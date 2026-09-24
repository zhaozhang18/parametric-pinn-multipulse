# -*- coding: utf-8 -*-
"""
train_inverse_multi_pulse_pinn_pure_physics.py

新的多脉冲纯物理逆向比较脚本。

本脚本把用户当前要比较的两种逆向思路放在同一个入口中：

方法1：冻结已经训练好的正向 PINN F*
    F* 是一个条件正向传播器 F*(z,t,P)->u,v。逆向时 P 虽然未知，但被写成
    M×4 trainable logits 或离散枚举候选，作为 F* 的输入参与前向计算；只更新 P，
    不更新 F* 的网络权重。

    方法1拆成两个独立子方案：
      1a. method1_adam：冻结 F*，AdamW 多 restart 梯度优化 P logits；
      1b. method1_enum：冻结 F*，枚举 4^M 个离散 P，只用终端误差选择最优 P。

方法2：不使用训练好的正向模型做传播器
    训练一个单样本轨迹 PINN A_theta(z,t)->u,v。网络输入只包含 z、t 及 Fourier 特征，
    不把未知 P 当作网络输入；P 是外部 M×4 trainable logits，只通过 z=0 的
    参数化初始条件 A0(t;P) 进入损失。
    v6 默认支持跨样本 sequential warm-start：第 k 个输出波形训练得到的 best A_theta
    会作为第 k+1 个输出波形的 restart-0 初始模型；每个新样本的 P logits 重新初始化。

样本选择：
    不再使用旧的固定 25% inverse test split。
    默认从 run_dir/dataset/unseen_combinations.csv，也就是正向 PINN 没见过的 90% 组合中，
    随机选 --n-samples 个样本；SSFM 末端波形和真实 P 从 inverse SSFM memmap 数据集中读取。

SSFM 逆向数据默认位置：
    <run_dir>/inverse/ssfm_terminal_datasets/<dataset_name>/
        Y_terminal_power.npy
        P_levels.npy
        P_class_indices.npy
        tau_input.npy
        meta.json

典型用法（Windows cmd 可识别，换行用 ^）：
    python train_inverse_multi_pulse_pinn_pure_physics.py ^
      --run-dir "./MULTIPULSE_FULL_RUNS\\M4_full" ^
      -M 4 ^
      --device cuda ^
      --n-samples 10 ^
      --method1-epochs 3000 ^
      --method1-restarts 8 ^
      --method2-adam-steps 2500 ^
      --method2-restarts 8
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlse import PAM4_LEVELS, load_combinations_csv, pulse_centers_t0, pinn_half_window_t0
from train_multi_pulse_pinn import ConditionalPINN, initial_condition, load_forward_checkpoint
from inverse_data_hybrid_mapping import (
    LEVELS,
    all_legal_pam4_candidates,
    build_or_load_inverse_ssfm_dataset,
    powers_to_class_indices,
    selected_indices_hash,
    load_pde_params_from_forward_checkpoint,
    pde_residual_for_inverse,
    terminal_power_forward_model,
    terminal_relative_mse,
)


# =============================================================================
# 基础工具
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_device(text: str) -> torch.device:
    text = str(text).strip().lower()
    if text == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def slug_float(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def combo_key(combo: Sequence[float], ndigits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in combo)


def combo_to_text(combo: Sequence[float]) -> str:
    return ";".join(f"{float(x):g}" for x in combo)


def parse_combo_text(text: str) -> list[float]:
    return [float(x) for x in str(text).split(";") if str(x).strip()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def powers_to_classes_np(powers: np.ndarray, levels: Sequence[float] = LEVELS) -> np.ndarray:
    p = np.asarray(powers, dtype=np.float32)
    lv = np.asarray(levels, dtype=np.float32)
    return np.argmin(np.abs(p[..., None] - lv.reshape(1, -1)), axis=-1).astype(np.int64)


def powers_from_logits(
    logits: torch.Tensor,
    levels: torch.Tensor,
    temperature: float = 1.0,
    mode: str = "straight_through",
) -> tuple[torch.Tensor, torch.Tensor]:
    """M×4 logits -> 1×M normalized amplitudes, plus M×4 probabilities."""
    probs = F.softmax(logits / max(float(temperature), 1e-8), dim=-1)
    mode = str(mode).strip().lower()
    if mode in {"straight_through", "st", "hard"}:
        hard_idx = torch.argmax(probs, dim=-1)
        hard = F.one_hot(hard_idx, num_classes=levels.numel()).to(dtype=probs.dtype, device=probs.device)
        weights = hard - probs.detach() + probs
    elif mode in {"soft", "softmax"}:
        weights = probs
    else:
        raise ValueError("power map mode must be 'straight_through' or 'soft'.")
    p = torch.sum(weights * levels.reshape(1, -1), dim=-1).reshape(1, -1)
    return p, probs


def discrete_from_logits(logits: torch.Tensor, levels: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits / max(float(temperature), 1e-8), dim=-1)
    cls = torch.argmax(probs, dim=-1)
    p = levels[cls].reshape(1, -1)
    conf = torch.max(probs, dim=-1).values
    return p, cls, conf


def entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()


def linear_anneal(start: float, end: float, frac: float) -> float:
    """Linearly interpolate start -> end for frac in [0, 1]."""
    f = min(1.0, max(0.0, float(frac)))
    return float(start) + (float(end) - float(start)) * f


def method2_current_temperature(args: argparse.Namespace, step: int, total_steps: int) -> float:
    """Temperature used only by method2 P relaxation.

    The original global --power-temperature is still used by method1.  For
    method2, a high initial temperature lets the soft power relaxation move
    between PAM4 levels; a low final temperature makes the logits discrete.
    """
    start = float(getattr(args, "method2_temperature_start", float(args.power_temperature)))
    end = float(getattr(args, "method2_temperature_end", float(args.power_temperature)))
    total = max(1, int(total_steps))
    return linear_anneal(start, end, (int(step) - 1) / max(1, total - 1))


def method2_entropy_weight_now(args: argparse.Namespace, step: int, total_steps: int) -> float:
    """Entropy regularization for method2.

    It is often harmful to force discreteness at the very beginning, because the
    optimizer may simply commit to a random initial class.  This ramp keeps the
    entropy term off for an initial fraction of Adam steps, then turns it on.
    """
    w = float(getattr(args, "method2_entropy_weight", 0.0))
    start_frac = float(getattr(args, "method2_entropy_start_frac", 0.0))
    if int(step) < int(max(1, total_steps) * max(0.0, min(1.0, start_frac))):
        return 0.0
    return w


def method2_power_map_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "method2_power_map_mode", "inherit")).strip().lower()
    if mode in {"inherit", ""}:
        return str(args.power_map_mode)
    return mode


def initial_power_np(tau: np.ndarray, powers: Sequence[float], centers: Sequence[float]) -> np.ndarray:
    y = np.zeros_like(np.asarray(tau, dtype=np.float64))
    for P, c in zip(powers, centers):
        y += float(P) * np.exp(-((tau - float(c)) ** 2) / 2.0)
    return y ** 2


def relative_l2_np(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-300))


def terminal_observable_mode(args: argparse.Namespace) -> str:
    mode = str(getattr(args, "terminal_observable", "power")).strip().lower()
    aliases = {
        "field": "complex",
        "complex_field": "complex",
        "uv": "complex",
        "both": "power_and_complex",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"power", "complex", "power_and_complex"}:
        raise ValueError("--terminal-observable must be power, complex, or power_and_complex.")
    return mode


def terminal_field_forward_model(
    model: nn.Module,
    tau: torch.Tensor,
    powers: torch.Tensor,
    zeta: float,
    chunk_t: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable terminal complex field through either F*(z,t,P) or A_theta(z,t)."""
    B = int(powers.shape[0])
    us: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    for start in range(0, int(tau.numel()), int(chunk_t)):
        end = min(int(tau.numel()), start + int(chunk_t))
        t_chunk = tau[start:end]
        T = int(t_chunk.numel())
        z = torch.full((B * T, 1), float(zeta), dtype=powers.dtype, device=powers.device)
        t = t_chunk.reshape(1, T, 1).expand(B, T, 1).reshape(B * T, 1).to(dtype=powers.dtype, device=powers.device)
        p_rep = powers.reshape(B, 1, -1).expand(B, T, -1).reshape(B * T, -1)
        u, v = model(z, t, p_rep)
        us.append(u.reshape(B, T))
        vs.append(v.reshape(B, T))
    return torch.cat(us, dim=1), torch.cat(vs, dim=1)


def terminal_relative_field_mse(
    pred_u: torch.Tensor,
    pred_v: torch.Tensor,
    ref_u: torch.Tensor,
    ref_v: torch.Tensor,
) -> torch.Tensor:
    num = torch.sum((pred_u - ref_u) ** 2 + (pred_v - ref_v) ** 2, dim=1)
    den = torch.sum(ref_u ** 2 + ref_v ** 2, dim=1) + 1e-12
    return torch.mean(num / den)


def terminal_loss_from_uv_power(
    pred_u: torch.Tensor,
    pred_v: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    observable: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return selected terminal loss plus separate power/complex components.

    observable="power" uses only intensity. observable="complex" uses the SSFM
    real/imag terminal field. observable="power_and_complex" averages both.
    """
    pred_power = pred_u ** 2 + pred_v ** 2
    power_loss = terminal_relative_mse(pred_power, y_power)
    mode = str(observable).strip().lower()
    if mode == "power":
        return power_loss, power_loss, None
    if y_real is None or y_imag is None:
        raise RuntimeError(
            "terminal observable requires complex SSFM field, but Y_terminal_real.npy/Y_terminal_imag.npy were not loaded. "
            "Run with --rebuild-inverse-dataset --terminal-observable complex."
        )
    complex_loss = terminal_relative_field_mse(pred_u, pred_v, y_real, y_imag)
    if mode == "complex":
        return complex_loss, power_loss, complex_loss
    if mode == "power_and_complex":
        return 0.5 * (power_loss + complex_loss), power_loss, complex_loss
    raise ValueError("unknown terminal observable mode")


class TrajectoryPINN(nn.Module):
    """Single-sample inverse PINN: A_theta(z,t) -> (u,v).

    This is the intended method2 formulation.  The unknown 4-PAM amplitudes P are not
    network inputs; P is a separate trainable latent vector that only defines the
    initial condition A0(t;P).

    The forward signature still accepts an optional `powers` argument so that the
    existing NLSE/terminal helper functions can call model(z,t,powers).  The
    argument is deliberately ignored.
    """

    def __init__(
        self,
        n_pulses: int = 0,
        hidden: int = 100,
        layers: int = 4,
        z_max_ld: float = 4.0,
        t_min: float = -36.0,
        t_max: float = 36.0,
        p_min: float = 0.25,
        p_max: float = 1.0,
        fourier_features: int = 0,
    ) -> None:
        super().__init__()
        if int(layers) < 1:
            raise ValueError("layers must be >= 1.")
        self.n_pulses = int(n_pulses)
        self.z_max_ld = float(z_max_ld)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.fourier_features = int(fourier_features)

        # Inputs are z, t and Fourier features only.  No P input here.
        in_dim = 2 + 4 * max(0, self.fourier_features)
        mods: list[nn.Module] = [nn.Linear(in_dim, int(hidden)), nn.Tanh()]
        for _ in range(int(layers) - 1):
            mods += [nn.Linear(int(hidden), int(hidden)), nn.Tanh()]
        mods.append(nn.Linear(int(hidden), 2))
        self.net = nn.Sequential(*mods)

    def _encode(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z_s = 2.0 * z / self.z_max_ld - 1.0
        t_s = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        feats: list[torch.Tensor] = [z_s, t_s]
        for k in range(1, self.fourier_features + 1):
            kk = float(k) * math.pi
            feats += [torch.sin(kk * z_s), torch.cos(kk * z_s), torch.sin(kk * t_s), torch.cos(kk * t_s)]
        return torch.cat(feats, dim=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor, powers: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(self._encode(z, t))
        return out[:, 0:1], out[:, 1:2]

    def config(self) -> dict[str, float | int | str]:
        return {
            "model_type": "trajectory_no_P_input",
            "n_pulses": int(self.n_pulses),
            "hidden": int(self.net[0].out_features),
            "layers": int((len(self.net) - 1) // 2),
            "z_max_ld": float(self.z_max_ld),
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
            "p_min": float(self.p_min),
            "p_max": float(self.p_max),
            "fourier_features": int(self.fourier_features),
        }


def make_method2_model(base_cfg: dict[str, Any], args: argparse.Namespace) -> nn.Module:
    formulation = str(getattr(args, "method2_formulation", "trajectory")).strip().lower()
    if formulation == "conditional":
        return ConditionalPINN(**base_cfg)
    if formulation == "trajectory":
        return TrajectoryPINN(**base_cfg)
    raise ValueError("--method2-formulation must be 'trajectory' or 'conditional'.")


# =============================================================================
# 正向模型、SSFM 数据集、unseen 样本选择
# =============================================================================


def resolve_path_inside_run_dir(path_like: str | Path, run_dir: Path) -> Path | None:
    """Resolve stale paths saved in run_manifest/run_config after moving runs.

    Results merged across machines may contain absolute paths or legacy run
    names. This helper first tries the path as-is, then re-anchors known subpaths
    (train/<model_dir>, eval/<eval_dir>, dataset/<file>) under the current
    --run-dir.  It prevents inverse runs from failing merely because the
    experiment folder was copied to another drive or renamed.
    """
    if not str(path_like).strip():
        return None
    p = Path(path_like)
    if p.exists():
        return p
    raw = str(path_like).replace("\\", "/")
    parts = [x for x in raw.split("/") if x]
    for anchor in ("train", "eval", "dataset", "inverse"):
        if anchor in parts:
            i = parts.index(anchor)
            cand = run_dir.joinpath(*parts[i:])
            if cand.exists():
                return cand
            # For checkpoint paths, using only the model directory name is often enough.
            if anchor == "train" and len(parts) > i + 2:
                cand = run_dir / "train" / parts[i + 1] / parts[-1]
                if cand.exists():
                    return cand
            if anchor == "eval" and len(parts) > i + 2:
                cand = run_dir / "eval" / parts[i + 1] / parts[-1]
                if cand.exists():
                    return cand
    return None


def find_latest_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "eval").glob("*/metrics_stream.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"找不到 forward metrics_stream.csv：{run_dir / 'eval'}")
    return candidates[0]


def find_forward_checkpoint(run_dir: Path, label: str) -> Path:
    label = str(label)
    candidates = list((run_dir / "train").glob(f"*{label}*/forward_pinn.pt"))
    if not candidates and label == "no_fourier":
        candidates = list((run_dir / "train").glob("*no*fourier*/forward_pinn.pt"))
    if not candidates:
        candidates = list((run_dir / "train").glob("*/forward_pinn.pt"))
    candidates = [p for p in candidates if label in p.parent.name or label == ""]
    if not candidates:
        raise FileNotFoundError(f"找不到 label={label!r} 的 forward_pinn.pt，目录：{run_dir / 'train'}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    return load_json(path) if path.exists() else {}


def select_best_forward(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    run_dir = Path(args.run_dir)
    manifest = load_manifest(run_dir)

    if str(args.forward_model_path).strip():
        ckpt = Path(args.forward_model_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"指定的正向模型不存在：{ckpt}")
        label = ckpt.parent.name
        return ckpt, label, {"manual": True, "checkpoint": str(ckpt), "label": label}

    metrics_csv = None
    if str(args.forward_metrics_csv).strip():
        metrics_csv = resolve_path_inside_run_dir(args.forward_metrics_csv, run_dir) or Path(args.forward_metrics_csv)
    elif manifest.get("forward_metrics_csv"):
        metrics_csv = resolve_path_inside_run_dir(manifest["forward_metrics_csv"], run_dir)
    if metrics_csv is None or not metrics_csv.exists():
        metrics_csv = find_latest_metrics_csv(run_dir)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"forward metrics CSV 不存在：{metrics_csv}")

    rows = read_csv_dicts(metrics_csv)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if "model" not in row or "rel_l2_power" not in row:
            continue
        try:
            grouped.setdefault(str(row["model"]), []).append(float(row["rel_l2_power"]))
        except Exception:
            continue
    if not grouped:
        raise ValueError(f"{metrics_csv} 中找不到 model 和 rel_l2_power 列。")

    stats: list[dict[str, Any]] = []
    ckpt_map = {str(k): str(v) for k, v in dict(manifest.get("forward_checkpoints", {}) or {}).items()}
    for label, vals in grouped.items():
        arr = np.asarray(vals, dtype=float)
        ckpt = None
        if label in ckpt_map:
            ckpt = resolve_path_inside_run_dir(ckpt_map[label], run_dir)
        if ckpt is None or not ckpt.exists():
            ckpt = find_forward_checkpoint(run_dir, label)
        stats.append({
            "label": label,
            "checkpoint": str(ckpt),
            "mean_rel_l2_power": float(np.mean(arr)),
            "p95_rel_l2_power": float(np.quantile(arr, 0.95)),
            "max_rel_l2_power": float(np.max(arr)),
            "n_rows": int(arr.size),
        })
    stats = sorted(stats, key=lambda r: (r["mean_rel_l2_power"], r["p95_rel_l2_power"], r["max_rel_l2_power"]))
    selected = stats[0]
    return Path(selected["checkpoint"]), str(selected["label"]), {
        "metrics_csv": str(metrics_csv),
        "selected": selected,
        "all_candidates": stats,
    }



def enforce_forward_quality_gate(args: argparse.Namespace, forward_info: dict[str, Any]) -> None:
    """Stop inverse comparison early when the selected forward model is too inaccurate.

    The thresholds are intentionally opt-in because different experiments may use
    different evaluation grids and acceptable error scales.
    """
    selected = dict(forward_info.get("selected", {}) or {})
    if not selected:
        # Manual --forward-model-path has no metrics unless the user also supplies them.
        return
    mean_thr = float(getattr(args, "max_forward_mean_rel_l2_power", 0.0) or 0.0)
    p95_thr = float(getattr(args, "max_forward_p95_rel_l2_power", 0.0) or 0.0)
    failures = []
    mean_val = selected.get("mean_rel_l2_power")
    p95_val = selected.get("p95_rel_l2_power")
    if mean_thr > 0 and mean_val is not None and float(mean_val) > mean_thr:
        failures.append(f"mean_rel_l2_power={float(mean_val):.6g} > threshold={mean_thr:.6g}")
    if p95_thr > 0 and p95_val is not None and float(p95_val) > p95_thr:
        failures.append(f"p95_rel_l2_power={float(p95_val):.6g} > threshold={p95_thr:.6g}")
    if failures:
        candidates = forward_info.get("all_candidates", [])
        msg = [
            "所选正向模型没有通过质量门槛，停止逆向比较。",
            "原因：" + "; ".join(failures),
            "所选模型：" + json.dumps(selected, ensure_ascii=False),
        ]
        if candidates:
            msg.append("所有候选模型：" + json.dumps(candidates, ensure_ascii=False))
        raise RuntimeError("\n".join(msg))


def infer_m_from_sources(args: argparse.Namespace, run_dir: Path, forward_ckpt: Path | None = None) -> int:
    if int(args.n_pulses) > 0:
        return int(args.n_pulses)
    manifest = load_manifest(run_dir)
    for key in ["ssfm_eval_grid"]:
        if isinstance(manifest.get(key), dict) and manifest[key].get("M"):
            return int(manifest[key]["M"])
    ds_summary = run_dir / "dataset" / "dataset_summary.json"
    if ds_summary.exists():
        js = load_json(ds_summary)
        if js.get("n_pulses"):
            return int(js["n_pulses"])
    if forward_ckpt is not None and forward_ckpt.exists():
        ckpt = torch.load(forward_ckpt, map_location="cpu")
        if ckpt.get("n_pulses"):
            return int(ckpt["n_pulses"])
        cfg = ckpt.get("model_config", {}) or {}
        if cfg.get("n_pulses"):
            return int(cfg["n_pulses"])
    raise ValueError("无法自动推断 M；请显式传入 -M/--n-pulses。")


def find_inverse_dataset(run_dir: Path, dataset_dir: str = "", require_field: bool = False) -> Path | None:
    if str(dataset_dir).strip():
        root = Path(dataset_dir)
        return root if root.exists() else None
    roots = sorted((run_dir / "inverse" / "ssfm_terminal_datasets").glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for meta in roots:
        root = meta.parent
        if not (root / "Y_terminal_power.npy").exists():
            continue
        if require_field and not ((root / "Y_terminal_real.npy").exists() and (root / "Y_terminal_imag.npy").exists()):
            continue
        return root
    return None


def infer_eval_grid(args: argparse.Namespace, run_dir: Path, M: int, forward_ckpt: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    grid = dict(manifest.get("ssfm_eval_grid", {}) or {})
    grid_json = run_dir / "ssfm_eval_grid.json"
    if not grid and grid_json.exists():
        grid = load_json(grid_json)

    ckpt = torch.load(forward_ckpt, map_location="cpu") if forward_ckpt.exists() else {}
    cfg = dict(ckpt.get("model_config", {}) or {})

    compare_half_default = pinn_half_window_t0(M)
    half_window = float(args.t_window_t0) if float(args.t_window_t0) > 0 else float(grid.get("eval_half_window_t0", 0) or compare_half_default)
    n_t = int(args.n_t) if int(args.n_t) > 0 else int(grid.get("eval_n_t", 2048))
    n_z = int(args.n_z) if int(args.n_z) > 0 else int(grid.get("eval_n_z", 500))
    compare_t_min = float(args.compare_t_min) if str(args.compare_t_min).strip() else float(grid.get("compare_t_min", cfg.get("t_min", -compare_half_default)))
    compare_t_max = float(args.compare_t_max) if str(args.compare_t_max).strip() else float(grid.get("compare_t_max", cfg.get("t_max", compare_half_default)))
    z_max_ld = float(args.z_max_ld) if float(args.z_max_ld) > 0 else float(grid.get("z_max_ld", cfg.get("z_max_ld", 4.0)))

    return {
        "M": int(M),
        "eval_half_window_t0": half_window,
        "eval_n_t": n_t,
        "eval_n_z": n_z,
        "compare_t_min": compare_t_min,
        "compare_t_max": compare_t_max,
        "z_max_ld": z_max_ld,
        "source_grid": grid,
    }


def make_dataset_build_namespace(args: argparse.Namespace, M: int, eval_grid: dict[str, Any], ds_root: Path) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.n_pulses = int(M)
    ns.inverse_dataset_dir = str(ds_root)
    ns.inverse_dataset_dtype = str(args.inverse_dataset_dtype)
    ns.inverse_dataset_mode = str(getattr(args, "inverse_dataset_mode", "selected"))
    ns.selected_dataset_indices = str(getattr(args, "selected_dataset_indices", ""))
    ns.selected_indices_source = str(getattr(args, "selected_indices_source", ""))
    ns.terminal_observable = terminal_observable_mode(args)
    ns.save_terminal_field = bool(args.save_terminal_field) or terminal_observable_mode(args) in {"complex", "power_and_complex"}
    ns.save_initial_power = bool(args.save_initial_power)
    ns.reuse_inverse_dataset = not bool(args.rebuild_inverse_dataset)
    ns.inverse_max_combos = int(args.inverse_max_combos)
    ns.seed = int(args.seed)
    ns.inverse_input_points = int(args.inverse_input_points)
    ns.inverse_dataset_log_every = int(args.inverse_dataset_log_every)
    ns.quiet_ssfm = not bool(args.verbose_ssfm)

    ns.t_window_t0 = float(eval_grid["eval_half_window_t0"])
    ns.n_t = int(eval_grid["eval_n_t"])
    ns.n_z = int(eval_grid["eval_n_z"])
    ns.z_max_ld = float(eval_grid["z_max_ld"])
    ns.compare_t_min = float(eval_grid["compare_t_min"])
    ns.compare_t_max = float(eval_grid["compare_t_max"])
    return ns


def get_or_build_inverse_dataset(args: argparse.Namespace, run_dir: Path, M: int, forward_ckpt: Path, device: torch.device) -> tuple[Path, dict[str, Any]]:
    require_field = terminal_observable_mode(args) in {"complex", "power_and_complex"} or bool(args.save_terminal_field)
    mode = str(getattr(args, "inverse_dataset_mode", "selected")).strip().lower()

    eval_grid = infer_eval_grid(args, run_dir, M, forward_ckpt)
    if str(args.inverse_dataset_dir).strip():
        ds_root = Path(args.inverse_dataset_dir)
    else:
        storage = "field" if require_field else "power"
        if bool(args.save_initial_power):
            storage += "_xinit"
        mode_part = "full"
        if mode == "selected":
            selected_indices = [int(x) for x in str(getattr(args, "selected_dataset_indices", "")).replace(",", " ").split() if x.strip()]
            mode_part = f"selectedN{len(selected_indices)}_h{selected_indices_hash(selected_indices)}" if selected_indices else "selectedN0"
        name = (
            f"inverse_ssfm_M{M}_win{eval_grid['eval_half_window_t0']:g}T0_nt{int(eval_grid['eval_n_t'])}_nz{int(eval_grid['eval_n_z'])}"
            f"_cmp{eval_grid['compare_t_min']:g}_to_{eval_grid['compare_t_max']:g}_Nin{int(args.inverse_input_points)}_{storage}_{args.inverse_dataset_dtype}_{mode_part}"
        ).replace(".", "p").replace("-", "m")
        ds_root = run_dir / "inverse" / "ssfm_terminal_datasets" / name

    existing = ds_root if ds_root.exists() else None
    if mode == "full" and existing is None:
        # Backward-compatible: full mode can reuse older full field datasets.
        existing = find_inverse_dataset(run_dir, args.inverse_dataset_dir, require_field=require_field)

    if existing is not None:
        required = ["Y_terminal_power.npy", "P_levels.npy", "P_class_indices.npy", "tau_input.npy", "meta.json", "completed_mask.npy"]
        if require_field:
            required += ["Y_terminal_real.npy", "Y_terminal_imag.npy"]
        if all((existing / name).exists() for name in required) and not args.rebuild_inverse_dataset:
            try:
                access_t0 = time.time()
                meta = load_json(existing / "meta.json")
                cm = np.load(existing / "completed_mask.npy", mmap_mode="r")
                complete_ok = int(np.sum(cm)) == int(meta.get("n_samples", len(cm)))
                if complete_ok:
                    meta = dict(meta)
                    meta["_dataset_access"] = {
                        "reused_this_call": True,
                        "generated_this_call": False,
                        "prepare_elapsed_sec_this_call": float(time.time() - access_t0),
                        "ssfm_generation_elapsed_sec_this_call": 0.0,
                        "ssfm_generation_elapsed_sec_original": float(meta.get("build_elapsed_sec", 0.0) or 0.0),
                    }
                    return existing, meta
            except Exception:
                pass

    if not args.build_dataset_if_missing and existing is None:
        raise FileNotFoundError(
            "未找到 inverse SSFM terminal dataset。默认位置为 "
            f"{run_dir / 'inverse' / 'ssfm_terminal_datasets'}，也可以用 --inverse-dataset-dir 指定。"
        )

    print("\n========== Build / load inverse SSFM terminal dataset ==========")
    print(f"dataset dir = {ds_root}")
    access_t0 = time.time()
    build_args = make_dataset_build_namespace(args, M, eval_grid, ds_root)
    info = build_or_load_inverse_ssfm_dataset(build_args, ds_root.parent, device)
    meta = dict(info["meta"])
    generated = not bool(meta.get("_dataset_access", {}).get("reused_this_call", False))
    original_build = float(meta.get("build_elapsed_sec", 0.0) or 0.0)
    meta["_dataset_access"] = {
        "reused_this_call": not generated,
        "generated_this_call": generated,
        "prepare_elapsed_sec_this_call": float(time.time() - access_t0),
        "ssfm_generation_elapsed_sec_this_call": original_build if generated else 0.0,
        "ssfm_generation_elapsed_sec_original": original_build,
    }
    return Path(info["root"]), meta

def load_unseen_combinations(run_dir: Path, args: argparse.Namespace, M: int) -> list[tuple[float, ...]]:
    if str(args.unseen_csv).strip():
        path = Path(args.unseen_csv)
    else:
        path = run_dir / "dataset" / "unseen_combinations.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 unseen_combinations.csv：{path}\n"
            "本脚本需要从正向 PINN 未见过的 90% 组合中抽样。"
        )
    rows = [tuple(float(x) for x in row) for row in load_combinations_csv(path)]
    rows = [r for r in rows if len(r) == int(M)]
    if not rows:
        raise RuntimeError(f"{path} 中没有 M={M} 的 unseen 组合。")
    return rows


def preselect_unseen_samples(args: argparse.Namespace, run_dir: Path, M: int, out_dir: Path) -> list[dict[str, Any]]:
    """Select inverse test samples before building SSFM data.

    v8 change: sample selection is now independent of the inverse SSFM cache.
    For --inverse-dataset-mode selected, these global dataset indices are passed
    to the SSFM builder, so only the selected terminal fields are generated.
    """
    all_p = all_legal_pam4_candidates(int(M))
    all_c = powers_to_class_indices(all_p)
    n_total = int(all_p.shape[0])

    if str(args.sample_indices).strip():
        indices = [int(x) for x in str(args.sample_indices).replace(",", " ").split()]
        selected: list[dict[str, Any]] = []
        for idx in indices:
            if idx < 0 or idx >= n_total:
                raise IndexError(f"sample index {idx} 超出全组合范围 [0,{n_total}) for M={M}")
            p = np.asarray(all_p[idx], dtype=np.float32)
            c = np.asarray(all_c[idx], dtype=np.int64)
            selected.append({
                "rank": len(selected),
                "dataset_index": int(idx),
                "amplitudes": [float(x) for x in p.tolist()],
                "powers": [float(x) for x in p.tolist()],  # legacy key; values are amplitudes
                "classes": [int(x) for x in c.tolist()],
                "source": "manual_sample_indices",
            })
        payload = {
            "rule": "manual global SSFM combination indices; no random sampling is used",
            "sample_seed": int(args.sample_seed),
            "requested_n_samples": len(selected),
            "available_unseen_in_dataset": None,
            "selected": selected,
        }
        write_json(out_dir / "selected_unseen_samples.json", payload)
        return selected

    unseen = load_unseen_combinations(run_dir, args, M)
    index_by_combo = {combo_key(all_p[i]): int(i) for i in range(n_total)}
    candidates: list[dict[str, Any]] = []
    for combo in unseen:
        key = combo_key(combo)
        if key not in index_by_combo:
            continue
        idx = index_by_combo[key]
        c = np.asarray(all_c[idx], dtype=np.int64)
        candidates.append({
            "dataset_index": int(idx),
            "powers": [float(x) for x in combo],
            "classes": [int(x) for x in c.tolist()],
            "source": "forward_unseen_combinations_csv",
        })
    if not candidates:
        raise RuntimeError("unseen_combinations.csv 与合法 PAM4 组合没有可匹配项。")

    rng = np.random.default_rng(int(args.sample_seed))
    n_pick = min(int(args.n_samples), len(candidates))
    pick = rng.choice(len(candidates), size=n_pick, replace=False)
    selected = []
    for rank, k in enumerate(pick.tolist()):
        item = dict(candidates[int(k)])
        item["rank"] = int(rank)
        selected.append(item)
    write_json(out_dir / "selected_unseen_samples.json", {
        "rule": "random sample from forward-unseen combinations only; no fixed 25% inverse test split is used",
        "sample_seed": int(args.sample_seed),
        "requested_n_samples": int(args.n_samples),
        "available_unseen_in_dataset": int(len(candidates)),
        "selected": selected,
    })
    return selected


def attach_dataset_row_indices(selected: list[dict[str, Any]], ds_root: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Attach row_index used to read arrays in either full or selected SSFM dataset.

    dataset_index always means the global lexicographic 4-PAM amplitude combination index.
    row_index means the row inside P_levels/Y_terminal arrays.  For full datasets
    they are identical.  For selected-only datasets they differ.
    """
    meta_path = ds_root / "meta.json"
    meta = load_json(meta_path) if meta_path.exists() else {}
    G_path = ds_root / "P_global_indices.npy"
    if G_path.exists():
        global_indices = [int(x) for x in np.asarray(np.load(G_path, mmap_mode="r")).tolist()]
    else:
        P_mem = np.load(ds_root / "P_levels.npy", mmap_mode="r")
        global_indices = list(range(int(P_mem.shape[0])))
    row_by_global = {int(g): int(i) for i, g in enumerate(global_indices)}
    out: list[dict[str, Any]] = []
    for sample in selected:
        item = dict(sample)
        g = int(item["dataset_index"])
        if g not in row_by_global:
            raise KeyError(
                f"selected dataset_index={g} 不在 inverse SSFM dataset 中：{ds_root}\n"
                "如果使用 selected-only 数据集，请确认 sample_seed/sample_indices 与数据集一致，或加 --rebuild-inverse-dataset。"
            )
        item["row_index"] = int(row_by_global[g])
        item["inverse_dataset_mode"] = str(meta.get("dataset_mode", "full"))
        out.append(item)
    # overwrite the sample file with row_index included for auditing.
    sample_file = out_dir / "selected_unseen_samples.json"
    payload = load_json(sample_file) if sample_file.exists() else {}
    if isinstance(payload, dict):
        payload["inverse_dataset_dir"] = str(ds_root)
        payload["inverse_dataset_mode"] = str(meta.get("dataset_mode", "full"))
        payload["selected_hash"] = selected_indices_hash([int(x["dataset_index"]) for x in out])
        payload["selected"] = out
        write_json(sample_file, payload)
    return out


# =============================================================================
# 物理损失与评价
# =============================================================================


class LossArgs:
    """给 nlse_loss_forward_pinn/pde residual 使用的轻量参数对象。"""

    def __init__(self, args: argparse.Namespace, z_max_ld: float, compare_t_min: float, compare_t_max: float, points: int) -> None:
        self.z_max_ld = float(z_max_ld)
        self.compare_t_min = float(compare_t_min)
        self.compare_t_max = float(compare_t_max)
        self.nlse_t_min = str(args.nlse_t_min)
        self.nlse_t_max = str(args.nlse_t_max)
        self.nlse_points_per_sample = int(points)


def nlse_loss_any_model(
    model: nn.Module,
    pde_params: Any,
    powers: torch.Tensor,
    z_max_ld: float,
    t_min: float,
    t_max: float,
    n_points: int,
) -> torch.Tensor:
    if int(n_points) <= 0:
        return torch.tensor(0.0, dtype=powers.dtype, device=powers.device)
    B = int(powers.shape[0])
    K = int(n_points)
    z = torch.rand((B, K, 1), dtype=powers.dtype, device=powers.device) * float(z_max_ld)
    t = float(t_min) + (float(t_max) - float(t_min)) * torch.rand((B, K, 1), dtype=powers.dtype, device=powers.device)
    p_rep = powers.reshape(B, 1, -1).expand(B, K, -1).reshape(B * K, -1)
    f, g = pde_residual_for_inverse(model, z.reshape(B * K, 1), t.reshape(B * K, 1), p_rep, pde_params)
    return (f ** 2).mean() + (g ** 2).mean()


def ic_loss_for_model(
    model: ConditionalPINN,
    powers: torch.Tensor,
    centers: Sequence[float],
    t_min: float,
    t_max: float,
    n_points: int,
) -> torch.Tensor:
    if int(n_points) <= 0:
        return torch.tensor(0.0, dtype=powers.dtype, device=powers.device)
    t = float(t_min) + (float(t_max) - float(t_min)) * torch.rand((int(n_points), 1), dtype=powers.dtype, device=powers.device)
    z = torch.zeros_like(t)
    p = powers.reshape(1, -1).expand(int(n_points), -1)
    u_pred, v_pred = model(z, t, p)
    u_true, v_true = initial_condition(t, p, centers)
    return F.mse_loss(u_pred, u_true) + F.mse_loss(v_pred, v_true)


def terminal_loss_for_model_subset(
    model: nn.Module,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    powers: torch.Tensor,
    zeta: float,
    n_points: int,
    observable: str = "power",
) -> torch.Tensor:
    n_tau = int(tau.numel())
    if int(n_points) <= 0 or int(n_points) >= n_tau:
        idx = torch.arange(n_tau, device=tau.device)
    else:
        idx = torch.randint(0, n_tau, (int(n_points),), device=tau.device)
    t = tau[idx].reshape(-1, 1)
    z = torch.full_like(t, float(zeta))
    p = powers.reshape(1, -1).expand(int(t.numel()), -1)
    u, v = model(z, t, p)
    pred_u = u.reshape(1, -1)
    pred_v = v.reshape(1, -1)
    yr = y_real[:, idx] if y_real is not None else None
    yi = y_imag[:, idx] if y_imag is not None else None
    loss, _power_loss, _complex_loss = terminal_loss_from_uv_power(
        pred_u, pred_v, y_power[:, idx], yr, yi, observable
    )
    return loss


def terminal_loss_for_forward_method1(
    forward_model: nn.Module,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    powers: torch.Tensor,
    zeta: float,
    n_points: int,
    chunk_t: int,
    observable: str = "power",
) -> torch.Tensor:
    """Method1 terminal loss.

    n_points<=0 means using the full terminal slice every step. A positive value
    randomly samples terminal t-points each step for faster exploratory runs;
    final evaluation still uses the full terminal slice.
    """
    n_tau = int(tau.numel())
    if int(n_points) > 0 and int(n_points) < n_tau:
        return terminal_loss_for_model_subset(
            forward_model, tau, y_power, y_real, y_imag, powers,
            zeta=zeta, n_points=int(n_points), observable=observable,
        )
    u, v = terminal_field_forward_model(forward_model, tau, powers, zeta=float(zeta), chunk_t=int(chunk_t))
    loss, _power_loss, _complex_loss = terminal_loss_from_uv_power(u, v, y_power, y_real, y_imag, observable)
    return loss


def evaluate_discrete_solution(
    model: nn.Module,
    pde_params: Any,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    p_disc: torch.Tensor,
    true_p: np.ndarray,
    true_cls: np.ndarray,
    z_max_ld: float,
    t_min: float,
    t_max: float,
    forward_time_chunk: int,
    nlse_points: int,
    terminal_observable: str = "power",
    ic_points: int = 0,
    centers: Sequence[float] | None = None,
    method2_model: bool = False,
) -> dict[str, Any]:
    u, v = terminal_field_forward_model(model, tau, p_disc, zeta=float(z_max_ld), chunk_t=int(forward_time_chunk))
    terminal_mse, terminal_power_mse, terminal_complex_mse = terminal_loss_from_uv_power(
        u, v, y_power, y_real, y_imag, str(terminal_observable)
    )
    terminal_rel_l2 = torch.sqrt(torch.clamp(terminal_mse, min=0.0))
    terminal_power_rel_l2 = torch.sqrt(torch.clamp(terminal_power_mse, min=0.0))
    terminal_complex_rel_l2 = None
    if terminal_complex_mse is not None:
        terminal_complex_rel_l2 = torch.sqrt(torch.clamp(terminal_complex_mse, min=0.0))
    pred_terminal = u ** 2 + v ** 2
    nlse = nlse_loss_any_model(model, pde_params, p_disc, z_max_ld, t_min, t_max, int(nlse_points))
    ic_val = torch.tensor(0.0, dtype=torch.float32, device=p_disc.device)
    if method2_model and ic_points > 0 and centers is not None:
        ic_val = ic_loss_for_model(model, p_disc, centers, t_min, t_max, int(ic_points))

    p_np = p_disc.detach().cpu().numpy().reshape(-1).astype(np.float32)
    pred_cls = powers_to_classes_np(p_np.reshape(1, -1)).reshape(-1)
    exact = bool(np.allclose(p_np, np.asarray(true_p, dtype=np.float32), atol=1e-6))
    per_pulse = float(np.mean(pred_cls == np.asarray(true_cls, dtype=np.int64)))
    return {
        "estimated_powers": [float(x) for x in p_np.tolist()],
        "estimated_classes": [int(x) for x in pred_cls.tolist()],
        "exact_match": exact,
        "per_pulse_accuracy": per_pulse,
        "terminal_observable": str(terminal_observable),
        "terminal_relative_mse": float(terminal_mse.detach().cpu()),
        "terminal_rel_l2": float(terminal_rel_l2.detach().cpu()),
        # Backward-compatible field name: if observable=power this is the old metric;
        # if observable=complex it is still reported separately below.
        "terminal_rel_l2_power": float(terminal_power_rel_l2.detach().cpu()),
        "terminal_relative_mse_power": float(terminal_power_mse.detach().cpu()),
        "terminal_relative_mse_complex": float(terminal_complex_mse.detach().cpu()) if terminal_complex_mse is not None else None,
        "terminal_rel_l2_complex": float(terminal_complex_rel_l2.detach().cpu()) if terminal_complex_rel_l2 is not None else None,
        "nlse_loss": float(nlse.detach().cpu()),
        "ic_loss": float(ic_val.detach().cpu()),
        "pred_terminal_power": pred_terminal.detach().cpu().numpy().reshape(-1).astype(float),
        "pred_terminal_real": u.detach().cpu().numpy().reshape(-1).astype(float),
        "pred_terminal_imag": v.detach().cpu().numpy().reshape(-1).astype(float),
    }


# =============================================================================
# 方法1：冻结正向模型，只优化 trainable P
# =============================================================================


def make_initial_logits(M: int, device: torch.device, seed: int, carry_logits: torch.Tensor | None, restart: int) -> torch.Tensor:
    set_seed(seed)
    if restart == 0 and carry_logits is not None:
        init = carry_logits.detach().clone().to(device=device, dtype=torch.float32)
        init = init + 0.02 * torch.randn_like(init)
        return init
    logits = 0.05 * torch.randn((int(M), len(LEVELS)), dtype=torch.float32, device=device)
    init_cls = torch.randint(low=0, high=len(LEVELS), size=(int(M),), device=device)
    logits[torch.arange(int(M), device=device), init_cls] += 1.0
    return logits


def strip_large_terminal_arrays(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in payload.items()
        if k not in {"pred_terminal_power", "pred_terminal_real", "pred_terminal_imag"}
    }


def enumerate_method1_discrete_candidates(
    sample: dict[str, Any],
    forward_model: nn.Module,
    pde_params: Any,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    z_max_ld: float,
    t_min: float,
    t_max: float,
) -> dict[str, Any] | None:
    """Exhaustively check all 4^M legal 4-PAM amplitude candidates under frozen F*.

    This is not required for large M, but for M=4/5/6 it is the cleanest way to
    separate an optimizer failure from a frozen-forward-model/observability
    failure.  The enumeration objective is the terminal observable loss; the
    selected candidate is then fully evaluated with NLSE/IC diagnostics.
    """
    M = len(sample["powers"])
    candidates = all_legal_pam4_candidates(M)
    if int(candidates.shape[0]) > int(args.method1_enumeration_max_combos):
        print(
            f"  [method1][enumeration] skip: 4^{M}={int(candidates.shape[0])} > "
            f"max={int(args.method1_enumeration_max_combos)}",
            flush=True,
        )
        return None

    observable = terminal_observable_mode(args)
    bs = max(1, int(args.method1_enumeration_batch_size))
    best_loss = float("inf")
    best_p_np: np.ndarray | None = None
    all_losses: list[tuple[float, list[float]]] = []
    # Gradients are not needed for enumeration; this is only a discrete diagnostic/search.
    with torch.no_grad():
        for start in range(0, int(candidates.shape[0]), bs):
            p_np = candidates[start:start + bs]
            p_t = torch.tensor(p_np, dtype=torch.float32, device=device)
            u, v = terminal_field_forward_model(forward_model, tau, p_t, zeta=float(z_max_ld), chunk_t=int(args.forward_time_chunk))
            pred_power = u ** 2 + v ** 2
            ref_power = y_power.expand_as(pred_power)
            power_loss_vec = torch.sum((pred_power - ref_power) ** 2, dim=1) / (torch.sum(ref_power ** 2, dim=1) + 1e-12)
            if observable == "power":
                loss_vec = power_loss_vec
            else:
                if y_real is None or y_imag is None:
                    raise RuntimeError("method1 enumeration with complex terminal observable requires terminal real/imag dataset.")
                ref_u = y_real.expand_as(u)
                ref_v = y_imag.expand_as(v)
                complex_loss_vec = torch.sum((u - ref_u) ** 2 + (v - ref_v) ** 2, dim=1) / (torch.sum(ref_u ** 2 + ref_v ** 2, dim=1) + 1e-12)
                loss_vec = complex_loss_vec if observable == "complex" else 0.5 * (power_loss_vec + complex_loss_vec)
            vals = loss_vec.detach().cpu().numpy().astype(float)
            for j, val in enumerate(vals.tolist()):
                pp = [float(x) for x in p_np[j].tolist()]
                all_losses.append((float(val), pp))
                if float(val) < best_loss:
                    best_loss = float(val)
                    best_p_np = p_np[j].astype(np.float32).copy()

    if best_p_np is None:
        return None
    true_p = np.asarray(sample["powers"], dtype=np.float32)
    true_cls = np.asarray(sample["classes"], dtype=np.int64)
    p_disc = torch.tensor(best_p_np.reshape(1, -1), dtype=torch.float32, device=device)
    eval_payload = evaluate_discrete_solution(
        forward_model,
        pde_params,
        tau,
        y_power,
        y_real,
        y_imag,
        p_disc,
        true_p,
        true_cls,
        z_max_ld,
        t_min,
        t_max,
        int(args.forward_time_chunk),
        int(args.method1_eval_nlse_points),
        terminal_observable=observable,
        ic_points=int(args.method1_eval_ic_points),
        centers=pulse_centers_t0(M),
        method2_model=True,
    )
    eval_payload.update({
        "restart": -1,
        "seed": None,
        "source": "exhaustive_discrete_enumeration",
        "enumerated_candidates": int(candidates.shape[0]),
        "enumeration_terminal_loss": float(best_loss),
        "best_training_loss": float(best_loss),
        "best_training_epoch": 0,
        "mean_confidence": 1.0,
        "min_confidence": 1.0,
    })
    # Enumeration is a discrete search under the frozen forward model.
    # Its search/ranking objective is intentionally the terminal observable loss only.
    eval_payload["selection_objective"] = float(eval_payload["terminal_relative_mse"])
    if int(args.method1_save_enumeration_topk) > 0:
        top = sorted(all_losses, key=lambda x: x[0])[:int(args.method1_save_enumeration_topk)]
        eval_payload["enumeration_topk"] = [
            {"rank": int(i + 1), "terminal_loss": float(v), "powers": pows}
            for i, (v, pows) in enumerate(top)
        ]
    return eval_payload


def run_method1_single_sample(
    sample: dict[str, Any],
    forward_model: nn.Module,
    pde_params: Any,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    carry_logits: torch.Tensor | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    M = len(sample["powers"])
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=device)
    true_p = np.asarray(sample["powers"], dtype=np.float32)
    true_cls = np.asarray(sample["classes"], dtype=np.int64)
    centers = pulse_centers_t0(M)
    sample_dir = ensure_dir(out_dir / f"s{int(sample['rank']):02d}_i{int(sample['dataset_index'])}")
    history_rows: list[dict[str, Any]] = []
    restart_summaries: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_logits: torch.Tensor | None = None
    t_start = time.time()

    z_max_ld = float(getattr(forward_model, "z_max_ld", args.z_max_ld if args.z_max_ld > 0 else 4.0))
    t_min = float(tau.min().detach().cpu())
    t_max = float(tau.max().detach().cpu())
    observable = terminal_observable_mode(args)

    for r in range(int(args.method1_restarts)):
        init = make_initial_logits(M, device, int(args.seed) + 10000 * int(sample["rank"]) + 1000 * r, carry_logits, r if args.method1_carry_state else r + 1)
        logits = torch.nn.Parameter(init)
        optimizer = torch.optim.AdamW([logits], lr=float(args.method1_lr), weight_decay=0.0)
        best_restart_loss = float("inf")
        best_restart_epoch = 0
        best_restart_logits = logits.detach().clone()
        hist_stride = max(1, int(args.method1_log_every))
        last_improve_epoch = 0
        last_p_change_epoch = 0
        prev_p_text = ""
        stopped_early = False
        stop_reason = ""

        for ep in range(1, int(args.method1_epochs) + 1):
            optimizer.zero_grad(set_to_none=True)
            powers, probs = powers_from_logits(logits, levels, float(args.power_temperature), str(args.power_map_mode))
            terminal = terminal_loss_for_forward_method1(
                forward_model,
                tau,
                y_power,
                y_real,
                y_imag,
                powers,
                zeta=z_max_ld,
                n_points=int(args.method1_terminal_points),
                chunk_t=int(args.forward_time_chunk),
                observable=observable,
            )
            ic = ic_loss_for_model(
                forward_model, powers, centers, t_min=t_min, t_max=t_max, n_points=int(args.method1_ic_points)
            )
            nlse = nlse_loss_any_model(
                forward_model, pde_params, powers, z_max_ld, t_min, t_max, int(args.method1_nlse_points)
            )
            ent = entropy_loss(probs)
            total = (
                float(args.method1_terminal_weight) * terminal
                + float(args.method1_ic_weight) * ic
                + float(args.method1_nlse_weight) * nlse
                + float(args.method1_entropy_weight) * ent
            )
            total.backward()
            if float(args.method1_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_([logits], float(args.method1_grad_clip))
            optimizer.step()

            total_float = float(total.detach().cpu())
            terminal_float = float(terminal.detach().cpu())
            min_delta = float(args.method1_early_stop_min_delta)
            improved = total_float < (best_restart_loss - min_delta * max(1.0, abs(best_restart_loss if math.isfinite(best_restart_loss) else total_float)))
            if improved or not math.isfinite(best_restart_loss):
                best_restart_loss = total_float
                best_restart_epoch = int(ep)
                best_restart_logits = logits.detach().clone()
            # Early stopping monitors terminal loss by default. Total loss contains random IC/NLSE mini-batches,
            # so using total often prevents early stopping even after the discrete P has stabilized.
            if str(getattr(args, "method1_early_stop_monitor", "terminal")) == "terminal":
                monitor_float = terminal_float
            else:
                monitor_float = total_float
            if ep == 1:
                best_monitor_loss = float("inf")
            monitor_improved = monitor_float < (best_monitor_loss - min_delta * max(1.0, abs(best_monitor_loss if math.isfinite(best_monitor_loss) else monitor_float)))
            if monitor_improved or not math.isfinite(best_monitor_loss):
                best_monitor_loss = monitor_float
                last_improve_epoch = int(ep)
            with torch.no_grad():
                p_disc_tmp, _cls_disc, _conf = discrete_from_logits(logits, levels, float(args.power_temperature))
                p_np_tmp = p_disc_tmp.detach().cpu().numpy().reshape(-1)
                p_text_now = combo_to_text(p_np_tmp)
            if p_text_now != prev_p_text:
                last_p_change_epoch = int(ep)
                prev_p_text = p_text_now

            log_now = ep == 1 or ep % hist_stride == 0 or ep == int(args.method1_epochs)
            if log_now:
                history_rows.append({
                    "restart": int(r),
                    "epoch": int(ep),
                    "total_loss": total_float,
                    "terminal_relative_mse": float(terminal.detach().cpu()),
                    "ic_loss": float(ic.detach().cpu()),
                    "nlse_loss": float(nlse.detach().cpu()),
                    "entropy": float(ent.detach().cpu()),
                    "estimated_powers": p_text_now,
                    "exact_match_current_discrete": int(bool(np.allclose(p_np_tmp, true_p, atol=1e-6))),
                    "early_stop_triggered": 0,
                    "stop_reason": "",
                })
                if not bool(getattr(args, "quiet_progress", False)):
                    print(
                        f"  [method1][sample {int(sample['rank']) + 1}] "
                        f"restart {r + 1}/{int(args.method1_restarts)} "
                        f"epoch {ep}/{int(args.method1_epochs)} "
                        f"loss={total_float:.4e} terminal={float(terminal.detach().cpu()):.4e} "
                        f"ic={float(ic.detach().cpu()):.4e} nlse={float(nlse.detach().cpu()):.4e} P={p_text_now}",
                        flush=True,
                    )

            if bool(args.method1_early_stop) and ep >= int(args.method1_early_stop_min_epochs):
                no_improve = (int(ep) - int(last_improve_epoch)) >= int(args.method1_early_stop_patience)
                p_stable = (int(ep) - int(last_p_change_epoch)) >= int(args.method1_early_stop_p_stable)
                if no_improve and p_stable:
                    stopped_early = True
                    stop_reason = f"no_improve_for_{int(ep)-int(last_improve_epoch)}_epochs_and_P_stable_for_{int(ep)-int(last_p_change_epoch)}_epochs"
                    history_rows.append({
                        "restart": int(r),
                        "epoch": int(ep),
                        "total_loss": total_float,
                        "terminal_relative_mse": float(terminal.detach().cpu()),
                        "ic_loss": float(ic.detach().cpu()),
                        "nlse_loss": float(nlse.detach().cpu()),
                        "entropy": float(ent.detach().cpu()),
                        "estimated_powers": p_text_now,
                        "exact_match_current_discrete": int(bool(np.allclose(p_np_tmp, true_p, atol=1e-6))),
                        "early_stop_triggered": 1,
                        "stop_reason": stop_reason,
                    })
                    if not bool(getattr(args, "quiet_progress", False)):
                        print(f"  [method1][sample {int(sample['rank']) + 1}] restart {r + 1} early stop at epoch {ep}: {stop_reason}", flush=True)
                    break

        with torch.no_grad():
            eval_logits = best_restart_logits.to(device=device, dtype=torch.float32)
            p_disc, cls_disc, conf = discrete_from_logits(eval_logits, levels, float(args.power_temperature))
        eval_payload = evaluate_discrete_solution(
            forward_model,
            pde_params,
            tau,
            y_power,
            y_real,
            y_imag,
            p_disc,
            true_p,
            true_cls,
            z_max_ld,
            t_min,
            t_max,
            int(args.forward_time_chunk),
            int(args.method1_eval_nlse_points),
            terminal_observable=observable,
            ic_points=int(args.method1_eval_ic_points),
            centers=centers,
            method2_model=True,
        )
        eval_payload.update({
            "restart": int(r),
            "seed": int(args.seed) + 10000 * int(sample["rank"]) + 1000 * r,
            "best_training_loss": float(best_restart_loss),
            "best_training_epoch": int(best_restart_epoch),
            "early_stop_triggered": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "mean_confidence": float(conf.mean().detach().cpu()),
            "min_confidence": float(conf.min().detach().cpu()),
        })
        eval_payload["selection_objective"] = (
            float(args.method1_terminal_weight) * eval_payload["terminal_relative_mse"]
            + float(args.method1_ic_weight) * eval_payload["ic_loss"]
            + float(args.method1_nlse_weight) * eval_payload["nlse_loss"]
        )
        restart_summaries.append(strip_large_terminal_arrays(eval_payload))
        if not bool(getattr(args, "quiet_progress", False)):
            print(
                f"  [method1][sample {int(sample['rank']) + 1}] restart {r + 1} done: "
                f"P_hat={eval_payload['estimated_powers']} exact={eval_payload['exact_match']} "
                f"per={eval_payload['per_pulse_accuracy']:.3f} obj={eval_payload['selection_objective']:.4e}",
                flush=True,
            )
        if best is None or eval_payload["selection_objective"] < best["selection_objective"]:
            best = eval_payload
            best_logits = best_restart_logits.detach().clone().cpu()

    if bool(args.method1_enumerate_candidates):
        enum_payload = enumerate_method1_discrete_candidates(
            sample, forward_model, pde_params, tau, y_power, y_real, y_imag, args, device, z_max_ld, t_min, t_max
        )
        if enum_payload is not None:
            restart_summaries.append(strip_large_terminal_arrays(enum_payload))
            write_json(sample_dir / "method1_enumeration_summary.json", strip_large_terminal_arrays(enum_payload))
            if best is None or bool(args.method1_select_enumeration_if_better) and enum_payload["selection_objective"] < best["selection_objective"]:
                best = enum_payload
                # Store a deterministic one-hot-ish logits tensor for carry state.
                cls = powers_to_classes_np(np.asarray(enum_payload["estimated_powers"], dtype=np.float32).reshape(1, -1)).reshape(-1)
                logits_enum = torch.full((M, len(LEVELS)), -2.0, dtype=torch.float32)
                logits_enum[torch.arange(M), torch.tensor(cls, dtype=torch.long)] = 2.0
                best_logits = logits_enum

    assert best is not None and best_logits is not None
    elapsed = time.time() - t_start
    # Save the terminal and reconstructed initial waveforms for this sample.
    tau_np_local = tau.detach().cpu().numpy().reshape(-1).astype(float)
    y_np_local = y_power.detach().cpu().numpy().reshape(-1).astype(float)
    y_real_np_local = y_real.detach().cpu().numpy().reshape(-1).astype(float) if y_real is not None else None
    y_imag_np_local = y_imag.detach().cpu().numpy().reshape(-1).astype(float) if y_imag is not None else None
    pred_terminal_np = np.asarray(best["pred_terminal_power"], dtype=float).reshape(-1)
    pred_real_np = np.asarray(best.get("pred_terminal_real", np.full_like(pred_terminal_np, np.nan)), dtype=float).reshape(-1)
    pred_imag_np = np.asarray(best.get("pred_terminal_imag", np.full_like(pred_terminal_np, np.nan)), dtype=float).reshape(-1)
    centers_local = pulse_centers_t0(M)
    write_csv_dicts(sample_dir / "method1_waveforms.csv", [
        {
            "tau": float(tau_np_local[i]),
            "target_terminal_power": float(y_np_local[i]),
            "target_terminal_real": float(y_real_np_local[i]) if y_real_np_local is not None else "",
            "target_terminal_imag": float(y_imag_np_local[i]) if y_imag_np_local is not None else "",
            "pred_terminal_power": float(pred_terminal_np[i]),
            "pred_terminal_real": float(pred_real_np[i]),
            "pred_terminal_imag": float(pred_imag_np[i]),
            "true_initial_power": float(initial_power_np(tau_np_local, true_p, centers_local)[i]),
            "estimated_initial_power": float(initial_power_np(tau_np_local, best["estimated_powers"], centers_local)[i]),
        }
        for i in range(len(tau_np_local))
    ])

    if int(args.method1_restarts) <= 0 and bool(args.method1_enumerate_candidates):
        method1_name = "method1b_enum_P"
    elif bool(args.method1_enumerate_candidates):
        method1_name = "method1_adam_plus_enumeration"
    else:
        method1_name = "method1a_adam_optimize_P"
    summary = {
        "method": method1_name,
        "sample_rank": int(sample["rank"]),
        "dataset_index": int(sample["dataset_index"]),
        "true_powers": [float(x) for x in true_p.tolist()],
        "true_classes": [int(x) for x in true_cls.tolist()],
        "best_restart": int(best["restart"]),
        "estimated_powers": best["estimated_powers"],
        "estimated_classes": best["estimated_classes"],
        "exact_match": bool(best["exact_match"]),
        "per_pulse_accuracy": float(best["per_pulse_accuracy"]),
        "terminal_observable": str(best.get("terminal_observable", observable)),
        "terminal_relative_mse": float(best["terminal_relative_mse"]),
        "terminal_rel_l2": float(best.get("terminal_rel_l2", math.sqrt(max(0.0, float(best["terminal_relative_mse"]))))),
        "terminal_rel_l2_power": float(best["terminal_rel_l2_power"]),
        "terminal_relative_mse_power": float(best.get("terminal_relative_mse_power", best["terminal_relative_mse"])),
        "terminal_relative_mse_complex": best.get("terminal_relative_mse_complex"),
        "terminal_rel_l2_complex": best.get("terminal_rel_l2_complex"),
        "nlse_loss": float(best["nlse_loss"]),
        "ic_loss": float(best.get("ic_loss", 0.0)),
        "selection_objective": float(best["selection_objective"]),
        "elapsed_sec": float(elapsed),
        "all_restarts": restart_summaries,
    }
    write_csv_dicts(sample_dir / "method1_history.csv", history_rows)
    write_csv_dicts(sample_dir / "method1_restart_summary.csv", restart_summaries)
    write_json(sample_dir / "method1_summary.json", summary)
    torch.save({"best_logits": best_logits, "summary": summary}, sample_dir / "method1_best_logits.pt")
    return summary, best_logits

# =============================================================================
# 方法2：从头训练单样本轨迹 PINN + P
# =============================================================================


def conditional_pinn_config_from_checkpoint(forward_ckpt: Path, M: int, args: argparse.Namespace) -> dict[str, Any]:
    ckpt = torch.load(forward_ckpt, map_location="cpu")
    cfg = dict(ckpt.get("model_config", {}) or {})
    if not cfg:
        compare_half = pinn_half_window_t0(M)
        cfg = {
            "n_pulses": int(M),
            "hidden": 100,
            "layers": 4,
            "z_max_ld": 4.0,
            "t_min": -compare_half,
            "t_max": compare_half,
            "p_min": 0.25,
            "p_max": 1.0,
            "fourier_features": 0,
        }
    cfg["n_pulses"] = int(M)
    if int(args.method2_hidden) > 0:
        cfg["hidden"] = int(args.method2_hidden)
    if int(args.method2_layers) > 0:
        cfg["layers"] = int(args.method2_layers)
    if int(args.method2_fourier_features) >= 0:
        cfg["fourier_features"] = int(args.method2_fourier_features)
    return cfg


def run_method2_single_sample(
    sample: dict[str, Any],
    base_cfg: dict[str, Any],
    pde_params: Any,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    warm_start_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    M = len(sample["powers"])
    levels = torch.tensor(LEVELS, dtype=torch.float32, device=device)
    true_p = np.asarray(sample["powers"], dtype=np.float32)
    true_cls = np.asarray(sample["classes"], dtype=np.int64)
    centers = pulse_centers_t0(M)
    sample_dir = ensure_dir(out_dir / f"s{int(sample['rank']):02d}_i{int(sample['dataset_index'])}")
    restart_summaries: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    t_start = time.time()

    z_max_ld = float(base_cfg.get("z_max_ld", 4.0))
    t_min = float(tau.min().detach().cpu())
    t_max = float(tau.max().detach().cpu())
    observable = terminal_observable_mode(args)

    for r in range(int(args.method2_restarts)):
        seed = int(args.seed) + 20000 * int(sample["rank"]) + 1000 * r
        set_seed(seed)
        model = make_method2_model(base_cfg, args).to(device)
        used_warm_start = False
        if r == 0 and warm_start_state is not None and bool(getattr(args, "method2_sequential_warm_start", True)):
            try:
                model.load_state_dict(warm_start_state["model_state"], strict=True)
                used_warm_start = True
                if not bool(getattr(args, "quiet_progress", False)):
                    print(f"  [method2][sample {int(sample['rank']) + 1}] restart 1 uses previous sample best model as warm start", flush=True)
            except Exception as exc:
                print(f"  [method2][sample {int(sample['rank']) + 1}] warm-start skipped because model state is incompatible: {exc}", flush=True)
        # P is sample-specific and is always re-initialized; only the PINN weights can be warm-started.
        logits = torch.nn.Parameter(make_initial_logits(M, device, seed, None, restart=r + 1))
        model_params = list(model.parameters())
        params = model_params + [logits]
        optimizer = torch.optim.AdamW(
            [
                {"params": model_params, "lr": float(args.method2_lr)},
                {"params": [logits], "lr": float(args.method2_p_lr)},
            ],
            weight_decay=float(args.method2_weight_decay),
        )
        hist_stride = max(1, int(args.method2_log_every))
        best_train_loss = float("inf")
        best_train_epoch = 0
        best_train_state: dict[str, Any] = {
            "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "logits": logits.detach().cpu().clone(),
        }
        last_improve_step = 0
        last_p_change_step = 0
        prev_p_text = ""
        stopped_early = False
        stop_reason = ""

        for step in range(1, int(args.method2_adam_steps) + 1):
            optimizer.zero_grad(set_to_none=True)
            temp_now = method2_current_temperature(args, step, int(args.method2_adam_steps))
            powers, probs = powers_from_logits(logits, levels, temp_now, method2_power_map_mode(args))
            terminal = terminal_loss_for_model_subset(
                model, tau, y_power, y_real, y_imag, powers,
                zeta=z_max_ld, n_points=int(args.method2_terminal_points), observable=observable,
            )
            ic = ic_loss_for_model(
                model, powers, centers, t_min=t_min, t_max=t_max, n_points=int(args.method2_ic_points)
            )
            nlse = nlse_loss_any_model(
                model, pde_params, powers, z_max_ld, t_min, t_max, int(args.method2_nlse_points)
            )
            ent = entropy_loss(probs)
            total = (
                float(args.method2_terminal_weight) * terminal
                + float(args.method2_ic_weight) * ic
                + float(args.method2_nlse_weight) * nlse
                + method2_entropy_weight_now(args, step, int(args.method2_adam_steps)) * ent
            )
            total.backward()
            if float(args.method2_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, float(args.method2_grad_clip))
            optimizer.step()

            total_float = float(total.detach().cpu())
            terminal_float = float(terminal.detach().cpu())
            min_delta = float(args.method2_early_stop_min_delta)
            improved = total_float < (best_train_loss - min_delta * max(1.0, abs(best_train_loss if math.isfinite(best_train_loss) else total_float)))
            if improved or not math.isfinite(best_train_loss):
                best_train_loss = total_float
                best_train_epoch = int(step)
                best_train_state = {
                    "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "logits": logits.detach().cpu().clone(),
                }
            if str(getattr(args, "method2_early_stop_monitor", "terminal")) == "terminal":
                monitor_float = terminal_float
            else:
                monitor_float = total_float
            if step == 1:
                best_monitor_loss = float("inf")
            monitor_improved = monitor_float < (best_monitor_loss - min_delta * max(1.0, abs(best_monitor_loss if math.isfinite(best_monitor_loss) else monitor_float)))
            if monitor_improved or not math.isfinite(best_monitor_loss):
                best_monitor_loss = monitor_float
                last_improve_step = int(step)
            with torch.no_grad():
                p_disc_tmp, _cls_disc, _conf = discrete_from_logits(logits, levels, temp_now)
                p_np_tmp = p_disc_tmp.detach().cpu().numpy().reshape(-1)
                p_text_now = combo_to_text(p_np_tmp)
            if p_text_now != prev_p_text:
                last_p_change_step = int(step)
                prev_p_text = p_text_now

            if step == 1 or step % hist_stride == 0 or step == int(args.method2_adam_steps):
                all_history.append({
                    "restart": int(r),
                    "step": int(step),
                    "total_loss": total_float,
                    "terminal_relative_mse_batch": float(terminal.detach().cpu()),
                    "ic_loss_batch": float(ic.detach().cpu()),
                    "nlse_loss_batch": float(nlse.detach().cpu()),
                    "entropy": float(ent.detach().cpu()),
                    "temperature": float(temp_now),
                    "p_lr": float(args.method2_p_lr),
                    "power_map_mode": method2_power_map_mode(args),
                    "estimated_powers": p_text_now,
                    "exact_match_current_discrete": int(bool(np.allclose(p_np_tmp, true_p, atol=1e-6))),
                    "early_stop_triggered": 0,
                    "stop_reason": "",
                })
                if not bool(getattr(args, "quiet_progress", False)):
                    print(
                        f"  [method2][sample {int(sample['rank']) + 1}] "
                        f"restart {r + 1}/{int(args.method2_restarts)} "
                        f"step {step}/{int(args.method2_adam_steps)} "
                        f"loss={total_float:.4e} terminal={float(terminal.detach().cpu()):.4e} "
                        f"ic={float(ic.detach().cpu()):.4e} nlse={float(nlse.detach().cpu()):.4e} "
                        f"temp={temp_now:.3f} P={p_text_now}",
                        flush=True,
                    )

            if bool(args.method2_early_stop) and step >= int(args.method2_early_stop_min_steps):
                no_improve = (int(step) - int(last_improve_step)) >= int(args.method2_early_stop_patience)
                p_stable = (int(step) - int(last_p_change_step)) >= int(args.method2_early_stop_p_stable)
                if no_improve and p_stable:
                    stopped_early = True
                    stop_reason = f"no_improve_for_{int(step)-int(last_improve_step)}_steps_and_P_stable_for_{int(step)-int(last_p_change_step)}_steps"
                    all_history.append({
                        "restart": int(r),
                        "step": int(step),
                        "total_loss": total_float,
                        "terminal_relative_mse_batch": float(terminal.detach().cpu()),
                        "ic_loss_batch": float(ic.detach().cpu()),
                        "nlse_loss_batch": float(nlse.detach().cpu()),
                        "entropy": float(ent.detach().cpu()),
                        "temperature": float(temp_now),
                        "p_lr": float(args.method2_p_lr),
                        "power_map_mode": method2_power_map_mode(args),
                        "estimated_powers": p_text_now,
                        "exact_match_current_discrete": int(bool(np.allclose(p_np_tmp, true_p, atol=1e-6))),
                        "early_stop_triggered": 1,
                        "stop_reason": stop_reason,
                    })
                    if not bool(getattr(args, "quiet_progress", False)):
                        print(f"  [method2][sample {int(sample['rank']) + 1}] restart {r + 1} early stop at step {step}: {stop_reason}", flush=True)
                    break

        if int(args.method2_lbfgs_steps) > 0:
            lbfgs = torch.optim.LBFGS(params, lr=float(args.method2_lbfgs_lr), max_iter=1, history_size=50, line_search_fn="strong_wolfe")
            for k in range(1, int(args.method2_lbfgs_steps) + 1):
                def closure() -> torch.Tensor:
                    lbfgs.zero_grad(set_to_none=True)
                    temp_c = float(args.method2_temperature_end)
                    powers, probs = powers_from_logits(logits, levels, temp_c, method2_power_map_mode(args))
                    terminal_c = terminal_loss_for_model_subset(
                        model, tau, y_power, y_real, y_imag, powers,
                        zeta=z_max_ld, n_points=int(args.method2_terminal_points), observable=observable,
                    )
                    ic_c = ic_loss_for_model(model, powers, centers, t_min=t_min, t_max=t_max, n_points=int(args.method2_ic_points))
                    nlse_c = nlse_loss_any_model(model, pde_params, powers, z_max_ld, t_min, t_max, int(args.method2_nlse_points))
                    ent_c = entropy_loss(probs)
                    total_c = (
                        float(args.method2_terminal_weight) * terminal_c
                        + float(args.method2_ic_weight) * ic_c
                        + float(args.method2_nlse_weight) * nlse_c
                        + float(args.method2_entropy_weight) * ent_c
                    )
                    total_c.backward()
                    return total_c
                loss_val = lbfgs.step(closure)
                if k == 1 or k % hist_stride == 0 or k == int(args.method2_lbfgs_steps):
                    all_history.append({
                        "restart": int(r),
                        "step": int(args.method2_adam_steps) + int(k),
                        "total_loss": float(loss_val.detach().cpu()) if torch.is_tensor(loss_val) else float(loss_val),
                        "terminal_relative_mse_batch": "",
                        "ic_loss_batch": "",
                        "nlse_loss_batch": "",
                        "entropy": "",
                        "estimated_powers": "",
                        "exact_match_current_discrete": "",
                        "early_stop_triggered": 0,
                        "stop_reason": "",
                    })

        # Evaluate the best observed training state, not necessarily the final state.
        model.load_state_dict(best_train_state["model_state"])
        logits_eval = best_train_state["logits"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            p_disc, cls_disc, conf = discrete_from_logits(logits_eval, levels, float(args.method2_temperature_end))
        eval_payload = evaluate_discrete_solution(
            model,
            pde_params,
            tau,
            y_power,
            y_real,
            y_imag,
            p_disc,
            true_p,
            true_cls,
            z_max_ld,
            t_min,
            t_max,
            int(args.forward_time_chunk),
            int(args.method2_eval_nlse_points),
            terminal_observable=observable,
            ic_points=int(args.method2_eval_ic_points),
            centers=centers,
            method2_model=True,
        )
        eval_payload.update({
            "restart": int(r),
            "seed": int(seed),
            "used_warm_start_model": bool(used_warm_start),
            "best_training_loss": float(best_train_loss),
            "best_training_epoch": int(best_train_epoch),
            "early_stop_triggered": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "mean_confidence": float(conf.mean().detach().cpu()),
            "min_confidence": float(conf.min().detach().cpu()),
        })
        eval_payload["selection_objective"] = (
            float(args.method2_terminal_weight) * eval_payload["terminal_relative_mse"]
            + float(args.method2_ic_weight) * eval_payload["ic_loss"]
            + float(args.method2_nlse_weight) * eval_payload["nlse_loss"]
        )
        restart_summaries.append(strip_large_terminal_arrays(eval_payload))
        if not bool(getattr(args, "quiet_progress", False)):
            print(
                f"  [method2][sample {int(sample['rank']) + 1}] restart {r + 1} done: "
                f"P_hat={eval_payload['estimated_powers']} exact={eval_payload['exact_match']} "
                f"per={eval_payload['per_pulse_accuracy']:.3f} obj={eval_payload['selection_objective']:.4e}",
                flush=True,
            )

        if best is None or eval_payload["selection_objective"] < best["selection_objective"]:
            best = eval_payload
            best_state = {
                "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "logits": logits_eval.detach().cpu().clone(),
                "restart": int(r),
            }

    assert best is not None and best_state is not None
    elapsed = time.time() - t_start
    tau_np_local = tau.detach().cpu().numpy().reshape(-1).astype(float)
    y_np_local = y_power.detach().cpu().numpy().reshape(-1).astype(float)
    y_real_np_local = y_real.detach().cpu().numpy().reshape(-1).astype(float) if y_real is not None else None
    y_imag_np_local = y_imag.detach().cpu().numpy().reshape(-1).astype(float) if y_imag is not None else None
    pred_terminal_np = np.asarray(best["pred_terminal_power"], dtype=float).reshape(-1)
    pred_real_np = np.asarray(best.get("pred_terminal_real", np.full_like(pred_terminal_np, np.nan)), dtype=float).reshape(-1)
    pred_imag_np = np.asarray(best.get("pred_terminal_imag", np.full_like(pred_terminal_np, np.nan)), dtype=float).reshape(-1)
    write_csv_dicts(sample_dir / "method2_waveforms.csv", [
        {
            "tau": float(tau_np_local[i]),
            "target_terminal_power": float(y_np_local[i]),
            "target_terminal_real": float(y_real_np_local[i]) if y_real_np_local is not None else "",
            "target_terminal_imag": float(y_imag_np_local[i]) if y_imag_np_local is not None else "",
            "pred_terminal_power": float(pred_terminal_np[i]),
            "pred_terminal_real": float(pred_real_np[i]),
            "pred_terminal_imag": float(pred_imag_np[i]),
            "true_initial_power": float(initial_power_np(tau_np_local, true_p, centers)[i]),
            "estimated_initial_power": float(initial_power_np(tau_np_local, best["estimated_powers"], centers)[i]),
        }
        for i in range(len(tau_np_local))
    ])

    summary = {
        "method": "method2_trajectory_PINN_and_P",
        "method2_formulation": str(getattr(args, "method2_formulation", "trajectory")),
        "sequential_warm_start": bool(getattr(args, "method2_sequential_warm_start", True)),
        "sample_rank": int(sample["rank"]),
        "dataset_index": int(sample["dataset_index"]),
        "true_powers": [float(x) for x in true_p.tolist()],
        "true_classes": [int(x) for x in true_cls.tolist()],
        "best_restart": int(best["restart"]),
        "estimated_powers": best["estimated_powers"],
        "estimated_classes": best["estimated_classes"],
        "exact_match": bool(best["exact_match"]),
        "per_pulse_accuracy": float(best["per_pulse_accuracy"]),
        "terminal_observable": str(best.get("terminal_observable", observable)),
        "terminal_relative_mse": float(best["terminal_relative_mse"]),
        "terminal_rel_l2": float(best.get("terminal_rel_l2", math.sqrt(max(0.0, float(best["terminal_relative_mse"]))))),
        "terminal_rel_l2_power": float(best["terminal_rel_l2_power"]),
        "terminal_relative_mse_power": float(best.get("terminal_relative_mse_power", best["terminal_relative_mse"])),
        "terminal_relative_mse_complex": best.get("terminal_relative_mse_complex"),
        "terminal_rel_l2_complex": best.get("terminal_rel_l2_complex"),
        "nlse_loss": float(best["nlse_loss"]),
        "ic_loss": float(best["ic_loss"]),
        "selection_objective": float(best["selection_objective"]),
        "elapsed_sec": float(elapsed),
        "all_restarts": restart_summaries,
    }
    write_csv_dicts(sample_dir / "method2_history.csv", all_history)
    write_csv_dicts(sample_dir / "method2_restart_summary.csv", restart_summaries)
    write_json(sample_dir / "method2_summary.json", summary)
    torch.save({"best_state": best_state, "summary": summary}, sample_dir / "method2_best_pinn_and_logits.pt")
    return summary, best_state


# =============================================================================
# 可视化与汇总
# =============================================================================


def plot_sample_waveforms(
    sample: dict[str, Any],
    tau_np: np.ndarray,
    y_obs: np.ndarray,
    method_summaries: dict[str, dict[str, Any]],
    out_path: Path,
    n_pulses: int,
) -> None:
    centers = pulse_centers_t0(n_pulses)
    true_p = sample["powers"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), squeeze=False)
    ax0 = axes[0, 0]
    ax1 = axes[1, 0]
    ax0.plot(tau_np, initial_power_np(tau_np, true_p, centers), label="true initial")
    for name, summary in method_summaries.items():
        if not summary:
            continue
        ax0.plot(tau_np, initial_power_np(tau_np, summary["estimated_powers"], centers), linestyle="--", label=f"{name} estimated")
    ax0.set_xlabel(r"$t/T_0$")
    ax0.set_ylabel("initial power")
    ax0.set_title(f"Initial reconstruction, idx={sample['dataset_index']}, true={combo_to_text(true_p)}")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=8)

    ax1.plot(tau_np, y_obs, label="SSFM observed terminal")
    ax1.set_xlabel(r"$t/T_0$")
    ax1.set_ylabel("terminal power")
    ax1.set_title("Observed terminal power waveform used by both inverse methods")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"skipped": True}
    n = len(rows)
    exact = np.asarray([float(bool(r["exact_match"])) for r in rows], dtype=float)
    per = np.asarray([float(r["per_pulse_accuracy"]) for r in rows], dtype=float)
    terr = np.asarray([float(r.get("terminal_rel_l2", r.get("terminal_rel_l2_power", np.nan))) for r in rows], dtype=float)
    terr_power = np.asarray([float(r.get("terminal_rel_l2_power", np.nan)) for r in rows], dtype=float)
    terr_complex_vals = [r.get("terminal_rel_l2_complex") for r in rows]
    terr_complex = np.asarray([np.nan if v is None or v == "" else float(v) for v in terr_complex_vals], dtype=float)
    elapsed = np.asarray([float(r["elapsed_sec"]) for r in rows], dtype=float)
    payload = {
        "n_samples": int(n),
        "exact_match_accuracy": float(exact.mean()),
        "exact_match_count": int(exact.sum()),
        "per_pulse_accuracy_mean": float(per.mean()),
        "terminal_rel_l2_mean": float(np.nanmean(terr)),
        "terminal_rel_l2_median": float(np.nanmedian(terr)),
        "terminal_rel_l2_power_mean": float(np.nanmean(terr_power)),
        "terminal_rel_l2_power_median": float(np.nanmedian(terr_power)),
        "elapsed_sec_total": float(elapsed.sum()),
        "elapsed_sec_mean_per_sample": float(elapsed.mean()),
    }
    if not np.all(np.isnan(terr_complex)):
        payload["terminal_rel_l2_complex_mean"] = float(np.nanmean(terr_complex))
        payload["terminal_rel_l2_complex_median"] = float(np.nanmedian(terr_complex))
    return payload


def plot_comparison(agg: dict[str, Any], out_dir: Path) -> None:
    order = ["method1_adam", "method1_enum", "method2", "method1"]
    methods = [m for m in order if m in agg and isinstance(agg[m], dict) and not agg[m].get("skipped")]
    if not methods:
        return
    label_map = {
        "method1_adam": "M1a: optimize P",
        "method1_enum": "M1b: enumerate P",
        "method1": "M1: frozen F*",
        "method2": "M2: train trajectory PINN",
    }
    labels = [label_map.get(m, m) for m in methods]
    exact = [float(agg[m]["exact_match_accuracy"]) for m in methods]
    per = [float(agg[m]["per_pulse_accuracy_mean"]) for m in methods]
    x = np.arange(len(methods), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    ax.bar(x - width / 2, exact, width, label="exact full-P accuracy")
    ax.bar(x + width / 2, per, width, label="per-pulse accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Inverse methods: accuracy on selected forward-unseen samples")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "two_inverse_ideas_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    times = [float(agg[m]["elapsed_sec_mean_per_sample"]) for m in methods]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, times)
    ax.set_ylabel("seconds / sample")
    ax.set_title("Inverse methods: mean runtime")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "two_inverse_ideas_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def flatten_summary_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "method": method,
            "sample_rank": int(r["sample_rank"]),
            "dataset_index": int(r["dataset_index"]),
            "true_powers": combo_to_text(r["true_powers"]),
            "estimated_powers": combo_to_text(r["estimated_powers"]),
            "true_classes": ";".join(str(int(x)) for x in r["true_classes"]),
            "estimated_classes": ";".join(str(int(x)) for x in r["estimated_classes"]),
            "exact_match": int(bool(r["exact_match"])),
            "per_pulse_accuracy": float(r["per_pulse_accuracy"]),
            "terminal_observable": str(r.get("terminal_observable", "power")),
            "terminal_relative_mse": float(r["terminal_relative_mse"]),
            "terminal_rel_l2": float(r.get("terminal_rel_l2", math.sqrt(max(0.0, float(r["terminal_relative_mse"]))))),
            "terminal_rel_l2_power": float(r["terminal_rel_l2_power"]),
            "terminal_relative_mse_power": float(r.get("terminal_relative_mse_power", r["terminal_relative_mse"])),
            "terminal_rel_l2_complex": "" if r.get("terminal_rel_l2_complex") is None else float(r.get("terminal_rel_l2_complex")),
            "terminal_relative_mse_complex": "" if r.get("terminal_relative_mse_complex") is None else float(r.get("terminal_relative_mse_complex")),
            "nlse_loss": float(r["nlse_loss"]),
            "ic_loss": float(r.get("ic_loss", 0.0)),
            "elapsed_sec": float(r["elapsed_sec"]),
            "best_restart": int(r["best_restart"]),
        })
    return out


# =============================================================================
# 主流程
# =============================================================================




def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    d = dict(vars(args))
    d.update(updates)
    return argparse.Namespace(**d)


def active_method_flags(args: argparse.Namespace) -> tuple[bool, bool, bool, bool]:
    """Return run flags: method1_adam, method1_enum, method2, method1_any.

    The old --skip-method1/--skip-method2 flags are still honored.  The new
    --run-mode makes it hard to accidentally run AdamW+enumeration when only
    enumeration was intended.
    """
    mode = str(getattr(args, "run_mode", "all"))
    if mode == "method1_adam":
        return (not bool(args.skip_method1), False, False, not bool(args.skip_method1))
    if mode == "method1_enum":
        return (False, not bool(args.skip_method1), False, not bool(args.skip_method1))
    if mode == "method1_both":
        return (not bool(args.skip_method1), not bool(args.skip_method1), False, not bool(args.skip_method1))
    if mode == "method2":
        return (False, False, not bool(args.skip_method2), False)
    # all: method1_adam is the default method1 branch; enum is added only if explicitly requested.
    run_m1_adam = not bool(args.skip_method1) and int(args.method1_restarts) > 0
    run_m1_enum = not bool(args.skip_method1) and bool(args.method1_enumerate_candidates)
    run_m2 = not bool(args.skip_method2)
    return run_m1_adam, run_m1_enum, run_m2, (run_m1_adam or run_m1_enum)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two pure-physics inverse ideas on forward-unseen multi-pulse SSFM samples.")
    p.add_argument("--run-dir", required=True, help="已有完整正向实验目录，例如 MULTIPULSE_FULL_RUNS/M4_full")
    p.add_argument("--n-pulses", "-M", type=int, default=0, help="脉冲数量 M；默认从 run_dir/ckpt 自动推断")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="", help="默认 run_dir/inverse/two_inverse_ideas_unseen<N>_seed<S>")
    p.add_argument(
        "--run-mode",
        choices=["all", "method1_adam", "method1_enum", "method1_both", "method2"],
        default="all",
        help=(
            "选择本次只跑哪个逆向流程："
            "method1_adam=冻结F*梯度优化P；"
            "method1_enum=冻结F*枚举P；"
            "method1_both=分别输出adam和enum两套结果；"
            "method2=只跑trajectory PINN；all=默认按skip参数跑。"
        ),
    )

    # Forward / dataset discovery.
    p.add_argument("--forward-model-path", default="", help="直接指定正向 forward_pinn.pt；否则按 metrics_stream.csv 选平均 rel_l2_power 最小者")
    p.add_argument("--forward-metrics-csv", default="", help="可选：指定 forward metrics_stream.csv")
    p.add_argument("--max-forward-mean-rel-l2-power", type=float, default=0.0, help="正向模型质量门槛；0=不启用。启用后所选 F* 的 mean rel_l2_power 超过该值会停止逆向。")
    p.add_argument("--max-forward-p95-rel-l2-power", type=float, default=0.0, help="正向模型质量门槛；0=不启用。启用后所选 F* 的 p95 rel_l2_power 超过该值会停止逆向。")
    p.add_argument("--inverse-dataset-dir", default="", help="可选：指定 SSFM terminal dataset 目录")
    p.add_argument("--build-dataset-if-missing", action="store_true", default=True)
    p.add_argument("--no-build-dataset", action="store_false", dest="build_dataset_if_missing")
    p.add_argument("--rebuild-inverse-dataset", action="store_true", help="强制重建 SSFM terminal dataset")
    p.add_argument("--inverse-dataset-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--inverse-dataset-mode", choices=["selected", "full"], default="selected", help="selected=先选测试样本，只生成这些样本的SSFM输出；full=生成全部4^M组合。当前逆向对比建议selected，M8会快很多。")
    p.add_argument("--terminal-observable", choices=["power", "complex", "power_and_complex"], default="power", help="逆向终端切片损失使用什么观测量：power=只用功率；complex=用 SSFM 输出复场实部/虚部；power_and_complex=两者平均。")
    p.add_argument("--save-terminal-field", action="store_true", help="构建 inverse SSFM dataset 时保存 Y_terminal_real.npy 和 Y_terminal_imag.npy。--terminal-observable complex/power_and_complex 会自动启用。")
    p.add_argument("--save-initial-power", action="store_true", default=True, help="构建 inverse SSFM dataset 时额外保存 X_initial_power.npy；它可由 P 重构，但便于检查。")
    p.add_argument("--no-save-initial-power", action="store_false", dest="save_initial_power")
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--inverse-dataset-log-every", type=int, default=20)
    p.add_argument("--inverse-max-combos", type=int, default=0, help="0=全部 4^M；调试大 M 时才建议 >0")
    p.add_argument("--verbose-ssfm", action="store_true")

    # If dataset must be built and run_manifest is incomplete.
    p.add_argument("--t-window-t0", type=float, default=0.0, help="SSFM half-window；默认从 run_manifest/ssfm_eval_grid 推断")
    p.add_argument("--n-t", type=int, default=0)
    p.add_argument("--n-z", type=int, default=0)
    p.add_argument("--z-max-ld", type=float, default=0.0)
    p.add_argument("--compare-t-min", default="")
    p.add_argument("--compare-t-max", default="")

    # Sample selection.
    p.add_argument("--unseen-csv", default="", help="默认 run_dir/dataset/unseen_combinations.csv")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--sample-indices", default="", help="直接指定 inverse dataset 全局 index，空格或逗号分隔；指定后不检查 unseen_csv")
    p.add_argument("--plot-examples", type=int, default=4)

    # Shared physics / power mapping.
    p.add_argument("--power-map-mode", choices=["straight_through", "soft"], default="straight_through")
    p.add_argument("--power-temperature", type=float, default=1.0)
    p.add_argument("--nlse-t-min", default="")
    p.add_argument("--nlse-t-max", default="")
    p.add_argument("--forward-time-chunk", type=int, default=512)
    p.add_argument("--skip-method1", action="store_true")
    p.add_argument("--skip-method2", action="store_true")
    p.add_argument("--quiet-progress", action="store_true", help="关闭每个 restart/epoch 的控制台进度输出。")

    # Method 1: frozen forward model, trainable P.
    p.add_argument("--method1-epochs", type=int, default=3000)
    p.add_argument("--method1-restarts", type=int, default=8)
    p.add_argument("--method1-lr", type=float, default=3e-2)
    p.add_argument("--method1-terminal-weight", type=float, default=1.0)
    p.add_argument("--method1-ic-weight", type=float, default=10.0, help="method1 的 z=0 初始条件一致性权重；不使用真实 P，只比较 F*(0,t,P) 与由当前 P 构造的 A0(t;P)。")
    p.add_argument("--method1-nlse-weight", type=float, default=10.0)
    p.add_argument("--method1-entropy-weight", type=float, default=0.0)
    p.add_argument("--method1-terminal-points", type=int, default=0, help="0=每步使用完整终端切片；>0=每步随机采样这些 t 点以加速，最终评价仍用完整切片")
    p.add_argument("--method1-ic-points", type=int, default=512)
    p.add_argument("--method1-nlse-points", type=int, default=16)
    p.add_argument("--method1-eval-ic-points", type=int, default=512)
    p.add_argument("--method1-eval-nlse-points", type=int, default=64)
    p.add_argument("--method1-grad-clip", type=float, default=1.0)
    p.add_argument("--method1-log-every", type=int, default=50)
    p.add_argument("--method1-carry-state", action="store_true", default=False, help="不建议用于method1正式实验；仅把上一样本P logits作为下一样本restart-0初值。默认关闭。")
    p.add_argument("--method1-no-carry-state", action="store_false", dest="method1_carry_state")
    p.add_argument("--method1-early-stop", action="store_true", default=True, help="按 loss 无改进且离散 P 稳定来提前停止；不保证全局最优。")
    p.add_argument("--method1-no-early-stop", action="store_false", dest="method1_early_stop")
    p.add_argument("--method1-early-stop-min-epochs", type=int, default=500)
    p.add_argument("--method1-early-stop-patience", type=int, default=300)
    p.add_argument("--method1-early-stop-p-stable", type=int, default=300)
    p.add_argument("--method1-early-stop-min-delta", type=float, default=1e-3)
    p.add_argument("--method1-early-stop-monitor", choices=["terminal", "total"], default="terminal", help="早停监控量。terminal更稳定、更快触发；total会受随机NLSE/IC采样波动影响。")
    p.add_argument("--method1-enumerate-candidates", action="store_true", help="对 4^M 个离散候选 P 做冻结 F* 终端误差枚举；M=4/5/6 强烈建议开启，用于排除优化未收敛。")
    p.add_argument("--method1-select-enumeration-if-better", action="store_true", default=True)
    p.add_argument("--method1-no-select-enumeration-if-better", action="store_false", dest="method1_select_enumeration_if_better")
    p.add_argument("--method1-enumeration-max-combos", type=int, default=4096)
    p.add_argument("--method1-enumeration-batch-size", type=int, default=1024)
    p.add_argument("--method1-save-enumeration-topk", type=int, default=10)

    # Method 2: train a new PINN and P from terminal slice.
    p.add_argument("--method2-formulation", choices=["trajectory", "conditional"], default="trajectory", help="method2 network formulation. trajectory: A_theta(z,t) with no P input, P only appears in IC loss; conditional: old A_theta(z,t,P).")
    p.add_argument("--method2-restarts", type=int, default=8)
    p.add_argument("--method2-adam-steps", type=int, default=2500)
    p.add_argument("--method2-lbfgs-steps", type=int, default=0)
    p.add_argument("--method2-lr", type=float, default=1e-3, help="method2 PINN network AdamW learning rate")
    p.add_argument("--method2-p-lr", type=float, default=3e-2, help="method2 P-logits AdamW learning rate; should usually be much larger than --method2-lr")
    p.add_argument("--method2-power-map-mode", choices=["inherit", "straight_through", "soft"], default="soft", help="method2-only P relaxation. soft is recommended so P can move between PAM4 levels before final discretization.")
    p.add_argument("--method2-temperature-start", type=float, default=2.0, help="method2 softmax temperature at Adam step 1")
    p.add_argument("--method2-temperature-end", type=float, default=0.25, help="method2 softmax temperature at final Adam/L-BFGS evaluation")
    p.add_argument("--method2-entropy-start-frac", type=float, default=0.5, help="fraction of Adam steps before turning on method2 entropy regularization")
    p.add_argument("--method2-lbfgs-lr", type=float, default=1.0)
    p.add_argument("--method2-weight-decay", type=float, default=0.0)
    p.add_argument("--method2-grad-clip", type=float, default=1.0)
    p.add_argument("--method2-log-every", type=int, default=50)
    p.add_argument("--method2-early-stop", action="store_true", default=True, help="按 loss 无改进且离散 P 稳定来提前停止；不保证全局最优。")
    p.add_argument("--method2-no-early-stop", action="store_false", dest="method2_early_stop")
    p.add_argument("--method2-early-stop-min-steps", type=int, default=1000)
    p.add_argument("--method2-early-stop-patience", type=int, default=500)
    p.add_argument("--method2-early-stop-p-stable", type=int, default=500)
    p.add_argument("--method2-early-stop-min-delta", type=float, default=1e-3)
    p.add_argument("--method2-early-stop-monitor", choices=["terminal", "total"], default="terminal", help="早停监控量。terminal更稳定；total更严格。")
    p.add_argument("--method2-sequential-warm-start", action="store_true", default=True, help="按用户方案2：sample k训练出的best模型作为sample k+1的restart-0初始化；每个样本的P logits仍重新初始化。")
    p.add_argument("--method2-no-sequential-warm-start", action="store_false", dest="method2_sequential_warm_start")
    p.add_argument("--method2-hidden", type=int, default=100, help="<=0 表示沿用所选 forward ckpt 配置")
    p.add_argument("--method2-layers", type=int, default=4, help="<=0 表示沿用所选 forward ckpt 配置")
    p.add_argument("--method2-fourier-features", type=int, default=-1, help="-1 表示沿用所选 forward ckpt 配置")
    p.add_argument("--method2-terminal-weight", type=float, default=1.0)
    p.add_argument("--method2-ic-weight", type=float, default=10.0)
    p.add_argument("--method2-nlse-weight", type=float, default=10.0)
    p.add_argument("--method2-entropy-weight", type=float, default=1e-4)
    p.add_argument("--method2-terminal-points", type=int, default=128)
    p.add_argument("--method2-ic-points", type=int, default=128)
    p.add_argument("--method2-nlse-points", type=int, default=64)
    p.add_argument("--method2-eval-nlse-points", type=int, default=128)
    p.add_argument("--method2-eval-ic-points", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = safe_device(args.device)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir 不存在：{run_dir}")

    forward_ckpt, forward_label, forward_info = select_best_forward(args)
    enforce_forward_quality_gate(args, forward_info)
    M = infer_m_from_sources(args, run_dir, forward_ckpt)
    if not args.out_dir:
        obs_slug = terminal_observable_mode(args).replace("_", "")
        suffix = "" if obs_slug == "power" else f"_{obs_slug}"
        out_dir = run_dir / "inverse" / f"two_inverse_ideas{suffix}_unseen{int(args.n_samples)}_seed{int(args.sample_seed)}"
    else:
        out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("\n========== Two inverse ideas on forward-unseen samples ==========")
    print(f"run_dir          = {run_dir}")
    print(f"M                = {M}")
    print(f"device           = {device}")
    print(f"selected F*      = {forward_label}: {forward_ckpt}")
    if isinstance(forward_info.get("selected"), dict):
        sel = forward_info["selected"]
        print(
            "selected F* metric = "
            f"mean_rel_l2_power={float(sel.get('mean_rel_l2_power', float('nan'))):.6g}, "
            f"p95={float(sel.get('p95_rel_l2_power', float('nan'))):.6g}, "
            f"max={float(sel.get('max_rel_l2_power', float('nan'))):.6g}"
        )
    print(f"out_dir          = {out_dir}")

    # Load frozen forward F* for method1 and for method2 config reference.
    forward_model = load_forward_checkpoint(forward_ckpt, device=device)
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad_(False)
    pde_params = load_pde_params_from_forward_checkpoint(forward_ckpt)

    # Select samples before building SSFM data.  In v8 selected-only mode, this
    # lets the builder generate only the 10 requested SSFM terminal fields rather
    # than all 4^M combinations.
    selected_samples_pre = preselect_unseen_samples(args, run_dir, M, out_dir)
    selected_global_indices = [int(s["dataset_index"]) for s in selected_samples_pre]
    setattr(args, "selected_dataset_indices", ",".join(str(x) for x in selected_global_indices))
    setattr(args, "selected_indices_source", str(out_dir / "selected_unseen_samples.json"))

    # Load / build inverse SSFM terminal dataset.
    ds_root, ds_meta = get_or_build_inverse_dataset(args, run_dir, M, forward_ckpt, device)
    tau_np = np.load(ds_root / "tau_input.npy").astype(np.float32)
    tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
    Y_mem = np.load(ds_root / "Y_terminal_power.npy", mmap_mode="r")
    need_field = terminal_observable_mode(args) in {"complex", "power_and_complex"}
    Yr_mem = None
    Yi_mem = None
    if (ds_root / "Y_terminal_real.npy").exists() and (ds_root / "Y_terminal_imag.npy").exists():
        Yr_mem = np.load(ds_root / "Y_terminal_real.npy", mmap_mode="r")
        Yi_mem = np.load(ds_root / "Y_terminal_imag.npy", mmap_mode="r")
    if need_field and (Yr_mem is None or Yi_mem is None):
        raise FileNotFoundError(
            f"当前 --terminal-observable={terminal_observable_mode(args)} 需要复场终端切片，"
            f"但 {ds_root} 下没有 Y_terminal_real.npy / Y_terminal_imag.npy。"
            "请加 --rebuild-inverse-dataset 或指定新的 --inverse-dataset-dir。"
        )

    selected_samples = attach_dataset_row_indices(selected_samples_pre, ds_root, out_dir)
    print(f"inverse SSFM dataset = {ds_root}")
    print(f"selected samples     = {len(selected_samples)} from forward-unseen combinations")
    print(f"selected indices     = {selected_global_indices}")

    base_cfg = conditional_pinn_config_from_checkpoint(forward_ckpt, M, args)

    run_config = {
        "script": Path(__file__).name,
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "device_requested": str(args.device),
        "device_actual": str(device),
        "n_pulses": int(M),
        "selected_forward": {"label": forward_label, "checkpoint": str(forward_ckpt), "selection_info": forward_info},
        "inverse_dataset_dir": str(ds_root),
        "inverse_dataset_meta": ds_meta,
        "sample_rule": "randomly select from run_dir/dataset/unseen_combinations.csv, i.e. forward-PINN unseen 90%; no fixed 25% inverse test split",
        "method1": {
            "description": "Frozen selected forward PINN F*, optimize only 4-PAM amplitude logits. Carry-state can pass the best logits from sample k to sample k+1.",
            "epochs": int(args.method1_epochs),
            "restarts": int(args.method1_restarts),
            "terminal_points_per_step": int(args.method1_terminal_points),
            "carry_state": bool(args.method1_carry_state),
            "terminal_observable": terminal_observable_mode(args),
            "loss": "lambda_T terminal_observable_loss + lambda_IC IC_consistency(F*(0,t,P), A0(t;P)) + lambda_f NLSE(F*) + lambda_H entropy",
        },
        "method2": {
            "description": "Train a fresh single-sample trajectory PINN A_theta(z,t) plus 4-PAM amplitude logits from terminal slice; no pretrained forward propagator is used. P is not a network input in the default formulation.",
            "formulation": str(getattr(args, "method2_formulation", "trajectory")),
            "model_config": base_cfg,
            "adam_steps": int(args.method2_adam_steps),
            "lbfgs_steps": int(args.method2_lbfgs_steps),
            "restarts": int(args.method2_restarts),
            "model_lr": float(args.method2_lr),
            "p_lr": float(args.method2_p_lr),
            "power_map_mode": method2_power_map_mode(args),
            "temperature_start": float(args.method2_temperature_start),
            "temperature_end": float(args.method2_temperature_end),
            "terminal_observable": terminal_observable_mode(args),
            "loss": "lambda_T terminal_observable_loss + lambda_IC IC_consistency(A_theta(0,t), A0(t;P)) + lambda_f NLSE + lambda_H entropy",
        },
        "cli_args": vars(args),
    }
    write_json(out_dir / "two_inverse_ideas_run_config.json", run_config)

    run_m1_adam, run_m1_enum, run_m2, run_m1_any = active_method_flags(args)
    method1_adam_dir = ensure_dir(out_dir / "m1_adam")
    method1_enum_dir = ensure_dir(out_dir / "m1_enum")
    method2_dir = ensure_dir(out_dir / "m2")
    method1_adam_rows: list[dict[str, Any]] = []
    method1_enum_rows: list[dict[str, Any]] = []
    method2_rows: list[dict[str, Any]] = []
    method2_warm_state: dict[str, Any] | None = None

    if run_m1_adam and int(args.method1_restarts) <= 0:
        raise ValueError("run-mode method1_adam/method1_both 需要 --method1-restarts > 0。纯枚举请用 --run-mode method1_enum。")
    if not (run_m1_adam or run_m1_enum or run_m2):
        raise ValueError("没有任何方法需要运行。请检查 --run-mode 与 --skip-method1/--skip-method2。")

    print(f"run_mode         = {getattr(args, 'run_mode', 'all')}")
    print(f"active methods   = method1_adam={run_m1_adam}, method1_enum={run_m1_enum}, method2={run_m2}")

    t_all = time.time()
    for k, sample in enumerate(selected_samples):
        idx = int(sample["dataset_index"])
        row_idx = int(sample.get("row_index", idx))
        y_np = np.asarray(Y_mem[row_idx], dtype=np.float32)
        y_power = torch.tensor(y_np.reshape(1, -1), dtype=torch.float32, device=device)
        yr_tensor = None
        yi_tensor = None
        if Yr_mem is not None and Yi_mem is not None:
            yr_np = np.asarray(Yr_mem[row_idx], dtype=np.float32)
            yi_np = np.asarray(Yi_mem[row_idx], dtype=np.float32)
            yr_tensor = torch.tensor(yr_np.reshape(1, -1), dtype=torch.float32, device=device)
            yi_tensor = torch.tensor(yi_np.reshape(1, -1), dtype=torch.float32, device=device)
        print("\n" + "=" * 80)
        print(f"Sample {k+1}/{len(selected_samples)} | dataset idx={idx} | true P={sample['powers']}")

        sample_method_plot_summaries: dict[str, dict[str, Any]] = {}

        if run_m1_adam:
            print("[method1a] frozen F* + AdamW optimize trainable P logits ...")
            adam_args = clone_args(args, method1_enumerate_candidates=False, method1_carry_state=False)
            s1a, _logits_unused = run_method1_single_sample(
                sample, forward_model, pde_params, tau, y_power, yr_tensor, yi_tensor,
                adam_args, device, method1_adam_dir, None,
            )
            s1a["method_variant"] = "method1a_adam_optimize_P"
            method1_adam_rows.append(s1a)
            sample_method_plot_summaries["method1a_adam"] = s1a
            print(f"[method1a] P_hat={s1a['estimated_powers']} exact={s1a['exact_match']} per={s1a['per_pulse_accuracy']:.3f} time={s1a['elapsed_sec']:.1f}s")

        if run_m1_enum:
            print("[method1b] frozen F* + exhaustive discrete P enumeration ...")
            enum_args = clone_args(
                args,
                method1_restarts=0,
                method1_enumerate_candidates=True,
                method1_carry_state=False,
                method1_select_enumeration_if_better=True,
            )
            s1b, _enum_logits_unused = run_method1_single_sample(
                sample, forward_model, pde_params, tau, y_power, yr_tensor, yi_tensor,
                enum_args, device, method1_enum_dir, None,
            )
            s1b["method_variant"] = "method1b_exhaustive_enum_P"
            method1_enum_rows.append(s1b)
            sample_method_plot_summaries["method1b_enum"] = s1b
            print(f"[method1b] P_hat={s1b['estimated_powers']} exact={s1b['exact_match']} per={s1b['per_pulse_accuracy']:.3f} time={s1b['elapsed_sec']:.1f}s")

        if run_m2:
            print(f"[method2] train {str(getattr(args, 'method2_formulation', 'trajectory'))} PINN from terminal slice + trainable P ...")
            s2, method2_warm_state = run_method2_single_sample(
                sample, base_cfg, pde_params, tau, y_power, yr_tensor, yi_tensor,
                args, device, method2_dir, method2_warm_state if bool(getattr(args, "method2_sequential_warm_start", True)) else None,
            )
            method2_rows.append(s2)
            sample_method_plot_summaries["method2"] = s2
            if bool(getattr(args, "method2_sequential_warm_start", True)):
                torch.save({"warm_start_state": method2_warm_state, "after_sample_rank": int(sample["rank"]), "after_dataset_index": idx}, method2_dir / "warm.pt")
            print(f"[method2] P_hat={s2['estimated_powers']} exact={s2['exact_match']} per={s2['per_pulse_accuracy']:.3f} time={s2['elapsed_sec']:.1f}s")

        if int(args.plot_examples) > 0 and k < int(args.plot_examples):
            plot_sample_waveforms(
                sample,
                tau_np,
                y_np,
                sample_method_plot_summaries,
                out_dir / f"s{k:02d}_i{idx}_waves.png",
                M,
            )

    flat_rows = (
        flatten_summary_rows("method1_adam", method1_adam_rows)
        + flatten_summary_rows("method1_enum", method1_enum_rows)
        + flatten_summary_rows("method2", method2_rows)
    )
    write_csv_dicts(out_dir / "two_inverse_ideas_per_sample_summary.csv", flat_rows)

    agg = {
        "method1_adam": aggregate_rows(method1_adam_rows),
        "method1_enum": aggregate_rows(method1_enum_rows),
        "method2": aggregate_rows(method2_rows),
        "total_elapsed_sec": float(time.time() - t_all),
        "n_selected_samples": int(len(selected_samples)),
        "selected_samples": selected_samples,
        "outputs": {
            "per_sample_csv": str(out_dir / "two_inverse_ideas_per_sample_summary.csv"),
            "accuracy_plot": str(out_dir / "two_inverse_ideas_accuracy.png"),
            "time_plot": str(out_dir / "two_inverse_ideas_time.png"),
        },
    }
    write_json(out_dir / "two_inverse_ideas_summary.json", agg)
    plot_comparison(agg, out_dir)

    print("\n========== Finished ==========")
    print(f"outputs -> {out_dir}")
    print(json.dumps({k: v for k, v in agg.items() if k in {"method1_adam", "method1_enum", "method2", "total_elapsed_sec"}}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
