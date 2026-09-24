# -*- coding: utf-8 -*-
"""
inverse_universal_unknownK_continuous_v3.py

冻结一个固定 8 槽位、K=1...8 的通用正向 PINN，同时反演：
1) 未知脉冲数量 K；
2) 激活脉冲的连续归一化场幅度 A in [A_min, 1]。

目标数据：
- 随机产生 true K；
- 按训练阶段相同的连续槽位规则，将 K 个脉冲嵌入固定 8 个中心；
- 用 SSFM 生成终端复场；
- 不向逆向优化器提供 true K 或 true A。

逆向方法：
- 枚举候选 K=1,...,8（只有 8 个模型阶数候选）；
- 对每个候选 K 使用多 restart AdamW 连续优化其激活幅度；
- 冻结通用正向 PINN，仅更新幅度变量；
- 默认用 BIC 型准则在 8 个候选 K 中选择预测 K，避免更大 K 因参数更多而总是占优；
- 将最终预测的 K 与幅度重新送入 SSFM，计算物理回代误差。

每个随机样本重点输出两个误差：
- best_restart_forward_inverse_loss：冻结正向 PINN 上的最小相对 MSE；
- ssfm_reconstruction_output_rel_l2：预测参数经 SSFM 回代后的终端相对 L2。

V2 新增：
- 每个样本单独列出真实 K、真实激活幅度、预测 K、预测激活幅度；
- 生成易读 CSV 与详细 PNG 表格；
- 表格只展示对应 K 的激活幅度，不再用 8 槽零填充向量造成阅读困难。

V4 新增：
- 支持 --sample-batch-size，将多个目标样本与其 K/restart 轨迹联合成一个 GPU 批次；
  例如 sample_batch_size=2、restarts=4 时，每批为 2×8×4=64 条轨迹；
- 每个样本保留独立的模型阶数选择、历史最优值和早停状态；已停止样本不再参与后续前向/反向计算；
- 计时采用实际批次墙钟时间按批内样本数均摊，能够反映跨样本并行后的真实吞吐率。

V3 新增：
- 支持大量 restart 的“轨迹微批次”计算。总轨迹数仍为 8×restarts，
  但每次只把 --trajectory-batch-size 条轨迹送入冻结正向网络，避免 6 GB 显存爆炸；
- 新增基于当前模型阶数选择结果的早停，不再等待几乎所有失败轨迹都停止改善；
- 逐样本记录逆向优化时间、SSFM 回代时间和总处理时间；
- 详细 CSV/PNG 表格中直接显示时间。

重要说明：
- 0 表示非激活槽位；激活槽位必须大于等于 --min-active-amplitude。
- 如果允许激活幅度无限接近 0，则 K 在物理上不可稳定辨识：一个额外的极弱脉冲
  与“没有该脉冲”几乎无法区分。因此默认检测下限设为 0.05。
- 四档训练值 0.25/0.5/0.75/1.0 是归一化场幅度；本脚本的连续幅度也按场幅度解释。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nlse import NLSEParams, pulse_centers_t0
from ssfm import run_ssfm
from train_multi_pulse_pinn import load_forward_checkpoint


# -----------------------------------------------------------------------------
# 基础工具
# -----------------------------------------------------------------------------


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def vector_text(x: Sequence[float]) -> str:
    return ";".join(f"{float(v):.8g}" for v in x)


def active_amplitudes_text(amplitudes8: Sequence[float], k: int, decimals: int = 5) -> str:
    """Return only the K active amplitudes, in physical slot order."""
    arr = np.asarray(amplitudes8, dtype=np.float64).reshape(-1)
    slots = active_slots_for_k(int(k), 8)
    values = [float(arr[s]) for s in slots]
    return "[" + ", ".join(f"{v:.{int(decimals)}f}" for v in values) + "]"


def active_slots_text(k: int) -> str:
    """Human-readable one-based slot indices for the contiguous K-pulse block."""
    return "[" + ", ".join(str(s + 1) for s in active_slots_for_k(int(k), 8)) + "]"


def wrap_amplitudes_for_table(text: str, items_per_line: int = 4) -> str:
    """Wrap a bracketed amplitude list so the PNG table remains readable."""
    content = str(text).strip().strip("[]")
    if not content:
        return "[]"
    items = [x.strip() for x in content.split(",") if x.strip()]
    lines = [", ".join(items[i:i + int(items_per_line)]) for i in range(0, len(items), int(items_per_line))]
    return "[" + ",\n".join(lines) + "]"


def active_slots_for_k(k: int, n_slots: int = 8) -> tuple[int, ...]:
    """与通用训练脚本相同：从 K=8 开始，交替删除左、右端槽位。"""
    k = int(k)
    if not 1 <= k <= int(n_slots):
        raise ValueError(f"K must lie in [1,{n_slots}], got {k}.")
    left, right = 0, int(n_slots) - 1
    remove_left = True
    while right - left + 1 > k:
        if remove_left:
            left += 1
        else:
            right -= 1
        remove_left = not remove_left
    return tuple(range(left, right + 1))


def mask_for_k(k: int, n_slots: int = 8) -> np.ndarray:
    mask = np.zeros(int(n_slots), dtype=np.float32)
    mask[list(active_slots_for_k(k, n_slots))] = 1.0
    return mask


def relative_l2_complex(pred_r: np.ndarray, pred_i: np.ndarray, ref_r: np.ndarray, ref_i: np.ndarray) -> float:
    num = np.sum((pred_r - ref_r) ** 2 + (pred_i - ref_i) ** 2)
    den = np.sum(ref_r ** 2 + ref_i ** 2) + 1e-12
    return float(np.sqrt(num / den))


def relative_l2_power(pred_r: np.ndarray, pred_i: np.ndarray, ref_r: np.ndarray, ref_i: np.ndarray) -> float:
    pred_p = pred_r ** 2 + pred_i ** 2
    ref_p = ref_r ** 2 + ref_i ** 2
    return float(np.linalg.norm(pred_p - ref_p) / (np.linalg.norm(ref_p) + 1e-12))


def interpolate_terminal_field(
    tau_full: np.ndarray,
    field_final: np.ndarray,
    tau_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real = np.interp(tau_input, tau_full, np.real(field_final)).astype(np.float32)
    imag = np.interp(tau_input, tau_full, np.imag(field_final)).astype(np.float32)
    power = (real ** 2 + imag ** 2).astype(np.float32)
    return power, real, imag


# -----------------------------------------------------------------------------
# SSFM 目标数据
# -----------------------------------------------------------------------------


def sample_true_k_values(n_samples: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    n_samples = int(n_samples)
    mode = str(mode).strip().lower()
    if mode == "uniform":
        return rng.integers(1, 9, size=n_samples, endpoint=False, dtype=np.int64)
    if mode != "balanced":
        raise ValueError("--k-sampling must be balanced or uniform.")
    values: list[int] = []
    while len(values) + 8 <= n_samples:
        values.extend(range(1, 9))
    remaining = n_samples - len(values)
    if remaining > 0:
        values.extend(rng.choice(np.arange(1, 9), size=remaining, replace=False).tolist())
    arr = np.asarray(values, dtype=np.int64)
    rng.shuffle(arr)
    return arr


def build_or_load_targets(
    *,
    out_dir: Path,
    n_samples: int,
    sample_seed: int,
    k_sampling: str,
    min_active_amplitude: float,
    inverse_input_points: int,
    ssfm_half_window: float,
    compare_half_window: float,
    n_t: int,
    n_z: int,
    z_max_ld: float,
    device: torch.device,
    rebuild: bool,
    verbose_ssfm: bool,
) -> dict[str, np.ndarray]:
    ds_dir = ensure_dir(out_dir / f"targets_N{int(n_samples)}_seed{int(sample_seed)}")
    paths = {
        "K": ds_dir / "true_K.npy",
        "A8": ds_dir / "true_amplitudes_8slots.npy",
        "tau": ds_dir / "tau_input.npy",
        "power": ds_dir / "Y_terminal_power.npy",
        "real": ds_dir / "Y_terminal_real.npy",
        "imag": ds_dir / "Y_terminal_imag.npy",
        "meta": ds_dir / "meta.json",
    }
    expected = {
        "version": 1,
        "n_samples": int(n_samples),
        "sample_seed": int(sample_seed),
        "k_sampling": str(k_sampling),
        "min_active_amplitude": float(min_active_amplitude),
        "inverse_input_points": int(inverse_input_points),
        "ssfm_half_window": float(ssfm_half_window),
        "compare_half_window": float(compare_half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "z_max_ld": float(z_max_ld),
        "level_quantity": "normalized_field_amplitude",
        "centers_t0": list(pulse_centers_t0(8)),
    }
    if not rebuild and all(p.exists() for p in paths.values()):
        try:
            old = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if all(old.get(k) == v for k, v in expected.items()):
                return {
                    "K": np.load(paths["K"]),
                    "A8": np.load(paths["A8"]),
                    "tau": np.load(paths["tau"]),
                    "power": np.load(paths["power"]),
                    "real": np.load(paths["real"]),
                    "imag": np.load(paths["imag"]),
                }
        except Exception:
            pass

    rng = np.random.default_rng(int(sample_seed))
    true_k = sample_true_k_values(n_samples, rng, k_sampling)
    true_a8 = np.zeros((int(n_samples), 8), dtype=np.float32)
    for i, k in enumerate(true_k.tolist()):
        slots = active_slots_for_k(int(k), 8)
        true_a8[i, list(slots)] = rng.uniform(
            float(min_active_amplitude), 1.0, size=int(k)
        ).astype(np.float32)

    tau_input = np.linspace(
        -float(compare_half_window),
        float(compare_half_window),
        int(inverse_input_points),
        dtype=np.float32,
    )
    y_power = np.empty((int(n_samples), int(inverse_input_points)), dtype=np.float32)
    y_real = np.empty_like(y_power)
    y_imag = np.empty_like(y_power)
    centers8 = pulse_centers_t0(8)

    t0 = time.time()
    for i in range(int(n_samples)):
        params = NLSEParams.paper_pam4(
            z_max_ld=float(z_max_ld),
            t_window_t0=float(ssfm_half_window),
            n_t=int(n_t),
            n_z=int(n_z),
        ).with_multi_pulse(
            tuple(float(x) for x in true_a8[i]),
            centers_t0=centers8,
            level_mode="field",
        )
        _, t_ps, field = run_ssfm(
            params,
            device=str(device),
            save_every=int(params.n_z),
            quiet=not bool(verbose_ssfm),
        )
        tau_full = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
        p, r, im = interpolate_terminal_field(tau_full, np.asarray(field[-1]), tau_input)
        y_power[i], y_real[i], y_imag[i] = p, r, im
        print(
            f"[target SSFM {i+1}/{n_samples}] true_K={int(true_k[i])} "
            f"A8={vector_text(true_a8[i])} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    np.save(paths["K"], true_k)
    np.save(paths["A8"], true_a8)
    np.save(paths["tau"], tau_input)
    np.save(paths["power"], y_power)
    np.save(paths["real"], y_real)
    np.save(paths["imag"], y_imag)
    write_json(paths["meta"], {**expected, "build_elapsed_sec": float(time.time() - t0)})
    write_csv(
        ds_dir / "true_parameters.csv",
        [
            {
                "sample": i,
                "true_K": int(true_k[i]),
                "true_amplitudes_8slots": vector_text(true_a8[i]),
            }
            for i in range(int(n_samples))
        ],
    )
    return {
        "K": true_k,
        "A8": true_a8,
        "tau": tau_input,
        "power": y_power,
        "real": y_real,
        "imag": y_imag,
    }


# -----------------------------------------------------------------------------
# 冻结通用正向模型的可微终端传播
# -----------------------------------------------------------------------------


def terminal_field_forward_model(
    model: torch.nn.Module,
    tau: torch.Tensor,
    amplitudes8: torch.Tensor,
    zeta: float,
    chunk_t: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if amplitudes8.ndim != 2 or amplitudes8.shape[1] != 8:
        raise ValueError(f"amplitudes8 must be [B,8], got {tuple(amplitudes8.shape)}")
    bsz = int(amplitudes8.shape[0])
    us: list[torch.Tensor] = []
    vs: list[torch.Tensor] = []
    for start in range(0, int(tau.numel()), int(chunk_t)):
        end = min(int(tau.numel()), start + int(chunk_t))
        t_chunk = tau[start:end]
        nt = int(t_chunk.numel())
        z = torch.full(
            (bsz * nt, 1), float(zeta), dtype=amplitudes8.dtype, device=amplitudes8.device
        )
        t = t_chunk.reshape(1, nt, 1).expand(bsz, nt, 1).reshape(bsz * nt, 1)
        a = amplitudes8.reshape(bsz, 1, 8).expand(bsz, nt, 8).reshape(bsz * nt, 8)
        u, v = model(z, t, a)
        us.append(u.reshape(bsz, nt))
        vs.append(v.reshape(bsz, nt))
    return torch.cat(us, dim=1), torch.cat(vs, dim=1)


def loss_vectors(
    *,
    model: torch.nn.Module,
    tau: torch.Tensor,
    amplitudes8: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor,
    y_imag: torch.Tensor,
    zeta: float,
    observable: str,
    forward_time_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, v = terminal_field_forward_model(
        model, tau, amplitudes8, zeta=float(zeta), chunk_t=int(forward_time_chunk)
    )
    p = u.square() + v.square()
    power_loss = torch.sum((p - y_power).square(), dim=1) / (
        torch.sum(y_power.square(), dim=1) + 1e-12
    )
    complex_loss = torch.sum((u - y_real).square() + (v - y_imag).square(), dim=1) / (
        torch.sum(y_real.square() + y_imag.square(), dim=1) + 1e-12
    )
    mode = str(observable).strip().lower()
    if mode == "power":
        selected = power_loss
    elif mode == "complex":
        selected = complex_loss
    elif mode == "power_and_complex":
        selected = 0.5 * (power_loss + complex_loss)
    else:
        raise ValueError("observable must be power, complex, or power_and_complex")
    return selected, power_loss, complex_loss


@dataclass
class OptimizationResult:
    amplitudes8: np.ndarray
    selected_loss: np.ndarray
    power_loss: np.ndarray
    complex_loss: np.ndarray
    best_epoch: np.ndarray
    history: list[dict[str, Any]]
    stopped_epoch: int
    stopped_reason: str
    total_trajectories: int
    trajectory_batch_size: int


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x)).astype(np.float32)


def optimize_one_sample_all_k(
    *,
    sample_index: int,
    model: torch.nn.Module,
    tau: torch.Tensor,
    y_power_one: torch.Tensor,
    y_real_one: torch.Tensor,
    y_imag_one: torch.Tensor,
    device: torch.device,
    min_active_amplitude: float,
    restarts: int,
    epochs: int,
    lr: float,
    min_lr: float,
    zeta: float,
    observable: str,
    forward_time_chunk: int,
    trajectory_batch_size: int,
    log_every: int,
    early_stop_min_epochs: int,
    early_stop_patience: int,
    early_stop_fraction: float,
    early_stop_rel_delta: float,
    early_stop_abs_delta: float,
    early_stop_mode: str,
    winner_stability_patience: int,
    early_stop_score_margin: float,
    selection: str,
    bic_weight: float,
    smallest_within_fraction: float,
    n_observations: int,
    seed: int,
) -> OptimizationResult:
    """Optimize all K candidates for one target.

    Total trajectories = 8 * restarts.  They are represented by one parameter
    tensor, but frozen-forward evaluations are split into trajectory mini-batches.
    Gradients from all mini-batches are accumulated before one AdamW update, so
    the mathematics matches the original all-at-once implementation while peak
    GPU memory is bounded by ``trajectory_batch_size``.
    """
    candidate_k = np.repeat(np.arange(1, 9, dtype=np.int64), int(restarts))
    restart_id = np.tile(np.arange(int(restarts), dtype=np.int64), 8)
    bsz = len(candidate_k)
    batch_size = int(trajectory_batch_size)
    if batch_size <= 0:
        batch_size = bsz
    batch_size = min(batch_size, bsz)
    masks_np = np.stack([mask_for_k(int(k), 8) for k in candidate_k], axis=0)

    rng = np.random.default_rng(int(seed) + 10007 * int(sample_index))
    init_a = np.zeros((bsz, 8), dtype=np.float32)
    raw_np = np.zeros_like(init_a)
    for b, k in enumerate(candidate_k.tolist()):
        slots = active_slots_for_k(int(k), 8)
        vals = rng.uniform(
            float(min_active_amplitude) + 0.03, 0.97, size=int(k)
        ).astype(np.float32)
        init_a[b, list(slots)] = vals
        scaled = (vals - float(min_active_amplitude)) / max(
            1.0 - float(min_active_amplitude), 1e-8
        )
        raw_np[b, list(slots)] = logit(scaled)

    mask_t = torch.tensor(masks_np, dtype=torch.float32, device=device)
    raw = torch.nn.Parameter(torch.tensor(raw_np, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([raw], lr=float(lr), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=float(min_lr)
    )

    best_loss = torch.full((bsz,), float("inf"), device=device)
    best_raw = raw.detach().clone()
    best_epoch = torch.zeros((bsz,), dtype=torch.long, device=device)
    last_improve = torch.zeros((bsz,), dtype=torch.long, device=device)

    # Candidate-level monitoring is what matters downstream: only the best
    # restart of each K enters model-order selection.
    candidate_monitor_best = torch.full((8,), float("inf"), device=device)
    candidate_last_improve = torch.zeros((8,), dtype=torch.long, device=device)
    winner_k_last = 0
    winner_since_epoch = 1

    history: list[dict[str, Any]] = []
    stopped_epoch = int(epochs)
    stopped_reason = "max_epochs"

    def decode(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        active = float(min_active_amplitude) + (
            1.0 - float(min_active_amplitude)
        ) * torch.sigmoid(x)
        return m * active

    def candidate_minima(loss_vec: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                loss_vec[(k - 1) * int(restarts): k * int(restarts)].min()
                for k in range(1, 9)
            ]
        )

    def current_winner(candidate_losses: torch.Tensor) -> tuple[int, float]:
        losses = candidate_losses.detach().cpu().numpy().astype(np.float64)
        mode = str(selection).strip().lower()
        if mode == "min_loss":
            scores = losses.copy()
            order = np.argsort(scores)
            margin = float(scores[order[1]] - scores[order[0]])
            return int(order[0] + 1), margin
        if mode == "smallest_within":
            min_loss = float(np.min(losses))
            eligible = np.where(
                losses <= min_loss * (1.0 + float(smallest_within_fraction))
            )[0]
            return int(eligible[0] + 1), float("inf")
        if mode != "bic":
            raise ValueError("selection must be bic, min_loss, or smallest_within")
        k_arr = np.arange(1, 9, dtype=np.float64)
        scores = float(n_observations) * np.log(np.maximum(losses, 1e-15))
        scores += float(bic_weight) * k_arr * math.log(float(n_observations))
        order = np.argsort(scores)
        margin = float(scores[order[1]] - scores[order[0]])
        return int(order[0] + 1), margin

    for ep in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        current_detached = torch.empty((bsz,), dtype=torch.float32, device=device)
        current_a = torch.empty((bsz, 8), dtype=torch.float32, device=device)

        # Trajectory mini-batching. Backward after every mini-batch frees that
        # frozen-forward graph before the next mini-batch is built.
        for b0 in range(0, bsz, batch_size):
            b1 = min(b0 + batch_size, bsz)
            a8_chunk = decode(raw[b0:b1], mask_t[b0:b1])
            chunk_n = b1 - b0
            selected, _, _ = loss_vectors(
                model=model,
                tau=tau,
                amplitudes8=a8_chunk,
                y_power=y_power_one.repeat(chunk_n, 1),
                y_real=y_real_one.repeat(chunk_n, 1),
                y_imag=y_imag_one.repeat(chunk_n, 1),
                zeta=zeta,
                observable=observable,
                forward_time_chunk=forward_time_chunk,
            )
            detached = selected.detach()
            current_detached[b0:b1] = detached
            current_a[b0:b1] = a8_chunk.detach()

            old_best = best_loss[b0:b1]
            threshold = torch.maximum(
                torch.full_like(old_best, float(early_stop_abs_delta)),
                torch.abs(old_best) * float(early_stop_rel_delta),
            )
            improved = (~torch.isfinite(old_best)) | (detached < old_best - threshold)
            if bool(torch.any(improved)):
                idx_local = torch.nonzero(improved, as_tuple=False).reshape(-1)
                idx_global = idx_local + b0
                best_loss[idx_global] = detached[idx_local]
                best_epoch[idx_global] = int(ep)
                last_improve[idx_global] = int(ep)
                best_raw[idx_global] = raw.detach()[idx_global]

            # Each trajectory owns a disjoint row of raw, so accumulating the
            # chunk sums is exactly equivalent to selected.sum().backward().
            selected.sum().backward()

        optimizer.step()
        scheduler.step()

        cand_best = candidate_minima(best_loss)
        cand_threshold = torch.maximum(
            torch.full_like(candidate_monitor_best, float(early_stop_abs_delta)),
            torch.abs(candidate_monitor_best) * float(early_stop_rel_delta),
        )
        cand_improved = (~torch.isfinite(candidate_monitor_best)) | (
            cand_best < candidate_monitor_best - cand_threshold
        )
        if bool(torch.any(cand_improved)):
            candidate_monitor_best = torch.where(
                cand_improved, cand_best, candidate_monitor_best
            )
            candidate_last_improve = torch.where(
                cand_improved,
                torch.full_like(candidate_last_improve, int(ep)),
                candidate_last_improve,
            )

        winner_k, score_margin = current_winner(cand_best)
        if winner_k != winner_k_last:
            winner_k_last = int(winner_k)
            winner_since_epoch = int(ep)

        trajectory_plateau = (ep - last_improve) >= int(early_stop_patience)
        trajectory_plateau_fraction = float(
            trajectory_plateau.float().mean().detach().cpu()
        )
        candidate_plateau = (
            ep - candidate_last_improve
        ) >= int(early_stop_patience)
        candidate_plateau_fraction = float(
            candidate_plateau.float().mean().detach().cpu()
        )
        winner_plateau_for = int(
            ep - int(candidate_last_improve[winner_k - 1].detach().cpu())
        )
        winner_stable_for = int(ep - winner_since_epoch)

        if ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs):
            current_np = current_detached.detach().cpu().numpy()
            best_np = best_loss.detach().cpu().numpy()
            a_np = current_a.detach().cpu().numpy()
            for b in range(bsz):
                history.append(
                    {
                        "sample": int(sample_index),
                        "candidate_K": int(candidate_k[b]),
                        "restart": int(restart_id[b]),
                        "epoch": int(ep),
                        "current_loss": float(current_np[b]),
                        "best_loss": float(best_np[b]),
                        "best_epoch": int(best_epoch[b].detach().cpu()),
                        "amplitudes_8slots": vector_text(a_np[b]),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "trajectory_plateau_fraction": trajectory_plateau_fraction,
                        "candidate_plateau_fraction": candidate_plateau_fraction,
                        "current_selected_K": int(winner_k),
                        "selected_K_stable_epochs": int(winner_stable_for),
                        "selected_K_plateau_epochs": int(winner_plateau_for),
                        "selection_score_margin": float(score_margin),
                        "trajectory_batch_size": int(batch_size),
                        "total_trajectories": int(bsz),
                    }
                )
            print(
                f"[inverse sample {sample_index}] ep={ep}/{epochs} "
                f"mean={float(current_detached.mean().cpu()):.4e} "
                f"best_mean={float(best_loss.mean().cpu()):.4e} "
                f"traj_plateau={trajectory_plateau_fraction:.1%} "
                f"K_plateau={candidate_plateau_fraction:.1%} "
                f"winner=K{winner_k} stable={winner_stable_for} "
                f"winner_plateau={winner_plateau_for} margin={score_margin:.2f}",
                flush=True,
            )

        if ep >= int(early_stop_min_epochs):
            mode = str(early_stop_mode).strip().lower()
            should_stop = False
            reason = ""
            if mode == "trajectory_fraction":
                should_stop = (
                    trajectory_plateau_fraction >= float(early_stop_fraction)
                )
                reason = "trajectory_plateau_fraction"
            elif mode == "candidate_fraction":
                should_stop = (
                    candidate_plateau_fraction >= float(early_stop_fraction)
                )
                reason = "candidate_plateau_fraction"
            elif mode == "selection":
                margin_ok = (
                    float(early_stop_score_margin) <= 0.0
                    or float(score_margin) >= float(early_stop_score_margin)
                )
                should_stop = (
                    winner_stable_for >= int(winner_stability_patience)
                    and winner_plateau_for >= int(early_stop_patience)
                    and margin_ok
                )
                reason = "selected_K_stable_and_plateaued"
            else:
                raise ValueError(
                    "--early-stop-mode must be selection, candidate_fraction, "
                    "or trajectory_fraction"
                )
            if should_stop:
                stopped_epoch = int(ep)
                stopped_reason = reason
                print(
                    f"[inverse sample {sample_index}] early stop at epoch {ep}: "
                    f"{reason}, selected K={winner_k}, score margin={score_margin:.3f}",
                    flush=True,
                )
                break

    selected_out = torch.empty((bsz,), dtype=torch.float32, device=device)
    power_out = torch.empty_like(selected_out)
    complex_out = torch.empty_like(selected_out)
    best_a8_out = torch.empty((bsz, 8), dtype=torch.float32, device=device)
    with torch.no_grad():
        for b0 in range(0, bsz, batch_size):
            b1 = min(b0 + batch_size, bsz)
            a8_chunk = decode(best_raw[b0:b1], mask_t[b0:b1])
            chunk_n = b1 - b0
            selected, power, complex_ = loss_vectors(
                model=model,
                tau=tau,
                amplitudes8=a8_chunk,
                y_power=y_power_one.repeat(chunk_n, 1),
                y_real=y_real_one.repeat(chunk_n, 1),
                y_imag=y_imag_one.repeat(chunk_n, 1),
                zeta=zeta,
                observable=observable,
                forward_time_chunk=forward_time_chunk,
            )
            best_a8_out[b0:b1] = a8_chunk
            selected_out[b0:b1] = selected
            power_out[b0:b1] = power
            complex_out[b0:b1] = complex_

    return OptimizationResult(
        amplitudes8=best_a8_out.detach().cpu().numpy(),
        selected_loss=selected_out.detach().cpu().numpy(),
        power_loss=power_out.detach().cpu().numpy(),
        complex_loss=complex_out.detach().cpu().numpy(),
        best_epoch=best_epoch.detach().cpu().numpy(),
        history=history,
        stopped_epoch=stopped_epoch,
        stopped_reason=stopped_reason,
        total_trajectories=int(bsz),
        trajectory_batch_size=int(batch_size),
    )



def optimize_sample_batch_all_k(
    *,
    sample_indices: Sequence[int],
    model: torch.nn.Module,
    tau: torch.Tensor,
    y_power_batch: torch.Tensor,
    y_real_batch: torch.Tensor,
    y_imag_batch: torch.Tensor,
    device: torch.device,
    min_active_amplitude: float,
    restarts: int,
    epochs: int,
    lr: float,
    min_lr: float,
    zeta: float,
    observable: str,
    forward_time_chunk: int,
    trajectory_batch_size: int,
    log_every: int,
    early_stop_min_epochs: int,
    early_stop_patience: int,
    early_stop_fraction: float,
    early_stop_rel_delta: float,
    early_stop_abs_delta: float,
    early_stop_mode: str,
    winner_stability_patience: int,
    early_stop_score_margin: float,
    selection: str,
    bic_weight: float,
    smallest_within_fraction: float,
    n_observations: int,
    seed: int,
) -> list[OptimizationResult]:
    """Jointly optimize several target samples on one GPU.

    Each sample still owns ``8 * restarts`` independent trajectories.  The
    combined trajectory axis is flattened only for frozen-forward evaluation,
    so samples never share amplitudes, losses, model-order scores or early-stop
    states.  A stopped sample is removed from subsequent forward/backward
    passes while unfinished samples continue.

    With two samples and four restarts, the full joint batch contains
    ``2 * 8 * 4 = 64`` trajectories.  Setting
    ``--trajectory-batch-size 64`` evaluates them in one trajectory chunk.
    """
    sample_indices = [int(x) for x in sample_indices]
    n_batch = len(sample_indices)
    if n_batch <= 0:
        return []
    if y_power_batch.ndim != 2 or int(y_power_batch.shape[0]) != n_batch:
        raise ValueError("y_power_batch must have shape [sample_batch, Nt].")
    if y_real_batch.shape != y_power_batch.shape or y_imag_batch.shape != y_power_batch.shape:
        raise ValueError("Power, real and imaginary target batches must have matching shapes.")

    restarts = int(restarts)
    trajectories_per_sample = 8 * restarts
    candidate_k = np.repeat(np.arange(1, 9, dtype=np.int64), restarts)
    restart_id = np.tile(np.arange(restarts, dtype=np.int64), 8)
    masks_np = np.stack([mask_for_k(int(k), 8) for k in candidate_k], axis=0)
    mask_t = torch.tensor(masks_np, dtype=torch.float32, device=device)

    total_joint_trajectories = n_batch * trajectories_per_sample
    chunk_size = int(trajectory_batch_size)
    if chunk_size <= 0:
        chunk_size = total_joint_trajectories
    chunk_size = min(chunk_size, total_joint_trajectories)

    raw_np = np.zeros((n_batch, trajectories_per_sample, 8), dtype=np.float32)
    for s_local, sample_index in enumerate(sample_indices):
        # Use the same per-sample RNG rule as the original sequential function,
        # so batching does not alter initial amplitudes for a given sample.
        rng = np.random.default_rng(int(seed) + 10007 * int(sample_index))
        for tr, k in enumerate(candidate_k.tolist()):
            slots = active_slots_for_k(int(k), 8)
            vals = rng.uniform(
                float(min_active_amplitude) + 0.03, 0.97, size=int(k)
            ).astype(np.float32)
            scaled = (vals - float(min_active_amplitude)) / max(
                1.0 - float(min_active_amplitude), 1e-8
            )
            raw_np[s_local, tr, list(slots)] = logit(scaled)

    raw = torch.nn.Parameter(torch.tensor(raw_np, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([raw], lr=float(lr), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=float(min_lr)
    )

    best_loss = torch.full(
        (n_batch, trajectories_per_sample), float("inf"), device=device
    )
    best_raw = raw.detach().clone()
    best_epoch = torch.zeros(
        (n_batch, trajectories_per_sample), dtype=torch.long, device=device
    )
    last_improve = torch.zeros_like(best_epoch)

    candidate_monitor_best = torch.full((n_batch, 8), float("inf"), device=device)
    candidate_last_improve = torch.zeros((n_batch, 8), dtype=torch.long, device=device)
    winner_k_last = np.zeros(n_batch, dtype=np.int64)
    winner_since_epoch = np.ones(n_batch, dtype=np.int64)
    active = np.ones(n_batch, dtype=bool)
    stopped_epoch = np.full(n_batch, int(epochs), dtype=np.int64)
    stopped_reason = ["max_epochs" for _ in range(n_batch)]
    histories: list[list[dict[str, Any]]] = [[] for _ in range(n_batch)]

    def decode(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        active_a = float(min_active_amplitude) + (
            1.0 - float(min_active_amplitude)
        ) * torch.sigmoid(x)
        return m * active_a

    def candidate_minima_batch(loss_matrix: torch.Tensor) -> torch.Tensor:
        return loss_matrix.reshape(n_batch, 8, restarts).min(dim=2).values

    def current_winner(candidate_losses: torch.Tensor) -> tuple[int, float]:
        losses = candidate_losses.detach().cpu().numpy().astype(np.float64)
        mode = str(selection).strip().lower()
        if mode == "min_loss":
            scores = losses.copy()
            order = np.argsort(scores)
            return int(order[0] + 1), float(scores[order[1]] - scores[order[0]])
        if mode == "smallest_within":
            min_loss = float(np.min(losses))
            eligible = np.where(
                losses <= min_loss * (1.0 + float(smallest_within_fraction))
            )[0]
            return int(eligible[0] + 1), float("inf")
        if mode != "bic":
            raise ValueError("selection must be bic, min_loss, or smallest_within")
        k_arr = np.arange(1, 9, dtype=np.float64)
        scores = float(n_observations) * np.log(np.maximum(losses, 1e-15))
        scores += float(bic_weight) * k_arr * math.log(float(n_observations))
        order = np.argsort(scores)
        return int(order[0] + 1), float(scores[order[1]] - scores[order[0]])

    all_trajectory_ids = torch.arange(
        trajectories_per_sample, dtype=torch.long, device=device
    )

    for ep in range(1, int(epochs) + 1):
        active_local = np.flatnonzero(active)
        if active_local.size == 0:
            break

        optimizer.zero_grad(set_to_none=True)
        current_detached = torch.full(
            (n_batch, trajectories_per_sample), float("nan"), device=device
        )
        current_a = torch.zeros(
            (n_batch, trajectories_per_sample, 8), dtype=torch.float32, device=device
        )

        sample_ids = torch.tensor(active_local, dtype=torch.long, device=device).repeat_interleave(
            trajectories_per_sample
        )
        trajectory_ids = all_trajectory_ids.repeat(int(active_local.size))
        n_active_trajectories = int(sample_ids.numel())

        for q0 in range(0, n_active_trajectories, chunk_size):
            q1 = min(q0 + chunk_size, n_active_trajectories)
            sids = sample_ids[q0:q1]
            tids = trajectory_ids[q0:q1]
            a8_chunk = decode(raw[sids, tids], mask_t[tids])
            selected, _, _ = loss_vectors(
                model=model,
                tau=tau,
                amplitudes8=a8_chunk,
                y_power=y_power_batch.index_select(0, sids),
                y_real=y_real_batch.index_select(0, sids),
                y_imag=y_imag_batch.index_select(0, sids),
                zeta=zeta,
                observable=observable,
                forward_time_chunk=forward_time_chunk,
            )
            detached = selected.detach()
            current_detached[sids, tids] = detached
            current_a[sids, tids] = a8_chunk.detach()

            old_best = best_loss[sids, tids]
            threshold = torch.maximum(
                torch.full_like(old_best, float(early_stop_abs_delta)),
                torch.abs(old_best) * float(early_stop_rel_delta),
            )
            improved = (~torch.isfinite(old_best)) | (detached < old_best - threshold)
            if bool(torch.any(improved)):
                s_imp = sids[improved]
                t_imp = tids[improved]
                best_loss[s_imp, t_imp] = detached[improved]
                best_epoch[s_imp, t_imp] = int(ep)
                last_improve[s_imp, t_imp] = int(ep)
                best_raw[s_imp, t_imp] = raw.detach()[s_imp, t_imp]

            # Rows of raw are independent. Summing all active trajectory losses
            # gives the same row-wise gradients as separate sample optimizers.
            selected.sum().backward()

        optimizer.step()
        scheduler.step()

        cand_best_all = candidate_minima_batch(best_loss)
        for s_local in active_local.tolist():
            cand_best = cand_best_all[s_local]
            old_monitor = candidate_monitor_best[s_local]
            cand_threshold = torch.maximum(
                torch.full_like(old_monitor, float(early_stop_abs_delta)),
                torch.abs(old_monitor) * float(early_stop_rel_delta),
            )
            cand_improved = (~torch.isfinite(old_monitor)) | (
                cand_best < old_monitor - cand_threshold
            )
            if bool(torch.any(cand_improved)):
                candidate_monitor_best[s_local] = torch.where(
                    cand_improved, cand_best, old_monitor
                )
                candidate_last_improve[s_local] = torch.where(
                    cand_improved,
                    torch.full_like(candidate_last_improve[s_local], int(ep)),
                    candidate_last_improve[s_local],
                )

            winner_k, score_margin = current_winner(cand_best)
            if winner_k != int(winner_k_last[s_local]):
                winner_k_last[s_local] = int(winner_k)
                winner_since_epoch[s_local] = int(ep)

            trajectory_plateau = (
                ep - last_improve[s_local]
            ) >= int(early_stop_patience)
            trajectory_plateau_fraction = float(
                trajectory_plateau.float().mean().detach().cpu()
            )
            candidate_plateau = (
                ep - candidate_last_improve[s_local]
            ) >= int(early_stop_patience)
            candidate_plateau_fraction = float(
                candidate_plateau.float().mean().detach().cpu()
            )
            winner_plateau_for = int(
                ep - int(candidate_last_improve[s_local, winner_k - 1].detach().cpu())
            )
            winner_stable_for = int(ep - int(winner_since_epoch[s_local]))

            should_log = (
                ep == 1
                or ep % max(1, int(log_every)) == 0
                or ep == int(epochs)
            )
            if should_log:
                current_np = current_detached[s_local].detach().cpu().numpy()
                best_np = best_loss[s_local].detach().cpu().numpy()
                a_np = current_a[s_local].detach().cpu().numpy()
                for tr in range(trajectories_per_sample):
                    histories[s_local].append(
                        {
                            "sample": int(sample_indices[s_local]),
                            "sample_batch_local_index": int(s_local),
                            "candidate_K": int(candidate_k[tr]),
                            "restart": int(restart_id[tr]),
                            "epoch": int(ep),
                            "current_loss": float(current_np[tr]),
                            "best_loss": float(best_np[tr]),
                            "best_epoch": int(best_epoch[s_local, tr].detach().cpu()),
                            "amplitudes_8slots": vector_text(a_np[tr]),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "trajectory_plateau_fraction": trajectory_plateau_fraction,
                            "candidate_plateau_fraction": candidate_plateau_fraction,
                            "current_selected_K": int(winner_k),
                            "selected_K_stable_epochs": int(winner_stable_for),
                            "selected_K_plateau_epochs": int(winner_plateau_for),
                            "selection_score_margin": float(score_margin),
                            "trajectory_batch_size": int(chunk_size),
                            "total_trajectories": int(trajectories_per_sample),
                            "sample_batch_size": int(n_batch),
                            "joint_trajectories": int(total_joint_trajectories),
                        }
                    )
                finite_current = current_detached[s_local][
                    torch.isfinite(current_detached[s_local])
                ]
                mean_current = float(finite_current.mean().cpu()) if finite_current.numel() else float("nan")
                print(
                    f"[inverse sample {sample_indices[s_local]} | joint batch {n_batch}] "
                    f"ep={ep}/{epochs} mean={mean_current:.4e} "
                    f"best_mean={float(best_loss[s_local].mean().cpu()):.4e} "
                    f"traj_plateau={trajectory_plateau_fraction:.1%} "
                    f"K_plateau={candidate_plateau_fraction:.1%} "
                    f"winner=K{winner_k} stable={winner_stable_for} "
                    f"winner_plateau={winner_plateau_for} margin={score_margin:.2f}",
                    flush=True,
                )

            if ep >= int(early_stop_min_epochs):
                mode = str(early_stop_mode).strip().lower()
                should_stop = False
                reason = ""
                if mode == "trajectory_fraction":
                    should_stop = trajectory_plateau_fraction >= float(early_stop_fraction)
                    reason = "trajectory_plateau_fraction"
                elif mode == "candidate_fraction":
                    should_stop = candidate_plateau_fraction >= float(early_stop_fraction)
                    reason = "candidate_plateau_fraction"
                elif mode == "selection":
                    margin_ok = (
                        float(early_stop_score_margin) <= 0.0
                        or float(score_margin) >= float(early_stop_score_margin)
                    )
                    should_stop = (
                        winner_stable_for >= int(winner_stability_patience)
                        and winner_plateau_for >= int(early_stop_patience)
                        and margin_ok
                    )
                    reason = "selected_K_stable_and_plateaued"
                else:
                    raise ValueError(
                        "--early-stop-mode must be selection, candidate_fraction, "
                        "or trajectory_fraction"
                    )
                if should_stop:
                    active[s_local] = False
                    stopped_epoch[s_local] = int(ep)
                    stopped_reason[s_local] = reason
                    print(
                        f"[inverse sample {sample_indices[s_local]}] early stop at epoch {ep}: "
                        f"{reason}, selected K={winner_k}, score margin={score_margin:.3f}",
                        flush=True,
                    )

    selected_out = torch.empty(
        (n_batch, trajectories_per_sample), dtype=torch.float32, device=device
    )
    power_out = torch.empty_like(selected_out)
    complex_out = torch.empty_like(selected_out)
    best_a8_out = torch.empty(
        (n_batch, trajectories_per_sample, 8), dtype=torch.float32, device=device
    )
    all_sids = torch.arange(n_batch, dtype=torch.long, device=device).repeat_interleave(
        trajectories_per_sample
    )
    all_tids = all_trajectory_ids.repeat(n_batch)
    with torch.no_grad():
        for q0 in range(0, total_joint_trajectories, chunk_size):
            q1 = min(q0 + chunk_size, total_joint_trajectories)
            sids = all_sids[q0:q1]
            tids = all_tids[q0:q1]
            a8_chunk = decode(best_raw[sids, tids], mask_t[tids])
            selected, power, complex_ = loss_vectors(
                model=model,
                tau=tau,
                amplitudes8=a8_chunk,
                y_power=y_power_batch.index_select(0, sids),
                y_real=y_real_batch.index_select(0, sids),
                y_imag=y_imag_batch.index_select(0, sids),
                zeta=zeta,
                observable=observable,
                forward_time_chunk=forward_time_chunk,
            )
            best_a8_out[sids, tids] = a8_chunk
            selected_out[sids, tids] = selected
            power_out[sids, tids] = power
            complex_out[sids, tids] = complex_

    results: list[OptimizationResult] = []
    for s_local in range(n_batch):
        results.append(
            OptimizationResult(
                amplitudes8=best_a8_out[s_local].detach().cpu().numpy(),
                selected_loss=selected_out[s_local].detach().cpu().numpy(),
                power_loss=power_out[s_local].detach().cpu().numpy(),
                complex_loss=complex_out[s_local].detach().cpu().numpy(),
                best_epoch=best_epoch[s_local].detach().cpu().numpy(),
                history=histories[s_local],
                stopped_epoch=int(stopped_epoch[s_local]),
                stopped_reason=str(stopped_reason[s_local]),
                total_trajectories=int(trajectories_per_sample),
                trajectory_batch_size=int(chunk_size),
            )
        )
    return results


# -----------------------------------------------------------------------------
# K 选择、SSFM 回代、结果汇总
# -----------------------------------------------------------------------------


def select_candidate_k(
    *,
    candidate_rows: list[dict[str, Any]],
    selection: str,
    n_observations: int,
    bic_weight: float,
    smallest_within_fraction: float,
) -> dict[str, Any]:
    selection = str(selection).strip().lower()
    if selection == "min_loss":
        return min(candidate_rows, key=lambda r: float(r["best_restart_forward_inverse_loss"]))
    if selection == "smallest_within":
        min_loss = min(float(r["best_restart_forward_inverse_loss"]) for r in candidate_rows)
        threshold = min_loss * (1.0 + float(smallest_within_fraction))
        eligible = [r for r in candidate_rows if float(r["best_restart_forward_inverse_loss"]) <= threshold]
        return min(eligible, key=lambda r: int(r["candidate_K"]))
    if selection != "bic":
        raise ValueError("--selection must be bic, min_loss, or smallest_within")
    for row in candidate_rows:
        loss = max(float(row["best_restart_forward_inverse_loss"]), 1e-15)
        k = int(row["candidate_K"])
        row["selection_score_bic"] = (
            float(n_observations) * math.log(loss)
            + float(bic_weight) * float(k) * math.log(float(n_observations))
        )
    return min(candidate_rows, key=lambda r: float(r["selection_score_bic"]))


def ssfm_terminal_for_a8(
    *,
    amplitudes8: np.ndarray,
    tau_input: np.ndarray,
    ssfm_half_window: float,
    n_t: int,
    n_z: int,
    z_max_ld: float,
    device: torch.device,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = NLSEParams.paper_pam4(
        z_max_ld=float(z_max_ld),
        t_window_t0=float(ssfm_half_window),
        n_t=int(n_t),
        n_z=int(n_z),
    ).with_multi_pulse(
        tuple(float(x) for x in amplitudes8),
        centers_t0=pulse_centers_t0(8),
        level_mode="field",
    )
    _, t_ps, field = run_ssfm(
        params, device=str(device), save_every=int(params.n_z), quiet=not bool(verbose)
    )
    tau_full = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
    return interpolate_terminal_field(tau_full, np.asarray(field[-1]), tau_input)


def make_plots(out_dir: Path, rows: list[dict[str, Any]], tau: np.ndarray, targets: dict[str, np.ndarray], reconstructed: list[np.ndarray]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot warning] {exc}", flush=True)
        return
    plot_dir = ensure_dir(out_dir / "plots")

    x = np.arange(len(rows))
    true_k = np.asarray([int(r["true_K"]) for r in rows])
    pred_k = np.asarray([int(r["predicted_K"]) for r in rows])
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(x, true_k, marker="o", label="True K")
    ax.plot(x, pred_k, marker="x", linestyle="--", label="Predicted K")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Pulse count K")
    ax.set_yticks(range(1, 9))
    ax.set_title("Unknown-K inverse: true and predicted pulse count")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "true_vs_predicted_K.png", dpi=180)
    plt.close(fig)

    inv = np.asarray([float(r["best_restart_forward_inverse_rel_l2"]) for r in rows])
    rec = np.asarray([float(r["ssfm_reconstruction_output_rel_l2"]) for r in rows])
    width = 0.38
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.bar(x - width / 2, inv, width=width, label="Frozen-forward inverse rel-L2")
    ax.bar(x + width / 2, rec, width=width, label="SSFM reconstruction rel-L2")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Two key inverse errors per random sample")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "two_key_errors_by_sample.png", dpi=180)
    plt.close(fig)

    n = len(rows)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, max(3.0, 3.0 * nrows)), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        target_p = targets["power"][i]
        recon_p = reconstructed[i]
        ax.plot(tau, target_p, label="Target SSFM")
        ax.plot(tau, recon_p, linestyle="--", label="Predicted-parameter SSFM")
        ax.set_title(
            f"S{i}: K {rows[i]['true_K']}→{rows[i]['predicted_K']}, "
            f"recon={float(rows[i]['ssfm_reconstruction_output_rel_l2']):.3g}"
        )
        ax.set_xlabel("t / T0")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "terminal_power_target_vs_reconstructed.png", dpi=170)
    plt.close(fig)

    # Compact summary table image.
    table_data = []
    for r in rows:
        table_data.append([
            int(r["sample"]),
            int(r["true_K"]),
            int(r["predicted_K"]),
            f"{float(r['best_restart_forward_inverse_rel_l2']):.4f}",
            f"{float(r['ssfm_reconstruction_output_rel_l2']):.4f}",
            f"{float(r['amplitude_8slot_mae']):.4f}",
            f"{float(r['inverse_optimization_sec']):.1f}",
        ])
    fig_h = max(3.0, 0.42 * len(table_data) + 1.8)
    fig, ax = plt.subplots(figsize=(13.0, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Sample", "True K", "Pred K", "Inverse rel-L2", "SSFM recon rel-L2", "A8 MAE", "Inverse time (s)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    ax.set_title("Universal frozen-forward inverse with unknown K", pad=16)
    fig.tight_layout()
    fig.savefig(plot_dir / "unknownK_inverse_summary_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Detailed per-sample table requested for direct inspection.
    detailed = []
    for r in rows:
        detailed.append([
            int(r["sample"]),
            int(r["true_K"]),
            wrap_amplitudes_for_table(str(r["true_active_amplitudes"])),
            int(r["predicted_K"]),
            wrap_amplitudes_for_table(str(r["pred_active_amplitudes"])),
            f"{float(r['best_restart_forward_inverse_loss']):.3e}",
            f"{float(r['ssfm_reconstruction_output_rel_l2']):.5f}",
            f"{float(r['inverse_optimization_sec']):.2f}",
            f"{float(r['sample_total_sec']):.2f}",
        ])
    detailed_h = max(4.0, 0.82 * len(detailed) + 2.0)
    fig, ax = plt.subplots(figsize=(26.0, detailed_h))
    ax.axis("off")
    table = ax.table(
        cellText=detailed,
        colLabels=[
            "Sample",
            "True K",
            "True active amplitudes",
            "Pred K",
            "Pred active amplitudes",
            "Frozen-forward inverse loss",
            "SSFM reconstruction rel-L2",
            "Inverse time (s)",
            "Total sample time (s)",
        ],
        cellLoc="center",
        colWidths=[0.04, 0.05, 0.25, 0.05, 0.25, 0.11, 0.11, 0.07, 0.07],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.9)
    ax.set_title(
        "Unknown-K continuous inverse: true/predicted K, active amplitudes, and two key losses",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(
        plot_dir / "unknownK_inverse_detailed_results_table.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Freeze universal sparse-8 forward PINN and jointly infer unknown K plus continuous amplitudes."
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-seed", type=int, default=2028)
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--k-sampling", choices=["balanced", "uniform"], default="balanced")
    p.add_argument("--min-active-amplitude", type=float, default=0.05)
    p.add_argument("--rebuild-targets", action="store_true")

    p.add_argument("--terminal-observable", choices=["complex", "power", "power_and_complex"], default="complex")
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--compare-half-window", type=float, default=44.0)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--verbose-ssfm", action="store_true")

    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--min-lr", type=float, default=5e-4)
    p.add_argument("--forward-time-chunk", type=int, default=256)
    p.add_argument(
        "--trajectory-batch-size",
        type=int,
        default=64,
        help=(
            "Maximum number of K/restart trajectories evaluated together. "
            "0 means all 8*restarts at once. Use 64 or 32 on a 6 GB GPU."
        ),
    )
    p.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help=(
            "Number of target samples optimized jointly. For restarts=4, "
            "sample-batch-size=2 gives 2*8*4=64 joint trajectories. "
            "Each sample keeps independent K selection and early stopping."
        ),
    )
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--early-stop-min-epochs", type=int, default=1000)
    p.add_argument("--early-stop-patience", type=int, default=250)
    p.add_argument("--early-stop-fraction", type=float, default=0.90)
    p.add_argument("--early-stop-rel-delta", type=float, default=1e-3)
    p.add_argument("--early-stop-abs-delta", type=float, default=1e-8)
    p.add_argument(
        "--early-stop-mode",
        choices=["selection", "candidate_fraction", "trajectory_fraction"],
        default="selection",
        help="Selection-aware early stopping is recommended for unknown K.",
    )
    p.add_argument("--winner-stability-patience", type=int, default=300)
    p.add_argument(
        "--early-stop-score-margin",
        type=float,
        default=20.0,
        help="Minimum BIC/min-loss score gap between winner and runner-up for early stop.",
    )

    p.add_argument("--selection", choices=["bic", "min_loss", "smallest_within"], default="bic")
    p.add_argument("--bic-weight", type=float, default=1.0)
    p.add_argument("--smallest-within-fraction", type=float, default=0.02)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < float(args.min_active_amplitude) < 1.0:
        raise ValueError("--min-active-amplitude must lie in (0,1).")
    if int(args.sample_batch_size) < 1:
        raise ValueError("--sample-batch-size must be at least 1.")
    if int(args.restarts) < 1:
        raise ValueError("--restarts must be at least 1.")

    set_seed(args.seed)
    device = safe_device(args.device)
    run_dir = Path(args.run_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve() if str(args.checkpoint).strip() else run_dir / "sparse8_forward_pinn.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Universal checkpoint not found: {checkpoint}")
    out_dir = ensure_dir(
        args.out_dir
        or (run_dir / f"inverse_unknownK_continuous_N{int(args.n_samples)}_seed{int(args.sample_seed)}")
    )

    model = load_forward_checkpoint(checkpoint, device)
    if int(getattr(model, "n_pulses", 0)) != 8:
        raise ValueError(
            f"This script requires an 8-slot universal model, got "
            f"n_pulses={getattr(model, 'n_pulses', None)}"
        )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    trajectories_per_sample = 8 * int(args.restarts)
    requested_sample_batch = int(args.sample_batch_size)
    full_joint_trajectories = requested_sample_batch * trajectories_per_sample
    effective_trajectory_chunk = (
        full_joint_trajectories
        if int(args.trajectory_batch_size) <= 0
        else min(full_joint_trajectories, int(args.trajectory_batch_size))
    )

    print("=" * 96, flush=True)
    print("Universal frozen-forward inverse: unknown K + continuous normalized amplitudes", flush=True)
    print(f"device={device}", flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(
        f"n_samples={args.n_samples}, sample_batch_size={requested_sample_batch}, "
        f"K candidates=1..8, restarts/K={args.restarts}",
        flush=True,
    )
    print(
        f"trajectories/sample={trajectories_per_sample}, "
        f"joint trajectories/full sample batch={full_joint_trajectories}",
        flush=True,
    )
    print(
        f"trajectory_batch_size={effective_trajectory_chunk} "
        f"({math.ceil(full_joint_trajectories / effective_trajectory_chunk)} "
        f"trajectory chunks per epoch for a full sample batch)",
        flush=True,
    )
    print(f"active amplitude range=[{args.min_active_amplitude}, 1], inactive slots=0", flush=True)
    print(f"selection={args.selection}, terminal_observable={args.terminal_observable}", flush=True)
    print("=" * 96, flush=True)

    targets = build_or_load_targets(
        out_dir=out_dir,
        n_samples=args.n_samples,
        sample_seed=args.sample_seed,
        k_sampling=args.k_sampling,
        min_active_amplitude=args.min_active_amplitude,
        inverse_input_points=args.inverse_input_points,
        ssfm_half_window=args.ssfm_half_window,
        compare_half_window=args.compare_half_window,
        n_t=args.n_t,
        n_z=args.n_z,
        z_max_ld=args.z_max_ld,
        device=device,
        rebuild=args.rebuild_targets,
        verbose_ssfm=args.verbose_ssfm,
    )

    tau_t = torch.tensor(targets["tau"], dtype=torch.float32, device=device)
    rows: list[dict[str, Any]] = []
    candidate_rows_all: list[dict[str, Any]] = []
    history_all: list[dict[str, Any]] = []
    reconstructed_power: list[np.ndarray] = []
    inverse_batch_wall_times: list[float] = []
    actual_batch_sizes: list[int] = []
    t_all = time.time()

    n_samples = int(args.n_samples)
    for batch_start in range(0, n_samples, requested_sample_batch):
        batch_indices = list(
            range(batch_start, min(batch_start + requested_sample_batch, n_samples))
        )
        actual_batch = len(batch_indices)
        actual_batch_sizes.append(actual_batch)
        joint_count = actual_batch * trajectories_per_sample
        print(
            f"\n=== sample batch {batch_start // requested_sample_batch + 1}: "
            f"samples {[i + 1 for i in batch_indices]} / {n_samples}, "
            f"joint trajectories={joint_count} ===",
            flush=True,
        )

        yp = torch.tensor(targets["power"][batch_indices], dtype=torch.float32, device=device)
        yr = torch.tensor(targets["real"][batch_indices], dtype=torch.float32, device=device)
        yi = torch.tensor(targets["imag"][batch_indices], dtype=torch.float32, device=device)

        inverse_start = time.time()
        opt_list = optimize_sample_batch_all_k(
            sample_indices=batch_indices,
            model=model,
            tau=tau_t,
            y_power_batch=yp,
            y_real_batch=yr,
            y_imag_batch=yi,
            device=device,
            min_active_amplitude=args.min_active_amplitude,
            restarts=args.restarts,
            epochs=args.epochs,
            lr=args.lr,
            min_lr=args.min_lr,
            zeta=args.z_max_ld,
            observable=args.terminal_observable,
            forward_time_chunk=args.forward_time_chunk,
            trajectory_batch_size=args.trajectory_batch_size,
            log_every=args.log_every,
            early_stop_min_epochs=args.early_stop_min_epochs,
            early_stop_patience=args.early_stop_patience,
            early_stop_fraction=args.early_stop_fraction,
            early_stop_rel_delta=args.early_stop_rel_delta,
            early_stop_abs_delta=args.early_stop_abs_delta,
            early_stop_mode=args.early_stop_mode,
            winner_stability_patience=args.winner_stability_patience,
            early_stop_score_margin=args.early_stop_score_margin,
            selection=args.selection,
            bic_weight=args.bic_weight,
            smallest_within_fraction=args.smallest_within_fraction,
            n_observations=int(args.inverse_input_points)
            * (2 if args.terminal_observable == "complex" else 1),
            seed=args.seed,
        )
        inverse_batch_wall_sec = float(time.time() - inverse_start)
        inverse_batch_wall_times.append(inverse_batch_wall_sec)
        inverse_amortized_sec = inverse_batch_wall_sec / float(actual_batch)
        print(
            f"[sample batch] inverse wall time={inverse_batch_wall_sec:.2f}s, "
            f"amortized={inverse_amortized_sec:.2f}s/sample",
            flush=True,
        )

        for local_index, (i, opt) in enumerate(zip(batch_indices, opt_list)):
            history_all.extend(opt.history)
            per_k: list[dict[str, Any]] = []
            for k in range(1, 9):
                start = (k - 1) * int(args.restarts)
                end = start + int(args.restarts)
                local = start + int(np.argmin(opt.selected_loss[start:end]))
                row_k = {
                    "sample": i,
                    "sample_batch_start": int(batch_start),
                    "sample_batch_local_index": int(local_index),
                    "sample_batch_size_actual": int(actual_batch),
                    "candidate_K": k,
                    "best_restart": int(local - start),
                    "best_epoch": int(opt.best_epoch[local]),
                    "best_restart_forward_inverse_loss": float(opt.selected_loss[local]),
                    "best_restart_forward_inverse_rel_l2": float(
                        math.sqrt(max(float(opt.selected_loss[local]), 0.0))
                    ),
                    "forward_inverse_power_rel_l2": float(
                        math.sqrt(max(float(opt.power_loss[local]), 0.0))
                    ),
                    "forward_inverse_complex_rel_l2": float(
                        math.sqrt(max(float(opt.complex_loss[local]), 0.0))
                    ),
                    "pred_amplitudes_8slots": vector_text(opt.amplitudes8[local]),
                    "stopped_epoch": int(opt.stopped_epoch),
                    "stopped_reason": str(opt.stopped_reason),
                    "total_trajectories": int(opt.total_trajectories),
                    "trajectory_batch_size": int(opt.trajectory_batch_size),
                    "joint_trajectories_in_sample_batch": int(joint_count),
                }
                per_k.append(row_k)

            n_obs = int(args.inverse_input_points) * (
                2 if args.terminal_observable == "complex" else 1
            )
            selected = select_candidate_k(
                candidate_rows=per_k,
                selection=args.selection,
                n_observations=n_obs,
                bic_weight=args.bic_weight,
                smallest_within_fraction=args.smallest_within_fraction,
            )
            candidate_rows_all.extend(per_k)

            pred_k = int(selected["candidate_K"])
            pred_a8 = np.asarray(
                [float(x) for x in str(selected["pred_amplitudes_8slots"]).split(";")],
                dtype=np.float32,
            )
            true_k = int(targets["K"][i])
            true_a8 = targets["A8"][i]

            reconstruction_start = time.time()
            recon_p, recon_r, recon_i = ssfm_terminal_for_a8(
                amplitudes8=pred_a8,
                tau_input=targets["tau"],
                ssfm_half_window=args.ssfm_half_window,
                n_t=args.n_t,
                n_z=args.n_z,
                z_max_ld=args.z_max_ld,
                device=device,
                verbose=args.verbose_ssfm,
            )
            ssfm_reconstruction_sec = float(time.time() - reconstruction_start)
            reconstructed_power.append(recon_p)
            recon_complex = relative_l2_complex(
                recon_r, recon_i, targets["real"][i], targets["imag"][i]
            )
            recon_power = relative_l2_power(
                recon_r, recon_i, targets["real"][i], targets["imag"][i]
            )
            if args.terminal_observable == "power":
                recon_selected = recon_power
            elif args.terminal_observable == "complex":
                recon_selected = recon_complex
            else:
                recon_selected = math.sqrt(
                    0.5 * (recon_power ** 2 + recon_complex ** 2)
                )

            min_loss_k = int(
                min(
                    per_k,
                    key=lambda r: float(r["best_restart_forward_inverse_loss"]),
                )["candidate_K"]
            )
            amp_abs = np.abs(pred_a8 - true_a8)
            sample_total_sec = float(inverse_amortized_sec + ssfm_reconstruction_sec)
            result_row = {
                "sample": i,
                "true_K": true_k,
                "predicted_K": pred_k,
                "K_exact": int(pred_k == true_k),
                "K_abs_error": abs(pred_k - true_k),
                "predicted_K_min_raw_loss": min_loss_k,
                "selection_method": args.selection,
                "selection_score": selected.get("selection_score_bic", ""),
                "best_restart": selected["best_restart"],
                "best_epoch": selected["best_epoch"],
                "best_restart_forward_inverse_loss": selected[
                    "best_restart_forward_inverse_loss"
                ],
                "best_restart_forward_inverse_rel_l2": selected[
                    "best_restart_forward_inverse_rel_l2"
                ],
                "forward_inverse_power_rel_l2": selected[
                    "forward_inverse_power_rel_l2"
                ],
                "forward_inverse_complex_rel_l2": selected[
                    "forward_inverse_complex_rel_l2"
                ],
                "ssfm_reconstruction_output_rel_l2": float(recon_selected),
                "ssfm_reconstruction_power_rel_l2": float(recon_power),
                "ssfm_reconstruction_complex_rel_l2": float(recon_complex),
                "true_active_slots_1based": active_slots_text(true_k),
                "pred_active_slots_1based": active_slots_text(pred_k),
                "true_active_amplitudes": active_amplitudes_text(true_a8, true_k),
                "pred_active_amplitudes": active_amplitudes_text(pred_a8, pred_k),
                "true_amplitudes_8slots": vector_text(true_a8),
                "pred_amplitudes_8slots": vector_text(pred_a8),
                "amplitude_8slot_mae": float(np.mean(amp_abs)),
                "amplitude_8slot_rmse": float(
                    np.sqrt(np.mean((pred_a8 - true_a8) ** 2))
                ),
                "amplitude_8slot_max_abs_error": float(np.max(amp_abs)),
                # Amortized wall time is the meaningful per-sample throughput
                # under joint sample batching.
                "inverse_optimization_sec": float(inverse_amortized_sec),
                "inverse_batch_wall_sec": float(inverse_batch_wall_sec),
                "inverse_time_accounting": "batch_wall_time_divided_by_actual_batch_size",
                "ssfm_reconstruction_sec": float(ssfm_reconstruction_sec),
                "sample_total_sec": sample_total_sec,
                "elapsed_total_sec": float(time.time() - t_all),
                "stopped_epoch": int(opt.stopped_epoch),
                "stopped_reason": str(opt.stopped_reason),
                "total_trajectories": int(opt.total_trajectories),
                "trajectory_batch_size": int(opt.trajectory_batch_size),
                "sample_batch_size_requested": int(requested_sample_batch),
                "sample_batch_size_actual": int(actual_batch),
                "joint_trajectories_in_sample_batch": int(joint_count),
            }
            rows.append(result_row)
            print(
                f"[sample {i}] true_K={true_k} "
                f"true_A={result_row['true_active_amplitudes']} | "
                f"pred_K={pred_k} pred_A={result_row['pred_active_amplitudes']} | "
                f"inverse_loss={float(result_row['best_restart_forward_inverse_loss']):.4e} | "
                f"SSFM_recon_relL2={float(recon_selected):.4e} | "
                f"inverse_time_amortized={inverse_amortized_sec:.2f}s "
                f"total_time_amortized={sample_total_sec:.2f}s",
                flush=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(out_dir / "per_sample_unknownK_two_losses.csv", rows)
    readable_rows = [
        {
            "sample": int(r["sample"]),
            "actual_M": int(r["true_K"]),
            "actual_active_slots_1based": r["true_active_slots_1based"],
            "actual_normalized_amplitudes": r["true_active_amplitudes"],
            "predicted_M": int(r["predicted_K"]),
            "predicted_active_slots_1based": r["pred_active_slots_1based"],
            "predicted_normalized_amplitudes": r["pred_active_amplitudes"],
            "M_correct": int(r["K_exact"]),
            "frozen_forward_inverse_loss_relative_MSE": float(
                r["best_restart_forward_inverse_loss"]
            ),
            "frozen_forward_inverse_rel_L2": float(
                r["best_restart_forward_inverse_rel_l2"]
            ),
            "SSFM_reconstruction_output_rel_L2": float(
                r["ssfm_reconstruction_output_rel_l2"]
            ),
            "inverse_optimization_sec": float(r["inverse_optimization_sec"]),
            "inverse_batch_wall_sec": float(r["inverse_batch_wall_sec"]),
            "ssfm_reconstruction_sec": float(r["ssfm_reconstruction_sec"]),
            "sample_total_sec": float(r["sample_total_sec"]),
            "stopped_epoch": int(r["stopped_epoch"]),
            "stopped_reason": str(r["stopped_reason"]),
            "total_trajectories": int(r["total_trajectories"]),
            "trajectory_batch_size": int(r["trajectory_batch_size"]),
            "sample_batch_size_requested": int(r["sample_batch_size_requested"]),
            "sample_batch_size_actual": int(r["sample_batch_size_actual"]),
            "joint_trajectories_in_sample_batch": int(
                r["joint_trajectories_in_sample_batch"]
            ),
        }
        for r in rows
    ]
    write_csv(out_dir / "per_sample_unknownK_detailed_readable.csv", readable_rows)
    write_csv(out_dir / "candidate_K_scores.csv", candidate_rows_all)
    write_csv(out_dir / "optimization_history.csv", history_all)

    k_acc = float(np.mean([int(r["K_exact"]) for r in rows]))
    summary = {
        "method": (
            "freeze universal sparse8 forward PINN; enumerate K=1..8; "
            "continuous amplitude optimization; model-order selection; "
            "joint target-sample batching"
        ),
        "checkpoint": str(checkpoint),
        "n_samples": len(rows),
        "K_exact_accuracy": k_acc,
        "K_mean_absolute_error": float(
            np.mean([float(r["K_abs_error"]) for r in rows])
        ),
        "mean_best_restart_forward_inverse_loss": float(
            np.mean([float(r["best_restart_forward_inverse_loss"]) for r in rows])
        ),
        "mean_best_restart_forward_inverse_rel_l2": float(
            np.mean([float(r["best_restart_forward_inverse_rel_l2"]) for r in rows])
        ),
        "mean_ssfm_reconstruction_output_rel_l2": float(
            np.mean([float(r["ssfm_reconstruction_output_rel_l2"]) for r in rows])
        ),
        "mean_ssfm_reconstruction_power_rel_l2": float(
            np.mean([float(r["ssfm_reconstruction_power_rel_l2"]) for r in rows])
        ),
        "mean_ssfm_reconstruction_complex_rel_l2": float(
            np.mean([float(r["ssfm_reconstruction_complex_rel_l2"]) for r in rows])
        ),
        "mean_amplitude_8slot_mae": float(
            np.mean([float(r["amplitude_8slot_mae"]) for r in rows])
        ),
        "mean_inverse_optimization_sec_per_sample": float(
            np.mean([float(r["inverse_optimization_sec"]) for r in rows])
        ),
        "mean_ssfm_reconstruction_sec_per_sample": float(
            np.mean([float(r["ssfm_reconstruction_sec"]) for r in rows])
        ),
        "mean_total_sec_per_sample": float(
            np.mean([float(r["sample_total_sec"]) for r in rows])
        ),
        "total_inverse_batch_wall_sec": float(np.sum(inverse_batch_wall_times)),
        "total_trajectories_per_sample": int(trajectories_per_sample),
        "sample_batch_size_requested": int(requested_sample_batch),
        "mean_actual_sample_batch_size": float(np.mean(actual_batch_sizes)),
        "joint_trajectories_per_full_sample_batch": int(full_joint_trajectories),
        "trajectory_batch_size": int(effective_trajectory_chunk),
        "time_accounting": (
            "inverse_optimization_sec is actual joint-batch wall time divided "
            "by the number of samples in that batch"
        ),
        "selection": args.selection,
        "bic_weight": float(args.bic_weight),
        "min_active_amplitude": float(args.min_active_amplitude),
        "terminal_observable": args.terminal_observable,
        "level_quantity": "normalized_field_amplitude",
        "note": (
            "0 denotes an inactive slot. Active amplitudes are constrained above "
            "the detection floor. Each target sample keeps independent K selection "
            "and early stopping even when samples are optimized jointly."
        ),
        "elapsed_sec": float(time.time() - t_all),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "run_config.json", vars(args))
    make_plots(out_dir, rows, targets["tau"], targets, reconstructed_power)

    print("\n========== Unknown-K inverse finished ==========", flush=True)
    print(f"K exact accuracy = {k_acc:.1%}", flush=True)
    print(
        f"mean frozen-forward inverse rel-L2 = "
        f"{summary['mean_best_restart_forward_inverse_rel_l2']:.6g}",
        flush=True,
    )
    print(
        f"mean SSFM reconstruction rel-L2 = "
        f"{summary['mean_ssfm_reconstruction_output_rel_l2']:.6g}",
        flush=True,
    )
    print(
        f"mean inverse wall time per sample (amortized) = "
        f"{summary['mean_inverse_optimization_sec_per_sample']:.3f}s",
        flush=True,
    )
    print(f"results -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
