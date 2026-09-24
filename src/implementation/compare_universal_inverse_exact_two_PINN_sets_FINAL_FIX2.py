# -*- coding: utf-8 -*-
"""
compare_universal_inverse_exact_two_PINN_sets_FINAL.py

Purpose
-------
Compare the universal DDNN and the two trained universal inverse CNNs on EXACTLY
the same two unknown-K continuous inverse test sets already used by the universal PINN:

1) N=100, Amin=0.05, R=4, seed=2030
2) N=100, Amin=0.20, R=4, seed=2030

The PINN is NOT rerun. Its existing per-sample results are read directly.

Important fairness details
--------------------------
- Exact true K and exact true sparse-8 amplitudes are read from the original PINN CSV.
- Exact saved PINN target terminal complex fields are read from the original target folders.
- DDNN uses the same unknown-K task:
    candidate K=1,...,8
    R=4 restarts for every candidate K
    3000 epoch cap
    lr=0.03 -> 0.0005 cosine schedule
    active amplitudes constrained to [Amin,1]
    512 terminal complex-field input points
    BIC model-order selection, bic_weight=1.0
- The DDNN optimizer settings are read from each PINN experiment's run_config.json,
  and the script checks the expected formal settings before running.
- CNNs are direct inverse networks and therefore do not use restart.
- For CNN unknown-K decoding, the known experimental detection floor Amin is used:
    estimated amplitude < Amin/2 -> inactive slot (0)
    otherwise retain the continuous estimated amplitude
  This avoids using the old PAM4 threshold 0.125 for the Amin=0.05 experiment.
- All recovered amplitude vectors are re-propagated by SSFM and evaluated against
  the exact saved PINN terminal target on the same tau_input grid.

Stages
------
--stage check
    Fast preflight only. Verifies all paths/checkpoints/configs/array shapes and
    reconstructs the original SSFM targets to check consistency. Does NOT run inverse.

--stage cnn
    Runs baseline CNN and scaled CNN on both exact test sets.

--stage ddnn
    Runs frozen-forward universal DDNN unknown-K inversion on both exact test sets.
    Results are cached per 2-sample batch, so rerunning resumes completed batches.

--stage all
    Runs check, CNN and DDNN.

Required project files in .
----------------------------------------------------
run_universal_full81_CNN_DDNN.py
run_universal_baseline_cnn_inverse.py OR run_universal_baseline_cnn_inverse_v2.py
run_universal_large_cnn_forward_inverse.py

Only FINAL checkpoints are used:
PINN config/physics metadata:
    sparse8_forward_pinn.pt
DDNN:
    universal_ddnn81.pt
CNN baseline inverse:
    baseline_universal_inverse_cnn.pt
CNN scaled inverse:
    large_universal_inverse_cnn.pt
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


SCRIPT_VERSION = "EXACT_TWO_PINN_UNKNOWNK_SETS_FINAL_FIX2_SELFCONTAINED_20260717"

EXP_REL_DIRS = (
    "inverse_unknownK_N100_R4_B2_v4_seed2030",
    "inverse_unknownK_N100_R4_B2_Amin0p20_v4_seed2030",
)

EXPECTED = {
    "inverse_unknownK_N100_R4_B2_v4_seed2030": {
        "n_samples": 100,
        "amin": 0.05,
        "sample_seed": 2030,
        "restarts": 4,
    },
    "inverse_unknownK_N100_R4_B2_Amin0p20_v4_seed2030": {
        "n_samples": 100,
        "amin": 0.20,
        "sample_seed": 2030,
        "restarts": 4,
    },
}

FINAL_REL_PATHS = {
    "pinn": "sparse8_forward_pinn.pt",
    "ddnn": (
        "universal_CNN_DDNN_full81/"
        "universal_DDNN81_same_as_PINN/final/universal_ddnn81.pt"
    ),
    "baseline_cnn": (
        "universal_CNN_DDNN_full81/"
        "universal_CNN_baseline_inverse/final/baseline_universal_inverse_cnn.pt"
    ),
    "scaled_cnn": (
        "universal_CNN_DDNN_full81/"
        "universal_CNN_large_scaled/inverse/final/large_universal_inverse_cnn.pt"
    ),
}

PULSE_CENTERS = np.asarray(
    [-28.0, -20.0, -12.0, -4.0, 4.0, 12.0, 20.0, 28.0],
    dtype=np.float64,
)


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError("Expected JSON object: %s" % path)
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_vector(text: Any) -> np.ndarray:
    s = str(text).strip().strip("[]()")
    for sep in [",", ";"]:
        s = s.replace(sep, " ")
    values = [float(x) for x in s.split()]
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != 8:
        raise ValueError("Expected 8 amplitudes, got %d from %r" % (arr.size, text))
    return arr


def vector_text(x: np.ndarray) -> str:
    return ";".join("%.9g" % float(v) for v in np.asarray(x).reshape(-1))


def rel_field_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(ref, dtype=np.float64)
    axes = tuple(range(1, p.ndim))
    return np.sqrt(
        np.sum((p - y) ** 2, axis=axes)
        / (np.sum(y ** 2, axis=axes) + 1e-300)
    )


def rel_power_l2(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    y = np.asarray(ref, dtype=np.float64)
    pp = np.sum(p ** 2, axis=1)
    yy = np.sum(y ** 2, axis=1)
    axes = tuple(range(1, pp.ndim))
    return np.sqrt(
        np.sum((pp - yy) ** 2, axis=axes)
        / (np.sum(yy ** 2, axis=axes) + 1e-300)
    )


def import_baseline_module():
    for name in (
        "run_universal_baseline_cnn_inverse_v2",
        "run_universal_baseline_cnn_inverse",
    ):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise RuntimeError(
        "Cannot import run_universal_baseline_cnn_inverse.py or _v2.py."
    )


def find_targets_dir(exp_dir: Path) -> Path:
    candidates = []
    for tau_path in exp_dir.rglob("tau_input.npy"):
        d = tau_path.parent
        if (
            d.name.startswith("targets_N")
            and (d / "Y_terminal_real.npy").is_file()
            and (d / "Y_terminal_imag.npy").is_file()
            and (d / "Y_terminal_power.npy").is_file()
        ):
            candidates.append(d.resolve())

    if not candidates:
        raise FileNotFoundError("No complete targets_N* folder under %s" % exp_dir)

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_exact_experiment(exp_dir: Path) -> Dict[str, Any]:
    cfg = read_json(exp_dir / "run_config.json")
    summary = read_json(exp_dir / "summary.json")
    rows = read_csv(exp_dir / "per_sample_unknownK_two_losses.csv")
    targets_dir = find_targets_dir(exp_dir)

    tau = np.asarray(np.load(targets_dir / "tau_input.npy"), dtype=np.float32)
    real = np.asarray(np.load(targets_dir / "Y_terminal_real.npy"), dtype=np.float32)
    imag = np.asarray(np.load(targets_dir / "Y_terminal_imag.npy"), dtype=np.float32)
    power = np.asarray(np.load(targets_dir / "Y_terminal_power.npy"), dtype=np.float32)

    if real.shape != imag.shape or real.shape != power.shape:
        raise RuntimeError(
            "Target shape mismatch in %s: real=%s imag=%s power=%s"
            % (targets_dir, real.shape, imag.shape, power.shape)
        )
    if real.shape[1] != len(tau):
        raise RuntimeError("tau_input length does not match saved target arrays.")

    rows = sorted(rows, key=lambda r: int(float(r["sample"])))
    sample_ids = np.asarray(
        [int(float(r["sample"])) for r in rows],
        dtype=np.int64,
    )
    if not np.array_equal(sample_ids, np.arange(len(rows), dtype=np.int64)):
        raise RuntimeError("Sample IDs are not exactly 0..N-1 in %s" % exp_dir)

    true_k = np.asarray(
        [int(float(r["true_K"])) for r in rows],
        dtype=np.int64,
    )
    true_a = np.stack(
        [parse_vector(r["true_amplitudes_8slots"]) for r in rows],
        axis=0,
    ).astype(np.float32)

    return {
        "exp_dir": exp_dir,
        "cfg": cfg,
        "summary": summary,
        "rows": rows,
        "targets_dir": targets_dir,
        "tau_input": tau,
        "target_field": np.stack([real, imag], axis=1).astype(np.float32),
        "target_power": power,
        "true_k": true_k,
        "true_a": true_a,
    }


def validate_formal_config(exp: Mapping[str, Any]) -> None:
    exp_dir = Path(exp["exp_dir"])
    name = exp_dir.name
    expected = EXPECTED[name]
    cfg = exp["cfg"]

    checks = {
        "n_samples": (int(cfg["n_samples"]), int(expected["n_samples"])),
        "sample_seed": (int(cfg["sample_seed"]), int(expected["sample_seed"])),
        "restarts": (int(cfg["restarts"]), int(expected["restarts"])),
        "min_active_amplitude": (
            float(cfg["min_active_amplitude"]),
            float(expected["amin"]),
        ),
        "epochs": (int(cfg["epochs"]), 3000),
        "inverse_input_points": (int(cfg["inverse_input_points"]), 512),
        "selection": (str(cfg["selection"]), "bic"),
        "bic_weight": (float(cfg["bic_weight"]), 1.0),
        "terminal_observable": (str(cfg["terminal_observable"]), "complex"),
    }

    bad = {}
    for key, pair in checks.items():
        actual, target = pair
        if isinstance(target, float):
            ok = abs(float(actual) - float(target)) <= 1e-12
        else:
            ok = actual == target
        if not ok:
            bad[key] = {"actual": actual, "expected": target}

    if bad:
        raise RuntimeError(
            "Formal PINN experiment config mismatch in %s:\n%s"
            % (exp_dir, json.dumps(bad, ensure_ascii=False, indent=2))
        )


def infer_active_layout(
    seen_k: np.ndarray,
    seen_a: np.ndarray,
    unseen_k: np.ndarray,
    unseen_a: np.ndarray,
) -> Dict[int, List[int]]:
    all_k = np.concatenate([seen_k, unseen_k], axis=0)
    all_a = np.concatenate([seen_a, unseen_a], axis=0)

    result: Dict[int, List[int]] = {}
    for K in range(1, 9):
        rows = all_a[all_k == K]
        if len(rows) == 0:
            raise RuntimeError("No configurations found for K=%d" % K)
        slots = np.flatnonzero(np.any(np.abs(rows) > 1e-12, axis=0)).tolist()
        if len(slots) != K:
            raise RuntimeError(
                "Expected exactly %d active slots for K=%d, got %s"
                % (K, K, slots)
            )
        result[K] = [int(x) for x in slots]
    return result



# =============================================================================
# Self-contained inverse CNN definitions/loaders
# =============================================================================

class BaselineResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.norm1 = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv1d(
            int(channels),
            int(channels),
            int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )
        self.norm2 = nn.GroupNorm(1, int(channels))
        self.conv2 = nn.Conv1d(
            int(channels),
            int(channels),
            int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class BaselineInverseCNNLocal(nn.Module):
    """Exact architecture used by baseline_universal_inverse_cnn.pt."""

    def __init__(
        self,
        hidden: int = 64,
        kernel_size: int = 11,
        dilations: Sequence[int] = (1, 4, 16, 64),
    ) -> None:
        super().__init__()
        self.input_layer = nn.Conv1d(
            2,
            int(hidden),
            int(kernel_size),
            padding=int(kernel_size) // 2,
        )
        self.blocks = nn.ModuleList(
            [
                BaselineResidualBlock(
                    int(hidden),
                    int(kernel_size),
                    int(d),
                )
                for d in dilations
            ]
        )
        self.output_norm = nn.GroupNorm(1, int(hidden))
        self.output_layer = nn.Conv1d(
            int(hidden),
            2,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.input_layer(waveform)
        for block in self.blocks:
            features = block(features)
        delta = self.output_layer(
            F.gelu(self.output_norm(features))
        )
        return waveform + delta


class LargeInverseResidualBlockLocal(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        padding = (int(kernel_size) // 2) * int(dilation)
        self.norm1 = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv1d(
            int(channels),
            int(channels),
            int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )
        self.norm2 = nn.GroupNorm(1, int(channels))
        self.conv2 = nn.Conv1d(
            int(channels),
            int(channels),
            int(kernel_size),
            padding=padding,
            dilation=int(dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class LargeInverseCNNLocal(nn.Module):
    """Exact architecture used by large_universal_inverse_cnn.pt."""

    def __init__(
        self,
        hidden: int = 80,
        kernel_size: int = 11,
        dilations: Sequence[int] = (1, 4, 16, 64, 128),
    ) -> None:
        super().__init__()
        if len(tuple(dilations)) != 5:
            raise ValueError(
                "Scaled inverse CNN checkpoint must use exactly five blocks."
            )

        self.input_layer = nn.Conv1d(
            2,
            int(hidden),
            int(kernel_size),
            padding=int(kernel_size) // 2,
        )
        self.blocks = nn.ModuleList(
            [
                LargeInverseResidualBlockLocal(
                    int(hidden),
                    int(kernel_size),
                    int(d),
                )
                for d in dilations
            ]
        )
        self.output_norm = nn.GroupNorm(1, int(hidden))
        self.output_layer = nn.Conv1d(
            int(hidden),
            2,
            kernel_size=1,
        )
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.input_layer(waveform)
        for block in self.blocks:
            features = block(features)
        delta = self.output_layer(
            F.gelu(self.output_norm(features))
        )
        return waveform + delta


def load_baseline_inverse_checkpoint_local(
    path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    cfg = dict(payload["model_config"])

    model = BaselineInverseCNNLocal(
        hidden=int(cfg["hidden"]),
        kernel_size=int(cfg["kernel_size"]),
        dilations=tuple(int(x) for x in cfg["dilations"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    if "field_scale" not in payload:
        raise RuntimeError(
            "Baseline inverse CNN checkpoint has no field_scale: %s" % path
        )
    return model, payload


def load_scaled_inverse_checkpoint_local(
    path: Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)

    task = str(payload.get("task", "inverse"))
    if task != "inverse":
        raise RuntimeError(
            "Scaled CNN checkpoint task is %r, expected 'inverse'." % task
        )

    cfg = dict(payload["model_config"])
    model = LargeInverseCNNLocal(
        hidden=int(cfg["hidden"]),
        kernel_size=int(cfg["kernel_size"]),
        dilations=tuple(int(x) for x in cfg["dilations"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    if "field_scale" not in payload:
        raise RuntimeError(
            "Scaled inverse CNN checkpoint has no field_scale: %s" % path
        )
    return model, payload


# =============================================================================
# Model terminal prediction for DDNN
# =============================================================================

def predict_terminal_torch(
    model: torch.nn.Module,
    amplitudes: torch.Tensor,
    tau: torch.Tensor,
    z_final: float,
    time_chunk: int,
) -> torch.Tensor:
    """
    amplitudes: [B,8]
    tau: [T]
    returns [B,2,T]
    """
    B = int(amplitudes.shape[0])
    outputs = []

    for start in range(0, int(tau.numel()), int(time_chunk)):
        end = min(int(tau.numel()), start + int(time_chunk))
        t_chunk = tau[start:end]
        T = int(t_chunk.numel())

        z = torch.full(
            (B * T, 1),
            float(z_final),
            dtype=torch.float32,
            device=amplitudes.device,
        )
        t = (
            t_chunk[None, :]
            .expand(B, T)
            .reshape(-1, 1)
        )
        a = (
            amplitudes[:, None, :]
            .expand(B, T, 8)
            .reshape(-1, 8)
        )

        u, v = model(z, t, a)
        chunk = (
            torch.cat([u, v], dim=1)
            .reshape(B, T, 2)
            .permute(0, 2, 1)
            .contiguous()
        )
        outputs.append(chunk)

    return torch.cat(outputs, dim=2)


def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def build_trajectory_metadata(
    batch_size: int,
    restarts: int,
    active_layout: Mapping[int, Sequence[int]],
    amin: float,
    optimizer_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      sample_id [T]
      candidate_k [T]
      mask [T,8]
      raw_init [T,8]

    Total trajectories = batch_size * 8 * restarts.
    """
    rng = np.random.default_rng(int(optimizer_seed))

    sample_ids = []
    candidate_ks = []
    masks = []
    raw_rows = []

    for sample in range(int(batch_size)):
        for K in range(1, 9):
            slots = list(active_layout[K])
            mask = np.zeros(8, dtype=np.float32)
            mask[slots] = 1.0

            for _ in range(int(restarts)):
                amp = rng.uniform(
                    max(float(amin), 0.05),
                    0.95,
                    size=K,
                )
                normalized = (
                    amp - float(amin)
                ) / max(1.0 - float(amin), 1e-12)
                raw_active = inverse_sigmoid(normalized)

                raw = np.zeros(8, dtype=np.float32)
                raw[slots] = raw_active.astype(np.float32)

                sample_ids.append(sample)
                candidate_ks.append(K)
                masks.append(mask.copy())
                raw_rows.append(raw)

    return (
        np.asarray(sample_ids, dtype=np.int64),
        np.asarray(candidate_ks, dtype=np.int64),
        np.stack(masks).astype(np.float32),
        np.stack(raw_rows).astype(np.float32),
    )


def amplitude_from_raw(
    raw: torch.Tensor,
    mask: torch.Tensor,
    amin: float,
) -> torch.Tensor:
    active = float(amin) + (1.0 - float(amin)) * torch.sigmoid(raw)
    return active * mask


def bic_scores_from_candidate_losses(
    candidate_loss: np.ndarray,
    n_observations: int,
    bic_weight: float,
) -> np.ndarray:
    """
    candidate_loss shape [B,8], relative squared terminal-field loss.
    """
    K = np.arange(1, 9, dtype=np.float64)[None, :]
    return (
        float(n_observations)
        * np.log(np.maximum(candidate_loss, 1e-15))
        + float(bic_weight)
        * K
        * math.log(float(n_observations))
    )


def optimize_ddnn_batch(
    *,
    model: torch.nn.Module,
    target: np.ndarray,
    tau: np.ndarray,
    active_layout: Mapping[int, Sequence[int]],
    z_final: float,
    cfg: Mapping[str, Any],
    device: torch.device,
    optimizer_seed: int,
) -> Dict[str, np.ndarray]:
    B = int(target.shape[0])
    R = int(cfg["restarts"])
    amin = float(cfg["min_active_amplitude"])

    sample_ids_np, candidate_k_np, mask_np, raw_init_np = (
        build_trajectory_metadata(
            B,
            R,
            active_layout,
            amin,
            optimizer_seed,
        )
    )

    trajectories = len(sample_ids_np)
    expected_trajectories = B * 8 * R
    if trajectories != expected_trajectories:
        raise RuntimeError("Trajectory count mismatch.")

    raw = torch.tensor(
        raw_init_np,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    mask = torch.from_numpy(mask_np).to(device)
    sample_ids = torch.from_numpy(sample_ids_np).to(device)
    target_t = torch.from_numpy(
        np.asarray(target, dtype=np.float32)
    ).to(device)
    tau_t = torch.from_numpy(
        np.asarray(tau, dtype=np.float32)
    ).to(device)

    optimizer = torch.optim.Adam(
        [raw],
        lr=float(cfg["lr"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(cfg["epochs"])),
        eta_min=float(cfg["min_lr"]),
    )

    best_loss = torch.full(
        (trajectories,),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    best_amp = torch.zeros(
        (trajectories, 8),
        dtype=torch.float32,
        device=device,
    )
    best_epoch = torch.zeros(
        trajectories,
        dtype=torch.long,
        device=device,
    )
    stale = torch.zeros(
        trajectories,
        dtype=torch.long,
        device=device,
    )

    # For selection-mode early stopping.
    last_winner = np.full(B, -1, dtype=np.int64)
    winner_stable = np.zeros(B, dtype=np.int64)

    n_obs = 2 * int(len(tau))
    stopped_epoch = int(cfg["epochs"])

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)

        amplitudes = amplitude_from_raw(
            raw,
            mask,
            amin,
        )

        pred = predict_terminal_torch(
            model,
            amplitudes,
            tau_t,
            float(z_final),
            int(cfg["forward_time_chunk"]),
        )

        target_per_traj = target_t.index_select(
            0,
            sample_ids,
        )

        num = torch.sum(
            (pred.float() - target_per_traj.float()) ** 2,
            dim=(1, 2),
        )
        den = torch.sum(
            target_per_traj.float() ** 2,
            dim=(1, 2),
        ) + 1e-12
        loss_vec = num / den
        loss = torch.mean(loss_vec)

        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DDNN inverse loss.")

        with torch.no_grad():
            threshold = torch.maximum(
                torch.full_like(
                    best_loss,
                    float(cfg["early_stop_abs_delta"]),
                ),
                float(cfg["early_stop_rel_delta"])
                * torch.abs(best_loss),
            )
            improved = torch.isinf(best_loss) | (
                loss_vec < best_loss - threshold
            )

            best_loss = torch.where(
                improved,
                loss_vec.detach(),
                best_loss,
            )
            best_amp = torch.where(
                improved[:, None],
                amplitudes.detach(),
                best_amp,
            )
            best_epoch = torch.where(
                improved,
                torch.full_like(
                    best_epoch,
                    int(epoch),
                ),
                best_epoch,
            )
            stale = torch.where(
                improved,
                torch.zeros_like(stale),
                stale + 1,
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [raw],
            max_norm=10.0,
        )
        optimizer.step()
        scheduler.step()

        # Candidate losses from best restart so far.
        if (
            epoch == 1
            or epoch % int(cfg["log_every"]) == 0
            or epoch >= int(cfg["early_stop_min_epochs"])
        ):
            best_loss_np = best_loss.detach().cpu().numpy()
            candidate_loss = np.full(
                (B, 8),
                np.inf,
                dtype=np.float64,
            )

            for b in range(B):
                for K in range(1, 9):
                    idx = np.flatnonzero(
                        (sample_ids_np == b)
                        & (candidate_k_np == K)
                    )
                    candidate_loss[b, K - 1] = float(
                        np.min(
                            best_loss_np[idx]
                        )
                    )

            bic = bic_scores_from_candidate_losses(
                candidate_loss,
                n_observations=n_obs,
                bic_weight=float(cfg["bic_weight"]),
            )
            winner = (
                np.argmin(
                    bic,
                    axis=1,
                )
                + 1
            ).astype(np.int64)

            sorted_bic = np.sort(
                bic,
                axis=1,
            )
            score_margin = (
                sorted_bic[:, 1]
                - sorted_bic[:, 0]
            )

            for b in range(B):
                if (
                    winner[b] == last_winner[b]
                    and score_margin[b]
                    >= float(cfg["early_stop_score_margin"])
                ):
                    winner_stable[b] += 1
                else:
                    winner_stable[b] = 0
                last_winner[b] = winner[b]

            plateau_fraction = float(
                torch.mean(
                    (
                        stale
                        >= int(cfg["early_stop_patience"])
                    ).float()
                ).detach().cpu()
            )

            if (
                epoch == 1
                or epoch % int(cfg["log_every"]) == 0
                or epoch == int(cfg["epochs"])
            ):
                print(
                    "[DDNN inverse] epoch=%4d/%d mean=%.4e best_mean=%.4e "
                    "plateau=%.1f%% winners=%s stable_min=%d lr=%.3e"
                    % (
                        epoch,
                        int(cfg["epochs"]),
                        float(loss.detach().cpu()),
                        float(torch.mean(best_loss).detach().cpu()),
                        100.0 * plateau_fraction,
                        winner.tolist(),
                        int(np.min(winner_stable)),
                        float(optimizer.param_groups[0]["lr"]),
                    ),
                    flush=True,
                )

            if (
                epoch >= int(cfg["early_stop_min_epochs"])
                and str(cfg["early_stop_mode"]) == "selection"
                and plateau_fraction
                >= float(cfg["early_stop_fraction"])
                and np.all(
                    winner_stable
                    >= int(cfg["winner_stability_patience"])
                )
            ):
                stopped_epoch = int(epoch)
                print(
                    "[DDNN inverse] selection-mode early stop at epoch=%d"
                    % stopped_epoch,
                    flush=True,
                )
                break

    # Final per-sample/per-K best restart.
    best_loss_np = best_loss.detach().cpu().numpy()
    best_amp_np = best_amp.detach().cpu().numpy()
    best_epoch_np = best_epoch.detach().cpu().numpy()

    candidate_loss = np.full(
        (B, 8),
        np.inf,
        dtype=np.float64,
    )
    candidate_amp = np.zeros(
        (B, 8, 8),
        dtype=np.float32,
    )
    candidate_restart = np.zeros(
        (B, 8),
        dtype=np.int64,
    )
    candidate_epoch = np.zeros(
        (B, 8),
        dtype=np.int64,
    )

    for b in range(B):
        for K in range(1, 9):
            idx = np.flatnonzero(
                (sample_ids_np == b)
                & (candidate_k_np == K)
            )
            local = int(
                np.argmin(
                    best_loss_np[idx]
                )
            )
            pos = int(idx[local])
            candidate_loss[b, K - 1] = float(
                best_loss_np[pos]
            )
            candidate_amp[b, K - 1] = best_amp_np[pos]
            candidate_restart[b, K - 1] = local
            candidate_epoch[b, K - 1] = int(
                best_epoch_np[pos]
            )

    bic = bic_scores_from_candidate_losses(
        candidate_loss,
        n_observations=n_obs,
        bic_weight=float(cfg["bic_weight"]),
    )
    selected_k = (
        np.argmin(
            bic,
            axis=1,
        )
        + 1
    ).astype(np.int64)

    selected_amp = np.zeros(
        (B, 8),
        dtype=np.float32,
    )
    selected_loss = np.zeros(
        B,
        dtype=np.float64,
    )
    selected_restart = np.zeros(
        B,
        dtype=np.int64,
    )
    selected_epoch = np.zeros(
        B,
        dtype=np.int64,
    )

    for b, K in enumerate(selected_k.tolist()):
        selected_amp[b] = candidate_amp[b, K - 1]
        selected_loss[b] = candidate_loss[b, K - 1]
        selected_restart[b] = candidate_restart[b, K - 1]
        selected_epoch[b] = candidate_epoch[b, K - 1]

    return {
        "pred_a": selected_amp,
        "pred_k": selected_k,
        "selected_loss": selected_loss,
        "selected_restart": selected_restart,
        "selected_epoch": selected_epoch,
        "candidate_loss": candidate_loss,
        "candidate_bic": bic,
        "stopped_epoch": np.asarray(
            [stopped_epoch],
            dtype=np.int64,
        ),
    }


# =============================================================================
# CNN prediction / amplitude decoding
# =============================================================================

def predict_cnn(
    model: torch.nn.Module,
    terminal_full: np.ndarray,
    scale: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(terminal_full), int(batch_size)):
            x = torch.from_numpy(
                np.asarray(
                    terminal_full[
                        start : start + int(batch_size)
                    ],
                    dtype=np.float32,
                )
                / float(scale)
            ).to(device)
            y = model(x).float() * float(scale)
            outputs.append(
                y.detach().cpu().numpy()
            )

    return np.concatenate(
        outputs,
        axis=0,
    ).astype(np.float32)


def decode_all8_amplitudes(
    initial_field: np.ndarray,
    tau_full: np.ndarray,
) -> np.ndarray:
    basis = np.stack(
        [
            np.exp(
                -0.5
                * (
                    np.asarray(tau_full, dtype=np.float64)
                    - center
                )
                ** 2
            )
            for center in PULSE_CENTERS
        ],
        axis=1,
    )  # [T,8]

    pinv = np.linalg.pinv(basis)
    real = np.asarray(
        initial_field[:, 0],
        dtype=np.float64,
    )
    amp = real @ pinv.T
    return np.clip(
        amp,
        0.0,
        1.0,
    ).astype(np.float32)


def sparsify_cnn_amplitudes(
    estimated: np.ndarray,
    amin: float,
) -> Tuple[np.ndarray, np.ndarray]:
    threshold = float(amin) / 2.0
    pred = np.asarray(
        estimated,
        dtype=np.float32,
    ).copy()
    pred[
        pred < threshold
    ] = 0.0

    pred_k = np.sum(
        pred > 0.0,
        axis=1,
    ).astype(np.int64)

    return pred, pred_k


# =============================================================================
# SSFM target generation / validation / back-check
# =============================================================================

def interpolate_field(
    field: np.ndarray,
    source_tau: np.ndarray,
    target_tau: np.ndarray,
) -> np.ndarray:
    if (
        len(source_tau) == len(target_tau)
        and np.allclose(
            source_tau,
            target_tau,
            rtol=0.0,
            atol=1e-6,
        )
    ):
        return np.asarray(
            field,
            dtype=np.float32,
        )

    out = np.empty(
        (
            field.shape[0],
            2,
            len(target_tau),
        ),
        dtype=np.float32,
    )

    for i in range(field.shape[0]):
        out[i, 0] = np.interp(
            target_tau,
            source_tau,
            field[i, 0],
        )
        out[i, 1] = np.interp(
            target_tau,
            source_tau,
            field[i, 1],
        )
    return out


def build_full_endpoint_grid(
    base: Any,
    model_cfg: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> Mapping[str, Any]:
    return base.make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=float(cfg["ssfm_half_window"]),
        n_t=int(cfg["n_t"]),
        n_z=int(cfg["n_z"]),
        n_slices=2,
        compare_t_min=None,
        compare_t_max=None,
    )


def generate_ssfm_terminal_full(
    base: Any,
    amplitudes: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    device: torch.device,
    complex64: bool,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    for start in range(0, len(amplitudes), int(batch_size)):
        maps = base.run_ssfm_batch_selected(
            amplitudes[
                start : start + int(batch_size)
            ],
            grid,
            pde,
            device,
            bool(complex64),
        )
        chunks.append(
            np.asarray(
                maps[:, -1],
                dtype=np.float32,
            )
        )
    return np.concatenate(
        chunks,
        axis=0,
    )


# =============================================================================
# Metrics
# =============================================================================

def normalize_existing_pinn_rows(
    exp: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for source in exp["rows"]:
        true_a = parse_vector(
            source["true_amplitudes_8slots"]
        )
        pred_a = parse_vector(
            source["pred_amplitudes_8slots"]
        )
        err = np.abs(
            pred_a.astype(np.float64)
            - true_a.astype(np.float64)
        )

        rows.append(
            {
                "method": "PINN",
                "sample": int(float(source["sample"])),
                "true_K": int(float(source["true_K"])),
                "predicted_K": int(float(source["predicted_K"])),
                "K_correct": int(
                    int(float(source["true_K"]))
                    == int(float(source["predicted_K"]))
                ),
                "true_amplitudes_8slots": vector_text(true_a),
                "pred_amplitudes_8slots": vector_text(pred_a),
                "amplitude_8slot_mae": float(np.mean(err)),
                "all_8slot_error_le_0p05": int(
                    bool(np.all(err <= 0.05))
                ),
                "ssfm_backcheck_field_rel_l2": float(
                    source[
                        "ssfm_reconstruction_output_rel_l2"
                    ]
                ),
            }
        )
    return rows


def make_result_rows(
    *,
    method: str,
    true_k: np.ndarray,
    true_a: np.ndarray,
    pred_k: np.ndarray,
    pred_a: np.ndarray,
    target_saved: np.ndarray,
    back_saved_grid: np.ndarray,
) -> List[Dict[str, Any]]:
    field_err = rel_field_l2(
        back_saved_grid,
        target_saved,
    )

    rows = []
    for i in range(len(true_a)):
        err = np.abs(
            pred_a[i].astype(np.float64)
            - true_a[i].astype(np.float64)
        )
        rows.append(
            {
                "method": method,
                "sample": int(i),
                "true_K": int(true_k[i]),
                "predicted_K": int(pred_k[i]),
                "K_correct": int(
                    int(pred_k[i])
                    == int(true_k[i])
                ),
                "true_amplitudes_8slots": vector_text(
                    true_a[i]
                ),
                "pred_amplitudes_8slots": vector_text(
                    pred_a[i]
                ),
                "amplitude_8slot_mae": float(
                    np.mean(err)
                ),
                "all_8slot_error_le_0p05": int(
                    bool(
                        np.all(
                            err <= 0.05
                        )
                    )
                ),
                "ssfm_backcheck_field_rel_l2": float(
                    field_err[i]
                ),
            }
        )
    return rows


def summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    experiment: str,
    amin: float,
) -> Dict[str, Any]:
    ssfm = np.asarray(
        [
            float(r["ssfm_backcheck_field_rel_l2"])
            for r in rows
        ],
        dtype=np.float64,
    )

    return {
        "experiment": experiment,
        "Amin": float(amin),
        "method": str(rows[0]["method"]),
        "n_samples": int(len(rows)),
        "K_accuracy": float(
            np.mean(
                [
                    float(r["K_correct"])
                    for r in rows
                ]
            )
        ),
        "amplitude_8slot_mae": float(
            np.mean(
                [
                    float(r["amplitude_8slot_mae"])
                    for r in rows
                ]
            )
        ),
        "vector_success_atol_0p05": float(
            np.mean(
                [
                    float(
                        r[
                            "all_8slot_error_le_0p05"
                        ]
                    )
                    for r in rows
                ]
            )
        ),
        "ssfm_backcheck_field_mean": float(
            np.mean(ssfm)
        ),
        "ssfm_backcheck_field_median": float(
            np.median(ssfm)
        ),
        "ssfm_backcheck_field_p95": float(
            np.quantile(
                ssfm,
                0.95,
            )
        ),
        "ssfm_backcheck_field_max": float(
            np.max(ssfm)
        ),
    }


# =============================================================================
# Main workflow
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-dir",
        required=True,
    )
    p.add_argument(
        "--stage",
        choices=(
            "check",
            "cnn",
            "ddnn",
            "all",
        ),
        default="check",
    )
    p.add_argument(
        "--device",
        default="cuda",
    )
    p.add_argument(
        "--cnn-batch-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--ssfm-batch-size",
        type=int,
        default=64,
    )
    p.add_argument(
        "--ssfm-complex64",
        action="store_true",
    )
    p.add_argument(
        "--optimizer-seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--output-dir",
        default="",
    )
    p.add_argument(
        "--force",
        action="store_true",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(
        args.run_dir
    ).expanduser().resolve()

    if not run_dir.is_dir():
        raise FileNotFoundError(
            "Run directory not found: %s"
            % run_dir
        )

    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(
            0,
            str(project_root),
        )

    import run_universal_full81_CNN_DDNN as base

    device = base.safe_device(
        str(args.device)
    )
    set_seed(
        int(args.optimizer_seed)
    )

    checkpoint_paths = {
        key: (
            run_dir
            / rel
        ).resolve()
        for key, rel in FINAL_REL_PATHS.items()
    }

    for key, path in checkpoint_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                "Missing FINAL %s checkpoint: %s"
                % (
                    key,
                    path,
                )
            )

    # Explicitly reject accidental use of stage PINN/DDNN checkpoints.
    if checkpoint_paths["pinn"].name != "sparse8_forward_pinn.pt":
        raise RuntimeError(
            "Only final sparse8_forward_pinn.pt is allowed."
        )
    if checkpoint_paths["ddnn"].name != "universal_ddnn81.pt":
        raise RuntimeError(
            "Only final universal_ddnn81.pt is allowed."
        )

    model_cfg, pde = (
        base.load_pinn_checkpoint_metadata(
            checkpoint_paths["pinn"]
        )
    )

    seen_k, seen_a = base.load_sparse8_csv(
        run_dir
        / "dataset"
        / "seen_sparse8_combinations.csv"
    )
    unseen_k, unseen_a = base.load_sparse8_csv(
        run_dir
        / "dataset"
        / "unseen_sparse8_combinations.csv"
    )
    active_layout = infer_active_layout(
        seen_k,
        seen_a,
        unseen_k,
        unseen_a,
    )

    experiments = []
    for rel in EXP_REL_DIRS:
        exp = load_exact_experiment(
            run_dir / rel
        )
        validate_formal_config(exp)
        experiments.append(exp)

    output_root = (
        Path(
            args.output_dir
        ).expanduser().resolve()
        if args.output_dir
        else (
            run_dir
            / "exact_two_PINN_unknownK_sets_CNN_DDNN_FINAL"
        )
    )
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 120, flush=True)
    print("SCRIPT VERSION :", SCRIPT_VERSION, flush=True)
    print("RUN DIR        :", run_dir, flush=True)
    print("PINN FINAL     :", checkpoint_paths["pinn"], flush=True)
    print("DDNN FINAL     :", checkpoint_paths["ddnn"], flush=True)
    print("BASE CNN FINAL :", checkpoint_paths["baseline_cnn"], flush=True)
    print("LARGE CNN FINAL:", checkpoint_paths["scaled_cnn"], flush=True)
    print("OUTPUT         :", output_root, flush=True)
    print("=" * 120, flush=True)

    # Load DDNN only if needed.
    ddnn_model = None
    if args.stage in ("ddnn", "all"):
        ddnn_model, _ = base.load_ddnn_checkpoint(
            checkpoint_paths["ddnn"],
            device,
        )

    baseline_model = None
    baseline_scale = None
    scaled_model = None
    scaled_scale = None

    if args.stage in ("cnn", "all"):
        (
            baseline_model,
            baseline_payload,
        ) = load_baseline_inverse_checkpoint_local(
            checkpoint_paths["baseline_cnn"],
            device,
        )
        baseline_scale = float(
            baseline_payload["field_scale"]
        )

        (
            scaled_model,
            scaled_payload,
        ) = load_scaled_inverse_checkpoint_local(
            checkpoint_paths["scaled_cnn"],
            device,
        )
        if str(
            scaled_payload.get(
                "task",
                "",
            )
        ) != "inverse":
            raise RuntimeError(
                "Scaled checkpoint is not inverse."
            )
        scaled_scale = float(
            scaled_payload["field_scale"]
        )

    combined_rows: List[Dict[str, Any]] = []
    combined_summary: List[Dict[str, Any]] = []

    for exp in experiments:
        cfg = exp["cfg"]
        exp_name = Path(
            exp["exp_dir"]
        ).name
        amin = float(
            cfg[
                "min_active_amplitude"
            ]
        )
        tag = (
            "N100_Amin0p05_R4_seed2030"
            if amin < 0.1
            else "N100_Amin0p20_R4_seed2030"
        )
        out_dir = (
            output_root
            / tag
        )
        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        grid_full = build_full_endpoint_grid(
            base,
            model_cfg,
            cfg,
        )

        print(
            "\n[%s] exact PINN source -> %s"
            % (
                tag,
                exp["exp_dir"],
            ),
            flush=True,
        )
        print(
            "[%s] saved target shape=%s tau=%d Amin=%.2f"
            % (
                tag,
                exp["target_field"].shape,
                len(
                    exp[
                        "tau_input"
                    ]
                ),
                amin,
            ),
            flush=True,
        )

        # Rebuild the SAME physical target at full 2048 resolution for CNN input.
        true_terminal_full = (
            generate_ssfm_terminal_full(
                base,
                exp["true_a"],
                grid_full,
                pde,
                device,
                bool(args.ssfm_complex64),
                int(args.ssfm_batch_size),
            )
        )

        regenerated_on_saved_tau = interpolate_field(
            true_terminal_full,
            np.asarray(
                grid_full["tau"],
                dtype=np.float32,
            ),
            exp["tau_input"],
        )
        target_consistency = rel_field_l2(
            regenerated_on_saved_tau,
            exp["target_field"],
        )

        print(
            "[%s] regenerated-vs-saved target: mean=%.6e max=%.6e"
            % (
                tag,
                float(
                    np.mean(
                        target_consistency
                    )
                ),
                float(
                    np.max(
                        target_consistency
                    )
                ),
            ),
            flush=True,
        )

        # Fail fast when the supposedly same target generation is clearly inconsistent.
        if float(
            np.mean(
                target_consistency
            )
        ) > 5e-3:
            raise RuntimeError(
                "Regenerated SSFM targets do not match the original saved PINN targets "
                "closely enough for %s. Mean rel-L2=%.6g. Stop before expensive inverse."
                % (
                    tag,
                    float(
                        np.mean(
                            target_consistency
                        )
                    ),
                )
            )

        # Existing PINN results are always included in summaries.
        pinn_rows = normalize_existing_pinn_rows(
            exp
        )
        save_csv(
            out_dir
            / "PINN_existing_results.csv",
            pinn_rows,
        )
        pinn_summary = summarize_rows(
            pinn_rows,
            tag,
            amin,
        )
        combined_rows.extend(
            [
                {
                    **row,
                    "experiment": tag,
                    "Amin": amin,
                }
                for row in pinn_rows
            ]
        )
        combined_summary.append(
            pinn_summary
        )

        if args.stage == "check":
            continue

        # ---------------------------------------------------------------------
        # CNN inverse
        # ---------------------------------------------------------------------
        if args.stage in ("cnn", "all"):
            for method, model, scale, filename in [
                (
                    "Baseline_CNN",
                    baseline_model,
                    baseline_scale,
                    "baseline_CNN_results.csv",
                ),
                (
                    "Scaled_CNN",
                    scaled_model,
                    scaled_scale,
                    "scaled_CNN_results.csv",
                ),
            ]:
                pred_initial = predict_cnn(
                    model,
                    true_terminal_full,
                    float(scale),
                    device,
                    int(args.cnn_batch_size),
                )
                estimated = decode_all8_amplitudes(
                    pred_initial,
                    np.asarray(
                        grid_full["tau"],
                        dtype=np.float32,
                    ),
                )
                pred_a, pred_k = (
                    sparsify_cnn_amplitudes(
                        estimated,
                        amin,
                    )
                )

                back_full = generate_ssfm_terminal_full(
                    base,
                    pred_a,
                    grid_full,
                    pde,
                    device,
                    bool(
                        args.ssfm_complex64
                    ),
                    int(
                        args.ssfm_batch_size
                    ),
                )
                back_saved_tau = interpolate_field(
                    back_full,
                    np.asarray(
                        grid_full["tau"],
                        dtype=np.float32,
                    ),
                    exp["tau_input"],
                )

                rows = make_result_rows(
                    method=method,
                    true_k=exp["true_k"],
                    true_a=exp["true_a"],
                    pred_k=pred_k,
                    pred_a=pred_a,
                    target_saved=exp["target_field"],
                    back_saved_grid=back_saved_tau,
                )
                save_csv(
                    out_dir / filename,
                    rows,
                )
                summary = summarize_rows(
                    rows,
                    tag,
                    amin,
                )
                combined_rows.extend(
                    [
                        {
                            **row,
                            "experiment": tag,
                            "Amin": amin,
                        }
                        for row in rows
                    ]
                )
                combined_summary.append(
                    summary
                )

                print(
                    "[%s %s] Kacc=%.2f%% A8_MAE=%.6f success=%.2f%% SSFM=%.3f%%"
                    % (
                        tag,
                        method,
                        100.0
                        * summary[
                            "K_accuracy"
                        ],
                        summary[
                            "amplitude_8slot_mae"
                        ],
                        100.0
                        * summary[
                            "vector_success_atol_0p05"
                        ],
                        100.0
                        * summary[
                            "ssfm_backcheck_field_mean"
                        ],
                    ),
                    flush=True,
                )

        # ---------------------------------------------------------------------
        # DDNN unknown-K inverse
        # ---------------------------------------------------------------------
        if args.stage in ("ddnn", "all"):
            ddnn_cfg = {
                "restarts": int(
                    cfg["restarts"]
                ),
                "epochs": int(
                    cfg["epochs"]
                ),
                "lr": float(
                    cfg["lr"]
                ),
                "min_lr": float(
                    cfg["min_lr"]
                ),
                "min_active_amplitude": float(
                    cfg[
                        "min_active_amplitude"
                    ]
                ),
                "forward_time_chunk": int(
                    cfg[
                        "forward_time_chunk"
                    ]
                ),
                "sample_batch_size": int(
                    cfg[
                        "sample_batch_size"
                    ]
                ),
                "trajectory_batch_size": int(
                    cfg[
                        "trajectory_batch_size"
                    ]
                ),
                "log_every": int(
                    cfg["log_every"]
                ),
                "early_stop_min_epochs": int(
                    cfg[
                        "early_stop_min_epochs"
                    ]
                ),
                "early_stop_patience": int(
                    cfg[
                        "early_stop_patience"
                    ]
                ),
                "early_stop_fraction": float(
                    cfg[
                        "early_stop_fraction"
                    ]
                ),
                "early_stop_rel_delta": float(
                    cfg[
                        "early_stop_rel_delta"
                    ]
                ),
                "early_stop_abs_delta": float(
                    cfg[
                        "early_stop_abs_delta"
                    ]
                ),
                "early_stop_mode": str(
                    cfg[
                        "early_stop_mode"
                    ]
                ),
                "winner_stability_patience": int(
                    cfg[
                        "winner_stability_patience"
                    ]
                ),
                "early_stop_score_margin": float(
                    cfg[
                        "early_stop_score_margin"
                    ]
                ),
                "selection": str(
                    cfg["selection"]
                ),
                "bic_weight": float(
                    cfg[
                        "bic_weight"
                    ]
                ),
                "smallest_within_fraction": float(
                    cfg[
                        "smallest_within_fraction"
                    ]
                ),
            }

            # In BIC mode, smallest_within_fraction is an unused alternative-selector
            # parameter; it is preserved in the manifest but selection is strict min-BIC,
            # matching run_config["selection"] == "bic".
            if ddnn_cfg["selection"] != "bic":
                raise RuntimeError(
                    "This final comparison expects selection='bic'."
                )

            batch_size = int(
                ddnn_cfg[
                    "sample_batch_size"
                ]
            )
            if (
                batch_size
                * 8
                * int(
                    ddnn_cfg[
                        "restarts"
                    ]
                )
                > int(
                    ddnn_cfg[
                        "trajectory_batch_size"
                    ]
                )
            ):
                raise RuntimeError(
                    "sample_batch_size * 8 * restarts exceeds trajectory_batch_size."
                )

            ddnn_out = (
                out_dir
                / "DDNN_unknownK_R4"
            )
            ddnn_out.mkdir(
                parents=True,
                exist_ok=True,
            )

            pred_a_all = np.zeros_like(
                exp["true_a"],
                dtype=np.float32,
            )
            pred_k_all = np.zeros(
                len(
                    exp[
                        "true_a"
                    ]
                ),
                dtype=np.int64,
            )

            for start in range(
                0,
                len(
                    exp[
                        "true_a"
                    ]
                ),
                batch_size,
            ):
                end = min(
                    len(
                        exp[
                            "true_a"
                        ]
                    ),
                    start
                    + batch_size,
                )
                cache = (
                    ddnn_out
                    / (
                        "samples_%03d_%03d.npz"
                        % (
                            start,
                            end - 1,
                        )
                    )
                )

                if (
                    cache.is_file()
                    and not args.force
                ):
                    saved = np.load(
                        cache
                    )
                    result = {
                        key: saved[key]
                        for key in saved.files
                    }
                    print(
                        "[%s DDNN] load cache samples %d-%d"
                        % (
                            tag,
                            start,
                            end - 1,
                        ),
                        flush=True,
                    )
                else:
                    print(
                        "\n[%s DDNN] optimize samples %d-%d | trajectories=%d"
                        % (
                            tag,
                            start,
                            end - 1,
                            (
                                end
                                - start
                            )
                            * 8
                            * int(
                                ddnn_cfg[
                                    "restarts"
                                ]
                            ),
                        ),
                        flush=True,
                    )

                    result = optimize_ddnn_batch(
                        model=ddnn_model,
                        target=exp[
                            "target_field"
                        ][
                            start:end
                        ],
                        tau=exp[
                            "tau_input"
                        ],
                        active_layout=active_layout,
                        z_final=float(
                            cfg[
                                "z_max_ld"
                            ]
                        ),
                        cfg=ddnn_cfg,
                        device=device,
                        optimizer_seed=(
                            int(
                                args.optimizer_seed
                            )
                            + int(
                                round(
                                    amin
                                    * 10000
                                )
                            )
                            + start
                        ),
                    )
                    np.savez_compressed(
                        cache,
                        **result,
                    )

                pred_a_all[
                    start:end
                ] = np.asarray(
                    result[
                        "pred_a"
                    ],
                    dtype=np.float32,
                )
                pred_k_all[
                    start:end
                ] = np.asarray(
                    result[
                        "pred_k"
                    ],
                    dtype=np.int64,
                )

            back_full = generate_ssfm_terminal_full(
                base,
                pred_a_all,
                grid_full,
                pde,
                device,
                bool(
                    args.ssfm_complex64
                ),
                int(
                    args.ssfm_batch_size
                ),
            )
            back_saved_tau = interpolate_field(
                back_full,
                np.asarray(
                    grid_full["tau"],
                    dtype=np.float32,
                ),
                exp["tau_input"],
            )

            rows = make_result_rows(
                method="DDNN",
                true_k=exp["true_k"],
                true_a=exp["true_a"],
                pred_k=pred_k_all,
                pred_a=pred_a_all,
                target_saved=exp["target_field"],
                back_saved_grid=back_saved_tau,
            )

            save_csv(
                out_dir
                / "DDNN_unknownK_results.csv",
                rows,
            )
            summary = summarize_rows(
                rows,
                tag,
                amin,
            )
            combined_rows.extend(
                [
                    {
                        **row,
                        "experiment": tag,
                        "Amin": amin,
                    }
                    for row in rows
                ]
            )
            combined_summary.append(
                summary
            )

            write_json(
                ddnn_out
                / "DDNN_inverse_config.json",
                {
                    "source_PINN_run_config": cfg,
                    "DDNN_effective_inverse_config": ddnn_cfg,
                    "note": (
                        "Same R/epochs/lr/min_lr/Amin/512 complex terminal points/"
                        "sample_batch_size/trajectory budget/BIC settings as source PINN run."
                    ),
                },
            )

            print(
                "[%s DDNN] Kacc=%.2f%% A8_MAE=%.6f success=%.2f%% SSFM=%.3f%%"
                % (
                    tag,
                    100.0
                    * summary[
                        "K_accuracy"
                    ],
                    summary[
                        "amplitude_8slot_mae"
                    ],
                    100.0
                    * summary[
                        "vector_success_atol_0p05"
                    ],
                    100.0
                    * summary[
                        "ssfm_backcheck_field_mean"
                    ],
                ),
                flush=True,
            )

        # Save current experiment summary from all accumulated rows for this tag.
        current = [
            row
            for row in combined_summary
            if row["experiment"]
            == tag
        ]
        save_csv(
            out_dir
            / "comparison_summary.csv",
            current,
        )

        write_json(
            out_dir
            / "preflight_and_manifest.json",
            {
                "script_version": SCRIPT_VERSION,
                "experiment": tag,
                "source_PINN_experiment": str(
                    exp["exp_dir"]
                ),
                "source_targets_dir": str(
                    exp[
                        "targets_dir"
                    ]
                ),
                "final_checkpoints": {
                    key: str(
                        value
                    )
                    for key, value in checkpoint_paths.items()
                },
                "source_run_config": cfg,
                "target_consistency_mean_rel_l2": float(
                    np.mean(
                        target_consistency
                    )
                ),
                "target_consistency_max_rel_l2": float(
                    np.max(
                        target_consistency
                    )
                ),
                "CNN_detection_threshold": float(
                    amin
                    / 2.0
                ),
                "active_layout_1based": {
                    str(K): [
                        int(x)
                        + 1
                        for x in active_layout[
                            K
                        ]
                    ]
                    for K in range(
                        1,
                        9,
                    )
                },
            },
        )

    save_csv(
        output_root
        / "combined_comparison_summary.csv",
        combined_summary,
    )
    save_csv(
        output_root
        / "combined_per_sample_results.csv",
        combined_rows,
    )

    if args.stage == "check":
        print(
            "\nPRECHECK PASSED. No inverse optimization was run.",
            flush=True,
        )
    else:
        print(
            "\nDone -> %s"
            % output_root,
            flush=True,
        )


if __name__ == "__main__":
    main()
