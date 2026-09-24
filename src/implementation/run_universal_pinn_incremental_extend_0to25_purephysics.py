# -*- coding: utf-8 -*-
"""
run_universal_pinn_incremental_extend_0to25_purephysics.py

Incrementally extend the existing universal sparse-8 PINN from 0-20 km (0-4 LD)
to 0-25 km (0-5 LD) by continuing to train THE SAME network.

This is different from the previous segmented child-network experiments:
- no second 20-25 km network is created;
- the final 0-20 km universal PINN checkpoint is used as initialization;
- all original weights remain trainable;
- most new PDE points are placed in z in [4,5], while a smaller rehearsal set
  remains in z in [0,4] to preserve the already learned region;
- analytic z=0 initial-condition constraints, time-edge constraints, NLSE
  residuals, and power conservation are used;
- NO SSFM propagation labels are used during training.

Important coordinate choice
---------------------------
The parent model_config is kept unchanged, including z_max_ld=4.0. Therefore the
network uses exactly the same coordinate normalization as the original 0-20 km
model. During incremental training it is simply evaluated and optimized at
z in (4,5] as well. This avoids changing the meaning of every old coordinate.

Default schedule
----------------
Stage 1: 800 Adam steps, 85% PDE points in [4,5]
Stage 2: 800 Adam steps, 65% PDE points in [4,5]
Stage 3: 600 Adam steps, 50% PDE points in [4,5]
Then:   4 x 100 block-wise L-BFGS steps, with fresh physics points per block.

The schedule is deliberately much shorter than training a new universal model
from random initialization. Whether it is actually cheaper and accurate enough
must be verified experimentally.

Expected files in project root:
    train_multi_pulse_pinn.py
    run_universal_full81_CNN_DDNN.py

Default parent checkpoint:
    <run-dir>/sparse8_forward_pinn.pt

Default output:
    <run-dir>/universal_pinn_incremental_0to25_purephysics/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_multi_pulse_pinn import ConditionalPINN
from run_universal_full81_CNN_DDNN import (
    load_sparse8_csv,
    metrics_from_maps,
    parse_done_indices,
    run_ssfm_batch_selected,
    safe_device,
    select_eval_positions,
)

SCRIPT_VERSION = "UNIVERSAL_PINN_INCREMENTAL_0TO25_PUREPHYSICS_V1_20260717"

SLOT_CENTERS = np.asarray(
    [-28.0, -20.0, -12.0, -4.0, 4.0, 12.0, 20.0, 28.0],
    dtype=np.float32,
)

K_WEIGHTS = {
    1: 0.5,
    2: 0.5,
    3: 0.7,
    4: 0.9,
    5: 1.1,
    6: 1.4,
    7: 1.8,
    8: 2.4,
}


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: Mapping[str, Any]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(dict(row))
        f.flush()


def parse_schedule(text: str) -> List[Tuple[int, float, float]]:
    """
    Format:
        steps:new_ratio:lr,steps:new_ratio:lr,...
    Example:
        800:0.85:2e-4,800:0.65:1e-4,600:0.50:5e-5
    """
    out: List[Tuple[int, float, float]] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("Bad stage schedule item: %s" % item)
        steps = int(parts[0])
        ratio = float(parts[1])
        lr = float(parts[2])
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if not (0.0 <= ratio <= 1.0):
            raise ValueError("new-region ratio must lie in [0,1]")
        if lr <= 0:
            raise ValueError("learning rate must be > 0")
        out.append((steps, ratio, lr))
    if not out:
        raise ValueError("Empty stage schedule.")
    return out


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> Tuple[ConditionalPINN, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected checkpoint payload: %s" % path)

    cfg = dict(payload.get("model_config", {}))
    state = payload.get("model_state", payload.get("state_dict"))
    if state is None:
        raise RuntimeError("Checkpoint has no model_state/state_dict: %s" % path)

    model = ConditionalPINN(**cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def save_checkpoint(
    path: Path,
    model: ConditionalPINN,
    parent_checkpoint: Path,
    model_cfg: Mapping[str, Any],
    pde_params: Mapping[str, Any],
    train_meta: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_state": model.state_dict(),
            "model_config": dict(model_cfg),
            "pde_params": dict(pde_params),
            "parent_checkpoint": str(parent_checkpoint),
            "incremental_meta": {
                "original_trained_z_max_ld": 4.0,
                "extended_training_z_max_ld": 5.0,
                "same_network_as_parent": True,
                "model_config_z_max_ld_kept_unchanged": True,
                "ssfm_training_labels": False,
            },
            "train_meta": dict(train_meta),
        },
        str(path),
    )


# =============================================================================
# K-balanced amplitude sampling
# =============================================================================

class AmplitudeSampler:
    def __init__(
        self,
        k_labels: np.ndarray,
        amplitudes: np.ndarray,
        mode: str,
    ) -> None:
        self.groups: Dict[int, np.ndarray] = {}
        for k in range(1, 9):
            arr = np.asarray(amplitudes[k_labels == k], dtype=np.float32)
            if len(arr) == 0:
                raise RuntimeError("No seen configurations for K=%d" % k)
            self.groups[k] = arr

        mode = str(mode).lower().strip()
        uniform = np.ones(8, dtype=np.float64)
        high = np.asarray(
            [K_WEIGHTS[k] for k in range(1, 9)],
            dtype=np.float64,
        )

        if mode == "balanced":
            probs = uniform
        elif mode == "highk":
            probs = high
        elif mode == "mixed":
            probs = 0.5 * uniform / uniform.sum() + 0.5 * high / high.sum()
        else:
            raise ValueError("Unknown k_sampling: %s" % mode)

        self.probs = probs / probs.sum()

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        k_values = rng.choice(
            np.arange(1, 9, dtype=np.int64),
            size=int(n),
            replace=True,
            p=self.probs,
        )
        amps = np.empty((int(n), 8), dtype=np.float32)

        for k in range(1, 9):
            pos = np.flatnonzero(k_values == k)
            if len(pos) == 0:
                continue
            pool = self.groups[k]
            pick = rng.integers(0, len(pool), size=len(pos), endpoint=False)
            amps[pos] = pool[pick]

        return k_values, amps


def choose_active_slot(
    active_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.sum(active_mask, axis=1).astype(np.int64)
    counts = np.maximum(counts, 1)
    rank = np.floor(rng.random(len(active_mask)) * counts).astype(np.int64)
    csum = np.cumsum(active_mask.astype(np.int64), axis=1)
    return np.argmax(csum > rank[:, None], axis=1)


def sample_time(
    amplitudes: np.ndarray,
    rng: np.random.Generator,
    t_min: float,
    t_max: float,
    global_fraction: float,
    pulse_fraction: float,
    midpoint_fraction: float,
    pulse_sigma: float,
    midpoint_sigma: float,
) -> np.ndarray:
    n = len(amplitudes)
    total = (
        float(global_fraction)
        + float(pulse_fraction)
        + float(midpoint_fraction)
    )
    if total <= 0:
        raise ValueError("Time fractions sum to zero.")

    pg = float(global_fraction) / total
    pp = float(pulse_fraction) / total

    r = rng.random(n)
    t = np.empty(n, dtype=np.float32)
    active = amplitudes > 1e-12

    global_pos = np.flatnonzero(r < pg)
    pulse_pos = np.flatnonzero((r >= pg) & (r < pg + pp))
    mid_pos = np.flatnonzero(r >= pg + pp)

    if len(global_pos):
        t[global_pos] = rng.uniform(
            float(t_min),
            float(t_max),
            size=len(global_pos),
        ).astype(np.float32)

    if len(pulse_pos):
        slots = choose_active_slot(active[pulse_pos], rng)
        centers = SLOT_CENTERS[slots]
        t[pulse_pos] = rng.normal(
            centers,
            float(pulse_sigma),
        ).astype(np.float32)

    if len(mid_pos):
        sub = active[mid_pos]
        counts = np.sum(sub, axis=1).astype(np.int64)
        first = np.argmax(sub, axis=1)
        rank = np.floor(
            rng.random(len(mid_pos)) * np.maximum(counts - 1, 1)
        ).astype(np.int64)
        left = np.minimum(first + rank, 7)
        right = np.minimum(left + 1, 7)
        centers = 0.5 * (SLOT_CENTERS[left] + SLOT_CENTERS[right])
        single = counts <= 1
        centers[single] = SLOT_CENTERS[first[single]]

        t[mid_pos] = rng.normal(
            centers,
            float(midpoint_sigma),
        ).astype(np.float32)

    return np.clip(t, float(t_min), float(t_max)).reshape(-1, 1)


def sample_z_mixed(
    n: int,
    new_ratio: float,
    rng: np.random.Generator,
    old_z_max: float = 4.0,
    new_z_max: float = 5.0,
) -> np.ndarray:
    n = int(n)
    choose_new = rng.random(n) < float(new_ratio)
    z = np.empty(n, dtype=np.float32)

    n_new = int(np.sum(choose_new))
    n_old = n - n_new

    if n_new:
        # Half uniform in [4,5], half slightly biased toward the frontier z=5.
        u = rng.random(n_new)
        front = rng.random(n_new) < 0.5
        vals = float(old_z_max) + (float(new_z_max) - float(old_z_max)) * u
        vals[front] = (
            float(new_z_max)
            - (float(new_z_max) - float(old_z_max)) * (u[front] ** 2)
        )
        z[choose_new] = vals.astype(np.float32)

    if n_old:
        # Old-region rehearsal: uniform plus some points near z=4.
        u = rng.random(n_old)
        near_interface = rng.random(n_old) < 0.35
        vals = float(old_z_max) * u
        vals[near_interface] = float(old_z_max) * (
            1.0 - u[near_interface] ** 2
        )
        z[~choose_new] = vals.astype(np.float32)

    return z.reshape(-1, 1)


# =============================================================================
# Physics losses
# =============================================================================

def analytic_initial_field(
    t: torch.Tensor,
    amplitudes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    centers = torch.as_tensor(
        SLOT_CENTERS,
        device=t.device,
        dtype=t.dtype,
    ).view(1, 8)

    # t: [N,1], amplitudes: [N,8]
    basis = torch.exp(
        -0.5 * (t - centers) ** 2
    )
    u0 = torch.sum(amplitudes * basis, dim=1, keepdim=True)
    v0 = torch.zeros_like(u0)
    return u0, v0


def sample_pde_batch(
    sampler: AmplitudeSampler,
    n: int,
    new_ratio: float,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    _, amps = sampler.sample(int(n), rng)
    z = sample_z_mixed(
        int(n),
        float(new_ratio),
        rng,
        old_z_max=4.0,
        new_z_max=5.0,
    )
    t = sample_time(
        amps,
        rng,
        float(cfg["t_min"]),
        float(cfg["t_max"]),
        float(args.time_global_fraction),
        float(args.time_pulse_fraction),
        float(args.time_midpoint_fraction),
        float(args.pulse_local_sigma),
        float(args.midpoint_local_sigma),
    )

    return {
        "z": torch.from_numpy(z).to(device),
        "t": torch.from_numpy(t).to(device),
        "a": torch.from_numpy(amps).to(device),
    }


def sample_ic_batch(
    sampler: AmplitudeSampler,
    n: int,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    _, amps = sampler.sample(int(n), rng)
    t = sample_time(
        amps,
        rng,
        float(cfg["t_min"]),
        float(cfg["t_max"]),
        float(args.time_global_fraction),
        float(args.time_pulse_fraction),
        float(args.time_midpoint_fraction),
        float(args.pulse_local_sigma),
        float(args.midpoint_local_sigma),
    )
    z = np.zeros((int(n), 1), dtype=np.float32)
    return {
        "z": torch.from_numpy(z).to(device),
        "t": torch.from_numpy(t).to(device),
        "a": torch.from_numpy(amps).to(device),
    }


def sample_edge_batch(
    sampler: AmplitudeSampler,
    n: int,
    new_ratio: float,
    cfg: Mapping[str, Any],
    rng: np.random.Generator,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    _, amps = sampler.sample(int(n), rng)
    z = sample_z_mixed(int(n), float(new_ratio), rng)
    side = rng.integers(0, 2, size=int(n))
    t = np.where(
        side == 0,
        float(cfg["t_min"]),
        float(cfg["t_max"]),
    ).astype(np.float32).reshape(-1, 1)

    return {
        "z": torch.from_numpy(z).to(device),
        "t": torch.from_numpy(t).to(device),
        "a": torch.from_numpy(amps).to(device),
    }


def pde_loss(
    model: ConditionalPINN,
    batch: Mapping[str, torch.Tensor],
    beta2_norm: float,
    n_sq: float,
) -> torch.Tensor:
    z = batch["z"].detach().clone().requires_grad_(True)
    t = batch["t"].detach().clone().requires_grad_(True)
    a = batch["a"]

    u, v = model(z, t, a)
    ones = torch.ones_like(u)

    u_z = torch.autograd.grad(
        u, z, ones, create_graph=True, retain_graph=True
    )[0]
    v_z = torch.autograd.grad(
        v, z, ones, create_graph=True, retain_graph=True
    )[0]

    u_t = torch.autograd.grad(
        u, t, ones, create_graph=True, retain_graph=True
    )[0]
    v_t = torch.autograd.grad(
        v, t, ones, create_graph=True, retain_graph=True
    )[0]

    u_tt = torch.autograd.grad(
        u_t, t, torch.ones_like(u_t), create_graph=True, retain_graph=True
    )[0]
    v_tt = torch.autograd.grad(
        v_t, t, torch.ones_like(v_t), create_graph=True, retain_graph=True
    )[0]

    power = u * u + v * v

    r_real = (
        -v_z
        + 0.5 * float(beta2_norm) * u_tt
        + float(n_sq) * power * u
    )
    r_imag = (
        u_z
        + 0.5 * float(beta2_norm) * v_tt
        + float(n_sq) * power * v
    )

    return torch.mean(r_real * r_real + r_imag * r_imag)


def ic_loss(
    model: ConditionalPINN,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    up, vp = model(batch["z"], batch["t"], batch["a"])
    ut, vt = analytic_initial_field(batch["t"], batch["a"])

    target_scale = torch.mean(ut * ut + vt * vt).detach().clamp_min(1e-8)
    return torch.mean(
        (up - ut) ** 2 + (vp - vt) ** 2
    ) / target_scale


def edge_loss(
    model: ConditionalPINN,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    u, v = model(batch["z"], batch["t"], batch["a"])
    return torch.mean(u * u + v * v)


def power_conservation_loss(
    model: ConditionalPINN,
    sampler: AmplitudeSampler,
    new_ratio: float,
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    b = int(args.power_cases_batch)
    nz = int(args.power_z_per_case)
    nt = int(args.power_t_points)

    _, amps_np = sampler.sample(b, rng)
    amps = torch.from_numpy(amps_np).to(device)

    t_grid = torch.linspace(
        float(cfg["t_min"]),
        float(cfg["t_max"]),
        nt,
        device=device,
        dtype=torch.float32,
    )

    z_np = sample_z_mixed(
        b * nz,
        float(new_ratio),
        rng,
    ).reshape(b, nz)
    z_rand = torch.from_numpy(z_np).to(device)

    z = z_rand[:, :, None].expand(b, nz, nt).reshape(-1, 1)
    t = t_grid[None, None, :].expand(b, nz, nt).reshape(-1, 1)
    a = amps[:, None, None, :].expand(b, nz, nt, 8).reshape(-1, 8)

    u, v = model(z, t, a)
    pz = (
        (u * u + v * v)
        .reshape(b, nz, nt)
        .mean(dim=2)
    )

    t0 = t_grid[None, :].expand(b, nt).reshape(-1, 1)
    a0 = amps[:, None, :].expand(b, nt, 8).reshape(-1, 8)
    u0, v0 = analytic_initial_field(t0, a0)
    p0 = (
        (u0 * u0 + v0 * v0)
        .reshape(b, nt)
        .mean(dim=1)
    )

    return torch.mean(
        ((pz - p0[:, None]) ** 2)
        / (p0[:, None] ** 2 + 1e-8)
    )


def compute_total_loss(
    model: ConditionalPINN,
    sampler: AmplitudeSampler,
    cfg: Mapping[str, Any],
    pde_params: Mapping[str, Any],
    args: argparse.Namespace,
    new_ratio: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    pde_batch = sample_pde_batch(
        sampler,
        int(args.pde_batch_points),
        float(new_ratio),
        cfg,
        args,
        rng,
        device,
    )
    ic_batch = sample_ic_batch(
        sampler,
        int(args.ic_batch_points),
        cfg,
        args,
        rng,
        device,
    )
    edge_batch = sample_edge_batch(
        sampler,
        int(args.edge_batch_points),
        float(new_ratio),
        cfg,
        rng,
        device,
    )

    lpde = pde_loss(
        model,
        pde_batch,
        float(pde_params["beta2_norm"]),
        float(pde_params["N_sq"]),
    )
    lic = ic_loss(model, ic_batch)

    if float(args.edge_weight) > 0:
        ledge = edge_loss(model, edge_batch)
    else:
        ledge = torch.tensor(0.0, device=device)

    if float(args.power_weight) > 0:
        lpower = power_conservation_loss(
            model,
            sampler,
            float(new_ratio),
            cfg,
            args,
            rng,
            device,
        )
    else:
        lpower = torch.tensor(0.0, device=device)

    loss = (
        float(args.pde_weight) * lpde
        + float(args.ic_weight) * lic
        + float(args.edge_weight) * ledge
        + float(args.power_weight) * lpower
    )

    return loss, {
        "pde": lpde,
        "ic": lic,
        "edge": ledge,
        "power": lpower,
    }


# =============================================================================
# Training
# =============================================================================

def train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    device = safe_device(args.device)

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(str(run_dir))

    parent_checkpoint = (
        Path(args.parent_checkpoint).expanduser().resolve()
        if args.parent_checkpoint
        else run_dir / "sparse8_forward_pinn.pt"
    )
    if not parent_checkpoint.is_file():
        raise FileNotFoundError(
            "Parent checkpoint not found: %s" % parent_checkpoint
        )

    model, parent_payload = load_checkpoint(parent_checkpoint, device)
    model.train()

    cfg = dict(parent_payload["model_config"])
    pde_raw = dict(parent_payload.get("pde_params", {}))
    pde_params = {
        "beta2_norm": float(pde_raw.get("beta2_norm", 1.0)),
        "N_sq": float(pde_raw.get("N_sq", 1.0)),
        "alpha_norm": float(pde_raw.get("alpha_norm", 0.0)),
        "has_tod": bool(pde_raw.get("has_tod", False)),
        "has_ss": bool(pde_raw.get("has_ss", False)),
        "has_irs": bool(pde_raw.get("has_irs", False)),
    }

    if int(cfg.get("n_pulses", -1)) != 8:
        raise RuntimeError("Expected universal n_pulses=8 checkpoint.")
    if abs(float(cfg.get("z_max_ld", 4.0)) - 4.0) > 1e-6:
        print(
            "[warning] parent model_config z_max_ld is not exactly 4.0:",
            cfg.get("z_max_ld"),
            flush=True,
        )
    if abs(float(pde_params["alpha_norm"])) > 1e-12:
        raise RuntimeError("Current script assumes alpha_norm=0.")
    if (
        bool(pde_params["has_tod"])
        or bool(pde_params["has_ss"])
        or bool(pde_params["has_irs"])
    ):
        raise RuntimeError(
            "Current script targets the GVD+SPM NLSE used by the universal model."
        )

    seen_k, seen_a = load_sparse8_csv(
        run_dir / "dataset" / "seen_sparse8_combinations.csv"
    )
    sampler = AmplitudeSampler(
        seen_k,
        seen_a,
        str(args.k_sampling),
    )

    output_root = ensure_dir(
        run_dir / "universal_pinn_incremental_0to25_purephysics"
    )
    train_root = ensure_dir(output_root / "train")
    history_path = train_root / "history.csv"

    final_checkpoint = train_root / "incremental_0to25_pinn.pt"
    if final_checkpoint.is_file() and not args.force:
        print("[skip] final checkpoint already exists:", final_checkpoint, flush=True)
        return final_checkpoint

    schedule = parse_schedule(args.adam_schedule)

    print("=" * 116, flush=True)
    print("SCRIPT VERSION :", SCRIPT_VERSION, flush=True)
    print("parent model   :", parent_checkpoint, flush=True)
    print("model config   :", cfg, flush=True)
    print("same network   : YES - all parent weights loaded and trainable", flush=True)
    print("coordinate map : parent z_max_ld kept unchanged =", cfg.get("z_max_ld"), flush=True)
    print("new domain     : z=4..5 LD (20..25 km)", flush=True)
    print("seen configs   :", len(seen_a), flush=True)
    print("Adam schedule  :", schedule, flush=True)
    print(
        "L-BFGS         : %d blocks x %d steps"
        % (int(args.lbfgs_blocks), int(args.lbfgs_steps_per_block)),
        flush=True,
    )
    print("SSFM labels    : NONE in training", flush=True)
    print("output         :", output_root, flush=True)
    print("=" * 116, flush=True)

    history: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(args.seed) + 424242)
    global_step = 0
    t0 = time.time()

    # -------------------------------------------------------------------------
    # Adam incremental adaptation.
    # -------------------------------------------------------------------------
    for stage_idx, (stage_steps, new_ratio, lr) in enumerate(
        schedule,
        start=1,
    ):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(args.weight_decay),
        )

        print(
            "\n[Adam stage %d/%d] steps=%d new-region-ratio=%.2f lr=%.2e"
            % (
                stage_idx,
                len(schedule),
                stage_steps,
                new_ratio,
                lr,
            ),
            flush=True,
        )

        for local_step in range(1, int(stage_steps) + 1):
            global_step += 1
            model.train()
            optimizer.zero_grad(set_to_none=True)

            loss, parts = compute_total_loss(
                model,
                sampler,
                cfg,
                pde_params,
                args,
                float(new_ratio),
                rng,
                device,
            )

            loss.backward()

            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(args.grad_clip),
                )

            optimizer.step()

            if (
                local_step == 1
                or local_step % int(args.log_every) == 0
                or local_step == int(stage_steps)
            ):
                row = {
                    "phase": "adam",
                    "stage": int(stage_idx),
                    "global_step": int(global_step),
                    "new_region_ratio": float(new_ratio),
                    "learning_rate": float(lr),
                    "loss": float(loss.detach().cpu()),
                    "loss_pde": float(parts["pde"].detach().cpu()),
                    "loss_ic": float(parts["ic"].detach().cpu()),
                    "loss_edge": float(parts["edge"].detach().cpu()),
                    "loss_power": float(parts["power"].detach().cpu()),
                    "elapsed_sec": float(time.time() - t0),
                }
                history.append(row)
                save_rows(history_path, history)

                print(
                    "[Adam %5d] loss=%.4e pde=%.4e ic=%.4e edge=%.4e power=%.4e"
                    % (
                        global_step,
                        row["loss"],
                        row["loss_pde"],
                        row["loss_ic"],
                        row["loss_edge"],
                        row["loss_power"],
                    ),
                    flush=True,
                )

        stage_ckpt = train_root / (
            "incremental_after_adam_stage%d.pt" % stage_idx
        )
        save_checkpoint(
            stage_ckpt,
            model,
            parent_checkpoint,
            cfg,
            pde_raw,
            {
                "phase": "adam_stage_%d" % stage_idx,
                "global_step": int(global_step),
                "schedule": schedule,
                "args": vars(args),
            },
        )

    # -------------------------------------------------------------------------
    # Block-wise L-BFGS on fresh pure-physics point sets.
    # -------------------------------------------------------------------------
    total_lbfgs_step = 0

    for block in range(1, int(args.lbfgs_blocks) + 1):
        block_rng = np.random.default_rng(
            int(args.seed) + 900000 + block
        )

        # Final refinement uses a balanced mix of old and new domains.
        fixed_pde = sample_pde_batch(
            sampler,
            int(args.lbfgs_pde_points),
            float(args.lbfgs_new_region_ratio),
            cfg,
            args,
            block_rng,
            device,
        )
        fixed_ic = sample_ic_batch(
            sampler,
            int(args.lbfgs_ic_points),
            cfg,
            args,
            block_rng,
            device,
        )
        fixed_edge = sample_edge_batch(
            sampler,
            int(args.lbfgs_edge_points),
            float(args.lbfgs_new_region_ratio),
            cfg,
            block_rng,
            device,
        )

        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=float(args.lbfgs_lr),
            max_iter=1,
            max_eval=int(args.lbfgs_max_eval),
            history_size=int(args.lbfgs_history_size),
            tolerance_grad=float(args.lbfgs_tolerance_grad),
            tolerance_change=float(args.lbfgs_tolerance_change),
            line_search_fn=(
                "strong_wolfe"
                if bool(args.lbfgs_strong_wolfe)
                else None
            ),
        )

        print(
            "\n[L-BFGS block %d/%d] fresh physics points"
            % (block, int(args.lbfgs_blocks)),
            flush=True,
        )

        for step in range(1, int(args.lbfgs_steps_per_block) + 1):
            total_lbfgs_step += 1
            last: Dict[str, float] = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)

                lpde = pde_loss(
                    model,
                    fixed_pde,
                    float(pde_params["beta2_norm"]),
                    float(pde_params["N_sq"]),
                )
                lic = ic_loss(model, fixed_ic)
                ledge = (
                    edge_loss(model, fixed_edge)
                    if float(args.edge_weight) > 0
                    else torch.tensor(0.0, device=device)
                )

                # Use a freshly sampled power batch for every closure call.
                lpower = (
                    power_conservation_loss(
                        model,
                        sampler,
                        float(args.lbfgs_new_region_ratio),
                        cfg,
                        args,
                        block_rng,
                        device,
                    )
                    if float(args.power_weight) > 0
                    else torch.tensor(0.0, device=device)
                )

                loss = (
                    float(args.pde_weight) * lpde
                    + float(args.ic_weight) * lic
                    + float(args.edge_weight) * ledge
                    + float(args.power_weight) * lpower
                )

                loss.backward()

                last["loss"] = float(loss.detach().cpu())
                last["pde"] = float(lpde.detach().cpu())
                last["ic"] = float(lic.detach().cpu())
                last["edge"] = float(ledge.detach().cpu())
                last["power"] = float(lpower.detach().cpu())
                return loss

            optimizer.step(closure)

            if (
                step == 1
                or step % int(args.log_every) == 0
                or step == int(args.lbfgs_steps_per_block)
            ):
                row = {
                    "phase": "lbfgs",
                    "stage": int(block),
                    "global_step": int(total_lbfgs_step),
                    "new_region_ratio": float(args.lbfgs_new_region_ratio),
                    "learning_rate": float(args.lbfgs_lr),
                    "loss": float(last["loss"]),
                    "loss_pde": float(last["pde"]),
                    "loss_ic": float(last["ic"]),
                    "loss_edge": float(last["edge"]),
                    "loss_power": float(last["power"]),
                    "elapsed_sec": float(time.time() - t0),
                }
                history.append(row)
                save_rows(history_path, history)

                print(
                    "[L-BFGS %4d] loss=%.4e pde=%.4e ic=%.4e edge=%.4e power=%.4e"
                    % (
                        total_lbfgs_step,
                        row["loss"],
                        row["loss_pde"],
                        row["loss_ic"],
                        row["loss_edge"],
                        row["loss_power"],
                    ),
                    flush=True,
                )

        block_ckpt = train_root / (
            "incremental_after_lbfgs_block%d.pt" % block
        )
        save_checkpoint(
            block_ckpt,
            model,
            parent_checkpoint,
            cfg,
            pde_raw,
            {
                "phase": "lbfgs_block_%d" % block,
                "total_lbfgs_step": int(total_lbfgs_step),
                "args": vars(args),
            },
        )

    save_checkpoint(
        final_checkpoint,
        model,
        parent_checkpoint,
        cfg,
        pde_raw,
        {
            "phase": "final",
            "adam_steps": int(sum(x[0] for x in schedule)),
            "lbfgs_steps": int(
                args.lbfgs_blocks * args.lbfgs_steps_per_block
            ),
            "args": vars(args),
        },
    )

    write_json(
        output_root / "train_config.json",
        {
            "script_version": SCRIPT_VERSION,
            "parent_checkpoint": str(parent_checkpoint),
            "final_checkpoint": str(final_checkpoint),
            "same_network_finetuning": True,
            "model_config_kept_unchanged": cfg,
            "effective_training_z_max_ld": 5.0,
            "pure_physics_training": True,
            "ssfm_training_labels": False,
            "adam_schedule": [
                {
                    "steps": int(s),
                    "new_region_ratio": float(r),
                    "learning_rate": float(lr),
                }
                for s, r, lr in schedule
            ],
            "args": vars(args),
        },
    )

    print("\n[done] final checkpoint:", final_checkpoint, flush=True)
    return final_checkpoint


# =============================================================================
# Evaluation
# =============================================================================

def build_eval_grid(
    cfg: Mapping[str, Any],
    ssfm_half_window: float,
    n_t: int,
    n_z: int,
    n_slices: int,
) -> Dict[str, Any]:
    tau_full = np.linspace(
        -float(ssfm_half_window),
        float(ssfm_half_window),
        int(n_t),
        endpoint=False,
    )
    t_min = float(cfg["t_min"])
    t_max = float(cfg["t_max"])
    mask = (
        (tau_full >= t_min - 1e-12)
        & (tau_full <= t_max + 1e-12)
    )

    selected_steps = np.rint(
        np.linspace(0, int(n_z), int(n_slices))
    ).astype(np.int64)

    if len(np.unique(selected_steps)) != len(selected_steps):
        raise ValueError(
            "Duplicate selected SSFM steps. Increase n_z or reduce n_slices."
        )

    zeta = 5.0 * selected_steps.astype(np.float64) / float(n_z)

    return {
        "ssfm_half_window": float(ssfm_half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "n_slices": int(n_slices),
        "z_max_ld": 5.0,
        "compare_t_min": t_min,
        "compare_t_max": t_max,
        "tau_full": tau_full,
        "time_mask": mask,
        "tau": tau_full[mask],
        "selected_steps": selected_steps,
        "zeta": zeta,
    }


def predict_maps(
    model: ConditionalPINN,
    amplitudes: np.ndarray,
    tau: np.ndarray,
    zeta: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    b = int(len(amplitudes))
    nz = int(len(zeta))
    nt = int(len(tau))
    total = b * nz * nt

    amp_t = torch.as_tensor(
        amplitudes,
        dtype=torch.float32,
        device=device,
    )
    tau_t = torch.as_tensor(
        tau,
        dtype=torch.float32,
        device=device,
    )
    z_t = torch.as_tensor(
        zeta,
        dtype=torch.float32,
        device=device,
    )

    out = np.empty((total, 2), dtype=np.float32)

    model.eval()
    with torch.inference_mode():
        for start in range(0, total, int(chunk_size)):
            end = min(total, start + int(chunk_size))
            flat = torch.arange(
                start,
                end,
                device=device,
                dtype=torch.long,
            )

            sample_idx = torch.div(
                flat,
                nz * nt,
                rounding_mode="floor",
            )
            rem = flat % (nz * nt)
            z_idx = torch.div(rem, nt, rounding_mode="floor")
            t_idx = rem % nt

            z = z_t[z_idx].reshape(-1, 1)
            t = tau_t[t_idx].reshape(-1, 1)
            a = amp_t[sample_idx]

            u, v = model(z, t, a)
            out[start:end] = (
                torch.cat([u, v], dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

    return out.reshape(b, nz, nt, 2).transpose(0, 1, 3, 2)


def rel_l2_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    eps = 1e-300

    field = (
        np.sqrt(np.sum((pred - ref) ** 2, axis=(1, 2, 3)))
        / np.maximum(
            np.sqrt(np.sum(ref ** 2, axis=(1, 2, 3))),
            eps,
        )
    )

    pp = np.sum(pred ** 2, axis=2)
    pr = np.sum(ref ** 2, axis=2)

    power = (
        np.sqrt(np.sum((pp - pr) ** 2, axis=(1, 2)))
        / np.maximum(
            np.sqrt(np.sum(pr ** 2, axis=(1, 2))),
            eps,
        )
    )

    return field, power


def summarize_eval(
    metrics_path: Path,
    output_dir: Path,
) -> None:
    with metrics_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError("No evaluation rows.")

    metrics = [
        "full_0to25_rel_l2_field",
        "full_0to25_rel_l2_power",
        "old_0to20_rel_l2_field",
        "old_0to20_rel_l2_power",
        "new_20to25_rel_l2_field",
        "new_20to25_rel_l2_power",
        "terminal_25km_rel_l2_field",
        "terminal_25km_rel_l2_power",
    ]

    kvals = sorted(set(int(r["K"]) for r in rows))
    summary_rows: List[Dict[str, Any]] = []

    for k in kvals:
        group = [r for r in rows if int(r["K"]) == k]
        item: Dict[str, Any] = {
            "K": int(k),
            "n_samples": int(len(group)),
        }
        for metric in metrics:
            values = np.asarray(
                [float(r[metric]) for r in group],
                dtype=np.float64,
            )
            item[metric + "_mean"] = float(np.mean(values))
            item[metric + "_median"] = float(np.median(values))
        summary_rows.append(item)

    save_rows(output_dir / "summary_by_k.csv", summary_rows)

    macro: Dict[str, float] = {}
    micro: Dict[str, float] = {}

    for metric in metrics:
        per_k_means = [
            float(row[metric + "_mean"])
            for row in summary_rows
        ]
        all_values = np.asarray(
            [float(r[metric]) for r in rows],
            dtype=np.float64,
        )

        macro[metric] = float(np.mean(per_k_means))
        micro[metric] = float(np.mean(all_values))

    write_json(
        output_dir / "overall_summary.json",
        {
            "n_samples": int(len(rows)),
            "macro_equal_K_weight": macro,
            "micro_sample_weighted": micro,
        },
    )

    print(
        "[summary] macro full 0-25 power=%.3f%% | old 0-20 power=%.3f%% | "
        "new 20-25 power=%.3f%% | 25km terminal power=%.3f%%"
        % (
            100.0 * macro["full_0to25_rel_l2_power"],
            100.0 * macro["old_0to20_rel_l2_power"],
            100.0 * macro["new_20to25_rel_l2_power"],
            100.0 * macro["terminal_25km_rel_l2_power"],
        ),
        flush=True,
    )


def evaluate(args: argparse.Namespace) -> None:
    device = safe_device(args.device)
    run_dir = Path(args.run_dir).expanduser().resolve()

    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else (
            run_dir
            / "universal_pinn_incremental_0to25_purephysics"
            / "train"
            / "incremental_0to25_pinn.pt"
        )
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))

    model, payload = load_checkpoint(checkpoint, device)
    cfg = dict(payload["model_config"])

    pde_raw = dict(payload.get("pde_params", {}))
    pde_params = {
        "beta2_norm": float(pde_raw.get("beta2_norm", 1.0)),
        "N_sq": float(pde_raw.get("N_sq", 1.0)),
        "alpha_norm": float(pde_raw.get("alpha_norm", 0.0)),
        "beta3_norm": float(pde_raw.get("beta3_norm", 0.0)),
        "has_tod": bool(pde_raw.get("has_tod", False)),
        "has_ss": bool(pde_raw.get("has_ss", False)),
        "has_irs": bool(pde_raw.get("has_irs", False)),
        "s": float(pde_raw.get("s", 0.0)),
        "ss_coef": float(pde_raw.get("ss_coef", 1.0)),
        "tau_R": float(pde_raw.get("tau_R", 0.0)),
    }

    unseen_k, unseen_a = load_sparse8_csv(
        run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    )

    positions = select_eval_positions(
        unseen_k,
        int(args.max_eval_per_k),
        int(args.eval_seed),
    )

    output_dir = ensure_dir(
        run_dir
        / "universal_pinn_incremental_0to25_purephysics"
        / "eval_0to25"
    )
    metrics_path = output_dir / "metrics_stream.csv"

    if args.overwrite_eval and metrics_path.exists():
        metrics_path.unlink()

    done = parse_done_indices(metrics_path)
    pending = [
        int(i)
        for i in positions.tolist()
        if int(i) not in done
    ]

    grid = build_eval_grid(
        cfg,
        float(args.ssfm_half_window),
        int(args.eval_n_t),
        int(args.eval_n_z),
        int(args.eval_slices),
    )

    zeta = np.asarray(grid["zeta"], dtype=np.float64)
    tau = np.asarray(grid["tau"], dtype=np.float64)

    old_mask = zeta <= 4.0 + 1e-12
    new_mask = zeta >= 4.0 - 1e-12

    fieldnames = (
        ["idx", "K"]
        + [f"A{i}" for i in range(1, 9)]
        + [
            "full_0to25_rel_l2_field",
            "full_0to25_rel_l2_power",
            "old_0to20_rel_l2_field",
            "old_0to20_rel_l2_power",
            "new_20to25_rel_l2_field",
            "new_20to25_rel_l2_power",
            "terminal_25km_rel_l2_field",
            "terminal_25km_rel_l2_power",
        ]
    )

    new_file = (
        not metrics_path.exists()
        or metrics_path.stat().st_size == 0
    )
    f = metrics_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    )
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()
        f.flush()

    print("=" * 112, flush=True)
    print("EVALUATION      : incremental same-network PINN 0->25 km", flush=True)
    print("checkpoint      :", checkpoint, flush=True)
    print("selected unseen :", len(positions), flush=True)
    print("pending         :", len(pending), flush=True)
    print("planes          :", len(zeta), flush=True)
    print("output          :", output_dir, flush=True)
    print("=" * 112, flush=True)

    completed = 0
    t0 = time.time()

    try:
        for start in range(
            0,
            len(pending),
            int(args.eval_batch_size),
        ):
            idx = np.asarray(
                pending[
                    start : start + int(args.eval_batch_size)
                ],
                dtype=np.int64,
            )
            amps = unseen_a[idx]
            kvals = unseen_k[idx]

            ref = run_ssfm_batch_selected(
                amps,
                grid,
                pde_params,
                device,
                bool(args.ssfm_complex64),
            )

            pred = predict_maps(
                model,
                amps,
                tau,
                zeta,
                device,
                int(args.eval_chunk_size),
            )

            full_field, full_power = rel_l2_metrics(pred, ref)
            old_field, old_power = rel_l2_metrics(
                pred[:, old_mask],
                ref[:, old_mask],
            )
            new_field, new_power = rel_l2_metrics(
                pred[:, new_mask],
                ref[:, new_mask],
            )

            terminal_pred = pred[:, -1:]
            terminal_ref = ref[:, -1:]
            term_field, term_power = rel_l2_metrics(
                terminal_pred,
                terminal_ref,
            )

            for i, sample_idx in enumerate(idx):
                row: Dict[str, Any] = {
                    "idx": int(sample_idx),
                    "K": int(kvals[i]),
                }
                for j in range(8):
                    row["A%d" % (j + 1)] = float(amps[i, j])

                row.update(
                    {
                        "full_0to25_rel_l2_field": float(full_field[i]),
                        "full_0to25_rel_l2_power": float(full_power[i]),
                        "old_0to20_rel_l2_field": float(old_field[i]),
                        "old_0to20_rel_l2_power": float(old_power[i]),
                        "new_20to25_rel_l2_field": float(new_field[i]),
                        "new_20to25_rel_l2_power": float(new_power[i]),
                        "terminal_25km_rel_l2_field": float(term_field[i]),
                        "terminal_25km_rel_l2_power": float(term_power[i]),
                    }
                )
                writer.writerow(row)

            f.flush()
            os.fsync(f.fileno())

            completed += len(idx)
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-9)
            eta = (
                len(pending) - completed
            ) / max(rate, 1e-12)

            print(
                "[eval] %d/%d | old20=%.2f%% new20-25=%.2f%% "
                "terminal25=%.2f%% | rate=%.2f sample/s ETA=%.1f min"
                % (
                    completed,
                    len(pending),
                    100.0 * float(np.mean(old_power)),
                    100.0 * float(np.mean(new_power)),
                    100.0 * float(np.mean(term_power)),
                    rate,
                    eta / 60.0,
                ),
                flush=True,
            )

            del ref, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        f.close()

    summarize_eval(metrics_path, output_dir)


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Incrementally fine-tune the SAME universal PINN from 0-20 km "
            "to 0-25 km using pure physics only."
        )
    )

    p.add_argument(
        "--stage",
        choices=("train", "eval", "all"),
        default="train",
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("--parent-checkpoint", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")

    # Training schedule.
    p.add_argument(
        "--adam-schedule",
        default="800:0.85:2e-4,800:0.65:1e-4,600:0.50:5e-5",
    )
    p.add_argument("--weight-decay", type=float, default=0.0)

    # Per-Adam-step physics samples.
    p.add_argument("--pde-batch-points", type=int, default=8192)
    p.add_argument("--ic-batch-points", type=int, default=2048)
    p.add_argument("--edge-batch-points", type=int, default=512)

    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--ic-weight", type=float, default=1.0)
    p.add_argument("--edge-weight", type=float, default=0.05)
    p.add_argument("--power-weight", type=float, default=0.05)

    p.add_argument("--power-cases-batch", type=int, default=6)
    p.add_argument("--power-z-per-case", type=int, default=3)
    p.add_argument("--power-t-points", type=int, default=256)

    # K/time sampling.
    p.add_argument(
        "--k-sampling",
        choices=("balanced", "highk", "mixed"),
        default="mixed",
    )
    p.add_argument("--time-global-fraction", type=float, default=0.4)
    p.add_argument("--time-pulse-fraction", type=float, default=0.4)
    p.add_argument("--time-midpoint-fraction", type=float, default=0.2)
    p.add_argument("--pulse-local-sigma", type=float, default=3.5)
    p.add_argument("--midpoint-local-sigma", type=float, default=2.5)

    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)

    # L-BFGS refinement.
    p.add_argument("--lbfgs-blocks", type=int, default=4)
    p.add_argument("--lbfgs-steps-per-block", type=int, default=100)
    p.add_argument("--lbfgs-new-region-ratio", type=float, default=0.5)
    p.add_argument("--lbfgs-pde-points", type=int, default=16384)
    p.add_argument("--lbfgs-ic-points", type=int, default=4096)
    p.add_argument("--lbfgs-edge-points", type=int, default=1024)

    p.add_argument("--lbfgs-lr", type=float, default=0.5)
    p.add_argument("--lbfgs-max-eval", type=int, default=4)
    p.add_argument("--lbfgs-history-size", type=int, default=50)
    p.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-10)
    p.add_argument("--lbfgs-tolerance-change", type=float, default=1e-13)
    p.add_argument("--lbfgs-strong-wolfe", action="store_true")

    # Evaluation.
    p.add_argument("--max-eval-per-k", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=2027)
    p.add_argument("--eval-slices", type=int, default=101)
    p.add_argument("--eval-n-t", type=int, default=2048)
    p.add_argument("--eval-n-z", type=int, default=625)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--eval-chunk-size", type=int, default=65536)
    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--overwrite-eval", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    trained_checkpoint: Optional[Path] = None

    if args.stage in ("train", "all"):
        trained_checkpoint = train(args)

    if args.stage in ("eval", "all"):
        if trained_checkpoint is not None and not args.checkpoint:
            args.checkpoint = str(trained_checkpoint)
        evaluate(args)


if __name__ == "__main__":
    main()
