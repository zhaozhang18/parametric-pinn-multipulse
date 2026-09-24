# -*- coding: utf-8 -*-
"""
train_fixedM_cnn_full_propagation_v2.py

Purely data-driven fixed-M CNN baseline for nonlinear multi-pulse propagation.

Task
----
    initial complex field h(0,t)  ->  full complex propagation field h(z,t)

The seen/unseen amplitude combinations are NOT re-sampled.  They are read
verbatim from each fixed-M PINN run directory:
    dataset/seen_combinations.csv
    dataset/unseen_combinations.csv

For storage control, the SSFM trajectory is sampled at a fixed number of
propagation planes (default 81) from z/L_D=0 to 4.  This includes the nine
paper-style waterfall planes
    0, 0.5, 1.0, ..., 4.0.

Directory layout under each selected M*_amp4_r* run:
    cnn_full_propagation/
        dataset/
        train/seed_<seed>/
        eval/seed_<seed>/
        eval/aggregate/

Python 3.8 compatible.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlse import NLSEParams, load_combinations_csv


SCRIPT_VERSION = "CNN_FULL_PROPAGATION_V2_20260626"


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

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


def safe_device(text: str) -> torch.device:
    if str(text).lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(text)
    print("[warning] CUDA is unavailable; falling back to CPU.")
    return torch.device("cpu")


def parse_ratio_from_name(name: str) -> float:
    match = re.search(r"_r([0-9]+(?:p[0-9]+)?)", str(name))
    if not match:
        return -1.0
    return float(match.group(1).replace("p", "."))


def parse_m_from_name(name: str) -> Optional[int]:
    match = re.match(r"M(\d+)_amp4_", str(name))
    return int(match.group(1)) if match else None


def find_run_dir(runs_root: Path, M: int) -> Path:
    candidates: List[Path] = []
    for path in runs_root.iterdir():
        if path.is_dir() and parse_m_from_name(path.name) == int(M):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No fixed-M run directory found for M=%d under %s" % (M, runs_root))
    selected = max(candidates, key=lambda p: (parse_ratio_from_name(p.name), p.stat().st_mtime))
    if len(candidates) > 1:
        print("[run selection] M=%d: choosing the largest training ratio:" % M)
        for p in sorted(candidates, key=lambda x: parse_ratio_from_name(x.name)):
            suffix = "  <-- selected" if p == selected else ""
            print("  ratio=%g  %s%s" % (parse_ratio_from_name(p.name), p, suffix))
    return selected


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def combo_text(combo: Sequence[float]) -> str:
    return ";".join("%g" % float(x) for x in combo)


def parse_seeds(values: Sequence[int]) -> List[int]:
    return sorted(set(int(x) for x in values))


def load_grid(run_dir: Path, n_t_override: int = 0, n_z_override: int = 0) -> Dict[str, Any]:
    path = run_dir / "ssfm_eval_grid.json"
    if not path.exists():
        raise FileNotFoundError("Missing SSFM grid file: %s" % path)
    grid = read_json(path)
    if int(n_t_override) > 0:
        grid["eval_n_t"] = int(n_t_override)
    if int(n_z_override) > 0:
        grid["eval_n_z"] = int(n_z_override)
    return grid


def make_tau_and_mask(grid: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = float(grid["eval_half_window_t0"])
    n_t = int(grid["eval_n_t"])
    tau_full = np.linspace(-half, half, n_t, endpoint=False, dtype=np.float64)
    compare_min = float(grid.get("compare_t_min", -half))
    compare_max = float(grid.get("compare_t_max", half))
    mask = (tau_full >= compare_min) & (tau_full <= compare_max)
    return tau_full, mask, tau_full[mask]


def make_z_sampling(n_z: int, z_max: float, n_slices: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_slices < 9:
        raise ValueError("--propagation-slices must be at least 9.")
    if n_slices > n_z + 1:
        raise ValueError("--propagation-slices cannot exceed n_z+1.")
    steps = np.rint(np.linspace(0, n_z, n_slices)).astype(np.int64)
    if len(np.unique(steps)) != len(steps):
        raise ValueError("Propagation-slice selection produced duplicate SSFM steps.")
    zeta_nominal = np.linspace(0.0, float(z_max), n_slices, dtype=np.float64)
    return steps, zeta_nominal


# -----------------------------------------------------------------------------
# Batched SSFM, numerically matching the project SSFM implementation
# -----------------------------------------------------------------------------

def build_initial_fields(
    combos: np.ndarray,
    tau_full: np.ndarray,
    centers: Sequence[float],
    phase_deg: float = 0.0,
) -> np.ndarray:
    combos = np.asarray(combos, dtype=np.float64)
    basis = np.stack(
        [np.exp(-0.5 * (tau_full - float(c)) ** 2) for c in centers],
        axis=0,
    )
    h0 = np.matmul(combos, basis)
    phase = np.exp(1j * np.deg2rad(float(phase_deg)))
    return (h0.astype(np.complex128) * phase).astype(np.complex128, copy=False)


def run_ssfm_batch_selected(
    params: NLSEParams,
    combos: np.ndarray,
    selected_steps: np.ndarray,
    time_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Return selected complex SSFM fields, shape [B, Z, 2, Tmask], float32."""
    combos = np.asarray(combos, dtype=np.float64)
    batch = int(combos.shape[0])
    tau_full = np.linspace(
        -float(params.t_window_t0),
        float(params.t_window_t0),
        int(params.n_t),
        endpoint=False,
        dtype=np.float64,
    )
    dtau = float(tau_full[1] - tau_full[0])
    omega = 2.0 * np.pi * np.fft.fftfreq(int(params.n_t), d=dtau)
    zeta = np.linspace(0.0, float(params.z_max_ld), int(params.n_z) + 1)
    dz = float(zeta[1] - zeta[0])

    centers = params.default_centers_t0(combos.shape[1], params.pulse_spacing_t0)
    h0 = build_initial_fields(combos, tau_full, centers, params.initial_phase_deg)
    h = torch.as_tensor(h0, dtype=torch.complex128, device=device)

    omega_t = torch.as_tensor(omega, dtype=torch.float64, device=device)
    i_omega = 1j * omega_t
    linear = -float(params.alpha_norm) / 2.0 + 1j * float(params.beta2_norm) / 2.0 * omega_t ** 2
    if params.has_tod:
        linear = linear - 1j * float(params.beta3_norm) / 6.0 * omega_t ** 3
    half_prop = torch.exp(linear * (dz / 2.0))

    selected_steps = np.asarray(selected_steps, dtype=np.int64)
    step_to_slot = {int(step): i for i, step in enumerate(selected_steps.tolist())}
    n_slices = int(len(selected_steps))
    n_time = int(np.count_nonzero(time_mask))
    out = np.empty((batch, n_slices, 2, n_time), dtype=np.float32)

    def capture(step: int) -> None:
        slot = step_to_slot[int(step)]
        arr = h.detach().cpu().numpy()[:, time_mask]
        out[:, slot, 0, :] = arr.real.astype(np.float32)
        out[:, slot, 1, :] = arr.imag.astype(np.float32)

    capture(0)
    for step in range(1, int(params.n_z) + 1):
        h = torch.fft.ifft(torch.fft.fft(h, dim=-1) * half_prop, dim=-1)
        intensity = torch.abs(h) ** 2
        h_out = h * torch.exp(1j * float(params.N_sq) * intensity * dz)
        if params.has_ss:
            p = intensity * h
            p_t = torch.fft.ifft(i_omega * torch.fft.fft(p, dim=-1), dim=-1)
            h_out = h_out - float(params.ss_coef) * float(params.N_sq) * float(params.s) * p_t * dz
        if params.has_irs:
            i_t = torch.fft.ifft(i_omega * torch.fft.fft(intensity, dim=-1), dim=-1)
            h_out = h_out - 1j * float(params.N_sq) * float(params.tau_R) * i_t * h * dz
        h = h_out
        h = torch.fft.ifft(torch.fft.fft(h, dim=-1) * half_prop, dim=-1)
        if step in step_to_slot:
            capture(step)

    return out


# -----------------------------------------------------------------------------
# Dataset generation/storage
# -----------------------------------------------------------------------------

def dataset_paths(dataset_dir: Path) -> Dict[str, Path]:
    return {
        "data": dataset_dir / "seen_complex_maps_float32.dat",
        "mask": dataset_dir / "complete_mask.npy",
        "meta": dataset_dir / "metadata.json",
        "seen_copy": dataset_dir / "seen_combinations_exact_copy.csv",
    }


def expected_dataset_meta(
    M: int,
    seen_csv: Path,
    n_seen: int,
    grid: Dict[str, Any],
    propagation_slices: int,
    tau_compare: np.ndarray,
    selected_steps: np.ndarray,
    zeta_nominal: np.ndarray,
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "M": int(M),
        "n_seen": int(n_seen),
        "seen_csv_sha256": sha256_file(seen_csv),
        "storage_dtype": "float32",
        "storage_shape": [int(n_seen), int(propagation_slices), 2, int(len(tau_compare))],
        "axis_order": ["configuration", "z_slice", "real_imag", "time"],
        "eval_half_window_t0": float(grid["eval_half_window_t0"]),
        "eval_n_t": int(grid["eval_n_t"]),
        "eval_n_z": int(grid["eval_n_z"]),
        "compare_t_min": float(grid["compare_t_min"]),
        "compare_t_max": float(grid["compare_t_max"]),
        "z_max_ld": float(grid.get("z_max_ld", 4.0)),
        "propagation_slices": int(propagation_slices),
        "selected_ssfm_steps": [int(x) for x in selected_steps.tolist()],
        "zeta_nominal": [float(x) for x in zeta_nominal.tolist()],
        "tau_compare": [float(x) for x in tau_compare.tolist()],
        "waterfall_zeta": [0.5 * i for i in range(9)],
        "contains_intermediate_propagation": True,
        "contains_only_seen_configurations": True,
    }


def verify_meta(existing: Dict[str, Any], expected: Dict[str, Any]) -> None:
    keys = [
        "M", "n_seen", "seen_csv_sha256", "storage_shape", "eval_n_t", "eval_n_z",
        "compare_t_min", "compare_t_max", "propagation_slices", "selected_ssfm_steps",
    ]
    mismatches = []
    for key in keys:
        if existing.get(key) != expected.get(key):
            mismatches.append("%s: existing=%r expected=%r" % (key, existing.get(key), expected.get(key)))
    if mismatches:
        raise RuntimeError(
            "Existing CNN dataset is incompatible with the requested settings. "
            "Use --force-data to rebuild it.\n" + "\n".join(mismatches)
        )


def recompute_field_scale(memmap: np.memmap, complete: np.ndarray, chunk: int = 8) -> float:
    maximum = 0.0
    indices = np.flatnonzero(complete)
    for start in range(0, len(indices), int(chunk)):
        idx = indices[start:start + int(chunk)]
        arr = np.asarray(memmap[idx], dtype=np.float32)
        maximum = max(maximum, float(np.max(np.abs(arr))))
    return max(maximum, 1e-8)


def generate_seen_dataset(
    run_dir: Path,
    cnn_root: Path,
    M: int,
    grid: Dict[str, Any],
    propagation_slices: int,
    device: torch.device,
    ssfm_batch_size: int,
    force_data: bool,
    max_data_samples: int,
) -> Dict[str, Any]:
    seen_csv = run_dir / "dataset" / "seen_combinations.csv"
    if not seen_csv.exists():
        raise FileNotFoundError("Missing seen CSV: %s" % seen_csv)
    combos = np.asarray(load_combinations_csv(seen_csv), dtype=np.float64)
    if combos.ndim != 2 or combos.shape[1] != int(M):
        raise ValueError("Unexpected seen-combination shape: %r" % (combos.shape,))
    if int(max_data_samples) > 0:
        combos = combos[:int(max_data_samples)]

    dataset_dir = cnn_root / "dataset"
    paths = dataset_paths(dataset_dir)
    if force_data and dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tau_full, time_mask, tau_compare = make_tau_and_mask(grid)
    selected_steps, zeta_nominal = make_z_sampling(
        int(grid["eval_n_z"]), float(grid.get("z_max_ld", 4.0)), int(propagation_slices)
    )
    expected = expected_dataset_meta(
        M, seen_csv, len(combos), grid, propagation_slices, tau_compare, selected_steps, zeta_nominal
    )

    if paths["meta"].exists():
        existing = read_json(paths["meta"])
        verify_meta(existing, expected)
    else:
        write_json(paths["meta"], expected)

    if not paths["seen_copy"].exists():
        shutil.copy2(str(seen_csv), str(paths["seen_copy"]))

    shape = tuple(int(x) for x in expected["storage_shape"])
    mode = "r+" if paths["data"].exists() else "w+"
    mmap = np.memmap(str(paths["data"]), dtype=np.float32, mode=mode, shape=shape)
    if paths["mask"].exists():
        complete = np.load(str(paths["mask"])).astype(bool)
        if complete.shape != (len(combos),):
            raise RuntimeError("complete_mask.npy has an incompatible shape.")
    else:
        complete = np.zeros(len(combos), dtype=bool)
        np.save(str(paths["mask"]), complete)

    params = NLSEParams.paper_pam4(
        z_max_ld=float(grid.get("z_max_ld", 4.0)),
        t_window_t0=float(grid["eval_half_window_t0"]),
        n_t=int(grid["eval_n_t"]),
        n_z=int(grid["eval_n_z"]),
    )

    remaining = np.flatnonzero(~complete)
    print("[dataset] M=%d seen=%d, remaining=%d, shape=%s" % (M, len(combos), len(remaining), shape))
    begin = time.perf_counter()
    for pos in range(0, len(remaining), int(ssfm_batch_size)):
        idx = remaining[pos:pos + int(ssfm_batch_size)]
        maps = run_ssfm_batch_selected(
            params=params,
            combos=combos[idx],
            selected_steps=selected_steps,
            time_mask=time_mask,
            device=device,
        )
        mmap[idx] = maps
        mmap.flush()
        complete[idx] = True
        np.save(str(paths["mask"]), complete)
        done = int(np.count_nonzero(complete))
        elapsed = time.perf_counter() - begin
        print("[dataset] %d/%d complete, elapsed %.1f s" % (done, len(combos), elapsed), flush=True)
        del maps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scale = recompute_field_scale(mmap, complete)
    meta = read_json(paths["meta"])
    meta["field_scale_max_abs_seen"] = float(scale)
    meta["generation_complete"] = bool(np.all(complete))
    meta["completed_count"] = int(np.count_nonzero(complete))
    meta["data_file_bytes"] = int(paths["data"].stat().st_size)
    write_json(paths["meta"], meta)
    del mmap
    return meta


class ComplexMapMemmapDataset(Dataset):
    def __init__(self, data_path: Path, shape: Sequence[int], indices: Sequence[int], scale: float):
        self.data_path = str(data_path)
        self.shape = tuple(int(x) for x in shape)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scale = float(scale)
        self._mmap = None

    def _array(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self.data_path, dtype=np.float32, mode="r", shape=self.shape)
        return self._mmap

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[int(item)])
        arr = np.asarray(self._array()[idx], dtype=np.float32)
        x = np.array(arr[0], dtype=np.float32, copy=True) / self.scale  # [2,T]
        y = np.array(arr.reshape(-1, arr.shape[-1]), dtype=np.float32, copy=True) / self.scale
        return torch.from_numpy(x), torch.from_numpy(y)


# -----------------------------------------------------------------------------
# CNN model
# -----------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    """One residual 1-D convolution block used by the data-driven CNN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = int(dilation) * (int(kernel_size) - 1) // 2
        self.conv = nn.Conv1d(
            int(in_channels), int(out_channels), kernel_size=int(kernel_size),
            padding=padding, dilation=int(dilation), bias=False,
        )
        self.bn = nn.BatchNorm1d(int(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Identity()
            if int(in_channels) == int(out_channels)
            else nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.bn(self.conv(x))
        out = self.relu(out + residual)
        return out


class CoarseGlobalContext1D(nn.Module):
    """Inject coarse, global temporal context without leaving the CNN family.

    The original four dilated convolution blocks are local operators.  This
    branch pools the feature sequence to a short coarse grid, processes it by
    convolution, and linearly upsamples it back to the original time length.
    It allows distant sub-pulses to influence one another while keeping the
    network lightweight and fully convolutional.
    """

    def __init__(self, channels: int, bins: int = 32):
        super().__init__()
        self.bins = int(max(4, bins))
        self.net = nn.Sequential(
            nn.Conv1d(int(channels), int(channels), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(int(channels)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool1d(x, self.bins)
        context = self.net(pooled)
        context = F.interpolate(context, size=x.shape[-1], mode="linear", align_corners=False)
        return x + context


class FullPropagationCNN(nn.Module):
    """Four residual convolution blocks plus global context and two projections.

    The network predicts the propagation *change* relative to the known input
    field.  The z=0 output is copied exactly from the input, so an untrained or
    poorly trained network can no longer produce an incorrect initial slice.
    No NLSE residual or physics loss is used; all learned propagation changes
    remain purely SSFM-supervised.
    """

    def __init__(
        self,
        n_slices: int,
        hidden: int = 32,
        kernel_size: int = 11,
        dilations: Sequence[int] = (1, 4, 16, 64),
        context_bins: int = 32,
    ):
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("Exactly four convolutional blocks are required.")
        blocks: List[nn.Module] = []
        in_ch = 2
        for dilation in dilations:
            blocks.append(
                ResidualConvBlock(in_ch, int(hidden), int(kernel_size), int(dilation))
            )
            in_ch = int(hidden)
        self.features = nn.Sequential(*blocks)
        self.context = CoarseGlobalContext1D(int(hidden), int(context_bins))
        self.linear1 = nn.Conv1d(int(hidden), int(hidden), kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Conv1d(int(hidden), 2 * int(n_slices), kernel_size=1)
        # Start from a stable identity-through-propagation baseline instead of
        # a random full-field map. Training then learns only the SSFM change.
        nn.init.zeros_(self.linear2.weight)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)
        self.n_slices = int(n_slices)

        # The first propagation plane is z=0.  Its learned residual is forced to
        # zero and the exact input field is copied to the output.
        residual_mask = torch.ones(1, self.n_slices, 1, 1, dtype=torch.float32)
        residual_mask[:, 0, :, :] = 0.0
        self.register_buffer("residual_mask", residual_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.context(h)
        h = self.relu(self.linear1(h))
        delta = self.linear2(h)
        batch, _, n_time = delta.shape
        delta = delta.view(batch, self.n_slices, 2, n_time)
        base = x.unsqueeze(1).expand(-1, self.n_slices, -1, -1)
        pred = base + delta * self.residual_mask.to(dtype=delta.dtype)
        return pred.reshape(batch, 2 * self.n_slices, n_time)

def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def save_checkpoint(
    path: Path,
    model: FullPropagationCNN,
    model_config: Dict[str, Any],
    train_config: Dict[str, Any],
    scale: float,
    best_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "train_config": train_config,
            "field_scale": float(scale),
            "best_epoch": int(best_epoch),
        },
        str(path),
    )


def load_cnn_checkpoint(path: Path, device: torch.device) -> Tuple[FullPropagationCNN, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    cfg = dict(payload["model_config"])
    model = FullPropagationCNN(
        n_slices=int(cfg["n_slices"]),
        hidden=int(cfg["hidden"]),
        kernel_size=int(cfg["kernel_size"]),
        dilations=tuple(int(x) for x in cfg.get("dilations", [1, 4, 16, 64])),
        context_bins=int(cfg.get("context_bins", 32)),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def make_split(n_seen: int, val_fraction: float, split_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_seen < 3:
        raise ValueError("At least three seen configurations are required for train/validation selection.")
    n_val = max(1, int(round(float(val_fraction) * n_seen)))
    n_val = min(n_val, n_seen - 2)
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n_seen)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def save_loss_plot(history: pd.DataFrame, path: Path, log_scale: bool, has_val: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(history["epoch"], history["train_loss"], label="Training loss", linewidth=1.8)
    if has_val and "val_loss" in history.columns:
        ax.plot(history["epoch"], history["val_loss"], label="Validation loss", linewidth=1.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Composite supervised loss")
    ax.set_title("CNN Training History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=220, bbox_inches="tight")
    plt.close(fig)


def supervised_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    n_slices: int,
    field_weight: float,
    power_weight: float,
    initial_weight: float,
    terminal_weight: float,
    peak_weight: float,
) -> Dict[str, torch.Tensor]:
    """Relative, signal-aware supervised loss for complex propagation maps.

    A plain elementwise MSE is dominated by the near-zero background.  This
    loss normalizes each sample by its own energy, adds a power-domain term,
    and reports initial/terminal slice terms separately.  The z=0 term should
    be numerically zero because the model copies the input exactly.
    """
    batch, _, n_time = prediction.shape
    pred = prediction.view(batch, int(n_slices), 2, n_time)
    ref = target.view(batch, int(n_slices), 2, n_time)
    eps = torch.finfo(pred.dtype).eps

    field_num = torch.sum((pred - ref) ** 2, dim=(1, 2, 3))
    field_den = torch.sum(ref ** 2, dim=(1, 2, 3)).clamp_min(eps)
    field_rel = torch.mean(field_num / field_den)

    pred_power = torch.sum(pred ** 2, dim=2)
    ref_power = torch.sum(ref ** 2, dim=2)
    peak = torch.amax(ref_power, dim=(1, 2), keepdim=True).clamp_min(eps)
    signal_weight = 1.0 + float(peak_weight) * ref_power / peak
    power_num = torch.sum(signal_weight * (pred_power - ref_power) ** 2, dim=(1, 2))
    power_den = torch.sum(signal_weight * ref_power ** 2, dim=(1, 2)).clamp_min(eps)
    power_rel = torch.mean(power_num / power_den)

    initial_num = torch.sum((pred[:, 0] - ref[:, 0]) ** 2, dim=(1, 2))
    initial_den = torch.sum(ref[:, 0] ** 2, dim=(1, 2)).clamp_min(eps)
    initial_rel = torch.mean(initial_num / initial_den)

    terminal_num = torch.sum((pred[:, -1] - ref[:, -1]) ** 2, dim=(1, 2))
    terminal_den = torch.sum(ref[:, -1] ** 2, dim=(1, 2)).clamp_min(eps)
    terminal_rel = torch.mean(terminal_num / terminal_den)

    total = (
        float(field_weight) * field_rel
        + float(power_weight) * power_rel
        + float(initial_weight) * initial_rel
        + float(terminal_weight) * terminal_rel
    )
    return {
        "loss": total,
        "field_rel_sq": field_rel,
        "power_rel_sq": power_rel,
        "initial_rel_sq": initial_rel,
        "terminal_rel_sq": terminal_rel,
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp_enabled: bool,
    n_slices: int,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: Dict[str, float] = {
        "loss": 0.0,
        "field_rel_sq": 0.0,
        "power_rel_sq": 0.0,
        "initial_rel_sq": 0.0,
        "terminal_rel_sq": 0.0,
    }
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(amp_enabled and device.type == "cuda")):
                pred = model(x)
                parts = supervised_loss_components(
                    pred,
                    y,
                    n_slices=int(n_slices),
                    field_weight=float(loss_weights["field"]),
                    power_weight=float(loss_weights["power"]),
                    initial_weight=float(loss_weights["initial"]),
                    terminal_weight=float(loss_weights["terminal"]),
                    peak_weight=float(loss_weights["peak"]),
                )
                loss = parts["loss"]
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            batch = int(x.shape[0])
            for key in totals:
                totals[key] += float(parts[key].detach().cpu()) * batch
            count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}

def compute_training_fit_diagnostics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_slices: int,
    amp_enabled: bool,
) -> Dict[str, Any]:
    model.eval()
    full_power_errors: List[float] = []
    terminal_power_errors: List[float] = []
    initial_field_errors: List[float] = []
    pred_sq_sum = 0.0
    ref_sq_sum = 0.0
    pred_peak = 0.0
    ref_peak = 0.0
    n_values = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(amp_enabled and device.type == "cuda")):
                pred_flat = model(x)
            b, _, t = pred_flat.shape
            pred = pred_flat.float().view(b, int(n_slices), 2, t)
            ref = y.float().view(b, int(n_slices), 2, t)
            pred_power = torch.sum(pred ** 2, dim=2)
            ref_power = torch.sum(ref ** 2, dim=2)
            eps = 1e-12
            full = torch.sqrt(
                torch.sum((pred_power - ref_power) ** 2, dim=(1, 2))
                / torch.sum(ref_power ** 2, dim=(1, 2)).clamp_min(eps)
            )
            terminal = torch.sqrt(
                torch.sum((pred_power[:, -1] - ref_power[:, -1]) ** 2, dim=1)
                / torch.sum(ref_power[:, -1] ** 2, dim=1).clamp_min(eps)
            )
            initial = torch.sqrt(
                torch.sum((pred[:, 0] - ref[:, 0]) ** 2, dim=(1, 2))
                / torch.sum(ref[:, 0] ** 2, dim=(1, 2)).clamp_min(eps)
            )
            full_power_errors.extend(full.detach().cpu().numpy().astype(float).tolist())
            terminal_power_errors.extend(terminal.detach().cpu().numpy().astype(float).tolist())
            initial_field_errors.extend(initial.detach().cpu().numpy().astype(float).tolist())
            pred_sq_sum += float(torch.sum(pred ** 2).detach().cpu())
            ref_sq_sum += float(torch.sum(ref ** 2).detach().cpu())
            pred_peak = max(pred_peak, float(torch.max(torch.abs(pred)).detach().cpu()))
            ref_peak = max(ref_peak, float(torch.max(torch.abs(ref)).detach().cpu()))
            n_values += int(pred.numel())

    rms_ratio = math.sqrt(pred_sq_sum / max(n_values, 1)) / (
        math.sqrt(ref_sq_sum / max(n_values, 1)) + 1e-12
    )
    peak_ratio = pred_peak / (ref_peak + 1e-12)
    mean_full = float(np.mean(full_power_errors)) if full_power_errors else float("nan")
    mean_terminal = float(np.mean(terminal_power_errors)) if terminal_power_errors else float("nan")
    max_initial = float(np.max(initial_field_errors)) if initial_field_errors else float("nan")

    collapsed = bool(
        (np.isfinite(mean_full) and mean_full > 0.95 and rms_ratio < 0.30)
        or peak_ratio < 0.15
        or not np.isfinite(mean_full)
    )
    quality_gate_failed = bool(
        collapsed
        or (np.isfinite(mean_full) and mean_full > 1.50)
        or rms_ratio < 0.20
        or rms_ratio > 3.00
        or peak_ratio < 0.20
        or peak_ratio > 5.00
        or (np.isfinite(max_initial) and max_initial > 1e-5)
    )
    return {
        "n_seen_cases": int(len(full_power_errors)),
        "seen_full_rel_l2_power_mean": mean_full,
        "seen_terminal_rel_l2_power_mean": mean_terminal,
        "seen_initial_rel_l2_field_max": max_initial,
        "prediction_to_reference_rms_ratio": float(rms_ratio),
        "prediction_to_reference_peak_ratio": float(peak_ratio),
        "collapsed": collapsed,
        "quality_gate_failed": quality_gate_failed,
        "collapse_rule": "near-zero/nonfinite output diagnostic",
        "quality_gate_rule": "seen full power rel-L2 <= 1.50, RMS ratio in [0.20,3.00], peak ratio in [0.20,5.00], exact z=0",
    }


def train_one_seed(
    cnn_root: Path,
    seed: int,
    dataset_meta: Dict[str, Any],
    device: torch.device,
    hidden: int,
    kernel_size: int,
    dilations: Sequence[int],
    context_bins: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    val_fraction: float,
    split_seed: int,
    amp_enabled: bool,
    force_train: bool,
    field_loss_weight: float,
    power_loss_weight: float,
    initial_loss_weight: float,
    terminal_loss_weight: float,
    peak_weight: float,
    allow_collapsed_model: bool,
) -> Path:
    train_root = cnn_root / "train" / ("seed_%d" % int(seed))
    final_ckpt = train_root / "final" / "final_cnn.pt"
    if final_ckpt.exists() and not force_train:
        diagnostic_path = train_root / "final" / "training_fit_diagnostics.json"
        if diagnostic_path.exists() and read_json(diagnostic_path).get("quality_gate_failed", read_json(diagnostic_path).get("collapsed", False)):
            if not allow_collapsed_model:
                raise RuntimeError(
                    "Existing final CNN is marked as collapsed. Use --force-train to retrain "
                    "or --allow-collapsed-model only for debugging: %s" % final_ckpt
                )
        print("[train] seed=%d already complete: %s" % (seed, final_ckpt))
        return final_ckpt

    shape = tuple(int(x) for x in dataset_meta["storage_shape"])
    n_seen = int(shape[0])
    n_slices = int(shape[1])
    scale = float(dataset_meta["field_scale_max_abs_seen"])
    data_path = cnn_root / "dataset" / "seen_complex_maps_float32.dat"
    train_idx, val_idx = make_split(n_seen, val_fraction, split_seed)
    split_info = {
        "n_seen": n_seen,
        "train_indices": [int(x) for x in train_idx.tolist()],
        "validation_indices": [int(x) for x in val_idx.tolist()],
        "validation_is_internal_to_seen_set": True,
        "unseen_test_set_used_for_model_selection": False,
        "split_seed": int(split_seed),
    }
    write_json(train_root / "model_selection" / "split_indices.json", split_info)

    model_cfg = {
        "n_slices": n_slices,
        "hidden": int(hidden),
        "kernel_size": int(kernel_size),
        "dilations": [int(x) for x in dilations],
        "context_bins": int(context_bins),
        "input_channels": 2,
        "output_channels": 2 * n_slices,
        "task": "initial_complex_field_to_full_complex_propagation",
        "prediction_form": "input_field_plus_learned_propagation_residual",
        "z0_output_is_exact_input_copy": True,
    }
    loss_weights = {
        "field": float(field_loss_weight),
        "power": float(power_loss_weight),
        "initial": float(initial_loss_weight),
        "terminal": float(terminal_loss_weight),
        "peak": float(peak_weight),
    }
    train_cfg = {
        "seed": int(seed),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "val_fraction": float(val_fraction),
        "split_seed": int(split_seed),
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "gradient_clip_norm": 5.0,
        "amp": bool(amp_enabled),
        "loss_weights": loss_weights,
    }

    # Stage A: select epoch count using only the seen-set internal split.
    set_seed(seed)
    model = FullPropagationCNN(
        n_slices, hidden, kernel_size, tuple(dilations), context_bins=context_bins
    ).to(device)
    print("[train] seed=%d model parameters=%d" % (seed, count_parameters(model)))
    print("[train] dilations=%s, context_bins=%d, loss_weights=%s" % (
        list(dilations), int(context_bins), loss_weights
    ))
    train_ds = ComplexMapMemmapDataset(data_path, shape, train_idx, scale)
    val_ds = ComplexMapMemmapDataset(data_path, shape, val_idx, scale)
    train_loader = DataLoader(
        train_ds, batch_size=min(int(batch_size), len(train_ds)), shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(int(batch_size), len(val_ds)), shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, patience // 4), min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and device.type == "cuda"))

    selection_dir = train_root / "model_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    best_path = selection_dir / "best_selection_cnn.pt"
    history_rows: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 1
    stale = 0
    for epoch in range(1, int(max_epochs) + 1):
        t0 = time.perf_counter()
        train_stats = run_epoch(
            model, train_loader, device, optimizer, scaler, amp_enabled,
            n_slices=n_slices, loss_weights=loss_weights,
        )
        val_stats = run_epoch(
            model, val_loader, device, None, None, amp_enabled,
            n_slices=n_slices, loss_weights=loss_weights,
        )
        train_loss = float(train_stats["loss"])
        val_loss = float(val_stats["loss"])
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        improved = val_loss < best_val - max(
            1e-10, 1e-5 * abs(best_val if np.isfinite(best_val) else 1.0)
        )
        if improved:
            best_val = val_loss
            best_epoch = int(epoch)
            stale = 0
            save_checkpoint(best_path, model, model_cfg, train_cfg, scale, best_epoch)
        else:
            stale += 1
        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "learning_rate": lr_now,
            "epoch_sec": time.perf_counter() - t0,
        }
        for key, value in train_stats.items():
            if key != "loss":
                row["train_" + key] = float(value)
        for key, value in val_stats.items():
            if key != "loss":
                row["val_" + key] = float(value)
        history_rows.append(row)
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                "[selection] seed=%d epoch=%d train=%.4e val=%.4e "
                "val_field=%.3e val_power=%.3e best_epoch=%d lr=%.2e"
                % (
                    seed, epoch, train_loss, val_loss,
                    val_stats["field_rel_sq"], val_stats["power_rel_sq"],
                    best_epoch, lr_now,
                ),
                flush=True,
            )
        if epoch >= int(min_epochs) and stale >= int(patience):
            print("[selection] early stop at epoch %d; best epoch=%d" % (epoch, best_epoch))
            break

    hist = pd.DataFrame(history_rows)
    hist.to_csv(selection_dir / "selection_history.csv", index=False, encoding="utf-8-sig")
    save_loss_plot(hist, selection_dir / "selection_loss_history_linear.png", False, True)
    save_loss_plot(hist, selection_dir / "selection_loss_history_log.png", True, True)
    write_json(selection_dir / "selection_summary.json", {
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val),
        "epochs_executed": int(len(hist)),
        "loss_weights": loss_weights,
        "note": "The unseen test set was not used to choose the epoch count.",
    })

    # Stage B: reset and train on ALL seen configurations for exactly best_epoch epochs.
    set_seed(seed)
    final_model = FullPropagationCNN(
        n_slices, hidden, kernel_size, tuple(dilations), context_bins=context_bins
    ).to(device)
    all_idx = np.arange(n_seen, dtype=np.int64)
    all_ds = ComplexMapMemmapDataset(data_path, shape, all_idx, scale)
    all_loader = DataLoader(
        all_ds, batch_size=min(int(batch_size), len(all_ds)), shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    diagnostic_loader = DataLoader(
        all_ds, batch_size=min(int(batch_size), len(all_ds)), shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    final_optimizer = torch.optim.Adam(
        final_model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    final_scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and device.type == "cuda"))
    final_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(best_epoch) + 1):
        t0 = time.perf_counter()
        train_stats = run_epoch(
            final_model, all_loader, device, final_optimizer, final_scaler, amp_enabled,
            n_slices=n_slices, loss_weights=loss_weights,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "epoch_sec": time.perf_counter() - t0,
        }
        for key, value in train_stats.items():
            if key != "loss":
                row["train_" + key] = float(value)
        final_rows.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(best_epoch):
            print(
                "[final] seed=%d epoch=%d/%d loss=%.4e field=%.3e power=%.3e"
                % (
                    seed, epoch, best_epoch, train_stats["loss"],
                    train_stats["field_rel_sq"], train_stats["power_rel_sq"],
                ),
                flush=True,
            )

    final_dir = train_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(final_ckpt, final_model, model_cfg, train_cfg, scale, best_epoch)
    final_hist = pd.DataFrame(final_rows)
    final_hist.to_csv(final_dir / "final_history.csv", index=False, encoding="utf-8-sig")
    save_loss_plot(final_hist, final_dir / "final_loss_history_linear.png", False, False)
    save_loss_plot(final_hist, final_dir / "final_loss_history_log.png", True, False)
    diagnostics = compute_training_fit_diagnostics(
        final_model, diagnostic_loader, device, n_slices, amp_enabled
    )
    write_json(final_dir / "training_fit_diagnostics.json", diagnostics)
    write_json(final_dir / "train_config.json", {
        "model_config": model_cfg,
        "train_config": train_cfg,
        "selected_epoch_count": int(best_epoch),
        "used_all_seen_configurations": True,
        "n_seen": int(n_seen),
        "field_scale": float(scale),
        "parameter_count": count_parameters(final_model),
        "training_fit_diagnostics": diagnostics,
    })
    print("[diagnostic] seed=%d %s" % (seed, diagnostics))
    if diagnostics.get("quality_gate_failed", diagnostics.get("collapsed", False)) and not allow_collapsed_model:
        raise RuntimeError(
            "CNN training failed the seen-set quality gate; evaluation was stopped. "
            "Inspect %s and retrain with --force-train."
            % (final_dir / "training_fit_diagnostics.json")
        )

    del model, final_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_ckpt


# -----------------------------------------------------------------------------
# Evaluation and aggregation
# -----------------------------------------------------------------------------

def metric_rows_for_batch(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> List[Dict[str, float]]:
    # [B,Z,2,T] -> complex
    ref = reference[:, :, 0, :].astype(np.float64) + 1j * reference[:, :, 1, :].astype(np.float64)
    pred = prediction[:, :, 0, :].astype(np.float64) + 1j * prediction[:, :, 1, :].astype(np.float64)
    rows: List[Dict[str, float]] = []
    for i in range(ref.shape[0]):
        r = ref[i]
        p = pred[i]
        pr = np.abs(r) ** 2
        pp = np.abs(p) ** 2
        full_field = float(np.linalg.norm(p - r) / (np.linalg.norm(r) + 1e-300))
        full_power = float(np.linalg.norm(pp - pr) / (np.linalg.norm(pr) + 1e-300))
        terminal_field = float(np.linalg.norm(p[-1] - r[-1]) / (np.linalg.norm(r[-1]) + 1e-300))
        terminal_power = float(np.linalg.norm(pp[-1] - pr[-1]) / (np.linalg.norm(pr[-1]) + 1e-300))
        rows.append({
            "full_rel_l2_field": full_field,
            "full_rel_l2_power": full_power,
            "full_e1": full_power,
            "full_e2": float(np.max(np.abs(pp - pr))),
            "full_mse_power": float(np.mean((pp - pr) ** 2)),
            "full_mae_power": float(np.mean(np.abs(pp - pr))),
            "terminal_rel_l2_field": terminal_field,
            "terminal_rel_l2_power": terminal_power,
            "terminal_e1": terminal_power,
            "terminal_e2": float(np.max(np.abs(pp[-1] - pr[-1]))),
            "terminal_mse_power": float(np.mean((pp[-1] - pr[-1]) ** 2)),
            "terminal_mae_power": float(np.mean(np.abs(pp[-1] - pr[-1]))),
        })
    return rows


def append_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def completed_indices(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["idx"])
        return set(int(x) for x in df["idx"].tolist())
    except Exception:
        return set()


def predict_complex_maps(
    model: FullPropagationCNN,
    payload: Dict[str, Any],
    reference: np.ndarray,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    scale = float(payload["field_scale"])
    x = torch.from_numpy(reference[:, 0, :, :].astype(np.float32) / scale).to(device)
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=bool(amp_enabled and device.type == "cuda")):
            y = model(x)
    b, _, t = y.shape
    z = int(payload["model_config"]["n_slices"])
    return (y.detach().float().cpu().numpy().reshape(b, z, 2, t) * scale).astype(np.float32)


def summarize_seed_metrics(metrics_csv: Path, out_json: Path) -> Dict[str, Any]:
    df = pd.read_csv(metrics_csv)
    cols = [c for c in df.columns if c.startswith("full_") or c.startswith("terminal_")]
    summary: Dict[str, Any] = {"n_cases": int(len(df)), "metrics": {}}
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        summary["metrics"][col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(vals.min()),
            "p50": float(vals.quantile(0.50)),
            "p90": float(vals.quantile(0.90)),
            "p95": float(vals.quantile(0.95)),
            "max": float(vals.max()),
        }
    write_json(out_json, summary)
    return summary


def evaluate_all_seeds(
    run_dir: Path,
    cnn_root: Path,
    M: int,
    grid: Dict[str, Any],
    propagation_slices: int,
    seeds: Sequence[int],
    device: torch.device,
    ssfm_batch_size: int,
    amp_enabled: bool,
    max_eval_samples: int,
) -> None:
    unseen_csv = run_dir / "dataset" / "unseen_combinations.csv"
    combos = np.asarray(load_combinations_csv(unseen_csv), dtype=np.float64)
    if int(max_eval_samples) > 0:
        combos = combos[:int(max_eval_samples)]
    if combos.ndim != 2 or combos.shape[1] != int(M):
        raise ValueError("Unexpected unseen-combination shape: %r" % (combos.shape,))

    tau_full, time_mask, tau_compare = make_tau_and_mask(grid)
    selected_steps, zeta_nominal = make_z_sampling(
        int(grid["eval_n_z"]), float(grid.get("z_max_ld", 4.0)), int(propagation_slices)
    )
    params = NLSEParams.paper_pam4(
        z_max_ld=float(grid.get("z_max_ld", 4.0)),
        t_window_t0=float(grid["eval_half_window_t0"]),
        n_t=int(grid["eval_n_t"]),
        n_z=int(grid["eval_n_z"]),
    )

    models: Dict[int, FullPropagationCNN] = {}
    payloads: Dict[int, Dict[str, Any]] = {}
    metric_paths: Dict[int, Path] = {}
    completed: Dict[int, set] = {}
    for seed in seeds:
        ckpt = cnn_root / "train" / ("seed_%d" % int(seed)) / "final" / "final_cnn.pt"
        if not ckpt.exists():
            raise FileNotFoundError("Missing final CNN checkpoint: %s" % ckpt)
        diagnostic_path = ckpt.parent / "training_fit_diagnostics.json"
        if diagnostic_path.exists():
            diagnostic = read_json(diagnostic_path)
            if diagnostic.get("quality_gate_failed", diagnostic.get("collapsed", False)):
                raise RuntimeError(
                    "Refusing to evaluate a collapsed CNN checkpoint: %s. "
                    "Retrain it with the V2 settings." % ckpt
                )
        model, payload = load_cnn_checkpoint(ckpt, device)
        if int(payload["model_config"]["n_slices"]) != int(propagation_slices):
            raise RuntimeError("Checkpoint propagation-slice count does not match evaluation settings.")
        models[int(seed)] = model
        payloads[int(seed)] = payload
        metric_path = cnn_root / "eval" / ("seed_%d" % int(seed)) / "metrics_stream.csv"
        metric_paths[int(seed)] = metric_path
        completed[int(seed)] = completed_indices(metric_path)

    all_indices = np.arange(len(combos), dtype=np.int64)
    pending = [idx for idx in all_indices.tolist() if any(idx not in completed[s] for s in seeds)]
    print("[eval] M=%d unseen=%d, pending=%d, seeds=%s" % (M, len(combos), len(pending), list(seeds)))
    begin = time.perf_counter()
    for pos in range(0, len(pending), int(ssfm_batch_size)):
        idx = np.asarray(pending[pos:pos + int(ssfm_batch_size)], dtype=np.int64)
        t0 = time.perf_counter()
        reference = run_ssfm_batch_selected(params, combos[idx], selected_steps, time_mask, device)
        ssfm_sec_each = (time.perf_counter() - t0) / max(len(idx), 1)
        for seed in seeds:
            needed_mask = np.asarray([int(i) not in completed[int(seed)] for i in idx], dtype=bool)
            if not np.any(needed_mask):
                continue
            t1 = time.perf_counter()
            prediction_all = predict_complex_maps(models[int(seed)], payloads[int(seed)], reference, device, amp_enabled)
            cnn_sec_each = (time.perf_counter() - t1) / max(len(idx), 1)
            metrics = metric_rows_for_batch(reference[needed_mask], prediction_all[needed_mask])
            rows: List[Dict[str, Any]] = []
            needed_indices = idx[needed_mask]
            for local, (case_idx, metric) in enumerate(zip(needed_indices.tolist(), metrics)):
                row: Dict[str, Any] = {
                    "idx": int(case_idx),
                    "M": int(M),
                    "seed": int(seed),
                    "levels": combo_text(combos[int(case_idx)]),
                    "ssfm_sec": float(ssfm_sec_each),
                    "cnn_sec": float(cnn_sec_each),
                }
                row.update(metric)
                rows.append(row)
                completed[int(seed)].add(int(case_idx))
            append_csv_rows(metric_paths[int(seed)], rows)
            del prediction_all
        done = pos + len(idx)
        print("[eval] processed %d/%d pending batches; elapsed %.1f s" % (
            min(done, len(pending)), len(pending), time.perf_counter() - begin
        ), flush=True)
        del reference
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for seed in seeds:
        summarize_seed_metrics(
            metric_paths[int(seed)],
            cnn_root / "eval" / ("seed_%d" % int(seed)) / "summary.json",
        )
    aggregate_evaluation(cnn_root, seeds)


def aggregate_evaluation(cnn_root: Path, seeds: Sequence[int]) -> Dict[str, Any]:
    frames = []
    seed_summary_rows = []
    for seed in seeds:
        path = cnn_root / "eval" / ("seed_%d" % int(seed)) / "metrics_stream.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df)
        seed_summary_rows.append({
            "seed": int(seed),
            "n_cases": int(len(df)),
            "full_rel_l2_power_mean": float(df["full_rel_l2_power"].mean()),
            "full_rel_l2_power_std": float(df["full_rel_l2_power"].std(ddof=0)),
            "terminal_rel_l2_power_mean": float(df["terminal_rel_l2_power"].mean()),
            "terminal_rel_l2_power_std": float(df["terminal_rel_l2_power"].std(ddof=0)),
        })
    if not frames:
        raise RuntimeError("No seed evaluation metrics are available for aggregation.")
    all_df = pd.concat(frames, ignore_index=True)
    aggregate_dir = cnn_root / "eval" / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(aggregate_dir / "all_seed_case_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seed_summary_rows).to_csv(aggregate_dir / "summary_by_seed.csv", index=False, encoding="utf-8-sig")

    metric_cols = [
        "full_rel_l2_field", "full_rel_l2_power", "full_e2", "full_mse_power", "full_mae_power",
        "terminal_rel_l2_field", "terminal_rel_l2_power", "terminal_e2", "terminal_mse_power", "terminal_mae_power",
    ]
    agg_spec: Dict[str, List[str]] = {col: ["mean", "std", "min", "max"] for col in metric_cols}
    case_df = all_df.groupby(["idx", "M", "levels"], as_index=False).agg(agg_spec)
    case_df.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in case_df.columns
    ]
    case_df = case_df.rename(columns={"idx_": "idx", "M_": "M", "levels_": "levels"})
    case_df.to_csv(aggregate_dir / "metrics_by_case_across_seeds.csv", index=False, encoding="utf-8-sig")

    selector = "full_rel_l2_power_mean"
    ordered = case_df.sort_values(selector).reset_index(drop=True)
    median_value = float(ordered[selector].median())
    best_row = ordered.iloc[0]
    worst_row = ordered.iloc[-1]
    middle_row = ordered.iloc[(ordered[selector] - median_value).abs().argmin()]

    seed_table = pd.DataFrame(seed_summary_rows)
    median_seed_mean = float(seed_table["full_rel_l2_power_mean"].median())
    representative_seed = int(seed_table.iloc[(seed_table["full_rel_l2_power_mean"] - median_seed_mean).abs().argmin()]["seed"])

    selected = {
        "selection_metric": selector,
        "selection_across_seeds": True,
        "representative_seed_for_visualization": representative_seed,
        "cases": {
            "best": {
                "idx": int(best_row["idx"]), "levels": str(best_row["levels"]),
                "metric_mean": float(best_row[selector]),
            },
            "middle": {
                "idx": int(middle_row["idx"]), "levels": str(middle_row["levels"]),
                "metric_mean": float(middle_row[selector]),
            },
            "worst": {
                "idx": int(worst_row["idx"]), "levels": str(worst_row["levels"]),
                "metric_mean": float(worst_row[selector]),
            },
        },
    }
    write_json(aggregate_dir / "selected_best_middle_worst.json", selected)

    summary: Dict[str, Any] = {
        "seed_count": int(len(seed_summary_rows)),
        "seeds": [int(x) for x in seeds],
        "n_unique_cases": int(len(case_df)),
        "selected_cases": selected,
        "metrics": {},
    }
    for col in metric_cols:
        vals = pd.to_numeric(all_df[col], errors="coerce").dropna()
        summary["metrics"][col] = {
            "mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
            "min": float(vals.min()), "p50": float(vals.quantile(0.50)),
            "p90": float(vals.quantile(0.90)), "p95": float(vals.quantile(0.95)),
            "max": float(vals.max()),
        }
    write_json(aggregate_dir / "summary.json", summary)
    return summary


def find_pinn_eval_dir(run_dir: Path) -> Optional[Path]:
    root = run_dir / "eval"
    if not root.exists():
        return None
    candidates = []
    for d in root.rglob("*"):
        if d.is_dir() and (d / "summary_by_model.json").exists():
            candidates.append(d)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def extract_fourier_pinn_summary(run_dir: Path) -> Dict[str, float]:
    eval_dir = find_pinn_eval_dir(run_dir)
    if eval_dir is None:
        return {}
    data = read_json(eval_dir / "summary_by_model.json").get("models", {})
    for key, rec in data.items():
        lower = str(key).lower()
        if "fourier" in lower and "no_fourier" not in lower:
            return {
                "pinn_terminal_rel_l2_power_mean": float(rec.get("rel_l2_power_mean", np.nan)),
                "pinn_terminal_rel_l2_power_std": float(rec.get("rel_l2_power_std", np.nan)),
                "pinn_terminal_e2_mean": float(rec.get("e2_mean", np.nan)),
                "pinn_terminal_e2_max": float(rec.get("e2_max", np.nan)),
            }
    return {}


def build_cross_m_summary(runs_root: Path, m_values: Sequence[int], cnn_folder_name: str) -> Path:
    rows = []
    for M in m_values:
        run_dir = find_run_dir(runs_root, int(M))
        summary_path = run_dir / cnn_folder_name / "eval" / "aggregate" / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        metrics = summary["metrics"]
        row: Dict[str, Any] = {
            "M": int(M),
            "run_dir": str(run_dir),
            "cnn_seed_count": int(summary["seed_count"]),
            "n_unseen": int(summary["n_unique_cases"]),
            "cnn_full_rel_l2_power_mean": float(metrics["full_rel_l2_power"]["mean"]),
            "cnn_full_rel_l2_power_std": float(metrics["full_rel_l2_power"]["std"]),
            "cnn_full_e2_mean": float(metrics["full_e2"]["mean"]),
            "cnn_terminal_rel_l2_power_mean": float(metrics["terminal_rel_l2_power"]["mean"]),
            "cnn_terminal_rel_l2_power_std": float(metrics["terminal_rel_l2_power"]["std"]),
            "cnn_terminal_e2_mean": float(metrics["terminal_e2"]["mean"]),
        }
        row.update(extract_fourier_pinn_summary(run_dir))
        rows.append(row)
    out = runs_root / "PINN_vs_CNN_full_propagation_fixedM_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print("[summary] %s" % out)
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fixed-M pure data-driven CNN for full multi-pulse propagation.")
    p.add_argument("--runs-root", default="./MULTIPULSE_AMPLITUDE_RUNS")
    p.add_argument("--M", nargs="+", type=int, required=True)
    p.add_argument("--stage", choices=["data", "train", "eval", "summary", "all"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    p.add_argument("--cnn-folder-name", default="cnn_full_propagation_v2")

    p.add_argument("--propagation-slices", type=int, default=81,
                   help="Complex propagation planes stored/predicted from z=0 to 4LD. 81 includes 0,0.5,...,4 exactly as nominal planes.")
    p.add_argument("--ssfm-batch-size", type=int, default=8)
    p.add_argument("--force-data", action="store_true")
    p.add_argument("--force-train", action="store_true")

    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--kernel-size", type=int, default=11)
    p.add_argument("--dilations", nargs=4, type=int, default=[1, 4, 16, 64],
                   help="Four dilation rates. The default gives a much larger temporal receptive field than V1.")
    p.add_argument("--context-bins", type=int, default=32,
                   help="Number of coarse temporal bins in the global-context CNN branch.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--min-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--field-loss-weight", type=float, default=1.0)
    p.add_argument("--power-loss-weight", type=float, default=1.0)
    p.add_argument("--initial-loss-weight", type=float, default=2.0)
    p.add_argument("--terminal-loss-weight", type=float, default=1.0)
    p.add_argument("--peak-weight", type=float, default=4.0,
                   help="Extra weight assigned to nonzero/high-power regions in the power loss.")
    p.add_argument("--allow-collapsed-model", action="store_true",
                   help="Debug only: allow evaluation even when the seen-set collapse diagnostic fails.")

    # Smoke/debug overrides; leave at zero for the formal experiment.
    p.add_argument("--n-t-override", type=int, default=0)
    p.add_argument("--n-z-override", type=int, default=0)
    p.add_argument("--max-data-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 78)
    print("SCRIPT VERSION:", SCRIPT_VERSION)
    print("TASK: h(0,t) -> full complex h(z,t), improved pure SSFM-supervised CNN")
    print("V2: exact z=0 copy, relative field/power loss, large receptive field, collapse check")
    print("WATERFALL PLANES: z/LD = 0, 0.5, 1.0, ..., 4.0")
    print("=" * 78)

    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.exists():
        raise FileNotFoundError("Runs root does not exist: %s" % runs_root)
    m_values = sorted(set(int(x) for x in args.M))
    seeds = parse_seeds(args.seeds)
    device = safe_device(args.device)
    amp_enabled = not bool(args.no_amp)

    if args.stage == "summary":
        build_cross_m_summary(runs_root, m_values, str(args.cnn_folder_name))
        return

    for M in m_values:
        if M < 2 or M > 8:
            raise ValueError("M must be between 2 and 8.")
        run_dir = find_run_dir(runs_root, M)
        cnn_root = run_dir / str(args.cnn_folder_name)
        cnn_root.mkdir(parents=True, exist_ok=True)
        grid = load_grid(run_dir, args.n_t_override, args.n_z_override)
        print("\n========== M=%d | %s ==========" % (M, run_dir.name))
        print("CNN root:", cnn_root)
        print("SSFM grid: nt=%d nz=%d window=±%gT0 compare=[%g,%g]" % (
            int(grid["eval_n_t"]), int(grid["eval_n_z"]), float(grid["eval_half_window_t0"]),
            float(grid["compare_t_min"]), float(grid["compare_t_max"]),
        ))

        if args.stage in ("data", "all"):
            dataset_meta = generate_seen_dataset(
                run_dir, cnn_root, M, grid, int(args.propagation_slices), device,
                int(args.ssfm_batch_size), bool(args.force_data), int(args.max_data_samples),
            )
        else:
            meta_path = cnn_root / "dataset" / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError("CNN dataset has not been generated: %s" % meta_path)
            dataset_meta = read_json(meta_path)

        if args.stage in ("train", "all"):
            for seed in seeds:
                train_one_seed(
                    cnn_root=cnn_root,
                    seed=seed,
                    dataset_meta=dataset_meta,
                    device=device,
                    hidden=int(args.hidden),
                    kernel_size=int(args.kernel_size),
                    dilations=tuple(int(x) for x in args.dilations),
                    context_bins=int(args.context_bins),
                    batch_size=int(args.batch_size),
                    learning_rate=float(args.learning_rate),
                    weight_decay=float(args.weight_decay),
                    max_epochs=int(args.max_epochs),
                    min_epochs=int(args.min_epochs),
                    patience=int(args.patience),
                    val_fraction=float(args.val_fraction),
                    split_seed=int(args.split_seed),
                    amp_enabled=amp_enabled,
                    force_train=bool(args.force_train),
                    field_loss_weight=float(args.field_loss_weight),
                    power_loss_weight=float(args.power_loss_weight),
                    initial_loss_weight=float(args.initial_loss_weight),
                    terminal_loss_weight=float(args.terminal_loss_weight),
                    peak_weight=float(args.peak_weight),
                    allow_collapsed_model=bool(args.allow_collapsed_model),
                )

        if args.stage in ("eval", "all"):
            evaluate_all_seeds(
                run_dir=run_dir,
                cnn_root=cnn_root,
                M=M,
                grid=grid,
                propagation_slices=int(args.propagation_slices),
                seeds=seeds,
                device=device,
                ssfm_batch_size=int(args.ssfm_batch_size),
                amp_enabled=amp_enabled,
                max_eval_samples=int(args.max_eval_samples),
            )

    if args.stage in ("all", "eval"):
        build_cross_m_summary(runs_root, m_values, str(args.cnn_folder_name))

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
