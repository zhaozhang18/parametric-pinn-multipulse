# -*- coding: utf-8 -*-
"""
run_universal_full81_CNN_DDNN.py

Universal K=1..8 data-driven forward baselines using exactly the same sparse-8
seen/unseen amplitude configurations as an existing universal PINN run.

Experiments
-----------
1) Universal CNN-81
   - Reuses the fixed-M FullPropagationCNN architecture.
   - Input: complex initial waveform h(0,t), 2 channels (real/imaginary).
   - Output: complete complex propagation map on 81 supervised z planes.

2) Universal DDNN-81
   - Reuses exactly the model_config of the universal PINN checkpoint
     (same ConditionalPINN architecture, including Fourier features).
   - Input: (z, t, A1,...,A8).
   - Output: (u,v).
   - Purely supervised by SSFM labels on all 81 z planes; no PDE/physics loss.

Data fairness
-------------
- The script reads dataset/seen_sparse8_combinations.csv and
  dataset/unseen_sparse8_combinations.csv from the universal PINN run.
- No amplitude combination is re-sampled.
- A single shared 81-plane SSFM seen-set dataset is generated once and reused by
  both CNN and DDNN.
- Evaluation uses the same unseen CSV and computes full-propagation metrics over
  all 81 planes by default.

The training protocol uses a stratified seen-set train/validation split only for
model selection, followed by final training on all seen configurations for the
selected epoch/step budget. The unseen set is never used for model selection.
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
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_fixedM_cnn_full_propagation_v2 import (
    ComplexMapMemmapDataset,
    FullPropagationCNN,
    count_parameters,
    run_epoch,
)
from train_multi_pulse_pinn import ConditionalPINN
from train_forward_sparse8_universal_highK import (
    DEFAULT_MAIN_PDE_COUNTS_BY_K,
    DEFAULT_FINETUNE_PDE_COUNTS_BY_K,
)


SCRIPT_VERSION = "universal_full81_cnn_ddnn_v2_adam_lbfgs_20260715"

METRIC_NAMES = [
    "full_rel_l2_field",
    "full_rel_l2_power",
    "terminal_rel_l2_field",
    "terminal_rel_l2_power",
    "max_abs_power_error",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_device(text: str) -> torch.device:
    requested = str(text)
    if requested != "cpu" and not torch.cuda.is_available():
        print(f"[device] CUDA unavailable; falling back from {requested!r} to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def combo_text(row: Sequence[float]) -> str:
    return ";".join(f"{float(x):g}" for x in row)


def save_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def find_universal_run(runs_root: Path, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"Universal PINN run directory does not exist: {p}")
        return p

    candidates: List[Path] = []
    for p in runs_root.rglob("universal_sparse8_K1to8*"):
        if not p.is_dir():
            continue
        if not (p / "dataset" / "seen_sparse8_combinations.csv").is_file():
            continue
        if not (p / "dataset" / "unseen_sparse8_combinations.csv").is_file():
            continue
        if not (p / "sparse8_forward_pinn.pt").is_file():
            continue
        candidates.append(p)
    if not candidates:
        raise FileNotFoundError(
            "No universal_sparse8_K1to8* run containing the PINN checkpoint and exact "
            f"seen/unseen CSV files was found under {runs_root}."
        )

    def score(p: Path) -> Tuple[int, float]:
        name = p.name.lower()
        priority = 0
        if "highk" in name:
            priority += 10
        if "try" in name:
            priority += 2
        return priority, (p / "sparse8_forward_pinn.pt").stat().st_mtime

    chosen = max(candidates, key=score)
    print(f"[discover] universal PINN run -> {chosen}", flush=True)
    return chosen


def resolve_checkpoint(run_dir: Path, explicit: str = "") -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
    else:
        p = run_dir / "sparse8_forward_pinn.pt"
    if not p.is_file():
        raise FileNotFoundError(f"Universal PINN checkpoint not found: {p}")
    return p


def load_sparse8_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    k_values: List[int] = []
    amps: List[List[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        required = ["K"] + [f"A{i}" for i in range(1, 9)]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"Missing columns {missing} in {path}")
        for row in reader:
            k_values.append(int(float(row["K"])))
            amps.append([float(row[f"A{i}"]) for i in range(1, 9)])
    if not amps:
        raise RuntimeError(f"No rows in {path}")
    return np.asarray(k_values, dtype=np.int64), np.asarray(amps, dtype=np.float64)


def load_pinn_checkpoint_metadata(checkpoint: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch.load(str(checkpoint), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint payload: {checkpoint}")
    model_cfg = dict(payload.get("model_config", {}))
    if int(model_cfg.get("n_pulses", -1)) != 8:
        raise RuntimeError(f"Checkpoint is not an 8-slot universal model: {model_cfg}")
    pde_raw = dict(payload.get("pde_params", {}))
    pde = {
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
    return model_cfg, pde


# -----------------------------------------------------------------------------
# Exact shared SSFM labels on all selected propagation planes
# -----------------------------------------------------------------------------

def make_grid(
    model_cfg: Mapping[str, Any],
    ssfm_half_window: float,
    n_t: int,
    n_z: int,
    n_slices: int,
    compare_t_min: Optional[float],
    compare_t_max: Optional[float],
) -> Dict[str, Any]:
    z_max = float(model_cfg.get("z_max_ld", 4.0))
    t_min = float(model_cfg.get("t_min", -44.0)) if compare_t_min is None else float(compare_t_min)
    t_max = float(model_cfg.get("t_max", 44.0)) if compare_t_max is None else float(compare_t_max)
    if not (-float(ssfm_half_window) <= t_min < t_max <= float(ssfm_half_window)):
        raise ValueError("Comparison window must lie inside the SSFM window.")
    if int(n_slices) < 2 or int(n_slices) > int(n_z) + 1:
        raise ValueError("n-slices must lie in [2, n-z+1].")

    tau_full = np.linspace(-float(ssfm_half_window), float(ssfm_half_window), int(n_t), endpoint=False)
    time_mask = (tau_full >= t_min - 1e-12) & (tau_full <= t_max + 1e-12)
    tau = tau_full[time_mask]
    steps = np.rint(np.linspace(0, int(n_z), int(n_slices))).astype(np.int64)
    if len(np.unique(steps)) != len(steps):
        raise ValueError("Selected SSFM steps contain duplicates; increase n-z or reduce n-slices.")
    zeta = np.linspace(0.0, z_max, int(n_slices), dtype=np.float64)
    return {
        "ssfm_half_window": float(ssfm_half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "n_slices": int(n_slices),
        "z_max_ld": z_max,
        "compare_t_min": t_min,
        "compare_t_max": t_max,
        "tau_full": tau_full,
        "time_mask": time_mask,
        "tau": tau,
        "selected_steps": steps,
        "zeta": zeta,
    }


def build_initial_fields(
    amplitudes: np.ndarray,
    tau: np.ndarray,
    centers: Sequence[float],
) -> np.ndarray:
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    basis = np.stack([np.exp(-0.5 * (tau - float(c)) ** 2) for c in centers], axis=0)
    return np.matmul(amplitudes, basis).astype(np.complex128, copy=False)


def run_ssfm_batch_selected(
    amplitudes: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    device: torch.device,
    use_complex64: bool,
) -> np.ndarray:
    """Return [B,Z,2,Tcompare] float32 complex-field labels."""
    amps = np.asarray(amplitudes, dtype=np.float64)
    if amps.ndim != 2 or amps.shape[1] != 8:
        raise ValueError(f"Expected amplitudes [B,8], got {amps.shape}")

    tau_full = np.asarray(grid["tau_full"], dtype=np.float64)
    mask = np.asarray(grid["time_mask"], dtype=bool)
    selected_steps = np.asarray(grid["selected_steps"], dtype=np.int64)
    n_z = int(grid["n_z"])
    z_max = float(grid["z_max_ld"])
    dt = float(tau_full[1] - tau_full[0])
    omega = 2.0 * np.pi * np.fft.fftfreq(len(tau_full), d=dt)
    dz = z_max / float(n_z)
    centers = tuple(-28.0 + 8.0 * i for i in range(8))

    real_dtype = torch.float32 if use_complex64 else torch.float64
    complex_dtype = torch.complex64 if use_complex64 else torch.complex128
    h0 = build_initial_fields(amps, tau_full, centers)
    h = torch.as_tensor(h0, dtype=complex_dtype, device=device)
    omega_t = torch.as_tensor(omega, dtype=real_dtype, device=device)
    i_omega = (1j * omega_t).to(complex_dtype)

    linear = -float(pde["alpha_norm"]) / 2.0 + 1j * float(pde["beta2_norm"]) / 2.0 * omega_t**2
    if bool(pde["has_tod"]):
        linear = linear - 1j * float(pde["beta3_norm"]) / 6.0 * omega_t**3
    half_prop = torch.exp(linear * (dz / 2.0)).to(complex_dtype)

    step_to_slot = {int(s): i for i, s in enumerate(selected_steps.tolist())}
    out = np.empty(
        (len(amps), len(selected_steps), 2, int(np.count_nonzero(mask))),
        dtype=np.float32,
    )

    def capture(step: int) -> None:
        slot = step_to_slot[int(step)]
        arr = h.detach().cpu().numpy()[:, mask]
        out[:, slot, 0, :] = arr.real.astype(np.float32)
        out[:, slot, 1, :] = arr.imag.astype(np.float32)

    capture(0)
    for step in range(1, n_z + 1):
        h = torch.fft.ifft(torch.fft.fft(h, dim=-1) * half_prop, dim=-1)
        intensity = torch.abs(h) ** 2
        h_out = h * torch.exp((1j * float(pde["N_sq"]) * intensity * dz).to(complex_dtype))
        if bool(pde["has_ss"]):
            prod = intensity * h
            prod_t = torch.fft.ifft(i_omega * torch.fft.fft(prod, dim=-1), dim=-1)
            h_out = h_out - float(pde["ss_coef"]) * float(pde["N_sq"]) * float(pde["s"]) * prod_t * dz
        if bool(pde["has_irs"]):
            intensity_c = intensity.to(complex_dtype)
            intensity_t = torch.fft.ifft(i_omega * torch.fft.fft(intensity_c, dim=-1), dim=-1)
            h_out = h_out - 1j * float(pde["N_sq"]) * float(pde["tau_R"]) * intensity_t * h * dz
        h = torch.fft.ifft(torch.fft.fft(h_out, dim=-1) * half_prop, dim=-1)
        if step in step_to_slot:
            capture(step)
    return out


def dataset_paths(root: Path) -> Dict[str, Path]:
    return {
        "data": root / "seen_complex_maps_float32.dat",
        "mask": root / "complete_mask.npy",
        "meta": root / "metadata.json",
        "seen_csv": root / "seen_sparse8_combinations_exact_copy.csv",
        "unseen_csv": root / "unseen_sparse8_combinations_exact_copy.csv",
    }


def expected_dataset_meta(
    seen_csv: Path,
    unseen_csv: Path,
    seen_k: np.ndarray,
    seen_a: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    checkpoint: Path,
) -> Dict[str, Any]:
    k_counts = {str(int(k)): int(np.sum(seen_k == k)) for k in sorted(np.unique(seen_k).tolist())}
    return {
        "script_version": SCRIPT_VERSION,
        "seen_csv_sha256": sha256_file(seen_csv),
        "unseen_csv_sha256": sha256_file(unseen_csv),
        "checkpoint_sha256": sha256_file(checkpoint),
        "n_seen": int(len(seen_a)),
        "seen_counts_by_k": k_counts,
        "storage_dtype": "float32",
        "storage_shape": [int(len(seen_a)), int(grid["n_slices"]), 2, int(len(grid["tau"]))],
        "axis_order": ["configuration", "z_slice", "real_imag", "time"],
        "ssfm_half_window": float(grid["ssfm_half_window"]),
        "n_t": int(grid["n_t"]),
        "n_z": int(grid["n_z"]),
        "n_slices": int(grid["n_slices"]),
        "z_max_ld": float(grid["z_max_ld"]),
        "compare_t_min": float(grid["compare_t_min"]),
        "compare_t_max": float(grid["compare_t_max"]),
        "selected_steps": [int(x) for x in np.asarray(grid["selected_steps"]).tolist()],
        "zeta": [float(x) for x in np.asarray(grid["zeta"]).tolist()],
        "tau": [float(x) for x in np.asarray(grid["tau"]).tolist()],
        "pde_params": dict(pde),
        "contains_only_exact_pinn_seen_configurations": True,
        "all_selected_planes_are_supervised": True,
    }


def verify_dataset_meta(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    keys = [
        "seen_csv_sha256", "unseen_csv_sha256", "checkpoint_sha256", "n_seen",
        "storage_shape", "n_t", "n_z", "n_slices", "z_max_ld",
        "compare_t_min", "compare_t_max", "selected_steps",
    ]
    mismatches = [f"{k}: existing={existing.get(k)!r}, expected={expected.get(k)!r}" for k in keys if existing.get(k) != expected.get(k)]
    if mismatches:
        raise RuntimeError(
            "Existing shared dataset is incompatible. Use --force-data to rebuild it.\n" + "\n".join(mismatches)
        )


def prepare_shared_dataset(
    dataset_root: Path,
    seen_csv: Path,
    unseen_csv: Path,
    seen_k: np.ndarray,
    seen_a: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
    use_complex64: bool,
    force_data: bool,
    max_data_samples: int,
) -> Dict[str, Any]:
    if int(max_data_samples) > 0:
        seen_k = seen_k[: int(max_data_samples)]
        seen_a = seen_a[: int(max_data_samples)]
    paths = dataset_paths(dataset_root)
    if force_data and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    expected = expected_dataset_meta(seen_csv, unseen_csv, seen_k, seen_a, grid, pde, checkpoint)
    if paths["meta"].exists():
        existing = read_json(paths["meta"])
        verify_dataset_meta(existing, expected)
    else:
        write_json(paths["meta"], expected)

    if not paths["seen_csv"].exists():
        shutil.copy2(str(seen_csv), str(paths["seen_csv"]))
    if not paths["unseen_csv"].exists():
        shutil.copy2(str(unseen_csv), str(paths["unseen_csv"]))

    shape = tuple(int(x) for x in expected["storage_shape"])
    mode = "r+" if paths["data"].exists() else "w+"
    mmap = np.memmap(str(paths["data"]), dtype=np.float32, mode=mode, shape=shape)
    if paths["mask"].exists():
        complete = np.load(str(paths["mask"])).astype(bool)
        if complete.shape != (len(seen_a),):
            raise RuntimeError("Existing complete_mask.npy has incompatible shape.")
    else:
        complete = np.zeros(len(seen_a), dtype=bool)
        np.save(str(paths["mask"]), complete)

    meta = read_json(paths["meta"])
    running_max = float(meta.get("field_scale_max_abs_seen", 0.0))
    remaining = np.flatnonzero(~complete)
    print(f"[data] shared 81-plane seen labels: total={len(seen_a)}, remaining={len(remaining)}, shape={shape}", flush=True)
    t0 = time.time()
    for start in range(0, len(remaining), int(batch_size)):
        idx = remaining[start : start + int(batch_size)]
        maps = run_ssfm_batch_selected(
            amplitudes=seen_a[idx],
            grid=grid,
            pde=pde,
            device=device,
            use_complex64=use_complex64,
        )
        mmap[idx] = maps
        mmap.flush()
        running_max = max(running_max, float(np.max(np.abs(maps))))
        complete[idx] = True
        np.save(str(paths["mask"]), complete)
        meta.update(
            field_scale_max_abs_seen=float(max(running_max, 1e-8)),
            completed_count=int(np.count_nonzero(complete)),
            generation_complete=bool(np.all(complete)),
            data_file_bytes=int(paths["data"].stat().st_size),
        )
        write_json(paths["meta"], meta)
        elapsed = time.time() - t0
        print(f"[data] {int(np.count_nonzero(complete))}/{len(seen_a)} complete | elapsed={elapsed/60:.1f} min", flush=True)
        del maps
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del mmap
    return read_json(paths["meta"])


def load_shared_dataset_meta(dataset_root: Path) -> Dict[str, Any]:
    path = dataset_paths(dataset_root)["meta"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared 81-plane dataset metadata not found: {path}. Run --stage data first."
        )
    meta = read_json(path)
    if not bool(meta.get("generation_complete", False)):
        raise RuntimeError("Shared dataset is incomplete. Re-run --stage data to resume generation.")
    return meta


# -----------------------------------------------------------------------------
# Stratified model-selection split and K-balanced samplers
# -----------------------------------------------------------------------------

def stratified_split(k_labels: np.ndarray, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    train: List[int] = []
    val: List[int] = []
    all_idx = np.arange(len(k_labels), dtype=np.int64)
    for k in sorted(np.unique(k_labels).tolist()):
        idx = all_idx[k_labels == int(k)]
        perm = rng.permutation(idx)
        if len(idx) < 3:
            raise ValueError(f"K={k} has only {len(idx)} seen cases; at least 3 are required.")
        n_val = max(1, int(round(float(val_fraction) * len(idx))))
        n_val = min(n_val, len(idx) - 2)
        val.extend(int(x) for x in perm[:n_val])
        train.extend(int(x) for x in perm[n_val:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(val), dtype=np.int64)


def make_weighted_sampler(indices: np.ndarray, k_labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = {int(k): int(np.sum(k_labels[indices] == int(k))) for k in np.unique(k_labels[indices])}
    weights = np.asarray([1.0 / float(counts[int(k_labels[i])]) for i in indices], dtype=np.float64)
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(len(indices)),
        replacement=True,
        generator=gen,
    )


def build_map_loader(
    data_path: Path,
    shape: Sequence[int],
    indices: np.ndarray,
    k_labels: np.ndarray,
    scale: float,
    batch_size: int,
    balanced_k: bool,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ComplexMapMemmapDataset(data_path, shape, indices, scale)
    if balanced_k:
        sampler = make_weighted_sampler(indices, k_labels, seed)
        return DataLoader(dataset, batch_size=int(batch_size), sampler=sampler, num_workers=0, pin_memory=True)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, pin_memory=True)


# -----------------------------------------------------------------------------
# Universal CNN-81 training
# -----------------------------------------------------------------------------

def save_cnn_checkpoint(
    path: Path,
    model: FullPropagationCNN,
    model_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    scale: float,
    selected_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "state_dict": model.state_dict(),
            "model_config": dict(model_cfg),
            "train_config": dict(train_cfg),
            "field_scale": float(scale),
            "selected_epoch": int(selected_epoch),
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
        dilations=tuple(int(x) for x in cfg["dilations"]),
        context_bins=int(cfg["context_bins"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def train_universal_cnn(
    method_root: Path,
    dataset_root: Path,
    dataset_meta: Mapping[str, Any],
    seen_k: np.ndarray,
    device: torch.device,
    seed: int,
    split_seed: int,
    val_fraction: float,
    balanced_k: bool,
    batch_size: int,
    hidden: int,
    kernel_size: int,
    dilations: Sequence[int],
    context_bins: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    field_loss_weight: float,
    power_loss_weight: float,
    initial_loss_weight: float,
    terminal_loss_weight: float,
    peak_weight: float,
    amp_enabled: bool,
    force_train: bool,
) -> Path:
    final_ckpt = method_root / "final" / "universal_cnn81.pt"
    if final_ckpt.is_file() and not force_train:
        print(f"[CNN] existing final checkpoint -> {final_ckpt}", flush=True)
        return final_ckpt

    shape = tuple(int(x) for x in dataset_meta["storage_shape"])
    scale = float(dataset_meta["field_scale_max_abs_seen"])
    data_path = dataset_paths(dataset_root)["data"]
    train_idx, val_idx = stratified_split(seen_k, val_fraction, split_seed)
    selection_dir = ensure_dir(method_root / "selection")
    final_dir = ensure_dir(method_root / "final")
    save_csv_rows(
        selection_dir / "configuration_split.csv",
        [
            {"idx": int(i), "K": int(seen_k[i]), "split": "train"} for i in train_idx
        ]
        + [{"idx": int(i), "K": int(seen_k[i]), "split": "val"} for i in val_idx],
    )

    model_cfg = {
        "n_slices": int(shape[1]),
        "hidden": int(hidden),
        "kernel_size": int(kernel_size),
        "dilations": [int(x) for x in dilations],
        "context_bins": int(context_bins),
    }
    loss_weights = {
        "field": float(field_loss_weight),
        "power": float(power_loss_weight),
        "initial": float(initial_loss_weight),
        "terminal": float(terminal_loss_weight),
        "peak": float(peak_weight),
    }

    set_seed(seed)
    model = FullPropagationCNN(**model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(3, int(patience) // 4), min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and device.type == "cuda"))
    train_loader = build_map_loader(data_path, shape, train_idx, seen_k, scale, batch_size, balanced_k, seed, True)
    val_loader = build_map_loader(data_path, shape, val_idx, seen_k, scale, batch_size, False, seed, False)

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: List[Dict[str, Any]] = []
    best_path = selection_dir / "best_selection_cnn81.pt"
    t0 = time.time()
    print(
        f"[CNN] selection start | params={count_parameters(model):,} | train={len(train_idx)} val={len(val_idx)} | balanced_k={balanced_k}",
        flush=True,
    )
    for epoch in range(1, int(max_epochs) + 1):
        tr = run_epoch(model, train_loader, device, optimizer, scaler, amp_enabled, int(shape[1]), loss_weights)
        va = run_epoch(model, val_loader, device, None, None, amp_enabled, int(shape[1]), loss_weights)
        scheduler.step(float(va["loss"]))
        lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch": epoch,
            "train_loss": float(tr["loss"]),
            "val_loss": float(va["loss"]),
            "train_field_rel_sq": float(tr["field_rel_sq"]),
            "val_field_rel_sq": float(va["field_rel_sq"]),
            "train_power_rel_sq": float(tr["power_rel_sq"]),
            "val_power_rel_sq": float(va["power_rel_sq"]),
            "learning_rate": lr,
            "elapsed_sec": time.time() - t0,
        }
        history.append(row)
        print(
            f"[CNN select] epoch={epoch:4d} train={row['train_loss']:.4e} val={row['val_loss']:.4e} lr={lr:.2e}",
            flush=True,
        )
        improvement = best_val - float(va["loss"])
        threshold = max(1e-12, 1e-5 * abs(best_val)) if math.isfinite(best_val) else 0.0
        if improvement > threshold:
            best_val = float(va["loss"])
            best_epoch = int(epoch)
            stale = 0
            save_cnn_checkpoint(
                best_path,
                model,
                model_cfg,
                {"phase": "selection", "best_val": best_val, "balanced_k": balanced_k},
                scale,
                best_epoch,
            )
        else:
            stale += 1
        if epoch >= int(min_epochs) and stale >= int(patience):
            print(f"[CNN select] early stop at epoch={epoch}, selected_epoch={best_epoch}", flush=True)
            break

    if best_epoch <= 0:
        raise RuntimeError("CNN model selection failed to produce a checkpoint.")
    save_csv_rows(selection_dir / "training_history.csv", history)
    write_json(
        selection_dir / "selection_summary.json",
        {
            "script_version": SCRIPT_VERSION,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "balanced_k_sampling": bool(balanced_k),
            "model_config": model_cfg,
            "loss_weights": loss_weights,
        },
    )

    # Final training on every exact PINN-seen configuration for selected epochs.
    set_seed(seed)
    final_model = FullPropagationCNN(**model_cfg).to(device)
    final_opt = torch.optim.Adam(final_model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    final_scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and device.type == "cuda"))
    all_idx = np.arange(shape[0], dtype=np.int64)
    final_loader = build_map_loader(data_path, shape, all_idx, seen_k, scale, batch_size, balanced_k, seed + 17, True)
    final_history: List[Dict[str, Any]] = []
    t1 = time.time()
    for epoch in range(1, best_epoch + 1):
        tr = run_epoch(final_model, final_loader, device, final_opt, final_scaler, amp_enabled, int(shape[1]), loss_weights)
        row = {"epoch": epoch, "train_loss": float(tr["loss"]), "elapsed_sec": time.time() - t1}
        final_history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == best_epoch:
            print(f"[CNN final] epoch={epoch}/{best_epoch} loss={row['train_loss']:.4e}", flush=True)

    save_cnn_checkpoint(
        final_ckpt,
        final_model,
        model_cfg,
        {
            "phase": "final",
            "final_epochs": best_epoch,
            "used_all_seen_configurations": True,
            "balanced_k_sampling": bool(balanced_k),
            "loss_weights": loss_weights,
        },
        scale,
        best_epoch,
    )
    save_csv_rows(final_dir / "training_history.csv", final_history)
    write_json(
        final_dir / "train_config.json",
        {
            "script_version": SCRIPT_VERSION,
            "final_epochs": best_epoch,
            "n_seen": int(shape[0]),
            "model_config": model_cfg,
            "field_scale": scale,
            "balanced_k_sampling": bool(balanced_k),
            "seed": int(seed),
        },
    )
    print(f"[CNN] final checkpoint -> {final_ckpt}", flush=True)
    del model, final_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_ckpt


# -----------------------------------------------------------------------------
# Universal DDNN-81 training: exactly the universal PINN architecture
# -----------------------------------------------------------------------------

class UniversalPointLabelSampler:
    def __init__(
        self,
        data_path: Path,
        shape: Sequence[int],
        amplitudes: np.ndarray,
        k_labels: np.ndarray,
        tau: np.ndarray,
        zeta: np.ndarray,
    ) -> None:
        self.data_path = str(data_path)
        self.shape = tuple(int(x) for x in shape)
        self.amplitudes = np.asarray(amplitudes, dtype=np.float32)
        self.k_labels = np.asarray(k_labels, dtype=np.int64)
        self.tau = np.asarray(tau, dtype=np.float32)
        self.zeta = np.asarray(zeta, dtype=np.float32)
        self._mmap: Optional[np.memmap] = None

    def array(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(self.data_path, dtype=np.float32, mode="r", shape=self.shape)
        return self._mmap

    def sample(
        self,
        allowed_indices: np.ndarray,
        n_points: int,
        rng: np.random.Generator,
        balanced_k: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        allowed = np.asarray(allowed_indices, dtype=np.int64)
        if balanced_k:
            k_unique = np.unique(self.k_labels[allowed])
            chosen_k = rng.choice(k_unique, size=int(n_points), replace=True)
            config_ids = np.empty(int(n_points), dtype=np.int64)
            for k in k_unique:
                pos = np.flatnonzero(chosen_k == int(k))
                pool = allowed[self.k_labels[allowed] == int(k)]
                config_ids[pos] = rng.choice(pool, size=len(pos), replace=True)
        else:
            config_ids = rng.choice(allowed, size=int(n_points), replace=True)

        plane_ids = rng.integers(0, self.shape[1], size=int(n_points), endpoint=False)
        time_ids = rng.integers(0, self.shape[3], size=int(n_points), endpoint=False)
        target = np.asarray(self.array()[config_ids, plane_ids, :, time_ids], dtype=np.float32)
        z = self.zeta[plane_ids].reshape(-1, 1).astype(np.float32, copy=False)
        t = self.tau[time_ids].reshape(-1, 1).astype(np.float32, copy=False)
        amps = self.amplitudes[config_ids].astype(np.float32, copy=False)
        return z, t, amps, target

    def sample_weighted_k(
        self,
        allowed_indices: np.ndarray,
        n_points: int,
        rng: np.random.Generator,
        k_weights: Mapping[int, float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample supervised points with an explicit K distribution.

        The weights are converted to probabilities only over K values that are
        actually present in ``allowed_indices``. Within each selected K, an
        available seen amplitude configuration is sampled uniformly, followed by
        a uniformly sampled supervised propagation plane and time coordinate.
        """
        allowed = np.asarray(allowed_indices, dtype=np.int64)
        if len(allowed) == 0:
            raise ValueError("allowed_indices is empty.")
        k_unique = np.unique(self.k_labels[allowed])
        probs = np.asarray([max(0.0, float(k_weights.get(int(k), 0.0))) for k in k_unique], dtype=np.float64)
        if float(probs.sum()) <= 0.0:
            probs = np.ones_like(probs, dtype=np.float64)
        probs /= probs.sum()
        chosen_k = rng.choice(k_unique, size=int(n_points), replace=True, p=probs)
        config_ids = np.empty(int(n_points), dtype=np.int64)
        for k in k_unique:
            pos = np.flatnonzero(chosen_k == int(k))
            if len(pos) == 0:
                continue
            pool = allowed[self.k_labels[allowed] == int(k)]
            config_ids[pos] = rng.choice(pool, size=len(pos), replace=True)

        plane_ids = rng.integers(0, self.shape[1], size=int(n_points), endpoint=False)
        time_ids = rng.integers(0, self.shape[3], size=int(n_points), endpoint=False)
        target = np.asarray(self.array()[config_ids, plane_ids, :, time_ids], dtype=np.float32)
        z = self.zeta[plane_ids].reshape(-1, 1).astype(np.float32, copy=False)
        t = self.tau[time_ids].reshape(-1, 1).astype(np.float32, copy=False)
        amps = self.amplitudes[config_ids].astype(np.float32, copy=False)
        return z, t, amps, target


def ddnn_mse_on_fixed_points(
    model: ConditionalPINN,
    z: np.ndarray,
    t: np.ndarray,
    amps: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(z), int(chunk_size)):
            end = min(len(z), start + int(chunk_size))
            zt = torch.from_numpy(z[start:end]).to(device)
            tt = torch.from_numpy(t[start:end]).to(device)
            at = torch.from_numpy(amps[start:end]).to(device)
            yt = torch.from_numpy(target[start:end]).to(device)
            up, vp = model(zt, tt, at)
            pred = torch.cat([up, vp], dim=1)
            total += float(F.mse_loss(pred, yt, reduction="sum").detach().cpu())
            count += int(yt.numel())
    return total / max(count, 1)


def save_ddnn_checkpoint(
    path: Path,
    model: ConditionalPINN,
    model_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_state": model.state_dict(),
            "model_config": dict(model_cfg),
            "train_config": dict(train_cfg),
        },
        str(path),
    )


def load_ddnn_checkpoint(path: Path, device: torch.device) -> Tuple[ConditionalPINN, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    model = ConditionalPINN(**dict(payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _ddnn_adam_stage(
    model: ConditionalPINN,
    labels: UniversalPointLabelSampler,
    allowed_indices: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    steps: int,
    batch_points: int,
    learning_rate: float,
    weight_decay: float,
    k_weights: Mapping[int, float],
    grad_clip: float,
    log_every: int,
    phase: str,
    history: List[Dict[str, Any]],
    t0: float,
) -> None:
    if int(steps) <= 0:
        return
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    for step in range(1, int(steps) + 1):
        model.train()
        z, t, amps, target = labels.sample_weighted_k(
            allowed_indices, int(batch_points), rng, k_weights
        )
        zt = torch.from_numpy(z).to(device)
        tt = torch.from_numpy(t).to(device)
        at = torch.from_numpy(amps).to(device)
        yt = torch.from_numpy(target).to(device)
        optimizer.zero_grad(set_to_none=True)
        up, vp = model(zt, tt, at)
        pred = torch.cat([up, vp], dim=1)
        loss = F.mse_loss(pred, yt)
        loss.backward()
        if float(grad_clip) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()

        if step == 1 or step % int(log_every) == 0 or step == int(steps):
            row = {
                "phase": phase,
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_sec": time.time() - t0,
            }
            history.append(row)
            print(
                f"[DDNN {phase}] step={step:6d}/{int(steps)} mse={row['loss']:.4e} lr={row['learning_rate']:.2e}",
                flush=True,
            )
        del zt, tt, at, yt, pred, loss


def _ddnn_lbfgs_closure_loss(
    model: ConditionalPINN,
    z: np.ndarray,
    t: np.ndarray,
    amps: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> float:
    """Compute exact mean supervised MSE and accumulate its gradient in chunks."""
    n = int(len(z))
    if n <= 0:
        raise ValueError("Empty L-BFGS supervised point set.")
    denom = float(n * target.shape[1])
    total_sum = 0.0
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        zt = torch.from_numpy(z[start:end]).to(device)
        tt = torch.from_numpy(t[start:end]).to(device)
        at = torch.from_numpy(amps[start:end]).to(device)
        yt = torch.from_numpy(target[start:end]).to(device)
        up, vp = model(zt, tt, at)
        pred = torch.cat([up, vp], dim=1)
        loss_sum = F.mse_loss(pred, yt, reduction="sum")
        (loss_sum / denom).backward()
        total_sum += float(loss_sum.detach().cpu())
        del zt, tt, at, yt, pred, loss_sum
    return total_sum / denom


def train_universal_ddnn(
    method_root: Path,
    dataset_root: Path,
    dataset_meta: Mapping[str, Any],
    seen_k: np.ndarray,
    seen_a: np.ndarray,
    exact_pinn_model_cfg: Mapping[str, Any],
    pinn_checkpoint: Path,
    device: torch.device,
    seed: int,
    batch_points: int,
    learning_rate: float,
    weight_decay: float,
    adam_steps: int,
    finetune_steps: int,
    finetune_learning_rate: float,
    lbfgs_points: int,
    lbfgs_epochs: int,
    lbfgs_max_iter: int,
    min_lbfgs_epochs: int,
    early_stop_eps: float,
    early_stop_patience: int,
    lbfgs_chunk_size: int,
    grad_clip: float,
    log_every: int,
    force_train: bool,
) -> Path:
    """Train the universal DDNN with the same optimizer stages as the universal PINN.

    Architecture:
      copied exactly from the universal PINN checkpoint.

    Optimization schedule:
      1) Adam main stage: 5000 steps by default;
      2) high-K-emphasis Adam fine-tuning: 1500 steps by default;
      3) L-BFGS refinement: up to 1200 epochs, max_iter=20 by default,
         with the same early-stop convention as the universal PINN.

    All stages use only the exact PINN-seen amplitude configurations and the
    shared SSFM labels on all 81 supervised planes. The unseen set is never used.
    """
    final_ckpt = method_root / "final" / "universal_ddnn81.pt"
    if final_ckpt.is_file() and not force_train:
        print(f"[DDNN] existing final checkpoint -> {final_ckpt}", flush=True)
        return final_ckpt

    shape = tuple(int(x) for x in dataset_meta["storage_shape"])
    tau = np.asarray(dataset_meta["tau"], dtype=np.float32)
    zeta = np.asarray(dataset_meta["zeta"], dtype=np.float32)
    labels = UniversalPointLabelSampler(
        dataset_paths(dataset_root)["data"], shape, seen_a, seen_k, tau, zeta
    )
    all_idx = np.arange(len(seen_a), dtype=np.int64)
    final_dir = ensure_dir(method_root / "final")

    # Match the universal PINN's K emphasis by reusing the same point-allocation
    # ratios for the two Adam stages and for L-BFGS refinement.
    main_k_weights = {int(k): float(v) for k, v in DEFAULT_MAIN_PDE_COUNTS_BY_K.items()}
    finetune_k_weights = {int(k): float(v) for k, v in DEFAULT_FINETUNE_PDE_COUNTS_BY_K.items()}

    model_cfg = dict(exact_pinn_model_cfg)
    set_seed(seed)
    model = ConditionalPINN(**model_cfg).to(device)
    history: List[Dict[str, Any]] = []
    t0 = time.time()
    print(
        f"[DDNN] training start | exact PINN architecture={model_cfg} | "
        f"params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
        flush=True,
    )
    print(
        f"[DDNN] schedule = Adam({adam_steps}) -> high-K Adam({finetune_steps}) -> "
        f"L-BFGS(up to {lbfgs_epochs}, max_iter={lbfgs_max_iter}) | supervised planes={shape[1]}",
        flush=True,
    )

    main_rng = np.random.default_rng(int(seed) + 81000)
    _ddnn_adam_stage(
        model=model,
        labels=labels,
        allowed_indices=all_idx,
        device=device,
        rng=main_rng,
        steps=adam_steps,
        batch_points=batch_points,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        k_weights=main_k_weights,
        grad_clip=grad_clip,
        log_every=log_every,
        phase="adam",
        history=history,
        t0=t0,
    )
    save_ddnn_checkpoint(
        final_dir / "universal_ddnn81_after_main_adam.pt",
        model,
        model_cfg,
        {
            "phase": "after_main_adam",
            "adam_steps": int(adam_steps),
            "supervised_planes": int(shape[1]),
            "pinn_architecture_checkpoint": str(pinn_checkpoint),
        },
    )

    finetune_rng = np.random.default_rng(int(seed) + 82000)
    _ddnn_adam_stage(
        model=model,
        labels=labels,
        allowed_indices=all_idx,
        device=device,
        rng=finetune_rng,
        steps=finetune_steps,
        batch_points=batch_points,
        learning_rate=finetune_learning_rate,
        weight_decay=weight_decay,
        k_weights=finetune_k_weights,
        grad_clip=grad_clip,
        log_every=log_every,
        phase="adam_highk_finetune",
        history=history,
        t0=t0,
    )
    save_ddnn_checkpoint(
        final_dir / "universal_ddnn81_after_highk_finetune.pt",
        model,
        model_cfg,
        {
            "phase": "after_highk_finetune",
            "adam_steps": int(adam_steps),
            "finetune_steps": int(finetune_steps),
            "supervised_planes": int(shape[1]),
            "pinn_architecture_checkpoint": str(pinn_checkpoint),
        },
    )

    # Fixed supervised point set for deterministic L-BFGS closures. The set is
    # drawn from all exact PINN-seen configurations and all 81 supervised planes.
    lbfgs_epochs_ran = 0
    best_lbfgs = float("inf")
    if int(lbfgs_epochs) > 0:
        lbfgs_rng = np.random.default_rng(int(seed) + 83000)
        lz, lt, la, ly = labels.sample_weighted_k(
            all_idx, int(lbfgs_points), lbfgs_rng, finetune_k_weights
        )
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(lbfgs_max_iter),
            max_eval=max(int(lbfgs_max_iter) + 10, int(lbfgs_max_iter)),
            line_search_fn="strong_wolfe",
        )
        stale_epochs = 0
        for ep in range(1, int(lbfgs_epochs) + 1):
            vals: Dict[str, float] = {}

            def closure() -> torch.Tensor:
                opt_lbfgs.zero_grad(set_to_none=True)
                model.train()
                mse = _ddnn_lbfgs_closure_loss(
                    model, lz, lt, la, ly, device, int(lbfgs_chunk_size)
                )
                vals["loss"] = float(mse)
                return torch.tensor(float(mse), device=device, dtype=torch.float32)

            loss_out = opt_lbfgs.step(closure)
            current = vals.get(
                "loss",
                float(loss_out.detach().cpu()) if torch.is_tensor(loss_out) else float(loss_out),
            )
            lbfgs_epochs_ran = int(ep)
            improvement = best_lbfgs - current
            threshold = (
                max(float(early_stop_eps), abs(best_lbfgs) * 1e-7)
                if math.isfinite(best_lbfgs)
                else 0.0
            )
            if current < best_lbfgs:
                best_lbfgs = float(current)
            if improvement > threshold:
                stale_epochs = 0
            else:
                stale_epochs += 1

            if ep == 1 or ep % int(log_every) == 0 or ep == int(lbfgs_epochs):
                row = {
                    "phase": "lbfgs",
                    "step": int(ep),
                    "loss": float(current),
                    "elapsed_sec": time.time() - t0,
                }
                history.append(row)
                print(
                    f"[DDNN lbfgs] epoch={ep:5d}/{int(lbfgs_epochs)} mse={current:.4e} "
                    f"best={best_lbfgs:.4e} stale={stale_epochs}",
                    flush=True,
                )

            if ep >= int(min_lbfgs_epochs) and stale_epochs >= int(early_stop_patience):
                row = {
                    "phase": "lbfgs_early_stop",
                    "step": int(ep),
                    "loss": float(current),
                    "elapsed_sec": time.time() - t0,
                }
                history.append(row)
                print(
                    f"[DDNN lbfgs] early stop at epoch={ep}, stale_epochs={stale_epochs}",
                    flush=True,
                )
                break

    train_cfg = {
        "phase": "final",
        "used_all_seen_configurations": True,
        "supervised_planes": int(shape[1]),
        "adam_steps": int(adam_steps),
        "adam_learning_rate": float(learning_rate),
        "finetune_steps": int(finetune_steps),
        "finetune_learning_rate": float(finetune_learning_rate),
        "lbfgs_points": int(lbfgs_points),
        "lbfgs_epochs_requested": int(lbfgs_epochs),
        "lbfgs_epochs_ran": int(lbfgs_epochs_ran),
        "lbfgs_max_iter": int(lbfgs_max_iter),
        "best_lbfgs_mse": float(best_lbfgs) if math.isfinite(best_lbfgs) else None,
        "main_k_sampling_weights": {str(k): float(v) for k, v in main_k_weights.items()},
        "finetune_k_sampling_weights": {str(k): float(v) for k, v in finetune_k_weights.items()},
        "pinn_architecture_checkpoint": str(pinn_checkpoint),
        "model_config": model_cfg,
        "seed": int(seed),
    }
    save_ddnn_checkpoint(final_ckpt, model, model_cfg, train_cfg)
    save_csv_rows(final_dir / "training_history.csv", history)
    write_json(final_dir / "train_config.json", {"script_version": SCRIPT_VERSION, **train_cfg})
    print(f"[DDNN] final checkpoint -> {final_ckpt}", flush=True)

    del model, labels
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return final_ckpt


# -----------------------------------------------------------------------------
# Full 81-plane unseen evaluation for either baseline
# -----------------------------------------------------------------------------

def predict_cnn_maps(
    model: FullPropagationCNN,
    payload: Mapping[str, Any],
    reference_maps: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    scale = float(payload["field_scale"])
    x = torch.from_numpy(np.asarray(reference_maps[:, 0], dtype=np.float32) / scale).to(device)
    with torch.inference_mode():
        pred = model(x)
    b, _, t = pred.shape
    z = int(payload["model_config"]["n_slices"])
    out = pred.float().view(b, z, 2, t).detach().cpu().numpy() * scale
    return out.astype(np.float32, copy=False)


def predict_ddnn_maps(
    model: ConditionalPINN,
    amplitudes: np.ndarray,
    tau: np.ndarray,
    zeta: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)
    zeta = np.asarray(zeta, dtype=np.float32)
    b, nz, nt = len(amps), len(zeta), len(tau)
    total = b * nz * nt
    out = np.empty((total, 2), dtype=np.float32)
    amp_t = torch.from_numpy(amps).to(device)
    zeta_t = torch.from_numpy(zeta).to(device)
    tau_t = torch.from_numpy(tau).to(device)
    model.eval()
    with torch.inference_mode():
        for start in range(0, total, int(chunk_size)):
            end = min(total, start + int(chunk_size))
            flat = torch.arange(start, end, device=device, dtype=torch.long)
            sample_idx = torch.div(flat, nz * nt, rounding_mode="floor")
            rem = flat % (nz * nt)
            z_idx = torch.div(rem, nt, rounding_mode="floor")
            t_idx = rem % nt
            z = zeta_t[z_idx].reshape(-1, 1)
            t = tau_t[t_idx].reshape(-1, 1)
            a = amp_t[sample_idx]
            u, v = model(z, t, a)
            pred = torch.cat([u, v], dim=1)
            out[start:end] = pred.detach().cpu().numpy().astype(np.float32, copy=False)
    return out.reshape(b, nz, nt, 2).transpose(0, 1, 3, 2)


def metrics_from_maps(pred: np.ndarray, ref: np.ndarray) -> Dict[str, np.ndarray]:
    pred64 = np.asarray(pred, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    eps = 1e-300
    diff = pred64 - ref64
    full_field = np.sqrt(np.sum(diff**2, axis=(1, 2, 3))) / np.maximum(
        np.sqrt(np.sum(ref64**2, axis=(1, 2, 3))), eps
    )
    p_pred = np.sum(pred64**2, axis=2)
    p_ref = np.sum(ref64**2, axis=2)
    p_diff = p_pred - p_ref
    full_power = np.sqrt(np.sum(p_diff**2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(p_ref**2, axis=(1, 2))), eps
    )
    term_diff = diff[:, -1]
    terminal_field = np.sqrt(np.sum(term_diff**2, axis=(1, 2))) / np.maximum(
        np.sqrt(np.sum(ref64[:, -1] ** 2, axis=(1, 2))), eps
    )
    terminal_power = np.sqrt(np.sum(p_diff[:, -1] ** 2, axis=1)) / np.maximum(
        np.sqrt(np.sum(p_ref[:, -1] ** 2, axis=1)), eps
    )
    max_abs_power = np.max(np.abs(p_diff), axis=(1, 2))
    return {
        "full_rel_l2_field": full_field,
        "full_rel_l2_power": full_power,
        "terminal_rel_l2_field": terminal_field,
        "terminal_rel_l2_power": terminal_power,
        "max_abs_power_error": max_abs_power,
    }


def parse_done_indices(path: Path) -> set[int]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["idx"]))
            except Exception:
                pass
    return done


def select_eval_positions(k_labels: np.ndarray, max_per_k: int, seed: int) -> np.ndarray:
    all_pos = np.arange(len(k_labels), dtype=np.int64)
    if int(max_per_k) <= 0:
        return all_pos
    rng = np.random.default_rng(int(seed))
    chosen: List[int] = []
    for k in sorted(np.unique(k_labels).tolist()):
        pos = all_pos[k_labels == int(k)]
        n = min(len(pos), int(max_per_k))
        if n < len(pos):
            pos = np.sort(rng.choice(pos, size=n, replace=False))
        chosen.extend(int(x) for x in pos)
    return np.asarray(sorted(chosen), dtype=np.int64)


def percentile_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def summarize_metrics(metrics_path: Path, out_dir: Path) -> None:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No metric rows in {metrics_path}")
    k_values = sorted(set(int(r["K"]) for r in rows))
    summary_rows: List[Dict[str, Any]] = []
    per_k_internal: Dict[int, Dict[str, Dict[str, float]]] = {}
    for k in k_values:
        group = [r for r in rows if int(r["K"]) == k]
        item: Dict[str, Any] = {"K": k, "n_samples": len(group)}
        per_k_internal[k] = {}
        for metric in METRIC_NAMES:
            vals = np.asarray([float(r[metric]) for r in group], dtype=float)
            stats = percentile_stats(vals)
            per_k_internal[k][metric] = stats
            for s, v in stats.items():
                item[f"{metric}_{s}"] = v
        summary_rows.append(item)
    save_csv_rows(out_dir / "summary_by_k.csv", summary_rows)

    micro: Dict[str, Any] = {}
    macro: Dict[str, Any] = {}
    for metric in METRIC_NAMES:
        all_vals = np.asarray([float(r[metric]) for r in rows], dtype=float)
        micro[metric] = percentile_stats(all_vals)
        macro[metric] = {
            stat: float(np.mean([per_k_internal[k][metric][stat] for k in k_values]))
            for stat in ["mean", "std", "p50", "p90", "p95", "max"]
        }
    write_json(
        out_dir / "overall_summary.json",
        {
            "n_samples": len(rows),
            "K_values": k_values,
            "micro_sample_weighted": micro,
            "macro_equal_K_weight": macro,
            "primary_metric": "macro_equal_K_weight/full_rel_l2_power/mean",
        },
    )
    print(
        f"[summary] macro full power rel-L2 = {100*macro['full_rel_l2_power']['mean']:.4f}% | "
        f"macro full field rel-L2 = {100*macro['full_rel_l2_field']['mean']:.4f}%",
        flush=True,
    )


def evaluate_method(
    method: str,
    method_root: Path,
    checkpoint: Path,
    unseen_k: np.ndarray,
    unseen_a: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    device: torch.device,
    ssfm_batch_size: int,
    ssfm_complex64: bool,
    ddnn_chunk_size: int,
    max_eval_per_k: int,
    eval_seed: int,
    overwrite: bool,
) -> None:
    out_dir = ensure_dir(method_root / "eval_full81_all_unseen")
    metrics_path = out_dir / "metrics_stream.csv"
    if overwrite and metrics_path.exists():
        metrics_path.unlink()

    positions = select_eval_positions(unseen_k, max_eval_per_k, eval_seed)
    selected_k = unseen_k[positions]
    selected_a = unseen_a[positions]
    selected_idx = positions.copy()  # original CSV row index because positions reference the full file.
    done = parse_done_indices(metrics_path)
    keep = np.asarray([int(i) not in done for i in selected_idx], dtype=bool)
    pending_idx = selected_idx[keep]
    pending_k = selected_k[keep]
    pending_a = selected_a[keep]

    method = method.lower()
    if method == "cnn":
        model, payload = load_cnn_checkpoint(checkpoint, device)
    elif method == "ddnn":
        model, payload = load_ddnn_checkpoint(checkpoint, device)
    else:
        raise ValueError(method)

    fieldnames = ["idx", "K"] + [f"A{i}" for i in range(1, 9)] + ["levels"] + METRIC_NAMES + ["ssfm_sec_per_sample", "model_sec_per_sample"]
    new_file = not metrics_path.exists() or metrics_path.stat().st_size == 0
    f = metrics_path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()
        f.flush()

    print(
        f"[{method.upper()} eval] selected={len(selected_idx)} already_done={len(selected_idx)-len(pending_idx)} pending={len(pending_idx)} | planes={grid['n_slices']}",
        flush=True,
    )
    t_all = time.time()
    completed = 0
    try:
        for start in range(0, len(pending_idx), int(ssfm_batch_size)):
            end = min(len(pending_idx), start + int(ssfm_batch_size))
            idx = pending_idx[start:end]
            kk = pending_k[start:end]
            aa = pending_a[start:end]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            ref = run_ssfm_batch_selected(aa, grid, pde, device, ssfm_complex64)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ssfm_sec = time.perf_counter() - t0

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            if method == "cnn":
                pred = predict_cnn_maps(model, payload, ref, device)
            else:
                pred = predict_ddnn_maps(
                    model, aa, np.asarray(grid["tau"]), np.asarray(grid["zeta"]), device, ddnn_chunk_size
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_sec = time.perf_counter() - t1

            metrics = metrics_from_maps(pred, ref)
            b = len(idx)
            for i in range(b):
                row: Dict[str, Any] = {
                    "idx": int(idx[i]),
                    "K": int(kk[i]),
                    **{f"A{j+1}": f"{float(aa[i,j]):g}" for j in range(8)},
                    "levels": combo_text(aa[i]),
                    **{m: float(metrics[m][i]) for m in METRIC_NAMES},
                    "ssfm_sec_per_sample": float(ssfm_sec / b),
                    "model_sec_per_sample": float(model_sec / b),
                }
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            completed += b
            elapsed = time.time() - t_all
            rate = completed / max(elapsed, 1e-9)
            eta = (len(pending_idx) - completed) / max(rate, 1e-12)
            print(
                f"[{method.upper()} eval] {completed}/{len(pending_idx)} | "
                f"batch full-power={100*np.mean(metrics['full_rel_l2_power']):.3f}% | "
                f"rate={rate:.2f} sample/s ETA={eta/60:.1f} min",
                flush=True,
            )
            del ref, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        f.close()

    summarize_metrics(metrics_path, out_dir)
    write_json(
        out_dir / "eval_config.json",
        {
            "script_version": SCRIPT_VERSION,
            "method": method,
            "checkpoint": str(checkpoint),
            "n_selected_unseen": int(len(selected_idx)),
            "max_eval_per_k": int(max_eval_per_k),
            "all_81_planes_evaluated": int(grid["n_slices"]) == 81,
            "grid": {k: v for k, v in grid.items() if k not in {"tau_full", "time_mask", "tau", "selected_steps", "zeta"}},
            "selected_steps": [int(x) for x in np.asarray(grid["selected_steps"]).tolist()],
        },
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train/evaluate universal K=1..8 CNN-81 and same-architecture DDNN-81 baselines using the exact universal-PINN split."
    )
    p.add_argument("--runs-root", default="./MULTIPULSE_AMPLITUDE_RUNS")
    p.add_argument("--pinn-run-dir", default="", help="Optional explicit universal PINN run. Otherwise auto-discover under --runs-root.")
    p.add_argument("--pinn-checkpoint", default="", help="Optional explicit sparse8_forward_pinn.pt.")
    p.add_argument("--output-root", default="", help="Default: <universal PINN run>/universal_CNN_DDNN_full81")
    p.add_argument("--method", choices=["cnn", "ddnn", "both"], default="both")
    p.add_argument("--stage", choices=["data", "train", "eval", "all"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.20)

    p.add_argument("--balanced-k-sampling", dest="balanced_k_sampling", action="store_true", default=True)
    p.add_argument("--no-balanced-k-sampling", dest="balanced_k_sampling", action="store_false")

    # Shared 81-plane SSFM label grid. Defaults match the current universal-PINN formal evaluation window.
    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--compare-t-min", type=float, default=None)
    p.add_argument("--compare-t-max", type=float, default=None)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--n-slices", type=int, default=81)
    p.add_argument("--ssfm-batch-size", type=int, default=4)
    p.add_argument("--ssfm-complex64", action="store_true", help="Faster/lower-memory SSFM. Default uses complex128, matching the formal fixed-M data pipeline.")
    p.add_argument("--force-data", action="store_true")
    p.add_argument("--max-data-samples", type=int, default=0, help="Debug only. 0 uses every PINN-seen waveform.")

    # CNN: architecture defaults copied from the fixed-M full-propagation CNN.
    p.add_argument("--cnn-hidden", type=int, default=32)
    p.add_argument("--cnn-kernel-size", type=int, default=11)
    p.add_argument("--cnn-dilations", nargs=4, type=int, default=[1, 4, 16, 64])
    p.add_argument("--cnn-context-bins", type=int, default=32)
    p.add_argument("--cnn-batch-size", type=int, default=16)
    p.add_argument("--cnn-learning-rate", type=float, default=1e-4)
    p.add_argument("--cnn-weight-decay", type=float, default=0.0)
    p.add_argument("--cnn-max-epochs", type=int, default=600)
    p.add_argument("--cnn-min-epochs", type=int, default=50)
    p.add_argument("--cnn-patience", type=int, default=60)
    p.add_argument("--cnn-field-loss-weight", type=float, default=1.0)
    p.add_argument("--cnn-power-loss-weight", type=float, default=1.0)
    p.add_argument("--cnn-initial-loss-weight", type=float, default=2.0)
    p.add_argument("--cnn-terminal-loss-weight", type=float, default=1.0)
    p.add_argument("--cnn-peak-weight", type=float, default=4.0)
    p.add_argument("--cnn-no-amp", action="store_true")

    # DDNN: architecture is copied exactly from the universal PINN checkpoint.
    # The optimizer schedule mirrors the universal PINN: Adam -> high-K Adam -> L-BFGS.
    p.add_argument("--ddnn-batch-points", type=int, default=16384)
    p.add_argument("--ddnn-learning-rate", type=float, default=1e-3)
    p.add_argument("--ddnn-weight-decay", type=float, default=0.0)
    p.add_argument("--ddnn-adam-steps", type=int, default=5000)
    p.add_argument("--ddnn-finetune-steps", type=int, default=1500)
    p.add_argument("--ddnn-finetune-learning-rate", type=float, default=3e-4)
    p.add_argument("--ddnn-lbfgs-points", type=int, default=65536)
    p.add_argument("--ddnn-lbfgs-epochs", type=int, default=1200)
    p.add_argument("--ddnn-lbfgs-max-iter", type=int, default=20)
    p.add_argument("--ddnn-min-lbfgs-epochs", type=int, default=100)
    p.add_argument("--ddnn-early-stop-eps", type=float, default=1e-8)
    p.add_argument("--ddnn-early-stop-patience", type=int, default=20)
    p.add_argument("--ddnn-lbfgs-chunk-size", type=int, default=16384)
    p.add_argument("--ddnn-grad-clip", type=float, default=10.0)
    p.add_argument("--ddnn-log-every", type=int, default=100)
    p.add_argument("--ddnn-chunk-size", type=int, default=65536, help="Inference/evaluation chunk size.")

    p.add_argument("--force-train", action="store_true")
    p.add_argument("--max-eval-per-k", type=int, default=0, help="0 evaluates every exact PINN-unseen waveform.")
    p.add_argument("--eval-seed", type=int, default=2027)
    p.add_argument("--overwrite-eval", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = safe_device(args.device)
    runs_root = Path(args.runs_root).expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    run_dir = find_universal_run(runs_root, args.pinn_run_dir)
    checkpoint = resolve_checkpoint(run_dir, args.pinn_checkpoint)
    model_cfg, pde = load_pinn_checkpoint_metadata(checkpoint)
    seen_csv = run_dir / "dataset" / "seen_sparse8_combinations.csv"
    unseen_csv = run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    seen_k, seen_a = load_sparse8_csv(seen_csv)
    unseen_k, unseen_a = load_sparse8_csv(unseen_csv)

    output_root = ensure_dir(
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else run_dir / "universal_CNN_DDNN_full81"
    )
    dataset_root = output_root / "shared_seen_SSFM_81planes"
    cnn_root = output_root / "universal_CNN81"
    ddnn_root = output_root / "universal_DDNN81_same_as_PINN"

    grid = make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=args.ssfm_half_window,
        n_t=args.n_t,
        n_z=args.n_z,
        n_slices=args.n_slices,
        compare_t_min=args.compare_t_min,
        compare_t_max=args.compare_t_max,
    )

    print("=" * 110, flush=True)
    print(f"SCRIPT VERSION : {SCRIPT_VERSION}", flush=True)
    print(f"PINN run       : {run_dir}", flush=True)
    print(f"PINN checkpoint: {checkpoint}", flush=True)
    print(f"seen/unseen    : {len(seen_a)} / {len(unseen_a)}", flush=True)
    print(f"seen by K      : { {int(k): int(np.sum(seen_k==k)) for k in np.unique(seen_k)} }", flush=True)
    print(f"unseen by K    : { {int(k): int(np.sum(unseen_k==k)) for k in np.unique(unseen_k)} }", flush=True)
    print(f"grid           : {args.n_slices} planes, nt={args.n_t}, nz={args.n_z}, compare=[{grid['compare_t_min']},{grid['compare_t_max']}]", flush=True)
    print(f"balanced K     : {args.balanced_k_sampling}", flush=True)
    print(f"output         : {output_root}", flush=True)
    print("=" * 110, flush=True)

    if args.stage in ("data", "all"):
        prepare_shared_dataset(
            dataset_root=dataset_root,
            seen_csv=seen_csv,
            unseen_csv=unseen_csv,
            seen_k=seen_k,
            seen_a=seen_a,
            grid=grid,
            pde=pde,
            checkpoint=checkpoint,
            device=device,
            batch_size=args.ssfm_batch_size,
            use_complex64=args.ssfm_complex64,
            force_data=args.force_data,
            max_data_samples=args.max_data_samples,
        )
        if args.stage == "data":
            print("[done] shared dataset prepared.", flush=True)
            return

    dataset_meta = load_shared_dataset_meta(dataset_root)
    if int(dataset_meta["n_seen"]) != len(seen_a):
        if int(args.max_data_samples) <= 0:
            raise RuntimeError(
                f"Shared dataset contains {dataset_meta['n_seen']} seen cases but exact PINN seen CSV contains {len(seen_a)}."
            )
        # Debug-only truncated-data mode.
        seen_k = seen_k[: int(dataset_meta["n_seen"])]
        seen_a = seen_a[: int(dataset_meta["n_seen"])]

    methods = ["cnn", "ddnn"] if args.method == "both" else [args.method]
    checkpoints: Dict[str, Path] = {}

    if args.stage in ("train", "all"):
        if "cnn" in methods:
            checkpoints["cnn"] = train_universal_cnn(
                method_root=cnn_root,
                dataset_root=dataset_root,
                dataset_meta=dataset_meta,
                seen_k=seen_k,
                device=device,
                seed=args.seed,
                split_seed=args.split_seed,
                val_fraction=args.val_fraction,
                balanced_k=args.balanced_k_sampling,
                batch_size=args.cnn_batch_size,
                hidden=args.cnn_hidden,
                kernel_size=args.cnn_kernel_size,
                dilations=args.cnn_dilations,
                context_bins=args.cnn_context_bins,
                learning_rate=args.cnn_learning_rate,
                weight_decay=args.cnn_weight_decay,
                max_epochs=args.cnn_max_epochs,
                min_epochs=args.cnn_min_epochs,
                patience=args.cnn_patience,
                field_loss_weight=args.cnn_field_loss_weight,
                power_loss_weight=args.cnn_power_loss_weight,
                initial_loss_weight=args.cnn_initial_loss_weight,
                terminal_loss_weight=args.cnn_terminal_loss_weight,
                peak_weight=args.cnn_peak_weight,
                amp_enabled=not args.cnn_no_amp,
                force_train=args.force_train,
            )
        if "ddnn" in methods:
            checkpoints["ddnn"] = train_universal_ddnn(
                method_root=ddnn_root,
                dataset_root=dataset_root,
                dataset_meta=dataset_meta,
                seen_k=seen_k,
                seen_a=seen_a,
                exact_pinn_model_cfg=model_cfg,
                pinn_checkpoint=checkpoint,
                device=device,
                seed=args.seed,
                batch_points=args.ddnn_batch_points,
                learning_rate=args.ddnn_learning_rate,
                weight_decay=args.ddnn_weight_decay,
                adam_steps=args.ddnn_adam_steps,
                finetune_steps=args.ddnn_finetune_steps,
                finetune_learning_rate=args.ddnn_finetune_learning_rate,
                lbfgs_points=args.ddnn_lbfgs_points,
                lbfgs_epochs=args.ddnn_lbfgs_epochs,
                lbfgs_max_iter=args.ddnn_lbfgs_max_iter,
                min_lbfgs_epochs=args.ddnn_min_lbfgs_epochs,
                early_stop_eps=args.ddnn_early_stop_eps,
                early_stop_patience=args.ddnn_early_stop_patience,
                lbfgs_chunk_size=args.ddnn_lbfgs_chunk_size,
                grad_clip=args.ddnn_grad_clip,
                log_every=args.ddnn_log_every,
                force_train=args.force_train,
            )

    if args.stage in ("eval", "all"):
        if "cnn" in methods:
            ckpt = checkpoints.get("cnn", cnn_root / "final" / "universal_cnn81.pt")
            if not ckpt.is_file():
                raise FileNotFoundError(f"CNN checkpoint not found: {ckpt}")
            evaluate_method(
                method="cnn",
                method_root=cnn_root,
                checkpoint=ckpt,
                unseen_k=unseen_k,
                unseen_a=unseen_a,
                grid=grid,
                pde=pde,
                device=device,
                ssfm_batch_size=args.ssfm_batch_size,
                ssfm_complex64=args.ssfm_complex64,
                ddnn_chunk_size=args.ddnn_chunk_size,
                max_eval_per_k=args.max_eval_per_k,
                eval_seed=args.eval_seed,
                overwrite=args.overwrite_eval,
            )
        if "ddnn" in methods:
            ckpt = checkpoints.get("ddnn", ddnn_root / "final" / "universal_ddnn81.pt")
            if not ckpt.is_file():
                raise FileNotFoundError(f"DDNN checkpoint not found: {ckpt}")
            evaluate_method(
                method="ddnn",
                method_root=ddnn_root,
                checkpoint=ckpt,
                unseen_k=unseen_k,
                unseen_a=unseen_a,
                grid=grid,
                pde=pde,
                device=device,
                ssfm_batch_size=args.ssfm_batch_size,
                ssfm_complex64=args.ssfm_complex64,
                ddnn_chunk_size=args.ddnn_chunk_size,
                max_eval_per_k=args.max_eval_per_k,
                eval_seed=args.eval_seed,
                overwrite=args.overwrite_eval,
            )

    print("\nAll requested stages are complete.", flush=True)


if __name__ == "__main__":
    main()
