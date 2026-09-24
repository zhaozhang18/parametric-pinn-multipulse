# -*- coding: utf-8 -*-
"""
inverse_data_hybrid_mapping.py

Fixed-M normalized-amplitude 4-PAM inverse mapping with two comparable inverse networks:

1) Pure data-driven inverse network
   received terminal SSFM power waveform y_L(t) -> M 4-PAM amplitude classes.
   Loss = data supervision only, i.e. cross entropy against the true initial P.

2) Physics-augmented inverse network
   The same black-box inverse network, optionally initialized from the pure-data model.
   Loss = data supervision + terminal physics consistency.
   The terminal consistency uses the selected best trained forward PINN:
       predicted P -> frozen forward PINN at z=L -> terminal power
   and compares that terminal power with the received SSFM terminal waveform.

Important storage design
------------------------
The inverse dataset stores only what the inverse task needs:
    - final/terminal SSFM power waveform sampled at inverse_input_points;
    - the true initial normalized amplitudes and class labels.
It does NOT store full SSFM z-trajectories for every sample.  This keeps disk/RAM
usage small and does not affect inverse mapping, because both the black-box data
loss and the terminal physics loss only need y_L(t) and P.

For large M, arrays are stored as .npy memmaps, so training can read mini-batches
without loading the entire SSFM dataset into memory.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import time
import hashlib
from dataclasses import dataclass, fields
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset, DataLoader

from nlse import NLSEParams, PAM4_LEVELS, pulse_centers_t0
from ssfm import run_ssfm
from train_multi_pulse_pinn import load_forward_checkpoint, PDEParams

LEVELS = tuple(float(x) for x in PAM4_LEVELS)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def safe_device(device_text: str) -> torch.device:
    return torch.device(device_text if (torch.cuda.is_available() or device_text == "cpu") else "cpu")


def combo_to_text(combo: Sequence[float]) -> str:
    return ";".join(f"{float(x):g}" for x in combo)


def all_legal_pam4_candidates(n_pulses: int) -> np.ndarray:
    return np.asarray(list(product(LEVELS, repeat=int(n_pulses))), dtype=np.float32)


def powers_to_class_indices(powers: np.ndarray) -> np.ndarray:
    powers = np.asarray(powers, dtype=np.float32)
    levels = np.asarray(LEVELS, dtype=np.float32)
    idx = np.argmin(np.abs(powers[..., None] - levels.reshape(1, -1)), axis=-1).astype(np.uint8)
    if powers.ndim == 2:
        reconstructed = levels[idx]
        if float(np.max(np.abs(reconstructed - powers))) > 1e-5:
            raise ValueError("Found an amplitude outside the legal 4-PAM set.")
    return idx


def parse_bool_text(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def initial_power_from_levels(tau: np.ndarray, combo: Sequence[float], centers: Sequence[float]) -> np.ndarray:
    h = np.zeros_like(tau, dtype=np.float64)
    for amp, c in zip(combo, centers):
        h += float(amp) * np.exp(-((tau - float(c)) ** 2) / 2.0)
    return h ** 2


def interp_power_to_input_grid(tau_src: np.ndarray, power_src: np.ndarray, tau_input: np.ndarray) -> np.ndarray:
    return np.interp(tau_input, tau_src, power_src).astype(np.float32)


def interp_complex_field_to_input_grid(
    tau_src: np.ndarray, field_src: np.ndarray, tau_input: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a complex terminal field to the inverse input grid.

    The inverse dataset still stores only terminal slices, not the full SSFM
    trajectory.  Real and imaginary parts are saved separately as memmaps so
    downstream code can choose power-only or complex-field terminal losses.
    """
    field_src = np.asarray(field_src, dtype=np.complex128)
    real = np.interp(tau_input, tau_src, np.real(field_src)).astype(np.float32)
    imag = np.interp(tau_input, tau_src, np.imag(field_src)).astype(np.float32)
    return real, imag


def relative_power_error_np(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-300))


def select_forward_model(
    metrics_csv: Path,
    model_labels: list[str],
    model_paths: list[str],
    mean_threshold: float,
    p95_threshold: float,
    out_dir: Path,
) -> dict:
    import pandas as pd

    if not metrics_csv.exists():
        raise FileNotFoundError(f"forward metrics CSV not found: {metrics_csv}")
    df = pd.read_csv(metrics_csv)
    rows: list[dict] = []
    for label, path in zip(model_labels, model_paths):
        g = df[df["model"].astype(str) == str(label)]
        if g.empty:
            continue
        vals = g["rel_l2_power"].astype(float)
        rows.append({
            "label": str(label),
            "path": str(path),
            "mean_rel_l2_power": float(vals.mean()),
            "p95_rel_l2_power": float(vals.quantile(0.95)),
            "max_rel_l2_power": float(vals.max()),
            "n_rows": int(len(g)),
        })
    if not rows:
        raise RuntimeError("No model rows found in forward metrics CSV.")
    rows = sorted(rows, key=lambda r: (r["mean_rel_l2_power"], r["p95_rel_l2_power"]))
    selected = rows[0]
    gate_passed = selected["mean_rel_l2_power"] <= float(mean_threshold)
    if float(p95_threshold) > 0:
        gate_passed = gate_passed and selected["p95_rel_l2_power"] <= float(p95_threshold)
    payload = {
        "selected_model": selected,
        "forward_gate_passed": bool(gate_passed),
        "mean_threshold": float(mean_threshold),
        "p95_threshold": float(p95_threshold),
        "all_models": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selected_forward_for_inverse.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_pde_params_from_forward_checkpoint(path: str | Path) -> PDEParams:
    """Load the PDE coefficients used by the selected forward PINN checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    raw = dict(ckpt.get("pde_params", {}) or {})
    allowed = {field.name for field in PDEParams.__dataclass_fields__.values()}
    clean = {k: v for k, v in raw.items() if k in allowed}
    return PDEParams(**clean)


def sanitize_for_name(x: float | str) -> str:
    return str(x).replace(".", "p").replace("-", "m").replace("+", "p").replace(",", "_").replace(":", "_")


def _parse_selected_dataset_indices(value: object) -> list[int]:
    """Parse selected global SSFM combination indices for selected-only datasets."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in list(value)]
    text = str(value).strip()
    if not text:
        return []
    return [int(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if str(x).strip()]


def selected_indices_hash(indices: Sequence[int]) -> str:
    text = ",".join(str(int(x)) for x in indices)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def dataset_dir_name(args: argparse.Namespace) -> str:
    dtype = str(args.inverse_dataset_dtype).lower()
    observable = str(getattr(args, "terminal_observable", "power")).strip().lower()
    save_field = bool(getattr(args, "save_terminal_field", False)) or observable in {"complex", "field", "complex_field", "power_and_complex", "both"}
    save_xinit = bool(getattr(args, "save_initial_power", False))
    storage = "field" if save_field else "power"
    if save_xinit:
        storage += "_xinit"
    mode = str(getattr(args, "inverse_dataset_mode", "full")).strip().lower()
    mode_part = "full"
    if mode == "selected":
        indices = _parse_selected_dataset_indices(getattr(args, "selected_dataset_indices", ""))
        if indices:
            mode_part = f"selectedN{len(indices)}_h{selected_indices_hash(indices)}"
        else:
            mode_part = "selectedN0"
    return (
        f"inverse_ssfm_M{args.n_pulses}_win{args.t_window_t0:g}T0_nt{args.n_t}_nz{args.n_z}"
        f"_cmp{args.compare_t_min:g}_to_{args.compare_t_max:g}_Nin{args.inverse_input_points}_{storage}_{dtype}_{mode_part}"
    ).replace(".", "p").replace("-", "m")


def build_or_load_inverse_ssfm_dataset(args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict:
    """Build a memmapped inverse dataset from SSFM terminal slices.

    v8 adds two dataset modes:
      - full: generate all 4^M legal PAM4 combinations, as older versions did.
      - selected: generate only the selected global combination indices.  This is
        the recommended mode for the current inverse comparison with only
        --n-samples=10, especially for M=7/M=8.

    In both modes only terminal slices are saved.  No intermediate z-slices are
    stored; run_ssfm is called with save_every=n_z, so A contains only z=0 and z=L.
    """
    ds_root = Path(args.inverse_dataset_dir) if args.inverse_dataset_dir else out_dir / dataset_dir_name(args)
    ds_root.mkdir(parents=True, exist_ok=True)
    meta_path = ds_root / "meta.json"
    Y_path = ds_root / "Y_terminal_power.npy"
    Yr_path = ds_root / "Y_terminal_real.npy"
    Yi_path = ds_root / "Y_terminal_imag.npy"
    X_path = ds_root / "X_initial_power.npy"
    A_path = ds_root / "A_levels.npy"
    AC_path = ds_root / "A_class_indices.npy"
    AG_path = ds_root / "A_global_indices.npy"
    # Legacy aliases are also written so older analysis scripts remain usable.
    P_path = ds_root / "P_levels.npy"
    C_path = ds_root / "P_class_indices.npy"
    G_path = ds_root / "P_global_indices.npy"
    tau_path = ds_root / "tau_input.npy"
    completed_path = ds_root / "completed_mask.npy"

    M = int(args.n_pulses)
    legal_combos = all_legal_pam4_candidates(M)
    n_legal = int(len(legal_combos))
    mode = str(getattr(args, "inverse_dataset_mode", "full")).strip().lower()
    if mode not in {"full", "selected"}:
        raise ValueError(f"inverse_dataset_mode must be full or selected, got {mode!r}")

    if mode == "selected":
        global_indices = _parse_selected_dataset_indices(getattr(args, "selected_dataset_indices", ""))
        if not global_indices:
            raise ValueError("--inverse-dataset-mode selected requires --selected-dataset-indices internally. Use train_inverse... with --n-samples/--sample-indices rather than calling this builder directly.")
        bad = [i for i in global_indices if i < 0 or i >= n_legal]
        if bad:
            raise IndexError(f"selected global indices out of range for M={M}, 4^M={n_legal}: {bad[:10]}")
        all_combos = legal_combos[np.asarray(global_indices, dtype=np.int64)]
        n_all = int(len(all_combos))
        subset_note = f"selected-only global indices; n_selected={n_all}; source={getattr(args, 'selected_indices_source', '')}"
    else:
        all_combos = legal_combos
        global_indices = list(range(n_legal))
        n_all = n_legal
        if int(getattr(args, "inverse_max_combos", 0)) > 0 and n_all > int(args.inverse_max_combos):
            rng = np.random.default_rng(int(args.seed))
            idx = rng.choice(n_all, size=int(args.inverse_max_combos), replace=False)
            idx = np.sort(idx)
            all_combos = all_combos[idx]
            global_indices = [int(x) for x in idx.tolist()]
            n_all = int(len(all_combos))
            subset_note = f"random subset of legal combinations, max={args.inverse_max_combos}"
        else:
            subset_note = "all legal combinations"

    observable = str(getattr(args, "terminal_observable", "power")).strip().lower()
    save_terminal_field = bool(getattr(args, "save_terminal_field", False)) or observable in {"complex", "field", "complex_field", "power_and_complex", "both"}
    save_initial_power = bool(getattr(args, "save_initial_power", False))

    dtype = np.float16 if str(args.inverse_dataset_dtype).lower() == "float16" else np.float32
    tau_input = np.linspace(float(args.compare_t_min), float(args.compare_t_max), int(args.inverse_input_points), dtype=np.float32)
    expected_meta = {
        "version": 8,
        "purpose": "inverse terminal SSFM dataset; stores final terminal slices only, not full z-trajectories",
        "dataset_mode": mode,
        "terminal_storage": "power_plus_real_imag" if save_terminal_field else "power_only",
        "save_terminal_field": bool(save_terminal_field),
        "save_initial_power": bool(save_initial_power),
        "n_pulses": M,
        "levels": list(LEVELS),
        "level_quantity": "normalized_field_amplitude",
        "isolated_peak_power_levels": [float(x) ** 2 for x in LEVELS],
        "n_samples": n_all,
        "n_legal_combinations": n_legal,
        "subset_note": subset_note,
        "selected_global_indices": [int(x) for x in global_indices],
        "selected_hash": selected_indices_hash(global_indices) if mode == "selected" else "",
        "inverse_input_points": int(args.inverse_input_points),
        "dtype": np.dtype(dtype).name,
        "t_window_t0": float(args.t_window_t0),
        "n_t": int(args.n_t),
        "n_z": int(args.n_z),
        "z_max_ld": float(args.z_max_ld),
        "compare_t_min": float(args.compare_t_min),
        "compare_t_max": float(args.compare_t_max),
        "seed": int(args.seed),
    }

    required_for_reuse = [Y_path, P_path, C_path, G_path, tau_path, meta_path, completed_path]
    if save_terminal_field:
        required_for_reuse += [Yr_path, Yi_path]
    if save_initial_power:
        required_for_reuse += [X_path]

    if bool(getattr(args, "reuse_inverse_dataset", True)) and all(p.exists() for p in required_for_reuse):
        old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        comparable_keys = [
            "dataset_mode", "n_pulses", "n_samples", "n_legal_combinations", "inverse_input_points", "dtype",
            "t_window_t0", "n_t", "n_z", "z_max_ld", "compare_t_min", "compare_t_max",
            "save_terminal_field", "save_initial_power", "selected_hash",
        ]
        complete_ok = True
        try:
            cm = np.load(completed_path, mmap_mode="r")
            complete_ok = int(np.sum(cm)) == int(old_meta.get("n_samples", len(cm)))
        except Exception:
            complete_ok = False
        if complete_ok and all(old_meta.get(k) == expected_meta.get(k) for k in comparable_keys):
            print(f"Reusing inverse SSFM dataset -> {ds_root}")
            return {"root": str(ds_root), "meta": old_meta}
        print("Existing inverse dataset is incomplete or metadata does not match current settings; rebuilding.")

    print("\n========== Building inverse SSFM terminal dataset ==========")
    print(f"dataset -> {ds_root}")
    print(f"mode    -> {mode}")
    print(f"samples -> {n_all} ({subset_note})")
    print(f"stored  -> terminal power{' + terminal complex field' if save_terminal_field else ''}{' + initial power' if save_initial_power else ''}, dtype={np.dtype(dtype).name}")
    print("note    -> no intermediate z-slices are saved")

    Y = open_memmap(Y_path, mode="w+", dtype=dtype, shape=(n_all, int(args.inverse_input_points)))
    Yr = open_memmap(Yr_path, mode="w+", dtype=dtype, shape=(n_all, int(args.inverse_input_points))) if save_terminal_field else None
    Yi = open_memmap(Yi_path, mode="w+", dtype=dtype, shape=(n_all, int(args.inverse_input_points))) if save_terminal_field else None
    X = open_memmap(X_path, mode="w+", dtype=dtype, shape=(n_all, int(args.inverse_input_points))) if save_initial_power else None
    P = open_memmap(P_path, mode="w+", dtype=np.float32, shape=(n_all, M))
    C = open_memmap(C_path, mode="w+", dtype=np.uint8, shape=(n_all, M))
    G = open_memmap(G_path, mode="w+", dtype=np.int64, shape=(n_all,))
    completed = open_memmap(completed_path, mode="w+", dtype=np.bool_, shape=(n_all,))
    completed[:] = False
    np.save(tau_path, tau_input)

    P[:] = all_combos.astype(np.float32)
    C[:] = powers_to_class_indices(all_combos)
    G[:] = np.asarray(global_indices, dtype=np.int64)
    P.flush(); C.flush(); G.flush(); completed.flush(); Y.flush()
    if Yr is not None: Yr.flush()
    if Yi is not None: Yi.flush()
    if X is not None: X.flush()

    centers = pulse_centers_t0(M)
    t0_all = time.time()
    for i, combo in enumerate(all_combos):
        params = NLSEParams.paper_pam4(
            z_max_ld=float(args.z_max_ld),
            t_window_t0=float(args.t_window_t0),
            n_t=int(args.n_t),
            n_z=int(args.n_z),
        ).with_multi_pulse(tuple(float(x) for x in combo))
        t0 = time.time()
        z_phys, t_ps, A = run_ssfm(params, device=str(device), save_every=params.n_z, quiet=args.quiet_ssfm)
        tau_full = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
        h_final = np.asarray(A[-1], dtype=np.complex128)
        power = np.abs(h_final) ** 2
        Y[i, :] = interp_power_to_input_grid(tau_full, power, tau_input).astype(dtype)
        if save_terminal_field:
            real_i, imag_i = interp_complex_field_to_input_grid(tau_full, h_final, tau_input)
            Yr[i, :] = real_i.astype(dtype)  # type: ignore[index]
            Yi[i, :] = imag_i.astype(dtype)  # type: ignore[index]
        if save_initial_power:
            X[i, :] = initial_power_from_levels(tau_input, combo, centers).astype(dtype)  # type: ignore[index]
        completed[i] = True
        if (i + 1) % max(1, int(args.inverse_dataset_log_every)) == 0 or i == 0 or i == n_all - 1:
            print(f"[{i+1}/{n_all}] global_idx={int(global_indices[i])} combo={combo_to_text(combo)} ssfm={time.time()-t0:.2f}s elapsed={time.time()-t0_all:.1f}s")
        if (i + 1) % 16 == 0:
            Y.flush(); completed.flush(); G.flush()
            if Yr is not None: Yr.flush()
            if Yi is not None: Yi.flush()
            if X is not None: X.flush()
        del A, h_final, power
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Y.flush(); P.flush(); C.flush(); G.flush(); completed.flush()
    if Yr is not None: Yr.flush()
    if Yi is not None: Yi.flush()
    if X is not None: X.flush()
    expected_meta["build_elapsed_sec"] = float(time.time() - t0_all)
    expected_meta["Y_terminal_power_path"] = str(Y_path)
    if save_terminal_field:
        expected_meta["Y_terminal_real_path"] = str(Yr_path)
        expected_meta["Y_terminal_imag_path"] = str(Yi_path)
    if save_initial_power:
        expected_meta["X_initial_power_path"] = str(X_path)
    # Write explicit amplitude-named copies for clear downstream use.
    np.save(A_path, np.asarray(P, dtype=np.float32))
    np.save(AC_path, np.asarray(C, dtype=np.uint8))
    np.save(AG_path, np.asarray(G, dtype=np.int64))
    expected_meta["A_levels_path"] = str(A_path)
    expected_meta["A_class_indices_path"] = str(AC_path)
    expected_meta["A_global_indices_path"] = str(AG_path)
    expected_meta["P_levels_path_legacy_alias"] = str(P_path)
    expected_meta["P_class_indices_path_legacy_alias"] = str(C_path)
    expected_meta["P_global_indices_path_legacy_alias"] = str(G_path)
    expected_meta["tau_input_path"] = str(tau_path)
    expected_meta["completed_count"] = int(np.sum(np.asarray(completed)))
    meta_path.write_text(json.dumps(expected_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("inverse SSFM dataset ready ->", ds_root)
    return {"root": str(ds_root), "meta": expected_meta}

def make_or_load_split(ds_root: Path, seed: int, train_ratio: float, max_train: int = 0, max_test: int = 0) -> dict:
    split_path = ds_root / f"split_train{train_ratio:g}_seed{seed}_maxtr{max_train}_maxte{max_test}.json".replace(".", "p")
    P = np.load(ds_root / "P_levels.npy", mmap_mode="r")
    n = int(P.shape[0])
    if split_path.exists():
        return json.loads(split_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_train = int(round(float(train_ratio) * n))
    n_train = max(1, min(n_train, n - 1))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if int(max_train) > 0:
        train_idx = train_idx[:int(max_train)]
    if int(max_test) > 0:
        test_idx = test_idx[:int(max_test)]
    payload = {
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "n_total": int(n),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_indices": [int(x) for x in train_idx.tolist()],
        "test_indices": [int(x) for x in test_idx.tolist()],
        "note": "Default split is 50 percent train / 50 percent test. This inverse split is independent of the forward PINN seen/unseen split r_F.",
    }
    split_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


class InverseMemmapDataset(Dataset):
    def __init__(self, ds_root: Path, indices: Sequence[int]) -> None:
        self.ds_root = Path(ds_root)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.Y = np.load(self.ds_root / "Y_terminal_power.npy", mmap_mode="r")
        self.P = np.load(self.ds_root / "P_levels.npy", mmap_mode="r")
        self.C = np.load(self.ds_root / "P_class_indices.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, k: int):
        i = int(self.indices[int(k)])
        return {
            "idx": np.int64(i),
            # copy() avoids a PyTorch warning about read-only memmap views.
            "y": np.asarray(self.Y[i], dtype=np.float32).copy(),
            "powers": np.asarray(self.P[i], dtype=np.float32).copy(),
            "classes": np.asarray(self.C[i], dtype=np.int64).copy(),
        }


class InverseMLP(nn.Module):
    """Black-box inverse mapper: terminal power waveform -> M PAM4 class logits."""
    def __init__(self, n_pulses: int, n_input_points: int, hidden: int = 256, layers: int = 4, y_scale: float = 1.0) -> None:
        super().__init__()
        self.n_pulses = int(n_pulses)
        self.n_input_points = int(n_input_points)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.y_scale = float(max(y_scale, 1e-12))
        mods: list[nn.Module] = []
        in_dim = self.n_input_points
        for layer in range(int(layers)):
            mods.append(nn.Linear(in_dim if layer == 0 else int(hidden), int(hidden)))
            mods.append(nn.GELU())
        mods.append(nn.Linear(int(hidden), self.n_pulses * len(LEVELS)))
        self.net = nn.Sequential(*mods)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y / self.y_scale
        return self.net(y).reshape(y.shape[0], self.n_pulses, len(LEVELS))

    def soft_powers(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        levels = torch.tensor(LEVELS, dtype=logits.dtype, device=logits.device).reshape(1, 1, -1)
        probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        return torch.sum(probs * levels, dim=-1)

    def differentiable_powers(self, logits: torch.Tensor, mode: str = "straight_through", temperature: float = 1.0) -> torch.Tensor:
        """Map class logits to legal PAM4 powers for physics losses.

        mode="straight_through" uses a hard one-hot PAM4 value in the forward pass
        while preserving softmax gradients. This keeps the predicted powers legal
        for terminal/NLSE losses. mode="soft" is smoother but the training-time
        powers are convex combinations of legal levels.
        """
        levels = torch.tensor(LEVELS, dtype=logits.dtype, device=logits.device).reshape(1, 1, -1)
        probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        mode = str(mode).strip().lower()
        if mode in {"straight_through", "st", "hard"}:
            hard_idx = torch.argmax(probs, dim=-1)
            hard = torch.nn.functional.one_hot(hard_idx, num_classes=len(LEVELS)).to(dtype=probs.dtype, device=probs.device)
            probs_used = hard + probs - probs.detach()
        elif mode in {"soft", "softmax"}:
            probs_used = probs
        else:
            raise ValueError("power-map-mode must be 'straight_through' or 'soft'.")
        return torch.sum(probs_used * levels, dim=-1)

    def hard_powers(self, logits: torch.Tensor) -> torch.Tensor:
        levels = torch.tensor(LEVELS, dtype=logits.dtype, device=logits.device)
        idx = torch.argmax(logits, dim=-1)
        return levels[idx]

    def config(self) -> dict:
        return {
            "model_type": "InverseMLP_terminal_waveform_to_PAM4_power_classes",
            "n_pulses": self.n_pulses,
            "n_input_points": self.n_input_points,
            "hidden": self.hidden,
            "layers": self.layers,
            "y_scale": self.y_scale,
            "levels": list(LEVELS),
        "level_quantity": "normalized_field_amplitude",
        "isolated_peak_power_levels": [float(x) ** 2 for x in LEVELS],
        }


def cross_entropy_pam4(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, len(LEVELS)), targets.reshape(-1))


def terminal_power_forward_model(
    forward_model: nn.Module,
    tau: torch.Tensor,
    powers: torch.Tensor,
    zeta: float,
    chunk_t: int = 512,
) -> torch.Tensor:
    """Differentiable terminal power through frozen forward PINN.

    forward_model parameters are frozen, but gradients still flow to `powers`.
    """
    B = int(powers.shape[0])
    outs: list[torch.Tensor] = []
    for start in range(0, int(tau.numel()), int(chunk_t)):
        end = min(int(tau.numel()), start + int(chunk_t))
        t_chunk = tau[start:end]
        T = int(t_chunk.numel())
        z = torch.full((B * T, 1), float(zeta), dtype=powers.dtype, device=powers.device)
        t = t_chunk.reshape(1, T, 1).expand(B, T, 1).reshape(B * T, 1).to(dtype=powers.dtype, device=powers.device)
        p = powers.reshape(B, 1, -1).expand(B, T, -1).reshape(B * T, -1)
        u, v = forward_model(z, t, p)
        outs.append((u.reshape(B, T) ** 2 + v.reshape(B, T) ** 2))
    return torch.cat(outs, dim=1)


def terminal_relative_mse(pred_power: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
    """Relative MSE used for terminal consistency during inverse training."""
    num = torch.sum((pred_power - y_ref) ** 2, dim=1)
    den = torch.sum(y_ref ** 2, dim=1) + 1e-12
    return torch.mean(num / den)


def pde_residual_for_inverse(
    forward_model: nn.Module,
    z: torch.Tensor,
    t: torch.Tensor,
    powers: torch.Tensor,
    params: PDEParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """NLSE residual of the frozen forward PINN, keeping gradients to powers.

    The forward PINN parameters are frozen, but the residual remains
    differentiable with respect to `powers`, so this loss can update the
    inverse network that generated those powers.
    """
    z = z.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    u, v = forward_model(z, t, powers)
    ones = torch.ones_like(u)
    u_z = torch.autograd.grad(u, z, ones, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z, ones, create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, ones, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, ones, create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, ones, create_graph=True, retain_graph=True)[0]
    v_tt = torch.autograd.grad(v_t, t, ones, create_graph=True, retain_graph=True)[0]

    r2 = u ** 2 + v ** 2
    disp_f = -(params.beta2_norm / 2.0) * v_tt
    disp_g = (params.beta2_norm / 2.0) * u_tt

    if params.has_tod:
        u_ttt = torch.autograd.grad(u_tt, t, ones, create_graph=True, retain_graph=True)[0]
        v_ttt = torch.autograd.grad(v_tt, t, ones, create_graph=True, retain_graph=True)[0]
        disp_f = disp_f - params.beta3_norm / 6.0 * u_ttt
        disp_g = disp_g - params.beta3_norm / 6.0 * v_ttt

    nl_f = params.N_sq * r2 * v
    nl_g = -params.N_sq * r2 * u

    P_t = None
    if params.has_ss:
        P_t = torch.autograd.grad(r2, t, ones, create_graph=True, retain_graph=True)[0]
        ss = params.s * params.ss_coef
        nl_f = nl_f + ss * (P_t * u + r2 * u_t)
        nl_g = nl_g + ss * (P_t * v + r2 * v_t)

    if params.has_irs:
        if P_t is None:
            P_t = torch.autograd.grad(r2, t, ones, create_graph=True, retain_graph=True)[0]
        nl_f = nl_f - params.N_sq * params.tau_R * P_t * v
        nl_g = nl_g + params.N_sq * params.tau_R * P_t * u

    f = u_z + (params.alpha_norm / 2.0) * u + disp_f + nl_f
    g = v_z + (params.alpha_norm / 2.0) * v + disp_g + nl_g
    return f, g


def nlse_loss_forward_pinn(
    forward_model: nn.Module,
    pde_params: PDEParams,
    powers: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Compute mean squared NLSE residual at random collocation points."""
    points_per_sample = int(args.nlse_points_per_sample)
    if points_per_sample <= 0:
        return torch.tensor(0.0, dtype=powers.dtype, device=powers.device)
    B = int(powers.shape[0])
    K = points_per_sample
    z = torch.rand((B, K, 1), dtype=powers.dtype, device=powers.device) * float(args.z_max_ld)
    t_min = float(args.nlse_t_min) if str(args.nlse_t_min) != "" else float(args.compare_t_min)
    t_max = float(args.nlse_t_max) if str(args.nlse_t_max) != "" else float(args.compare_t_max)
    t = t_min + (t_max - t_min) * torch.rand((B, K, 1), dtype=powers.dtype, device=powers.device)
    P_rep = powers.reshape(B, 1, -1).expand(B, K, -1).reshape(B * K, -1)
    f, g = pde_residual_for_inverse(forward_model, z.reshape(B * K, 1), t.reshape(B * K, 1), P_rep, pde_params)
    return (f ** 2).mean() + (g ** 2).mean()


@dataclass
class TrainResult:
    model: InverseMLP
    summary: dict


def train_inverse_model(
    method: str,
    model: InverseMLP,
    train_ds: InverseMemmapDataset,
    test_ds: InverseMemmapDataset,
    tau_input: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    forward_model: nn.Module | None = None,
    pde_params: PDEParams | None = None,
    init_state_path: Path | None = None,
) -> TrainResult:
    """Train one inverse method.

    Methods:
        pure_data     : data CE only.
        hybrid        : data CE + terminal consistency + NLSE residual.
        pure_physics  : terminal consistency + NLSE residual, no true-P label in loss.
    """
    method = str(method)
    if method not in {"pure_data", "hybrid", "pure_physics"}:
        raise ValueError("method must be pure_data, hybrid, or pure_physics")
    method_dir = out_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    if init_state_path is not None and init_state_path.exists():
        ckpt = torch.load(init_state_path, map_location=device)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        print(f"[{method}] initialized from {init_state_path}")

    train_loader = DataLoader(train_ds, batch_size=int(args.inverse_batch_size), shuffle=True, num_workers=int(args.inverse_num_workers), pin_memory=torch.cuda.is_available())
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.inverse_lr), weight_decay=float(args.inverse_weight_decay))
    tau_t = torch.tensor(tau_input, dtype=torch.float32, device=device)
    history: list[dict] = []
    best_score = float("inf")
    best_state = None
    patience = 0
    t0_all = time.time()

    use_data = method in {"pure_data", "hybrid"}
    use_physics = method in {"hybrid", "pure_physics"}
    if use_physics and forward_model is None:
        raise RuntimeError(f"{method} requires a selected frozen forward PINN model")
    if use_physics and pde_params is None:
        raise RuntimeError(f"{method} requires pde_params from the selected forward checkpoint")

    print(f"\n========== Train {method} ==========")
    print(f"train/test = {len(train_ds)}/{len(test_ds)}")
    print(f"loss weights: data={float(args.data_loss_weight):g}, terminal={float(args.terminal_loss_weight):g}, nlse={float(args.nlse_loss_weight):g}")
    print(f"power map  : {args.power_map_mode}, temperature={float(args.power_temperature):g}")

    for epoch in range(int(args.inverse_epochs)):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "terminal": 0.0, "nlse": 0.0, "n": 0}
        for batch in train_loader:
            y = batch["y"].to(device=device, dtype=torch.float32, non_blocking=True)
            target = batch["classes"].to(device=device, dtype=torch.long, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(y)

            # CE is always computed for logging. It is included in the gradient
            # only for pure_data and hybrid; pure_physics does not use labels.
            ce = cross_entropy_pam4(logits, target)
            terminal = torch.tensor(0.0, dtype=torch.float32, device=device)
            nlse = torch.tensor(0.0, dtype=torch.float32, device=device)

            if use_physics:
                pred_p = model.differentiable_powers(logits, mode=str(args.power_map_mode), temperature=float(args.power_temperature))
                if float(args.terminal_loss_weight) != 0.0:
                    pred_terminal = terminal_power_forward_model(forward_model, tau_t, pred_p, float(args.z_max_ld), chunk_t=int(args.forward_time_chunk))
                    terminal = terminal_relative_mse(pred_terminal, y)
                if float(args.nlse_loss_weight) != 0.0 and int(args.nlse_points_per_sample) > 0:
                    nlse = nlse_loss_forward_pinn(forward_model, pde_params, pred_p, args)

            loss = torch.tensor(0.0, dtype=torch.float32, device=device)
            if use_data:
                loss = loss + float(args.data_loss_weight) * ce
            if use_physics:
                loss = loss + float(args.terminal_loss_weight) * terminal + float(args.nlse_loss_weight) * nlse

            loss.backward()
            if float(args.inverse_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.inverse_grad_clip))
            opt.step()
            B = int(y.shape[0])
            sums["loss"] += float(loss.detach().cpu()) * B
            sums["ce"] += float(ce.detach().cpu()) * B
            sums["terminal"] += float(terminal.detach().cpu()) * B
            sums["nlse"] += float(nlse.detach().cpu()) * B
            sums["n"] += B

        row = {
            "epoch": int(epoch),
            "loss": sums["loss"] / max(1, sums["n"]),
            "ce": sums["ce"] / max(1, sums["n"]),
            "terminal": sums["terminal"] / max(1, sums["n"]),
            "nlse": sums["nlse"] / max(1, sums["n"]),
            "elapsed_sec": float(time.time() - t0_all),
        }
        history.append(row)
        # Score by the loss that actually trains the model. This keeps early stopping simple.
        score = row["loss"]
        if score < best_score - 1e-8:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % int(args.inverse_log_every) == 0 or epoch == int(args.inverse_epochs) - 1:
            print(
                f"{method:13s} epoch {epoch:5d} | loss={row['loss']:.4e} | "
                f"ce={row['ce']:.4e} | terminal={row['terminal']:.4e} | nlse={row['nlse']:.4e} | {row['elapsed_sec']:.1f}s"
            )
        if epoch >= int(args.inverse_min_epochs) and patience >= int(args.inverse_early_stop_patience):
            print(f"[{method}] early stop at epoch {epoch}; best loss={best_score:.4e}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist_path = method_dir / f"{method}_loss_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss", "ce", "terminal", "nlse", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(history)
    plot_loss_curve(hist_path, method_dir / f"{method}_loss_curve.png", title=f"{method} inverse training loss")

    ckpt_path = method_dir / f"{method}_inverse_model.pt"
    torch.save({
        "model_config": model.config(),
        "model_state": model.state_dict(),
        "method": method,
        "levels": list(LEVELS),
        "level_quantity": "normalized_field_amplitude",
        "isolated_peak_power_levels": [float(x) ** 2 for x in LEVELS],
        "history": history,
    }, ckpt_path)

    summary = evaluate_inverse_model(method, model, test_ds, tau_input, args, device, method_dir, forward_model=forward_model)
    summary.update({
        "method": method,
        "checkpoint": str(ckpt_path),
        "loss_history_csv": str(hist_path),
        "train_size": int(len(train_ds)),
        "test_size": int(len(test_ds)),
        "data_loss_weight": float(args.data_loss_weight if use_data else 0.0),
        "terminal_loss_weight": float(args.terminal_loss_weight if use_physics else 0.0),
        "nlse_loss_weight": float(args.nlse_loss_weight if use_physics else 0.0),
        "nlse_points_per_sample": int(args.nlse_points_per_sample if use_physics else 0),
        "power_map_mode": str(args.power_map_mode),
        "power_temperature": float(args.power_temperature),
    })
    (method_dir / f"{method}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return TrainResult(model=model, summary=summary)

def evaluate_inverse_model(
    method: str,
    model: InverseMLP,
    test_ds: InverseMemmapDataset,
    tau_input: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    method_dir: Path,
    forward_model: nn.Module | None = None,
) -> dict:
    loader = DataLoader(test_ds, batch_size=int(args.inverse_eval_batch_size), shuffle=False, num_workers=int(args.inverse_num_workers), pin_memory=torch.cuda.is_available())
    model.eval()
    tau_t = torch.tensor(tau_input, dtype=torch.float32, device=device)
    rows: list[dict] = []
    n_total = 0
    n_symbol_correct = 0
    n_symbols = 0
    n_exact = 0
    ce_vals = []
    terminal_vals = []
    terminal_examples: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            idx = batch["idx"].cpu().numpy().astype(int)
            y = batch["y"].to(device=device, dtype=torch.float32)
            target = batch["classes"].to(device=device, dtype=torch.long)
            p_true = batch["powers"].cpu().numpy().astype(float)
            logits = model(y)
            ce = cross_entropy_pam4(logits, target)
            pred_cls = torch.argmax(logits, dim=-1)
            levels_t = torch.tensor(LEVELS, dtype=torch.float32, device=device)
            p_pred = levels_t[pred_cls].cpu().numpy().astype(float)
            cls_true_np = target.cpu().numpy().astype(int)
            cls_pred_np = pred_cls.cpu().numpy().astype(int)
            correct = (cls_true_np == cls_pred_np)
            exact = np.all(correct, axis=1)
            n_total += int(len(idx))
            n_symbol_correct += int(np.sum(correct))
            n_symbols += int(np.size(correct))
            n_exact += int(np.sum(exact))
            ce_vals.append(float(ce.detach().cpu()))

            term_rel = np.full(len(idx), np.nan, dtype=float)
            if forward_model is not None:
                p_pred_t = torch.tensor(p_pred, dtype=torch.float32, device=device)
                pred_terminal = terminal_power_forward_model(forward_model, tau_t, p_pred_t, float(args.z_max_ld), chunk_t=int(args.forward_time_chunk))
                y_np = y.detach().cpu().numpy()
                pt_np = pred_terminal.detach().cpu().numpy()
                for j in range(len(idx)):
                    term_rel[j] = relative_power_error_np(pt_np[j], y_np[j])
                terminal_vals.extend([float(x) for x in term_rel if np.isfinite(x)])
                for j in range(len(idx)):
                    if len(terminal_examples) < int(args.inverse_plot_examples):
                        terminal_examples.append({
                            "idx": int(idx[j]),
                            "y_obs": y_np[j].astype(float),
                            "y_pred": pt_np[j].astype(float),
                            "true_levels": combo_to_text(p_true[j]),
                            "pred_levels": combo_to_text(p_pred[j]),
                        })

            for j in range(len(idx)):
                rows.append({
                    "idx": int(idx[j]),
                    "true_levels": combo_to_text(p_true[j]),
                    "pred_levels": combo_to_text(p_pred[j]),
                    "true_classes": ";".join(str(int(x)) for x in cls_true_np[j]),
                    "pred_classes": ";".join(str(int(x)) for x in cls_pred_np[j]),
                    "exact_match": int(bool(exact[j])),
                    "per_pulse_accuracy": float(np.mean(correct[j])),
                    "terminal_rel_l2_power_forward_check": float(term_rel[j]) if np.isfinite(term_rel[j]) else "",
                })

    pred_csv = method_dir / f"{method}_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["idx", "true_levels", "pred_levels", "true_classes", "pred_classes", "exact_match", "per_pulse_accuracy", "terminal_rel_l2_power_forward_check"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_test": int(n_total),
        "all_pulses_accuracy": float(n_exact / max(1, n_total)),
        "per_pulse_accuracy": float(n_symbol_correct / max(1, n_symbols)),
        "ce_mean": float(np.mean(ce_vals)) if ce_vals else None,
        "terminal_rel_l2_power_mean_forward_check": float(np.mean(terminal_vals)) if terminal_vals else None,
        "terminal_rel_l2_power_p95_forward_check": float(np.quantile(terminal_vals, 0.95)) if terminal_vals else None,
        "predictions_csv": str(pred_csv),
    }
    plot_initial_examples(method, rows, tau_input, args.n_pulses, method_dir / f"{method}_initial_reconstruction_examples.png", n_examples=int(args.inverse_plot_examples))
    plot_terminal_examples(method, terminal_examples, tau_input, method_dir / f"{method}_terminal_consistency_examples.png")
    return summary


def plot_loss_curve(csv_path: Path, out_path: Path, title: str) -> None:
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["loss"], label="total")
    if "ce" in df.columns:
        ax.plot(df["epoch"], df["ce"], label="data CE")
    if "terminal" in df.columns and float(df["terminal"].abs().sum()) > 0:
        ax.plot(df["epoch"], df["terminal"], label="terminal")
    if "nlse" in df.columns and float(df["nlse"].abs().sum()) > 0:
        ax.plot(df["epoch"], df["nlse"], label="NLSE")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_initial_examples(method: str, rows: list[dict], tau: np.ndarray, n_pulses: int, out_path: Path, n_examples: int = 4) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    centers = pulse_centers_t0(n_pulses)
    chosen = rows[:max(1, min(int(n_examples), len(rows)))]
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, row in enumerate(chosen):
        true = [float(x) for x in str(row["true_levels"]).split(";")]
        pred = [float(x) for x in str(row["pred_levels"]).split(";")]
        y_true = initial_power_from_levels(tau, true, centers)
        y_pred = initial_power_from_levels(tau, pred, centers)
        ax.plot(tau, y_true + 1.25 * k, linewidth=1.2, label=f"true idx={row['idx']}" if k == 0 else None)
        ax.plot(tau, y_pred + 1.25 * k, linestyle="--", linewidth=1.2, label=f"pred ({method})" if k == 0 else None)
    ax.set_xlabel(r"$t/T_0$")
    ax.set_ylabel("initial power + offset")
    ax.set_title(f"{method}: predicted vs true initial powers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_examples(method: str, examples: list[dict], tau: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not examples:
        return
    n = len(examples)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.4 * n)), squeeze=False)
    for k, ex in enumerate(examples):
        ax = axes[k, 0]
        ax.plot(tau, ex["y_obs"], label="observed terminal", linewidth=1.2)
        ax.plot(tau, ex["y_pred"], linestyle="--", label="forward(P_pred)", linewidth=1.2)
        ax.set_ylabel("power")
        ax.set_title(f"idx={ex['idx']} true={ex['true_levels']} pred={ex['pred_levels']}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8)
    axes[-1, 0].set_xlabel(r"$t/T_0$")
    fig.suptitle(f"{method}: terminal waveform consistency examples", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(out_dir: Path, summaries: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    methods = []
    exact = []
    symbol = []
    terminal = []
    for key in ["pure_data", "hybrid", "pure_physics"]:
        if key in summaries and summaries[key] and not summaries[key].get("skipped"):
            methods.append(key)
            exact.append(float(summaries[key].get("all_pulses_accuracy", 0.0)))
            symbol.append(float(summaries[key].get("per_pulse_accuracy", 0.0)))
            val = summaries[key].get("terminal_rel_l2_power_mean_forward_check")
            terminal.append(np.nan if val is None else float(val))
    if not methods:
        return
    x = np.arange(len(methods), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(x - width/2, exact, width, label="all pulses exact")
    ax.bar(x + width/2, symbol, width, label="per pulse")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("Inverse mapping accuracy comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "inverse_accuracy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not np.all(np.isnan(terminal)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(methods, terminal)
        ax.set_ylabel("mean terminal rel-L2 power error")
        ax.set_title("Forward-PINN terminal consistency of predicted P")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "inverse_terminal_consistency_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_three_method_initial_comparison(out_dir: Path, tau: np.ndarray, n_pulses: int, n_examples: int = 4) -> None:
    """Overlay true and predicted initial waveforms from all inverse methods."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    method_csvs = {
        "pure_data": out_dir / "pure_data" / "pure_data_predictions.csv",
        "hybrid": out_dir / "hybrid" / "hybrid_predictions.csv",
        "pure_physics": out_dir / "pure_physics" / "pure_physics_predictions.csv",
    }
    dfs = {m: pd.read_csv(path) for m, path in method_csvs.items() if path.exists()}
    if not dfs:
        return
    first_method = next(iter(dfs))
    base = dfs[first_method]
    if base.empty:
        return
    chosen_idx = base["idx"].astype(int).head(max(1, int(n_examples))).tolist()
    centers = pulse_centers_t0(n_pulses)
    fig, ax = plt.subplots(figsize=(11, 6))
    for row_i, idx in enumerate(chosen_idx):
        offset = 1.35 * row_i
        # True curve from the first method's row.
        row0 = base[base["idx"].astype(int) == int(idx)].iloc[0]
        true = [float(x) for x in str(row0["true_levels"]).split(";")]
        y_true = initial_power_from_levels(tau, true, centers)
        ax.plot(tau, y_true + offset, linewidth=1.8, label="true" if row_i == 0 else None)
        for method, df in dfs.items():
            sub = df[df["idx"].astype(int) == int(idx)]
            if sub.empty:
                continue
            pred = [float(x) for x in str(sub.iloc[0]["pred_levels"]).split(";")]
            y_pred = initial_power_from_levels(tau, pred, centers)
            ax.plot(tau, y_pred + offset, linestyle="--", linewidth=1.1, label=method if row_i == 0 else None)
    ax.set_xlabel(r"$t/T_0$")
    ax.set_ylabel("initial power + offset")
    ax.set_title("Three inverse schemes: initial waveform reconstruction examples")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / "inverse_three_method_initial_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data-only and forward-PINN physics-augmented inverse mapping for fixed-M PAM4 multi-pulses.")
    p.add_argument("--n-pulses", "-M", type=int, required=True)
    p.add_argument("--model-labels", nargs="+", required=True)
    p.add_argument("--model-paths", nargs="+", required=True)
    p.add_argument("--forward-metrics-csv", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # SSFM grid and observation grid.
    p.add_argument("--t-window-t0", type=float, required=True)
    p.add_argument("--n-t", type=int, required=True)
    p.add_argument("--n-z", type=int, required=True)
    p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--compare-t-min", type=float, required=True)
    p.add_argument("--compare-t-max", type=float, required=True)
    p.add_argument("--quiet-ssfm", action="store_true", default=True)
    p.add_argument("--verbose-ssfm", action="store_false", dest="quiet_ssfm")

    # Inverse dataset.
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--inverse-dataset-dir", type=str, default="")
    p.add_argument("--reuse-inverse-dataset", action="store_true", default=True)
    p.add_argument("--rebuild-inverse-dataset", action="store_false", dest="reuse_inverse_dataset")
    p.add_argument("--inverse-dataset-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--inverse-dataset-log-every", type=int, default=20)
    p.add_argument("--inverse-max-combos", type=int, default=0, help="0 means all legal combinations. Use >0 only for very large M debugging.")
    p.add_argument("--inverse-train-ratio", type=float, default=0.50, help="Inverse train/test split. Default 0.50 means 50 percent train / 50 percent test. Ignored when --inverse-split-json is provided.")
    p.add_argument("--inverse-split-json", type=str, default="", help="Optional JSON file containing explicit train_indices and test_indices. Use this for fixed-test/nested ratio searches.")
    p.add_argument("--max-inverse-train-samples", type=int, default=0)
    p.add_argument("--max-inverse-test-samples", type=int, default=0)

    # Selected forward model for physics-augmented inverse.
    p.add_argument("--forward-mean-threshold", type=float, default=0.05)
    p.add_argument("--forward-p95-threshold", type=float, default=0.10)
    p.add_argument("--force-physics-inverse", action="store_true", help="Run hybrid physics inverse even if selected forward model fails the threshold gate.")

    # Inverse model and training.
    p.add_argument("--inverse-hidden", type=int, default=256)
    p.add_argument("--inverse-layers", type=int, default=4)
    p.add_argument("--inverse-epochs", type=int, default=1000)
    p.add_argument("--inverse-min-epochs", type=int, default=100)
    p.add_argument("--inverse-early-stop-patience", type=int, default=150)
    p.add_argument("--inverse-batch-size", type=int, default=32)
    p.add_argument("--inverse-eval-batch-size", type=int, default=128)
    p.add_argument("--inverse-lr", type=float, default=1e-3)
    p.add_argument("--inverse-weight-decay", type=float, default=1e-6)
    p.add_argument("--inverse-grad-clip", type=float, default=1.0)
    p.add_argument("--inverse-log-every", type=int, default=50)
    p.add_argument("--inverse-num-workers", type=int, default=0)
    p.add_argument("--inverse-plot-examples", type=int, default=4)

    # Loss weights. Defaults are deliberately simple; optimize in separate script.
    p.add_argument("--data-loss-weight", type=float, default=1.0)
    p.add_argument("--terminal-loss-weight", type=float, default=1.0)
    p.add_argument("--nlse-loss-weight", type=float, default=0.01)
    p.add_argument("--nlse-points-per-sample", type=int, default=16, help="Random PDE residual points per inverse training sample. Set 0 to disable NLSE loss.")
    p.add_argument("--nlse-t-min", type=str, default="", help="Optional NLSE residual t-min; default uses compare-t-min.")
    p.add_argument("--nlse-t-max", type=str, default="", help="Optional NLSE residual t-max; default uses compare-t-max.")
    p.add_argument("--power-map-mode", choices=["straight_through", "soft"], default="straight_through", help="How logits are converted to legal PAM4 powers for terminal/NLSE losses.")
    p.add_argument("--power-temperature", type=float, default=1.0)
    p.add_argument("--hybrid-init-from-data", action="store_true", default=True)
    p.add_argument("--hybrid-random-init", action="store_false", dest="hybrid_init_from_data")
    p.add_argument("--pure-data-init-checkpoint", type=str, default="", help="Optional saved pure_data checkpoint used to initialize hybrid when pure_data is skipped.")
    p.add_argument("--forward-time-chunk", type=int, default=512)

    p.add_argument("--skip-pure-data", action="store_true")
    p.add_argument("--skip-hybrid", action="store_true")
    p.add_argument("--skip-hybrid-physics", action="store_true", dest="skip_hybrid", help="Backward-compatible alias for --skip-hybrid.")
    p.add_argument("--skip-pure-physics", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = safe_device(args.device)
    if len(args.model_labels) != len(args.model_paths):
        raise ValueError("model-labels and model-paths must have the same length")

    gate = select_forward_model(
        Path(args.forward_metrics_csv),
        [str(x) for x in args.model_labels],
        [str(x) for x in args.model_paths],
        float(args.forward_mean_threshold),
        float(args.forward_p95_threshold),
        out_dir,
    )
    selected = gate["selected_model"]
    forward_model = load_forward_checkpoint(selected["path"], device)
    pde_params = load_pde_params_from_forward_checkpoint(selected["path"])
    forward_model.eval()
    for param in forward_model.parameters():
        param.requires_grad_(False)
    print("\nSelected forward model for inverse physics losses:")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print("PDE params:", json.dumps(pde_params.__dict__, indent=2, ensure_ascii=False))

    ds_info = build_or_load_inverse_ssfm_dataset(args, out_dir, device)
    ds_root = Path(ds_info["root"])
    tau_input = np.load(ds_root / "tau_input.npy").astype(np.float32)
    if str(args.inverse_split_json).strip():
        split_path = Path(str(args.inverse_split_json))
        if not split_path.exists():
            raise FileNotFoundError(f"inverse split JSON not found: {split_path}")
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if "train_indices" not in split or "test_indices" not in split:
            raise ValueError(f"inverse split JSON must contain train_indices and test_indices: {split_path}")
        split["source_split_json"] = str(split_path)
        P_for_check = np.load(ds_root / "P_levels.npy", mmap_mode="r")
        n_total_check = int(P_for_check.shape[0])
        bad = [int(i) for i in list(split.get("train_indices", [])) + list(split.get("test_indices", [])) if int(i) < 0 or int(i) >= n_total_check]
        if bad:
            raise ValueError(f"inverse split contains out-of-range indices for dataset size {n_total_check}; examples: {bad[:5]}")
        split["n_total"] = int(n_total_check)
        split["n_train"] = int(len(split["train_indices"]))
        split["n_test"] = int(len(split["test_indices"]))
    else:
        split = make_or_load_split(ds_root, int(args.seed), float(args.inverse_train_ratio), int(args.max_inverse_train_samples), int(args.max_inverse_test_samples))
    (out_dir / "inverse_split_used.json").write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    train_ds = InverseMemmapDataset(ds_root, split["train_indices"])
    test_ds = InverseMemmapDataset(ds_root, split["test_indices"])

    # Scale by the max terminal power in the stored dataset, but read it chunkwise to avoid RAM spikes.
    Y_mem = np.load(ds_root / "Y_terminal_power.npy", mmap_mode="r")
    y_scale = float(np.max(Y_mem)) if Y_mem.size else 1.0
    if not np.isfinite(y_scale) or y_scale <= 0:
        y_scale = 1.0

    pde_params = load_pde_params_from_forward_checkpoint(selected["path"])

    run_config = vars(args).copy()
    run_config.update({
        "device_actual": str(device),
        "selected_forward_model": selected,
        "forward_gate_passed": bool(gate["forward_gate_passed"]),
        "inverse_dataset_root": str(ds_root),
        "split": {k: v for k, v in split.items() if k not in {"train_indices", "test_indices"}},
        "network_design": "terminal SSFM power waveform -> M PAM4 class heads; pure_data, hybrid, pure_physics inverse methods",
        "losses": {
            "pure_data": "L_data",
            "hybrid": "lambda_d L_data + lambda_T L_terminal + lambda_f L_NLSE",
            "pure_physics": "lambda_T L_terminal + lambda_f L_NLSE; true labels are used only for evaluation",
        },
        "pde_params": pde_params.__dict__,
        "note": "The frozen forward PINN F* is selected once, then used by hybrid and pure_physics. NLSE loss is a PDE residual on collocation points, not a full-field label comparison.",
    })
    (out_dir / "inverse_three_methods_run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    summaries: dict = {"selected_forward_model": selected, "forward_gate_passed": bool(gate["forward_gate_passed"])}

    pure_ckpt_path: Path | None = Path(args.pure_data_init_checkpoint) if str(args.pure_data_init_checkpoint).strip() else None
    if pure_ckpt_path is not None and not pure_ckpt_path.exists():
        raise FileNotFoundError(f"pure_data init checkpoint not found: {pure_ckpt_path}")
    if not args.skip_pure_data:
        pure_model = InverseMLP(args.n_pulses, args.inverse_input_points, args.inverse_hidden, args.inverse_layers, y_scale=y_scale)
        pure_result = train_inverse_model("pure_data", pure_model, train_ds, test_ds, tau_input, args, device, out_dir)
        summaries["pure_data"] = pure_result.summary
        pure_ckpt_path = Path(pure_result.summary["checkpoint"])
    else:
        summaries["pure_data"] = {"skipped": True}

    physics_allowed = bool(gate["forward_gate_passed"]) or bool(args.force_physics_inverse)

    if not args.skip_hybrid:
        if not physics_allowed:
            summaries["hybrid"] = {
                "skipped": True,
                "reason": "Selected forward model failed the accuracy gate. Use --force-physics-inverse to run anyway.",
            }
            hyb_dir = out_dir / "hybrid"
            hyb_dir.mkdir(exist_ok=True)
            (hyb_dir / "hybrid_summary.json").write_text(json.dumps(summaries["hybrid"], indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            hyb_model = InverseMLP(args.n_pulses, args.inverse_input_points, args.inverse_hidden, args.inverse_layers, y_scale=y_scale)
            init_path = pure_ckpt_path if (args.hybrid_init_from_data and pure_ckpt_path is not None) else None
            if init_path is None and args.hybrid_init_from_data and str(args.pure_data_init_checkpoint).strip():
                init_path = Path(args.pure_data_init_checkpoint)
            hyb_result = train_inverse_model("hybrid", hyb_model, train_ds, test_ds, tau_input, args, device, out_dir, forward_model=forward_model, pde_params=pde_params, init_state_path=init_path)
            summaries["hybrid"] = hyb_result.summary
    else:
        summaries["hybrid"] = {"skipped": True}

    if not args.skip_pure_physics:
        if not physics_allowed:
            summaries["pure_physics"] = {
                "skipped": True,
                "reason": "Selected forward model failed the accuracy gate. Use --force-physics-inverse to run anyway.",
            }
            pp_dir = out_dir / "pure_physics"
            pp_dir.mkdir(exist_ok=True)
            (pp_dir / "pure_physics_summary.json").write_text(json.dumps(summaries["pure_physics"], indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            pp_model = InverseMLP(args.n_pulses, args.inverse_input_points, args.inverse_hidden, args.inverse_layers, y_scale=y_scale)
            pp_result = train_inverse_model("pure_physics", pp_model, train_ds, test_ds, tau_input, args, device, out_dir, forward_model=forward_model, pde_params=pde_params)
            summaries["pure_physics"] = pp_result.summary
    else:
        summaries["pure_physics"] = {"skipped": True}

    plot_method_comparison(out_dir, summaries)
    (out_dir / "inverse_three_methods_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    # Backward-compatible summary filename for old scripts.
    (out_dir / "inverse_data_vs_physics_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n========== Inverse three-method stage finished ==========")
    print("outputs ->", out_dir)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
