# -*- coding: utf-8 -*-
"""Train a dispersion-conditioned sparse-8 Universal PINN initialized from the original non-D Universal PINN.

The original Universal PINN has inputs (z,t,A1..A8) plus Fourier(z,t), whereas the
new model adds one continuous dispersion input D.  Direct strict state loading is
therefore impossible because the first linear layer has one extra input column.
This script performs a function-preserving transfer initialization:
1) copy every compatible parameter from the original Universal PINN;
2) expand the first layer by inserting the new D column between A8 and Fourier features;
3) initialize the new D-column weights to zero;
4) verify numerically that, before D-training, the new model reproduces the original
   model output for D=0.8, 1.0, and 1.2;
5) continue with the same pure-physics D-conditioned training pipeline.


This script deliberately reuses the successful K=1..8 dataset split, High-K PDE
allocation, K loss weights, mixed-time collocation, Adam -> High-K Adam -> L-BFGS
schedule, boundary loss and conservation loss from train_forward_sparse8_universal_highK.py.

Only the necessary changes are introduced:
1) model input becomes (z,t,A1..A8,D);
2) fixed-reference NLSE uses beta2_norm * D;
3) train D values are balanced inside every K stratum;
4) losses average equally over D inside each K before applying the original K weights;
5) PDE residual is processed in chunks to avoid 6 GB CUDA OOM.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import train_forward_sparse8_universal_highK as base
from train_multi_pulse_pinn import ConditionalPINN, initial_condition, load_pde_params_from_nlse
from nlse import pulse_centers_t0
from universal_beta2_common import (
    UniversalDispersionConditionalPINN,
    parse_float_list,
    pde_residual_variable_d,
    write_json,
)

SCRIPT_VERSION = "universal_sparse8_highK_beta2_transfer_from_original_v2_schema_aligned_20260722"


def read_sparse_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ks: list[int] = []
    amps: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            ks.append(int(float(row["K"])))
            amps.append([float(row[f"A{i}"]) for i in range(1, 9)])
    return np.asarray(ks, dtype=np.int64), np.asarray(amps, dtype=np.float32)


def load_or_build_dataset(args, out_dir: Path, K_values: Sequence[int], levels_nonzero: Sequence[float]):
    dataset_dir = base.ensure_dir(out_dir / "dataset")
    source = Path(args.dataset_source_dir).expanduser().resolve() if str(args.dataset_source_dir).strip() else None
    if source is not None:
        seen_src = source / "seen_sparse8_combinations.csv"
        unseen_src = source / "unseen_sparse8_combinations.csv"
        if not seen_src.is_file() or not unseen_src.is_file():
            raise FileNotFoundError(
                f"dataset-source-dir must contain seen_sparse8_combinations.csv and unseen_sparse8_combinations.csv: {source}"
            )
        train_k, train_combos = read_sparse_csv(seen_src)
        test_k, test_combos = read_sparse_csv(unseen_src)
        train_by_k = {int(k): train_combos[train_k == int(k)] for k in K_values}
        test_by_k = {int(k): test_combos[test_k == int(k)] for k in K_values}
        shutil.copy2(seen_src, dataset_dir / "seen_sparse8_combinations.csv")
        shutil.copy2(unseen_src, dataset_dir / "unseen_sparse8_combinations.csv")
        for K in K_values:
            base.write_combos_with_k_csv(
                dataset_dir / f"K{K}_seen_sparse8_combinations.csv",
                train_by_k[int(K)],
                np.full(len(train_by_k[int(K)]), int(K), dtype=np.int64),
            )
            base.write_combos_with_k_csv(
                dataset_dir / f"K{K}_unseen_sparse8_combinations.csv",
                test_by_k[int(K)],
                np.full(len(test_by_k[int(K)]), int(K), dtype=np.int64),
            )
        source_summary_path = source / "dataset_summary.json"
        source_summary = {}
        if source_summary_path.is_file():
            try:
                source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
            except Exception:
                source_summary = {}
        total_capacity = int(sum(4 ** int(k) for k in K_values))
        dataset_summary = {
            **source_summary,
            "dataset_source_dir": str(source),
            "dataset_reused_exactly": True,
            "K_values": [int(k) for k in K_values],
            "train_counts_by_k": {str(k): int(len(train_by_k[int(k)])) for k in K_values},
            "unseen_counts_by_k": {str(k): int(len(test_by_k[int(k)])) for k in K_values},
            "n_train": int(len(train_combos)),
            "n_unseen_saved": int(len(test_combos)),
            "total_capacity": total_capacity,
            "overall_train_fraction": float(len(train_combos) / total_capacity),
        }
    else:
        train_counts = base.parse_int_map(args.train_counts_by_k)
        (
            train_by_k,
            test_by_k,
            train_combos,
            test_combos,
            train_k,
            test_k,
            sample_stats,
        ) = base.build_stratified_dataset(
            levels_nonzero=levels_nonzero,
            K_values=K_values,
            train_counts_by_k=train_counts,
            seed=args.seed,
            sampling_strategy=args.sampling_strategy,
            test_per_k=None,
        )
        base.write_combos_with_k_csv(dataset_dir / "seen_sparse8_combinations.csv", train_combos, train_k)
        base.write_combos_with_k_csv(dataset_dir / "unseen_sparse8_combinations.csv", test_combos, test_k)
        for K in K_values:
            base.write_combos_with_k_csv(
                dataset_dir / f"K{K}_seen_sparse8_combinations.csv",
                train_by_k[int(K)],
                np.full(len(train_by_k[int(K)]), int(K), dtype=np.int64),
            )
            base.write_combos_with_k_csv(
                dataset_dir / f"K{K}_unseen_sparse8_combinations.csv",
                test_by_k[int(K)],
                np.full(len(test_by_k[int(K)]), int(K), dtype=np.int64),
            )
        total_capacity = int(sum(4 ** int(k) for k in K_values))
        dataset_summary = {
            "dataset_reused_exactly": False,
            "K_values": [int(k) for k in K_values],
            "train_counts_by_k": {str(k): int(len(train_by_k[int(k)])) for k in K_values},
            "unseen_counts_by_k": {str(k): int(len(test_by_k[int(k)])) for k in K_values},
            "n_train": int(len(train_combos)),
            "n_unseen_saved": int(len(test_combos)),
            "total_capacity": total_capacity,
            "overall_train_fraction": float(len(train_combos) / total_capacity),
            "sample_stats": sample_stats,
        }
    write_json(dataset_dir / "dataset_summary.json", dataset_summary)
    return train_by_k, test_by_k, dataset_summary


def balanced_d_labels(k_labels: torch.Tensor, d_values: Sequence[float], seed: int) -> torch.Tensor:
    labels_np = k_labels.detach().cpu().numpy().reshape(-1)
    out = np.empty((len(labels_np), 1), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    d_arr = np.asarray(d_values, dtype=np.float32)
    for K in sorted(np.unique(labels_np).tolist()):
        idx = np.flatnonzero(labels_np == int(K))
        vals = np.resize(d_arr, len(idx)).copy()
        rng.shuffle(vals)
        out[idx, 0] = vals
    return torch.tensor(out, dtype=torch.float32, device=k_labels.device)


def attach_d_to_collocation(tensors: dict[str, torch.Tensor], d_values: Sequence[float], seed: int) -> dict[str, torch.Tensor]:
    tensors = dict(tensors)
    tensors["D_pde"] = balanced_d_labels(tensors["K_pde"], d_values, seed + 11)
    tensors["D_ic"] = balanced_d_labels(tensors["K_ic"], d_values, seed + 23)
    tensors["D_boundary"] = balanced_d_labels(tensors["K_boundary"], d_values, seed + 37)

    # Conservation is small, so explicitly evaluate every sampled (K,A,z) combination at every D.
    n0 = int(tensors["A_cons_combo"].shape[0])
    nd = len(d_values)
    tensors["A_cons_combo"] = tensors["A_cons_combo"].repeat_interleave(nd, dim=0)
    tensors["K_cons_combo"] = tensors["K_cons_combo"].repeat_interleave(nd, dim=0)
    tensors["z_cons_combo"] = tensors["z_cons_combo"].repeat_interleave(nd, dim=0)
    d_tile = torch.tensor(d_values, dtype=torch.float32, device=tensors["A_cons_combo"].device).reshape(1, nd, 1)
    tensors["D_cons_combo"] = d_tile.expand(n0, nd, 1).reshape(n0 * nd, 1).contiguous()
    return tensors


def validate_groups(K_rows: torch.Tensor, D_rows: torch.Tensor, K_values: Sequence[int], D_values: Sequence[float]) -> None:
    kf = K_rows.reshape(-1)
    df = D_rows.reshape(-1)
    for K in K_values:
        for D in D_values:
            if not bool(torch.any((kf == int(K)) & (torch.abs(df - float(D)) < 1e-6))):
                raise RuntimeError(f"Missing K={K}, D={D} group in collocation set.")


def weighted_kd_mean(
    values: torch.Tensor,
    K_rows: torch.Tensor,
    D_rows: torch.Tensor,
    K_values: Sequence[int],
    D_values: Sequence[float],
    k_weights: Mapping[int, float],
) -> torch.Tensor:
    flat = values.reshape(-1)
    kf = K_rows.reshape(-1)
    df = D_rows.reshape(-1)
    wk = np.asarray([float(k_weights[int(k)]) for k in K_values], dtype=np.float64)
    wk = wk / wk.sum()
    per_k: list[torch.Tensor] = []
    for K in K_values:
        d_means: list[torch.Tensor] = []
        for D in D_values:
            mask = (kf == int(K)) & (torch.abs(df - float(D)) < 1e-6)
            if not bool(torch.any(mask)):
                raise RuntimeError(f"Missing K={K}, D={D} rows.")
            d_means.append(flat[mask].mean())
        per_k.append(torch.stack(d_means).mean())
    w = torch.tensor(wk, dtype=per_k[0].dtype, device=per_k[0].device)
    return torch.sum(torch.stack(per_k) * w)


def kd_full_counts(K_rows: torch.Tensor, D_rows: torch.Tensor, K_values: Sequence[int], D_values: Sequence[float]) -> dict[tuple[int, float], int]:
    kf = K_rows.reshape(-1)
    df = D_rows.reshape(-1)
    out: dict[tuple[int, float], int] = {}
    for K in K_values:
        for D in D_values:
            n = int(((kf == int(K)) & (torch.abs(df - float(D)) < 1e-6)).sum().item())
            if n <= 0:
                raise RuntimeError(f"Missing K={K},D={D} rows.")
            out[(int(K), float(D))] = n
    return out


def weighted_kd_chunk_contribution(
    values: torch.Tensor,
    K_rows: torch.Tensor,
    D_rows: torch.Tensor,
    K_values: Sequence[int],
    D_values: Sequence[float],
    k_weights: Mapping[int, float],
    full_counts: Mapping[tuple[int, float], int],
) -> torch.Tensor:
    flat = values.reshape(-1)
    kf = K_rows.reshape(-1)
    df = D_rows.reshape(-1)
    weight_sum = float(sum(float(k_weights[int(k)]) for k in K_values))
    out = torch.zeros((), dtype=flat.dtype, device=flat.device)
    for K in K_values:
        k_weight = float(k_weights[int(K)]) / weight_sum
        for D in D_values:
            mask = (kf == int(K)) & (torch.abs(df - float(D)) < 1e-6)
            if bool(torch.any(mask)):
                out = out + (k_weight / float(len(D_values))) * flat[mask].sum() / float(full_counts[(int(K), float(D))])
    return out



def transfer_original_universal_to_d_model(
    target_model: UniversalDispersionConditionalPINN,
    source_checkpoint: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Function-preserving expansion from the original Universal PINN to the D-conditioned model."""
    source_checkpoint = Path(source_checkpoint).expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(f"Original Universal PINN checkpoint not found: {source_checkpoint}")

    payload = torch.load(str(source_checkpoint), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected checkpoint payload: {source_checkpoint}")

    source_cfg = dict(payload.get("model_config", {}))
    source_state = payload.get("model_state", payload.get("state_dict"))
    if source_state is None:
        raise RuntimeError(f"Checkpoint has no model_state/state_dict: {source_checkpoint}")

    required_cfg = [
        "n_pulses", "hidden", "layers", "z_max_ld",
        "t_min", "t_max", "p_min", "p_max", "fourier_features",
    ]
    missing = [key for key in required_cfg if key not in source_cfg]
    if missing:
        raise RuntimeError(f"Original checkpoint is missing model_config keys: {missing}")

    target_cfg = target_model.config()
    compare_keys = [
        "n_pulses", "hidden", "layers", "z_max_ld",
        "t_min", "t_max", "p_min", "p_max", "fourier_features",
    ]
    mismatches = {
        key: (source_cfg.get(key), target_cfg.get(key))
        for key in compare_keys
        if source_cfg.get(key) != target_cfg.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Original Universal PINN architecture/config does not match the target D-model "
            f"apart from the intended extra D input. Mismatches: {mismatches}"
        )

    # Instantiate the source model exactly as it was trained. This also validates the checkpoint.
    source_model = ConditionalPINN(**source_cfg).to(device)
    source_model.load_state_dict(source_state, strict=True)
    source_model.eval()

    src = source_model.state_dict()
    dst = target_model.state_dict()
    first_weight_key = "net.0.weight"

    if first_weight_key not in src or first_weight_key not in dst:
        raise RuntimeError(
            f"Expected first-layer key {first_weight_key!r}; "
            f"source keys include {list(src.keys())[:8]}, target keys include {list(dst.keys())[:8]}"
        )

    src_w = src[first_weight_key]
    dst_w = dst[first_weight_key]
    n_pulses = int(target_cfg["n_pulses"])
    raw_old_dim = 2 + n_pulses  # z, t, A1..A8
    d_column = raw_old_dim      # new order: z, t, A1..A8, D, Fourier(z,t)

    if src_w.ndim != 2 or dst_w.ndim != 2:
        raise RuntimeError("First-layer tensors are not matrices.")
    if dst_w.shape[0] != src_w.shape[0] or dst_w.shape[1] != src_w.shape[1] + 1:
        raise RuntimeError(
            "Expected the D-conditioned first layer to have exactly one extra input column. "
            f"source={tuple(src_w.shape)}, target={tuple(dst_w.shape)}"
        )
    if raw_old_dim > src_w.shape[1]:
        raise RuntimeError(
            f"Invalid raw input dimension {raw_old_dim} for source first layer {tuple(src_w.shape)}"
        )

    transferred = {}
    for key, target_tensor in dst.items():
        if key == first_weight_key:
            expanded = target_tensor.detach().clone()
            # Raw variables z,t,A1..A8 retain their exact columns.
            expanded[:, :raw_old_dim] = src_w[:, :raw_old_dim]
            # New dispersion input D starts as a zero-effect feature.
            expanded[:, d_column].zero_()
            # Fourier(z,t) columns move right by one position.
            expanded[:, d_column + 1:] = src_w[:, raw_old_dim:]
            transferred[key] = expanded
        else:
            if key not in src:
                raise RuntimeError(f"Target parameter {key!r} has no source counterpart.")
            if tuple(src[key].shape) != tuple(target_tensor.shape):
                raise RuntimeError(
                    f"Shape mismatch for {key}: source={tuple(src[key].shape)}, "
                    f"target={tuple(target_tensor.shape)}"
                )
            transferred[key] = src[key].detach().clone()

    target_model.load_state_dict(transferred, strict=True)
    target_model.eval()

    # Numerical verification: with zero D-column weights, the new model must initially
    # reproduce the original model for every D.  We explicitly test all training D values.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260722)
    n_check = 64
    z = torch.rand((n_check, 1), generator=generator, dtype=torch.float32)
    z = z * float(target_cfg["z_max_ld"])
    t = torch.rand((n_check, 1), generator=generator, dtype=torch.float32)
    t = float(target_cfg["t_min"]) + t * (
        float(target_cfg["t_max"]) - float(target_cfg["t_min"])
    )
    amplitudes = torch.rand(
        (n_check, n_pulses), generator=generator, dtype=torch.float32
    )

    z = z.to(device)
    t = t.to(device)
    amplitudes = amplitudes.to(device)

    with torch.no_grad():
        src_u, src_v = source_model(z, t, amplitudes)
        max_abs_diff = 0.0
        per_d_diff = {}
        for d_value in (0.8, 1.0, 1.2):
            d = torch.full((n_check, 1), float(d_value), device=device, dtype=torch.float32)
            dst_u, dst_v = target_model(z, t, amplitudes, d)
            diff = max(
                float(torch.max(torch.abs(dst_u - src_u)).detach().cpu()),
                float(torch.max(torch.abs(dst_v - src_v)).detach().cpu()),
            )
            per_d_diff[str(d_value)] = diff
            max_abs_diff = max(max_abs_diff, diff)

    tolerance = 2e-6
    if max_abs_diff > tolerance:
        raise RuntimeError(
            "Transfer verification failed: the expanded D-conditioned model does not "
            f"reproduce the original model before D-training. max_abs_diff={max_abs_diff:.3e}"
        )

    target_model.train()
    meta = {
        "initialization": "function_preserving_transfer_from_original_nonD_universal_PINN",
        "source_checkpoint": str(source_checkpoint),
        "source_model_config": source_cfg,
        "target_model_config": target_cfg,
        "source_first_layer_shape": [int(x) for x in src_w.shape],
        "target_first_layer_shape": [int(x) for x in dst_w.shape],
        "inserted_D_column_index_zero_based": int(d_column),
        "inserted_D_column_weight_init": 0.0,
        "all_later_layers_copied_exactly": True,
        "verification_D_values": [0.8, 1.0, 1.2],
        "verification_max_abs_diff": float(max_abs_diff),
        "verification_per_D_max_abs_diff": per_d_diff,
        "verification_tolerance": float(tolerance),
    }
    return meta

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sparse-8 K=1..8 High-K PINN conditioned on dispersion ratio D.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-source-dir", default="", help="Recommended: dataset folder from the successful original universal run.")
    p.add_argument(
        "--source-universal-checkpoint",
        default="",
        help="Original non-D Universal PINN checkpoint used for transfer initialization. Required unless --resume-checkpoint is supplied.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--train-d", default="0.8,1.0,1.2")
    p.add_argument("--levels-nonzero", default="0.25,0.5,0.75,1.0")
    p.add_argument("--k-values", default="1,2,3,4,5,6,7,8")
    p.add_argument("--train-counts-by-k", default=base.train_count_map_to_text(base.DEFAULT_TRAIN_COUNTS_BY_K))
    p.add_argument("--sampling-strategy", choices=["balanced", "random"], default="balanced")

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--fourier-features", type=int, default=4)
    p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--t-min", type=float, default=-44.0)
    p.add_argument("--t-max", type=float, default=44.0)

    p.add_argument("--n-ic", type=int, default=7500)
    p.add_argument("--n-pde", type=int, default=40000)
    p.add_argument("--pde-chunk-size", type=int, default=8192)
    p.add_argument("--ensure-all-seen-ic", dest="ensure_all_seen_ic", action="store_true", default=True)
    p.add_argument("--no-ensure-all-seen-ic", dest="ensure_all_seen_ic", action="store_false")
    p.add_argument("--ic-points-per-seen", type=int, default=2)
    p.add_argument("--power-strata", type=int, default=3)
    p.add_argument("--pde-counts-by-k", default=base.train_count_map_to_text(base.DEFAULT_MAIN_PDE_COUNTS_BY_K))
    p.add_argument("--finetune-pde-counts-by-k", default=base.train_count_map_to_text(base.DEFAULT_FINETUNE_PDE_COUNTS_BY_K))
    p.add_argument("--main-k-loss-weights", default=base.float_map_to_text(base.DEFAULT_MAIN_K_LOSS_WEIGHTS))
    p.add_argument("--finetune-k-loss-weights", default=base.float_map_to_text(base.DEFAULT_FINETUNE_K_LOSS_WEIGHTS))

    p.add_argument("--pde-global-fraction", type=float, default=0.40)
    p.add_argument("--pde-pulse-fraction", type=float, default=0.40)
    p.add_argument("--pde-midpoint-fraction", type=float, default=0.20)
    p.add_argument("--ic-global-fraction", type=float, default=0.20)
    p.add_argument("--ic-pulse-fraction", type=float, default=0.55)
    p.add_argument("--ic-midpoint-fraction", type=float, default=0.25)
    p.add_argument("--pulse-local-sigma", type=float, default=3.5)
    p.add_argument("--midpoint-local-sigma", type=float, default=2.5)
    p.add_argument("--n-boundary", type=int, default=1024)
    p.add_argument("--conservation-combos-per-k", type=int, default=2)
    p.add_argument("--conservation-nt", type=int, default=128)

    p.add_argument("--adam-steps", type=int, default=5000)
    p.add_argument("--finetune-steps", type=int, default=1500)
    p.add_argument("--finetune-lr", type=float, default=3e-4)
    p.add_argument("--finetune-resample-every", type=int, default=100)
    p.add_argument("--ic-warmup-steps", type=int, default=500)
    p.add_argument("--resample-every", type=int, default=100)
    p.add_argument("--lbfgs-epochs", type=int, default=1200)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--min-lbfgs-epochs", type=int, default=100)
    p.add_argument("--early-stop-eps", type=float, default=1e-8)
    p.add_argument("--early-stop-patience", type=int, default=20)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ic-weight", type=float, default=2.0)
    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--boundary-weight", type=float, default=0.05)
    p.add_argument("--conservation-weight", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume-checkpoint", default="", help="Optional stage checkpoint to load before --start-stage.")
    p.add_argument("--start-stage", choices=["adam", "finetune", "lbfgs"], default="adam")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    out_dir = base.ensure_dir(args.out_dir)
    K_values = sorted(set(int(x) for x in str(args.k_values).replace(",", " ").split() if x.strip()))
    D_values = parse_float_list(args.train_d)
    if len(D_values) < 2:
        raise ValueError("Use at least two train-D values.")
    levels_nonzero = base.parse_levels(args.levels_nonzero)
    main_pde_counts = base.parse_int_map(args.pde_counts_by_k)
    finetune_pde_counts = base.parse_int_map(args.finetune_pde_counts_by_k)
    main_k_weights = base.parse_float_map(args.main_k_loss_weights)
    finetune_k_weights = base.parse_float_map(args.finetune_k_loss_weights)
    base.validate_complete_k_map("main_pde_counts", main_pde_counts, K_values)
    base.validate_complete_k_map("finetune_pde_counts", finetune_pde_counts, K_values)
    base.validate_complete_k_map("main_k_weights", main_k_weights, K_values)
    base.validate_complete_k_map("finetune_k_weights", finetune_k_weights, K_values)
    if sum(main_pde_counts.values()) != int(args.n_pde) or sum(finetune_pde_counts.values()) != int(args.n_pde):
        raise ValueError("PDE count maps must each sum to --n-pde.")

    train_by_k, _test_by_k, dataset_summary = load_or_build_dataset(args, out_dir, K_values, levels_nonzero)
    dataset_summary.update({
        "train_D_values": [float(x) for x in D_values],
        "D_definition": "D=beta2/beta2_ref under fixed reference normalization",
        "physics_parameter_generalization": True,
        "script_version": SCRIPT_VERSION,
    })
    write_json(out_dir / "dataset" / "dataset_summary.json", dataset_summary)

    model = UniversalDispersionConditionalPINN(
        n_pulses=8,
        hidden=args.hidden,
        layers=args.layers,
        z_max_ld=args.z_max_ld,
        t_min=args.t_min,
        t_max=args.t_max,
        p_min=0.0,
        p_max=1.0,
        d_min=min(D_values),
        d_max=max(D_values),
        fourier_features=args.fourier_features,
    ).to(device)
    pde_obj = load_pde_params_from_nlse()
    pde = dict(pde_obj.__dict__)
    centers = pulse_centers_t0(8)

    initialization_meta: dict[str, Any]
    if str(args.resume_checkpoint).strip():
        resume = Path(args.resume_checkpoint).expanduser().resolve()
        if not resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        payload = torch.load(str(resume), map_location=device)
        state = payload.get("model_state", payload.get("state_dict"))
        if state is None:
            raise RuntimeError(f"Resume checkpoint has no model_state/state_dict: {resume}")
        model.load_state_dict(state, strict=True)
        initialization_meta = dict(payload.get("initialization_meta", {}))
        initialization_meta.setdefault("resumed_from", str(resume))
        print(f"[resume] loaded D-conditioned model state from {resume}", flush=True)
    else:
        if not str(args.source_universal_checkpoint).strip():
            raise ValueError(
                "--source-universal-checkpoint is required for a fresh transfer-initialized run."
            )
        initialization_meta = transfer_original_universal_to_d_model(
            model,
            Path(args.source_universal_checkpoint),
            device,
        )
        # Keep transfer-only metadata separate so the standard D-model checkpoint schema
        # remains identical to the direct-training pipeline.
        write_json(out_dir / "transfer_initialization_meta.json", initialization_meta)
        print(
            "[transfer] original non-D Universal PINN -> D-conditioned model completed; "
            f"verification max_abs_diff={initialization_meta['verification_max_abs_diff']:.3e}",
            flush=True,
        )

    latest_diag: dict[str, Any] = {}

    def resample(seed_offset: int, stage: str):
        nonlocal latest_diag
        pde_counts = main_pde_counts if stage == "main" else finetune_pde_counts
        tensors, diag = base.sample_collocation_stratified_k(
            train_by_k=train_by_k,
            K_values=K_values,
            n_pde=args.n_pde,
            n_ic=args.n_ic,
            z_min=0.0,
            z_max=args.z_max_ld,
            t_min=args.t_min,
            t_max=args.t_max,
            seed=args.seed + int(seed_offset),
            device=device,
            ensure_all_seen_ic=args.ensure_all_seen_ic,
            ic_points_per_seen=args.ic_points_per_seen,
            power_strata=args.power_strata,
            pde_time_fractions=(args.pde_global_fraction, args.pde_pulse_fraction, args.pde_midpoint_fraction),
            ic_time_fractions=(args.ic_global_fraction, args.ic_pulse_fraction, args.ic_midpoint_fraction),
            pulse_local_sigma=args.pulse_local_sigma,
            midpoint_local_sigma=args.midpoint_local_sigma,
            n_boundary=args.n_boundary,
            conservation_combos_per_k=args.conservation_combos_per_k,
            conservation_nt=args.conservation_nt,
            pde_counts_by_k=pde_counts,
        )
        tensors = attach_d_to_collocation(tensors, D_values, args.seed + int(seed_offset) * 101)
        for family in ["pde", "ic", "boundary", "cons_combo"]:
            kkey = "K_" + family
            dkey = "D_" + family
            validate_groups(tensors[kkey], tensors[dkey], K_values, D_values)

        u0, v0 = initial_condition(tensors["t_ic"], tensors["A_ic"], centers)
        n_cons = int(tensors["A_cons_combo"].shape[0])
        nt_cons = int(tensors["t_cons_grid"].shape[0])
        t0_exp = tensors["t_cons_grid"].reshape(1, nt_cons, 1).expand(n_cons, nt_cons, 1)
        A0_exp = tensors["A_cons_combo"].reshape(n_cons, 1, 8).expand(n_cons, nt_cons, 8)
        u_cons0, v_cons0 = initial_condition(t0_exp.reshape(-1, 1), A0_exp.reshape(-1, 8), centers)
        p0 = (u_cons0.square() + v_cons0.square()).reshape(n_cons, nt_cons)
        tensors["E0_cons_combo"] = torch.trapz(p0, tensors["t_cons_grid"].reshape(-1), dim=1).detach()
        diag.update({
            "training_stage": stage,
            "train_D_values": [float(x) for x in D_values],
            "loss_aggregation": "mean inside every (K,D), equal mean over D, then original normalized K difficulty weights",
            "pde_chunk_size": int(args.pde_chunk_size),
            "conservation_rows_after_D_expansion": n_cons,
        })
        latest_diag = diag
        return tensors, u0.detach(), v0.detach()

    tensors, u0, v0 = resample(0, "main" if args.start_stage == "adam" else "finetune")
    write_json(out_dir / "collocation_summary.json", latest_diag)

    def backward_losses(k_weights: Mapping[int, float], physics_scale: float) -> dict[str, float]:
        # IC
        up, vp = model(tensors["z_ic"], tensors["t_ic"], tensors["A_ic"], tensors["D_ic"])
        ic = weighted_kd_mean((up - u0).square() + (vp - v0).square(), tensors["K_ic"], tensors["D_ic"], K_values, D_values, k_weights)

        # Boundary
        ub, vb = model(tensors["z_boundary"], tensors["t_boundary"], tensors["A_boundary"], tensors["D_boundary"])
        boundary = weighted_kd_mean(ub.square() + vb.square(), tensors["K_boundary"], tensors["D_boundary"], K_values, D_values, k_weights)

        # Conservation
        n_cons = int(tensors["A_cons_combo"].shape[0])
        nt_cons = int(tensors["t_cons_grid"].shape[0])
        zc = tensors["z_cons_combo"].reshape(n_cons, 1, 1).expand(n_cons, nt_cons, 1)
        tc = tensors["t_cons_grid"].reshape(1, nt_cons, 1).expand(n_cons, nt_cons, 1)
        Ac = tensors["A_cons_combo"].reshape(n_cons, 1, 8).expand(n_cons, nt_cons, 8)
        Dc = tensors["D_cons_combo"].reshape(n_cons, 1, 1).expand(n_cons, nt_cons, 1)
        uc, vc = model(zc.reshape(-1, 1), tc.reshape(-1, 1), Ac.reshape(-1, 8), Dc.reshape(-1, 1))
        pc = (uc.square() + vc.square()).reshape(n_cons, nt_cons)
        Ec = torch.trapz(pc, tensors["t_cons_grid"].reshape(-1), dim=1)
        cons_rows = torch.log((Ec + 1e-8) / (tensors["E0_cons_combo"] + 1e-8)).square()
        conservation = weighted_kd_mean(cons_rows, tensors["K_cons_combo"], tensors["D_cons_combo"], K_values, D_values, k_weights)

        non_pde_total = (
            args.ic_weight * ic
            + float(physics_scale) * (args.boundary_weight * boundary + args.conservation_weight * conservation)
        )
        non_pde_total.backward()

        # Exact chunked PDE contribution. Each chunk backpropagates immediately, so its second-derivative graph is freed.
        counts = kd_full_counts(tensors["K_pde"], tensors["D_pde"], K_values, D_values)
        pde_value = 0.0
        n_pde = int(tensors["z_pde"].shape[0])
        chunk = max(1, int(args.pde_chunk_size))
        for s in range(0, n_pde, chunk):
            e = min(n_pde, s + chunk)
            f, g = pde_residual_variable_d(
                model,
                tensors["z_pde"][s:e],
                tensors["t_pde"][s:e],
                tensors["A_pde"][s:e],
                tensors["D_pde"][s:e],
                pde,
            )
            contrib = weighted_kd_chunk_contribution(
                f.square() + g.square(),
                tensors["K_pde"][s:e],
                tensors["D_pde"][s:e],
                K_values,
                D_values,
                k_weights,
                counts,
            )
            (float(physics_scale) * args.pde_weight * contrib).backward()
            pde_value += float(contrib.detach().cpu())

        ic_v = float(ic.detach().cpu())
        boundary_v = float(boundary.detach().cpu())
        cons_v = float(conservation.detach().cpu())
        total_v = args.ic_weight * ic_v + float(physics_scale) * (
            args.pde_weight * pde_value + args.boundary_weight * boundary_v + args.conservation_weight * cons_v
        )
        return {
            "loss": total_v,
            "ic": ic_v,
            "pde": pde_value,
            "boundary": boundary_v,
            "conservation": cons_v,
            "physics_scale": float(physics_scale),
        }

    logs: list[dict[str, Any]] = []
    t_start = time.time()

    def save_stage(name: str, stage: str) -> Path:
        path = out_dir / name
        torch.save({
            "model_state": model.state_dict(),
            "model_config": model.config(),
            "pde_params": pde,
            "dataset_summary": dataset_summary,
            "collocation_summary": latest_diag,
            "train_args": vars(args),
            "training_stage": stage,
            "script_version": SCRIPT_VERSION,
            "final_loss": logs[-1] if logs else None,
        }, path)
        print(f"stage checkpoint -> {path}", flush=True)
        return path

    print("=" * 110, flush=True)
    print("Sparse-8 universal High-K PINN + dispersion parameter D", flush=True)
    print(f"device={device} | D_train={D_values}", flush=True)
    print(f"network=8 amplitudes + D, hidden={args.hidden}, layers={args.layers}, Fourier(z,t)={args.fourier_features}", flush=True)
    print(f"initialization={initialization_meta.get('initialization', 'resume')} | source={initialization_meta.get('source_checkpoint', initialization_meta.get('resumed_from', ''))}", flush=True)
    print(f"dataset exact reuse={dataset_summary.get('dataset_reused_exactly')} | seen={dataset_summary.get('n_train')} unseen={dataset_summary.get('n_unseen_saved')}", flush=True)
    print(f"PDE total={args.n_pde}, chunk={args.pde_chunk_size}; IC actual={latest_diag.get('ic_points_actual')}", flush=True)
    print(f"main PDE by K={main_pde_counts}", flush=True)
    print(f"finetune PDE by K={finetune_pde_counts}", flush=True)
    print("=" * 110, flush=True)

    if args.start_stage == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for step in range(1, int(args.adam_steps) + 1):
            if args.resample_every > 0 and step > 1 and step % int(args.resample_every) == 0:
                tensors, u0, v0 = resample(step, "main")
                write_json(out_dir / "collocation_summary.json", latest_diag)
            physics_scale = min(1.0, max(0.05, step / float(args.ic_warmup_steps))) if args.ic_warmup_steps > 0 else 1.0
            opt.zero_grad(set_to_none=True)
            vals = backward_losses(main_k_weights, physics_scale)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.adam_steps:
                row = {"phase": "adam", "step": step, **vals, "elapsed_sec": time.time() - t_start}
                logs.append(row); print(row, flush=True)
        save_stage("sparse8_beta2_forward_pinn_after_main_adam.pt", "after_main_adam")

    if args.start_stage in {"adam", "finetune"} and args.finetune_steps > 0:
        tensors, u0, v0 = resample(args.adam_steps + 1, "finetune")
        write_json(out_dir / "collocation_summary.json", latest_diag)
        opt_ft = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        for step in range(1, int(args.finetune_steps) + 1):
            if args.finetune_resample_every > 0 and step > 1 and step % int(args.finetune_resample_every) == 0:
                tensors, u0, v0 = resample(args.adam_steps + step, "finetune")
                write_json(out_dir / "collocation_summary.json", latest_diag)
            opt_ft.zero_grad(set_to_none=True)
            vals = backward_losses(finetune_k_weights, 1.0)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt_ft.step()
            if step == 1 or step % args.log_every == 0 or step == args.finetune_steps:
                row = {"phase": "adam_highk_finetune", "step": step, **vals, "elapsed_sec": time.time() - t_start}
                logs.append(row); print(row, flush=True)
        save_stage("sparse8_beta2_forward_pinn_after_highk_finetune.pt", "after_highk_finetune")

    if args.lbfgs_epochs > 0:
        # Keep one fixed fine-tune-weighted collocation set during L-BFGS, matching the original strategy.
        if args.start_stage == "lbfgs":
            tensors, u0, v0 = resample(args.adam_steps + args.finetune_steps + 1, "finetune")
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=args.lbfgs_max_iter,
            max_eval=max(args.lbfgs_max_iter + 10, args.lbfgs_max_iter), line_search_fn="strong_wolfe"
        )
        best = float("inf"); stale = 0
        for ep in range(1, int(args.lbfgs_epochs) + 1):
            vals_holder: dict[str, float] = {}
            def closure():
                opt_lbfgs.zero_grad(set_to_none=True)
                vals = backward_losses(finetune_k_weights, 1.0)
                vals_holder.clear(); vals_holder.update(vals)
                return torch.tensor(vals["loss"], dtype=torch.float32, device=device)
            loss_out = opt_lbfgs.step(closure)
            current = float(vals_holder.get("loss", float(loss_out.detach().cpu()) if torch.is_tensor(loss_out) else loss_out))
            improvement = best - current
            threshold = max(args.early_stop_eps, abs(best) * 1e-7) if math.isfinite(best) else 0.0
            if current < best: best = current
            stale = 0 if improvement > threshold else stale + 1
            if ep == 1 or ep % args.log_every == 0 or ep == args.lbfgs_epochs:
                row = {"phase": "lbfgs", "step": ep, **vals_holder, "elapsed_sec": time.time() - t_start}
                logs.append(row); print(row, flush=True)
            if ep >= args.min_lbfgs_epochs and stale >= args.early_stop_patience:
                row = {"phase": "lbfgs_early_stop", "step": ep, **vals_holder, "elapsed_sec": time.time() - t_start}
                logs.append(row); print(f"L-BFGS early stop at epoch={ep}, stale={stale}", flush=True)
                break

    if logs:
        keys = ["phase", "step", "loss", "ic", "pde", "boundary", "conservation", "physics_scale", "elapsed_sec"]
        with (out_dir / "sparse8_beta2_train_log.csv").open("w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(logs)

    final_path = out_dir / "sparse8_beta2_forward_pinn.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "pde_params": pde,
        "dataset_summary": dataset_summary,
        "collocation_summary": latest_diag,
        "train_args": vars(args),
        "script_version": SCRIPT_VERSION,
        "final_loss": logs[-1] if logs else None,
    }, final_path)
    write_json(out_dir / "run_summary.json", {
        "checkpoint": str(final_path),
        "elapsed_sec": time.time() - t_start,
        "train_D_values": D_values,
        "dataset_reused_exactly": dataset_summary.get("dataset_reused_exactly"),
        "final_log": logs[-1] if logs else None,
    })
    print(f"checkpoint -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
