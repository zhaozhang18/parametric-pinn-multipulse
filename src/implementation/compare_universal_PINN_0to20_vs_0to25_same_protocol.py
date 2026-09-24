# -*- coding: utf-8 -*-
"""
compare_universal_PINN_0to20_vs_0to25_same_protocol.py

Purpose
-------
Compare the original 0-20 km universal PINN and the direct 0-25 km universal PINN
under the same evaluation protocol:
- exactly the same unseen samples
- exactly the same SSFM reference fields
- exactly the same z positions
- exactly the same metrics

Default comparison range
------------------------
0-20 km, with 0.25 km step.

Reference SSFM range
--------------------
By default 0-25 km with n_z=1000 so that every 0.25 km location is hit exactly.
This makes the comparison grid identical to the user's zero-shot fine-grid analysis.

Outputs
-------
- per_sample_per_distance.csv
- summary_by_distance.csv
- overall_summary.csv
- overall_summary.json
- run_config.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from train_multi_pulse_pinn import ConditionalPINN
import run_universal_full81_CNN_DDNN as base

SCRIPT_VERSION = "COMPARE_UNIVERSAL_PINN_0TO20_VS_0TO25_SAME_PROTOCOL_V1_20260718"


def load_pinn_checkpoint(path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected PINN checkpoint payload: %s" % path)
    cfg = dict(payload.get("model_config", {}))
    if int(cfg.get("n_pulses", -1)) != 8:
        raise RuntimeError("Checkpoint is not the 8-slot universal PINN: %s" % path)
    state = payload.get("model_state")
    if state is None:
        raise RuntimeError("Checkpoint has no model_state: %s" % path)
    model = ConditionalPINN(**cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def percentile_stats(x: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
        "fraction_below_0p10": float(np.mean(arr <= 0.10)),
    }


def macro_mean_over_k(values: np.ndarray, ks: np.ndarray) -> float:
    unique_k = np.unique(ks)
    per_k = []
    for k in unique_k:
        mask = ks == k
        per_k.append(float(np.mean(values[mask])))
    return float(np.mean(per_k))


def per_plane_metrics(pred: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred64 = np.asarray(pred, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    eps = 1e-300
    diff = pred64 - ref64
    field = np.sqrt(np.sum(diff ** 2, axis=(2, 3))) / np.maximum(np.sqrt(np.sum(ref64 ** 2, axis=(2, 3))), eps)
    p_pred = np.sum(pred64 ** 2, axis=2)
    p_ref = np.sum(ref64 ** 2, axis=2)
    power = np.sqrt(np.sum((p_pred - p_ref) ** 2, axis=2)) / np.maximum(np.sqrt(np.sum(p_ref ** 2, axis=2)), eps)
    return field, power


def make_ssfm_grid(
    model_cfg: Mapping[str, Any],
    ssfm_half_window: float,
    n_t: int,
    n_z: int,
    ssfm_total_end_km: float,
    eval_start_km: float,
    eval_end_km: float,
    step_km: float,
    ld_km: float,
) -> Dict[str, Any]:
    z_km = np.arange(eval_start_km, eval_end_km + 0.5 * step_km, step_km, dtype=np.float64)
    zeta = z_km / float(ld_km)
    z_max_ld_ssfm = float(ssfm_total_end_km) / float(ld_km)

    tau_full = np.linspace(-float(ssfm_half_window), float(ssfm_half_window), int(n_t), endpoint=False, dtype=np.float64)
    t_min = float(model_cfg.get("t_min", -44.0))
    t_max = float(model_cfg.get("t_max", 44.0))
    time_mask = (tau_full >= t_min - 1e-12) & (tau_full <= t_max + 1e-12)
    tau = tau_full[time_mask]

    requested_steps_float = zeta / z_max_ld_ssfm * float(n_z)
    requested_steps = np.rint(requested_steps_float).astype(np.int64)
    reconstructed_zeta = requested_steps.astype(np.float64) * z_max_ld_ssfm / float(n_z)
    max_position_error = float(np.max(np.abs(reconstructed_zeta - zeta)))
    if max_position_error > 1e-10:
        raise ValueError("SSFM grid does not hit requested z exactly. max error=%.6g" % max_position_error)

    selected_steps = np.unique(np.concatenate([np.asarray([0], dtype=np.int64), requested_steps])).astype(np.int64)
    step_to_slot = {int(step): int(i) for i, step in enumerate(selected_steps.tolist())}
    requested_slots = np.asarray([step_to_slot[int(step)] for step in requested_steps.tolist()], dtype=np.int64)

    return {
        "ssfm_half_window": float(ssfm_half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "z_max_ld": float(z_max_ld_ssfm),
        "compare_t_min": float(t_min),
        "compare_t_max": float(t_max),
        "tau_full": tau_full,
        "time_mask": time_mask,
        "tau": tau,
        "selected_steps": selected_steps,
        "requested_steps": requested_steps,
        "requested_slots": requested_slots,
        "zeta": zeta,
        "z_km": z_km,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pinn20-ckpt", default="")
    p.add_argument("--pinn25-ckpt", required=True)
    p.add_argument("--device", default="cuda")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--use-all-unseen", action="store_true")
    group.add_argument("--max-eval-per-k", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=2027)

    p.add_argument("--compare-start-km", type=float, default=0.0)
    p.add_argument("--compare-end-km", type=float, default=20.0)
    p.add_argument("--step-km", type=float, default=0.25)
    p.add_argument("--ssfm-total-end-km", type=float, default=25.0)
    p.add_argument("--ld-km", type=float, default=5.0)

    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=1000)
    p.add_argument("--ssfm-batch-size", type=int, default=64)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--model-chunk-size", type=int, default=131072)
    p.add_argument("--out-dir", default="")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(str(run_dir))

    device = base.safe_device(str(args.device))
    pinn20_ckpt = Path(args.pinn20_ckpt).expanduser().resolve() if args.pinn20_ckpt else (run_dir / "sparse8_forward_pinn.pt")
    pinn25_ckpt = Path(args.pinn25_ckpt).expanduser().resolve()
    if not pinn20_ckpt.is_file():
        raise FileNotFoundError("0-20 PINN checkpoint not found: %s" % pinn20_ckpt)
    if not pinn25_ckpt.is_file():
        raise FileNotFoundError("0-25 PINN checkpoint not found: %s" % pinn25_ckpt)

    model20_cfg, pde = base.load_pinn_checkpoint_metadata(pinn20_ckpt)
    model25_cfg, _ = base.load_pinn_checkpoint_metadata(pinn25_ckpt)
    if abs(float(model20_cfg.get("z_max_ld", 4.0)) - 4.0) > 1e-12:
        raise RuntimeError("The 0-20 checkpoint does not have z_max_ld=4.0: %s" % pinn20_ckpt)
    if abs(float(model25_cfg.get("z_max_ld", 5.0)) - 5.0) > 1e-12:
        raise RuntimeError("The 0-25 checkpoint does not have z_max_ld=5.0: %s" % pinn25_ckpt)

    unseen_k, unseen_a = base.load_sparse8_csv(run_dir / "dataset" / "unseen_sparse8_combinations.csv")
    if args.use_all_unseen:
        positions = np.arange(len(unseen_a), dtype=np.int64)
    else:
        positions = base.select_eval_positions(unseen_k, int(args.max_eval_per_k), int(args.eval_seed))
    selected_k = unseen_k[positions]
    selected_a = unseen_a[positions]

    grid = make_ssfm_grid(
        model_cfg=model20_cfg,
        ssfm_half_window=float(args.ssfm_half_window),
        n_t=int(args.n_t),
        n_z=int(args.n_z),
        ssfm_total_end_km=float(args.ssfm_total_end_km),
        eval_start_km=float(args.compare_start_km),
        eval_end_km=float(args.compare_end_km),
        step_km=float(args.step_km),
        ld_km=float(args.ld_km),
    )

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        label = "all_unseen" if args.use_all_unseen else ("quick_max%d_seed%d" % (int(args.max_eval_per_k), int(args.eval_seed)))
        out_dir = run_dir / "compare_universal_PINN_0to20_vs_0to25_same_protocol" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = out_dir / "per_sample_per_distance.csv"
    summary_path = out_dir / "summary_by_distance.csv"
    overall_csv = out_dir / "overall_summary.csv"
    overall_json = out_dir / "overall_summary.json"
    if per_sample_path.is_file() and not args.overwrite:
        raise FileExistsError("Output already exists. Use --overwrite: %s" % per_sample_path)

    pinn20_model, _ = load_pinn_checkpoint(pinn20_ckpt, device)
    pinn25_model, _ = load_pinn_checkpoint(pinn25_ckpt, device)

    print("=" * 120, flush=True)
    print("SAME-PROTOCOL COMPARISON: 0-20 PINN vs 0-25 PINN", flush=True)
    print("PINN 0-20 :", pinn20_ckpt, flush=True)
    print("PINN 0-25 :", pinn25_ckpt, flush=True)
    print("COMPARE Z  : %.2f-%.2f km step %.2f km" % (float(args.compare_start_km), float(args.compare_end_km), float(args.step_km)), flush=True)
    print("SSFM REF   : 0-%.2f km with n_z=%d" % (float(args.ssfm_total_end_km), int(args.n_z)), flush=True)
    print("SAMPLES    : %d" % len(selected_a), flush=True)
    print("by K       : %s" % {int(k): int(np.sum(selected_k == k)) for k in np.unique(selected_k)}, flush=True)
    print("OUTPUT     :", out_dir, flush=True)
    print("=" * 120, flush=True)

    rows: List[Dict[str, Any]] = []
    all20_field = []
    all20_power = []
    all25_field = []
    all25_power = []

    total = len(selected_a)
    t_all = time.time()
    for start in range(0, total, int(args.ssfm_batch_size)):
        end = min(total, start + int(args.ssfm_batch_size))
        idx = positions[start:end]
        kk = selected_k[start:end]
        aa = selected_a[start:end]

        ref_all = base.run_ssfm_batch_selected(aa, grid, pde, device, bool(args.ssfm_complex64))
        ref = ref_all[:, np.asarray(grid["requested_slots"], dtype=np.int64), :, :]

        tau32 = np.asarray(grid["tau"], dtype=np.float32)
        zeta32 = np.asarray(grid["zeta"], dtype=np.float32)
        pred20 = base.predict_ddnn_maps(pinn20_model, aa, tau32, zeta32, device, int(args.model_chunk_size))
        pred25 = base.predict_ddnn_maps(pinn25_model, aa, tau32, zeta32, device, int(args.model_chunk_size))

        field20, power20 = per_plane_metrics(pred20, ref)
        field25, power25 = per_plane_metrics(pred25, ref)

        all20_field.append(field20)
        all20_power.append(power20)
        all25_field.append(field25)
        all25_power.append(power25)

        for bi in range(len(aa)):
            for zi, z_km in enumerate(grid["z_km"]):
                common = {
                    "idx": int(idx[bi]),
                    "K": int(kk[bi]),
                    "z_km": float(z_km),
                    "z_over_LD": float(grid["zeta"][zi]),
                }
                rows.append({**common, "method": "PINN_0to20", "field_rel_l2": float(field20[bi, zi]), "power_rel_l2": float(power20[bi, zi])})
                rows.append({**common, "method": "PINN_0to25", "field_rel_l2": float(field25[bi, zi]), "power_rel_l2": float(power25[bi, zi])})

        completed = end
        elapsed = time.time() - t_all
        rate = completed / max(elapsed, 1e-9)
        eta = (total - completed) / max(rate, 1e-12)
        term_idx = int(np.argmin(np.abs(np.asarray(grid["z_km"]) - float(args.compare_end_km))))
        print(
            "[eval] %d/%d | 0-20PINN terminal=%.2f%% | 0-25PINN terminal=%.2f%% | rate=%.2f sample/s ETA=%.1f min"
            % (
                completed,
                total,
                100.0 * float(np.mean(power20[:, term_idx])),
                100.0 * float(np.mean(power25[:, term_idx])),
                rate,
                eta / 60.0,
            ),
            flush=True,
        )

        del ref_all, ref, pred20, pred25
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all20_field = np.concatenate(all20_field, axis=0)
    all20_power = np.concatenate(all20_power, axis=0)
    all25_field = np.concatenate(all25_field, axis=0)
    all25_power = np.concatenate(all25_power, axis=0)

    save_csv(per_sample_path, rows)

    summary_rows: List[Dict[str, Any]] = []
    for method, field_arr, power_arr in (
        ("PINN_0to20", all20_field, all20_power),
        ("PINN_0to25", all25_field, all25_power),
    ):
        for zi, z_km in enumerate(grid["z_km"]):
            field_stats = percentile_stats(field_arr[:, zi])
            power_stats = percentile_stats(power_arr[:, zi])
            summary_rows.append({
                "method": method,
                "z_km": float(z_km),
                "z_over_LD": float(grid["zeta"][zi]),
                "n_samples": int(len(selected_k)),
                "power_mean": power_stats["mean"],
                "power_median": power_stats["median"],
                "power_p95": power_stats["p95"],
                "power_max": power_stats["max"],
                "power_fraction_below_10pct": power_stats["fraction_below_0p10"],
                "field_mean": field_stats["mean"],
                "field_median": field_stats["median"],
                "field_p95": field_stats["p95"],
                "field_max": field_stats["max"],
                "field_fraction_below_10pct": field_stats["fraction_below_0p10"],
                "power_macro_mean_by_K": macro_mean_over_k(power_arr[:, zi], selected_k),
                "field_macro_mean_by_K": macro_mean_over_k(field_arr[:, zi], selected_k),
            })
    save_csv(summary_path, summary_rows)

    compare_end_idx = int(np.argmin(np.abs(np.asarray(grid["z_km"]) - float(args.compare_end_km))))
    overall_rows = []
    for method, field_arr, power_arr in (
        ("PINN_0to20", all20_field, all20_power),
        ("PINN_0to25", all25_field, all25_power),
    ):
        overall_rows.append({
            "method": method,
            "n_samples": int(len(selected_k)),
            "range_km": "%.2f-%.2f" % (float(args.compare_start_km), float(args.compare_end_km)),
            "micro_full_field": float(np.mean(field_arr)),
            "micro_full_power": float(np.mean(power_arr)),
            "macro_full_field": macro_mean_over_k(np.mean(field_arr, axis=1), selected_k),
            "macro_full_power": macro_mean_over_k(np.mean(power_arr, axis=1), selected_k),
            "micro_terminal_field": float(np.mean(field_arr[:, compare_end_idx])),
            "micro_terminal_power": float(np.mean(power_arr[:, compare_end_idx])),
            "macro_terminal_field": macro_mean_over_k(field_arr[:, compare_end_idx], selected_k),
            "macro_terminal_power": macro_mean_over_k(power_arr[:, compare_end_idx], selected_k),
            "fraction_terminal_power_below_10pct": float(np.mean(power_arr[:, compare_end_idx] <= 0.10)),
            "fraction_full_power_mean_below_10pct": float(np.mean(np.mean(power_arr, axis=1) <= 0.10)),
        })
    save_csv(overall_csv, overall_rows)

    overall_payload = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "pinn20_checkpoint": str(pinn20_ckpt),
        "pinn25_checkpoint": str(pinn25_ckpt),
        "n_selected": int(len(selected_k)),
        "selected_by_K": {str(int(k)): int(np.sum(selected_k == k)) for k in np.unique(selected_k)},
        "compare_range_km": [float(args.compare_start_km), float(args.compare_end_km)],
        "step_km": float(args.step_km),
        "overall_rows": overall_rows,
    }
    overall_json.write_text(json.dumps(overall_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    config = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "pinn20_checkpoint": str(pinn20_ckpt),
        "pinn25_checkpoint": str(pinn25_ckpt),
        "compare_range_km": [float(args.compare_start_km), float(args.compare_end_km)],
        "step_km": float(args.step_km),
        "ssfm_total_end_km": float(args.ssfm_total_end_km),
        "use_all_unseen": bool(args.use_all_unseen),
        "max_eval_per_k": None if args.use_all_unseen else int(args.max_eval_per_k),
        "eval_seed": int(args.eval_seed),
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone -> %s" % out_dir, flush=True)
    print("Distance summary -> %s" % summary_path, flush=True)
    print("Overall summary  -> %s" % overall_csv, flush=True)


if __name__ == "__main__":
    main()
