# -*- coding: utf-8 -*-
"""
eval_universal_pinn_full81.py

Evaluate the existing universal sparse-8 PINN on the exact unseen split used by
the universal PINN/CNN/DDNN comparison, over the full 81-plane propagation map.

This script intentionally reuses the SSFM grid, metric definitions, unseen-set
selection, streaming CSV format, and summary logic from
run_universal_full81_CNN_DDNN.py so that the three universal models are evaluated
with the same protocol.

No training is performed.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from train_multi_pulse_pinn import ConditionalPINN
from run_universal_full81_CNN_DDNN import (
    METRIC_NAMES,
    combo_text,
    ensure_dir,
    find_universal_run,
    load_pinn_checkpoint_metadata,
    load_sparse8_csv,
    make_grid,
    metrics_from_maps,
    parse_done_indices,
    predict_ddnn_maps,
    resolve_checkpoint,
    run_ssfm_batch_selected,
    safe_device,
    select_eval_positions,
    set_seed,
    summarize_metrics,
    write_json,
)

SCRIPT_VERSION = "universal_pinn_full81_eval_v1_20260716"


def load_pinn_checkpoint(path: Path, device: torch.device):
    payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected PINN checkpoint payload: {path}")
    model_cfg = dict(payload.get("model_config", {}))
    if int(model_cfg.get("n_pulses", -1)) != 8:
        raise RuntimeError(f"Checkpoint is not an 8-slot universal PINN: {model_cfg}")
    if "model_state" not in payload:
        raise RuntimeError(f"Checkpoint has no 'model_state': {path}")
    model = ConditionalPINN(**model_cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full-81-plane evaluation of the existing universal K=1..8 PINN."
    )
    p.add_argument("--runs-root", default="./MULTIPULSE_AMPLITUDE_RUNS")
    p.add_argument("--pinn-run-dir", default="")
    p.add_argument("--pinn-checkpoint", default="")
    p.add_argument("--out-dir", default="", help="Default: <PINN run>/universal_PINN_full81/eval_full81_all_unseen")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--compare-t-min", type=float, default=None)
    p.add_argument("--compare-t-max", type=float, default=None)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--n-slices", type=int, default=81)
    p.add_argument("--ssfm-batch-size", type=int, default=64)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--pinn-chunk-size", type=int, default=65536)

    p.add_argument("--max-eval-per-k", type=int, default=0, help="0 evaluates all exact PINN-unseen waveforms.")
    p.add_argument("--eval-seed", type=int, default=2027)
    p.add_argument("--overwrite-eval", action="store_true")
    p.add_argument("--summary-only", action="store_true")
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

    unseen_csv = run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    if not unseen_csv.is_file():
        raise FileNotFoundError(f"Unseen CSV not found: {unseen_csv}")
    unseen_k, unseen_a = load_sparse8_csv(unseen_csv)

    out_dir = ensure_dir(
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else run_dir / "universal_PINN_full81" / "eval_full81_all_unseen"
    )
    metrics_path = out_dir / "metrics_stream.csv"

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
    print(f"unseen         : {len(unseen_a)}", flush=True)
    print(f"unseen by K    : { {int(k): int(np.sum(unseen_k==k)) for k in np.unique(unseen_k)} }", flush=True)
    print(f"grid           : {args.n_slices} planes, nt={args.n_t}, nz={args.n_z}, compare=[{grid['compare_t_min']},{grid['compare_t_max']}]", flush=True)
    print(f"output         : {out_dir}", flush=True)
    print("=" * 110, flush=True)

    if args.summary_only:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"No metrics CSV found: {metrics_path}")
        summarize_metrics(metrics_path, out_dir)
        return

    if args.overwrite_eval and metrics_path.exists():
        metrics_path.unlink()

    positions = select_eval_positions(unseen_k, args.max_eval_per_k, args.eval_seed)
    selected_k = unseen_k[positions]
    selected_a = unseen_a[positions]
    selected_idx = positions.copy()

    done = parse_done_indices(metrics_path)
    keep = np.asarray([int(i) not in done for i in selected_idx], dtype=bool)
    pending_idx = selected_idx[keep]
    pending_k = selected_k[keep]
    pending_a = selected_a[keep]

    model, payload = load_pinn_checkpoint(checkpoint, device)

    fieldnames = (
        ["idx", "K"]
        + [f"A{i}" for i in range(1, 9)]
        + ["levels"]
        + METRIC_NAMES
        + ["ssfm_sec_per_sample", "pinn_sec_per_sample"]
    )
    new_file = not metrics_path.exists() or metrics_path.stat().st_size == 0
    f = metrics_path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()
        f.flush()

    print(
        f"[PINN eval] selected={len(selected_idx)} already_done={len(selected_idx)-len(pending_idx)} "
        f"pending={len(pending_idx)} | planes={grid['n_slices']}",
        flush=True,
    )

    t_all = time.time()
    completed = 0
    try:
        for start in range(0, len(pending_idx), int(args.ssfm_batch_size)):
            end = min(len(pending_idx), start + int(args.ssfm_batch_size))
            idx = pending_idx[start:end]
            kk = pending_k[start:end]
            aa = pending_a[start:end]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            ref = run_ssfm_batch_selected(
                aa, grid, pde, device, bool(args.ssfm_complex64)
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ssfm_sec = time.perf_counter() - t0

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            # The universal PINN and DDNN share the same ConditionalPINN
            # architecture and therefore the same map-inference routine.
            pred = predict_ddnn_maps(
                model,
                aa,
                np.asarray(grid["tau"]),
                np.asarray(grid["zeta"]),
                device,
                int(args.pinn_chunk_size),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pinn_sec = time.perf_counter() - t1

            metrics = metrics_from_maps(pred, ref)
            b = len(idx)
            for i in range(b):
                row: Dict[str, Any] = {
                    "idx": int(idx[i]),
                    "K": int(kk[i]),
                    **{f"A{j+1}": f"{float(aa[i, j]):g}" for j in range(8)},
                    "levels": combo_text(aa[i]),
                    **{m: float(metrics[m][i]) for m in METRIC_NAMES},
                    "ssfm_sec_per_sample": float(ssfm_sec / b),
                    "pinn_sec_per_sample": float(pinn_sec / b),
                }
                writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

            completed += b
            elapsed = time.time() - t_all
            rate = completed / max(elapsed, 1e-9)
            eta = (len(pending_idx) - completed) / max(rate, 1e-12)
            print(
                f"[PINN eval] {completed}/{len(pending_idx)} | "
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
            "method": "pinn",
            "checkpoint": str(checkpoint),
            "n_selected_unseen": int(len(selected_idx)),
            "max_eval_per_k": int(args.max_eval_per_k),
            "all_81_planes_evaluated": int(grid["n_slices"]) == 81,
            "grid": {
                k: v
                for k, v in grid.items()
                if k not in {"tau_full", "time_mask", "tau", "selected_steps", "zeta"}
            },
            "selected_steps": [
                int(x) for x in np.asarray(grid["selected_steps"]).tolist()
            ],
        },
    )
    print("\nPINN full-81-plane evaluation complete.", flush=True)


if __name__ == "__main__":
    main()
