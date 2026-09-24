# -*- coding: utf-8 -*-
"""Streaming full-propagation evaluation for the K=1..8 beta2-conditioned PINN.

The exact original unseen amplitude split is reused from RUN_DIR/dataset.
For every requested D, the same selected unseen amplitude rows are evaluated.
Supports balanced subset evaluation (--max-eval-per-k > 0) and formal all-unseen
(--max-eval-per-k 0). Results are resumable.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from universal_beta2_common import (
    METRIC_NAMES,
    ensure_dir,
    load_beta2_checkpoint,
    load_sparse8_csv,
    make_grid,
    metrics_from_maps,
    parse_float_list,
    predict_maps_variable_d,
    run_ssfm_batch_selected_variable_d,
    safe_device,
    save_csv_rows,
    select_eval_positions,
    write_json,
)

SCRIPT_VERSION = "universal_beta2_full81_eval_v1_20260719"


def d_key(x: float) -> str:
    return f"{float(x):.8f}"


def combo_text(row: Sequence[float]) -> str:
    return ";".join(f"{float(x):g}" for x in row)


def parse_done(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.is_file() or path.stat().st_size == 0:
        return done
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add((int(row["idx"]), d_key(float(row["D"]))))
            except Exception:
                pass
    return done


def stats(vals: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def summarize(metrics_path: Path, out_dir: Path) -> None:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {metrics_path}")
    D_values = sorted(set(float(r["D"]) for r in rows))
    K_values = sorted(set(int(r["K"]) for r in rows))

    by_kd: list[dict[str, Any]] = []
    kd_means: dict[tuple[float, int, str], float] = {}
    for D in D_values:
        for K in K_values:
            group = [r for r in rows if abs(float(r["D"]) - D) < 1e-7 and int(r["K"]) == K]
            if not group:
                continue
            item: dict[str, Any] = {"D": D, "group": group[0]["group"], "K": K, "n_samples": len(group)}
            for m in METRIC_NAMES:
                s = stats([float(r[m]) for r in group])
                kd_means[(D, K, m)] = s["mean"]
                for name, val in s.items():
                    item[f"{m}_{name}"] = val
            by_kd.append(item)
    save_csv_rows(out_dir / "summary_by_K_D.csv", by_kd)

    by_d: list[dict[str, Any]] = []
    for D in D_values:
        group = [r for r in rows if abs(float(r["D"]) - D) < 1e-7]
        present_k = sorted(set(int(r["K"]) for r in group))
        item: dict[str, Any] = {"D": D, "group": group[0]["group"], "n_samples": len(group), "n_K": len(present_k)}
        for m in METRIC_NAMES:
            micro = float(np.mean([float(r[m]) for r in group]))
            macro = float(np.mean([kd_means[(D, K, m)] for K in present_k]))
            item[f"{m}_micro_mean"] = micro
            item[f"{m}_macro_equal_K_mean"] = macro
        by_d.append(item)
    save_csv_rows(out_dir / "summary_by_D.csv", by_d)

    by_group: list[dict[str, Any]] = []
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    for g in groups:
        group_rows = [r for r in rows if r["group"] == g]
        d_in_group = sorted(set(float(r["D"]) for r in group_rows))
        item: dict[str, Any] = {"group": g, "n_samples": len(group_rows), "n_D": len(d_in_group)}
        for m in METRIC_NAMES:
            item[f"{m}_micro_mean"] = float(np.mean([float(r[m]) for r in group_rows]))
            kd_vals = [
                kd_means[(D, K, m)]
                for D in d_in_group
                for K in K_values
                if (D, K, m) in kd_means
            ]
            item[f"{m}_macro_equal_D_K_mean"] = float(np.mean(kd_vals))
        by_group.append(item)
    save_csv_rows(out_dir / "summary_by_group.csv", by_group)

    write_json(out_dir / "overall_summary.json", {
        "script_version": SCRIPT_VERSION,
        "n_rows": len(rows),
        "D_values": D_values,
        "K_values": K_values,
        "primary_forward_metric": "summary_by_D/full_rel_l2_power_macro_equal_K_mean",
        "primary_group_metric": "summary_by_group/full_rel_l2_power_macro_equal_D_K_mean",
    })
    print("[summary]", flush=True)
    for row in by_d:
        print(
            f"  D={row['D']:.3f} {row['group']}: "
            f"full power Macro={100*row['full_rel_l2_power_macro_equal_K_mean']:.4f}% | "
            f"terminal power Macro={100*row['terminal_rel_l2_power_macro_equal_K_mean']:.4f}%",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-81-plane evaluation for universal beta2-conditioned sparse-8 PINN.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-d", default="0.8,1.0,1.2")
    p.add_argument("--interp-d", default="0.9,1.1")
    p.add_argument("--extra-d", default="0.7,1.3")
    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--compare-t-min", type=float, default=None)
    p.add_argument("--compare-t-max", type=float, default=None)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--n-slices", type=int, default=81)
    p.add_argument("--ssfm-batch-size", type=int, default=32)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--pinn-chunk-size", type=int, default=65536)
    p.add_argument("--max-eval-per-k", type=int, default=256, help="0 = all exact unseen rows for every D; 256 is recommended first.")
    p.add_argument("--eval-seed", type=int, default=2027)
    p.add_argument("--overwrite-eval", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = safe_device(args.device)
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve() if str(args.checkpoint).strip() else run_dir / "sparse8_beta2_forward_pinn.pt"
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else run_dir / "eval_beta2_full81")
    metrics_path = out_dir / "metrics_stream.csv"
    if args.summary_only:
        summarize(metrics_path, out_dir); return
    if args.overwrite_eval and metrics_path.exists():
        metrics_path.unlink()

    model, payload = load_beta2_checkpoint(checkpoint, device)
    cfg = dict(payload["model_config"])
    pde = dict(payload.get("pde_params", {}))
    unseen_k, unseen_a = load_sparse8_csv(run_dir / "dataset" / "unseen_sparse8_combinations.csv")
    pos = select_eval_positions(unseen_k, args.max_eval_per_k, args.eval_seed)
    selected_k, selected_a = unseen_k[pos], unseen_a[pos]
    grid = make_grid(cfg, args.ssfm_half_window, args.n_t, args.n_z, args.n_slices, args.compare_t_min, args.compare_t_max)

    groups: list[tuple[str, float]] = []
    for name, text in [("seen_D", args.train_d), ("interpolation_D", args.interp_d), ("extrapolation_D", args.extra_d)]:
        for D in parse_float_list(text):
            if not any(abs(D - oldD) < 1e-9 for _, oldD in groups):
                groups.append((name, float(D)))

    done = parse_done(metrics_path)
    fields = ["idx", "K", "D", "group"] + [f"A{i}" for i in range(1, 9)] + ["levels"] + METRIC_NAMES + ["ssfm_sec_per_sample", "pinn_sec_per_sample"]
    new = not metrics_path.exists() or metrics_path.stat().st_size == 0
    f = metrics_path.open("a", encoding="utf-8-sig", newline="")
    wr = csv.DictWriter(f, fieldnames=fields)
    if new: wr.writeheader(); f.flush()

    print("=" * 110, flush=True)
    print(f"checkpoint={checkpoint}", flush=True)
    print(f"selected amplitudes={len(pos)} | max-per-K={args.max_eval_per_k}", flush=True)
    print(f"D conditions={groups}", flush=True)
    print(f"grid={args.n_slices} planes, nt={args.n_t}, nz={args.n_z}", flush=True)
    print(f"output={out_dir}", flush=True)
    print("=" * 110, flush=True)

    total_pending = sum(1 for _, D in groups for idx in pos if (int(idx), d_key(D)) not in done)
    completed = 0; t_all = time.time()
    try:
        for group_name, D in groups:
            keep = np.asarray([(int(idx), d_key(D)) not in done for idx in pos], dtype=bool)
            idx_D, k_D, a_D = pos[keep], selected_k[keep], selected_a[keep]
            for start in range(0, len(idx_D), int(args.ssfm_batch_size)):
                end = min(len(idx_D), start + int(args.ssfm_batch_size))
                idx = idx_D[start:end]; kk = k_D[start:end]; aa = a_D[start:end]
                dd = np.full(len(aa), float(D), dtype=np.float32)
                if device.type == "cuda": torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                ref = run_ssfm_batch_selected_variable_d(aa, dd, grid, pde, device, bool(args.ssfm_complex64))
                if device.type == "cuda": torch.cuda.synchronize(device)
                ssfm_sec = time.perf_counter() - t0
                if device.type == "cuda": torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                pred = predict_maps_variable_d(model, aa, dd, np.asarray(grid["tau"]), np.asarray(grid["zeta"]), device, args.pinn_chunk_size)
                if device.type == "cuda": torch.cuda.synchronize(device)
                pinn_sec = time.perf_counter() - t1
                mets = metrics_from_maps(pred, ref)
                for i in range(len(idx)):
                    row: Dict[str, Any] = {
                        "idx": int(idx[i]), "K": int(kk[i]), "D": float(D), "group": group_name,
                        **{f"A{j+1}": f"{float(aa[i,j]):g}" for j in range(8)},
                        "levels": combo_text(aa[i]),
                        **{m: float(mets[m][i]) for m in METRIC_NAMES},
                        "ssfm_sec_per_sample": float(ssfm_sec / len(idx)),
                        "pinn_sec_per_sample": float(pinn_sec / len(idx)),
                    }
                    wr.writerow(row)
                f.flush(); os.fsync(f.fileno())
                completed += len(idx)
                elapsed = time.time() - t_all
                rate = completed / max(elapsed, 1e-9)
                eta = (total_pending - completed) / max(rate, 1e-12)
                print(
                    f"[eval D={D:.3f} {group_name}] {completed}/{total_pending} | "
                    f"batch full-power={100*np.mean(mets['full_rel_l2_power']):.3f}% | ETA={eta/60:.1f} min",
                    flush=True,
                )
                del ref, pred
                if device.type == "cuda": torch.cuda.empty_cache()
    finally:
        f.close()

    summarize(metrics_path, out_dir)
    write_json(out_dir / "eval_config.json", {
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(checkpoint),
        "n_selected_amplitudes": int(len(pos)),
        "max_eval_per_k": int(args.max_eval_per_k),
        "D_conditions": [{"group": g, "D": D} for g, D in groups],
        "grid": {"n_slices": args.n_slices, "n_t": args.n_t, "n_z": args.n_z, "compare_t_min": grid["compare_t_min"], "compare_t_max": grid["compare_t_max"]},
    })


if __name__ == "__main__":
    main()
