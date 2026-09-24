# -*- coding: utf-8 -*-
"""
eval_0to25_budget_10_25_50_full78624_resume.py

Formal all-unseen evaluation for the 0-20 -> 0-25 km low-cost domain-extension experiment.

Evaluates SIX trained 0-25 km PINNs in ONE pass over the full unseen set:
    10% Scratch vs From-20
    25% Scratch vs From-20
    50% Scratch vs From-20

Why one pass?
    SSFM reference propagation is computed only once per batch, then reused by all six models.
    This is much faster than running three separate two-model evaluators.

Protocol is kept consistent with the earlier quick558 evaluator:
    - same unseen dataset
    - same 0-25 km grid (101 planes by default)
    - same region-level relative L2 definitions
    - same regions:
        full_0to25
        old_0to20
        new_20to25  (includes the shared 20 km boundary, exactly as the quick evaluator)
        terminal_25
    - both sample-weighted mean and equal-K macro mean

Resume support:
    Results are written into NumPy memmap files after batches.
    If evaluation is interrupted, rerun the same command/BAT and it resumes from the last
    completed checkpoint instead of starting over.

Outputs:
    summary.csv
    summary_by_K.csv
    comparison_by_budget.csv
    run_config.json
    progress.json
    field_metrics.npy      (resume storage)
    power_metrics.npy      (resume storage)
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


SCRIPT_VERSION = "FULL78624_BUDGET_10_25_50_RESUME_V1_20260719"
REGION_NAMES = ["full_0to25", "old_0to20", "new_20to25", "terminal_25"]


def load_model(path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    payload = torch.load(str(path), map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected checkpoint payload: %s" % path)
    cfg = dict(payload.get("model_config", {}))
    state = payload.get("model_state")
    if state is None:
        raise RuntimeError("Checkpoint has no model_state: %s" % path)
    model = ConditionalPINN(**cfg).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def region_errors(pred: np.ndarray, ref: np.ndarray, plane_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred[:, plane_mask], dtype=np.float64)
    y = np.asarray(ref[:, plane_mask], dtype=np.float64)

    field_num = np.sum((p - y) ** 2, axis=(1, 2, 3))
    field_den = np.sum(y ** 2, axis=(1, 2, 3)) + 1e-300
    field = np.sqrt(field_num / field_den)

    pp = np.sum(p ** 2, axis=2)
    yy = np.sum(y ** 2, axis=2)
    power_num = np.sum((pp - yy) ** 2, axis=(1, 2))
    power_den = np.sum(yy ** 2, axis=(1, 2)) + 1e-300
    power = np.sqrt(power_num / power_den)
    return field, power


def aggregate(values: np.ndarray, k_labels: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    by_k = [float(np.mean(values[k_labels == k])) for k in sorted(np.unique(k_labels).tolist())]
    return {
        "sample_mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "macro_equal_K_mean": float(np.mean(by_k)),
    }


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="cuda")

    p.add_argument("--scratch10", required=True)
    p.add_argument("--from20-10", dest="from20_10", required=True)
    p.add_argument("--scratch25", required=True)
    p.add_argument("--from20-25", dest="from20_25", required=True)
    p.add_argument("--scratch50", required=True)
    p.add_argument("--from20-50", dest="from20_50", required=True)

    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--n-slices", type=int, default=101)
    p.add_argument("--ssfm-batch-size", type=int, default=64)
    p.add_argument("--model-chunk-size", type=int, default=131072)
    p.add_argument("--checkpoint-every-batches", type=int, default=5)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--restart", action="store_true", help="Delete resume arrays/progress and start from sample 0.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = base.safe_device(str(args.device))

    model_specs = [
        ("scratch_10pct_actual", Path(args.scratch10).expanduser().resolve(), "10pct_actual", "scratch"),
        ("from20_10pct_actual", Path(args.from20_10).expanduser().resolve(), "10pct_actual", "from20"),
        ("scratch_25pct_actual", Path(args.scratch25).expanduser().resolve(), "25pct_actual", "scratch"),
        ("from20_25pct_actual", Path(args.from20_25).expanduser().resolve(), "25pct_actual", "from20"),
        ("scratch_50pct_actual", Path(args.scratch50).expanduser().resolve(), "50pct_actual", "scratch"),
        ("from20_50pct_actual", Path(args.from20_50).expanduser().resolve(), "50pct_actual", "from20"),
    ]

    for _, path, _, _ in model_specs:
        if not path.is_file():
            raise FileNotFoundError("Checkpoint not found: %s" % path)

    original_pinn = run_dir / "sparse8_forward_pinn.pt"
    if not original_pinn.is_file():
        raise FileNotFoundError("Original 0-20 final PINN not found: %s" % original_pinn)
    _, pde = base.load_pinn_checkpoint_metadata(original_pinn)

    unseen_k, unseen_a = base.load_sparse8_csv(run_dir / "dataset" / "unseen_sparse8_combinations.csv")
    # Formal evaluation: ALL unseen samples, no per-K cap.
    positions = np.arange(len(unseen_a), dtype=np.int64)
    selected_k = unseen_k
    selected_a = unseen_a
    total = len(selected_a)

    # Load all six models once.
    models: List[torch.nn.Module] = []
    payloads: List[Dict[str, Any]] = []
    for label, path, _, _ in model_specs:
        print("[load] %-24s %s" % (label, path), flush=True)
        model, payload = load_model(path, device)
        models.append(model)
        payloads.append(payload)

    reference_cfg = dict(payloads[0]["model_config"])
    for i, payload in enumerate(payloads[1:], start=1):
        cfg = dict(payload["model_config"])
        if cfg != reference_cfg:
            raise RuntimeError(
                "Model config mismatch between %s and %s" % (model_specs[0][0], model_specs[i][0])
            )
    if abs(float(reference_cfg.get("z_max_ld", -1.0)) - 5.0) > 1e-12:
        raise RuntimeError("Expected all six evaluated models to have z_max_ld=5.0")

    grid = base.make_grid(
        model_cfg=reference_cfg,
        ssfm_half_window=float(args.ssfm_half_window),
        n_t=int(args.n_t),
        n_z=int(args.n_z),
        n_slices=int(args.n_slices),
        compare_t_min=float(reference_cfg["t_min"]),
        compare_t_max=float(reference_cfg["t_max"]),
    )

    zeta = np.asarray(grid["zeta"], dtype=np.float64)
    regions = {
        "full_0to25": np.ones(len(zeta), dtype=bool),
        "old_0to20": zeta <= 4.0 + 1e-12,
        # Keep exactly the same boundary convention as the earlier quick558 evaluator.
        "new_20to25": zeta >= 4.0 - 1e-12,
        "terminal_25": np.arange(len(zeta)) == (len(zeta) - 1),
    }

    field_path = out_dir / "field_metrics.npy"
    power_path = out_dir / "power_metrics.npy"
    progress_path = out_dir / "progress.json"
    config_path = out_dir / "run_config.json"
    summary_path = out_dir / "summary.csv"

    if args.restart:
        for path in (field_path, power_path, progress_path, summary_path, out_dir / "summary_by_K.csv", out_dir / "comparison_by_budget.csv"):
            if path.exists():
                path.unlink()

    shape = (len(model_specs), len(REGION_NAMES), total)

    config_payload = {
        "script_version": SCRIPT_VERSION,
        "run_dir": str(run_dir),
        "n_samples": int(total),
        "selected_by_K": {str(int(k)): int(np.sum(selected_k == k)) for k in np.unique(selected_k)},
        "model_specs": [
            {"label": label, "checkpoint": str(path), "budget": budget, "initialization": init}
            for label, path, budget, init in model_specs
        ],
        "regions": REGION_NAMES,
        "grid": {
            "n_t": int(args.n_t),
            "n_z": int(args.n_z),
            "n_slices": int(args.n_slices),
            "ssfm_half_window": float(args.ssfm_half_window),
        },
        "metric_protocol": "Same as eval_0to25_budget_scratch_vs_from20_quick558.py; formal all-unseen evaluation.",
    }

    if config_path.is_file() and not args.restart:
        old_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if old_cfg.get("model_specs") != config_payload.get("model_specs") or int(old_cfg.get("n_samples", -1)) != total:
            raise RuntimeError("Existing resume directory was created for different checkpoints/data. Use a new --out-dir or --restart.")
    else:
        atomic_write_json(config_path, config_payload)

    if field_path.is_file() and power_path.is_file() and not args.restart:
        field_mm = np.load(field_path, mmap_mode="r+")
        power_mm = np.load(power_path, mmap_mode="r+")
        if field_mm.shape != shape or power_mm.shape != shape:
            raise RuntimeError("Resume array shape mismatch. Use --restart or a new output directory.")
    else:
        field_mm = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float64, shape=shape)
        power_mm = np.lib.format.open_memmap(power_path, mode="w+", dtype=np.float64, shape=shape)
        field_mm[:] = np.nan
        power_mm[:] = np.nan
        field_mm.flush()
        power_mm.flush()

    completed = 0
    if progress_path.is_file() and not args.restart:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed = int(progress.get("completed_samples", 0))
        completed = max(0, min(completed, total))

    if summary_path.is_file() and completed >= total and not args.restart:
        print("[done] Full evaluation already completed: %s" % summary_path, flush=True)
        return

    print("=" * 120, flush=True)
    print("FORMAL ALL-UNSEEN 0-25 km DOMAIN-EXTENSION EVALUATION", flush=True)
    print("Samples      : %d (ALL unseen)" % total, flush=True)
    print("By K         : %s" % {int(k): int(np.sum(selected_k == k)) for k in np.unique(selected_k)}, flush=True)
    print("Models       : %d" % len(model_specs), flush=True)
    print("Resume from  : %d/%d" % (completed, total), flush=True)
    print("Output       : %s" % out_dir, flush=True)
    print("=" * 120, flush=True)

    batch_size = int(args.ssfm_batch_size)
    checkpoint_every = max(1, int(args.checkpoint_every_batches))
    start_time = time.time()
    batch_counter = 0

    try:
        for start in range(completed, total, batch_size):
            end = min(total, start + batch_size)
            aa = selected_a[start:end]

            # Common SSFM reference: computed ONCE and reused by all six models.
            ref = base.run_ssfm_batch_selected(aa, grid, pde, device, False)

            tau32 = np.asarray(grid["tau"], dtype=np.float32)
            zeta32 = np.asarray(grid["zeta"], dtype=np.float32)

            for model_idx, (label, _, _, _) in enumerate(model_specs):
                pred = base.predict_ddnn_maps(
                    models[model_idx],
                    aa,
                    tau32,
                    zeta32,
                    device,
                    int(args.model_chunk_size),
                )

                for region_idx, region_name in enumerate(REGION_NAMES):
                    field, power = region_errors(pred, ref, regions[region_name])
                    field_mm[model_idx, region_idx, start:end] = field
                    power_mm[model_idx, region_idx, start:end] = power

                del pred
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            del ref
            if device.type == "cuda":
                torch.cuda.empty_cache()

            batch_counter += 1
            should_checkpoint = (batch_counter % checkpoint_every == 0) or (end >= total)
            if should_checkpoint:
                field_mm.flush()
                power_mm.flush()
                elapsed = time.time() - start_time
                done_this_run = end - completed
                rate = done_this_run / max(elapsed, 1e-9)
                eta = (total - end) / max(rate, 1e-12)
                atomic_write_json(
                    progress_path,
                    {
                        "script_version": SCRIPT_VERSION,
                        "completed_samples": int(end),
                        "total_samples": int(total),
                        "percent": 100.0 * float(end) / float(total),
                        "elapsed_this_run_sec": float(elapsed),
                        "rate_samples_per_sec": float(rate),
                        "eta_sec": float(eta),
                        "updated_at_unix": float(time.time()),
                    },
                )
                print(
                    "[eval] %d/%d (%.2f%%) | %.3f sample/s | ETA %.1f min"
                    % (end, total, 100.0 * end / total, rate, eta / 60.0),
                    flush=True,
                )

    except KeyboardInterrupt:
        field_mm.flush()
        power_mm.flush()
        print("\n[interrupted] Progress up to the last checkpoint is resumable. Rerun the same command/BAT.", flush=True)
        raise

    # Final summaries.
    summary_rows: List[Dict[str, Any]] = []
    by_k_rows: List[Dict[str, Any]] = []

    for model_idx, (label, _, budget, init) in enumerate(model_specs):
        for region_idx, region_name in enumerate(REGION_NAMES):
            field = np.asarray(field_mm[model_idx, region_idx, :], dtype=np.float64)
            power = np.asarray(power_mm[model_idx, region_idx, :], dtype=np.float64)
            if np.isnan(field).any() or np.isnan(power).any():
                raise RuntimeError("NaN/unfilled metrics remain for %s / %s" % (label, region_name))

            fs = aggregate(field, selected_k)
            ps = aggregate(power, selected_k)
            summary_rows.append({
                "method": label,
                "budget": budget,
                "initialization": init,
                "region": region_name,
                "n_samples": int(total),
                "field_sample_mean": fs["sample_mean"],
                "field_macro_equal_K_mean": fs["macro_equal_K_mean"],
                "field_median": fs["median"],
                "field_p95": fs["p95"],
                "field_max": fs["max"],
                "power_sample_mean": ps["sample_mean"],
                "power_macro_equal_K_mean": ps["macro_equal_K_mean"],
                "power_median": ps["median"],
                "power_p95": ps["p95"],
                "power_max": ps["max"],
            })

            for k in sorted(np.unique(selected_k).tolist()):
                mask = selected_k == k
                by_k_rows.append({
                    "method": label,
                    "budget": budget,
                    "initialization": init,
                    "region": region_name,
                    "K": int(k),
                    "n_samples": int(np.sum(mask)),
                    "field_mean": float(np.mean(field[mask])),
                    "power_mean": float(np.mean(power[mask])),
                    "field_median": float(np.median(field[mask])),
                    "power_median": float(np.median(power[mask])),
                })

    save_csv(summary_path, summary_rows)
    save_csv(out_dir / "summary_by_K.csv", by_k_rows)

    # Direct scratch-vs-from20 comparison for each budget/region.
    comparison_rows: List[Dict[str, Any]] = []
    for budget in ("10pct_actual", "25pct_actual", "50pct_actual"):
        for region_name in REGION_NAMES:
            scratch = next(r for r in summary_rows if r["budget"] == budget and r["initialization"] == "scratch" and r["region"] == region_name)
            from20 = next(r for r in summary_rows if r["budget"] == budget and r["initialization"] == "from20" and r["region"] == region_name)
            for metric in ("power_sample_mean", "power_macro_equal_K_mean", "field_sample_mean", "field_macro_equal_K_mean"):
                s = float(scratch[metric])
                f = float(from20[metric])
                comparison_rows.append({
                    "budget": budget,
                    "region": region_name,
                    "metric": metric,
                    "scratch": s,
                    "from20": f,
                    "absolute_reduction": s - f,
                    "relative_reduction_vs_scratch": (s - f) / s if s != 0 else np.nan,
                    "better": "from20" if f < s else ("scratch" if s < f else "tie"),
                })
    save_csv(out_dir / "comparison_by_budget.csv", comparison_rows)

    field_mm.flush()
    power_mm.flush()
    atomic_write_json(
        progress_path,
        {
            "script_version": SCRIPT_VERSION,
            "completed_samples": int(total),
            "total_samples": int(total),
            "percent": 100.0,
            "finished": True,
            "updated_at_unix": float(time.time()),
        },
    )

    print("\n" + "=" * 120, flush=True)
    print("FULL 78,624-SAMPLE EVALUATION COMPLETED", flush=True)
    for row in summary_rows:
        if row["region"] in ("full_0to25", "new_20to25", "terminal_25"):
            print(
                "%-24s %-12s | power micro=%.3f%% | power macro=%.3f%%"
                % (
                    row["method"],
                    row["region"],
                    100.0 * float(row["power_sample_mean"]),
                    100.0 * float(row["power_macro_equal_K_mean"]),
                ),
                flush=True,
            )
    print("Summary     : %s" % summary_path, flush=True)
    print("Comparison  : %s" % (out_dir / "comparison_by_budget.csv"), flush=True)
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()
