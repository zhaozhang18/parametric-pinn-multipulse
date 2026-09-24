# -*- coding: utf-8 -*-
"""One-command fixed-M 4-PAM normalized-amplitude forward/inverse workflow.

Definition used everywhere in this package
------------------------------------------
The four symbols 0.25, 0.50, 0.75 and 1.00 are NORMALIZED FIELD AMPLITUDES.
For pulse j,
    h_j(0,tau) = A_j * exp(-(tau-c_j)^2/2)
and the full initial field is the coherent sum over j. Thus the isolated peak
normalized powers are 0.0625, 0.25, 0.5625 and 1.0.

Pipeline
--------
1. Build seen/unseen amplitude combinations.
2. Search the minimum reliable SSFM half-window, n_t and n_z.
3. Apply n_t/n_z safety factors to obtain the formal evaluation grid.
4. Train no-Fourier and Fourier-K pure-physics forward PINNs.
5. Evaluate both models against SSFM on forward-unseen combinations.
6. Select the model with the lowest mean terminal power relative-L2 error.
7. Run five clearly separated inverse experiment groups and build one consolidated inverse summary:
   01_continuous_0to1_amplitude4
       Continuous A in [0,1], batched restarts/samples, then nearest-four-level diagnostic.
   02_enum_amplitude4
       Exhaustive enumeration of {0.25,0.5,0.75,1.0}^M with observations batched.
   03_fine10
       Same 10 fine-amplitude SSFM cases, with both discrete logits and
       continuous_round optimization.
   04_fine20
       Same 20-level idea, with both optimization modes.
   05_random_continuous_0to1_ssfm
       Ten truly continuous random amplitude vectors in [0,1]^M; SSFM observations,
       batched frozen-forward inversion, then SSFM reconstruction verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from nlse import make_and_save_seen_unseen, pulse_centers_t0, pinn_half_window_t0

SCRIPT_DIR = Path(__file__).resolve().parent


def slug(x: float) -> str:
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def pick_best_model(metrics_csv: Path, ckpts: dict[str, Path]) -> tuple[str, Path, dict[str, Any]]:
    grouped: dict[str, list[float]] = {}
    with metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                grouped.setdefault(str(row["model"]), []).append(float(row["rel_l2_power"]))
            except Exception:
                continue
    if not grouped:
        raise RuntimeError(f"No usable model/rel_l2_power rows in {metrics_csv}")
    stats = []
    for label, vals in grouped.items():
        arr = sorted(vals)
        n = len(arr)
        p95 = arr[min(n - 1, max(0, int(round(0.95 * (n - 1)))))]
        stats.append({
            "label": label,
            "checkpoint": str(ckpts[label]),
            "n": n,
            "mean_rel_l2_power": sum(arr) / n,
            "p95_rel_l2_power": p95,
            "max_rel_l2_power": max(arr),
        })
    stats.sort(key=lambda r: (r["mean_rel_l2_power"], r["p95_rel_l2_power"], r["max_rel_l2_power"]))
    best = stats[0]
    return str(best["label"]), Path(best["checkpoint"]), {"selected": best, "all_models": stats}



def collect_results_overview(run_dir: Path, best_info: dict[str, Any]) -> None:
    inverse_dir = run_dir / "inverse"
    overview: dict[str, Any] = {
        "level_quantity": "normalized_field_amplitude",
        "amplitude_levels": [0.25, 0.5, 0.75, 1.0],
        "isolated_peak_power_levels": [0.0625, 0.25, 0.5625, 1.0],
        "forward_selection": best_info,
        "inverse": {},
    }
    targets = {
        "continuous_0to1": inverse_dir / "01_continuous_0to1_amplitude4" / "summary.json",
        "enum_amplitude4": inverse_dir / "02_enum_amplitude4" / "summary.json",
        "fine10_both_modes": inverse_dir / "03_fine10" / "summary.json",
        "fine20_both_modes": inverse_dir / "04_fine20" / "summary.json",
        "random_continuous": inverse_dir / "05_random_continuous_0to1_ssfm" / "summary.json",
        "consolidated_summary": inverse_dir / "inverse_summary.json",
    }
    for key, path in targets.items():
        if path.exists():
            overview["inverse"][key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            overview["inverse"][key] = {"status": "not_finished", "expected_summary": str(path)}
    (run_dir / "results_overview.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-M normalized-amplitude 4-PAM forward + inverse workflow.")
    p.add_argument("-M", "--n-pulses", type=int, required=True)
    p.add_argument("--train-fraction", type=float, required=True)
    p.add_argument("--root-dir", default="./MULTIPULSE_AMPLITUDE_RUNS")
    p.add_argument("--run-name", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling-strategy", choices=["balanced", "random"], default="balanced")

    # Forward model/training.
    p.add_argument("--hidden", type=int, default=100)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--fourier-features", type=int, default=4)
    p.add_argument("--n-ic", type=int, default=7500)
    p.add_argument("--n-pde", type=int, default=115000)
    p.add_argument("--adam-steps", type=int, default=5000)
    p.add_argument("--lbfgs-epochs", type=int, default=1200)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--min-lbfgs-epochs", type=int, default=100)
    p.add_argument("--early-stop-eps", type=float, default=1e-8)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--resample-every", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--pinn-guard-t0", type=float, default=16.0)

    # SSFM grid search. The formal evaluation grid uses the recommended half-window
    # and multiplies recommended n_t/n_z by the safety factors (default 2x), matching
    # the original full pipeline logic.
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--force-grid", action="store_true")
    p.add_argument("--half-windows", default="auto")
    p.add_argument("--guards", default="4,8,12,16,20,24,28,32")
    p.add_argument("--nts", default="1024,2048,4096")
    p.add_argument("--nzs", default="250,500,1000,2000")
    p.add_argument("--ref-nt", type=int, default=16384)
    p.add_argument("--ref-nz", type=int, default=4000)
    p.add_argument("--ref-half-window", default="")
    p.add_argument("--ref-window-margin", type=float, default=20.0)
    p.add_argument("--tol-field", type=float, default=1e-4)
    p.add_argument("--tol-power", type=float, default=1e-4)
    p.add_argument("--tol-time-edge", type=float, default=1e-8)
    p.add_argument("--tol-freq-edge", type=float, default=1e-8)
    p.add_argument("--nt-safety", type=float, default=2.0)
    p.add_argument("--nz-safety", type=float, default=2.0)
    p.add_argument("--fixed-half-window", type=float, default=0.0)
    p.add_argument("--fixed-nt", type=int, default=0)
    p.add_argument("--fixed-nz", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0, help="0 = evaluate every unseen combination")
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--no-forward-plots", action="store_true")
    p.add_argument("--no-waterfalls", action="store_true")
    p.add_argument("--no-final-slices", action="store_true")

    # Inverse experiments.
    p.add_argument("--inverse-samples", type=int, default=10)
    p.add_argument("--inverse-sample-seed", type=int, default=2026)
    p.add_argument("--random-continuous-seed", type=int, default=2027)
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--inverse-epochs", type=int, default=3000)
    p.add_argument("--inverse-restarts", type=int, default=4)
    p.add_argument("--inverse-batch-mode", choices=["auto", "all_samples", "per_sample"], default="auto")
    p.add_argument("--inverse-early-stop-min-epochs", type=int, default=1000)
    p.add_argument("--inverse-early-stop-patience", type=int, default=300)
    p.add_argument("--inverse-early-stop-fraction", type=float, default=0.90)
    p.add_argument("--inverse-early-stop-rel-delta", type=float, default=1e-4)
    p.add_argument("--inverse-early-stop-abs-delta", type=float, default=1e-8)
    p.add_argument("--inverse-lr", type=float, default=3e-2)
    p.add_argument("--inverse-min-lr", type=float, default=5e-4)
    p.add_argument("--terminal-observable", choices=["power", "complex", "power_and_complex"], default="complex")
    p.add_argument("--enum-batch-size", type=int, default=1024)
    p.add_argument("--enum-sample-batch-size", type=int, default=10)
    p.add_argument("--forward-time-chunk", type=int, default=512)

    # Stage/resume controls.
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-inverse", action="store_true")
    p.add_argument("--force", action="store_true", help="rerun stages even when final output files exist")
    p.add_argument("--clear-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    M = int(args.n_pulses)
    if M < 2 or M > 8:
        raise ValueError("This project workflow expects M in [2,8].")
    if not (0.0 < float(args.train_fraction) < 1.0):
        raise ValueError("--train-fraction must be in (0,1).")

    run_name = args.run_name or f"M{M}_amp4_r{slug(args.train_fraction)}"
    run_dir = Path(args.root_dir).resolve() / run_name
    if args.clear_run and run_dir.exists():
        shutil.rmtree(run_dir)
    dataset_dir = run_dir / "dataset"
    grid_dir = run_dir / "grid_search"
    train_dir = run_dir / "train"
    eval_base = run_dir / "eval"
    inverse_dir = run_dir / "inverse"
    for d in (dataset_dir, grid_dir, train_dir, eval_base, inverse_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Dataset.
    ds_summary_path = dataset_dir / "dataset_summary.json"
    if args.force or not ds_summary_path.exists():
        ds_summary = make_and_save_seen_unseen(
            M,
            dataset_dir,
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
            sampling_strategy=str(args.sampling_strategy),
        )
    else:
        ds_summary = json.loads(ds_summary_path.read_text(encoding="utf-8"))
    seen_csv = dataset_dir / "seen_combinations.csv"
    unseen_csv = dataset_dir / "unseen_combinations.csv"

    # 2) Search the SSFM grid, then apply the same safety rule as the original pipeline.
    centers = pulse_centers_t0(M)
    compare_half = pinn_half_window_t0(M, guard_t0=float(args.pinn_guard_t0))
    rec_json = grid_dir / f"recommendation_M{M}.json"
    if not args.skip_grid and (args.force_grid or not rec_json.exists()):
        cmd = [
            sys.executable, str(SCRIPT_DIR / "find_min_ssfm_grid_multi_pulse.py"),
            "--device", str(args.device),
            "--pulse-counts", str(M),
            "--z-max-ld", "4.0",
            "--pulse-spacing-t0", "8.0",
            "--compare-t-min", str(-compare_half),
            "--compare-t-max", str(compare_half),
            "--half-windows", str(args.half_windows),
            "--guards", str(args.guards),
            "--nts", str(args.nts),
            "--nzs", str(args.nzs),
            "--ref-nt", str(args.ref_nt),
            "--ref-nz", str(args.ref_nz),
            "--ref-window-margin", str(args.ref_window_margin),
            "--tol-field", str(args.tol_field),
            "--tol-power", str(args.tol_power),
            "--tol-time-edge", str(args.tol_time_edge),
            "--tol-freq-edge", str(args.tol_freq_edge),
            "--quiet-ssfm",
            "--out-dir", str(grid_dir),
        ]
        if str(args.ref_half_window).strip():
            cmd += ["--ref-half-window", str(args.ref_half_window)]
        run_cmd(cmd)
    elif args.skip_grid:
        print("\nSkipping grid search by request; an existing recommendation file is required.")

    if not rec_json.exists():
        raise FileNotFoundError(
            f"Missing grid recommendation: {rec_json}. Run without --skip-grid first."
        )
    rec_payload = json.loads(rec_json.read_text(encoding="utf-8"))
    rec = rec_payload.get("recommendation")
    if not rec:
        raise RuntimeError(
            f"No grid candidate passed in {rec_json}. Increase the candidate ranges or relax tolerances."
        )

    eval_half = (
        float(args.fixed_half_window)
        if float(args.fixed_half_window) > 0
        else float(rec["half_window_t0"])
    )
    eval_n_t = (
        int(args.fixed_nt)
        if int(args.fixed_nt) > 0
        else int(round(float(args.nt_safety) * int(rec["n_t"])))
    )
    eval_n_z = (
        int(args.fixed_nz)
        if int(args.fixed_nz) > 0
        else int(round(float(args.nz_safety) * int(rec["n_z"])))
    )
    eval_grid = {
        "M": M,
        "level_quantity": "normalized_field_amplitude",
        "amplitude_levels": [0.25, 0.5, 0.75, 1.0],
        "isolated_peak_power_levels": [0.0625, 0.25, 0.5625, 1.0],
        "centers_t0": list(centers),
        "minimum_recommendation": rec,
        "safety_rule": {
            "nt_safety": float(args.nt_safety),
            "nz_safety": float(args.nz_safety),
            "half_window_safety": "use recommended half-window unchanged",
        },
        "eval_half_window_t0": float(eval_half),
        "eval_total_window_t0": float(2.0 * eval_half),
        "eval_n_t": int(eval_n_t),
        "eval_n_z": int(eval_n_z),
        "compare_t_min": float(-compare_half),
        "compare_t_max": float(compare_half),
        "z_max_ld": 4.0,
    }
    (run_dir / "ssfm_eval_grid.json").write_text(
        json.dumps(eval_grid, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSelected formal SSFM evaluation grid:")
    print(json.dumps(eval_grid, indent=2, ensure_ascii=False))

    # 3) Train two forward models.
    ckpts: dict[str, Path] = {}
    settings = [("no_fourier", 0), (f"fourier{int(args.fourier_features)}", int(args.fourier_features))]
    for label, ff in settings:
        model_dir = train_dir / (
            f"M{M}_{label}_amp4_net{args.layers}x{args.hidden}_seen{int(ds_summary['n_seen'])}"
            f"_ic{args.n_ic}_pde{args.n_pde}"
        )
        ckpt = model_dir / "forward_pinn.pt"
        ckpts[label] = ckpt
        if not args.skip_train and (args.force or not ckpt.exists()):
            run_cmd([
                sys.executable, str(SCRIPT_DIR / "train_multi_pulse_pinn.py"),
                "-M", str(M),
                "--seen-csv", str(seen_csv),
                "--out-dir", str(model_dir),
                "--device", str(args.device),
                "--seed", str(args.seed),
                "--train-fraction", str(args.train_fraction),
                "--hidden", str(args.hidden),
                "--layers", str(args.layers),
                "--fourier-features", str(ff),
                "--auto-t-window",
                "--pinn-guard-t0", str(args.pinn_guard_t0),
                "--n-ic", str(args.n_ic),
                "--n-pde", str(args.n_pde),
                "--adam-steps", str(args.adam_steps),
                "--lbfgs-epochs", str(args.lbfgs_epochs),
                "--lbfgs-max-iter", str(args.lbfgs_max_iter),
                "--min-lbfgs-epochs", str(args.min_lbfgs_epochs),
                "--early-stop-eps", str(args.early_stop_eps),
                "--early-stop-patience", str(args.early_stop_patience),
                "--resample-every", str(args.resample_every),
                "--log-every", str(args.log_every),
            ])
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing forward checkpoint: {ckpt}")

    # 4) Forward evaluation.
    eval_out = eval_base / (
        f"M{M}_amp4_eval_win{slug(eval_half)}T0_nt{eval_n_t}_nz{eval_n_z}"
        f"_compare{slug(compare_half)}T0"
    )
    metrics_csv = eval_out / "metrics_stream.csv"
    if not args.skip_eval and (args.force or not metrics_csv.exists()):
        labels = list(ckpts.keys())
        cmd = [
            sys.executable, str(SCRIPT_DIR / "stream_eval_multi_pulse_vs_ssfm_enhanced.py"),
            "-M", str(M),
            "--unseen-csv", str(unseen_csv),
            "--model-labels", *labels,
            "--model-paths", *[str(ckpts[x]) for x in labels],
            "--device", str(args.device),
            "--t-window-t0", str(eval_half),
            "--n-t", str(eval_n_t),
            "--n-z", str(eval_n_z),
            "--compare-t-min", str(-compare_half),
            "--compare-t-max", str(compare_half),
            "--manual-compare-window",
            "--chunk-size", str(args.chunk_size),
            "--out-dir", str(eval_out),
            "--resume",
        ]
        if int(args.max_eval_samples) > 0:
            cmd += ["--max-samples", str(args.max_eval_samples)]
        if args.no_forward_plots:
            cmd += ["--no-plots"]
        if args.no_waterfalls:
            cmd += ["--no-waterfalls"]
        if args.no_final_slices:
            cmd += ["--no-final-slices"]
        run_cmd(cmd)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing forward metrics: {metrics_csv}")

    best_label, best_ckpt, best_info = pick_best_model(metrics_csv, ckpts)
    (run_dir / "selected_forward_for_inverse.json").write_text(
        json.dumps(best_info, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest = {
        "pipeline_version": "amplitude4_v4_inverse_summary_random_continuous",
        "level_quantity": "normalized_field_amplitude",
        "amplitude_levels": [0.25, 0.5, 0.75, 1.0],
        "isolated_peak_power_levels": [0.0625, 0.25, 0.5625, 1.0],
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "grid_dir": str(grid_dir),
        "grid_recommendation_json": str(rec_json),
        "train_dir": str(train_dir),
        "eval_dir": str(eval_out),
        "seen_csv": str(seen_csv),
        "unseen_csv": str(unseen_csv),
        "forward_checkpoints": {k: str(v) for k, v in ckpts.items()},
        "forward_metrics_csv": str(metrics_csv),
        "selected_forward_label": best_label,
        "selected_forward_checkpoint": str(best_ckpt),
        "ssfm_eval_grid": eval_grid,
        "inverse_layout": {
            "01_continuous_0to1_amplitude4": "continuous A in [0,1]",
            "02_enum_amplitude4": "exhaustive 4-level amplitude enumeration",
            "03_fine10": "10 amplitude levels; discrete and continuous_round",
            "04_fine20": "20 amplitude levels; discrete and continuous_round",
            "05_random_continuous_0to1_ssfm": "continuous random A in [0,1], SSFM observations and SSFM reconstruction verification",
            "inverse_summary": "consolidated method/sample tables and plots",
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 5) Inverse experiment groups.
    if not args.skip_inverse:
        common_dataset = inverse_dir / f"shared_ssfm_amplitude4_N{args.inverse_samples}_seed{args.inverse_sample_seed}"
        common = [
            "--run-dir", str(run_dir),
            "-M", str(M),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--forward-model-path", str(best_ckpt),
            "--forward-metrics-csv", str(metrics_csv),
            "--unseen-csv", str(unseen_csv),
            "--n-samples", str(args.inverse_samples),
            "--sample-seed", str(args.inverse_sample_seed),
            "--terminal-observable", str(args.terminal_observable),
            "--inverse-dataset-mode", "selected",
            "--inverse-dataset-dir", str(common_dataset),
            "--inverse-input-points", str(args.inverse_input_points),
            "--t-window-t0", str(eval_half),
            "--n-t", str(eval_n_t),
            "--n-z", str(eval_n_z),
            "--z-max-ld", "4.0",
            "--compare-t-min", str(-compare_half),
            "--compare-t-max", str(compare_half),
            "--forward-time-chunk", str(args.forward_time_chunk),
        ]

        out_cont = inverse_dir / "01_continuous_0to1_amplitude4"
        if args.force or not (out_cont / "summary.json").exists():
            cmd = [
                sys.executable, str(SCRIPT_DIR / "inverse_idea1a_continuous_terminal_only.py"),
                *common,
                "--out-dir", str(out_cont),
                "--p-min", "0.0", "--p-max", "1.0",
                "--epochs", str(args.inverse_epochs),
                "--restarts", str(args.inverse_restarts),
                "--lr", str(args.inverse_lr),
                "--min-lr", str(args.inverse_min_lr),
                "--batch-mode", str(args.inverse_batch_mode),
                "--early-stop",
                "--early-stop-min-epochs", str(args.inverse_early_stop_min_epochs),
                "--early-stop-patience", str(args.inverse_early_stop_patience),
                "--early-stop-fraction", str(args.inverse_early_stop_fraction),
                "--early-stop-min-delta", str(args.inverse_early_stop_rel_delta),
                "--early-stop-abs-delta", str(args.inverse_early_stop_abs_delta),
            ]
            if args.force:
                cmd += ["--rebuild-inverse-dataset"]
            run_cmd(cmd)

        out_enum = inverse_dir / "02_enum_amplitude4"
        if args.force or not (out_enum / "summary.json").exists():
            cmd = [
                sys.executable, str(SCRIPT_DIR / "inverse_enum_amplitude4_terminal_only.py"),
                *common,
                "--out-dir", str(out_enum),
                "--batch-size", str(args.enum_batch_size),
                "--sample-batch-size", str(args.enum_sample_batch_size),
            ]
            if args.force:
                cmd += ["--rebuild-inverse-dataset"]
            run_cmd(cmd)

        for folder, levels in [
            ("03_fine10", ",".join(f"{i/10:.2f}" for i in range(1, 11))),
            ("04_fine20", ",".join(f"{i/20:.2f}" for i in range(1, 21))),
        ]:
            out_fine = inverse_dir / folder
            if args.force or not (out_fine / "summary.json").exists():
                cmd = [
                    sys.executable, str(SCRIPT_DIR / "inverse_idea1b_fine_levels_terminal_only.py"),
                    "--run-dir", str(run_dir),
                    "-M", str(M),
                    "--device", str(args.device),
                    "--seed", str(args.seed),
                    "--forward-model-path", str(best_ckpt),
                    "--forward-metrics-csv", str(metrics_csv),
                    "--out-dir", str(out_fine),
                    "--terminal-observable", str(args.terminal_observable),
                    "--levels", levels,
                    "--n-samples", str(args.inverse_samples),
                    "--sample-seed", str(args.inverse_sample_seed),
                    "--inverse-input-points", str(args.inverse_input_points),
                    "--t-window-t0", str(eval_half),
                    "--n-t", str(eval_n_t),
                    "--n-z", str(eval_n_z),
                    "--z-max-ld", "4.0",
                    "--compare-t-min", str(-compare_half),
                    "--compare-t-max", str(compare_half),
                    "--run-mode", "both",
                    "--epochs", str(args.inverse_epochs),
                    "--restarts", str(args.inverse_restarts),
                    "--lr", str(args.inverse_lr),
                    "--min-lr", str(args.inverse_min_lr),
                    "--batch-mode", str(args.inverse_batch_mode),
                    "--early-stop",
                    "--early-stop-min-epochs", str(args.inverse_early_stop_min_epochs),
                    "--early-stop-patience", str(args.inverse_early_stop_patience),
                    "--early-stop-fraction", str(args.inverse_early_stop_fraction),
                    "--early-stop-min-delta", str(args.inverse_early_stop_rel_delta),
                    "--early-stop-abs-delta", str(args.inverse_early_stop_abs_delta),
                    "--forward-time-chunk", str(args.forward_time_chunk),
                ]
                if args.force:
                    cmd += ["--rebuild-ssfm"]
                run_cmd(cmd)

        out_random = inverse_dir / "05_random_continuous_0to1_ssfm"
        if args.force or not (out_random / "summary.json").exists():
            cmd = [
                sys.executable, str(SCRIPT_DIR / "inverse_random_continuous_ssfm_terminal_only.py"),
                "--run-dir", str(run_dir),
                "-M", str(M),
                "--device", str(args.device),
                "--seed", str(args.seed),
                "--sample-seed", str(args.random_continuous_seed),
                "--n-samples", str(args.inverse_samples),
                "--out-dir", str(out_random),
                "--forward-model-path", str(best_ckpt),
                "--forward-metrics-csv", str(metrics_csv),
                "--terminal-observable", str(args.terminal_observable),
                "--inverse-input-points", str(args.inverse_input_points),
                "--t-window-t0", str(eval_half),
                "--n-t", str(eval_n_t),
                "--n-z", str(eval_n_z),
                "--z-max-ld", "4.0",
                "--compare-t-min", str(-compare_half),
                "--compare-t-max", str(compare_half),
                "--epochs", str(args.inverse_epochs),
                "--restarts", str(args.inverse_restarts),
                "--batch-mode", str(args.inverse_batch_mode),
                "--lr", str(args.inverse_lr),
                "--min-lr", str(args.inverse_min_lr),
                "--early-stop",
                "--early-stop-min-epochs", str(args.inverse_early_stop_min_epochs),
                "--early-stop-patience", str(args.inverse_early_stop_patience),
                "--early-stop-fraction", str(args.inverse_early_stop_fraction),
                "--early-stop-min-delta", str(args.inverse_early_stop_rel_delta),
                "--early-stop-abs-delta", str(args.inverse_early_stop_abs_delta),
                "--forward-time-chunk", str(args.forward_time_chunk),
            ]
            if args.force:
                cmd += ["--rebuild-random-dataset"]
            run_cmd(cmd)

        run_cmd([
            sys.executable, str(SCRIPT_DIR / "summarize_fixedM_inverse_results.py"),
            "--run-dir", str(run_dir),
        ])

    collect_results_overview(run_dir, best_info)
    layout_text = """INVERSE RESULT LAYOUT\n=====================\n01_continuous_0to1_amplitude4/  continuous amplitude A in [0,1] on PAM4 samples\n02_enum_amplitude4/             exhaustive {0.25,0.5,0.75,1.0}^M search\n03_fine10/                      10-level discrete + continuous_round\n04_fine20/                      20-level discrete + continuous_round\n05_random_continuous_0to1_ssfm/ random continuous amplitudes + SSFM reconstruction verification\ninverse_methods_summary.csv      one row per inverse method/mode\ninverse_samples_summary.csv      all per-sample rows combined\ninverse_summary.json             consolidated machine-readable summary\ninverse_summary_report.md        consolidated readable table\nsummary_plots/                    accuracy, MAE, runtime and terminal-error figures\nshared_ssfm_amplitude4_N<samples>_seed<seed>/  shared SSFM observations for 01 and 02\n"""
    (inverse_dir / "README_RESULTS_LAYOUT.txt").write_text(layout_text, encoding="utf-8")

    print("\n========== AMPLITUDE 4-PAM PIPELINE FINISHED ==========")
    print(f"run_dir      -> {run_dir}")
    print(f"best forward -> {best_label}: {best_ckpt}")
    print(f"manifest     -> {run_dir / 'run_manifest.json'}")
    print(f"inverse      -> {inverse_dir}")


if __name__ == "__main__":
    main()
