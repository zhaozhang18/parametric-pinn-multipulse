# -*- coding: utf-8 -*-
"""Run every inverse strategy for one completed fixed-M forward experiment.

This script does not train or evaluate the forward PINN. It reads the completed
``run_manifest.json`` and runs every inverse strategy in one command.

Runtime reporting follows the "problem-solving time" convention:
* target SSFM generation is treated as preparation of the test question;
* optional SSFM reconstruction is treated as post-solution verification;
* the comparison table reports only inverse optimization/search time.

A consolidated CSV/JSON/Markdown/table-image summary is generated automatically.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: list[str]) -> None:
    print("\n$ " + " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all frozen-forward inverse methods for an existing fixed-M run.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inverse-samples", type=int, default=10)
    p.add_argument("--inverse-sample-seed", type=int, default=2026)
    p.add_argument("--random-continuous-seed", type=int, default=2027)
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--inverse-epochs", type=int, default=3000, help="Maximum budget; automatic stopping may finish earlier.")
    p.add_argument("--inverse-restarts", type=int, default=4)
    p.add_argument("--inverse-batch-mode", choices=["auto", "all_samples", "per_sample"], default="auto")
    p.add_argument("--inverse-lr", type=float, default=3e-2)
    p.add_argument("--inverse-min-lr", type=float, default=5e-4)
    p.add_argument("--inverse-early-stop-min-epochs", type=int, default=1000)
    p.add_argument("--inverse-early-stop-patience", type=int, default=300)
    p.add_argument("--inverse-early-stop-fraction", type=float, default=0.90)
    p.add_argument("--inverse-early-stop-rel-delta", type=float, default=1e-4)
    p.add_argument("--inverse-early-stop-abs-delta", type=float, default=1e-8)
    p.add_argument("--terminal-observable", choices=["power", "complex", "power_and_complex"], default="complex")
    p.add_argument("--enum-batch-size", type=int, default=1024)
    p.add_argument("--enum-sample-batch-size", type=int, default=10)
    p.add_argument("--forward-time-chunk", type=int, default=512)
    p.add_argument("--force", action="store_true", help="Backward compatible: rerun all methods and rebuild SSFM target datasets.")
    p.add_argument("--rerun-all", action="store_true", help="Rerun every inverse method but reuse existing SSFM target datasets.")
    p.add_argument("--rebuild-ssfm", action="store_true", help="Rebuild SSFM target datasets; also reruns all methods.")
    p.add_argument("--summary-only", action="store_true", help="Do not rerun inverse methods; regenerate only consolidated tables and plots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "run_manifest.json"
    grid_path = run_dir / "ssfm_eval_grid.json"
    if not manifest_path.exists() or not grid_path.exists():
        raise FileNotFoundError(
            "A completed fixed-M run is required. Missing run_manifest.json or ssfm_eval_grid.json."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    M = int(grid["M"])

    def manifest_path(value: str) -> Path:
        """Resolve paths written by both old and new forward pipelines.

        Older manifests store paths relative to the project root, for example::

            <project-root>/MULTIPULSE_AMPLITUDE_RUNS/M4_amp4_r0p1/train/.../forward_pinn.pt

        Newer manifests may store paths relative to ``run_dir`` such as::

            train/.../forward_pinn.pt

        Blindly joining every relative path to ``run_dir`` duplicates the run
        prefix for old manifests.  Try all supported bases and return the first
        existing path.
        """
        import os

        raw = str(value).strip()
        # Normalise separators so copied result folders also work across OSes.
        q = Path(raw.replace("\\", os.sep).replace("/", os.sep))
        candidates: list[Path] = []

        # Stored paths may refer to the machine on which the forward run was
        # created, while the result folder may later be moved to another project
        # root. Use the stored absolute path only when it still exists; otherwise
        # rebuild the suffix under the current run_dir.
        if q.is_absolute():
            candidates.append(q.resolve())
        else:
            # Old manifest format: path relative to the project/script directory.
            candidates.append((SCRIPT_DIR / q).resolve())
            candidates.append((Path.cwd() / q).resolve())

            # New manifest format: path relative to the selected run directory.
            candidates.append((run_dir / q).resolve())

        # Robust fallback for both absolute and relative stored paths: if the
        # path contains the current run-folder name, strip everything through
        # that name and join the remaining suffix to the current run_dir.
        parts = q.parts
        matching = [i for i, part in enumerate(parts) if part == run_dir.name]
        if matching:
            suffix = Path(*parts[matching[-1] + 1 :])
            candidates.insert(0, (run_dir / suffix).resolve())

        # Last fallback for copied/renamed runs: locate the filename under the
        # current run directory. This is safe because the manifest still
        # supplies the expected basename (e.g. forward_pinn.pt or metrics CSV).
        if q.name:
            matches = list(run_dir.rglob(q.name))
            candidates.extend(x.resolve() for x in matches)

        seen: set[str] = set()
        unique: list[Path] = []
        for candidate in candidates:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)

        for candidate in unique:
            if candidate.exists():
                return candidate

        attempted = "\n  - ".join(str(x) for x in unique)
        raise FileNotFoundError(
            f"Could not resolve manifest path: {value}\nAttempted:\n  - {attempted}"
        )

    forward_ckpt = manifest_path(manifest["selected_forward_checkpoint"])
    metrics_csv = manifest_path(manifest["forward_metrics_csv"])
    unseen_csv = manifest_path(manifest["unseen_csv"])
    for q in (forward_ckpt, metrics_csv, unseen_csv):
        if not q.exists():
            raise FileNotFoundError(q)

    inverse_dir = run_dir / "inverse"
    inverse_dir.mkdir(parents=True, exist_ok=True)
    common_dataset = inverse_dir / f"shared_ssfm_amplitude4_N{args.inverse_samples}_seed{args.inverse_sample_seed}"
    common = [
        "--run-dir", str(run_dir),
        "-M", str(M),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--forward-model-path", str(forward_ckpt),
        "--forward-metrics-csv", str(metrics_csv),
        "--unseen-csv", str(unseen_csv),
        "--n-samples", str(args.inverse_samples),
        "--sample-seed", str(args.inverse_sample_seed),
        "--terminal-observable", str(args.terminal_observable),
        "--inverse-dataset-mode", "selected",
        "--inverse-dataset-dir", str(common_dataset),
        "--inverse-input-points", str(args.inverse_input_points),
        "--t-window-t0", str(grid["eval_half_window_t0"]),
        "--n-t", str(grid["eval_n_t"]),
        "--n-z", str(grid["eval_n_z"]),
        "--z-max-ld", str(grid.get("z_max_ld", 4.0)),
        "--compare-t-min", str(grid["compare_t_min"]),
        "--compare-t-max", str(grid["compare_t_max"]),
        "--forward-time-chunk", str(args.forward_time_chunk),
    ]
    early = [
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
    ]

    rerun_all = bool(args.force or args.rerun_all or args.rebuild_ssfm)
    rebuild_ssfm = bool(args.force or args.rebuild_ssfm)

    out_cont = inverse_dir / "01_pam4_target_continuous_blind_0to1"
    if (not args.summary_only) and (rerun_all or not (out_cont / "summary.json").exists()):
        cmd = [
            sys.executable, str(SCRIPT_DIR / "inverse_idea1a_continuous_terminal_only.py"),
            *common, "--out-dir", str(out_cont), "--p-min", "0.0", "--p-max", "1.0", *early,
        ]
        if rebuild_ssfm:
            cmd += ["--rebuild-inverse-dataset"]
        run_cmd(cmd)

    out_enum = inverse_dir / "02_pam4_exhaustive_enumeration"
    if (not args.summary_only) and (rerun_all or not (out_enum / "summary.json").exists()):
        cmd = [
            sys.executable, str(SCRIPT_DIR / "inverse_enum_amplitude4_terminal_only.py"),
            *common, "--out-dir", str(out_enum),
            "--batch-size", str(args.enum_batch_size),
            "--sample-batch-size", str(args.enum_sample_batch_size),
        ]
        run_cmd(cmd)

    for folder, levels in [
        ("03_levels10_two_optimizers", ",".join(f"{i/10:.2f}" for i in range(1, 11))),
        ("04_levels20_two_optimizers", ",".join(f"{i/20:.2f}" for i in range(1, 21))),
    ]:
        out_fine = inverse_dir / folder
        if (not args.summary_only) and (rerun_all or not (out_fine / "summary.json").exists()):
            cmd = [
                sys.executable, str(SCRIPT_DIR / "inverse_idea1b_fine_levels_terminal_only.py"),
                "--run-dir", str(run_dir),
                "-M", str(M),
                "--device", str(args.device),
                "--seed", str(args.seed),
                "--forward-model-path", str(forward_ckpt),
                "--forward-metrics-csv", str(metrics_csv),
                "--out-dir", str(out_fine),
                "--terminal-observable", str(args.terminal_observable),
                "--levels", levels,
                "--n-samples", str(args.inverse_samples),
                "--sample-seed", str(args.inverse_sample_seed),
                "--inverse-input-points", str(args.inverse_input_points),
                "--t-window-t0", str(grid["eval_half_window_t0"]),
                "--n-t", str(grid["eval_n_t"]),
                "--n-z", str(grid["eval_n_z"]),
                "--z-max-ld", str(grid.get("z_max_ld", 4.0)),
                "--compare-t-min", str(grid["compare_t_min"]),
                "--compare-t-max", str(grid["compare_t_max"]),
                "--run-mode", "both",
                *early,
                "--forward-time-chunk", str(args.forward_time_chunk),
            ]
            if rebuild_ssfm:
                cmd += ["--rebuild-ssfm"]
            run_cmd(cmd)

    out_random = inverse_dir / "05_random_continuous_target_blind_0to1"
    if (not args.summary_only) and (rerun_all or not (out_random / "summary.json").exists()):
        cmd = [
            sys.executable, str(SCRIPT_DIR / "inverse_random_continuous_ssfm_terminal_only.py"),
            "--run-dir", str(run_dir),
            "-M", str(M),
            "--device", str(args.device),
            "--seed", str(args.seed),
            "--sample-seed", str(args.random_continuous_seed),
            "--n-samples", str(args.inverse_samples),
            "--out-dir", str(out_random),
            "--forward-model-path", str(forward_ckpt),
            "--forward-metrics-csv", str(metrics_csv),
            "--terminal-observable", str(args.terminal_observable),
            "--inverse-input-points", str(args.inverse_input_points),
            "--t-window-t0", str(grid["eval_half_window_t0"]),
            "--n-t", str(grid["eval_n_t"]),
            "--n-z", str(grid["eval_n_z"]),
            "--z-max-ld", str(grid.get("z_max_ld", 4.0)),
            "--compare-t-min", str(grid["compare_t_min"]),
            "--compare-t-max", str(grid["compare_t_max"]),
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
        if rebuild_ssfm:
            cmd += ["--rebuild-random-dataset"]
        run_cmd(cmd)

    # One consolidated summary across every completed inverse method.
    run_cmd([
        sys.executable, str(SCRIPT_DIR / "summarize_fixedM_inverse_results.py"),
        "--run-dir", str(run_dir),
    ])

    layout = {
        "run_dir": str(run_dir),
        "M": M,
        "forward_checkpoint": str(forward_ckpt),
        "inverse_outputs": {
            "pam4_target_continuous_blind": str(out_cont),
            "pam4_exhaustive_enumeration": str(out_enum),
            "levels10_two_optimizers": str(inverse_dir / "03_levels10_two_optimizers"),
            "levels20_two_optimizers": str(inverse_dir / "04_levels20_two_optimizers"),
            "random_continuous_target_blind": str(inverse_dir / "05_random_continuous_target_blind_0to1"),
            "consolidated_summary": str(inverse_dir / "inverse_summary.json"),
        },
        "random_continuous_seed": int(args.random_continuous_seed),
        "batching": {
            "optimization": args.inverse_batch_mode,
            "samples": args.inverse_samples,
            "restarts_per_sample": args.inverse_restarts,
            "nominal_joint_restarts": args.inverse_samples * args.inverse_restarts,
        },
        "runtime_reporting": {
            "reported_runtime": "inverse optimization/search time only",
            "excluded": ["target SSFM generation", "SSFM reconstruction verification"],
        },
    }
    (inverse_dir / "inverse_all_run_manifest.json").write_text(
        json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(layout, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
