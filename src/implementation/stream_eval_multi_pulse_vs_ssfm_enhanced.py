# -*- coding: utf-8 -*-
"""
stream_eval_multi_pulse_vs_ssfm_enhanced.py

Enhanced streaming evaluation for M-pulse forward PINNs against SSFM.

What this version adds compared with the original evaluator
-----------------------------------------------------------
1. Keeps the original streaming metric CSV behavior.
2. Adds visual metric comparisons for no-Fourier vs Fourier PINNs.
3. Finds the best and worst unseen case for every forward model.
4. Re-runs those selected cases and saves 3D waterfall plots comparing SSFM
   and PINN over the propagation distance.
5. Saves final-position 2D slice plots at z = z_max = 4 L_D for the same
   best/worst cases of every model.

The script does not save full SSFM datasets for every unseen sample. It only
re-generates SSFM trajectories for the selected best/worst visualization cases.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from nlse import NLSEParams, load_combinations_csv, make_and_save_seen_unseen, pinn_half_window_t0
from ssfm import run_ssfm
from train_multi_pulse_pinn import load_forward_checkpoint


def _safe_device(device_text: str) -> torch.device:
    return torch.device(device_text if (torch.cuda.is_available() or device_text == "cpu") else "cpu")


def parse_levels(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in str(text).replace(",", ";").split(";") if str(x).strip())


def combo_to_text(combo: Sequence[float]) -> str:
    return ";".join(f"{float(x):g}" for x in combo)


def predict_model(
    model,
    tau: np.ndarray,
    zeta: float,
    combo: Sequence[float],
    device: torch.device,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Predict complex field h(zeta, tau) for one normalized-amplitude combination."""
    outs: list[np.ndarray] = []
    amplitudes_np = np.asarray(combo, dtype=np.float32).reshape(1, -1)
    with torch.no_grad():
        for start in range(0, len(tau), int(chunk_size)):
            end = min(len(tau), start + int(chunk_size))
            n = end - start
            z = torch.full((n, 1), float(zeta), dtype=torch.float32, device=device)
            t = torch.tensor(tau[start:end].reshape(-1, 1), dtype=torch.float32, device=device)
            p = torch.tensor(np.repeat(amplitudes_np, n, axis=0), dtype=torch.float32, device=device)
            u, v = model(z, t, p)
            h = u.detach().cpu().numpy().reshape(-1) + 1j * v.detach().cpu().numpy().reshape(-1)
            outs.append(h)
    return np.concatenate(outs, axis=0)


def predict_model_grid(
    model,
    tau: np.ndarray,
    zeta_values: np.ndarray,
    combo: Sequence[float],
    device: torch.device,
    chunk_size: int = 65536,
) -> np.ndarray:
    """Predict complex field on a zeta x tau grid."""
    rows = []
    for zeta in zeta_values:
        rows.append(predict_model(model, tau, float(zeta), combo, device, chunk_size=chunk_size))
    return np.asarray(rows, dtype=np.complex128)


def compute_metrics(h_pred: np.ndarray, h_ref: np.ndarray) -> dict:
    """Metrics consistent with the previous evaluator and the paper-style power errors."""
    rel_l2_field = float(np.linalg.norm(h_pred - h_ref) / (np.linalg.norm(h_ref) + 1e-300))
    P_pred = np.abs(h_pred) ** 2
    P_ref = np.abs(h_ref) ** 2
    rel_l2_power = float(np.linalg.norm(P_pred - P_ref) / (np.linalg.norm(P_ref) + 1e-300))
    return {
        "rel_l2_field": rel_l2_field,
        "rel_l2_power": rel_l2_power,
        "e1": rel_l2_power,
        "e2": float(np.max(np.abs(P_pred - P_ref))),
        "mse_power": float(np.mean((P_pred - P_ref) ** 2)),
        "mae_power": float(np.mean(np.abs(P_pred - P_ref))),
    }


def summarize_metrics(csv_path: Path, out_json: Path) -> dict:
    import pandas as pd

    df = pd.read_csv(csv_path)
    numeric_cols = [
        "rel_l2_field", "rel_l2_power", "e1", "e2", "mse_power", "mae_power", "ssfm_sec", "pinn_sec"
    ]
    summary: dict = {"models": {}, "n_metric_rows": int(len(df))}
    if len(df) == 0:
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    for model, g in df.groupby("model"):
        d: dict = {"n_rows": int(len(g))}
        for col in numeric_cols:
            vals = g[col].astype(float)
            d[f"{col}_mean"] = float(vals.mean())
            d[f"{col}_std"] = float(vals.std(ddof=0))
            d[f"{col}_min"] = float(vals.min())
            d[f"{col}_p50"] = float(vals.quantile(0.50))
            d[f"{col}_p90"] = float(vals.quantile(0.90))
            d[f"{col}_p95"] = float(vals.quantile(0.95))
            d[f"{col}_max"] = float(vals.max())
        best_row = g.loc[g["rel_l2_power"].astype(float).idxmin()]
        worst_row = g.loc[g["rel_l2_power"].astype(float).idxmax()]
        d["best_idx"] = int(best_row["idx"])
        d["best_levels"] = str(best_row["levels"])
        d["best_rel_l2_power"] = float(best_row["rel_l2_power"])
        d["worst_idx"] = int(worst_row["idx"])
        d["worst_levels"] = str(worst_row["levels"])
        d["worst_rel_l2_power"] = float(worst_row["rel_l2_power"])
        summary["models"][str(model)] = d

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def plot_metric_figures(metrics_path: Path, out_dir: Path) -> list[str]:
    """Create simple visual comparisons of model metrics."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(metrics_path)
    if df.empty:
        return []
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    models = list(df["model"].astype(str).unique())

    # Boxplot: rel-L2 power error distribution.
    fig, ax = plt.subplots(figsize=(8, 5))
    data = [df.loc[df["model"].astype(str) == m, "rel_l2_power"].astype(float).values for m in models]
    ax.boxplot(data, labels=models, showmeans=True)
    ax.set_ylabel("relative L2 power error / e1")
    ax.set_title("Forward PINN test error distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = plot_dir / "forward_rel_l2_power_boxplot.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # Bar: mean, p95, max.
    x = np.arange(len(models), dtype=float)
    width = 0.25
    means = []
    p95s = []
    maxs = []
    for m in models:
        vals = df.loc[df["model"].astype(str) == m, "rel_l2_power"].astype(float)
        means.append(float(vals.mean()))
        p95s.append(float(vals.quantile(0.95)))
        maxs.append(float(vals.max()))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, means, width, label="mean")
    ax.bar(x, p95s, width, label="p95")
    ax.bar(x + width, maxs, width, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("relative L2 power error / e1")
    ax.set_title("Forward model summary errors")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "forward_rel_l2_power_mean_p95_max.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # Histogram overlay.
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in models:
        vals = df.loc[df["model"].astype(str) == m, "rel_l2_power"].astype(float).values
        ax.hist(vals, bins=30, alpha=0.45, label=m)
    ax.set_xlabel("relative L2 power error / e1")
    ax.set_ylabel("count")
    ax.set_title("Forward test-error histogram")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "forward_rel_l2_power_histogram.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    # e1/e2 scatter.
    fig, ax = plt.subplots(figsize=(7, 6))
    for m in models:
        sub = df[df["model"].astype(str) == m]
        ax.scatter(sub["e1"].astype(float), sub["e2"].astype(float), s=18, alpha=0.65, label=m)
    ax.set_xlabel("e1 = relative L2 power error")
    ax.set_ylabel("e2 = max absolute power error")
    ax.set_title("Forward e1/e2 scatter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = plot_dir / "forward_e1_e2_scatter.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(p))

    return paths


def plot_waterfall(
    tau: np.ndarray,
    zeta: np.ndarray,
    A_ssfm: np.ndarray,
    A_pinn: np.ndarray,
    out_path: Path,
    title: str,
    max_time_points: int = 900,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if len(tau) > max_time_points:
        step = int(math.ceil(len(tau) / max_time_points))
        tau_plot = tau[::step]
        ssfm_plot = A_ssfm[:, ::step]
        pinn_plot = A_pinn[:, ::step]
    else:
        tau_plot = tau
        ssfm_plot = A_ssfm
        pinn_plot = A_pinn

    P_ssfm = np.abs(ssfm_plot) ** 2
    P_pinn = np.abs(pinn_plot) ** 2

    fig = plt.figure(figsize=(8.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    for i, z in enumerate(zeta):
        y = np.full_like(tau_plot, float(z), dtype=float)
        ax.plot(tau_plot, y, P_ssfm[i], color="blue", linewidth=0.8)
        ax.plot(tau_plot, y, P_pinn[i], color="red", linestyle="--", linewidth=0.9)

    ax.set_xlabel(r"$t/T_0$")
    ax.set_ylabel(r"$z/L_D$")
    ax.set_zlabel("Normalized power")
    ax.set_title(title)
    ax.view_init(elev=26, azim=-58)
    ax.legend(
        handles=[
            Line2D([0], [0], color="blue", lw=1.0, label="SSFM"),
            Line2D([0], [0], color="red", lw=1.0, linestyle="--", label="PINN"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_slice(
    tau: np.ndarray,
    h_ssfm_final: np.ndarray,
    h_pinn_final: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save a 2D final-position slice: SSFM vs PINN at z = z_max.

    This is the direct time-domain cut at the final propagation distance,
    complementary to the 3D waterfall view. It helps identify whether the model
    misses peak height, broadening, crosstalk-generated peaks, or edge tails at
    the actual receiver plane.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P_ssfm = np.abs(h_ssfm_final) ** 2
    P_pinn = np.abs(h_pinn_final) ** 2
    abs_err = np.abs(P_pinn - P_ssfm)
    rel_l2 = float(np.linalg.norm(P_pinn - P_ssfm) / (np.linalg.norm(P_ssfm) + 1e-300))
    max_abs = float(np.max(abs_err)) if abs_err.size else 0.0

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    ax0.plot(tau, P_ssfm, color="blue", linewidth=1.2, label="SSFM")
    ax0.plot(tau, P_pinn, color="red", linestyle="--", linewidth=1.2, label="PINN")
    ax0.set_ylabel("Normalized power")
    ax0.set_title(f"{title}\nfinal slice at z = 4 L_D, e1={rel_l2:.3e}, e2={max_abs:.3e}")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1.plot(tau, abs_err, color="black", linewidth=1.0)
    ax1.set_xlabel(r"$t/T_0$")
    ax1.set_ylabel("|error|")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_best_worst_waterfalls(
    args: argparse.Namespace,
    metrics_path: Path,
    models: Dict[str, object],
    device: torch.device,
    out_dir: Path,
) -> dict:
    import pandas as pd

    df = pd.read_csv(metrics_path)
    out: dict = {"cases": []}
    if df.empty:
        return out

    wf_dir = out_dir / "waterfalls_best_worst"
    slice_dir = out_dir / "final_slices_best_worst"
    if args.make_waterfalls:
        wf_dir.mkdir(parents=True, exist_ok=True)
    if args.make_final_slices:
        slice_dir.mkdir(parents=True, exist_ok=True)

    for label, model in models.items():
        g = df[df["model"].astype(str) == str(label)].copy()
        if g.empty:
            continue
        chosen = [
            ("best", g.loc[g["rel_l2_power"].astype(float).idxmin()]),
            ("worst", g.loc[g["rel_l2_power"].astype(float).idxmax()]),
        ]
        for case_name, row in chosen:
            idx = int(row["idx"])
            combo = parse_levels(str(row["levels"]))
            params = NLSEParams.paper_pam4(
                z_max_ld=args.z_max_ld,
                t_window_t0=args.t_window_t0,
                n_t=args.n_t,
                n_z=args.n_z,
            ).with_multi_pulse(combo)
            # Save about waterfall_slices slices, including z=0 and z=z_max.
            save_every = max(1, int(round(args.n_z / max(1, int(args.waterfall_slices) - 1))))
            t0 = time.time()
            z_phys, t_ps, A = run_ssfm(params, device=str(device), save_every=save_every, quiet=args.quiet_ssfm)
            ssfm_sec = time.time() - t0
            tau = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
            zeta = np.asarray(z_phys, dtype=np.float64) / float(params.LD)
            mask = (tau >= float(args.compare_t_min)) & (tau <= float(args.compare_t_max))
            tau_cmp = tau[mask]
            A_ssfm = np.asarray(A[:, mask], dtype=np.complex128)
            A_pinn = predict_model_grid(model, tau_cmp, zeta, combo, device, chunk_size=args.chunk_size)
            metric_text = f"e1={float(row['rel_l2_power']):.3e}, e2={float(row['e2']):.3e}"
            combo_text = combo_to_text(combo).replace(";", ",")

            waterfall_path = None
            if args.make_waterfalls:
                waterfall_path = wf_dir / f"waterfall_{label}_{case_name}_idx{idx}_levels_{combo_text}.png"
                plot_waterfall(
                    tau_cmp,
                    zeta,
                    A_ssfm,
                    A_pinn,
                    waterfall_path,
                    title=f"{label} {case_name} case, idx={idx}, levels=({combo_text})\n{metric_text}",
                    max_time_points=args.waterfall_max_time_points,
                )
                print(f"waterfall saved -> {waterfall_path}")

            final_slice_path = None
            if args.make_final_slices:
                final_slice_path = slice_dir / f"final_slice_z4LD_{label}_{case_name}_idx{idx}_levels_{combo_text}.png"
                plot_final_slice(
                    tau_cmp,
                    A_ssfm[-1],
                    A_pinn[-1],
                    final_slice_path,
                    title=f"{label} {case_name} case, idx={idx}, levels=({combo_text})",
                )
                print(f"final slice saved -> {final_slice_path}")

            rec = {
                "model": str(label),
                "case": case_name,
                "idx": idx,
                "levels": list(combo),
                "rel_l2_power": float(row["rel_l2_power"]),
                "e2": float(row["e2"]),
                "waterfall_plot": None if waterfall_path is None else str(waterfall_path),
                "final_slice_plot": None if final_slice_path is None else str(final_slice_path),
                "ssfm_sec_for_plot": float(ssfm_sec),
                "save_every": int(save_every),
                "n_z_slices": int(len(zeta)),
            }
            out["cases"].append(rec)
            del A, A_ssfm, A_pinn
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    vis_summary = out_dir / "best_worst_visualization_cases.json"
    vis_summary.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.make_waterfalls:
        (wf_dir / "best_worst_waterfall_cases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.make_final_slices:
        (slice_dir / "best_worst_final_slice_cases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced streaming evaluation of generic M-pulse PINNs against SSFM.")
    p.add_argument("--n-pulses", "-M", type=int, required=True)
    p.add_argument("--unseen-csv", type=str, default="", help="CSV of unseen combinations. If omitted, generated using train_fraction and seed.")
    p.add_argument("--train-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--model-labels", nargs="+", required=True)
    p.add_argument("--model-paths", nargs="+", required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, required=True)

    p.add_argument("--t-window-t0", type=float, required=True, help="SSFM half time window in T0.")
    p.add_argument("--n-t", type=int, required=True)
    p.add_argument("--n-z", type=int, required=True)
    p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--compare-t-min", type=float, default=None)
    p.add_argument("--compare-t-max", type=float, default=None)
    p.add_argument("--auto-compare-window", action="store_true", default=True)
    p.add_argument("--manual-compare-window", action="store_false", dest="auto_compare_window")
    p.add_argument("--pinn-guard-t0", type=float, default=16.0)

    p.add_argument("--max-samples", type=int, default=0, help="0 means all unseen combinations.")
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--quiet-ssfm", action="store_true", default=True)
    p.add_argument("--verbose-ssfm", action="store_false", dest="quiet_ssfm")

    p.add_argument("--make-plots", action="store_true", default=True)
    p.add_argument("--no-plots", action="store_false", dest="make_plots")
    p.add_argument("--make-waterfalls", action="store_true", default=True)
    p.add_argument("--no-waterfalls", action="store_false", dest="make_waterfalls")
    p.add_argument("--waterfall-slices", type=int, default=24)
    p.add_argument("--waterfall-max-time-points", type=int, default=900)
    p.add_argument("--make-final-slices", action="store_true", default=True)
    p.add_argument("--no-final-slices", action="store_false", dest="make_final_slices")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.model_labels) != len(args.model_paths):
        raise ValueError("model-labels and model-paths must have the same length.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_stream.csv"
    summary_path = out_dir / "summary_by_model.json"

    if args.unseen_csv:
        unseen = load_combinations_csv(args.unseen_csv)
    else:
        dataset_dir = out_dir / "dataset"
        make_and_save_seen_unseen(args.n_pulses, dataset_dir, train_fraction=args.train_fraction, seed=args.seed)
        unseen = load_combinations_csv(dataset_dir / "unseen_combinations.csv")
    if args.max_samples and args.max_samples > 0:
        unseen = unseen[:int(args.max_samples)]

    if args.auto_compare_window:
        half_cmp = pinn_half_window_t0(args.n_pulses, guard_t0=args.pinn_guard_t0)
        args.compare_t_min = -half_cmp
        args.compare_t_max = half_cmp

    dev = _safe_device(args.device)
    models: Dict[str, object] = {}
    for label, path in zip(args.model_labels, args.model_paths):
        models[str(label)] = load_forward_checkpoint(path, dev)

    completed: set[tuple[int, str]] = set()
    write_header = True
    if args.resume and metrics_path.exists():
        import pandas as pd
        old = pd.read_csv(metrics_path)
        if "idx" in old.columns and "model" in old.columns:
            completed = set((int(r.idx), str(r.model)) for r in old.itertuples())
            write_header = False

    run_config = vars(args).copy()
    run_config.update({
        "n_eval_combos": len(unseen),
        "device_actual": str(dev),
        "model_labels": args.model_labels,
        "model_paths": args.model_paths,
        "note": "Evaluation set is the unseen_combinations.csv by default; max_samples=0 means all unseen combinations.",
    })
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = [
        "idx", "model", "levels", "rel_l2_field", "rel_l2_power", "e1", "e2", "mse_power", "mae_power", "ssfm_sec", "pinn_sec"
    ]
    mode = "a" if (args.resume and metrics_path.exists()) else "w"
    with metrics_path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        print("\n========== Enhanced streaming evaluation ==========")
        print(f"M={args.n_pulses}, n_eval={len(unseen)}, models={args.model_labels}")
        print(f"SSFM window=±{args.t_window_t0}T0, nt={args.n_t}, nz={args.n_z}")
        print(f"compare window=[{args.compare_t_min}, {args.compare_t_max}]T0")
        print(f"out_dir={out_dir}")

        for idx, combo in enumerate(unseen):
            if all((idx, label) in completed for label in args.model_labels):
                continue
            params = NLSEParams.paper_pam4(
                z_max_ld=args.z_max_ld,
                t_window_t0=args.t_window_t0,
                n_t=args.n_t,
                n_z=args.n_z,
            ).with_multi_pulse(combo)

            t0 = time.time()
            z_phys, t_ps, A = run_ssfm(params, device=str(dev), save_every=params.n_z, quiet=args.quiet_ssfm)
            ssfm_sec = time.time() - t0
            tau = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
            h_final = np.asarray(A[-1], dtype=np.complex128)
            mask = (tau >= float(args.compare_t_min)) & (tau <= float(args.compare_t_max))
            tau_cmp = tau[mask]
            h_ref = h_final[mask]

            for label, model in models.items():
                if (idx, label) in completed:
                    continue
                t1 = time.time()
                h_pred = predict_model(model, tau_cmp, args.z_max_ld, combo, dev, chunk_size=args.chunk_size)
                pinn_sec = time.time() - t1
                row = {"idx": idx, "model": label, "levels": combo_to_text(combo),
                       "ssfm_sec": ssfm_sec, "pinn_sec": pinn_sec}
                row.update(compute_metrics(h_pred, h_ref))
                writer.writerow(row)
                f.flush()

            del A, h_final, h_ref, tau_cmp
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if idx % 20 == 0 or idx == len(unseen) - 1:
                print(f"[{idx+1}/{len(unseen)}] combo={combo} ssfm={ssfm_sec:.2f}s")

    summary = summarize_metrics(metrics_path, summary_path)
    plot_paths = plot_metric_figures(metrics_path, out_dir) if args.make_plots else []
    waterfall_summary = make_best_worst_waterfalls(args, metrics_path, models, dev, out_dir) if (args.make_waterfalls or args.make_final_slices) else {"cases": []}

    print("\n========== Done ==========")
    print("metrics ->", metrics_path)
    print("summary ->", summary_path)
    if plot_paths:
        print("metric plots:")
        for p in plot_paths:
            print("  ", p)
    if waterfall_summary.get("cases"):
        if args.make_waterfalls:
            print("waterfall cases ->", out_dir / "waterfalls_best_worst" / "best_worst_waterfall_cases.json")
        if args.make_final_slices:
            print("final slice cases ->", out_dir / "final_slices_best_worst" / "best_worst_final_slice_cases.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
