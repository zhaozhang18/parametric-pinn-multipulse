# -*- coding: utf-8 -*-
"""Frozen-forward inversion for truly continuous random amplitudes in [0, 1].

For each fixed M this script:
1. draws ``n_samples`` random amplitude vectors A ~ Uniform([0,1]^M);
2. uses SSFM to generate the terminal complex waveform for every vector;
3. freezes the selected forward PINN and optimizes four (or more) restarts per
   sample, with all samples/restarts optionally evaluated in one GPU batch;
4. chooses the lowest frozen-forward terminal loss among the restarts;
5. sends that predicted amplitude vector through SSFM again and compares the
   reconstructed SSFM terminal waveform with the true SSFM observation.

The two main per-sample quantities requested for auditing are written to
``per_sample_summary.csv``:
- ``best_restart_forward_inverse_loss``: minimum frozen-forward objective among
  the restarts (relative MSE in the selected observable);
- ``ssfm_reconstruction_output_rel_l2``: terminal waveform error after the
  predicted amplitudes are propagated by SSFM.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inverse_batched_utils import run_batched_adamw, terminal_loss_vector
from inverse_data_hybrid_mapping import (
    interp_complex_field_to_input_grid,
    interp_power_to_input_grid,
)
from inverse_idea1a_continuous_terminal_only import init_raw_for_uniform_p, raw_to_p
from nlse import NLSEParams
from ssfm import run_ssfm
from train_multi_pulse_pinn import load_forward_checkpoint
from train_inverse_multi_pulse_pinn_pure_physics import (
    combo_to_text,
    ensure_dir,
    infer_eval_grid,
    infer_m_from_sources,
    safe_device,
    select_best_forward,
    set_seed,
    terminal_observable_mode,
    write_csv_dicts,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random continuous-amplitude SSFM observations -> frozen-forward amplitude inversion."
    )
    p.add_argument("--run-dir", required=True)
    p.add_argument("-M", "--n-pulses", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42, help="Optimizer/restart seed.")
    p.add_argument("--sample-seed", type=int, default=2027, help="Continuous true-amplitude seed.")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--out-dir", default="")
    p.add_argument("--forward-model-path", default="")
    p.add_argument("--forward-metrics-csv", default="")

    p.add_argument("--terminal-observable", default="complex", choices=["power", "complex", "power_and_complex"])
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--t-window-t0", type=float, default=0.0)
    p.add_argument("--n-t", type=int, default=0)
    p.add_argument("--n-z", type=int, default=0)
    p.add_argument("--z-max-ld", type=float, default=0.0)
    p.add_argument("--compare-t-min", default="")
    p.add_argument("--compare-t-max", default="")
    p.add_argument("--rebuild-random-dataset", action="store_true")
    p.add_argument("--verbose-ssfm", action="store_true")

    p.add_argument("--p-min", type=float, default=0.0)
    p.add_argument("--p-max", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=3000, help="Maximum epoch budget.")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--batch-mode", choices=["auto", "all_samples", "per_sample"], default="auto")
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--min-lr", type=float, default=5e-4)
    p.add_argument("--cosine-anneal", action="store_true", default=True)
    p.add_argument("--no-cosine-anneal", action="store_false", dest="cosine_anneal")
    p.add_argument("--terminal-points", type=int, default=0)
    p.add_argument("--forward-time-chunk", type=int, default=512)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", action="store_false", dest="early_stop")
    p.add_argument("--early-stop-min-epochs", type=int, default=1000)
    p.add_argument("--early-stop-patience", type=int, default=300)
    p.add_argument("--early-stop-fraction", type=float, default=0.90)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--early-stop-abs-delta", type=float, default=1e-8)
    return p.parse_args()


def _selected_rel_l2(power_rel_l2: float, complex_rel_l2: float, observable: str) -> float:
    if observable == "power":
        return float(power_rel_l2)
    if observable == "complex":
        return float(complex_rel_l2)
    return float(math.sqrt(0.5 * (power_rel_l2 ** 2 + complex_rel_l2 ** 2)))


def _build_or_load_random_ssfm_dataset(
    *,
    out_dir: Path,
    M: int,
    n_samples: int,
    sample_seed: int,
    inverse_input_points: int,
    grid: dict[str, Any],
    device: torch.device,
    rebuild: bool,
    verbose_ssfm: bool,
) -> tuple[Path, dict[str, Any]]:
    ds = ensure_dir(out_dir / f"random_continuous_ssfm_N{n_samples}_seed{sample_seed}")
    paths = {
        "A": ds / "true_amplitudes.npy",
        "tau": ds / "tau_input.npy",
        "power": ds / "Y_terminal_power.npy",
        "real": ds / "Y_terminal_real.npy",
        "imag": ds / "Y_terminal_imag.npy",
        "meta": ds / "meta.json",
    }
    expected = {
        "version": 1,
        "purpose": "random continuous amplitudes in [0,1], terminal SSFM field only",
        "M": int(M),
        "n_samples": int(n_samples),
        "sample_seed": int(sample_seed),
        "inverse_input_points": int(inverse_input_points),
        "t_window_t0": float(grid["eval_half_window_t0"]),
        "n_t": int(grid["eval_n_t"]),
        "n_z": int(grid["eval_n_z"]),
        "z_max_ld": float(grid.get("z_max_ld", 4.0)),
        "compare_t_min": float(grid["compare_t_min"]),
        "compare_t_max": float(grid["compare_t_max"]),
    }
    if not rebuild and all(p.exists() for p in paths.values()):
        try:
            access_t0 = time.time()
            old = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if all(old.get(k) == v for k, v in expected.items()):
                old = dict(old)
                old["_dataset_access"] = {
                    "reused_this_call": True,
                    "generated_this_call": False,
                    "prepare_elapsed_sec_this_call": float(time.time() - access_t0),
                    "ssfm_generation_elapsed_sec_this_call": 0.0,
                    "ssfm_generation_elapsed_sec_original": float(old.get("build_elapsed_sec", 0.0) or 0.0),
                }
                return ds, old
        except Exception:
            pass

    rng = np.random.default_rng(int(sample_seed))
    amplitudes = rng.uniform(0.0, 1.0, size=(int(n_samples), int(M))).astype(np.float32)
    # Avoid exact saturation of the sigmoid parameterization while preserving an
    # effectively continuous [0,1] experiment.
    amplitudes = np.clip(amplitudes, 1e-6, 1.0 - 1e-6)
    tau_input = np.linspace(
        float(grid["compare_t_min"]),
        float(grid["compare_t_max"]),
        int(inverse_input_points),
        dtype=np.float32,
    )
    yp = np.empty((int(n_samples), int(inverse_input_points)), dtype=np.float32)
    yr = np.empty_like(yp)
    yi = np.empty_like(yp)

    t0 = time.time()
    for i, amp in enumerate(amplitudes):
        params = NLSEParams.paper_pam4(
            z_max_ld=float(grid.get("z_max_ld", 4.0)),
            t_window_t0=float(grid["eval_half_window_t0"]),
            n_t=int(grid["eval_n_t"]),
            n_z=int(grid["eval_n_z"]),
        ).with_multi_pulse(tuple(float(x) for x in amp), level_mode="field")
        _, t_ps, field = run_ssfm(
            params,
            device=str(device),
            save_every=int(params.n_z),
            quiet=not bool(verbose_ssfm),
        )
        tau_full = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
        h_final = np.asarray(field[-1], dtype=np.complex128)
        yp[i] = interp_power_to_input_grid(tau_full, np.abs(h_final) ** 2, tau_input)
        yr[i], yi[i] = interp_complex_field_to_input_grid(tau_full, h_final, tau_input)
        print(
            f"[random continuous SSFM {i+1}/{n_samples}] "
            f"A={combo_to_text(amp)} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    np.save(paths["A"], amplitudes)
    np.save(paths["tau"], tau_input)
    np.save(paths["power"], yp)
    np.save(paths["real"], yr)
    np.save(paths["imag"], yi)
    meta = {
        **expected,
        "build_elapsed_sec": float(time.time() - t0),
        "amplitude_distribution": "independent uniform continuous values in [0,1]",
        "true_amplitudes_path": str(paths["A"]),
        "tau_input_path": str(paths["tau"]),
        "Y_terminal_power_path": str(paths["power"]),
        "Y_terminal_real_path": str(paths["real"]),
        "Y_terminal_imag_path": str(paths["imag"]),
    }
    write_json(paths["meta"], meta)
    meta["_dataset_access"] = {
        "reused_this_call": False,
        "generated_this_call": True,
        "prepare_elapsed_sec_this_call": float(meta.get("build_elapsed_sec", 0.0)),
        "ssfm_generation_elapsed_sec_this_call": float(meta.get("build_elapsed_sec", 0.0)),
        "ssfm_generation_elapsed_sec_original": float(meta.get("build_elapsed_sec", 0.0)),
    }
    write_csv_dicts(
        ds / "true_amplitudes.csv",
        [
            {"sample_rank": i, **{f"A{j+1}": float(amplitudes[i, j]) for j in range(M)}}
            for i in range(int(n_samples))
        ],
    )
    return ds, meta


def _run_optimization(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tau: torch.Tensor,
    true_a: np.ndarray,
    yp: torch.Tensor,
    yr: torch.Tensor,
    yi: torch.Tensor,
    zeta: float,
    observable: str,
    device: torch.device,
    actual_batch_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    S, M = true_a.shape
    R = int(args.restarts)
    groups = [list(range(S))] if actual_batch_mode == "all_samples" else [[i] for i in range(S)]
    per_sample_best: list[dict[str, Any] | None] = [None] * S
    restart_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []

    for group_no, group in enumerate(groups):
        sample_ids = np.repeat(np.asarray(group, dtype=np.int64), R)
        restart_ids = np.tile(np.arange(R, dtype=np.int64), len(group))
        B = len(sample_ids)
        raw_np = np.empty((B, M), dtype=np.float32)
        for b, (si, ri) in enumerate(zip(sample_ids.tolist(), restart_ids.tolist())):
            raw_np[b] = init_raw_for_uniform_p(
                M, args.p_min, args.p_max,
                args.seed + 1000 * int(si) + int(ri),
            )
        initial = torch.tensor(raw_np, dtype=torch.float32, device=device)
        yp_b = yp[group].repeat_interleave(R, dim=0)
        yr_b = yr[group].repeat_interleave(R, dim=0)
        yi_b = yi[group].repeat_interleave(R, dim=0)
        result = run_batched_adamw(
            model=model,
            tau=tau,
            y_power=yp_b,
            y_real=yr_b,
            y_imag=yi_b,
            initial_parameter=initial,
            decode_parameter=lambda x: raw_to_p(x, args.p_min, args.p_max),
            zeta=zeta,
            observable=observable,
            epochs=args.epochs,
            lr=args.lr,
            min_lr=args.min_lr,
            cosine_anneal=args.cosine_anneal,
            terminal_points=args.terminal_points,
            forward_time_chunk=args.forward_time_chunk,
            log_every=args.log_every,
            early_stop=args.early_stop,
            early_stop_min_epochs=args.early_stop_min_epochs,
            early_stop_patience=args.early_stop_patience,
            early_stop_fraction=args.early_stop_fraction,
            early_stop_rel_delta=args.early_stop_min_delta,
            early_stop_abs_delta=args.early_stop_abs_delta,
            sample_ids=sample_ids,
            restart_ids=restart_ids,
            mode="random_continuous_ssfm_batched",
            history_extra=lambda a: [{"A": combo_to_text(row)} for row in a],
        )
        history_rows.extend(result.history)
        stop_rows.append({
            "group": int(group_no),
            "sample_indices": [int(x) for x in group],
            "batch_size": int(B),
            "stopped_epoch": int(result.stopped_epoch),
            "plateau_fraction": float(result.plateau_fraction),
        })

        with torch.no_grad():
            pred = raw_to_p(result.best_parameter.to(device), args.p_min, args.p_max)
            loss_v, power_v, complex_v = terminal_loss_vector(
                model, tau, pred, yp_b, yr_b, yi_b,
                zeta, observable, 0, args.forward_time_chunk,
            )
        pred_np = pred.detach().cpu().numpy()
        loss_np = loss_v.detach().cpu().numpy()
        power_np = power_v.detach().cpu().numpy()
        complex_np = complex_v.detach().cpu().numpy() if complex_v is not None else None
        best_ep_np = result.best_epoch.detach().cpu().numpy()

        for b, (si, ri) in enumerate(zip(sample_ids.tolist(), restart_ids.tolist())):
            err = pred_np[b] - true_a[si]
            row: dict[str, Any] = {
                "sample_rank": int(si),
                "restart": int(ri),
                "best_epoch": int(best_ep_np[b]),
                "best_restart_forward_inverse_loss": float(loss_np[b]),
                "best_restart_forward_inverse_rel_l2": float(math.sqrt(max(0.0, float(loss_np[b])))),
                "forward_inverse_power_rel_l2": float(math.sqrt(max(0.0, float(power_np[b])))),
                "forward_inverse_complex_rel_l2": "" if complex_np is None else float(math.sqrt(max(0.0, float(complex_np[b])))),
                "true_amplitudes": combo_to_text(true_a[si]),
                "pred_amplitudes": combo_to_text(pred_np[b]),
                "amplitude_mae": float(np.mean(np.abs(err))),
                "amplitude_rmse": float(np.sqrt(np.mean(err ** 2))),
                "amplitude_max_abs_error": float(np.max(np.abs(err))),
            }
            restart_rows.append(row)
            old = per_sample_best[si]
            if old is None or row["best_restart_forward_inverse_loss"] < old["best_restart_forward_inverse_loss"]:
                per_sample_best[si] = row

    return [r for r in per_sample_best if r is not None], restart_rows, history_rows, stop_rows


def _ssfm_reconstruct_predictions(
    *,
    best_rows: list[dict[str, Any]],
    target_power: np.ndarray,
    target_real: np.ndarray,
    target_imag: np.ndarray,
    tau_input: np.ndarray,
    M: int,
    grid: dict[str, Any],
    device: torch.device,
    observable: str,
    verbose_ssfm: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    S = len(best_rows)
    pred_power = np.empty_like(target_power, dtype=np.float32)
    pred_real = np.empty_like(target_real, dtype=np.float32)
    pred_imag = np.empty_like(target_imag, dtype=np.float32)
    t0 = time.time()
    for i, row in enumerate(best_rows):
        pred_a = np.asarray([float(x) for x in str(row["pred_amplitudes"]).split(";")], dtype=np.float32)
        if pred_a.shape != (M,):
            raise ValueError(f"Bad predicted amplitude shape for sample {i}: {pred_a.shape}")
        params = NLSEParams.paper_pam4(
            z_max_ld=float(grid.get("z_max_ld", 4.0)),
            t_window_t0=float(grid["eval_half_window_t0"]),
            n_t=int(grid["eval_n_t"]),
            n_z=int(grid["eval_n_z"]),
        ).with_multi_pulse(tuple(float(x) for x in pred_a), level_mode="field")
        _, t_ps, field = run_ssfm(
            params, device=str(device), save_every=int(params.n_z), quiet=not bool(verbose_ssfm)
        )
        tau_full = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
        h_final = np.asarray(field[-1], dtype=np.complex128)
        pred_power[i] = interp_power_to_input_grid(tau_full, np.abs(h_final) ** 2, tau_input)
        pred_real[i], pred_imag[i] = interp_complex_field_to_input_grid(tau_full, h_final, tau_input)

        p_rel = float(np.linalg.norm(pred_power[i] - target_power[i]) / (np.linalg.norm(target_power[i]) + 1e-12))
        c_num = np.sum((pred_real[i] - target_real[i]) ** 2 + (pred_imag[i] - target_imag[i]) ** 2)
        c_den = np.sum(target_real[i] ** 2 + target_imag[i] ** 2) + 1e-12
        c_rel = float(np.sqrt(c_num / c_den))
        row["ssfm_reconstruction_power_rel_l2"] = p_rel
        row["ssfm_reconstruction_complex_rel_l2"] = c_rel
        row["ssfm_reconstruction_output_rel_l2"] = _selected_rel_l2(p_rel, c_rel, observable)
        print(
            f"[SSFM reconstruction {i+1}/{S}] selected_error="
            f"{row['ssfm_reconstruction_output_rel_l2']:.4e} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
    return pred_power, pred_real, pred_imag, float(time.time() - t0)


def _add_tolerance_metrics(best_rows: list[dict[str, Any]]) -> None:
    for row in best_rows:
        true_a = np.asarray([float(x) for x in str(row["true_amplitudes"]).split(";")], dtype=np.float32)
        pred_a = np.asarray([float(x) for x in str(row["pred_amplitudes"]).split(";")], dtype=np.float32)
        ae = np.abs(pred_a - true_a)
        for tol, key in [(0.01, "0p01"), (0.025, "0p025"), (0.05, "0p05")]:
            row[f"all_pulses_within_{key}"] = int(bool(np.all(ae <= tol)))
            row[f"per_pulse_within_{key}"] = float(np.mean(ae <= tol))


def _make_plots(
    out_dir: Path,
    best_rows: list[dict[str, Any]],
    target_power: np.ndarray,
    reconstructed_power: np.ndarray,
    tau: np.ndarray,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot warning] matplotlib unavailable: {exc}", flush=True)
        return
    plot_dir = ensure_dir(out_dir / "plots")
    true_a = np.asarray([[float(x) for x in str(r["true_amplitudes"]).split(";")] for r in best_rows])
    pred_a = np.asarray([[float(x) for x in str(r["pred_amplitudes"]).split(";")] for r in best_rows])

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(true_a.reshape(-1), pred_a.reshape(-1), alpha=0.75)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    ax.set_xlabel("True normalized amplitude")
    ax.set_ylabel("Predicted normalized amplitude")
    ax.set_title("Random continuous amplitudes: true vs predicted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "true_vs_predicted_amplitudes.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(best_rows))
    inv_rel = np.asarray([float(r["best_restart_forward_inverse_rel_l2"]) for r in best_rows])
    ssfm_rel = np.asarray([float(r["ssfm_reconstruction_output_rel_l2"]) for r in best_rows])
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(x - width / 2, inv_rel, width=width, label="Frozen-forward inverse rel-L2")
    ax.bar(x + width / 2, ssfm_rel, width=width, label="SSFM reconstruction rel-L2")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Random continuous inverse errors by sample")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "inverse_and_ssfm_reconstruction_errors.png", dpi=180)
    plt.close(fig)

    n = len(best_rows)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12.0, max(3.0, 2.8 * nrows)), squeeze=False)
    for i in range(nrows * ncols):
        ax = axes[i // ncols][i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(tau, target_power[i], label="True SSFM")
        ax.plot(tau, reconstructed_power[i], linestyle="--", label="Predicted-A SSFM")
        ax.set_title(f"Sample {i}: output rel-L2={best_rows[i]['ssfm_reconstruction_output_rel_l2']:.3g}")
        ax.set_xlabel("t / T0")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "terminal_power_true_vs_reconstructed_all_samples.png", dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = safe_device(args.device)
    run_dir = Path(args.run_dir).resolve()
    out_dir = ensure_dir(args.out_dir or (run_dir / "inverse" / "05_random_continuous_0to1_ssfm"))

    ckpt, forward_label, forward_info = select_best_forward(args)
    M = infer_m_from_sources(args, run_dir, ckpt)
    args.n_pulses = int(M)
    grid = infer_eval_grid(args, run_dir, M, ckpt)
    # Explicit CLI values, when supplied by the orchestrator, are already folded
    # into infer_eval_grid. Keep one authoritative dictionary for SSFM and PINN.
    model = load_forward_checkpoint(ckpt, device).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    observable = terminal_observable_mode(args)

    total_t0 = time.time()
    ds_root, ds_meta = _build_or_load_random_ssfm_dataset(
        out_dir=out_dir,
        M=M,
        n_samples=args.n_samples,
        sample_seed=args.sample_seed,
        inverse_input_points=args.inverse_input_points,
        grid=grid,
        device=device,
        rebuild=args.rebuild_random_dataset,
        verbose_ssfm=args.verbose_ssfm,
    )
    true_a = np.load(ds_root / "true_amplitudes.npy").astype(np.float32)
    tau_np = np.load(ds_root / "tau_input.npy").astype(np.float32)
    yp_np = np.load(ds_root / "Y_terminal_power.npy").astype(np.float32)
    yr_np = np.load(ds_root / "Y_terminal_real.npy").astype(np.float32)
    yi_np = np.load(ds_root / "Y_terminal_imag.npy").astype(np.float32)
    tau = torch.tensor(tau_np, dtype=torch.float32, device=device)
    yp = torch.tensor(yp_np, dtype=torch.float32, device=device)
    yr = torch.tensor(yr_np, dtype=torch.float32, device=device)
    yi = torch.tensor(yi_np, dtype=torch.float32, device=device)
    zeta = float(getattr(model, "z_max_ld", grid.get("z_max_ld", 4.0)))

    requested_mode = args.batch_mode
    actual_mode = "all_samples" if requested_mode in {"auto", "all_samples"} else "per_sample"
    optimization_t0 = time.time()
    try:
        best_rows, restart_rows, history_rows, stop_rows = _run_optimization(
            args=args, model=model, tau=tau, true_a=true_a,
            yp=yp, yr=yr, yi=yi, zeta=zeta, observable=observable,
            device=device, actual_batch_mode=actual_mode,
        )
    except RuntimeError as exc:
        if requested_mode == "auto" and "out of memory" in str(exc).lower():
            print("[auto batch] GPU OOM; falling back to per-sample batched restarts.", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            actual_mode = "per_sample"
            best_rows, restart_rows, history_rows, stop_rows = _run_optimization(
                args=args, model=model, tau=tau, true_a=true_a,
                yp=yp, yr=yr, yi=yi, zeta=zeta, observable=observable,
                device=device, actual_batch_mode=actual_mode,
            )
        else:
            raise
    optimization_elapsed = float(time.time() - optimization_t0)

    _add_tolerance_metrics(best_rows)
    for row in best_rows:
        row["selected_restart"] = int(row["restart"])
    recon_power, recon_real, recon_imag, reconstruction_elapsed = _ssfm_reconstruct_predictions(
        best_rows=best_rows,
        target_power=yp_np,
        target_real=yr_np,
        target_imag=yi_np,
        tau_input=tau_np,
        M=M,
        grid=grid,
        device=device,
        observable=observable,
        verbose_ssfm=args.verbose_ssfm,
    )

    np.save(out_dir / "predicted_amplitudes.npy", np.asarray([
        [float(x) for x in str(r["pred_amplitudes"]).split(";")] for r in best_rows
    ], dtype=np.float32))
    np.save(out_dir / "reconstructed_ssfm_terminal_power.npy", recon_power)
    np.save(out_dir / "reconstructed_ssfm_terminal_real.npy", recon_real)
    np.save(out_dir / "reconstructed_ssfm_terminal_imag.npy", recon_imag)
    write_csv_dicts(out_dir / "per_sample_summary.csv", best_rows)
    write_csv_dicts(
        out_dir / "per_sample_two_key_errors.csv",
        [
            {
                "sample_rank": r["sample_rank"],
                "true_amplitudes": r["true_amplitudes"],
                "pred_amplitudes": r["pred_amplitudes"],
                "selected_restart": r["selected_restart"],
                "best_epoch": r["best_epoch"],
                "best_restart_forward_inverse_loss": r["best_restart_forward_inverse_loss"],
                "best_restart_forward_inverse_rel_l2": r["best_restart_forward_inverse_rel_l2"],
                "ssfm_reconstruction_output_rel_l2": r["ssfm_reconstruction_output_rel_l2"],
            }
            for r in best_rows
        ],
    )
    write_csv_dicts(out_dir / "all_restart_summary.csv", restart_rows)
    write_csv_dicts(out_dir / "batched_history.csv", history_rows)
    write_json(out_dir / "automatic_epoch_stop.json", {
        "maximum_epoch_budget": int(args.epochs),
        "requested_batch_mode": requested_mode,
        "actual_batch_mode": actual_mode,
        "stop_groups": stop_rows,
        "recommended_observed_stop_epoch": int(max(r["stopped_epoch"] for r in stop_rows)),
        "criterion": {
            "plateau_fraction": float(args.early_stop_fraction),
            "patience": int(args.early_stop_patience),
            "min_epochs": int(args.early_stop_min_epochs),
            "relative_delta": float(args.early_stop_min_delta),
            "absolute_delta": float(args.early_stop_abs_delta),
        },
    })

    terminal_complex_values = [float(r["forward_inverse_complex_rel_l2"]) for r in best_rows if r["forward_inverse_complex_rel_l2"] != ""]
    total_elapsed = float(time.time() - total_t0)
    access = dict(ds_meta.get("_dataset_access", {}) or {})
    target_ssfm_original = float(access.get("ssfm_generation_elapsed_sec_original", ds_meta.get("build_elapsed_sec", 0.0)) or 0.0)
    target_ssfm_this_run = float(access.get("ssfm_generation_elapsed_sec_this_call", 0.0) or 0.0)
    dataset_prepare_this_run = float(access.get("prepare_elapsed_sec_this_call", 0.0) or 0.0)
    summary = {
        "method": "random_continuous_0to1_ssfm_frozen_forward_batched",
        "method_semantics": {
            "target_waveform_generator": "SSFM",
            "target_true_amplitude_distribution": "independent Uniform(0,1)",
            "inverse_search_space": "continuous [0,1]^M",
            "inverse_knows_discrete_levels": False,
        },
        "M": int(M),
        "n_samples": int(len(best_rows)),
        "restarts_per_sample": int(args.restarts),
        "true_amplitude_distribution": "independent continuous Uniform(0,1)",
        "terminal_observable": observable,
        "requested_batch_mode": requested_mode,
        "actual_batch_mode": actual_mode,
        "amplitude_mae_mean": float(np.mean([r["amplitude_mae"] for r in best_rows])),
        "amplitude_rmse_mean": float(np.mean([r["amplitude_rmse"] for r in best_rows])),
        "amplitude_max_abs_error_mean": float(np.mean([r["amplitude_max_abs_error"] for r in best_rows])),
        "sample_all_pulses_within_0p01_accuracy": float(np.mean([r["all_pulses_within_0p01"] for r in best_rows])),
        "sample_all_pulses_within_0p025_accuracy": float(np.mean([r["all_pulses_within_0p025"] for r in best_rows])),
        "sample_all_pulses_within_0p05_accuracy": float(np.mean([r["all_pulses_within_0p05"] for r in best_rows])),
        "per_pulse_within_0p01_accuracy": float(np.mean([r["per_pulse_within_0p01"] for r in best_rows])),
        "per_pulse_within_0p025_accuracy": float(np.mean([r["per_pulse_within_0p025"] for r in best_rows])),
        "per_pulse_within_0p05_accuracy": float(np.mean([r["per_pulse_within_0p05"] for r in best_rows])),
        "best_restart_forward_inverse_loss_mean": float(np.mean([r["best_restart_forward_inverse_loss"] for r in best_rows])),
        "best_restart_forward_inverse_rel_l2_mean": float(np.mean([r["best_restart_forward_inverse_rel_l2"] for r in best_rows])),
        "forward_inverse_power_rel_l2_mean": float(np.mean([r["forward_inverse_power_rel_l2"] for r in best_rows])),
        "forward_inverse_complex_rel_l2_mean": float(np.mean(terminal_complex_values)) if terminal_complex_values else None,
        "ssfm_reconstruction_output_rel_l2_mean": float(np.mean([r["ssfm_reconstruction_output_rel_l2"] for r in best_rows])),
        "ssfm_reconstruction_output_rel_l2_p95": float(np.percentile([r["ssfm_reconstruction_output_rel_l2"] for r in best_rows], 95)),
        "ssfm_reconstruction_output_rel_l2_max": float(np.max([r["ssfm_reconstruction_output_rel_l2"] for r in best_rows])),
        "ssfm_reconstruction_power_rel_l2_mean": float(np.mean([r["ssfm_reconstruction_power_rel_l2"] for r in best_rows])),
        "ssfm_reconstruction_complex_rel_l2_mean": float(np.mean([r["ssfm_reconstruction_complex_rel_l2"] for r in best_rows])),
        "best_epoch_p50_all_restarts": float(np.percentile([r["best_epoch"] for r in restart_rows], 50)),
        "best_epoch_p90_all_restarts": float(np.percentile([r["best_epoch"] for r in restart_rows], 90)),
        "best_epoch_max_all_restarts": int(max(r["best_epoch"] for r in restart_rows)),
        "ssfm_true_dataset_build_elapsed_sec": target_ssfm_original,
        "target_ssfm_dataset_reused_this_run": bool(access.get("reused_this_call", False)),
        "target_ssfm_generation_sec_original": target_ssfm_original,
        "target_ssfm_generation_sec_this_run": target_ssfm_this_run,
        "dataset_prepare_sec_this_run": dataset_prepare_this_run,
        "optimization_elapsed_sec": optimization_elapsed,
        "inverse_core_elapsed_sec": optimization_elapsed,
        "inverse_runtime_sec_for_method_comparison": optimization_elapsed,
        "inverse_runtime_sec_per_sample": float(optimization_elapsed / max(1, len(best_rows))),
        "ssfm_reconstruction_elapsed_sec": reconstruction_elapsed,
        "end_to_end_sec_from_scratch": float(target_ssfm_original + optimization_elapsed + reconstruction_elapsed),
        "wall_clock_sec_this_run": total_elapsed,
        "elapsed_sec_total": total_elapsed,
        "random_ssfm_dataset": str(ds_root),
        "forward_model": str(ckpt),
        "forward_label": forward_label,
        "forward_selection": forward_info,
        "args": vars(args),
    }
    write_json(out_dir / "summary.json", summary)
    _make_plots(out_dir, best_rows, yp_np, recon_power, tau_np)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
