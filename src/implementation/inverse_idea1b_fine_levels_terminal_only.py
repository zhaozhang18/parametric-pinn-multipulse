# -*- coding: utf-8 -*-
"""
inverse_idea1b_fine_levels_terminal_only.py  (v11 batched)

逆向大思路 1b：冻结正向模型，在更细归一化幅度档位上反推 P。

v11 在 v10 基础上的主要改动：
1. 增加 forward diagnostic：先把 true fine-level amplitude A 输入冻结正向模型 F*，
   检查 F* 对这些细档位 P 的正向预测是否准确。这样可以区分：
   - 反演失败是因为 F* 对细档位 P 本身预测不准；
   - 还是因为逆向优化 P 没收敛。
2. 修复 discrete logits 模式的 best-state 保存逻辑：保存“当前 loss 对应的 logits”，
   再 optimizer.step()，避免保存到 step 之后已经跨档位的 logits。
3. 增加 continuous_round 模式：连续优化 P，再 round 到最近 fine levels。
   这比 hard straight-through 离散 logits 更平滑，适合细档位反演。
4. 4 个 restart 可并行，默认可将 10 个 sample 的 40 条轨迹同时优化。
5. 3000 轮作为最大预算，默认按 90% 轨迹平台期自动停止。

典型用途：
- --run-mode diagnostic：只检查冻结正向模型对 fine-level amplitude A 的正向预测能力。
- --run-mode discrete：fine levels hard logits，修复 best-state 后的离散优化。
- --run-mode continuous_round：A 连续优化，最后 round 到最近 fine level。
- --run-mode both：同时跑 discrete 和 continuous_round。
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

from nlse import NLSEParams, pulse_centers_t0
from ssfm import run_ssfm
from train_multi_pulse_pinn import load_forward_checkpoint
from inverse_data_hybrid_mapping import interp_power_to_input_grid, interp_complex_field_to_input_grid, initial_power_from_levels
from train_inverse_multi_pulse_pinn_pure_physics import (
    safe_device, set_seed, ensure_dir, write_json, write_csv_dicts,
    select_best_forward, infer_m_from_sources, infer_eval_grid,
    terminal_field_forward_model, terminal_loss_from_uv_power,
    terminal_observable_mode, combo_to_text,
)
from inverse_batched_utils import run_batched_adamw, terminal_loss_vector


def parse_levels(text: str) -> list[float]:
    """Parse comma-separated levels. Shortcuts: 10 / ten -> 0.1,...,1.0."""
    s = str(text).strip().lower()
    if s in {"10", "ten", "0.1"}:
        return [round(0.1 * i, 10) for i in range(1, 11)]
    vals = [float(x) for x in str(text).replace(";", ",").replace(" ", ",").split(",") if x.strip()]
    if not vals:
        raise ValueError("levels is empty")
    vals = sorted(set(float(x) for x in vals))
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze forward model and invert fine-level powers with terminal-only loss.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("-M", "--n-pulses", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="")
    p.add_argument("--forward-model-path", default="")
    p.add_argument("--forward-metrics-csv", default="")
    p.add_argument("--terminal-observable", default="complex", choices=["power", "complex", "power_and_complex"])
    p.add_argument("--levels", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--inverse-dataset-mode", default="selected", choices=["selected"], help="Compatibility option. This script always builds selected-only fine-level SSFM data.")
    p.add_argument("--t-window-t0", type=float, default=0.0)
    p.add_argument("--n-t", type=int, default=0)
    p.add_argument("--n-z", type=int, default=0)
    p.add_argument("--z-max-ld", type=float, default=0.0)
    p.add_argument("--compare-t-min", default="")
    p.add_argument("--compare-t-max", default="")
    p.add_argument("--reuse-ssfm", action="store_true", default=True)
    p.add_argument("--rebuild-ssfm", action="store_true", default=False)
    p.add_argument("--verbose-ssfm", action="store_true")

    p.add_argument("--run-mode", default="discrete", choices=["diagnostic", "discrete", "continuous_round", "both"], help="diagnostic only, hard fine-level logits, continuous amplitude A then round, or both inverse modes.")
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--batch-mode", choices=["auto", "all_samples", "per_sample"], default="auto")
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--min-lr", type=float, default=5e-4)
    p.add_argument("--cosine-anneal", action="store_true", default=True)
    p.add_argument("--no-cosine-anneal", action="store_false", dest="cosine_anneal")
    p.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for discrete mode.")
    p.add_argument("--terminal-points", type=int, default=0)
    p.add_argument("--forward-time-chunk", type=int, default=512)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", action="store_false", dest="early_stop")
    p.add_argument("--early-stop-min-epochs", type=int, default=1000)
    p.add_argument("--early-stop-patience", type=int, default=300)
    p.add_argument("--early-stop-fraction", type=float, default=0.90)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Relative improvement threshold.")
    p.add_argument("--early-stop-abs-delta", type=float, default=1e-8)
    p.add_argument("--p-min", default="", help="Continuous mode lower bound. Empty -> min(levels).")
    p.add_argument("--p-max", default="", help="Continuous mode upper bound. Empty -> max(levels).")
    return p.parse_args()


def levels_hash(levels: Sequence[float]) -> str:
    import hashlib
    text = ",".join(f"{float(x):.10g}" for x in levels)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def make_dataset(args: argparse.Namespace, M: int, eval_grid: dict[str, Any], levels: list[float], out_root: Path, device: torch.device) -> Path:
    """Build selected-only fine-level SSFM data. Only n_samples are generated."""
    h = levels_hash(levels)
    ds_root = out_root / "fine_level_ssfm_datasets" / f"fine_M{M}_N{args.n_samples}_levels{len(levels)}_h{h}_seed{args.sample_seed}_Nin{args.inverse_input_points}"
    meta_path = ds_root / "meta.json"
    required = [
        meta_path, ds_root / "A_levels.npy", ds_root / "tau_input.npy",
        ds_root / "Y_terminal_power.npy", ds_root / "Y_terminal_real.npy", ds_root / "Y_terminal_imag.npy",
    ]
    if (not args.rebuild_ssfm) and all(q.exists() for q in required):
        return ds_root

    ds_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.sample_seed))
    L = np.asarray(levels, dtype=np.float32)
    P = L[rng.integers(0, len(L), size=(int(args.n_samples), M))].astype(np.float32)
    tau_input = np.linspace(float(eval_grid["compare_t_min"]), float(eval_grid["compare_t_max"]), int(args.inverse_input_points), dtype=np.float32)
    np.save(ds_root / "tau_input.npy", tau_input)
    np.save(ds_root / "A_levels.npy", P)
    C = np.argmin(np.abs(P[..., None] - L.reshape(1, 1, -1)), axis=-1).astype(np.int64)
    np.save(ds_root / "A_class_indices.npy", C)

    Y = open_memmap(ds_root / "Y_terminal_power.npy", mode="w+", dtype=np.float32, shape=(int(args.n_samples), int(args.inverse_input_points)))
    Yr = open_memmap(ds_root / "Y_terminal_real.npy", mode="w+", dtype=np.float32, shape=(int(args.n_samples), int(args.inverse_input_points)))
    Yi = open_memmap(ds_root / "Y_terminal_imag.npy", mode="w+", dtype=np.float32, shape=(int(args.n_samples), int(args.inverse_input_points)))
    X = open_memmap(ds_root / "X_initial_power.npy", mode="w+", dtype=np.float32, shape=(int(args.n_samples), int(args.inverse_input_points)))
    centers = pulse_centers_t0(M)
    t0_all = time.time()
    for i, combo in enumerate(P):
        params = NLSEParams.paper_pam4(
            z_max_ld=float(eval_grid["z_max_ld"]),
            t_window_t0=float(eval_grid["eval_half_window_t0"]),
            n_t=int(eval_grid["eval_n_t"]),
            n_z=int(eval_grid["eval_n_z"]),
        ).with_multi_pulse(tuple(float(x) for x in combo))
        z_phys, t_ps, A = run_ssfm(params, device=str(device), save_every=params.n_z, quiet=not bool(args.verbose_ssfm))
        tau_full = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
        h_final = np.asarray(A[-1], dtype=np.complex128)
        power = np.abs(h_final) ** 2
        Y[i, :] = interp_power_to_input_grid(tau_full, power, tau_input)
        real_i, imag_i = interp_complex_field_to_input_grid(tau_full, h_final, tau_input)
        Yr[i, :] = real_i
        Yi[i, :] = imag_i
        X[i, :] = initial_power_from_levels(tau_input, combo, centers)
        print(f"[SSFM fine {i+1}/{args.n_samples}] P={combo_to_text(combo)} elapsed={time.time()-t0_all:.1f}s", flush=True)
        del A, h_final, power
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    Y.flush(); Yr.flush(); Yi.flush(); X.flush()
    write_json(meta_path, {
        "dataset_type": "fine_amplitude_levels_selected_only",
        "level_quantity": "normalized_field_amplitude",
        "n_pulses": int(M),
        "levels": [float(x) for x in levels],
        "n_samples": int(args.n_samples),
        "sample_seed": int(args.sample_seed),
        "inverse_input_points": int(args.inverse_input_points),
        "eval_grid": eval_grid,
        "build_elapsed_sec": float(time.time() - t0_all),
        "note": "Only selected fine-level samples are generated by SSFM; no full grid is enumerated.",
    })
    return ds_root


def powers_from_logits_discrete(logits: torch.Tensor, levels: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight-through hard level selection."""
    probs = F.softmax(logits / max(float(temperature), 1e-8), dim=-1)
    hard_idx = torch.argmax(probs, dim=-1)
    hard = F.one_hot(hard_idx, num_classes=levels.numel()).to(dtype=probs.dtype, device=probs.device)
    weights = hard - probs.detach() + probs
    p = torch.sum(weights * levels.reshape(*([1] * (weights.ndim - 1)), -1), dim=-1)
    return p, probs


def powers_from_raw_continuous(raw: torch.Tensor, p_min: float, p_max: float) -> torch.Tensor:
    return float(p_min) + (float(p_max) - float(p_min)) * torch.sigmoid(raw)


def nearest_levels_np(p: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    L = np.asarray(levels, dtype=np.float32)
    idx = np.argmin(np.abs(p[..., None] - L.reshape(1, -1)), axis=-1)
    return L[idx]


def terminal_loss(model, tau, y_power, y_real, y_imag, powers, zeta, observable, n_points, chunk):
    if int(n_points) > 0 and int(n_points) < int(tau.numel()):
        idx = torch.randperm(int(tau.numel()), device=tau.device)[:int(n_points)]
        tau_use = tau[idx]
        yp = y_power[idx].reshape(1, -1)
        yr = y_real[idx].reshape(1, -1) if y_real is not None else None
        yi = y_imag[idx].reshape(1, -1) if y_imag is not None else None
    else:
        tau_use = tau
        yp = y_power.reshape(1, -1)
        yr = y_real.reshape(1, -1) if y_real is not None else None
        yi = y_imag.reshape(1, -1) if y_imag is not None else None
    u, v = terminal_field_forward_model(model, tau_use, powers, zeta=zeta, chunk_t=int(chunk))
    return terminal_loss_from_uv_power(u, v, yp, yr, yi, observable)


def run_forward_diagnostic(model, tau, P_true, Yp, Yr, Yi, zeta, observable, args, device, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for i in range(int(P_true.shape[0])):
        true = np.asarray(P_true[i], dtype=np.float32)
        p_t = torch.tensor(true.reshape(1, -1), dtype=torch.float32, device=device)
        y_power = torch.tensor(np.asarray(Yp[i], dtype=np.float32), dtype=torch.float32, device=device)
        y_real = torch.tensor(np.asarray(Yr[i], dtype=np.float32), dtype=torch.float32, device=device)
        y_imag = torch.tensor(np.asarray(Yi[i], dtype=np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            loss, pow_loss, cplx_loss = terminal_loss(model, tau, y_power, y_real, y_imag, p_t, zeta, observable, 0, args.forward_time_chunk)
        rows.append({
            "sample_rank": i,
            "true_amplitudes": combo_to_text(true),
            "forward_terminal_loss": float(loss.detach().cpu()),
            "forward_power_loss": float(pow_loss.detach().cpu()),
            "forward_complex_loss": float(cplx_loss.detach().cpu()) if cplx_loss is not None else "",
        })
    write_csv_dicts(out_dir / "forward_diagnostic.csv", rows)
    summary = {
        "n_samples": int(P_true.shape[0]),
        "forward_terminal_loss_mean": float(np.mean([r["forward_terminal_loss"] for r in rows])),
        "forward_terminal_loss_max": float(np.max([r["forward_terminal_loss"] for r in rows])),
        "forward_power_loss_mean": float(np.mean([r["forward_power_loss"] for r in rows])),
        "forward_complex_loss_mean": float(np.mean([r["forward_complex_loss"] for r in rows if r["forward_complex_loss"] != ""])) if any(r["forward_complex_loss"] != "" for r in rows) else None,
        "elapsed_sec": float(time.time() - t0),
        "purpose": "Check whether the frozen forward model F* predicts fine-level true P accurately before inverse optimization.",
    }
    write_json(out_dir / "forward_diagnostic_summary.json", summary)
    return summary



def _make_groups(n_samples: int, mode: str) -> list[list[int]]:
    return [list(range(n_samples))] if mode == "all_samples" else [[i] for i in range(n_samples)]


def _run_batched_mode(
    *,
    mode: str,
    model,
    tau: torch.Tensor,
    P_true: np.ndarray,
    Yp_t: torch.Tensor,
    Yr_t: torch.Tensor,
    Yi_t: torch.Tensor,
    levels: list[float],
    levels_t: torch.Tensor,
    zeta: float,
    observable: str,
    args: argparse.Namespace,
    device: torch.device,
    actual_batch_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    S, M = int(P_true.shape[0]), int(P_true.shape[1])
    R = int(args.restarts)
    per_sample_best: list[dict[str, Any] | None] = [None] * S
    restart_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    stop_rows: list[dict[str, Any]] = []
    p_min = float(min(levels) if str(args.p_min).strip() == "" else float(args.p_min))
    p_max = float(max(levels) if str(args.p_max).strip() == "" else float(args.p_max))

    for group_no, g in enumerate(_make_groups(S, actual_batch_mode)):
        sample_ids = np.repeat(np.asarray(g, dtype=np.int64), R)
        restart_ids = np.tile(np.arange(R, dtype=np.int64), len(g))
        B = len(sample_ids)
        yp = Yp_t[g].repeat_interleave(R, dim=0)
        yr = Yr_t[g].repeat_interleave(R, dim=0)
        yi = Yi_t[g].repeat_interleave(R, dim=0)

        if mode == "discrete":
            init = np.empty((B, M, len(levels)), dtype=np.float32)
            for b, (si, ri) in enumerate(zip(sample_ids.tolist(), restart_ids.tolist())):
                rng = np.random.default_rng(args.seed + 100000 + 1000 * si + ri)
                init[b] = rng.normal(0.0, 0.1, size=(M, len(levels))).astype(np.float32)
            initial = torch.tensor(init, dtype=torch.float32, device=device)
            decode = lambda x: powers_from_logits_discrete(x, levels_t, args.temperature)[0]
            history_extra = lambda p: [
                {"P": combo_to_text(row), "P_round": combo_to_text(row)} for row in p
            ]
        else:
            init = np.empty((B, M), dtype=np.float32)
            for b, (si, ri) in enumerate(zip(sample_ids.tolist(), restart_ids.tolist())):
                rng = np.random.default_rng(args.seed + 200000 + 1000 * si + ri)
                init[b] = rng.normal(0.0, 0.5, size=M).astype(np.float32)
            initial = torch.tensor(init, dtype=torch.float32, device=device)
            decode = lambda x: powers_from_raw_continuous(x, p_min, p_max)
            history_extra = lambda p: [
                {"P_cont": combo_to_text(row), "P_round": combo_to_text(nearest_levels_np(row, levels))}
                for row in p
            ]

        result = run_batched_adamw(
            model=model,
            tau=tau,
            y_power=yp,
            y_real=yr,
            y_imag=yi,
            initial_parameter=initial,
            decode_parameter=decode,
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
            mode=f"idea1b_{mode}_batched",
            history_extra=history_extra,
        )
        history_rows.extend(result.history)
        stop_rows.append({
            "group": group_no,
            "sample_indices": [int(x) for x in g],
            "batch_size": int(B),
            "stopped_epoch": int(result.stopped_epoch),
            "plateau_fraction": float(result.plateau_fraction),
        })

        with torch.no_grad():
            pbest = decode(result.best_parameter.to(device))
            loss_v, power_v, complex_v = terminal_loss_vector(
                model, tau, pbest, yp, yr, yi, zeta, observable, 0, args.forward_time_chunk
            )
        p_np = pbest.detach().cpu().numpy().astype(np.float32)
        lv = loss_v.detach().cpu().numpy()
        pv = power_v.detach().cpu().numpy()
        cv = complex_v.detach().cpu().numpy() if complex_v is not None else None
        best_ep = result.best_epoch.detach().cpu().numpy()

        for b, (si, ri) in enumerate(zip(sample_ids.tolist(), restart_ids.tolist())):
            true = np.asarray(P_true[si], dtype=np.float32)
            pred_cont = p_np[b]
            pred_round = pred_cont if mode == "discrete" else nearest_levels_np(pred_cont, levels).astype(np.float32)
            cand: dict[str, Any] = {
                "sample_rank": int(si),
                "mode": mode,
                "restart": int(ri),
                "best_epoch": int(best_ep[b]),
                "terminal_loss": float(lv[b]),
                "terminal_power_loss": float(pv[b]),
                "terminal_complex_loss": "" if cv is None else float(cv[b]),
                "true_amplitudes": combo_to_text(true),
                "pred_amplitudes": [float(x) for x in pred_cont.tolist()],
                "rounded_pred_amplitudes": [float(x) for x in pred_round.tolist()],
                "continuous_amplitude_mae": float(np.mean(np.abs(pred_cont - true))),
                "continuous_amplitude_rmse": float(np.sqrt(np.mean((pred_cont - true) ** 2))),
                "amplitude_mae": float(np.mean(np.abs(pred_round - true))),
                "amplitude_rmse": float(np.sqrt(np.mean((pred_round - true) ** 2))),
                "level_exact_match": int(bool(np.allclose(pred_round, true, atol=1e-6))),
                "per_pulse_accuracy": float(np.mean(np.isclose(pred_round, true, atol=1e-6))),
            }
            restart_rows.append(cand)
            old = per_sample_best[si]
            if old is None or cand["terminal_loss"] < old["terminal_loss"]:
                per_sample_best[si] = cand

    return [x for x in per_sample_best if x is not None], restart_rows, history_rows, stop_rows


def main() -> None:
    wall_clock_t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    device = safe_device(args.device)
    run_dir = Path(args.run_dir)
    levels = parse_levels(args.levels)
    out_dir = ensure_dir(args.out_dir or (run_dir / "inverse" / f"fine_amplitude_{args.run_mode}_L{len(levels)}_N{args.n_samples}_seed{args.sample_seed}"))

    ckpt, label, info = select_best_forward(args)
    M = infer_m_from_sources(args, run_dir, ckpt)
    args.n_pulses = M
    eval_grid = infer_eval_grid(args, run_dir, M, ckpt)
    h = levels_hash(levels)
    expected_ds_root = out_dir / "fine_level_ssfm_datasets" / f"fine_M{M}_N{args.n_samples}_levels{len(levels)}_h{h}_seed{args.sample_seed}_Nin{args.inverse_input_points}"
    meta_before = expected_ds_root / "meta.json"
    meta_mtime_before = meta_before.stat().st_mtime_ns if meta_before.exists() else None
    dataset_prepare_t0 = time.time()
    ds_root = make_dataset(args, M, eval_grid, levels, out_dir, device)
    dataset_prepare_sec_this_run = float(time.time() - dataset_prepare_t0)
    ds_meta = json.loads((ds_root / "meta.json").read_text(encoding="utf-8"))
    meta_mtime_after = (ds_root / "meta.json").stat().st_mtime_ns
    dataset_generated_this_run = bool(args.rebuild_ssfm or meta_mtime_before is None or meta_mtime_after != meta_mtime_before)
    target_ssfm_original = float(ds_meta.get("build_elapsed_sec", 0.0) or 0.0)
    target_ssfm_this_run = target_ssfm_original if dataset_generated_this_run else 0.0

    model = load_forward_checkpoint(ckpt, device).to(device)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    tau = torch.tensor(np.load(ds_root / "tau_input.npy"), dtype=torch.float32, device=device)
    P_true = np.load(ds_root / "A_levels.npy").astype(np.float32)
    Yp = np.load(ds_root / "Y_terminal_power.npy", mmap_mode="r")
    Yr = np.load(ds_root / "Y_terminal_real.npy", mmap_mode="r")
    Yi = np.load(ds_root / "Y_terminal_imag.npy", mmap_mode="r")
    Yp_t = torch.tensor(np.asarray(Yp, dtype=np.float32), dtype=torch.float32, device=device)
    Yr_t = torch.tensor(np.asarray(Yr, dtype=np.float32), dtype=torch.float32, device=device)
    Yi_t = torch.tensor(np.asarray(Yi, dtype=np.float32), dtype=torch.float32, device=device)
    levels_t = torch.tensor(levels, dtype=torch.float32, device=device)
    zeta = float(eval_grid["z_max_ld"])
    observable = terminal_observable_mode(args)

    diag = run_forward_diagnostic(model, tau, P_true, Yp, Yr, Yi, zeta, observable, args, device, out_dir)
    print("[forward diagnostic]", json.dumps(diag, indent=2, ensure_ascii=False), flush=True)
    if args.run_mode == "diagnostic":
        write_json(out_dir / "summary.json", {
            "method": "idea1b_fine_levels_terminal_only_batched",
            "run_mode": args.run_mode,
            "M": int(M),
            "levels": [float(x) for x in levels],
            "n_samples": int(args.n_samples),
            "forward_diagnostic": diag,
            "forward_model": str(ckpt),
            "args": vars(args),
        })
        return

    modes = ["discrete"] if args.run_mode == "discrete" else ["continuous_round"] if args.run_mode == "continuous_round" else ["discrete", "continuous_round"]
    all_rows: list[dict[str, Any]] = []
    t_all = time.time()
    requested_batch_mode = args.batch_mode
    for mode in modes:
        actual_batch_mode = "all_samples" if requested_batch_mode in {"auto", "all_samples"} else "per_sample"
        t_mode = time.time()
        try:
            rows, restart_rows, history, stop_rows = _run_batched_mode(
                mode=mode, model=model, tau=tau, P_true=P_true,
                Yp_t=Yp_t, Yr_t=Yr_t, Yi_t=Yi_t, levels=levels,
                levels_t=levels_t, zeta=zeta, observable=observable,
                args=args, device=device, actual_batch_mode=actual_batch_mode,
            )
        except RuntimeError as exc:
            if requested_batch_mode == "auto" and "out of memory" in str(exc).lower():
                print(f"[auto batch:{mode}] GPU OOM; falling back to per-sample batched restarts.", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                actual_batch_mode = "per_sample"
                rows, restart_rows, history, stop_rows = _run_batched_mode(
                    mode=mode, model=model, tau=tau, P_true=P_true,
                    Yp_t=Yp_t, Yr_t=Yr_t, Yi_t=Yi_t, levels=levels,
                    levels_t=levels_t, zeta=zeta, observable=observable,
                    args=args, device=device, actual_batch_mode=actual_batch_mode,
                )
            else:
                raise

        all_rows.extend(rows)
        write_csv_dicts(out_dir / f"per_sample_summary_{mode}.csv", rows)
        write_csv_dicts(out_dir / f"all_restart_summary_{mode}.csv", restart_rows)
        write_csv_dicts(out_dir / f"batched_history_{mode}.csv", history)
        for i in range(int(args.n_samples)):
            write_csv_dicts(out_dir / f"sample_{i:02d}_{mode}_history.csv", [h for h in history if int(h["sample"]) == i])
        write_json(out_dir / f"automatic_epoch_stop_{mode}.json", {
            "maximum_epoch_budget": int(args.epochs),
            "requested_batch_mode": requested_batch_mode,
            "actual_batch_mode": actual_batch_mode,
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
        inverse_core_elapsed = float(time.time() - t_mode)
        mode_summary = {
            "mode": mode,
            "method_semantics": {
                "target_waveform_generator": "SSFM",
                "target_true_amplitude_levels": [float(x) for x in levels],
                "inverse_search_space": ("discrete listed levels" if mode == "discrete" else "continuous interval followed by post-optimization nearest-level rounding"),
                "rounding_stage": ("not applicable" if mode == "discrete" else "post-optimization evaluation"),
            },
            "requested_batch_mode": requested_batch_mode,
            "actual_batch_mode": actual_batch_mode,
            "exact_match_accuracy": float(np.mean([r["level_exact_match"] for r in rows])),
            "per_pulse_accuracy_mean": float(np.mean([r["per_pulse_accuracy"] for r in rows])),
            "amplitude_mae_mean_after_rounding_or_discrete": float(np.mean([r["amplitude_mae"] for r in rows])),
            "amplitude_rmse_mean_after_rounding_or_discrete": float(np.mean([r["amplitude_rmse"] for r in rows])),
            "continuous_amplitude_mae_mean": float(np.mean([r.get("continuous_amplitude_mae", r["amplitude_mae"]) for r in rows])),
            "continuous_amplitude_rmse_mean": float(np.mean([r.get("continuous_amplitude_rmse", r["amplitude_rmse"]) for r in rows])),
            "terminal_loss_mean": float(np.mean([r["terminal_loss"] for r in rows])),
            "best_epoch_p50": float(np.percentile([r["best_epoch"] for r in restart_rows], 50)),
            "best_epoch_p90": float(np.percentile([r["best_epoch"] for r in restart_rows], 90)),
            "best_epoch_max": int(max(r["best_epoch"] for r in restart_rows)),
            "target_ssfm_generation_sec_original": target_ssfm_original,
            "target_ssfm_generation_sec_this_run": target_ssfm_this_run,
            "dataset_prepare_sec_this_run": dataset_prepare_sec_this_run,
            "inverse_core_elapsed_sec": inverse_core_elapsed,
            "ssfm_reconstruction_elapsed_sec": 0.0,
            "end_to_end_sec_from_scratch": float(target_ssfm_original + inverse_core_elapsed),
            "elapsed_sec": inverse_core_elapsed,
        }
        write_json(out_dir / f"summary_{mode}.json", mode_summary)
        print(json.dumps(mode_summary, indent=2, ensure_ascii=False), flush=True)

    write_csv_dicts(out_dir / "per_sample_summary_all_modes.csv", all_rows)
    summary = {
        "method": "idea1b_fine_levels_terminal_only_batched",
        "run_mode": args.run_mode,
        "M": int(M),
        "levels": [float(x) for x in levels],
        "n_samples": int(args.n_samples),
        "forward_diagnostic": diag,
        "target_ssfm_dataset": str(ds_root),
        "target_ssfm_dataset_reused_this_run": not dataset_generated_this_run,
        "target_ssfm_generation_sec_original": target_ssfm_original,
        "target_ssfm_generation_sec_this_run": target_ssfm_this_run,
        "dataset_prepare_sec_this_run": dataset_prepare_sec_this_run,
        "inverse_modes_elapsed_sec_total": float(time.time() - t_all),
        "wall_clock_sec_this_run": float(time.time() - wall_clock_t0),
        "elapsed_sec_total": float(time.time() - t_all),
        "forward_model": str(ckpt),
        "forward_label": label,
        "forward_selection": info,
        "args": vars(args),
    }
    for mode in modes:
        s_path = out_dir / f"summary_{mode}.json"
        if s_path.exists():
            summary[mode] = json.loads(s_path.read_text(encoding="utf-8"))
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
