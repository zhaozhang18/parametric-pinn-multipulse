# -*- coding: utf-8 -*-
"""Freeze the universal beta2-conditioned sparse-8 PINN and jointly recover:
    1) unknown pulse count K in {1,...,8};
    2) continuous active amplitude vector A;
    3) continuous dispersion ratio D = beta2 / beta2_ref.

For every target sample, all 8 candidate K values and all restarts are represented
by one optimization tensor. Target samples can also be optimized jointly. Frozen-
forward evaluations are trajectory-chunked for 6 GB GPU safety, but gradients for
all trajectories are accumulated before one AdamW step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from universal_beta2_common import (
    active_slots_for_k,
    ensure_dir,
    load_beta2_checkpoint,
    make_grid,
    mask_for_k,
    parse_float_list,
    run_ssfm_batch_selected_variable_d,
    safe_device,
    save_csv_rows,
    write_json,
)

SCRIPT_VERSION = "universal_unknownK_A_D_inverse_v2_power_backcheck_20260723"


def set_seed(seed: int) -> None:
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def vector_text(x: Sequence[float]) -> str:
    return ";".join(f"{float(v):.8g}" for v in x)


def sigmoid_logit_unit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(x / (1.0 - x)).astype(np.float32)


def map_raw_amplitudes(raw: torch.Tensor, masks: torch.Tensor, min_amp: float) -> torch.Tensor:
    active = float(min_amp) + (1.0 - float(min_amp)) * torch.sigmoid(raw)
    return active * masks


def map_raw_d(raw_d: torch.Tensor, d_lower: float, d_upper: float) -> torch.Tensor:
    return float(d_lower) + (float(d_upper) - float(d_lower)) * torch.sigmoid(raw_d)


def raw_from_d(d: np.ndarray, d_lower: float, d_upper: float) -> np.ndarray:
    unit = (np.asarray(d, dtype=np.float32) - float(d_lower)) / (float(d_upper) - float(d_lower))
    return sigmoid_logit_unit(unit)


def terminal_forward(model, tau: torch.Tensor, amps: torch.Tensor, d: torch.Tensor, zeta: float, time_chunk: int):
    B = int(amps.shape[0]); us: list[torch.Tensor] = []; vs: list[torch.Tensor] = []
    for s in range(0, int(tau.numel()), int(time_chunk)):
        e = min(int(tau.numel()), s + int(time_chunk)); tc = tau[s:e]; nt = int(tc.numel())
        z = torch.full((B * nt, 1), float(zeta), dtype=amps.dtype, device=amps.device)
        t = tc.reshape(1, nt, 1).expand(B, nt, 1).reshape(B * nt, 1)
        a = amps.reshape(B, 1, 8).expand(B, nt, 8).reshape(B * nt, 8)
        dd = d.reshape(B, 1, 1).expand(B, nt, 1).reshape(B * nt, 1)
        u, v = model(z, t, a, dd)
        us.append(u.reshape(B, nt)); vs.append(v.reshape(B, nt))
    return torch.cat(us, dim=1), torch.cat(vs, dim=1)


def loss_vectors(model, tau, amps, d, y_power, y_real, y_imag, zeta, observable, time_chunk):
    u, v = terminal_forward(model, tau, amps, d, zeta, time_chunk)
    p = u.square() + v.square()
    power = torch.sum((p - y_power).square(), dim=1) / (torch.sum(y_power.square(), dim=1) + 1e-12)
    complex_ = torch.sum((u - y_real).square() + (v - y_imag).square(), dim=1) / (
        torch.sum(y_real.square() + y_imag.square(), dim=1) + 1e-12
    )
    if observable == "complex": selected = complex_
    elif observable == "power": selected = power
    else: selected = 0.5 * (power + complex_)
    return selected, power, complex_


def build_targets(args, out_dir: Path, model_payload: dict[str, Any], device: torch.device) -> dict[str, np.ndarray]:
    target_dir = ensure_dir(out_dir / f"targets_seed{args.sample_seed}")
    paths = {
        "K": target_dir / "true_K.npy", "A": target_dir / "true_A8.npy", "D": target_dir / "true_D.npy",
        "group": target_dir / "groups.npy", "tau": target_dir / "tau.npy",
        "power": target_dir / "Y_power.npy", "real": target_dir / "Y_real.npy", "imag": target_dir / "Y_imag.npy",
        "meta": target_dir / "meta.json",
    }
    D_specs: list[tuple[str, float]] = []
    for g, text in [("interpolation_D", args.interp_d), ("extrapolation_D", args.extra_d)]:
        for D in parse_float_list(text): D_specs.append((g, float(D)))
    expected = {
        "version": 2, "sample_seed": int(args.sample_seed), "samples_per_d": int(args.samples_per_d),
        "D_specs": D_specs, "min_active_amplitude": float(args.min_active_amplitude),
        "inverse_input_points": int(args.inverse_input_points), "n_t": int(args.n_t), "n_z": int(args.n_z),
        "z_max_ld": float(args.z_max_ld), "ssfm_half_window": float(args.ssfm_half_window),
    }
    if not args.rebuild_targets and all(p.exists() for p in paths.values()):
        try:
            old = json.loads(paths["meta"].read_text(encoding="utf-8"))
            if all(old.get(k) == v for k, v in expected.items()):
                return {k: np.load(paths[k], allow_pickle=(k == "group")) for k in ["K","A","D","group","tau","power","real","imag"]}
        except Exception: pass

    rng = np.random.default_rng(int(args.sample_seed))
    true_K: list[int] = []; true_A: list[np.ndarray] = []; true_D: list[float] = []; groups: list[str] = []
    for g, D in D_specs:
        # Balanced as far as possible inside every D condition.
        ks = np.resize(np.arange(1, 9, dtype=np.int64), int(args.samples_per_d)).copy(); rng.shuffle(ks)
        for K in ks.tolist():
            a = np.zeros(8, dtype=np.float32); slots = active_slots_for_k(int(K), 8)
            a[list(slots)] = rng.uniform(float(args.min_active_amplitude), 1.0, size=int(K)).astype(np.float32)
            true_K.append(int(K)); true_A.append(a); true_D.append(float(D)); groups.append(g)
    true_K_np = np.asarray(true_K, dtype=np.int64); true_A_np = np.stack(true_A).astype(np.float32)
    true_D_np = np.asarray(true_D, dtype=np.float32); groups_np = np.asarray(groups, dtype=object)

    cfg = dict(model_payload["model_config"]); pde = dict(model_payload.get("pde_params", {}))
    grid = make_grid(cfg, args.ssfm_half_window, args.n_t, args.n_z, 2, -44.0, 44.0)
    tau_input = np.linspace(-44.0, 44.0, int(args.inverse_input_points), dtype=np.float32)
    y_real = np.empty((len(true_A_np), len(tau_input)), dtype=np.float32); y_imag = np.empty_like(y_real)
    batch = max(1, int(args.target_ssfm_batch_size)); tau_ref = np.asarray(grid["tau"], dtype=np.float64)
    t0 = time.time()
    for s in range(0, len(true_A_np), batch):
        e = min(len(true_A_np), s + batch)
        maps = run_ssfm_batch_selected_variable_d(true_A_np[s:e], true_D_np[s:e], grid, pde, device, bool(args.ssfm_complex64))
        for j in range(e - s):
            y_real[s+j] = np.interp(tau_input, tau_ref, maps[j, -1, 0]).astype(np.float32)
            y_imag[s+j] = np.interp(tau_input, tau_ref, maps[j, -1, 1]).astype(np.float32)
        print(f"[target SSFM] {e}/{len(true_A_np)} elapsed={time.time()-t0:.1f}s", flush=True)
        del maps
        if device.type == "cuda": torch.cuda.empty_cache()
    y_power = np.square(y_real) + np.square(y_imag)

    np.save(paths["K"], true_K_np); np.save(paths["A"], true_A_np); np.save(paths["D"], true_D_np)
    np.save(paths["group"], groups_np); np.save(paths["tau"], tau_input); np.save(paths["power"], y_power)
    np.save(paths["real"], y_real); np.save(paths["imag"], y_imag); write_json(paths["meta"], expected)
    save_csv_rows(target_dir / "true_parameters.csv", [
        {"sample": i, "group": groups[i], "true_K": int(true_K_np[i]), "true_D": float(true_D_np[i]), "true_A8": vector_text(true_A_np[i])}
        for i in range(len(true_K_np))
    ])
    return {"K": true_K_np, "A": true_A_np, "D": true_D_np, "group": groups_np, "tau": tau_input, "power": y_power, "real": y_real, "imag": y_imag}


def select_candidate(candidate_rows: list[dict[str, Any]], selection: str, n_obs: int, bic_weight: float, smallest_within_fraction: float):
    if selection == "min_loss": return min(candidate_rows, key=lambda r: float(r["loss"]))
    if selection == "smallest_within":
        best = min(float(r["loss"]) for r in candidate_rows); threshold = best * (1.0 + float(smallest_within_fraction))
        return min([r for r in candidate_rows if float(r["loss"]) <= threshold], key=lambda r: int(r["K"]))
    for r in candidate_rows:
        # K active amplitudes + one continuous D parameter.
        n_params = int(r["candidate_K"]) + 1
        r["selection_score_bic"] = float(n_obs) * math.log(max(float(r["loss"]), 1e-15)) + float(bic_weight) * n_params * math.log(float(n_obs))
    return min(candidate_rows, key=lambda r: float(r["selection_score_bic"]))


def relative_l2_complex(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12))


def relative_l2_power(pred_power: np.ndarray, ref_power: np.ndarray) -> float:
    pred = np.asarray(pred_power, dtype=np.float64)
    ref = np.asarray(ref_power, dtype=np.float64)
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-12))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unknown K + continuous amplitudes + beta2 inverse using frozen universal PINN.")
    p.add_argument("--run-dir", required=True); p.add_argument("--checkpoint", default=""); p.add_argument("--out-dir", default="")
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=43); p.add_argument("--sample-seed", type=int, default=2030)
    p.add_argument("--interp-d", default="0.85,0.9,0.95,1.05,1.1,1.15"); p.add_argument("--extra-d", default="0.7,0.75,1.25,1.3")
    p.add_argument("--samples-per-d", type=int, default=10); p.add_argument("--rebuild-targets", action="store_true")
    p.add_argument("--min-active-amplitude", type=float, default=0.05); p.add_argument("--d-lower", type=float, default=0.6); p.add_argument("--d-upper", type=float, default=1.4)
    p.add_argument("--d-restarts", default="0.7,0.9,1.1,1.3"); p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3000); p.add_argument("--lr-a", type=float, default=3e-2); p.add_argument("--lr-d", type=float, default=3e-2)
    p.add_argument("--min-lr", type=float, default=5e-4); p.add_argument("--sample-batch-size", type=int, default=10); p.add_argument("--trajectory-batch-size", type=int, default=64)
    p.add_argument("--forward-time-chunk", type=int, default=256); p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--early-stop-min-epochs", type=int, default=1000); p.add_argument("--early-stop-patience", type=int, default=300); p.add_argument("--early-stop-rel-delta", type=float, default=1e-4)
    p.add_argument("--terminal-observable", choices=["complex","power","power_and_complex"], default="complex")
    p.add_argument("--selection", choices=["bic","min_loss","smallest_within"], default="bic"); p.add_argument("--bic-weight", type=float, default=1.0); p.add_argument("--smallest-within-fraction", type=float, default=0.02)
    p.add_argument("--inverse-input-points", type=int, default=512); p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--n-t", type=int, default=2048); p.add_argument("--n-z", type=int, default=500); p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--target-ssfm-batch-size", type=int, default=32); p.add_argument("--ssfm-complex64", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args(); set_seed(args.seed); device = safe_device(args.device)
    run_dir = Path(args.run_dir).expanduser().resolve(); ckpt = Path(args.checkpoint).expanduser().resolve() if str(args.checkpoint).strip() else run_dir / "sparse8_beta2_forward_pinn.pt"
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else run_dir / "inverse_unknownK_A_D")
    model, payload = load_beta2_checkpoint(ckpt, device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    targets = build_targets(args, out_dir, payload, device)
    tau = torch.tensor(targets["tau"], dtype=torch.float32, device=device)

    R = int(args.restarts); d_starts = parse_float_list(args.d_restarts)
    if len(d_starts) != R: raise ValueError("Number of --d-restarts values must equal --restarts.")
    candidate_k = np.repeat(np.arange(1, 9, dtype=np.int64), R); restart_id = np.tile(np.arange(R, dtype=np.int64), 8)
    masks_np = np.stack([mask_for_k(int(k), 8) for k in candidate_k], axis=0).astype(np.float32)
    masks_t_base = torch.tensor(masks_np, dtype=torch.float32, device=device)
    trajectories_per_sample = 8 * R
    n_samples = len(targets["K"])

    result_rows: list[dict[str, Any]] = []; candidate_rows_all: list[dict[str, Any]] = []; history_rows: list[dict[str, Any]] = []
    t_all = time.time()
    print("=" * 110, flush=True)
    print("Universal inverse: unknown K + A + D", flush=True)
    print(f"samples={n_samples}, sample_batch={args.sample_batch_size}, trajectories/sample={trajectories_per_sample}, trajectory_chunk={args.trajectory_batch_size}", flush=True)
    print(f"D restarts={d_starts}, D bounds=[{args.d_lower},{args.d_upper}], amplitude active range=[{args.min_active_amplitude},1]", flush=True)
    print("=" * 110, flush=True)

    for bs in range(0, n_samples, int(args.sample_batch_size)):
        be = min(n_samples, bs + int(args.sample_batch_size)); B = be - bs; T = trajectories_per_sample
        rng = np.random.default_rng(int(args.seed) + 100003 * bs)
        init_a = rng.uniform(float(args.min_active_amplitude), 1.0, size=(B, T, 8)).astype(np.float32)
        unit_a = (init_a - float(args.min_active_amplitude)) / (1.0 - float(args.min_active_amplitude))
        raw_a = torch.nn.Parameter(torch.tensor(sigmoid_logit_unit(unit_a), dtype=torch.float32, device=device))
        init_d = np.tile(np.asarray(d_starts, dtype=np.float32), 8).reshape(1, T, 1).repeat(B, axis=0)
        raw_d = torch.nn.Parameter(torch.tensor(raw_from_d(init_d, args.d_lower, args.d_upper), dtype=torch.float32, device=device))
        opt = torch.optim.AdamW([
            {"params": [raw_a], "lr": float(args.lr_a)}, {"params": [raw_d], "lr": float(args.lr_d)}
        ], weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.epochs)), eta_min=float(args.min_lr))

        masks = masks_t_base.reshape(1, T, 8).expand(B, T, 8)
        best_loss = torch.full((B, T), float("inf"), device=device); best_power = torch.full_like(best_loss, float("inf")); best_complex = torch.full_like(best_loss, float("inf"))
        best_a = torch.zeros((B, T, 8), dtype=torch.float32, device=device); best_d = torch.zeros((B, T, 1), dtype=torch.float32, device=device); best_epoch = torch.zeros((B, T), dtype=torch.long, device=device)
        monitor = np.full(B, np.inf, dtype=np.float64); stale = np.zeros(B, dtype=np.int64)

        yP = torch.tensor(targets["power"][bs:be], dtype=torch.float32, device=device)
        yR = torch.tensor(targets["real"][bs:be], dtype=torch.float32, device=device)
        yI = torch.tensor(targets["imag"][bs:be], dtype=torch.float32, device=device)

        for ep in range(1, int(args.epochs) + 1):
            opt.zero_grad(set_to_none=True)
            current_selected = torch.empty((B, T), dtype=torch.float32, device=device)
            current_power = torch.empty_like(current_selected)
            current_complex = torch.empty_like(current_selected)
            current_a = torch.empty((B, T, 8), dtype=torch.float32, device=device)
            current_d = torch.empty((B, T, 1), dtype=torch.float32, device=device)

            total_joint = B * T
            chunk = total_joint if int(args.trajectory_batch_size) <= 0 else int(args.trajectory_batch_size)
            for s in range(0, total_joint, chunk):
                e = min(total_joint, s + chunk)
                flat = torch.arange(s, e, dtype=torch.long, device=device)
                sid = torch.div(flat, T, rounding_mode="floor")
                tid = torch.remainder(flat, T)

                # IMPORTANT: build the A/D mapping graph independently for every trajectory chunk.
                # The old implementation created amps_all/d_all once and then called backward()
                # repeatedly on slices of that shared graph, which caused
                # "Trying to backward through the graph a second time" on the second chunk.
                aa = map_raw_amplitudes(raw_a[sid, tid], masks[sid, tid], args.min_active_amplitude)
                dd = map_raw_d(raw_d[sid, tid], args.d_lower, args.d_upper)

                sel, pw, cx = loss_vectors(
                    model,
                    tau,
                    aa,
                    dd,
                    yP.index_select(0, sid),
                    yR.index_select(0, sid),
                    yI.index_select(0, sid),
                    args.z_max_ld,
                    args.terminal_observable,
                    args.forward_time_chunk,
                )
                # Each chunk owns an independent autograd graph. Gradients on raw_a/raw_d
                # still accumulate across all chunks before the single optimizer step.
                sel.sum().backward()

                current_selected[sid, tid] = sel.detach()
                current_power[sid, tid] = pw.detach()
                current_complex[sid, tid] = cx.detach()
                current_a[sid, tid] = aa.detach()
                current_d[sid, tid] = dd.detach()

            # Store the best parameters corresponding to the just-evaluated losses before
            # updating raw_a/raw_d. This is numerically equivalent to the old intent and
            # avoids any mismatch between metrics and parameter snapshots.
            with torch.no_grad():
                improved = current_selected < best_loss
                best_loss = torch.where(improved, current_selected, best_loss)
                best_power = torch.where(improved, current_power, best_power)
                best_complex = torch.where(improved, current_complex, best_complex)
                best_a = torch.where(improved.unsqueeze(-1), current_a, best_a)
                best_d = torch.where(improved.unsqueeze(-1), current_d, best_d)
                best_epoch = torch.where(improved, torch.full_like(best_epoch, ep), best_epoch)

            opt.step()
            sched.step()

            # Selection-aware batch early stopping: monitor the best candidate loss per sample.
            cand = best_loss.reshape(B, 8, R).min(dim=2).values
            current_best = cand.min(dim=1).values.detach().cpu().numpy()
            for i in range(B):
                threshold = max(1e-10, abs(monitor[i]) * float(args.early_stop_rel_delta)) if np.isfinite(monitor[i]) else 0.0
                if current_best[i] < monitor[i] - threshold: monitor[i] = current_best[i]; stale[i] = 0
                else: stale[i] += 1
            if ep == 1 or ep % int(args.log_every) == 0 or ep == args.epochs:
                print(f"[inverse batch {bs}:{be}] step {ep}/{args.epochs} mean_best={float(np.mean(current_best)):.6e}", flush=True)
                history_rows.append({"batch_start": bs, "batch_end": be, "step": ep, "mean_best_loss": float(np.mean(current_best))})
            if ep >= int(args.early_stop_min_epochs) and bool(np.all(stale >= int(args.early_stop_patience))):
                print(f"[inverse batch {bs}:{be}] early stop at {ep}", flush=True); break

        best_loss_np = best_loss.detach().cpu().numpy(); best_power_np = best_power.detach().cpu().numpy(); best_complex_np = best_complex.detach().cpu().numpy()
        best_a_np = best_a.detach().cpu().numpy(); best_d_np = best_d.detach().cpu().numpy().reshape(B, T); best_epoch_np = best_epoch.detach().cpu().numpy()

        # Per-sample model-order selection and SSFM back-substitution.
        cfg = dict(payload["model_config"]); pde = dict(payload.get("pde_params", {})); grid2 = make_grid(cfg, args.ssfm_half_window, args.n_t, args.n_z, 2, -44.0, 44.0)
        tau_ref = np.asarray(grid2["tau"])
        for bi in range(B):
            sample = bs + bi; per_k: list[dict[str, Any]] = []
            for K in range(1, 9):
                s0 = (K - 1) * R; s1 = s0 + R; local = s0 + int(np.argmin(best_loss_np[bi, s0:s1]))
                row = {
                    "sample": sample, "candidate_K": K, "best_restart": int(local - s0), "best_epoch": int(best_epoch_np[bi, local]),
                    "loss": float(best_loss_np[bi, local]), "forward_power_rel_l2": float(math.sqrt(max(best_power_np[bi, local], 0.0))),
                    "forward_complex_rel_l2": float(math.sqrt(max(best_complex_np[bi, local], 0.0))),
                    "pred_D": float(best_d_np[bi, local]), "pred_A8": vector_text(best_a_np[bi, local]),
                }
                per_k.append(row)
            n_obs = int(args.inverse_input_points) * (2 if args.terminal_observable == "complex" else 1)
            selected = select_candidate(per_k, args.selection, n_obs, args.bic_weight, args.smallest_within_fraction)
            candidate_rows_all.extend(per_k)
            pred_K = int(selected["candidate_K"]); pred_A = np.asarray([float(x) for x in str(selected["pred_A8"]).split(";")], dtype=np.float32); pred_D = float(selected["pred_D"])
            ref_complex = targets["real"][sample].astype(np.float64) + 1j * targets["imag"][sample].astype(np.float64)
            maps = run_ssfm_batch_selected_variable_d(pred_A.reshape(1,8), np.asarray([pred_D], dtype=np.float32), grid2, pde, device, bool(args.ssfm_complex64))
            recon_r = np.interp(targets["tau"], tau_ref, maps[0,-1,0]); recon_i = np.interp(targets["tau"], tau_ref, maps[0,-1,1]); recon = recon_r + 1j * recon_i
            recon_power = np.square(recon_r) + np.square(recon_i)
            ref_power = np.asarray(targets["power"][sample], dtype=np.float64)
            true_K = int(targets["K"][sample]); true_A = targets["A"][sample]; true_D = float(targets["D"][sample])
            amp_err = np.abs(pred_A - true_A)
            result_rows.append({
                "sample": sample, "group": str(targets["group"][sample]), "true_K": true_K, "predicted_K": pred_K, "K_exact": int(pred_K == true_K),
                "true_D": true_D, "pred_D": pred_D, "D_abs_error": abs(pred_D - true_D), "D_relative_error": abs(pred_D - true_D) / max(abs(true_D), 1e-12),
                "true_A8": vector_text(true_A), "pred_A8": vector_text(pred_A), "amplitude_8slot_MAE": float(np.mean(amp_err)),
                "active_amplitude_MAE_if_K_correct": float(np.mean(amp_err[list(active_slots_for_k(true_K,8))])) if pred_K == true_K else float("nan"),
                "forward_inverse_loss": float(selected["loss"]), "forward_complex_rel_l2": float(selected["forward_complex_rel_l2"]), "forward_power_rel_l2": float(selected["forward_power_rel_l2"]),
                "ssfm_backsub_terminal_power_rel_l2": relative_l2_power(recon_power, ref_power),
                "ssfm_backsub_terminal_complex_rel_l2": relative_l2_complex(recon, ref_complex),
                "best_restart": int(selected["best_restart"]), "best_epoch": int(selected["best_epoch"]), "selection": args.selection,
            })
            print(
                f"[result {sample+1}/{n_samples}] K {true_K}->{pred_K} | "
                f"D {true_D:.3f}->{pred_D:.5f} | A8_MAE={np.mean(amp_err):.5f} | "
                f"SSFM power backsub={100*result_rows[-1]['ssfm_backsub_terminal_power_rel_l2']:.3f}%",
                flush=True,
            )
            del maps
            if device.type == "cuda": torch.cuda.empty_cache()

    save_csv_rows(out_dir / "per_case.csv", result_rows); save_csv_rows(out_dir / "candidate_K_scores.csv", candidate_rows_all); save_csv_rows(out_dir / "optimization_history.csv", history_rows)

    def summarize_group(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        correct = [r for r in rows if int(r["K_exact"]) == 1]
        return {
            "group": name, "n": len(rows), "K_accuracy": float(np.mean([int(r["K_exact"]) for r in rows])),
            "amplitude_8slot_MAE": float(np.mean([float(r["amplitude_8slot_MAE"]) for r in rows])),
            "active_amplitude_MAE_K_correct": float(np.mean([float(r["active_amplitude_MAE_if_K_correct"]) for r in correct])) if correct else float("nan"),
            "D_MAE": float(np.mean([float(r["D_abs_error"]) for r in rows])),
            "D_relative_MAE": float(np.mean([float(r["D_relative_error"]) for r in rows])),
            "SSFM_backsub_terminal_power_mean": float(np.mean([float(r["ssfm_backsub_terminal_power_rel_l2"]) for r in rows])),
            "SSFM_backsub_terminal_complex_mean": float(np.mean([float(r["ssfm_backsub_terminal_complex_rel_l2"]) for r in rows])),
        }
    group_names = []
    for r in result_rows:
        if r["group"] not in group_names: group_names.append(r["group"])
    summary_rows = [summarize_group(g, [r for r in result_rows if r["group"] == g]) for g in group_names]
    summary_rows.append(summarize_group("overall", result_rows)); save_csv_rows(out_dir / "summary_by_group.csv", summary_rows)
    write_json(out_dir / "summary_overall.json", {**summary_rows[-1], "script_version": SCRIPT_VERSION, "elapsed_sec": time.time() - t_all})
    print("[done]", summary_rows[-1], flush=True)


if __name__ == "__main__":
    main()
