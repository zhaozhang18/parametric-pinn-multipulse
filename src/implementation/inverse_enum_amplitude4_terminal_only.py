# -*- coding: utf-8 -*-
"""Exhaustive 4-PAM normalized-amplitude inversion with a frozen forward PINN.

The legal normalized field-amplitude symbols are {0.25, 0.50, 0.75, 1.00}.
For each selected forward-unseen SSFM terminal waveform, all 4^M amplitude
vectors are evaluated through the frozen forward PINN. The candidate with the
smallest terminal observable loss is selected.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nlse import PAM4_AMPLITUDE_LEVELS
from train_multi_pulse_pinn import load_forward_checkpoint
from train_inverse_multi_pulse_pinn_pure_physics import (
    safe_device,
    set_seed,
    ensure_dir,
    write_json,
    write_csv_dicts,
    select_best_forward,
    infer_m_from_sources,
    preselect_unseen_samples,
    get_or_build_inverse_dataset,
    attach_dataset_row_indices,
    terminal_field_forward_model,
    terminal_observable_mode,
    combo_to_text,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze forward PINN and enumerate all 4-PAM amplitude candidates.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("-M", "--n-pulses", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="")
    p.add_argument("--forward-model-path", default="")
    p.add_argument("--forward-metrics-csv", default="")
    p.add_argument("--unseen-csv", default="")
    p.add_argument("--sample-indices", default="")
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--terminal-observable", default="complex", choices=["power", "complex", "power_and_complex"])
    p.add_argument("--inverse-dataset-mode", default="selected", choices=["selected", "full"])
    p.add_argument("--inverse-dataset-dir", default="")
    p.add_argument("--rebuild-inverse-dataset", action="store_true")
    p.add_argument("--inverse-dataset-dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--inverse-input-points", type=int, default=512)
    p.add_argument("--inverse-dataset-log-every", type=int, default=1)
    p.add_argument("--verbose-ssfm", action="store_true")
    p.add_argument("--t-window-t0", type=float, default=0.0)
    p.add_argument("--n-t", type=int, default=0)
    p.add_argument("--n-z", type=int, default=0)
    p.add_argument("--z-max-ld", type=float, default=0.0)
    p.add_argument("--compare-t-min", default="")
    p.add_argument("--compare-t-max", default="")
    p.add_argument("--save-terminal-field", action="store_true", default=True)
    p.add_argument("--save-initial-power", action="store_true", default=True)
    p.add_argument("--build-dataset-if-missing", action="store_true", default=True)
    p.add_argument("--inverse-max-combos", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1024, help="Number of candidate amplitude vectors evaluated together.")
    p.add_argument("--sample-batch-size", type=int, default=10, help="Number of observed waveforms compared together; 10 evaluates all default samples in parallel.")
    p.add_argument("--forward-time-chunk", type=int, default=512)
    p.add_argument("--save-topk", type=int, default=10)
    return p.parse_args()


def _loss_vector(
    u: torch.Tensor,
    v: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    observable: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    pred_power = u.square() + v.square()
    ref_power = y_power.reshape(1, -1).expand_as(pred_power)
    power_loss = torch.sum((pred_power - ref_power).square(), dim=1) / (torch.sum(ref_power.square(), dim=1) + 1e-12)
    if observable == "power":
        return power_loss, power_loss, None
    if y_real is None or y_imag is None:
        raise RuntimeError("Complex terminal enumeration requires Y_terminal_real.npy and Y_terminal_imag.npy.")
    ref_u = y_real.reshape(1, -1).expand_as(u)
    ref_v = y_imag.reshape(1, -1).expand_as(v)
    complex_loss = torch.sum((u - ref_u).square() + (v - ref_v).square(), dim=1) / (
        torch.sum(ref_u.square() + ref_v.square(), dim=1) + 1e-12
    )
    if observable == "complex":
        return complex_loss, power_loss, complex_loss
    return 0.5 * (power_loss + complex_loss), power_loss, complex_loss



def _loss_matrix(
    u: torch.Tensor,
    v: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    observable: str,
) -> torch.Tensor:
    """Return [S,C] losses for S observations and C candidate amplitude vectors."""
    pred_power = u.square() + v.square()  # [C,T]
    power_num = torch.sum((pred_power.unsqueeze(0) - y_power.unsqueeze(1)).square(), dim=2)
    power_den = torch.sum(y_power.square(), dim=1, keepdim=True) + 1e-12
    power_loss = power_num / power_den
    if observable == "power":
        return power_loss
    if y_real is None or y_imag is None:
        raise RuntimeError("Complex terminal enumeration requires real and imaginary SSFM observations.")
    complex_num = torch.sum(
        (u.unsqueeze(0) - y_real.unsqueeze(1)).square()
        + (v.unsqueeze(0) - y_imag.unsqueeze(1)).square(),
        dim=2,
    )
    complex_den = torch.sum(y_real.square() + y_imag.square(), dim=1, keepdim=True) + 1e-12
    complex_loss = complex_num / complex_den
    if observable == "complex":
        return complex_loss
    return 0.5 * (power_loss + complex_loss)


def main() -> None:
    wall_clock_t0 = time.time()
    args = parse_args()
    set_seed(args.seed)
    device = safe_device(args.device)
    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(args.out_dir or (run_dir / "inverse" / "02_enum_amplitude4"))

    ckpt, label, info = select_best_forward(args)
    M = infer_m_from_sources(args, run_dir, ckpt)
    args.n_pulses = M
    selected = preselect_unseen_samples(args, run_dir, M, out_dir)
    args.selected_dataset_indices = ",".join(str(int(s["dataset_index"])) for s in selected)
    args.selected_indices_source = "amplitude4_enum_preselection"
    ds_root, ds_meta = get_or_build_inverse_dataset(args, run_dir, M, ckpt, device)
    selected = attach_dataset_row_indices(selected, ds_root, out_dir)

    model = load_forward_checkpoint(ckpt, device).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tau = torch.tensor(np.load(ds_root / "tau_input.npy"), dtype=torch.float32, device=device)
    Yp = np.load(ds_root / "Y_terminal_power.npy", mmap_mode="r")
    Yr = np.load(ds_root / "Y_terminal_real.npy", mmap_mode="r") if (ds_root / "Y_terminal_real.npy").exists() else None
    Yi = np.load(ds_root / "Y_terminal_imag.npy", mmap_mode="r") if (ds_root / "Y_terminal_imag.npy").exists() else None
    row_indices = [int(s["row_index"]) for s in selected]
    Yp_t = torch.tensor(np.asarray(Yp[row_indices], dtype=np.float32), dtype=torch.float32, device=device)
    Yr_t = torch.tensor(np.asarray(Yr[row_indices], dtype=np.float32), dtype=torch.float32, device=device) if Yr is not None else None
    Yi_t = torch.tensor(np.asarray(Yi[row_indices], dtype=np.float32), dtype=torch.float32, device=device) if Yi is not None else None
    true_all = np.asarray([s["powers"] for s in selected], dtype=np.float32)
    zeta = float(getattr(model, "z_max_ld", ds_meta.get("z_max_ld", 4.0)))
    observable = terminal_observable_mode(args)

    candidates = np.asarray(list(product(PAM4_AMPLITUDE_LEVELS, repeat=M)), dtype=np.float32)
    S = len(selected)
    all_losses = np.empty((S, len(candidates)), dtype=np.float64)
    t_all = time.time()
    sample_batch = max(1, int(args.sample_batch_size))
    candidate_batch = max(1, int(args.batch_size))

    # Candidate forward fields are shared by every observation in a sample batch.
    # This avoids evaluating the same 4^M candidates ten separate times.
    with torch.no_grad():
        for ss in range(0, S, sample_batch):
            se = min(S, ss + sample_batch)
            yp = Yp_t[ss:se]
            yr = Yr_t[ss:se] if Yr_t is not None else None
            yi = Yi_t[ss:se] if Yi_t is not None else None
            for cs in range(0, len(candidates), candidate_batch):
                ce = min(len(candidates), cs + candidate_batch)
                a_t = torch.tensor(candidates[cs:ce], dtype=torch.float32, device=device)
                u, v = terminal_field_forward_model(
                    model, tau, a_t, zeta=zeta, chunk_t=int(args.forward_time_chunk)
                )
                loss_mat = _loss_matrix(u, v, yp, yr, yi, observable)
                all_losses[ss:se, cs:ce] = loss_mat.detach().cpu().numpy().astype(np.float64)
            print(
                f"[enum amplitude4] observations {ss+1}-{se}/{S}, "
                f"candidates={len(candidates)}, elapsed={time.time()-t_all:.1f}s",
                flush=True,
            )

    best_indices = np.argmin(all_losses, axis=1)
    best_a_np = candidates[best_indices]
    best_a_t = torch.tensor(best_a_np, dtype=torch.float32, device=device)
    # Re-evaluate selected candidates to report separate power and complex losses.
    with torch.no_grad():
        u, v = terminal_field_forward_model(
            model, tau, best_a_t, zeta=zeta, chunk_t=int(args.forward_time_chunk)
        )
        pred_power = u.square() + v.square()
        power_v = torch.sum((pred_power - Yp_t).square(), dim=1) / (torch.sum(Yp_t.square(), dim=1) + 1e-12)
        if Yr_t is not None and Yi_t is not None:
            complex_v = torch.sum((u - Yr_t).square() + (v - Yi_t).square(), dim=1) / (
                torch.sum(Yr_t.square() + Yi_t.square(), dim=1) + 1e-12
            )
        else:
            complex_v = None

    power_np = power_v.detach().cpu().numpy()
    complex_np = complex_v.detach().cpu().numpy() if complex_v is not None else None
    rows: list[dict[str, Any]] = []
    elapsed_total = float(time.time() - t_all)
    for i, sample in enumerate(selected):
        true_a = true_all[i]
        best_a = best_a_np[i]
        best_loss = float(all_losses[i, best_indices[i]])
        exact = int(bool(np.allclose(best_a, true_a, atol=1e-6)))
        per_pulse = float(np.mean(np.isclose(best_a, true_a, atol=1e-6)))
        item: dict[str, Any] = {
            "sample_rank": int(sample["rank"]),
            "dataset_index": int(sample["dataset_index"]),
            "true_amplitudes": combo_to_text(true_a),
            "pred_amplitudes": combo_to_text(best_a),
            "exact_match": exact,
            "per_pulse_accuracy": per_pulse,
            "amplitude_mae": float(np.mean(np.abs(best_a - true_a))),
            "amplitude_rmse": float(np.sqrt(np.mean((best_a - true_a) ** 2))),
            "terminal_loss": best_loss,
            "terminal_power_loss": float(power_np[i]),
            "terminal_complex_loss": "" if complex_np is None else float(complex_np[i]),
            "enumerated_candidates": int(len(candidates)),
            "elapsed_sec_shared_total": elapsed_total,
        }
        rows.append(item)
        if int(args.save_topk) > 0:
            top_idx = np.argsort(all_losses[i])[: int(args.save_topk)]
            write_json(out_dir / f"sample_{int(sample['rank']):02d}_topk.json", [
                {
                    "rank": rank + 1,
                    "terminal_loss": float(all_losses[i, j]),
                    "amplitudes": [float(x) for x in candidates[j].tolist()],
                }
                for rank, j in enumerate(top_idx.tolist())
            ])
        print(
            f"[enum amplitude4 {i+1}/{S}] true={combo_to_text(true_a)} "
            f"pred={combo_to_text(best_a)} loss={best_loss:.4e} exact={exact}",
            flush=True,
        )

    write_csv_dicts(out_dir / "per_sample_summary.csv", rows)
    access = dict(ds_meta.get("_dataset_access", {}) or {})
    target_ssfm_original = float(access.get("ssfm_generation_elapsed_sec_original", ds_meta.get("build_elapsed_sec", 0.0)) or 0.0)
    target_ssfm_this_run = float(access.get("ssfm_generation_elapsed_sec_this_call", 0.0) or 0.0)
    dataset_prepare_this_run = float(access.get("prepare_elapsed_sec_this_call", 0.0) or 0.0)
    summary = {
        "method": "frozen_forward_exhaustive_amplitude4_enumeration_batched_samples",
        "method_semantics": {
            "target_waveform_generator": "SSFM",
            "target_true_amplitude_levels": [0.25, 0.5, 0.75, 1.0],
            "inverse_search_space": "all 4^M PAM4 candidates",
            "inverse_knows_pam4_levels": True,
        },
        "level_quantity": "normalized_field_amplitude",
        "amplitude_levels": [float(x) for x in PAM4_AMPLITUDE_LEVELS],
        "M": int(M),
        "n_samples": int(len(rows)),
        "candidate_batch_size": candidate_batch,
        "sample_batch_size": sample_batch,
        "enumerated_candidates_per_sample": int(len(candidates)),
        "exact_match_count": int(sum(int(r["exact_match"]) for r in rows)),
        "exact_match_accuracy": float(np.mean([r["exact_match"] for r in rows])) if rows else 0.0,
        "per_pulse_accuracy_mean": float(np.mean([r["per_pulse_accuracy"] for r in rows])) if rows else 0.0,
        "amplitude_mae_mean": float(np.mean([r["amplitude_mae"] for r in rows])) if rows else 0.0,
        "amplitude_rmse_mean": float(np.mean([r["amplitude_rmse"] for r in rows])) if rows else 0.0,
        "terminal_loss_mean": float(np.mean([r["terminal_loss"] for r in rows])) if rows else 0.0,
        "target_ssfm_dataset_reused_this_run": bool(access.get("reused_this_call", False)),
        "target_ssfm_generation_sec_original": target_ssfm_original,
        "target_ssfm_generation_sec_this_run": target_ssfm_this_run,
        "dataset_prepare_sec_this_run": dataset_prepare_this_run,
        "inverse_core_elapsed_sec": elapsed_total,
        "ssfm_reconstruction_elapsed_sec": 0.0,
        "end_to_end_sec_from_scratch": float(target_ssfm_original + elapsed_total),
        "wall_clock_sec_this_run": float(time.time() - wall_clock_t0),
        "elapsed_sec_total": elapsed_total,
        "forward_model": str(ckpt),
        "forward_label": label,
        "forward_selection": info,
        "inverse_dataset": str(ds_root),
        "args": vars(args),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
