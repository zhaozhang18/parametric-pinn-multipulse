# -*- coding: utf-8 -*-
"""
train_universal_pinn_direct_0to25_fromscratch.py

Direct universal sparse-8 PINN baseline on 0-25 km, trained from random initialization.

The existing 0-20 km final checkpoint is used ONLY to copy:
- model architecture / Fourier-feature configuration
- t-domain configuration
- NLSE physical parameters

Its weights are NOT loaded into the new model.

Default:
- Main Adam: 8000 steps, mixed-K sampling
- High-K refinement: 2000 steps
- L-BFGS: 6 x 100 steps
- No SSFM propagation-field labels in training
- Evaluation batch size defaults to 64

Required helper in the same project folder:
run_universal_pinn_incremental_extend_0to25_purephysics.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

SCRIPT_VERSION = "DIRECT_UNIVERSAL_PINN_0TO25_FROMSCRATCH_V1_20260717"


def load_base():
    path = Path(__file__).resolve().parent / "run_universal_pinn_incremental_extend_0to25_purephysics.py"
    if not path.is_file():
        raise FileNotFoundError("Required helper not found: %s" % path)
    spec = importlib.util.spec_from_file_location("_direct25_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import helper: %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def save_ckpt(path, model, cfg, pde_raw, template_ckpt, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "model_state": model.state_dict(),
            "model_config": dict(cfg),
            "pde_params": dict(pde_raw),
            "direct_0to25_from_scratch": True,
            "template_checkpoint_for_config_only": str(template_ckpt),
            "template_weights_loaded": False,
            "train_meta": dict(meta),
        },
        str(path),
    )


def sample_z_full(n, rng, far_focus, far_start=3.75):
    n = int(n)
    z = rng.uniform(0.0, 5.0, size=n)
    mask = rng.random(n) < float(far_focus)
    m = int(mask.sum())
    if m:
        u = rng.random(m)
        vals = float(far_start) + (5.0 - float(far_start)) * u
        front = rng.random(m) < 0.5
        vals[front] = 5.0 - (5.0 - float(far_start)) * (u[front] ** 2)
        z[mask] = vals
    return z.astype(np.float32).reshape(-1, 1)


def sample_pde(base, sampler, n, cfg, args, rng, device, far_focus):
    _, amps = sampler.sample(int(n), rng)
    z = sample_z_full(int(n), rng, far_focus, args.far_start_ld)
    t = base.sample_time(
        amps, rng,
        float(cfg["t_min"]), float(cfg["t_max"]),
        args.time_global_fraction,
        args.time_pulse_fraction,
        args.time_midpoint_fraction,
        args.pulse_local_sigma,
        args.midpoint_local_sigma,
    )
    return {
        "z": torch.from_numpy(z).to(device),
        "t": torch.from_numpy(t).to(device),
        "a": torch.from_numpy(amps).to(device),
    }


def sample_edge(base, sampler, n, cfg, args, rng, device, far_focus):
    _, amps = sampler.sample(int(n), rng)
    z = sample_z_full(int(n), rng, far_focus, args.far_start_ld)
    side = rng.integers(0, 2, size=int(n))
    t = np.where(side == 0, float(cfg["t_min"]), float(cfg["t_max"]))
    t = t.astype(np.float32).reshape(-1, 1)
    return {
        "z": torch.from_numpy(z).to(device),
        "t": torch.from_numpy(t).to(device),
        "a": torch.from_numpy(amps).to(device),
    }


def power_loss(base, model, sampler, cfg, args, rng, device, far_focus):
    b = int(args.power_cases_batch)
    nz = int(args.power_z_per_case)
    nt = int(args.power_t_points)

    _, amps_np = sampler.sample(b, rng)
    amps = torch.from_numpy(amps_np).to(device)

    t_grid = torch.linspace(
        float(cfg["t_min"]), float(cfg["t_max"]), nt,
        device=device, dtype=torch.float32
    )

    z_np = sample_z_full(b * nz, rng, far_focus, args.far_start_ld).reshape(b, nz)
    z_rand = torch.from_numpy(z_np).to(device)

    z = z_rand[:, :, None].expand(b, nz, nt).reshape(-1, 1)
    t = t_grid[None, None, :].expand(b, nz, nt).reshape(-1, 1)
    a = amps[:, None, None, :].expand(b, nz, nt, 8).reshape(-1, 8)

    u, v = model(z, t, a)
    pz = (u * u + v * v).reshape(b, nz, nt).mean(dim=2)

    t0 = t_grid[None, :].expand(b, nt).reshape(-1, 1)
    a0 = amps[:, None, :].expand(b, nt, 8).reshape(-1, 8)
    u0, v0 = base.analytic_initial_field(t0, a0)
    p0 = (u0 * u0 + v0 * v0).reshape(b, nt).mean(dim=1)

    return torch.mean(((pz - p0[:, None]) ** 2) / (p0[:, None] ** 2 + 1e-8))


def compute_loss(base, model, sampler, cfg, pde_params, args, rng, device, far_focus):
    bpde = sample_pde(base, sampler, args.pde_batch_points, cfg, args, rng, device, far_focus)
    bic = base.sample_ic_batch(sampler, args.ic_batch_points, cfg, args, rng, device)
    bedge = sample_edge(base, sampler, args.edge_batch_points, cfg, args, rng, device, far_focus)

    lpde = base.pde_loss(model, bpde, pde_params["beta2_norm"], pde_params["N_sq"])
    lic = base.ic_loss(model, bic)
    ledge = base.edge_loss(model, bedge) if args.edge_weight > 0 else torch.tensor(0.0, device=device)
    lpower = power_loss(base, model, sampler, cfg, args, rng, device, far_focus) if args.power_weight > 0 else torch.tensor(0.0, device=device)

    loss = (
        args.pde_weight * lpde
        + args.ic_weight * lic
        + args.edge_weight * ledge
        + args.power_weight * lpower
    )
    return loss, {"pde": lpde, "ic": lic, "edge": ledge, "power": lpower}


def train(args):
    base = load_base()
    base.set_seed(args.seed)
    device = base.safe_device(args.device)

    run_dir = Path(args.run_dir).expanduser().resolve()
    template = (
        Path(args.template_checkpoint).expanduser().resolve()
        if args.template_checkpoint
        else run_dir / "sparse8_forward_pinn.pt"
    )
    if not template.is_file():
        raise FileNotFoundError(str(template))

    _, template_payload = base.load_checkpoint(template, device)
    cfg = dict(template_payload["model_config"])
    cfg["z_max_ld"] = 5.0

    pde_raw = dict(template_payload.get("pde_params", {}))
    pde_params = {
        "beta2_norm": float(pde_raw.get("beta2_norm", 1.0)),
        "N_sq": float(pde_raw.get("N_sq", 1.0)),
    }

    exp_root = base.ensure_dir(run_dir / args.exp_name)
    train_root = base.ensure_dir(exp_root / "train")
    history_path = train_root / "history.csv"

    if args.resume_checkpoint:
        resume = Path(args.resume_checkpoint).expanduser().resolve()
        model, payload = base.load_checkpoint(resume, device)
        if abs(float(payload["model_config"].get("z_max_ld", 0.0)) - 5.0) > 1e-6:
            raise RuntimeError("Resume checkpoint is not a direct 0-25 model.")
        cfg = dict(payload["model_config"])
        print("[resume]", resume, flush=True)
    else:
        model = base.ConditionalPINN(**cfg).to(device)
        print("[fresh] random initialization; 0-20 weights NOT loaded.", flush=True)

    model.train()

    seen_k, seen_a = base.load_sparse8_csv(
        run_dir / "dataset" / "seen_sparse8_combinations.csv"
    )
    sampler_main = base.AmplitudeSampler(seen_k, seen_a, "mixed")
    sampler_high = base.AmplitudeSampler(seen_k, seen_a, "highk")

    final_ckpt = train_root / "direct_0to25_final.pt"
    if final_ckpt.is_file() and not args.force:
        print("[skip] final exists:", final_ckpt, flush=True)
        return final_ckpt

    print("=" * 112, flush=True)
    print("DIRECT 0-25 UNIVERSAL PINN FROM SCRATCH", flush=True)
    print("template config :", template, flush=True)
    print("weights loaded  : NO" if not args.resume_checkpoint else "RESUME DIRECT MODEL", flush=True)
    print("model config    :", cfg, flush=True)
    print("main Adam       :", args.main_adam_steps, flush=True)
    print("high-K Adam     :", args.highk_adam_steps, flush=True)
    print("L-BFGS          : %d x %d" % (args.lbfgs_blocks, args.lbfgs_steps_per_block), flush=True)
    print("SSFM labels     : NONE", flush=True)
    print("output          :", exp_root, flush=True)
    print("=" * 112, flush=True)

    rows = []
    rng = np.random.default_rng(args.seed + 252525)
    t0 = time.time()
    global_step = 0

    def run_adam(name, sampler, steps, lr, far_focus):
        nonlocal global_step
        if steps <= 0:
            return
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

        for step in range(1, steps + 1):
            global_step += 1
            opt.zero_grad(set_to_none=True)
            loss, parts = compute_loss(
                base, model, sampler, cfg, pde_params, args, rng, device, far_focus
            )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step == 1 or step % args.log_every == 0 or step == steps:
                row = {
                    "phase": name,
                    "global_step": global_step,
                    "local_step": step,
                    "loss": float(loss.detach().cpu()),
                    "loss_pde": float(parts["pde"].detach().cpu()),
                    "loss_ic": float(parts["ic"].detach().cpu()),
                    "loss_edge": float(parts["edge"].detach().cpu()),
                    "loss_power": float(parts["power"].detach().cpu()),
                    "elapsed_sec": time.time() - t0,
                }
                rows.append(row)
                base.save_rows(history_path, rows)
                print(
                    "[%s %5d] loss=%.4e pde=%.4e ic=%.4e power=%.4e"
                    % (
                        name, step, row["loss"], row["loss_pde"],
                        row["loss_ic"], row["loss_power"]
                    ),
                    flush=True,
                )

        save_ckpt(
            train_root / ("direct_0to25_after_%s.pt" % name),
            model, cfg, pde_raw, template,
            {"phase": name, "global_step": global_step, "args": vars(args)}
        )

    run_adam(
        "main_adam", sampler_main,
        int(args.main_adam_steps), float(args.main_lr),
        float(args.main_far_focus)
    )

    run_adam(
        "highk_adam", sampler_high,
        int(args.highk_adam_steps), float(args.highk_lr),
        float(args.highk_far_focus)
    )

    total_lbfgs = 0
    for block in range(1, args.lbfgs_blocks + 1):
        brng = np.random.default_rng(args.seed + 700000 + block)
        sampler = sampler_high if block % 2 == 0 else sampler_main

        fixed_pde = sample_pde(
            base, sampler, args.lbfgs_pde_points, cfg, args, brng, device,
            args.lbfgs_far_focus
        )
        fixed_ic = base.sample_ic_batch(
            sampler, args.lbfgs_ic_points, cfg, args, brng, device
        )
        fixed_edge = sample_edge(
            base, sampler, args.lbfgs_edge_points, cfg, args, brng, device,
            args.lbfgs_far_focus
        )

        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=1,
            max_eval=args.lbfgs_max_eval,
            history_size=args.lbfgs_history_size,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe" if args.lbfgs_strong_wolfe else None,
        )

        print("\n[L-BFGS block %d/%d]" % (block, args.lbfgs_blocks), flush=True)

        for step in range(1, args.lbfgs_steps_per_block + 1):
            total_lbfgs += 1
            last = {}

            def closure():
                opt.zero_grad(set_to_none=True)
                lpde = base.pde_loss(
                    model, fixed_pde, pde_params["beta2_norm"], pde_params["N_sq"]
                )
                lic = base.ic_loss(model, fixed_ic)
                ledge = base.edge_loss(model, fixed_edge)
                loss = (
                    args.pde_weight * lpde
                    + args.ic_weight * lic
                    + args.edge_weight * ledge
                )
                loss.backward()
                last["loss"] = float(loss.detach().cpu())
                last["pde"] = float(lpde.detach().cpu())
                last["ic"] = float(lic.detach().cpu())
                last["edge"] = float(ledge.detach().cpu())
                return loss

            opt.step(closure)

            if step == 1 or step % args.log_every == 0 or step == args.lbfgs_steps_per_block:
                row = {
                    "phase": "lbfgs",
                    "global_step": total_lbfgs,
                    "local_step": step,
                    "loss": last["loss"],
                    "loss_pde": last["pde"],
                    "loss_ic": last["ic"],
                    "loss_edge": last["edge"],
                    "loss_power": float("nan"),
                    "elapsed_sec": time.time() - t0,
                }
                rows.append(row)
                base.save_rows(history_path, rows)
                print(
                    "[LBFGS %4d] loss=%.4e pde=%.4e ic=%.4e"
                    % (total_lbfgs, row["loss"], row["loss_pde"], row["loss_ic"]),
                    flush=True,
                )

        save_ckpt(
            train_root / ("direct_0to25_after_lbfgs_block%d.pt" % block),
            model, cfg, pde_raw, template,
            {"phase": "lbfgs_block_%d" % block, "args": vars(args)}
        )

    save_ckpt(
        final_ckpt, model, cfg, pde_raw, template,
        {
            "phase": "final",
            "main_adam_steps": args.main_adam_steps,
            "highk_adam_steps": args.highk_adam_steps,
            "lbfgs_steps": args.lbfgs_blocks * args.lbfgs_steps_per_block,
            "args": vars(args),
        }
    )

    base.write_json(
        exp_root / "train_config.json",
        {
            "script_version": SCRIPT_VERSION,
            "template_checkpoint_for_config_only": str(template),
            "template_weights_loaded": False,
            "resume_checkpoint": args.resume_checkpoint,
            "final_checkpoint": str(final_ckpt),
            "model_config": cfg,
            "args": vars(args),
        }
    )

    print("\n[done]", final_ckpt, flush=True)
    return final_ckpt


def evaluate(args):
    base = load_base()
    device = base.safe_device(args.device)

    run_dir = Path(args.run_dir).expanduser().resolve()
    exp_root = run_dir / args.exp_name
    ckpt = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else exp_root / "train" / "direct_0to25_final.pt"
    )
    if not ckpt.is_file():
        raise FileNotFoundError(str(ckpt))

    model, payload = base.load_checkpoint(ckpt, device)
    cfg = dict(payload["model_config"])

    pde_raw = dict(payload.get("pde_params", {}))
    pde_params = {
        "beta2_norm": float(pde_raw.get("beta2_norm", 1.0)),
        "N_sq": float(pde_raw.get("N_sq", 1.0)),
        "alpha_norm": float(pde_raw.get("alpha_norm", 0.0)),
        "beta3_norm": float(pde_raw.get("beta3_norm", 0.0)),
        "has_tod": bool(pde_raw.get("has_tod", False)),
        "has_ss": bool(pde_raw.get("has_ss", False)),
        "has_irs": bool(pde_raw.get("has_irs", False)),
        "s": float(pde_raw.get("s", 0.0)),
        "ss_coef": float(pde_raw.get("ss_coef", 1.0)),
        "tau_R": float(pde_raw.get("tau_R", 0.0)),
    }

    unseen_k, unseen_a = base.load_sparse8_csv(
        run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    )
    pos = base.select_eval_positions(
        unseen_k, int(args.max_eval_per_k), int(args.eval_seed)
    )

    out = base.ensure_dir(exp_root / "eval_0to25")
    metrics_path = out / "metrics_stream.csv"
    if args.overwrite_eval and metrics_path.exists():
        metrics_path.unlink()

    done = set()
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done.add(int(row["idx"]))
                except Exception:
                    pass

    pending = [int(i) for i in pos.tolist() if int(i) not in done]

    grid = base.build_eval_grid(
        cfg, args.ssfm_half_window, args.eval_n_t, args.eval_n_z, args.eval_slices
    )
    zeta = np.asarray(grid["zeta"], dtype=np.float64)
    tau = np.asarray(grid["tau"], dtype=np.float64)
    old_mask = zeta <= 4.0 + 1e-12
    new_mask = zeta >= 4.0 - 1e-12

    fields = (
        ["idx", "K"] + ["A%d" % i for i in range(1, 9)] +
        [
            "full_0to25_rel_l2_field",
            "full_0to25_rel_l2_power",
            "old_0to20_rel_l2_field",
            "old_0to20_rel_l2_power",
            "new_20to25_rel_l2_field",
            "new_20to25_rel_l2_power",
            "terminal_25km_rel_l2_field",
            "terminal_25km_rel_l2_power",
        ]
    )

    new_file = not metrics_path.exists() or metrics_path.stat().st_size == 0
    f = metrics_path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        writer.writeheader()

    print("=" * 110, flush=True)
    print("EVALUATION: direct universal PINN 0-25", flush=True)
    print("checkpoint      :", ckpt, flush=True)
    print("selected unseen :", len(pos), flush=True)
    print("pending         :", len(pending), flush=True)
    print("batch size      :", args.eval_batch_size, flush=True)
    print("output          :", out, flush=True)
    print("=" * 110, flush=True)

    completed = 0
    t0 = time.time()

    try:
        for start in range(0, len(pending), args.eval_batch_size):
            idx = np.asarray(
                pending[start:start + args.eval_batch_size], dtype=np.int64
            )
            amps = unseen_a[idx]
            kvals = unseen_k[idx]

            ref = base.run_ssfm_batch_selected(
                amps, grid, pde_params, device, bool(args.ssfm_complex64)
            )
            pred = base.predict_maps(
                model, amps, tau, zeta, device, int(args.eval_chunk_size)
            )

            ff, fp = base.rel_l2_metrics(pred, ref)
            of, op = base.rel_l2_metrics(pred[:, old_mask], ref[:, old_mask])
            nf, npow = base.rel_l2_metrics(pred[:, new_mask], ref[:, new_mask])
            tf, tp = base.rel_l2_metrics(pred[:, -1:], ref[:, -1:])

            for j, sample_idx in enumerate(idx):
                row = {"idx": int(sample_idx), "K": int(kvals[j])}
                for aidx in range(8):
                    row["A%d" % (aidx + 1)] = float(amps[j, aidx])
                row.update(
                    {
                        "full_0to25_rel_l2_field": float(ff[j]),
                        "full_0to25_rel_l2_power": float(fp[j]),
                        "old_0to20_rel_l2_field": float(of[j]),
                        "old_0to20_rel_l2_power": float(op[j]),
                        "new_20to25_rel_l2_field": float(nf[j]),
                        "new_20to25_rel_l2_power": float(npow[j]),
                        "terminal_25km_rel_l2_field": float(tf[j]),
                        "terminal_25km_rel_l2_power": float(tp[j]),
                    }
                )
                writer.writerow(row)

            f.flush()
            os.fsync(f.fileno())

            completed += len(idx)
            elapsed = time.time() - t0
            rate = completed / max(elapsed, 1e-9)
            eta = (len(pending) - completed) / max(rate, 1e-12)

            print(
                "[eval] %d/%d | old20=%.2f%% new20-25=%.2f%% terminal25=%.2f%% "
                "| rate=%.2f sample/s ETA=%.1f min"
                % (
                    completed, len(pending),
                    100.0 * float(np.mean(op)),
                    100.0 * float(np.mean(npow)),
                    100.0 * float(np.mean(tp)),
                    rate, eta / 60.0
                ),
                flush=True,
            )

            del ref, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        f.close()

    base.summarize_eval(metrics_path, out)


def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--stage", choices=("train", "eval", "all"), default="train")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--exp-name", default="universal_pinn_direct_0to25_baseline_v1")
    p.add_argument("--template-checkpoint", default="")
    p.add_argument("--resume-checkpoint", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")

    p.add_argument("--main-adam-steps", type=int, default=8000)
    p.add_argument("--main-lr", type=float, default=1e-3)
    p.add_argument("--main-far-focus", type=float, default=0.30)

    p.add_argument("--highk-adam-steps", type=int, default=2000)
    p.add_argument("--highk-lr", type=float, default=2e-4)
    p.add_argument("--highk-far-focus", type=float, default=0.55)

    p.add_argument("--pde-batch-points", type=int, default=8192)
    p.add_argument("--ic-batch-points", type=int, default=2048)
    p.add_argument("--edge-batch-points", type=int, default=512)

    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--ic-weight", type=float, default=1.0)
    p.add_argument("--edge-weight", type=float, default=0.05)
    p.add_argument("--power-weight", type=float, default=0.05)

    p.add_argument("--power-cases-batch", type=int, default=8)
    p.add_argument("--power-z-per-case", type=int, default=4)
    p.add_argument("--power-t-points", type=int, default=384)

    p.add_argument("--far-start-ld", type=float, default=3.75)
    p.add_argument("--time-global-fraction", type=float, default=0.4)
    p.add_argument("--time-pulse-fraction", type=float, default=0.4)
    p.add_argument("--time-midpoint-fraction", type=float, default=0.2)
    p.add_argument("--pulse-local-sigma", type=float, default=3.5)
    p.add_argument("--midpoint-local-sigma", type=float, default=2.5)

    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--lbfgs-blocks", type=int, default=6)
    p.add_argument("--lbfgs-steps-per-block", type=int, default=100)
    p.add_argument("--lbfgs-far-focus", type=float, default=0.45)
    p.add_argument("--lbfgs-pde-points", type=int, default=16384)
    p.add_argument("--lbfgs-ic-points", type=int, default=4096)
    p.add_argument("--lbfgs-edge-points", type=int, default=1024)
    p.add_argument("--lbfgs-lr", type=float, default=0.5)
    p.add_argument("--lbfgs-max-eval", type=int, default=4)
    p.add_argument("--lbfgs-history-size", type=int, default=50)
    p.add_argument("--lbfgs-tolerance-grad", type=float, default=1e-10)
    p.add_argument("--lbfgs-tolerance-change", type=float, default=1e-13)
    p.add_argument("--lbfgs-strong-wolfe", action="store_true")

    p.add_argument("--max-eval-per-k", type=int, default=100)
    p.add_argument("--eval-seed", type=int, default=2027)
    p.add_argument("--eval-slices", type=int, default=101)
    p.add_argument("--eval-n-t", type=int, default=2048)
    p.add_argument("--eval-n-z", type=int, default=625)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-chunk-size", type=int, default=131072)
    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--overwrite-eval", action="store_true")

    return p


def main():
    args = build_parser().parse_args()
    trained = None
    if args.stage in ("train", "all"):
        trained = train(args)
    if args.stage in ("eval", "all"):
        if trained is not None and not args.checkpoint:
            args.checkpoint = str(trained)
        evaluate(args)


if __name__ == "__main__":
    main()
