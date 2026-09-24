# -*- coding: utf-8 -*-
"""
train_multi_pulse_pinn.py

Generic M-pulse parameter-conditioned forward PINN.

Inputs:
    z/L_D, t/T0, A1,...,AM (normalized field amplitudes)
Outputs:
    Re(h), Im(h)

Training is pure-physics driven:
    IC loss at z=0 + NLSE PDE residual loss.
No SSFM labels are used in training; SSFM is only used for post-training evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats.qmc import LatinHypercube

from nlse import (
    NLSEParams,
    load_combinations_csv,
    make_and_save_seen_unseen,
    pinn_half_window_t0,
    pulse_centers_t0,
)


@dataclass
class PDEParams:
    beta2_norm: float = 1.0
    N_sq: float = 1.0
    alpha_norm: float = 0.0
    beta3_norm: float = 0.0
    has_tod: bool = False
    has_ss: bool = False
    has_irs: bool = False
    s: float = 0.0
    ss_coef: float = 0.0
    tau_R: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_pde_params_from_nlse(beta2_norm: float = 1.0, n_sq: float = 1.0) -> PDEParams:
    params = PDEParams(beta2_norm=float(beta2_norm), N_sq=float(n_sq))
    try:
        p = NLSEParams.paper_pam4()
        params.beta2_norm = float(getattr(p, "beta2_norm", beta2_norm))
        params.N_sq = float(getattr(p, "N_sq", n_sq))
        params.alpha_norm = float(getattr(p, "alpha_norm", 0.0))
        params.beta3_norm = float(getattr(p, "beta3_norm", 0.0))
        params.has_tod = bool(getattr(p, "has_tod", False))
        params.has_ss = bool(getattr(p, "has_ss", False))
        params.has_irs = bool(getattr(p, "has_irs", False))
        params.s = float(getattr(p, "s", 0.0))
        params.ss_coef = float(getattr(p, "ss_coef", 0.0))
        params.tau_R = float(getattr(p, "tau_R", 0.0))
    except Exception as exc:
        warnings.warn(f"Cannot read NLSEParams.paper_pam4(); using CLI defaults. Reason: {exc}")
    return params


class ConditionalPINN(nn.Module):
    """M-pulse conditional PINN conditioned on normalized field amplitudes."""

    def __init__(
        self,
        n_pulses: int,
        hidden: int = 100,
        layers: int = 4,
        z_max_ld: float = 4.0,
        t_min: float = -36.0,
        t_max: float = 36.0,
        p_min: float = 0.25,
        p_max: float = 1.0,
        fourier_features: int = 0,
    ) -> None:
        super().__init__()
        if int(n_pulses) <= 0:
            raise ValueError("n_pulses must be positive.")
        if int(layers) < 1:
            raise ValueError("layers must be >= 1.")
        self.n_pulses = int(n_pulses)
        self.z_max_ld = float(z_max_ld)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.fourier_features = int(fourier_features)

        in_dim = 2 + self.n_pulses + 4 * max(0, self.fourier_features)
        mods: List[nn.Module] = [nn.Linear(in_dim, int(hidden)), nn.Tanh()]
        for _ in range(int(layers) - 1):
            mods += [nn.Linear(int(hidden), int(hidden)), nn.Tanh()]
        mods.append(nn.Linear(int(hidden), 2))
        self.net = nn.Sequential(*mods)

    def _encode(self, z: torch.Tensor, t: torch.Tensor, amplitudes: torch.Tensor) -> torch.Tensor:
        z_s = 2.0 * z / self.z_max_ld - 1.0
        t_s = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        p_s = 2.0 * (amplitudes - self.p_min) / (self.p_max - self.p_min) - 1.0
        feats: List[torch.Tensor] = [z_s, t_s, p_s]
        for k in range(1, self.fourier_features + 1):
            kk = float(k) * math.pi
            feats += [torch.sin(kk * z_s), torch.cos(kk * z_s), torch.sin(kk * t_s), torch.cos(kk * t_s)]
        return torch.cat(feats, dim=1)

    def forward(self, z: torch.Tensor, t: torch.Tensor, amplitudes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._encode(z, t, amplitudes)
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]

    def config(self) -> Dict[str, float | int]:
        # layers = number of hidden layers
        return {
            "n_pulses": int(self.n_pulses),
            "hidden": int(self.net[0].out_features),
            "layers": int((len(self.net) - 1) // 2),
            "z_max_ld": float(self.z_max_ld),
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
            "p_min": float(self.p_min),
            "p_max": float(self.p_max),
            "fourier_features": int(self.fourier_features),
        }


def initial_condition(t: torch.Tensor, amplitudes: torch.Tensor, centers: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """M-Gaussian initial condition h(0,t)."""
    u = torch.zeros_like(t)
    for j, c in enumerate(centers):
        amp = amplitudes[:, j:j+1]
        u = u + amp * torch.exp(-((t - float(c)) ** 2) / 2.0)
    v = torch.zeros_like(u)
    return u, v


def sample_collocation_cond(
    seen_combos: np.ndarray,
    n_pde: int,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    seed: int,
    device: torch.device,
    n_ic: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample PDE and IC collocation points.

    IC sampling is stratified when possible: if n_ic >= n_seen, each seen
    combination appears at least once.
    """
    rng = np.random.default_rng(int(seed))
    seen_combos = np.asarray(seen_combos, dtype=np.float32)
    n_seen = int(len(seen_combos))
    if n_seen <= 0:
        raise RuntimeError("seen_combos is empty.")

    # PDE combinations sampled uniformly from seen set.
    pde_indices = rng.integers(0, n_seen, size=int(n_pde))
    pde_powers = seen_combos[pde_indices]

    # Latin-hypercube in (z,t).
    lh = LatinHypercube(d=2, seed=int(seed))
    zt_samples = lh.random(n=int(n_pde))
    z_pde = z_min + (z_max - z_min) * zt_samples[:, 0:1].astype(np.float32)
    t_pde = t_min + (t_max - t_min) * zt_samples[:, 1:2].astype(np.float32)

    if int(n_ic) <= 0:
        n_ic = max(1, int(n_pde) // 2)
    n_ic = int(n_ic)

    if n_ic >= n_seen:
        base = np.arange(n_seen, dtype=int)
        extra = rng.integers(0, n_seen, size=n_ic - n_seen)
        ic_indices = np.concatenate([base, extra])
        rng.shuffle(ic_indices)
    else:
        warnings.warn(
            f"n_ic={n_ic} < n_seen={n_seen}. Some seen combinations will not appear in this static IC set. "
            "Use dynamic resampling or increase n_ic for full IC coverage."
        )
        ic_indices = rng.choice(n_seen, size=n_ic, replace=False)
    ic_powers = seen_combos[ic_indices]

    lh_t = LatinHypercube(d=1, seed=int(seed) + 1000)
    t_samples = lh_t.random(n=n_ic)
    t_ic = t_min + (t_max - t_min) * t_samples.astype(np.float32)
    z_ic = np.zeros((n_ic, 1), dtype=np.float32)

    tensors = [
        z_pde.astype(np.float32), t_pde.astype(np.float32), pde_powers.astype(np.float32),
        z_ic.astype(np.float32), t_ic.astype(np.float32), ic_powers.astype(np.float32),
    ]
    return tuple(torch.tensor(a, dtype=torch.float32, device=device) for a in tensors)  # type: ignore[return-value]


def pde_residual_cond(
    model: ConditionalPINN,
    z: torch.Tensor,
    t: torch.Tensor,
    amplitudes: torch.Tensor,
    params: PDEParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = z.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    amplitudes = amplitudes.detach()

    u, v = model(z, t, amplitudes)
    ones = torch.ones_like(u)

    u_z = torch.autograd.grad(u, z, ones, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z, ones, create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t, ones, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, ones, create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, ones, create_graph=True, retain_graph=True)[0]
    v_tt = torch.autograd.grad(v_t, t, ones, create_graph=True, retain_graph=True)[0]

    r2 = u ** 2 + v ** 2
    disp_f = -(params.beta2_norm / 2.0) * v_tt
    disp_g = (params.beta2_norm / 2.0) * u_tt

    if params.has_tod:
        u_ttt = torch.autograd.grad(u_tt, t, ones, create_graph=True, retain_graph=True)[0]
        v_ttt = torch.autograd.grad(v_tt, t, ones, create_graph=True, retain_graph=True)[0]
        disp_f = disp_f - params.beta3_norm / 6.0 * u_ttt
        disp_g = disp_g - params.beta3_norm / 6.0 * v_ttt

    nl_f = params.N_sq * r2 * v
    nl_g = -params.N_sq * r2 * u

    P_t = None
    if params.has_ss:
        P_t = torch.autograd.grad(r2, t, ones, create_graph=True, retain_graph=True)[0]
        ss = params.s * params.ss_coef
        nl_f = nl_f + ss * (P_t * u + r2 * u_t)
        nl_g = nl_g + ss * (P_t * v + r2 * v_t)

    if params.has_irs:
        if P_t is None:
            P_t = torch.autograd.grad(r2, t, ones, create_graph=True, retain_graph=True)[0]
        nl_f = nl_f - params.N_sq * params.tau_R * P_t * v
        nl_g = nl_g + params.N_sq * params.tau_R * P_t * u

    f = u_z + (params.alpha_norm / 2.0) * u + disp_f + nl_f
    g = v_z + (params.alpha_norm / 2.0) * v + disp_g + nl_g
    return f, g


def plot_loss_history(loss_history: Dict[str, List[float]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    # log-scale plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if loss_history.get("adam"):
        ax.plot(loss_history["adam"], label="Adam")
    if loss_history.get("lbfgs"):
        offset = len(loss_history.get("adam", []))
        xs = np.arange(len(loss_history["lbfgs"])) + offset
        ax.plot(xs, loss_history["lbfgs"], label="L-BFGS")
    ax.set_xlabel("step / epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_history_log.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # linear-scale plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if loss_history.get("adam"):
        ax.plot(loss_history["adam"], label="Adam")
    if loss_history.get("lbfgs"):
        offset = len(loss_history.get("adam", []))
        xs = np.arange(len(loss_history["lbfgs"])) + offset
        ax.plot(xs, loss_history["lbfgs"], label="L-BFGS")
    ax.set_xlabel("step / epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_history_linear.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_forward(args: argparse.Namespace) -> str:
    set_seed(args.seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.seen_csv:
        seen_combos = load_combinations_csv(args.seen_csv)
    else:
        dataset_dir = out_dir / "dataset"
        make_and_save_seen_unseen(args.n_pulses, dataset_dir, train_fraction=args.train_fraction, seed=args.seed)
        seen_combos = load_combinations_csv(dataset_dir / "seen_combinations.csv")
    seen_arr = np.asarray(seen_combos, dtype=np.float32)
    if seen_arr.shape[1] != int(args.n_pulses):
        raise ValueError(f"seen_csv has {seen_arr.shape[1]} columns but n_pulses={args.n_pulses}")

    if args.auto_t_window:
        half = pinn_half_window_t0(args.n_pulses, guard_t0=args.pinn_guard_t0)
        args.t_min = -half
        args.t_max = half

    z_min, z_max = 0.0, float(args.z_max_ld)
    t_min, t_max = float(args.t_min), float(args.t_max)
    centers = pulse_centers_t0(args.n_pulses)
    p_min = float(seen_arr.min())
    p_max = float(seen_arr.max())

    pde_params = load_pde_params_from_nlse(args.beta2_norm, args.n_sq)
    if args.force_gvd_spm_only:
        pde_params.has_tod = False
        pde_params.has_ss = False
        pde_params.has_irs = False
        pde_params.alpha_norm = 0.0
        pde_params.beta3_norm = 0.0
    pde_params.beta2_norm = float(args.beta2_norm)
    pde_params.N_sq = float(args.n_sq)

    model = ConditionalPINN(
        n_pulses=args.n_pulses,
        hidden=args.hidden,
        layers=args.layers,
        z_max_ld=args.z_max_ld,
        t_min=t_min,
        t_max=t_max,
        p_min=p_min,
        p_max=p_max,
        fourier_features=args.fourier_features,
    ).to(device)

    run_args = vars(args).copy()
    run_args.update({
        "actual_t_min": t_min,
        "actual_t_max": t_max,
        "centers_t0": list(centers),
        "seen_count": int(len(seen_arr)),
        "level_quantity": "normalized_field_amplitude",
        "amplitude_levels": sorted({float(x) for x in seen_arr.reshape(-1).tolist()}),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    })
    with (out_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump(run_args, f, indent=2, ensure_ascii=False)

    # Also save seen combinations in model folder for reproducibility.
    with (out_dir / "seen_combinations.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([f"A{i+1}" for i in range(args.n_pulses)])
        writer.writerows(seen_arr.tolist())

    print("\n========== Train generic multi-pulse PINN ==========")
    print(f"M                 = {args.n_pulses}")
    print(f"device            = {device}")
    print(f"domain            = z/L_D [{z_min}, {z_max}], t/T0 [{t_min}, {t_max}]")
    print(f"seen combinations = {len(seen_arr)} unique")
    print(f"centers           = {centers}")
    print(f"network           = hidden={args.hidden}, layers={args.layers}, fourier_features={args.fourier_features}")
    print(f"parameters        = {sum(p.numel() for p in model.parameters())}")
    print(f"loss weights      = ic={args.ic_weight}, pde={args.pde_weight}")
    print(f"N_IC/N_PDE        = {args.n_ic}/{args.n_pde}")
    print(f"out_dir           = {out_dir}")

    z_pde, t_pde, P_pde, z_ic, t_ic, P_ic = sample_collocation_cond(
        seen_arr, args.n_pde, z_min, z_max, t_min, t_max, args.seed, device, n_ic=args.n_ic
    )
    u0, v0 = initial_condition(t_ic, P_ic, centers)

    def _resample(seed_offset: int):
        nonlocal z_pde, t_pde, P_pde, z_ic, t_ic, P_ic, u0, v0
        z_pde, t_pde, P_pde, z_ic, t_ic, P_ic = sample_collocation_cond(
            seen_arr, args.n_pde, z_min, z_max, t_min, t_max, args.seed + seed_offset, device, n_ic=args.n_ic
        )
        u0, v0 = initial_condition(t_ic, P_ic, centers)

    def _pde_loss() -> torch.Tensor:
        f, g = pde_residual_cond(model, z_pde, t_pde, P_pde, pde_params)
        return (f ** 2).mean() + (g ** 2).mean()

    loss_history: Dict[str, List[float]] = {"adam": [], "lbfgs": []}
    log_rows: List[Dict[str, float | str | int]] = []
    t0 = time.time()

    if args.adam_steps > 0:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"\nAdam steps = {args.adam_steps}")
        for step in range(int(args.adam_steps)):
            resampled = False
            if args.resample_every > 0 and step > 0 and step % args.resample_every == 0:
                _resample(step)
                resampled = True
            opt.zero_grad(set_to_none=True)
            up, vp = model(z_ic, t_ic, P_ic)
            loss_ic = F.mse_loss(up, u0) + F.mse_loss(vp, v0)
            loss_pde = _pde_loss()
            loss = args.ic_weight * loss_ic + args.pde_weight * loss_pde
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            loss_history["adam"].append(float(loss.detach().cpu()))
            if step % args.log_every == 0 or step == args.adam_steps - 1 or resampled:
                elapsed = time.time() - t0
                row = {"phase": "adam", "step": step, "loss": float(loss.detach().cpu()),
                       "ic": float(loss_ic.detach().cpu()), "pde": float(loss_pde.detach().cpu()), "elapsed_sec": elapsed}
                log_rows.append(row)
                print(f"adam {step:6d} | loss={row['loss']:.4e} | ic={row['ic']:.4e} | pde={row['pde']:.4e} | {elapsed:.1f}s")

    early_stop_triggered = False
    if args.lbfgs_epochs > 0:
        print(f"\nL-BFGS epochs <= {args.lbfgs_epochs}, max_iter={args.lbfgs_max_iter}")
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(args.lbfgs_max_iter),
            max_eval=int(args.lbfgs_max_iter) + 10,
            tolerance_grad=1e-11,
            tolerance_change=1e-13,
            line_search_fn="strong_wolfe",
        )
        prev_loss: float | None = None
        patience_count = 0

        for ep in range(int(args.lbfgs_epochs)):
            if args.resample_every > 0 and ep > 0 and ep % args.resample_every == 0:
                _resample(10000 + ep)

            vals = {"ic": math.nan, "pde": math.nan}

            def closure() -> torch.Tensor:
                opt_lbfgs.zero_grad(set_to_none=True)
                up, vp = model(z_ic, t_ic, P_ic)
                loss_ic = F.mse_loss(up, u0) + F.mse_loss(vp, v0)
                loss_pde = _pde_loss()
                loss = args.ic_weight * loss_ic + args.pde_weight * loss_pde
                loss.backward()
                vals["ic"] = float(loss_ic.detach().cpu())
                vals["pde"] = float(loss_pde.detach().cpu())
                return loss

            loss = opt_lbfgs.step(closure)
            loss_val = float(loss.detach().cpu()) if torch.is_tensor(loss) else float(loss)
            loss_history["lbfgs"].append(loss_val)

            # Early stopping based on relative/absolute loss change.
            if prev_loss is not None and ep >= int(args.min_lbfgs_epochs):
                rel_change = abs(loss_val - prev_loss) / max(abs(loss_val), abs(prev_loss), 1.0)
                if rel_change < float(args.early_stop_eps):
                    patience_count += 1
                else:
                    patience_count = 0
                if patience_count >= int(args.early_stop_patience):
                    early_stop_triggered = True
            prev_loss = loss_val

            if ep % args.log_every == 0 or ep == args.lbfgs_epochs - 1 or early_stop_triggered:
                elapsed = time.time() - t0
                row = {"phase": "lbfgs", "step": ep, "loss": loss_val,
                       "ic": vals["ic"], "pde": vals["pde"], "elapsed_sec": elapsed}
                log_rows.append(row)
                print(f"lbfgs {ep:6d} | loss={loss_val:.4e} | ic={vals['ic']:.4e} | pde={vals['pde']:.4e} | patience={patience_count} | {elapsed:.1f}s")
            if early_stop_triggered:
                print(f"Early stopping triggered at L-BFGS epoch {ep}.")
                break

    log_path = out_dir / "forward_train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["phase", "step", "loss", "ic", "pde", "elapsed_sec"])
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)

    ckpt_path = out_dir / "forward_pinn.pt"
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "args": vars(args),
        "pde_params": pde_params.__dict__,
        "n_pulses": int(args.n_pulses),
        "centers_t0": list(centers),
        "seen_combinations": seen_arr.tolist(),
        "level_quantity": "normalized_field_amplitude",
        "seen_combos_count": int(len(seen_arr)),
        "early_stop_triggered": bool(early_stop_triggered),
        "final_loss": float(log_rows[-1]["loss"]) if log_rows else None,
        "final_ic_loss": float(log_rows[-1]["ic"]) if log_rows else None,
        "final_pde_loss": float(log_rows[-1]["pde"]) if log_rows else None,
    }
    torch.save(ckpt, ckpt_path)
    plot_loss_history(loss_history, str(out_dir))

    print("\n========== Training done ==========")
    print("checkpoint ->", ckpt_path)
    print("log        ->", log_path)
    print("loss plots ->", out_dir / "loss_history_log.png", "and", out_dir / "loss_history_linear.png")
    return str(ckpt_path)


def load_forward_checkpoint(path: str | Path, device: torch.device | str) -> ConditionalPINN:
    dev = torch.device(device)
    ckpt = torch.load(path, map_location=dev)
    cfg = dict(ckpt["model_config"])
    model = ConditionalPINN(**cfg).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train generic M-pulse pure-physics forward PINN.")
    p.add_argument("--n-pulses", "-M", type=int, required=True)
    p.add_argument("--seen-csv", type=str, default="", help="CSV of seen initial-condition combinations. If omitted, generated inside out-dir/dataset.")
    p.add_argument("--train-fraction", type=float, default=0.10)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--auto-t-window", action="store_true", default=True)
    p.add_argument("--manual-t-window", action="store_false", dest="auto_t_window")
    p.add_argument("--pinn-guard-t0", type=float, default=16.0)
    p.add_argument("--t-min", type=float, default=-36.0)
    p.add_argument("--t-max", type=float, default=36.0)
    p.add_argument("--beta2-norm", type=float, default=1.0)
    p.add_argument("--n-sq", type=float, default=1.0)
    p.add_argument("--force-gvd-spm-only", action="store_true", default=True)
    p.add_argument("--allow-extra-effects", action="store_false", dest="force_gvd_spm_only")

    p.add_argument("--hidden", type=int, default=100)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--fourier-features", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adam-steps", type=int, default=5000)
    p.add_argument("--lbfgs-epochs", type=int, default=1200)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--min-lbfgs-epochs", type=int, default=100)
    p.add_argument("--early-stop-eps", type=float, default=1e-8)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--n-pde", type=int, default=115000)
    p.add_argument("--n-ic", type=int, default=7500)
    p.add_argument("--resample-every", type=int, default=0)
    p.add_argument("--ic-weight", type=float, default=1.0)
    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_forward(args)


if __name__ == "__main__":
    main()
