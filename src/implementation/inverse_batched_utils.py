# -*- coding: utf-8 -*-
"""Shared vectorized utilities for frozen-forward inverse optimization.

The core speedup is to evaluate several restarts and, optionally, several
observed samples in one forward/backward pass.  For S samples and R restarts,
the optimization batch contains B=S*R amplitude vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from train_inverse_multi_pulse_pinn_pure_physics import terminal_field_forward_model


@dataclass
class BatchedOptimizationResult:
    best_parameter: torch.Tensor
    best_loss: torch.Tensor
    best_epoch: torch.Tensor
    last_improve_epoch: torch.Tensor
    stopped_epoch: int
    plateau_fraction: float
    history: list[dict[str, Any]]


def terminal_loss_vector(
    model: torch.nn.Module,
    tau: torch.Tensor,
    powers: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    zeta: float,
    observable: str,
    terminal_points: int,
    forward_time_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return one relative terminal loss per candidate in ``powers``.

    Shapes
    ------
    powers:  [B, M]
    y_*:     [B, T]
    returns: [B]
    """
    if powers.ndim != 2:
        raise ValueError(f"powers must be [B,M], got {tuple(powers.shape)}")
    if y_power.ndim != 2 or y_power.shape[0] != powers.shape[0]:
        raise ValueError(
            f"y_power must be [B,T] with B={powers.shape[0]}, got {tuple(y_power.shape)}"
        )

    if 0 < int(terminal_points) < int(tau.numel()):
        idx = torch.randperm(int(tau.numel()), device=tau.device)[: int(terminal_points)]
        tau_use = tau[idx]
        yp = y_power[:, idx]
        yr = y_real[:, idx] if y_real is not None else None
        yi = y_imag[:, idx] if y_imag is not None else None
    else:
        tau_use = tau
        yp, yr, yi = y_power, y_real, y_imag

    u, v = terminal_field_forward_model(
        model, tau_use, powers, zeta=float(zeta), chunk_t=int(forward_time_chunk)
    )
    pred_power = u.square() + v.square()
    power_loss = torch.sum((pred_power - yp).square(), dim=1) / (
        torch.sum(yp.square(), dim=1) + 1e-12
    )

    mode = str(observable).strip().lower()
    if mode == "power":
        return power_loss, power_loss, None
    if yr is None or yi is None:
        raise RuntimeError(
            "Complex terminal loss requires Y_terminal_real.npy and Y_terminal_imag.npy."
        )
    complex_loss = torch.sum((u - yr).square() + (v - yi).square(), dim=1) / (
        torch.sum(yr.square() + yi.square(), dim=1) + 1e-12
    )
    if mode == "complex":
        return complex_loss, power_loss, complex_loss
    if mode == "power_and_complex":
        return 0.5 * (power_loss + complex_loss), power_loss, complex_loss
    raise ValueError(f"Unknown terminal observable: {observable}")


def _improved_mask(
    current: torch.Tensor,
    best: torch.Tensor,
    rel_delta: float,
    abs_delta: float,
) -> torch.Tensor:
    finite = torch.isfinite(best)
    threshold = torch.maximum(
        torch.full_like(best, float(abs_delta)),
        torch.abs(best) * float(rel_delta),
    )
    return (~finite) | (current < best - threshold)


def run_batched_adamw(
    *,
    model: torch.nn.Module,
    tau: torch.Tensor,
    y_power: torch.Tensor,
    y_real: torch.Tensor | None,
    y_imag: torch.Tensor | None,
    initial_parameter: torch.Tensor,
    decode_parameter: Callable[[torch.Tensor], torch.Tensor],
    zeta: float,
    observable: str,
    epochs: int,
    lr: float,
    min_lr: float,
    cosine_anneal: bool,
    terminal_points: int,
    forward_time_chunk: int,
    log_every: int,
    early_stop: bool,
    early_stop_min_epochs: int,
    early_stop_patience: int,
    early_stop_fraction: float,
    early_stop_rel_delta: float,
    early_stop_abs_delta: float,
    sample_ids: np.ndarray,
    restart_ids: np.ndarray,
    mode: str,
    history_extra: Callable[[np.ndarray], list[dict[str, Any]]] | None = None,
) -> BatchedOptimizationResult:
    """Optimize a whole [samples x restarts] batch with one AdamW optimizer.

    The maximum epoch budget can remain 3000.  Training stops automatically
    once the requested fraction (default 90%) of restart trajectories has not
    improved for ``patience`` epochs after ``min_epochs``.  The best parameter
    state is retained independently for every restart.
    """
    if not (0.0 < float(early_stop_fraction) <= 1.0):
        raise ValueError("early_stop_fraction must be in (0,1].")
    parameter = torch.nn.Parameter(initial_parameter.detach().clone())
    optimizer = torch.optim.AdamW([parameter], lr=float(lr), weight_decay=0.0)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(epochs)), eta_min=float(min_lr)
        )
        if cosine_anneal
        else None
    )

    B = int(parameter.shape[0])
    if len(sample_ids) != B or len(restart_ids) != B:
        raise ValueError("sample_ids/restart_ids length must equal optimization batch size.")
    best_loss = torch.full((B,), float("inf"), dtype=torch.float32, device=parameter.device)
    best_epoch = torch.zeros((B,), dtype=torch.long, device=parameter.device)
    last_improve = torch.zeros((B,), dtype=torch.long, device=parameter.device)
    best_parameter = parameter.detach().clone()
    history: list[dict[str, Any]] = []
    stopped_epoch = int(epochs)
    plateau_fraction = 0.0

    for ep in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        powers = decode_parameter(parameter)
        losses, _, _ = terminal_loss_vector(
            model,
            tau,
            powers,
            y_power,
            y_real,
            y_imag,
            zeta,
            observable,
            terminal_points,
            forward_time_chunk,
        )
        detached = losses.detach()
        improved = _improved_mask(
            detached, best_loss, float(early_stop_rel_delta), float(early_stop_abs_delta)
        )
        if bool(torch.any(improved)):
            best_loss = torch.where(improved, detached, best_loss)
            best_epoch = torch.where(
                improved, torch.full_like(best_epoch, ep), best_epoch
            )
            last_improve = torch.where(
                improved, torch.full_like(last_improve, ep), last_improve
            )
            best_parameter[improved] = parameter.detach()[improved]

        # Sum preserves the per-restart gradient that separate optimizers would see.
        # The parameter blocks are independent, so no cross-restart gradient is introduced.
        losses.sum().backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        plateau = (ep - last_improve) >= int(early_stop_patience)
        plateau_fraction = float(plateau.float().mean().detach().cpu())
        should_log = ep == 1 or ep % max(1, int(log_every)) == 0 or ep == int(epochs)
        if should_log:
            p_np = powers.detach().cpu().numpy()
            cur_np = detached.cpu().numpy()
            best_np = best_loss.detach().cpu().numpy()
            best_ep_np = best_epoch.detach().cpu().numpy()
            extras = history_extra(p_np) if history_extra is not None else [{} for _ in range(B)]
            for b in range(B):
                history.append(
                    {
                        "mode": mode,
                        "sample": int(sample_ids[b]),
                        "restart": int(restart_ids[b]),
                        "epoch": int(ep),
                        "terminal_loss": float(cur_np[b]),
                        "best_loss": float(best_np[b]),
                        "best_epoch": int(best_ep_np[b]),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "plateau": int(bool(plateau[b].detach().cpu())),
                        "plateau_fraction": plateau_fraction,
                        **extras[b],
                    }
                )
            print(
                f"[{mode}] ep={ep}/{epochs} mean={float(detached.mean().cpu()):.4e} "
                f"best_mean={float(best_loss.mean().cpu()):.4e} "
                f"plateau={plateau_fraction:.1%}",
                flush=True,
            )

        if (
            early_stop
            and ep >= int(early_stop_min_epochs)
            and plateau_fraction >= float(early_stop_fraction)
        ):
            stopped_epoch = ep
            break

    return BatchedOptimizationResult(
        best_parameter=best_parameter.detach(),
        best_loss=best_loss.detach(),
        best_epoch=best_epoch.detach(),
        last_improve_epoch=last_improve.detach(),
        stopped_epoch=int(stopped_epoch),
        plateau_fraction=float(plateau_fraction),
        history=history,
    )
