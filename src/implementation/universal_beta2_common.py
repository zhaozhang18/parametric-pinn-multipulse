# -*- coding: utf-8 -*-
"""Shared utilities for the K=1..8 sparse-8 PINN conditioned on dispersion ratio D.

The normalization is fixed to the reference fiber used by NLSEParams.paper_pam4().
D = beta2 / beta2_ref multiplies only the normalized second-order dispersion term;
N^2 remains referenced to the same fixed L_D,ref. This is the same convention used
by the successful fixed-M=4 beta2-generalization experiment.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

try:
    from nlse import pulse_centers_t0
except Exception:
    pulse_centers_t0 = None


METRIC_NAMES = [
    "full_rel_l2_field",
    "full_rel_l2_power",
    "terminal_rel_l2_field",
    "terminal_rel_l2_power",
    "max_abs_power_error",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def safe_device(text: str) -> torch.device:
    requested = str(text).strip().lower()
    if requested != "cpu" and not torch.cuda.is_available():
        print(f"[warning] CUDA unavailable; falling back from {requested!r} to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def parse_float_list(text: str) -> list[float]:
    vals = [
        float(x)
        for x in str(text).replace(";", ",").replace(" ", ",").split(",")
        if x.strip()
    ]
    if not vals:
        raise ValueError("Expected at least one floating-point value.")
    return vals


def active_slots_for_k(k: int, n_slots: int = 8) -> tuple[int, ...]:
    k = int(k)
    if not 1 <= k <= int(n_slots):
        raise ValueError(f"K must lie in [1,{n_slots}], got {k}.")
    left, right = 0, int(n_slots) - 1
    remove_left = True
    while right - left + 1 > k:
        if remove_left:
            left += 1
        else:
            right -= 1
        remove_left = not remove_left
    return tuple(range(left, right + 1))


def mask_for_k(k: int, n_slots: int = 8) -> np.ndarray:
    out = np.zeros(int(n_slots), dtype=np.float32)
    out[list(active_slots_for_k(k, n_slots))] = 1.0
    return out


def load_sparse8_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    ks: list[int] = []
    amps: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        required = ["K"] + [f"A{i}" for i in range(1, 9)]
        missing = [x for x in required if x not in (rd.fieldnames or [])]
        if missing:
            raise RuntimeError(f"Missing columns {missing} in {path}")
        for row in rd:
            ks.append(int(float(row["K"])))
            amps.append([float(row[f"A{i}"]) for i in range(1, 9)])
    if not amps:
        raise RuntimeError(f"No rows found in {path}")
    return np.asarray(ks, dtype=np.int64), np.asarray(amps, dtype=np.float32)


class UniversalDispersionConditionalPINN(nn.Module):
    """Sparse-8 universal PINN with one additional continuous dispersion input D."""

    def __init__(
        self,
        n_pulses: int = 8,
        hidden: int = 128,
        layers: int = 5,
        z_max_ld: float = 4.0,
        t_min: float = -44.0,
        t_max: float = 44.0,
        p_min: float = 0.0,
        p_max: float = 1.0,
        d_min: float = 0.8,
        d_max: float = 1.2,
        fourier_features: int = 4,
    ) -> None:
        super().__init__()
        if int(n_pulses) != 8:
            raise ValueError("This universal model requires exactly 8 amplitude slots.")
        if int(layers) < 1:
            raise ValueError("layers must be >= 1")
        if not float(d_max) > float(d_min):
            raise ValueError("d_max must be greater than d_min")
        self.n_pulses = int(n_pulses)
        self.z_max_ld = float(z_max_ld)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.fourier_features = int(fourier_features)

        in_dim = 3 + self.n_pulses + 4 * max(0, self.fourier_features)
        mods: list[nn.Module] = [nn.Linear(in_dim, int(hidden)), nn.Tanh()]
        for _ in range(int(layers) - 1):
            mods += [nn.Linear(int(hidden), int(hidden)), nn.Tanh()]
        mods.append(nn.Linear(int(hidden), 2))
        self.net = nn.Sequential(*mods)

    def _encode(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        amplitudes: torch.Tensor,
        d_ratio: torch.Tensor,
    ) -> torch.Tensor:
        z_s = 2.0 * z / self.z_max_ld - 1.0
        t_s = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        p_s = 2.0 * (amplitudes - self.p_min) / (self.p_max - self.p_min) - 1.0
        d_s = 2.0 * (d_ratio - self.d_min) / (self.d_max - self.d_min) - 1.0
        feats: list[torch.Tensor] = [z_s, t_s, p_s, d_s]
        for k in range(1, self.fourier_features + 1):
            kk = float(k) * math.pi
            feats += [
                torch.sin(kk * z_s),
                torch.cos(kk * z_s),
                torch.sin(kk * t_s),
                torch.cos(kk * t_s),
            ]
        return torch.cat(feats, dim=1)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        amplitudes: torch.Tensor,
        d_ratio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(self._encode(z, t, amplitudes, d_ratio))
        return out[:, 0:1], out[:, 1:2]

    def config(self) -> Dict[str, float | int]:
        return {
            "n_pulses": int(self.n_pulses),
            "hidden": int(self.net[0].out_features),
            "layers": int((len(self.net) - 1) // 2),
            "z_max_ld": float(self.z_max_ld),
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
            "p_min": float(self.p_min),
            "p_max": float(self.p_max),
            "d_min": float(self.d_min),
            "d_max": float(self.d_max),
            "fourier_features": int(self.fourier_features),
        }


def load_beta2_checkpoint(
    checkpoint: str | Path, device: torch.device
) -> tuple[UniversalDispersionConditionalPINN, dict[str, Any]]:
    checkpoint = Path(checkpoint)
    payload = torch.load(str(checkpoint), map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint}")
    cfg = dict(payload.get("model_config", {}))
    if int(cfg.get("n_pulses", -1)) != 8 or "d_min" not in cfg or "d_max" not in cfg:
        raise RuntimeError(f"Checkpoint is not an 8-slot beta2-conditioned PINN: {cfg}")
    model = UniversalDispersionConditionalPINN(**cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def pde_residual_variable_d(
    model: UniversalDispersionConditionalPINN,
    z: torch.Tensor,
    t: torch.Tensor,
    amplitudes: torch.Tensor,
    d_ratio: torch.Tensor,
    pde: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    z_req = z.detach().clone().requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    amps = amplitudes.detach()
    d = d_ratio.detach()
    u, v = model(z_req, t_req, amps, d)
    ones = torch.ones_like(u)
    u_z = torch.autograd.grad(u, z_req, ones, create_graph=True, retain_graph=True)[0]
    v_z = torch.autograd.grad(v, z_req, ones, create_graph=True, retain_graph=True)[0]
    u_t = torch.autograd.grad(u, t_req, ones, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t_req, ones, create_graph=True, retain_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_req, ones, create_graph=True, retain_graph=True)[0]
    v_tt = torch.autograd.grad(v_t, t_req, ones, create_graph=True, retain_graph=True)[0]

    r2 = u.square() + v.square()
    beta2_eff = float(pde.get("beta2_norm", 1.0)) * d
    disp_f = -(beta2_eff / 2.0) * v_tt
    disp_g = (beta2_eff / 2.0) * u_tt

    if bool(pde.get("has_tod", False)):
        u_ttt = torch.autograd.grad(u_tt, t_req, ones, create_graph=True, retain_graph=True)[0]
        v_ttt = torch.autograd.grad(v_tt, t_req, ones, create_graph=True, retain_graph=True)[0]
        disp_f = disp_f - float(pde.get("beta3_norm", 0.0)) / 6.0 * u_ttt
        disp_g = disp_g - float(pde.get("beta3_norm", 0.0)) / 6.0 * v_ttt

    nl_f = float(pde.get("N_sq", 1.0)) * r2 * v
    nl_g = -float(pde.get("N_sq", 1.0)) * r2 * u
    P_t = None
    if bool(pde.get("has_ss", False)):
        P_t = torch.autograd.grad(r2, t_req, ones, create_graph=True, retain_graph=True)[0]
        ss = float(pde.get("s", 0.0)) * float(pde.get("ss_coef", 0.0))
        nl_f = nl_f + ss * (P_t * u + r2 * u_t)
        nl_g = nl_g + ss * (P_t * v + r2 * v_t)
    if bool(pde.get("has_irs", False)):
        if P_t is None:
            P_t = torch.autograd.grad(r2, t_req, ones, create_graph=True, retain_graph=True)[0]
        N_sq = float(pde.get("N_sq", 1.0))
        tau_R = float(pde.get("tau_R", 0.0))
        nl_f = nl_f - N_sq * tau_R * P_t * v
        nl_g = nl_g + N_sq * tau_R * P_t * u

    f = u_z + float(pde.get("alpha_norm", 0.0)) / 2.0 * u + disp_f + nl_f
    g = v_z + float(pde.get("alpha_norm", 0.0)) / 2.0 * v + disp_g + nl_g
    return f, g


def make_grid(
    model_cfg: Mapping[str, Any],
    ssfm_half_window: float,
    n_t: int,
    n_z: int,
    n_slices: int,
    compare_t_min: Optional[float] = None,
    compare_t_max: Optional[float] = None,
) -> Dict[str, Any]:
    z_max = float(model_cfg.get("z_max_ld", 4.0))
    t_min = float(model_cfg.get("t_min", -44.0)) if compare_t_min is None else float(compare_t_min)
    t_max = float(model_cfg.get("t_max", 44.0)) if compare_t_max is None else float(compare_t_max)
    if not (-float(ssfm_half_window) <= t_min < t_max <= float(ssfm_half_window)):
        raise ValueError("Comparison window must lie inside the SSFM window.")
    if int(n_slices) < 2 or int(n_slices) > int(n_z) + 1:
        raise ValueError("n-slices must lie in [2,n-z+1].")
    tau_full = np.linspace(-float(ssfm_half_window), float(ssfm_half_window), int(n_t), endpoint=False)
    time_mask = (tau_full >= t_min - 1e-12) & (tau_full <= t_max + 1e-12)
    steps = np.rint(np.linspace(0, int(n_z), int(n_slices))).astype(np.int64)
    if len(np.unique(steps)) != len(steps):
        raise ValueError("Selected SSFM steps contain duplicates.")
    return {
        "ssfm_half_window": float(ssfm_half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "n_slices": int(n_slices),
        "z_max_ld": z_max,
        "compare_t_min": t_min,
        "compare_t_max": t_max,
        "tau_full": tau_full,
        "time_mask": time_mask,
        "tau": tau_full[time_mask],
        "selected_steps": steps,
        "zeta": np.linspace(0.0, z_max, int(n_slices), dtype=np.float64),
    }


def build_initial_fields(amplitudes: np.ndarray, tau: np.ndarray) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=np.float64)
    centers = tuple(-28.0 + 8.0 * i for i in range(8))
    basis = np.stack([np.exp(-0.5 * (tau - float(c)) ** 2) for c in centers], axis=0)
    return np.matmul(amps, basis).astype(np.complex128, copy=False)


def run_ssfm_batch_selected_variable_d(
    amplitudes: np.ndarray,
    d_ratio: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    device: torch.device,
    use_complex64: bool = False,
) -> np.ndarray:
    """Normalized SSFM for a batch with per-sample D. Returns [B,Z,2,Tcompare]."""
    amps = np.asarray(amplitudes, dtype=np.float64)
    d_np = np.asarray(d_ratio, dtype=np.float64).reshape(-1)
    if amps.ndim != 2 or amps.shape[1] != 8:
        raise ValueError(f"Expected amplitudes [B,8], got {amps.shape}")
    if len(d_np) != len(amps):
        raise ValueError("D batch size must equal amplitude batch size.")

    tau_full = np.asarray(grid["tau_full"], dtype=np.float64)
    mask = np.asarray(grid["time_mask"], dtype=bool)
    selected_steps = np.asarray(grid["selected_steps"], dtype=np.int64)
    n_z = int(grid["n_z"])
    z_max = float(grid["z_max_ld"])
    dt = float(tau_full[1] - tau_full[0])
    omega = 2.0 * np.pi * np.fft.fftfreq(len(tau_full), d=dt)
    dz = z_max / float(n_z)

    real_dtype = torch.float32 if use_complex64 else torch.float64
    complex_dtype = torch.complex64 if use_complex64 else torch.complex128
    h = torch.as_tensor(build_initial_fields(amps, tau_full), dtype=complex_dtype, device=device)
    omega_t = torch.as_tensor(omega, dtype=real_dtype, device=device).reshape(1, -1)
    d_t = torch.as_tensor(d_np, dtype=real_dtype, device=device).reshape(-1, 1)
    i_omega = (1j * omega_t).to(complex_dtype)

    beta2_eff = float(pde.get("beta2_norm", 1.0)) * d_t
    linear = -float(pde.get("alpha_norm", 0.0)) / 2.0 + 1j * beta2_eff / 2.0 * omega_t.square()
    if bool(pde.get("has_tod", False)):
        linear = linear - 1j * float(pde.get("beta3_norm", 0.0)) / 6.0 * omega_t.pow(3)
    half_prop = torch.exp(linear * (dz / 2.0)).to(complex_dtype)

    step_to_slot = {int(s): i for i, s in enumerate(selected_steps.tolist())}
    out = np.empty((len(amps), len(selected_steps), 2, int(np.count_nonzero(mask))), dtype=np.float32)

    def capture(step: int) -> None:
        slot = step_to_slot[int(step)]
        arr = h.detach().cpu().numpy()[:, mask]
        out[:, slot, 0, :] = arr.real.astype(np.float32)
        out[:, slot, 1, :] = arr.imag.astype(np.float32)

    capture(0)
    N_sq = float(pde.get("N_sq", 1.0))
    for step in range(1, n_z + 1):
        h = torch.fft.ifft(torch.fft.fft(h, dim=-1) * half_prop, dim=-1)
        intensity = torch.abs(h).square()
        h_out = h * torch.exp((1j * N_sq * intensity * dz).to(complex_dtype))
        if bool(pde.get("has_ss", False)):
            prod = intensity * h
            prod_t = torch.fft.ifft(i_omega * torch.fft.fft(prod, dim=-1), dim=-1)
            h_out = h_out - float(pde.get("ss_coef", 0.0)) * N_sq * float(pde.get("s", 0.0)) * prod_t * dz
        if bool(pde.get("has_irs", False)):
            intensity_c = intensity.to(complex_dtype)
            intensity_t = torch.fft.ifft(i_omega * torch.fft.fft(intensity_c, dim=-1), dim=-1)
            h_out = h_out - 1j * N_sq * float(pde.get("tau_R", 0.0)) * intensity_t * h * dz
        h = torch.fft.ifft(torch.fft.fft(h_out, dim=-1) * half_prop, dim=-1)
        if step in step_to_slot:
            capture(step)
    return out


def predict_maps_variable_d(
    model: UniversalDispersionConditionalPINN,
    amplitudes: np.ndarray,
    d_ratio: np.ndarray,
    tau: np.ndarray,
    zeta: np.ndarray,
    device: torch.device,
    flat_chunk_size: int,
) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=np.float32)
    d_np = np.asarray(d_ratio, dtype=np.float32).reshape(-1)
    tau_np = np.asarray(tau, dtype=np.float32)
    zeta_np = np.asarray(zeta, dtype=np.float32)
    b, nz, nt = len(amps), len(zeta_np), len(tau_np)
    total = b * nz * nt
    out = np.empty((total, 2), dtype=np.float32)
    amp_t = torch.as_tensor(amps, dtype=torch.float32, device=device)
    d_t = torch.as_tensor(d_np, dtype=torch.float32, device=device)
    tau_t = torch.as_tensor(tau_np, dtype=torch.float32, device=device)
    zeta_t = torch.as_tensor(zeta_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, total, int(flat_chunk_size)):
            end = min(total, start + int(flat_chunk_size))
            idx = torch.arange(start, end, dtype=torch.long, device=device)
            sample_idx = torch.div(idx, nz * nt, rounding_mode="floor")
            rem = torch.remainder(idx, nz * nt)
            z_idx = torch.div(rem, nt, rounding_mode="floor")
            t_idx = torch.remainder(rem, nt)
            z = zeta_t[z_idx].reshape(-1, 1)
            t = tau_t[t_idx].reshape(-1, 1)
            a = amp_t[sample_idx]
            d = d_t[sample_idx].reshape(-1, 1)
            u, v = model(z, t, a, d)
            out[start:end, 0] = u.reshape(-1).detach().cpu().numpy()
            out[start:end, 1] = v.reshape(-1).detach().cpu().numpy()
    return out.reshape(b, nz, nt, 2).transpose(0, 1, 3, 2)


def metrics_from_maps(pred: np.ndarray, ref: np.ndarray) -> Dict[str, np.ndarray]:
    pred64 = np.asarray(pred, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    eps = 1e-300
    diff = pred64 - ref64
    full_field = np.sqrt(np.sum(diff**2, axis=(1, 2, 3))) / np.maximum(np.sqrt(np.sum(ref64**2, axis=(1, 2, 3))), eps)
    p_pred = np.sum(pred64**2, axis=2)
    p_ref = np.sum(ref64**2, axis=2)
    p_diff = p_pred - p_ref
    full_power = np.sqrt(np.sum(p_diff**2, axis=(1, 2))) / np.maximum(np.sqrt(np.sum(p_ref**2, axis=(1, 2))), eps)
    term_diff = diff[:, -1]
    terminal_field = np.sqrt(np.sum(term_diff**2, axis=(1, 2))) / np.maximum(np.sqrt(np.sum(ref64[:, -1]**2, axis=(1, 2))), eps)
    terminal_power = np.sqrt(np.sum(p_diff[:, -1]**2, axis=1)) / np.maximum(np.sqrt(np.sum(p_ref[:, -1]**2, axis=1)), eps)
    return {
        "full_rel_l2_field": full_field,
        "full_rel_l2_power": full_power,
        "terminal_rel_l2_field": terminal_field,
        "terminal_rel_l2_power": terminal_power,
        "max_abs_power_error": np.max(np.abs(p_diff), axis=(1, 2)),
    }


def select_eval_positions(k_labels: np.ndarray, max_per_k: int, seed: int) -> np.ndarray:
    all_pos = np.arange(len(k_labels), dtype=np.int64)
    if int(max_per_k) <= 0:
        return all_pos
    rng = np.random.default_rng(int(seed))
    chosen: list[int] = []
    for k in sorted(np.unique(k_labels).tolist()):
        pos = all_pos[k_labels == int(k)]
        n = min(len(pos), int(max_per_k))
        if n < len(pos):
            pos = np.sort(rng.choice(pos, size=n, replace=False))
        chosen.extend(int(x) for x in pos.tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def percentile_stats(x: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }
