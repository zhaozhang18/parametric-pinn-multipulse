"""
ssfm.py

Symmetric split-step Fourier method (Strang splitting) for the normalized NLSE.
This version supports arbitrary M-pulse initial conditions through NLSEParams.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.io import savemat, loadmat

from nlse import NLSEParams


def _spectral_diff(f: torch.Tensor, i_omega: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifft(i_omega * torch.fft.fft(f))


def _nl_step(
    h: torch.Tensor,
    dz: float,
    N2: float,
    s: float,
    tau_R: float,
    ss_coef: float,
    i_omega: torch.Tensor,
    params: NLSEParams,
) -> torch.Tensor:
    I = torch.abs(h) ** 2
    h_out = h * torch.exp(1j * N2 * I * dz)

    if params.has_ss:
        P = I * h
        P_t = _spectral_diff(P, i_omega)
        h_out = h_out - ss_coef * N2 * s * P_t * dz

    if params.has_irs:
        I_t = _spectral_diff(I, i_omega)
        h_out = h_out - 1j * N2 * tau_R * I_t * h * dz

    return h_out


def run_ssfm(params: NLSEParams, device: str = "cuda", save_every: int = 1, quiet: bool = False):
    """Run Strang-splitting SSFM for normalized NLSE.

    Returns
    -------
    z_phys : ndarray, physical distance [m], shape (n_save,)
    t_ps   : ndarray, physical time [ps], shape (n_t,)
    A      : ndarray, normalized complex field h, shape (n_save, n_t)
    """
    if save_every <= 0:
        raise ValueError("save_every must be positive.")
    dev = torch.device(device if (torch.cuda.is_available() and device != "cpu") else "cpu")

    tau = np.linspace(-params.t_window_t0, params.t_window_t0, int(params.n_t), endpoint=False)
    dtau = float(tau[1] - tau[0])
    omega = 2.0 * np.pi * np.fft.fftfreq(int(params.n_t), d=dtau)

    zeta = np.linspace(0.0, float(params.z_max_ld), int(params.n_z) + 1)
    dz = float(zeta[1] - zeta[0])
    dz_half = dz / 2.0

    save_steps = list(range(0, int(params.n_z) + 1, int(save_every)))
    if save_steps[-1] != int(params.n_z):
        save_steps.append(int(params.n_z))
    save_step_to_idx = {step: idx for idx, step in enumerate(save_steps)}

    h0 = params.initial_pulse(tau)
    A = np.zeros((len(save_steps), int(params.n_t)), dtype=np.complex128)
    A[0] = h0

    h = torch.tensor(h0, dtype=torch.complex128, device=dev)
    omega_t = torch.tensor(omega, dtype=torch.float64, device=dev)
    i_omega = 1j * omega_t

    L = -params.alpha_norm / 2.0 + 1j * params.beta2_norm / 2.0 * omega_t ** 2
    if params.has_tod:
        L -= 1j * params.beta3_norm / 6.0 * omega_t ** 3
    half_linear_prop = torch.exp(L * dz_half)

    if not quiet:
        print(f"  SSFM: device={dev}, nt={params.n_t}, nz={params.n_z}, dzeta={dz:.3e}, window=±{params.t_window_t0:g}T0")
        print(f"  beta2_norm={params.beta2_norm:g}, N2={params.N_sq:.6g}")

    for step in range(1, int(params.n_z) + 1):
        h = torch.fft.ifft(torch.fft.fft(h) * half_linear_prop)
        h = _nl_step(h, dz, params.N_sq, params.s, params.tau_R, params.ss_coef, i_omega, params)
        h = torch.fft.ifft(torch.fft.fft(h) * half_linear_prop)

        if step in save_step_to_idx:
            A[save_step_to_idx[step]] = h.detach().cpu().numpy()

        if (not quiet) and (step % 500 == 0 or step == int(params.n_z)):
            print(f"  Progress: {step}/{params.n_z}")

    zeta_saved = np.array(save_steps, dtype=float) * dz
    z_phys = zeta_saved * params.LD
    t_ps = tau * params.T0_ps
    return z_phys, t_ps, A


def save_ssfm(
    z: np.ndarray,
    t: np.ndarray,
    A: np.ndarray,
    params: NLSEParams,
    path: str | Path,
    *,
    sample_idx: int | None = None,
    split: str | None = None,
    save_complex: bool = True,
    save_power: bool = True,
) -> None:
    """Save SSFM results to .mat. Streaming evaluation does not need this."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d: dict[str, Any] = {
        "z_grid": z,
        "t_grid": t,
        "beta2": params.beta2,
        "gamma": params.gamma,
        "P0": params.P0,
        "T0_ps": params.T0_ps,
        "LD": params.LD,
        "N2": params.N_sq,
        "z_max_LD": params.z_max_ld,
        "n_z": params.n_z,
        "n_t": params.n_t,
        "beta2_norm": params.beta2_norm,
        "alpha_norm": params.alpha_norm,
        "t_window_t0": params.t_window_t0,
        "pulse_spacing_t0": params.pulse_spacing_t0,
        "level_mode": params.multi_pulse_level_mode,
    }
    if sample_idx is not None:
        d["sample_idx"] = int(sample_idx)
    if split is not None:
        d["split"] = split
    if params.multi_pulse_levels is not None:
        levels = np.array(params.multi_pulse_levels, dtype=float)
        centers = np.array(params.resolved_multi_pulse_centers_t0, dtype=float)
        d["pam4_levels"] = levels
        d["pulse_centers_t0"] = centers
        d["pulse_centers_ps"] = centers * params.T0_ps
        d["pam4_field_amplitudes"] = np.array([params.level_to_field_amplitude(x) for x in levels], dtype=float)
    if save_complex:
        d["A_field"] = A.astype(np.complex64)
    if save_power:
        d["P_power"] = (np.abs(A) ** 2).astype(np.float32)
        d["initial_power"] = (np.abs(A[0]) ** 2).astype(np.float32)
        d["final_power"] = (np.abs(A[-1]) ** 2).astype(np.float32)
    savemat(str(path), d, do_compression=True)


def load_ssfm(path: str | Path):
    d = loadmat(str(path))
    A = d.get("A_field", d.get("P_power"))
    if A is None:
        raise KeyError("No A_field or P_power in .mat file.")
    z = d["z_grid"].flatten()
    t = d["t_grid"].flatten()
    return A, z, t, d
