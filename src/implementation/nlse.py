"""
nlse.py

Generic NLSE parameters, multi-Gaussian initial conditions, and 4-level pulse-amplitude
modulation (4-PAM) seen/unseen combination utilities.

This file is designed for multi-pulse PINN + SSFM experiments where the
number of Gaussian sub-pulses M is chosen from the command line.

Main conventions
----------------
- Normalized variables are used internally by SSFM/PINN:
    zeta = Z / LD, tau = T / T0, h = H / sqrt(P0)
- The four symbols 0.25/0.50/0.75/1.00 are normalized FIELD AMPLITUDES.
  Therefore one sub-pulse is level * exp(-(tau-center)^2/2), and its isolated
  peak normalized power is level^2.
- Pulse centers are uniformly spaced by spacing_t0=8 and symmetrically arranged:
    center_k/T0 = (k - (M-1)/2) * spacing_t0, k=0,...,M-1.
- Seen/unseen sets are sets of parameter combinations, not supervised SSFM labels.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence
import csv
import json
import math

import numpy as np

PAM4_AMPLITUDE_LEVELS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
# Backward-compatible alias used by older helper functions. Values now mean
# normalized field amplitudes, not peak powers.
PAM4_LEVELS: tuple[float, ...] = PAM4_AMPLITUDE_LEVELS


def pulse_centers_t0(n_pulses: int, spacing_t0: float = 8.0) -> tuple[float, ...]:
    """Return symmetric pulse centers in units of T0."""
    if int(n_pulses) <= 0:
        raise ValueError("n_pulses must be a positive integer.")
    centers = (np.arange(int(n_pulses), dtype=float) - (int(n_pulses) - 1) / 2.0) * float(spacing_t0)
    return tuple(float(x) for x in centers)


def pinn_half_window_t0(n_pulses: int, spacing_t0: float = 8.0, guard_t0: float = 16.0) -> float:
    """PINN/evaluation half-window in T0 units: outermost center + guard.

    For spacing=8, outermost center = 4*(M-1), so for M=6 and guard=16,
    this returns 36, matching the earlier six-pulse setting.
    """
    centers = pulse_centers_t0(n_pulses, spacing_t0)
    return float(max(abs(c) for c in centers) + float(guard_t0))


def all_pam4_combinations(n_pulses: int, levels: Sequence[float] = PAM4_LEVELS) -> list[tuple[float, ...]]:
    """All |levels|^M normalized 4-PAM amplitude combinations."""
    return [tuple(float(x) for x in c) for c in product(tuple(float(x) for x in levels), repeat=int(n_pulses))]


def _marginal_level_counts(
    combos: Sequence[Sequence[float]],
    levels: Sequence[float] = PAM4_LEVELS,
) -> dict[str, dict[str, int]]:
    """Counts of each normalized amplitude level at each pulse position."""
    if not combos:
        return {}
    n_pulses = len(combos[0])
    out: dict[str, dict[str, int]] = {}
    for j in range(n_pulses):
        counts = {f"{float(level):g}": 0 for level in levels}
        for row in combos:
            key = f"{float(row[j]):g}"
            counts[key] = counts.get(key, 0) + 1
        out[f"P{j+1}"] = counts
    return out


def _balanced_unique_sample(
    all_combos: Sequence[tuple[float, ...]],
    n_seen: int,
    seed: int,
    levels: Sequence[float] = PAM4_LEVELS,
) -> list[tuple[float, ...]]:
    """Balanced random sample without replacement.

    The goal is not a deterministic hand-picked design. It is a reproducible
    random design with approximately balanced one-dimensional marginals: for
    every pulse position, the four PAM4 levels appear as equally often as
    possible in the seen set. This is a lightweight stratified design suitable
    for arbitrary M. Pairwise balance is not enforced.
    """
    if n_seen <= 0:
        return []
    rng = np.random.default_rng(int(seed))
    all_set = set(all_combos)
    n_pulses = len(all_combos[0]) if all_combos else 0
    levels = tuple(float(x) for x in levels)
    seen: list[tuple[float, ...]] = []
    seen_set: set[tuple[float, ...]] = set()

    # Generate rows from independently balanced columns. This keeps every
    # position close to uniform over the four PAM4 levels, while the random
    # permutations avoid a regular lattice pattern.
    attempts = 0
    while len(seen) < n_seen and attempts < 200:
        attempts += 1
        cols: list[np.ndarray] = []
        for _j in range(n_pulses):
            base = n_seen // len(levels)
            rem = n_seen % len(levels)
            counts = np.full(len(levels), base, dtype=int)
            if rem:
                # Randomize which levels get the extra occurrences, but keep it
                # reproducible through the seed.
                extra_order = rng.permutation(len(levels))[:rem]
                counts[extra_order] += 1
            col: list[float] = []
            for level, count in zip(levels, counts):
                col.extend([float(level)] * int(count))
            col_arr = np.array(col, dtype=float)
            rng.shuffle(col_arr)
            cols.append(col_arr)
        rows = list(zip(*cols))
        rng.shuffle(rows)
        for row in rows:
            tup = tuple(float(x) for x in row)
            if tup in all_set and tup not in seen_set:
                seen.append(tup)
                seen_set.add(tup)
                if len(seen) >= n_seen:
                    break

    # Fallback: if duplicate rows prevented reaching n_seen, fill uniformly
    # without replacement from remaining combinations. This should rarely be
    # needed except for tiny M.
    if len(seen) < n_seen:
        remaining = [c for c in all_combos if c not in seen_set]
        need = n_seen - len(seen)
        idx = rng.choice(len(remaining), size=need, replace=False)
        for i in idx.tolist():
            seen.append(remaining[int(i)])

    return seen[:n_seen]


def split_seen_unseen_combinations(
    n_pulses: int,
    train_fraction: float = 0.10,
    seed: int = 42,
    levels: Sequence[float] = PAM4_LEVELS,
    sampling_strategy: str = "balanced",
) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], dict]:
    """Create seen/unseen PAM4 parameter-combination sets.

    Parameters
    ----------
    sampling_strategy:
        "balanced" (default): reproducible stratified random sampling. It
        approximately balances the marginal counts of the four PAM4 levels at
        each pulse position.

        "random": reproducible uniform random sampling without replacement.

    Returns
    -------
    seen, unseen, summary
    """
    all_combos = all_pam4_combinations(n_pulses, levels)
    n_all = len(all_combos)
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")
    n_seen = int(round(float(train_fraction) * n_all))
    n_seen = max(1, min(n_seen, n_all - 1))

    strategy = str(sampling_strategy).strip().lower()
    rng = np.random.default_rng(int(seed))
    if strategy in {"random", "uniform"}:
        idx = rng.choice(n_all, size=n_seen, replace=False)
        idx_set = set(int(i) for i in idx.tolist())
        seen = [all_combos[i] for i in sorted(idx_set)]
    elif strategy in {"balanced", "stratified", "balanced-stratified"}:
        seen = _balanced_unique_sample(all_combos, n_seen, seed=seed, levels=levels)
        strategy = "balanced"
    else:
        raise ValueError("sampling_strategy must be 'balanced' or 'random'.")

    seen_set = set(seen)
    unseen = [c for c in all_combos if c not in seen_set]

    # Marginal-balance diagnostic. For ideal balance, each level appears
    # n_seen/4 times at every pulse position, up to rounding.
    marginal_counts = _marginal_level_counts(seen, levels)
    ideal = len(seen) / float(len(levels))
    max_marginal_deviation = 0.0
    for pos_counts in marginal_counts.values():
        for count in pos_counts.values():
            max_marginal_deviation = max(max_marginal_deviation, abs(float(count) - ideal))

    summary = {
        "n_pulses": int(n_pulses),
        "levels": [float(x) for x in levels],
        "level_quantity": "normalized_field_amplitude",
        "isolated_peak_power_levels": [float(x) ** 2 for x in levels],
        "train_fraction": float(train_fraction),
        "sampling_strategy": strategy,
        "seed": int(seed),
        "n_all": int(n_all),
        "n_seen": int(len(seen)),
        "n_unseen": int(len(unseen)),
        "seen_unique": int(len(set(seen))),
        "unseen_unique": int(len(set(unseen))),
        "seen_unseen_overlap": int(len(set(seen) & set(unseen))),
        "centers_t0": list(pulse_centers_t0(n_pulses)),
        "seen_marginal_level_counts": marginal_counts,
        "seen_marginal_ideal_count_per_level": ideal,
        "seen_marginal_max_deviation_from_ideal": max_marginal_deviation,
    }
    return seen, unseen, summary


def save_combinations_csv(path: str | Path, combos: Sequence[Sequence[float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pulses = len(combos[0]) if combos else 0
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([f"A{i+1}" for i in range(n_pulses)])
        for row in combos:
            writer.writerow([float(x) for x in row])


def load_combinations_csv(path: str | Path) -> list[tuple[float, ...]]:
    path = Path(path)
    rows: list[tuple[float, ...]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for line in reader:
            if not line:
                continue
            rows.append(tuple(float(x) for x in line))
    return rows


def make_and_save_seen_unseen(
    n_pulses: int,
    out_dir: str | Path,
    train_fraction: float = 0.10,
    seed: int = 42,
    sampling_strategy: str = "balanced",
) -> dict:
    """Create seen/unseen combination CSVs and a dataset_summary.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen, unseen, summary = split_seen_unseen_combinations(
        n_pulses, train_fraction=train_fraction, seed=seed, sampling_strategy=sampling_strategy
    )
    save_combinations_csv(out_dir / "seen_combinations.csv", seen)
    save_combinations_csv(out_dir / "unseen_combinations.csv", unseen)
    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


@dataclass(frozen=True)
class NLSEParams:
    """Normalized NLSE parameter set and initial condition definition."""

    # Physical parameters
    beta2: float                 # [s^2/m]
    gamma: float                 # [1/(W*m)]
    P0: float                    # [W]
    T0_ps: float                 # [ps]

    # Optional effects; zero means disabled
    alpha_dB_km: float = 0.0
    beta2_for_ld: float | None = None
    beta3: float = 0.0
    s: float = 0.0
    ss_coef: float = 1.0
    tau_R: float = 0.0

    # Simulation domain
    t_window_t0: float = 52.0
    z_max_ld: float = 4.0
    n_t: int = 2048
    n_z: int = 500

    # Initial condition
    initial_phase_deg: float = 0.0
    pulse_type: str = "gaussian"
    pulse_amplitude: float = 1.0

    # Multi-pulse condition
    multi_pulse_levels: tuple[float, ...] | None = None
    multi_pulse_centers_t0: tuple[float, ...] | None = None
    multi_pulse_level_mode: str = "field"  # amplitude levels; legacy "power" remains supported
    pulse_spacing_t0: float = 8.0

    @property
    def T0_s(self) -> float:
        return self.T0_ps * 1e-12

    @property
    def LD(self) -> float:
        b2 = self.beta2_for_ld if self.beta2_for_ld is not None else self.beta2
        if b2 == 0:
            raise ValueError("beta2_for_ld cannot be zero when beta2 is zero.")
        return self.T0_s ** 2 / abs(b2)

    @property
    def N_sq(self) -> float:
        return self.gamma * self.P0 * self.LD

    @property
    def beta2_norm(self) -> float:
        if self.beta2 > 0:
            return 1.0
        if self.beta2 < 0:
            return -1.0
        return 0.0

    @property
    def alpha_np_m(self) -> float:
        return self.alpha_dB_km / (1000.0 * 10.0 * np.log10(np.e))

    @property
    def alpha_norm(self) -> float:
        return self.alpha_np_m * self.LD

    @property
    def beta3_norm(self) -> float:
        return self.beta3 * self.LD / (self.T0_s ** 3)

    @property
    def z_max_m(self) -> float:
        return self.z_max_ld * self.LD

    @property
    def t_min_ps(self) -> float:
        return -self.t_window_t0 * self.T0_ps

    @property
    def t_max_ps(self) -> float:
        return self.t_window_t0 * self.T0_ps

    @property
    def has_tod(self) -> bool:
        return abs(self.beta3) > 0.0

    @property
    def has_ss(self) -> bool:
        return abs(self.s) > 0.0

    @property
    def has_irs(self) -> bool:
        return abs(self.tau_R) > 0.0

    @property
    def has_loss(self) -> bool:
        return abs(self.alpha_dB_km) > 0.0

    @staticmethod
    def default_centers_t0(n_pulses: int, spacing_t0: float = 8.0) -> tuple[float, ...]:
        return pulse_centers_t0(n_pulses, spacing_t0)

    @property
    def resolved_multi_pulse_centers_t0(self) -> tuple[float, ...] | None:
        if self.multi_pulse_levels is None:
            return None
        if self.multi_pulse_centers_t0 is not None:
            if len(self.multi_pulse_centers_t0) != len(self.multi_pulse_levels):
                raise ValueError("multi_pulse_centers_t0 and multi_pulse_levels must have the same length.")
            return self.multi_pulse_centers_t0
        return self.default_centers_t0(len(self.multi_pulse_levels), self.pulse_spacing_t0)

    def level_to_field_amplitude(self, level: float) -> float:
        if level < 0:
            raise ValueError("Amplitude level cannot be negative.")
        mode = self.multi_pulse_level_mode.lower()
        if mode == "power":
            return float(np.sqrt(level))
        if mode == "field":
            return float(level)
        raise ValueError("multi_pulse_level_mode must be 'power' or 'field'.")

    def with_multi_pulse(
        self,
        levels: Sequence[float],
        centers_t0: Sequence[float] | None = None,
        level_mode: str | None = None,
    ) -> "NLSEParams":
        levels_tuple = tuple(float(x) for x in levels)
        centers_tuple = None if centers_t0 is None else tuple(float(x) for x in centers_t0)
        return replace(
            self,
            multi_pulse_levels=levels_tuple,
            multi_pulse_centers_t0=centers_tuple,
            multi_pulse_level_mode=self.multi_pulse_level_mode if level_mode is None else str(level_mode),
        )

    def _single_shape(self, tau: np.ndarray, center: float, amp: float) -> np.ndarray:
        x = tau - center
        if self.pulse_type.lower() == "sech":
            return amp / np.cosh(x)
        if self.pulse_type.lower() == "gaussian":
            return amp * np.exp(-(x ** 2) / 2.0)
        raise ValueError("pulse_type must be 'gaussian' or 'sech'.")

    def initial_pulse(self, tau: np.ndarray) -> np.ndarray:
        phase = np.exp(1j * np.deg2rad(self.initial_phase_deg))
        if self.multi_pulse_levels is not None:
            centers = self.resolved_multi_pulse_centers_t0
            h = np.zeros_like(tau, dtype=np.complex128)
            for level, center in zip(self.multi_pulse_levels, centers):
                amp = self.level_to_field_amplitude(float(level))
                h += self._single_shape(tau, float(center), amp)
            return h * phase
        return self._single_shape(tau, 0.0, self.pulse_amplitude).astype(np.complex128) * phase

    @staticmethod
    def paper_pam4(**kw) -> "NLSEParams":
        defaults = dict(
            beta2=20e-27,          # 20 ps^2/km = 20e-27 s^2/m
            gamma=2e-3,            # 2 W^-1 km^-1 = 2e-3 W^-1 m^-1
            P0=0.1,
            T0_ps=10.0,
            alpha_dB_km=0.0,
            beta3=0.0,
            s=0.0,
            tau_R=0.0,
            t_window_t0=52.0,
            z_max_ld=4.0,
            n_t=2048,
            n_z=500,
            initial_phase_deg=0.0,
            pulse_type="gaussian",
            pulse_amplitude=1.0,
            multi_pulse_levels=None,
            multi_pulse_centers_t0=None,
            multi_pulse_level_mode="field",
            pulse_spacing_t0=8.0,
        )
        defaults.update(kw)
        return NLSEParams(**defaults)

    def info_str(self) -> str:
        disp = "N" if self.beta2 > 0 else "A" if self.beta2 < 0 else "0"
        parts = [
            f"beta2={self.beta2:.3e}({disp})",
            f"gamma={self.gamma:.3e}",
            f"P0={self.P0:g} W",
            f"T0={self.T0_ps:g} ps",
            f"LD={self.LD/1000:g} km",
            f"N2={self.N_sq:g}",
            f"z={self.z_max_ld:g} LD",
            f"window=±{self.t_window_t0:g}T0",
            f"nt={self.n_t}",
            f"nz={self.n_z}",
        ]
        if self.multi_pulse_levels is not None:
            parts.append(f"M={len(self.multi_pulse_levels)}")
            parts.append(f"centers={self.resolved_multi_pulse_centers_t0}")
        return " | ".join(parts)


if __name__ == "__main__":
    for M in range(2, 10):
        seen, unseen, summary = split_seen_unseen_combinations(M, train_fraction=0.10, seed=42)
        print(summary)
