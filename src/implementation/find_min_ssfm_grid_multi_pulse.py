# -*- coding: utf-8 -*-
"""
Multi-pulse SSFM grid search: minimal time window, n_t, and n_z.

Place this file in the same directory as nlse.py and ssfm.py.
It supports 2,3,4,5,6,7,8,9... Gaussian pulses, with equally spaced centers.

Core idea
---------
For each pulse count M and each candidate grid (half_window_t0, n_t, n_z), run SSFM
and compare the final field against a higher-resolution reference SSFM. A candidate
passes only when:

  1) relative complex-field L2 error <= tol_field
  2) relative power L2 error <= tol_power
  3) time-domain edge leakage <= tol_time_edge
  4) frequency-domain edge leakage <= tol_freq_edge

The recommended grid is the passed candidate with the smallest approximate SSFM cost
    cost = n_z * n_t * log2(n_t)
then smaller n_t, n_z, and half-window as tie breakers.

Examples
--------
# Search only 6 pulses, using your PINN comparison domain [-36,36]
python find_min_ssfm_grid_multi_pulse.py ^
  --device cuda ^
  --pulse-counts 6 ^
  --compare-t-min -36 --compare-t-max 36 ^
  --nts 1024,2048,4096 ^
  --nzs 250,500,1000,2000 ^
  --ref-nt 16384 --ref-nz 4000 ^
  --out-dir GRID_SEARCH_MULTI

# Search 2--9 pulses. This can take time; start with fewer candidates first.
python find_min_ssfm_grid_multi_pulse.py ^
  --device cuda ^
  --pulse-counts 2,3,4,5,6,7,8,9 ^
  --compare-t-min -36 --compare-t-max 36 ^
  --nts 1024,2048,4096 ^
  --nzs 500,1000,2000 ^
  --ref-nt 16384 --ref-nz 4000 ^
  --out-dir GRID_SEARCH_2_TO_9
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from nlse import NLSEParams
from ssfm import run_ssfm


LEVELS = (0.25, 0.5, 0.75, 1.0)


def parse_float_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(float(x.strip())) for x in str(text).split(",") if x.strip()]


def unique_preserve_order(items: Iterable[tuple[float, ...]]) -> list[tuple[float, ...]]:
    seen: set[tuple[float, ...]] = set()
    out: list[tuple[float, ...]] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def default_centers_t0(n_pulses: int, spacing_t0: float = 8.0) -> tuple[float, ...]:
    """Equal-spaced pulse centers centered at zero.

    center_k = (k - (M-1)/2) * spacing_t0, k=0,...,M-1.
    For M=6 and spacing=8, this gives [-20,-12,-4,4,12,20].
    """
    if n_pulses <= 0:
        raise ValueError("n_pulses must be positive")
    centers = (np.arange(n_pulses, dtype=float) - (n_pulses - 1) / 2.0) * spacing_t0
    return tuple(float(x) for x in centers)


def default_diag_combos(n_pulses: int) -> list[tuple[float, ...]]:
    """Representative level combinations used for grid diagnostics.

    These are not training samples; they are stress-test patterns. The all-ones case is
    usually conservative for SPM-induced spectral broadening. Alternating and edge-high
    cases check interaction/edge behavior.
    """
    hi = 1.0
    lo = 0.25
    mid1 = 0.5
    mid2 = 0.75
    combos: list[tuple[float, ...]] = []
    combos.append(tuple([hi] * n_pulses))
    combos.append(tuple([lo] * n_pulses))
    combos.append(tuple(hi if i % 2 == 0 else lo for i in range(n_pulses)))
    combos.append(tuple(lo if i % 2 == 0 else hi for i in range(n_pulses)))

    if n_pulses >= 2:
        combos.append(tuple([hi] + [lo] * (n_pulses - 2) + [hi]))  # high edges
        combos.append(tuple([lo] + [hi] * (n_pulses - 2) + [lo]))  # high center block

    ramp_base = [lo, mid1, mid2, hi]
    ramp = tuple((ramp_base * ((n_pulses + 3) // 4))[:n_pulses])
    combos.append(ramp)
    combos.append(tuple(reversed(ramp)))

    return unique_preserve_order(combos)


def maybe_all_combos(n_pulses: int, threshold: int) -> list[tuple[float, ...]] | None:
    n_all = len(LEVELS) ** n_pulses
    if threshold > 0 and n_all <= threshold:
        return [tuple(float(x) for x in c) for c in product(LEVELS, repeat=n_pulses)]
    return None


def auto_half_windows(
    n_pulses: int,
    spacing_t0: float,
    compare_t_min: float | None,
    compare_t_max: float | None,
    guards: Sequence[float],
) -> list[float]:
    centers = default_centers_t0(n_pulses, spacing_t0)
    center_extent = max(abs(x) for x in centers)
    compare_extent = 0.0
    if compare_t_min is not None:
        compare_extent = max(compare_extent, abs(float(compare_t_min)))
    if compare_t_max is not None:
        compare_extent = max(compare_extent, abs(float(compare_t_max)))
    base = max(center_extent, compare_extent)
    vals = [base + float(g) for g in guards]
    # Keep clean numbers for CSV/JSON.
    return sorted({float(round(x, 10)) for x in vals})


def edge_metrics_time(h: np.ndarray, edge_fraction: float) -> tuple[float, float]:
    I = np.abs(h) ** 2
    peak = float(np.max(I)) + 1e-300
    total = float(np.sum(I)) + 1e-300
    m = max(1, int(round(edge_fraction * I.size)))
    edge = np.concatenate([I[:m], I[-m:]])
    return float(np.max(edge) / peak), float(np.sum(edge) / total)


def edge_metrics_freq(h: np.ndarray, edge_fraction: float) -> tuple[float, float]:
    S = np.abs(np.fft.fftshift(np.fft.fft(h))) ** 2
    peak = float(np.max(S)) + 1e-300
    total = float(np.sum(S)) + 1e-300
    m = max(1, int(round(edge_fraction * S.size)))
    edge = np.concatenate([S[:m], S[-m:]])
    return float(np.max(edge) / peak), float(np.sum(edge) / total)


def interp_complex(x_src: np.ndarray, y_src: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    re = np.interp(x_new, x_src, np.real(y_src))
    im = np.interp(x_new, x_src, np.imag(y_src))
    return re + 1j * im


def relative_errors(
    tau: np.ndarray,
    h: np.ndarray,
    tau_ref: np.ndarray,
    h_ref: np.ndarray,
    compare_t_min: float | None,
    compare_t_max: float | None,
) -> tuple[float, float, int]:
    lo = max(float(tau[0]), float(tau_ref[0]))
    hi = min(float(tau[-1]), float(tau_ref[-1]))
    if compare_t_min is not None:
        lo = max(lo, float(compare_t_min))
    if compare_t_max is not None:
        hi = min(hi, float(compare_t_max))
    mask = (tau >= lo) & (tau <= hi)
    n = int(np.sum(mask))
    if n < 8:
        raise RuntimeError(f"Comparison interval has too few points: lo={lo}, hi={hi}, n={n}")

    x = tau[mask]
    hc = h[mask]
    hr = interp_complex(tau_ref, h_ref, x)

    field_rel = float(np.linalg.norm(hc - hr) / (np.linalg.norm(hr) + 1e-300))
    Pc = np.abs(hc) ** 2
    Pr = np.abs(hr) ** 2
    power_rel = float(np.linalg.norm(Pc - Pr) / (np.linalg.norm(Pr) + 1e-300))
    return field_rel, power_rel, n


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def run_final_field(
    base_params: NLSEParams,
    combo: Sequence[float],
    half_window_t0: float,
    n_t: int,
    n_z: int,
    device: str,
    pulse_spacing_t0: float,
    quiet_ssfm: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    params = NLSEParams.paper_pam4(
        z_max_ld=base_params.z_max_ld,
        n_t=int(n_t),
        n_z=int(n_z),
        t_window_t0=float(half_window_t0),
        pulse_spacing_t0=float(pulse_spacing_t0),
    ).with_multi_pulse(tuple(float(x) for x in combo))

    t0 = time.time()
    with maybe_suppress_stdout(quiet_ssfm):
        z_phys, t_ps, A = run_ssfm(params, device=device, save_every=params.n_z)
    elapsed = time.time() - t0

    tau = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
    h_final = np.asarray(A[-1], dtype=np.complex128)

    del A, z_phys, t_ps
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tau, h_final, elapsed


def candidate_cost(n_t: int, n_z: int) -> float:
    return float(n_z) * float(n_t) * math.log2(max(2, int(n_t)))


def choose_recommendation(rows: list[dict]) -> dict | None:
    passed = [r for r in rows if bool(r.get("passed", False))]
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda r: (
            float(r["cost_proxy"]),
            int(r["n_t"]),
            int(r["n_z"]),
            float(r["half_window_t0"]),
            float(r["worst_relL2_power"]),
        ),
    )[0]


def search_one_pulse_count(args: argparse.Namespace, n_pulses: int) -> dict:
    centers = default_centers_t0(n_pulses, args.pulse_spacing_t0)
    half_windows = parse_float_csv(args.half_windows) if args.half_windows.lower() != "auto" else auto_half_windows(
        n_pulses,
        args.pulse_spacing_t0,
        args.compare_t_min,
        args.compare_t_max,
        parse_float_csv(args.guards),
    )
    nts = parse_int_csv(args.nts)
    nzs = parse_int_csv(args.nzs)

    # Reference window: automatic default is slightly larger than the largest candidate.
    ref_half_window = float(args.ref_half_window) if args.ref_half_window is not None else max(half_windows) + float(args.ref_window_margin)

    all_combos = maybe_all_combos(n_pulses, args.use_all_combos_up_to)
    combos = all_combos if all_combos is not None else default_diag_combos(n_pulses)

    base = NLSEParams.paper_pam4(
        z_max_ld=args.z_max_ld,
        n_t=1,
        n_z=args.ref_nz,
        t_window_t0=max(half_windows),
        pulse_spacing_t0=args.pulse_spacing_t0,
    )

    print("\n" + "=" * 72)
    print(f"Pulse count M = {n_pulses}")
    print(f"centers/T0    = {centers}")
    print(f"candidate half windows/T0 = {half_windows}")
    print(f"candidate n_t = {nts}")
    print(f"candidate n_z = {nzs}")
    print(f"reference     = half_window={ref_half_window}, n_t={args.ref_nt}, n_z={args.ref_nz}")
    print(f"compare domain= [{args.compare_t_min}, {args.compare_t_max}] T0")
    print(f"diagnostic combos = {len(combos)}")

    ref: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
    for i, c in enumerate(combos, start=1):
        print(f"[reference {i}/{len(combos)}] combo={c}")
        tau_r, h_r, sec = run_final_field(
            base,
            c,
            ref_half_window,
            args.ref_nt,
            args.ref_nz,
            args.device,
            args.pulse_spacing_t0,
            args.quiet_ssfm,
        )
        ref[c] = (tau_r, h_r)
        print(f"  done in {sec:.2f}s")

    rows: list[dict] = []
    total_candidates = len(half_windows) * len(nts) * len(nzs)
    k = 0
    for nt in nts:
        for nz in nzs:
            for hw in half_windows:
                k += 1
                worst_field = 0.0
                worst_power = 0.0
                worst_time_edge = 0.0
                worst_time_edge_energy = 0.0
                worst_freq_edge = 0.0
                worst_freq_edge_energy = 0.0
                total_sec = 0.0
                compare_points_min = 10**18
                worst_combo = ""

                print(f"\n[candidate {k}/{total_candidates}] M={n_pulses}, half_window={hw:g}T0, n_t={nt}, n_z={nz}")
                for c in combos:
                    tau, h, sec = run_final_field(
                        base,
                        c,
                        hw,
                        nt,
                        nz,
                        args.device,
                        args.pulse_spacing_t0,
                        args.quiet_ssfm,
                    )
                    total_sec += sec
                    time_edge, time_energy = edge_metrics_time(h, args.edge_fraction)
                    freq_edge, freq_energy = edge_metrics_freq(h, args.edge_fraction)
                    tau_ref, h_ref = ref[c]
                    field_rel, power_rel, n_cmp = relative_errors(
                        tau, h, tau_ref, h_ref, args.compare_t_min, args.compare_t_max
                    )
                    score = max(field_rel, power_rel, time_edge, freq_edge)
                    old_score = max(worst_field, worst_power, worst_time_edge, worst_freq_edge)
                    if score >= old_score:
                        worst_combo = ",".join(f"{x:g}" for x in c)
                    worst_field = max(worst_field, field_rel)
                    worst_power = max(worst_power, power_rel)
                    worst_time_edge = max(worst_time_edge, time_edge)
                    worst_time_edge_energy = max(worst_time_edge_energy, time_energy)
                    worst_freq_edge = max(worst_freq_edge, freq_edge)
                    worst_freq_edge_energy = max(worst_freq_edge_energy, freq_energy)
                    compare_points_min = min(compare_points_min, n_cmp)
                    print(
                        f"  combo={c}: field={field_rel:.3e}, power={power_rel:.3e}, "
                        f"time_edge={time_edge:.3e}, freq_edge={freq_edge:.3e}, sec={sec:.2f}"
                    )

                passed = (
                    worst_field <= args.tol_field
                    and worst_power <= args.tol_power
                    and worst_time_edge <= args.tol_time_edge
                    and worst_freq_edge <= args.tol_freq_edge
                )
                dtau = float(2 * hw / nt)
                row = {
                    "pulse_count": int(n_pulses),
                    "centers_t0": json.dumps(list(centers), ensure_ascii=False),
                    "pulse_spacing_t0": float(args.pulse_spacing_t0),
                    "half_window_t0": float(hw),
                    "total_window_t0": float(2 * hw),
                    "n_t": int(nt),
                    "n_z": int(nz),
                    "dtau": dtau,
                    "points_per_T0": float(nt / (2 * hw)),
                    "omega_nyquist_rad_per_T0": float(math.pi / dtau),
                    "dzeta": float(args.z_max_ld / nz),
                    "cost_proxy": candidate_cost(nt, nz),
                    "worst_relL2_field": worst_field,
                    "worst_relL2_power": worst_power,
                    "worst_time_edge_max_over_peak": worst_time_edge,
                    "worst_time_edge_energy_over_total": worst_time_edge_energy,
                    "worst_freq_edge_max_over_peak": worst_freq_edge,
                    "worst_freq_edge_energy_over_total": worst_freq_edge_energy,
                    "compare_points_min": int(compare_points_min),
                    "worst_combo_hint": worst_combo,
                    "diag_combo_count": int(len(combos)),
                    "elapsed_sec_total": float(total_sec),
                    "passed": bool(passed),
                }
                rows.append(row)
                print("  =>", "PASS" if passed else "FAIL", json.dumps(row, ensure_ascii=False))

    recommendation = choose_recommendation(rows)
    return {
        "pulse_count": int(n_pulses),
        "centers_t0": list(centers),
        "candidate_half_windows_t0": half_windows,
        "candidate_nts": nts,
        "candidate_nzs": nzs,
        "reference": {"half_window_t0": ref_half_window, "n_t": args.ref_nt, "n_z": args.ref_nz},
        "diag_combo_count": len(combos),
        "n_candidates": len(rows),
        "n_passed": sum(1 for r in rows if r["passed"]),
        "recommendation": recommendation,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Find minimal SSFM half-window, n_t, and n_z for M-pulse Gaussian inputs.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pulse-counts", default="6", help="Pulse counts, e.g. 2,3,4,5,6,7,8,9")
    ap.add_argument("--z-max-ld", type=float, default=4.0)
    ap.add_argument("--pulse-spacing-t0", type=float, default=8.0)

    ap.add_argument("--half-windows", default="auto", help="Candidate half windows in T0, or auto")
    ap.add_argument("--guards", default="4,8,12,16,20,24,28,32", help="For auto half-windows: base + guards")
    ap.add_argument("--nts", default="1024,2048,4096", help="Candidate n_t values")
    ap.add_argument("--nzs", default="250,500,1000,2000", help="Candidate n_z values")

    ap.add_argument("--ref-half-window", type=float, default=None, help="Reference half window. Default: max(candidate)+ref_window_margin")
    ap.add_argument("--ref-window-margin", type=float, default=20.0)
    ap.add_argument("--ref-nt", type=int, default=16384)
    ap.add_argument("--ref-nz", type=int, default=4000)

    ap.add_argument("--compare-t-min", type=float, default=None)
    ap.add_argument("--compare-t-max", type=float, default=None)

    ap.add_argument("--tol-field", type=float, default=1e-4)
    ap.add_argument("--tol-power", type=float, default=1e-4)
    ap.add_argument("--tol-time-edge", type=float, default=1e-8)
    ap.add_argument("--tol-freq-edge", type=float, default=1e-8)
    ap.add_argument("--edge-fraction", type=float, default=0.05)

    ap.add_argument("--use-all-combos-up-to", type=int, default=0,
                    help="If 4^M <= this value, use all level combinations for diagnostics; otherwise representative combos are used. 0 disables.")
    ap.add_argument("--quiet-ssfm", action="store_true", help="Suppress verbose output inside run_ssfm")
    ap.add_argument("--out-dir", default="GRID_SEARCH_MULTI_PULSE")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pulse_counts = parse_int_csv(args.pulse_counts)

    all_summaries: list[dict] = []
    all_rows: list[dict] = []
    for m in pulse_counts:
        summary = search_one_pulse_count(args, m)
        all_summaries.append({k: v for k, v in summary.items() if k != "rows"})
        all_rows.extend(summary["rows"])

        # Per-pulse-count CSV/JSON.
        per_csv = out_dir / f"grid_search_M{m}.csv"
        if summary["rows"]:
            with per_csv.open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary["rows"][0].keys()))
                writer.writeheader()
                writer.writerows(summary["rows"])
        per_json = out_dir / f"recommendation_M{m}.json"
        with per_json.open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in summary.items() if k != "rows"}, f, indent=2, ensure_ascii=False)
        print(f"\n[M={m}] CSV  -> {per_csv}")
        print(f"[M={m}] JSON -> {per_json}")
        if summary["recommendation"]:
            print(f"[M={m}] Recommended grid:")
            print(json.dumps(summary["recommendation"], indent=2, ensure_ascii=False))
        else:
            print(f"[M={m}] No candidate passed. Increase candidates or relax tolerances.")

    # Combined outputs.
    if all_rows:
        combined_csv = out_dir / "grid_search_all_pulse_counts.csv"
        with combined_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    combined_json = out_dir / "recommendations_all_pulse_counts.json"
    payload = {
        "thresholds": {
            "tol_field": args.tol_field,
            "tol_power": args.tol_power,
            "tol_time_edge": args.tol_time_edge,
            "tol_freq_edge": args.tol_freq_edge,
        },
        "compare_t_min": args.compare_t_min,
        "compare_t_max": args.compare_t_max,
        "pulse_spacing_t0": args.pulse_spacing_t0,
        "selection_rule": "minimum cost_proxy=n_z*n_t*log2(n_t) among passed candidates; then n_t, n_z, half_window_t0 as tie breakers",
        "summaries": all_summaries,
    }
    with combined_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f"Combined JSON -> {combined_json}")
    if all_rows:
        print(f"Combined CSV  -> {out_dir / 'grid_search_all_pulse_counts.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
