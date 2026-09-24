# -*- coding: utf-8 -*-
"""
visualize_universal_beta2_forward_representativeK8_D11_paper_SAMESTYLE_TIMES14.py

用途：
- 完整保留第一个通用模型正向可视化脚本的画图布局、配色、坐标、线型、字号与输出格式。
- 数据源改为加入色散条件 D 的第二个通用模型。
- 固定 K=8、D=1.1，并沿用原来的最佳样本标准：
  full_rel_l2_power 最小为主，terminal_rel_l2_power 最小为次，idx 最小为最终并列规则。
- 仅更换模型、SSFM/PINN 推理接口以及样本数据源，绘图函数不作改动。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from train_multi_pulse_pinn import ConditionalPINN

SSFM_COLOR = "#0000FF"
PINN_COLOR = "#FF0000"
GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#111111"
BASELINE_COLOR = "#C9CDD2"
AXIS_COLOR = "#555555"

REFERENCE_CMAP = LinearSegmentedColormap.from_list(
    "reference_parula",
    [
        "#4B67AE", "#4D82B8", "#4FA7B3", "#62C2A5", "#A4D98B",
        "#E8EA79", "#F9D65C", "#F5A44A", "#E45B3B", "#B51F2E",
    ],
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "font.size": 14.0,
    "font.weight": "bold",
    "axes.titlesize": 14.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 14.0,
    "axes.labelweight": "bold",
    "legend.fontsize": 14.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def read_sparse8_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    k_values = []
    amplitudes = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            k_values.append(int(float(row["K"])))
            amplitudes.append([float(row[f"A{i}"]) for i in range(1, 9)])
    return np.asarray(k_values, dtype=np.int64), np.asarray(amplitudes, dtype=np.float32)


def select_best_k_d_from_metrics(
    metrics_csv: Path,
    target_k: int,
    target_d: float,
    d_tolerance: float,
    amplitudes_all: np.ndarray,
    min_distinct_amplitudes: int,
    max_adjacent_run: int,
    min_adjacent_changes: int,
    max_same_level_count: int,
) -> dict[str, Any]:
    """Select the lowest-error representative mixed-amplitude K-D sample.

    The model-error ranking remains unchanged after a transparent waveform-complexity
    filter. The filter prevents visually artificial cases such as all-equal amplitudes
    or long blocks of repeated levels:
      - at least ``min_distinct_amplitudes`` distinct amplitude levels;
      - no amplitude level occurs more than ``max_same_level_count`` times;
      - the longest adjacent run is at most ``max_adjacent_run``;
      - at least ``min_adjacent_changes`` of the seven adjacent pairs change level.
    """
    candidates: list[dict[str, Any]] = []
    matched_k_d_count = 0
    rejected_distinct_count = 0
    rejected_level_count = 0
    rejected_run_count = 0
    rejected_change_count = 0

    with metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"idx", "K", "D", "full_rel_l2_power", "terminal_rel_l2_power"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise KeyError(f"Missing columns {missing} in {metrics_csv}")

        for row in reader:
            k_value = int(float(row["K"]))
            d_value = float(row["D"])
            if k_value != int(target_k):
                continue
            if abs(d_value - float(target_d)) > float(d_tolerance):
                continue

            matched_k_d_count += 1
            idx_value = int(float(row["idx"]))
            if not (0 <= idx_value < len(amplitudes_all)):
                raise IndexError(
                    f"metrics idx={idx_value} is outside unseen amplitude table range "
                    f"[0, {len(amplitudes_all) - 1}]"
                )

            sample_amplitudes = np.asarray(amplitudes_all[idx_value], dtype=np.float64).reshape(-1)
            rounded = np.round(sample_amplitudes, decimals=8)
            unique_values, counts = np.unique(rounded, return_counts=True)
            distinct_count = int(len(unique_values))
            max_level_count = int(np.max(counts))
            adjacent_changes = int(np.count_nonzero(np.abs(np.diff(rounded)) > 1e-8))

            run_lengths: list[int] = []
            current_run = 1
            for i in range(1, len(rounded)):
                if abs(float(rounded[i]) - float(rounded[i - 1])) <= 1e-8:
                    current_run += 1
                else:
                    run_lengths.append(current_run)
                    current_run = 1
            run_lengths.append(current_run)
            longest_run = int(max(run_lengths))

            if distinct_count < int(min_distinct_amplitudes):
                rejected_distinct_count += 1
                continue
            if max_level_count > int(max_same_level_count):
                rejected_level_count += 1
                continue
            if longest_run > int(max_adjacent_run):
                rejected_run_count += 1
                continue
            if adjacent_changes < int(min_adjacent_changes):
                rejected_change_count += 1
                continue

            candidates.append({
                "idx": idx_value,
                "K": k_value,
                "D": d_value,
                "full_rel_l2_power": float(row["full_rel_l2_power"]),
                "terminal_rel_l2_power": float(row["terminal_rel_l2_power"]),
                "distinct_amplitude_count": distinct_count,
                "max_same_level_count": max_level_count,
                "longest_adjacent_run": longest_run,
                "adjacent_changes": adjacent_changes,
                "amplitudes_A1_to_A8": [float(x) for x in sample_amplitudes.tolist()],
                "metrics_row": row,
            })

    if not candidates:
        raise RuntimeError(
            f"No representative mixed-amplitude sample found for K={target_k}, D={target_d} "
            f"in {metrics_csv}. Matched K-D rows={matched_k_d_count}; "
            f"rejected distinct={rejected_distinct_count}, level-count={rejected_level_count}, "
            f"long-run={rejected_run_count}, adjacent-change={rejected_change_count}. "
            "Relax --min-distinct-amplitudes, --max-same-level-count, "
            "--max-adjacent-run or --min-adjacent-changes if needed."
        )

    # After the representative-input filter, retain the original best-error ranking.
    candidates.sort(
        key=lambda item: (
            float(item["full_rel_l2_power"]),
            float(item["terminal_rel_l2_power"]),
            int(item["idx"]),
        )
    )
    best = candidates[0]
    best["matched_k_d_count"] = int(matched_k_d_count)
    best["rejected_distinct_count"] = int(rejected_distinct_count)
    best["rejected_level_count"] = int(rejected_level_count)
    best["rejected_run_count"] = int(rejected_run_count)
    best["rejected_change_count"] = int(rejected_change_count)
    best["candidate_count"] = int(len(candidates))
    return best


def resolve_metrics_csv(run_dir: Path, explicit_path: Path | None) -> Path:
    """Locate the beta2 forward-evaluation metrics file without changing sample-selection logic."""
    required = {"idx", "K", "D", "full_rel_l2_power", "terminal_rel_l2_power"}

    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Specified metrics CSV does not exist: {path}")
        return path

    preferred = run_dir / "eval_beta2_full81" / "metrics_stream.csv"
    if preferred.is_file():
        return preferred

    valid_candidates: list[Path] = []
    for path in run_dir.rglob("metrics_stream.csv"):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                header = set(next(csv.reader(f)))
        except Exception:
            continue
        if required.issubset(header):
            valid_candidates.append(path.resolve())

    if not valid_candidates:
        all_found = sorted(str(p.resolve()) for p in run_dir.rglob("metrics_stream.csv"))
        details = "\n".join(all_found) if all_found else "(none found)"
        raise FileNotFoundError(
            "Could not find a beta2 metrics_stream.csv containing columns "
            f"{sorted(required)} under {run_dir}.\nAll metrics_stream.csv files found:\n{details}"
        )

    def rank(path: Path) -> tuple[int, int, str]:
        lower = str(path).lower()
        return (
            0 if "eval_beta2" in lower else 1,
            len(path.parts),
            lower,
        )

    valid_candidates.sort(key=rank)
    chosen = valid_candidates[0]
    print(f"Auto-detected metrics CSV: {chosen}")
    if len(valid_candidates) > 1:
        print("Other valid metrics CSV candidates:")
        for p in valid_candidates[1:]:
            print(f"  {p}")
    return chosen


def exact_grid(model_cfg: dict[str, Any]) -> dict[str, Any]:
    compare_t_min = float(model_cfg.get("t_min", -44.0))
    compare_t_max = float(model_cfg.get("t_max", 44.0))
    n_t = 2048
    n_z = 500
    n_slices = 81
    z_max_ld = 4.0
    half_window = max(abs(compare_t_min), abs(compare_t_max))
    tau_full = np.linspace(-half_window, half_window, n_t, endpoint=False, dtype=np.float64)
    mask = (tau_full >= compare_t_min - 1e-12) & (tau_full <= compare_t_max + 1e-12)
    return {
        "ssfm_half_window": float(half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "n_slices": int(n_slices),
        "z_max_ld": float(z_max_ld),
        "compare_t_min": compare_t_min,
        "compare_t_max": compare_t_max,
        "tau_full": tau_full,
        "time_mask": mask,
        "tau": tau_full[mask].astype(np.float32),
        "selected_steps": np.rint(np.linspace(0, n_z, n_slices)).astype(np.int64),
        "zeta": np.linspace(0.0, z_max_ld, n_slices, dtype=np.float32),
        "z_km": np.linspace(0.0, z_max_ld, n_slices, dtype=np.float32) * 5.0,
    }


def load_pinn(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(str(checkpoint), map_location=device)
    model_cfg = dict(payload.get("model_config", {}))
    state = payload.get("model_state")
    model = ConditionalPINN(**model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def save_three_formats(fig: plt.Figure, stem: Path) -> None:
    # Fixed axes positions are intentional; do not use bbox_inches="tight".
    fig.savefig(str(stem) + ".png", dpi=320, facecolor="white", edgecolor="white", transparent=False, pad_inches=0.0)
    fig.savefig(str(stem) + ".pdf", facecolor="white", edgecolor="white", transparent=False, pad_inches=0.0)
    fig.savefig(str(stem) + ".svg", facecolor="white", edgecolor="white", transparent=False, pad_inches=0.0)


def _reference_time_ticks(values: np.ndarray) -> list[float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo <= -19 and hi >= 19:
        return [-20.0, 0.0, 20.0]
    return [lo, 0.5 * (lo + hi), hi]


def draw_waterfall_panel(ax: plt.Axes, tau: np.ndarray, zeta: np.ndarray, power_ref: np.ndarray, power_pred: np.ndarray) -> None:
    ax.set_axis_off()
    display_z = np.arange(0.0, 4.0 + 1e-12, 0.5, dtype=np.float64)
    display_idx = np.asarray([int(np.argmin(np.abs(zeta - zz))) for zz in display_z], dtype=np.int64)
    time_idx = np.unique(np.rint(np.linspace(0, len(tau) - 1, min(700, len(tau)))).astype(np.int64))
    tau_d = tau[time_idx].astype(np.float64)
    p_ref = power_ref[display_idx][:, time_idx].astype(np.float64)
    p_pred = power_pred[display_idx][:, time_idx].astype(np.float64)
    peak = float(np.max(p_ref)) + 1e-300
    p_ref /= peak
    p_pred /= peak
    tmin = float(tau_d[0]); tmax = float(tau_d[-1])
    q = (tau_d - tmin) / max(tmax - tmin, 1e-12)
    time_dx, time_dy, power_scale = 0.72, 0.28, 0.68
    xmin, xmax = 0.0, 4.0
    for y in (0.0, 0.5, 1.0):
        ax.plot([xmin, xmax], [y * power_scale, y * power_scale], color=GRID_COLOR, lw=0.65, zorder=0)
    for x in display_z:
        ax.plot([x, x], [0.0, power_scale], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([xmax, xmax], [0.0, power_scale], color=GRID_COLOR, lw=0.90, zorder=6)
    final_x = xmax
    ax.plot([final_x, final_x + time_dx], [0.0, -time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([final_x, final_x + time_dx], [power_scale, power_scale - time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot(
        [final_x + time_dx, final_x + time_dx],
        [-time_dy, power_scale - time_dy],
        color="#D8DADC",
        lw=0.75,
        zorder=7,
    )
    # Complete right-side power wall: bottom, middle and top slanted grid lines.
    # The missing line in the previous versions was the p_tick=0.5 line.
    for p_tick in (0.0, 0.5, 1.0):
        y_tick = p_tick * power_scale
        ax.plot(
            [final_x, final_x + time_dx],
            [y_tick, y_tick - time_dy],
            color=GRID_COLOR,
            lw=0.65,
            zorder=6,
        )
    for x0 in display_z:
        ax.plot(x0 + time_dx * q, -time_dy * q, color=BASELINE_COLOR, lw=0.65, zorder=1)
    for row in reversed(range(len(display_z))):
        x0 = display_z[row]
        x = x0 + time_dx * q
        baseline = -time_dy * q
        ax.plot(x, baseline + power_scale * p_ref[row], color=SSFM_COLOR, lw=1.80, solid_capstyle="round", zorder=3)
        ax.plot(x, baseline + power_scale * p_pred[row], color=PINN_COLOR, lw=1.60, ls=(0, (4.0, 2.5)), zorder=4)
    ax.plot([xmin + time_dx, xmax + time_dx], [-time_dy, -time_dy], color=AXIS_COLOR, lw=0.95, zorder=5)
    for xx in display_z:
        xt = xx + time_dx
        ax.plot([xt, xt - 0.035], [-time_dy, -time_dy - 0.023], color=AXIS_COLOR, lw=0.75, zorder=5)
        ax.text(xt, -time_dy - 0.072, f"{xx:g}", ha="center", va="top", fontsize=14, fontweight="bold")
    first_x = xmin
    for tv in _reference_time_ticks(tau_d):
        qv = (tv - tmin) / max(tmax - tmin, 1e-12)
        xt = first_x + time_dx * qv
        yt = -time_dy * qv
        ax.plot([xt, xt - 0.03], [yt, yt - 0.018], color=AXIS_COLOR, lw=0.75, zorder=5)
        ax.text(xt - 0.045, yt - 0.040, f"{tv:g}", ha="right", va="top", fontsize=14, fontweight="bold")
    ax.plot([xmin, xmin], [0.0, power_scale], color=AXIS_COLOR, lw=0.95, zorder=5)
    for val in (0.0, 0.5, 1.0):
        yt = val * power_scale
        ax.plot([xmin - 0.03, xmin], [yt, yt], color=AXIS_COLOR, lw=0.75, zorder=5)
        ax.text(xmin - 0.055, yt, f"{val:g}", ha="right", va="center", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.22, xmax + time_dx + 0.20)
    ax.set_ylim(-0.44, 0.87)
    ax.text(0.52, 0.045, r"$z/L_D$", transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(first_x + time_dx * 0.24, -time_dy - 0.005, r"$t/T_0$", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.005, 0.550, "Normalized Power", transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=14, fontweight="bold")
    ax.legend(handles=[Line2D([0],[0],color=SSFM_COLOR,lw=1.9,label="SSFM"), Line2D([0],[0],color=PINN_COLOR,lw=1.7,ls=(0,(4.0,2.5)),label="PINN")], loc="upper right", frameon=False, bbox_to_anchor=(0.915,0.875), handlelength=1.8, handletextpad=0.55, prop={"family":"Times New Roman","weight":"bold","size":14})
    ax.text(0.045, 0.635, "(a)", ha="left", va="top", fontsize=14, fontweight="bold", zorder=20)


def terminal_spectral_power(field_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(field_1d, dtype=np.complex128)
    spec = np.fft.fftshift(np.fft.fft(y))
    n = len(y)
    df = 1.0 / max(n, 1)
    f = np.fft.fftshift(np.fft.fftfreq(n, d=df))
    power = np.abs(spec) ** 2
    if np.max(power) > 0:
        power /= np.max(power)
    f = f / (np.max(np.abs(f)) + 1e-300) * 2.0
    return f, power


def draw_forward_figure(out_stem: Path, tau: np.ndarray, zeta: np.ndarray, ref_map: np.ndarray, pred_map: np.ndarray, meta: dict[str, Any]) -> None:
    ref_field = ref_map[:, 0, :] + 1j * ref_map[:, 1, :]
    pred_field = pred_map[:, 0, :] + 1j * pred_map[:, 1, :]
    ref_power = np.sum(ref_map ** 2, axis=1)
    pred_power = np.sum(pred_map ** 2, axis=1)

    reference_peak = float(np.max(ref_power)) + 1e-300
    ref_power_n = ref_power / reference_peak
    pred_power_n = pred_power / reference_peak
    err_power_n = np.abs(pred_power_n - ref_power_n)

    # Fixed canvas and fixed axes positions prevent title/label overlap.
    fig = plt.figure(figsize=(15.2, 9.20), facecolor="white")

    # Panel (a) is moved slightly upward. Fixed positions are used so that
    # its labels cannot overlap the lower four panels.
    ax_a = fig.add_axes([0.055, 0.605, 0.890, 0.305])
    draw_waterfall_panel(ax_a, tau, zeta, ref_power_n, pred_power_n)

    # Lower four panels: explicit positions, no gridspec.
    left_x = 0.085
    left_w = 0.475
    cbar_x = 0.568
    cbar_w = 0.018
    right_x = 0.675
    right_w = 0.275
    # Larger vertical gaps prevent the upper x-labels from touching the
    # lower-panel titles.
    upper_y = 0.360
    lower_y = 0.060
    panel_h = 0.205

    z_min = float(np.min(zeta))
    z_max = float(np.max(zeta))
    tau_min = float(np.min(tau))
    tau_max = float(np.max(tau))
    z_ticks = np.arange(0.0, 4.0 + 1e-9, 0.5)
    time_ticks = [-20.0, 0.0, 20.0]

    # (b) PINN power evolution.
    ax_b = fig.add_axes([left_x, upper_y, left_w, panel_h])
    mesh_b = ax_b.pcolormesh(
        zeta, tau, pred_power_n.T,
        shading="auto", cmap=REFERENCE_CMAP,
        vmin=0.0, vmax=1.0, rasterized=True,
    )
    cax_b = fig.add_axes([cbar_x, upper_y, cbar_w, panel_h])
    cb_b = fig.colorbar(mesh_b, cax=cax_b)
    cb_b.set_label("Normalized Power", labelpad=3, fontsize=14, fontweight="bold")
    cb_b.set_ticks(np.linspace(0.0, 1.0, 6))
    cb_b.ax.tick_params(labelsize=14, width=0.8, length=3)
    ax_b.set_title("Pulse Evolution Generated by PINN", pad=6)
    ax_b.set_xlabel(r"$z$", labelpad=4)
    ax_b.set_ylabel(r"$t$", labelpad=4)
    ax_b.set_xlim(z_min, z_max)
    ax_b.set_ylim(tau_min, tau_max)
    ax_b.set_xticks(z_ticks)
    ax_b.set_yticks(time_ticks)
    ax_b.tick_params(direction="out", width=0.9, length=4)
    ax_b.text(0.010, 0.880, "(b)", transform=ax_b.transAxes, fontsize=14, fontweight="bold")

    # (c) Absolute normalized-power error, fixed display range [0, 1].
    ax_c = fig.add_axes([left_x, lower_y, left_w, panel_h])
    mesh_c = ax_c.pcolormesh(
        zeta, tau, err_power_n.T,
        shading="auto", cmap=REFERENCE_CMAP,
        vmin=0.0, vmax=1.0, rasterized=True,
    )
    cax_c = fig.add_axes([cbar_x, lower_y, cbar_w, panel_h])
    cb_c = fig.colorbar(mesh_c, cax=cax_c)
    cb_c.set_label("Normalized Absolute Error", labelpad=3, fontsize=14, fontweight="bold")
    cb_c.set_ticks(np.linspace(0.0, 1.0, 6))
    cb_c.ax.tick_params(labelsize=14, width=0.8, length=3)
    ax_c.set_title("Absolute Error Distribution", pad=9)
    ax_c.set_xlabel(r"$z$", labelpad=2)
    ax_c.set_ylabel(r"$t$", labelpad=4)
    ax_c.set_xlim(z_min, z_max)
    ax_c.set_ylim(tau_min, tau_max)
    ax_c.set_xticks(z_ticks)
    ax_c.set_yticks(time_ticks)
    ax_c.tick_params(direction="out", width=0.9, length=4)
    ax_c.text(0.010, 0.880, "(c)", transform=ax_c.transAxes, fontsize=14, fontweight="bold")

    # (d) Terminal time-domain result.
    ax_d = fig.add_axes([right_x, upper_y, right_w, panel_h])
    ax_d.plot(tau, ref_power_n[-1], color=SSFM_COLOR, lw=2.10, label="SSFM")
    ax_d.plot(tau, pred_power_n[-1], color=PINN_COLOR, lw=1.90, ls=(0, (5.0, 3.0)), label="PINN")
    ax_d.set_title("Time-Domain Result", pad=6)
    ax_d.set_xlabel(r"$t$", labelpad=4)
    ax_d.set_ylabel("Normalized Power", labelpad=3)
    time_margin = 0.08 * (tau_max - tau_min)
    ax_d.set_xlim(tau_min - time_margin, tau_max + time_margin)
    ax_d.set_ylim(bottom=0.0)
    ax_d.grid(True, color=GRID_COLOR, linewidth=0.70, alpha=0.65)
    ax_d.legend(
        frameon=False, loc="upper right",
        bbox_to_anchor=(1.008, 1.000), borderaxespad=0.0,
        handlelength=1.8, handletextpad=0.55,
        prop={"family": "Times New Roman", "weight": "bold", "size": 14},
    )
    ax_d.text(0.010, 0.880, "(d)", transform=ax_d.transAxes, fontsize=14, fontweight="bold")

    # (e) Terminal frequency-domain result, requested range [-0.1, 0.1].
    ax_e = fig.add_axes([right_x, lower_y, right_w, panel_h])
    f_ref, s_ref = terminal_spectral_power(ref_field[-1])
    f_pred, s_pred = terminal_spectral_power(pred_field[-1])
    freq_mask = (f_ref >= -0.1) & (f_ref <= 0.1)
    ax_e.plot(f_ref[freq_mask], s_ref[freq_mask], color=SSFM_COLOR, lw=2.10, label="SSFM")
    ax_e.plot(f_pred[freq_mask], s_pred[freq_mask], color=PINN_COLOR, lw=1.90, ls=(0, (5.0, 3.0)), label="PINN")
    ax_e.set_title("Frequency-Domain Result", pad=9)
    ax_e.set_xlabel(r"$fT_0$", labelpad=4)
    ax_e.set_ylabel("Spectral Power", labelpad=3)
    ax_e.set_xlim(-0.1, 0.1)
    ax_e.set_ylim(bottom=0.0)
    ax_e.set_xticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    ax_e.grid(True, color=GRID_COLOR, linewidth=0.70, alpha=0.65)
    ax_e.text(0.010, 0.880, "(e)", transform=ax_e.transAxes, fontsize=14, fontweight="bold")

    # Global publication font control.
    # Change FONT_SIZE here to modify every visible text element in the figure.
    FONT_NAME = "Times New Roman"
    FONT_SIZE = 14
    for text_object in fig.findobj(matplotlib.text.Text):
        text_object.set_fontfamily(FONT_NAME)
        text_object.set_fontname(FONT_NAME)
        text_object.set_fontsize(FONT_SIZE)
        text_object.set_fontweight("bold")

    save_three_formats(fig, out_stem)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(r"."))
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--target-K", type=int, dest="target_K", default=8)
    p.add_argument("--target-D", type=float, dest="target_D", default=1.1)
    p.add_argument("--d-tolerance", type=float, default=1e-6)
    p.add_argument(
        "--idx",
        type=int,
        default=-1,
        help="-1 means automatically select the best K=8, D=1.1 sample from metrics_stream.csv.",
    )
    p.add_argument("--metrics-csv", type=Path, default=None)
    p.add_argument(
        "--min-distinct-amplitudes",
        type=int,
        default=3,
        help="Minimum distinct levels among A1..A8; default 3 avoids overly simple inputs.",
    )
    p.add_argument(
        "--max-same-level-count",
        type=int,
        default=4,
        help="Maximum number of slots allowed to share one amplitude level.",
    )
    p.add_argument(
        "--max-adjacent-run",
        type=int,
        default=2,
        help="Maximum allowed length of one adjacent equal-amplitude block.",
    )
    p.add_argument(
        "--min-adjacent-changes",
        type=int,
        default=5,
        help="Minimum number of amplitude changes among the seven adjacent slot pairs.",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--chunk-size", type=int, default=131072)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    run_dir = (
        args.run_dir.expanduser().resolve()
        if args.run_dir
        else project_root
        / "MULTIPULSE_AMPLITUDE_RUNS"
        / "universal_sparse8_K1to8_beta2_D081012_highK_transfer_from_original_v2_continue80k"
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else run_dir / "paper_forward_bestK8_D1p1_visualization"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from universal_beta2_common import (
        load_beta2_checkpoint,
        make_grid,
        predict_maps_variable_d,
        run_ssfm_batch_selected_variable_d,
        safe_device,
    )

    device = safe_device(str(args.device))
    pinn_ckpt = run_dir / "sparse8_beta2_forward_pinn.pt"
    unseen_csv = run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    metrics_csv = resolve_metrics_csv(run_dir, args.metrics_csv)

    k_all, amp_all = read_sparse8_csv(unseen_csv)

    if int(args.idx) >= 0:
        idx = int(args.idx)
        selected = {
            "idx": idx,
            "K": int(args.target_K),
            "D": float(args.target_D),
            "full_rel_l2_power": None,
            "terminal_rel_l2_power": None,
            "candidate_count": None,
            "metrics_row": None,
        }
    else:
        selected = select_best_k_d_from_metrics(
            metrics_csv=metrics_csv,
            target_k=int(args.target_K),
            target_d=float(args.target_D),
            d_tolerance=float(args.d_tolerance),
            amplitudes_all=amp_all,
            min_distinct_amplitudes=int(args.min_distinct_amplitudes),
            max_adjacent_run=int(args.max_adjacent_run),
            min_adjacent_changes=int(args.min_adjacent_changes),
            max_same_level_count=int(args.max_same_level_count),
        )
        idx = int(selected["idx"])

    if not (0 <= idx < len(amp_all)):
        raise IndexError(f"idx out of range: {idx}")

    sample_a = amp_all[idx:idx + 1]
    sample_k = int(k_all[idx])
    if sample_k != int(args.target_K):
        raise RuntimeError(
            f"Selected idx={idx} is not K={args.target_K}, got K={sample_k}"
        )

    sample_d_value = float(selected["D"])
    sample_d = np.asarray([sample_d_value], dtype=np.float32)

    pinn, payload = load_beta2_checkpoint(pinn_ckpt, device)
    model_cfg = dict(payload.get("model_config", {}))
    pde = dict(payload.get("pde_params", {}))

    compare_t_min = float(model_cfg.get("t_min", -44.0))
    compare_t_max = float(model_cfg.get("t_max", 44.0))
    grid = make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=max(abs(compare_t_min), abs(compare_t_max)),
        n_t=2048,
        n_z=500,
        n_slices=81,
        compare_t_min=compare_t_min,
        compare_t_max=compare_t_max,
    )
    tau = np.asarray(grid["tau"], dtype=np.float32)
    zeta = np.asarray(grid["zeta"], dtype=np.float32)

    ref_map = run_ssfm_batch_selected_variable_d(
        sample_a,
        sample_d,
        grid,
        pde,
        device,
        bool(args.ssfm_complex64),
    )[0]
    pred_map = predict_maps_variable_d(
        pinn,
        sample_a,
        sample_d,
        tau,
        zeta,
        device,
        int(args.chunk_size),
    )[0]

    meta = {
        "selection_rule": (
            "K=8 and D=1.1; representative mixed input with at least three distinct "
            "amplitude levels, no level used more than four times, no adjacent equal-level "
            "run longer than two, and at least five adjacent changes; minimum "
            "full_rel_l2_power; tie-break by minimum terminal_rel_l2_power, then minimum idx"
        ),
        "min_distinct_amplitudes": int(args.min_distinct_amplitudes),
        "max_same_level_count": int(args.max_same_level_count),
        "max_adjacent_run": int(args.max_adjacent_run),
        "min_adjacent_changes": int(args.min_adjacent_changes),
        "matched_k_d_count": selected.get("matched_k_d_count"),
        "rejected_distinct_count": selected.get("rejected_distinct_count"),
        "rejected_level_count": selected.get("rejected_level_count"),
        "rejected_run_count": selected.get("rejected_run_count"),
        "rejected_change_count": selected.get("rejected_change_count"),
        "candidate_count": selected.get("candidate_count"),
        "idx": idx,
        "K": int(args.target_K),
        "D": sample_d_value,
        "amplitudes_A1_to_A8": [float(x) for x in sample_a[0].tolist()],
        "full_rel_l2_power": selected.get("full_rel_l2_power"),
        "terminal_rel_l2_power": selected.get("terminal_rel_l2_power"),
        "checkpoint": str(pinn_ckpt),
        "metrics_csv": str(metrics_csv),
        "unseen_csv": str(unseen_csv),
    }
    (out_dir / "selected_representative_k8_d11_forward_sample.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    draw_forward_figure(
        out_dir / "universal_beta2_forward_representativeK8_D11_paper",
        tau.astype(np.float64),
        zeta.astype(np.float64),
        np.asarray(ref_map, dtype=np.float64),
        np.asarray(pred_map, dtype=np.float64),
        meta,
    )

    print("Selection completed.")
    print(f"Selected idx: {idx}")
    print(f"Selected K: {sample_k}")
    print(f"Selected D: {sample_d_value}")
    print(f"Selected amplitudes: {[float(x) for x in sample_a[0].tolist()]}")
    rounded_selected = np.round(sample_a[0], decimals=8)
    unique_selected, counts_selected = np.unique(rounded_selected, return_counts=True)
    adjacent_changes_selected = int(np.count_nonzero(np.abs(np.diff(rounded_selected)) > 1e-8))
    longest_run_selected = 1
    current_run_selected = 1
    for i in range(1, len(rounded_selected)):
        if abs(float(rounded_selected[i]) - float(rounded_selected[i - 1])) <= 1e-8:
            current_run_selected += 1
            longest_run_selected = max(longest_run_selected, current_run_selected)
        else:
            current_run_selected = 1
    print(f"Distinct amplitude levels: {len(unique_selected)}")
    print(f"Maximum count of one level: {int(np.max(counts_selected))}")
    print(f"Adjacent amplitude changes: {adjacent_changes_selected}/7")
    print(f"Longest adjacent equal-level run: {longest_run_selected}")
    if selected.get("matched_k_d_count") is not None:
        print(f"All evaluated K-D rows: {selected['matched_k_d_count']}")
        print(f"Rejected by distinct-level rule: {selected['rejected_distinct_count']}")
        print(f"Rejected by same-level count rule: {selected['rejected_level_count']}")
        print(f"Rejected by adjacent-run rule: {selected['rejected_run_count']}")
        print(f"Rejected by adjacent-change rule: {selected['rejected_change_count']}")
        print(f"Representative mixed candidates: {selected['candidate_count']}")
    if selected.get("full_rel_l2_power") is not None:
        print(f"Full relative L2 power: {selected['full_rel_l2_power']:.12g}")
        print(f"Terminal relative L2 power: {selected['terminal_rel_l2_power']:.12g}")
    print("Output:")
    print(out_dir / "universal_beta2_forward_representativeK8_D11_paper.png")


if __name__ == "__main__":
    main()
