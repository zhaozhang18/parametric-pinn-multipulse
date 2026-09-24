# -*- coding: utf-8 -*-
"""
visualize_universal_forward_bestK8_paper_fixedsample.py

用途：
- 不再重新筛选/评估所有 K=8 样本。
- 直接使用已确认的最佳 K=8 正向样本：idx=21845。
- 只对这一个样本重新运行一次 SSFM 和冻结通用 PINN，
  生成固定 M 论文风格的正向综合可视化图。

默认最佳样本（已由 metrics_stream.csv 确认）：
  idx = 21845
  K   = 8
  A   = [0.25, 0.25, 0.75, 0.5, 0.75, 0.5, 0.75, 0.25]
  full_rel_l2_power     = 0.023460267275296903
  terminal_rel_l2_power = 0.04189491842888051
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
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "font.weight": "bold",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "mathtext.fontset": "stix",
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
    ax.legend(handles=[Line2D([0],[0],color=SSFM_COLOR,lw=1.9,label="SSFM"), Line2D([0],[0],color=PINN_COLOR,lw=1.7,ls=(0,(4.0,2.5)),label="PINN")], loc="upper right", frameon=False, bbox_to_anchor=(0.915,0.875), handlelength=1.8, handletextpad=0.55, prop={"family":"serif","weight":"bold","size":14})
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
    ax_b.set_xlabel(r"$z/L_D$", labelpad=4)
    ax_b.set_ylabel(r"$t/T_0$", labelpad=4)
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
    ax_c.set_xlabel(r"$z/L_D$", labelpad=2)
    ax_c.set_ylabel(r"$t/T_0$", labelpad=4)
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
    ax_d.set_xlabel(r"$t/T_0$", labelpad=4)
    ax_d.set_ylabel("Normalized Power", labelpad=3)
    time_margin = 0.08 * (tau_max - tau_min)
    ax_d.set_xlim(tau_min - time_margin, tau_max + time_margin)
    ax_d.set_ylim(bottom=0.0)
    ax_d.grid(True, color=GRID_COLOR, linewidth=0.70, alpha=0.65)
    ax_d.legend(
        frameon=False, loc="upper right",
        bbox_to_anchor=(1.008, 1.000), borderaxespad=0.0,
        handlelength=1.8, handletextpad=0.55,
        prop={"family": "serif", "weight": "bold", "size": 14},
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
    ax_e.set_xlabel(r"$f/f_0$", labelpad=4)
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
    p.add_argument("--idx", type=int, default=21845)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ssfm-complex64", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else project_root / "MULTIPULSE_AMPLITUDE_RUNS" / "universal_sparse8_K1to8_highK_try1"
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else run_dir / "paper_forward_bestK8_visualization_fixedsample"
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    base = load_module("run_universal_full81_CNN_DDNN_fixedsample_forward", project_root / "run_universal_full81_CNN_DDNN.py")
    device = base.safe_device(str(args.device))
    pinn_ckpt = run_dir / "sparse8_forward_pinn.pt"
    unseen_csv = run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    metrics_csv = run_dir / "universal_PINN_full81" / "eval_full81_all_unseen" / "metrics_stream.csv"
    k_all, amp_all = read_sparse8_csv(unseen_csv)
    idx = int(args.idx)
    if not (0 <= idx < len(amp_all)):
        raise IndexError(f"idx out of range: {idx}")
    sample_a = amp_all[idx:idx+1]
    sample_k = int(k_all[idx])
    if sample_k != 8:
        raise RuntimeError(f"Selected idx={idx} is not K=8, got K={sample_k}")
    meta = {
        "idx": idx,
        "K": 8,
        "amplitudes_A1_to_A8": [float(x) for x in sample_a[0].tolist()],
        "full_rel_l2_power": 0.023460267275296903,
        "terminal_rel_l2_power": 0.04189491842888051,
        "metrics_csv": str(metrics_csv),
    }
    model_cfg, pde = base.load_pinn_checkpoint_metadata(pinn_ckpt)
    grid = exact_grid(model_cfg)
    tau = np.asarray(grid["tau"], dtype=np.float32)
    zeta = np.asarray(grid["zeta"], dtype=np.float32)
    pinn = load_pinn(pinn_ckpt, device)
    ref_map = base.run_ssfm_batch_selected(sample_a, grid, pde, device, bool(args.ssfm_complex64))[0]
    pred_map = base.predict_ddnn_maps(pinn, sample_a, tau, zeta, device, 131072)[0]
    (out_dir / "selected_best_k8_forward_sample.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_forward_figure(out_dir / "universal_forward_bestK8_paper", tau.astype(np.float64), zeta.astype(np.float64), np.asarray(ref_map, dtype=np.float64), np.asarray(pred_map, dtype=np.float64), meta)
    print("Done.")
    print(out_dir / "universal_forward_bestK8_paper.png")


if __name__ == "__main__":
    main()
