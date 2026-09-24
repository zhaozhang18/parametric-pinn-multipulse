# -*- coding: utf-8 -*-
"""
visualize_universal_beta2_inverse_best200_K8_D11_paper_SAMESTYLE_TIMES14.py

用途：
- 完整保留第一个通用模型逆向三联图的布局、配色、坐标、图例、字号与输出格式。
- 数据源改为加入色散条件 D 的第二个通用模型逆向结果。
- 合并第二个通用模型的两组 N=100 正式逆向实验，共 200 个样本。
- 固定真实 K=8、真实 D=1.1；若存在预测 K=8 的样本，则优先仅在这些样本中选择。
- 沿用原选择标准：8 槽幅值 MAE 最小为主、SSFM 终端功率相对 L2 最小为次。
- 仅更换模型、样本数据源和 D 条件推理接口，绘图函数不作改动。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from nlse import NLSEParams
from train_multi_pulse_pinn import ConditionalPINN

TRUE_COLOR = "#0000FF"
PINN_COLOR = "#FF0000"
SSFM_RECON_COLOR = "#2CA02C"
GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#111111"
BASELINE_COLOR = "#C9CDD2"
AXIS_COLOR = "#555555"

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

BETA2_INVERSE_EXPERIMENT_DIRS = {
    "N100_Amin0p05_R4_seed2030": "inverse_unknownK_A_D_N100_Amin0p05_R4_seed2030_power",
    "N100_Amin0p20_R4_seed2030": "inverse_unknownK_A_D_N100_Amin0p20_R4_seed2030_power",
}



def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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


def parse_amplitudes(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    text = str(value).strip().strip("[]()")
    parts = [x for x in re.split(r"[;,\s]+", text) if x]
    return np.asarray([float(x) for x in parts], dtype=np.float64)


def find_first_available(row: pd.Series, candidates: Sequence[str]) -> np.ndarray:
    for c in candidates:
        if c in row.index and str(row[c]).strip() != "":
            return parse_amplitudes(row[c])
    raise KeyError(f"Missing columns among {list(candidates)}")


def field_to_power(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.iscomplexobj(arr):
        return (np.abs(arr) ** 2).astype(np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim >= 2 and arr.shape[-2] == 2:
        return np.sum(arr ** 2, axis=-2)
    return arr.astype(np.float64)


def initial_power(tau: np.ndarray, amplitudes: Sequence[float]) -> np.ndarray:
    params = NLSEParams.paper_pam4().with_multi_pulse(tuple(float(v) for v in amplitudes), level_mode="field")
    field = params.initial_pulse(np.asarray(tau, dtype=np.float64))
    return np.abs(field) ** 2


def choose_zeta_slices() -> np.ndarray:
    return np.arange(0.0, 4.0 + 1e-12, 0.5, dtype=np.float64)


def predict_pinn_grid(
    model: torch.nn.Module,
    tau: np.ndarray,
    zeta: np.ndarray,
    amplitudes: Sequence[float],
    d_ratio: float,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float32)
    zeta = np.asarray(zeta, dtype=np.float32)
    amplitudes_np = np.asarray(amplitudes, dtype=np.float32).reshape(1, -1)
    rows = []
    with torch.inference_mode():
        for z_value in zeta:
            chunks = []
            for start in range(0, len(tau), int(chunk_size)):
                end = min(len(tau), start + int(chunk_size))
                n = end - start
                z_tensor = torch.full((n, 1), float(z_value), dtype=torch.float32, device=device)
                t_tensor = torch.as_tensor(tau[start:end, None], dtype=torch.float32, device=device)
                a_tensor = torch.as_tensor(np.repeat(amplitudes_np, n, axis=0), dtype=torch.float32, device=device)
                d_tensor = torch.full((n, 1), float(d_ratio), dtype=torch.float32, device=device)
                u, v = model(z_tensor, t_tensor, a_tensor, d_tensor)
                chunks.append(u.detach().cpu().numpy().reshape(-1) + 1j * v.detach().cpu().numpy().reshape(-1))
            rows.append(np.concatenate(chunks))
    return np.asarray(rows, dtype=np.complex128)


def save_three_formats(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(str(stem) + ".png", dpi=320, facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(str(stem) + ".pdf", facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(str(stem) + ".svg", facecolor="white", edgecolor="white", transparent=False)


def draw_inverse_waterfall(ax: plt.Axes, tau: np.ndarray, zeta: np.ndarray, true_input_power: np.ndarray, recovered_input_power: np.ndarray, target_output_power: np.ndarray, pinn_power_grid: np.ndarray, ssfm_reconstructed_output_power: np.ndarray) -> None:
    ax.set_axis_off()
    time_idx = np.unique(np.rint(np.linspace(0, len(tau) - 1, min(700, len(tau)))).astype(np.int64))
    tau_d = tau[time_idx].astype(np.float64)
    display_z = choose_zeta_slices()
    peak = max(float(np.max(true_input_power)), float(np.max(recovered_input_power)), float(np.max(target_output_power)), float(np.max(pinn_power_grid)), float(np.max(ssfm_reconstructed_output_power)), 1e-12)
    q = (tau_d - float(tau_d[0])) / max(float(tau_d[-1] - tau_d[0]), 1e-12)
    time_dx, time_dy, power_scale = 0.72, 0.28, 0.68

    # frame / grid
    for y in (0.0, 0.5, 1.0):
        ax.plot([0.0, 4.0], [y * power_scale, y * power_scale], color=GRID_COLOR, lw=0.65, zorder=0)
    for x in display_z:
        ax.plot([x, x], [0.0, power_scale], color=GRID_COLOR, lw=0.65, zorder=0)
    # right wall
    ax.plot([4.0, 4.0 + time_dx], [0.0, -time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([4.0, 4.0 + time_dx], [0.5 * power_scale, 0.5 * power_scale - time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([4.0, 4.0 + time_dx], [power_scale, power_scale - time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([4.0 + time_dx, 4.0 + time_dx], [-time_dy, power_scale - time_dy], color=GRID_COLOR, lw=0.65, zorder=0)

    for x0 in display_z:
        ax.plot(x0 + time_dx * q, -time_dy * q, color=BASELINE_COLOR, lw=0.65, zorder=1)

    # z=0 input pair
    base0 = -time_dy * q
    ax.plot(0.0 + time_dx*q, base0 + power_scale*(true_input_power[time_idx]/peak), color=TRUE_COLOR, lw=1.8, zorder=4)
    ax.plot(0.0 + time_dx*q, base0 + power_scale*(recovered_input_power[time_idx]/peak), color=PINN_COLOR, lw=1.55, ls=(0,(4.0,2.5)), zorder=5)

    # intermediate PINN propagation
    for z_show in display_z[1:-1]:
        idx = int(np.argmin(np.abs(zeta - z_show)))
        ax.plot(z_show + time_dx*q, -time_dy*q + power_scale*(np.asarray(pinn_power_grid[idx, time_idx], dtype=np.float64)/peak), color=PINN_COLOR, lw=1.45, ls=(0,(4.0,2.5)), zorder=3)

    # output slice at z=4
    xx = 4.0
    ax.plot(xx + time_dx*q, -time_dy*q + power_scale*(target_output_power[time_idx]/peak), color=TRUE_COLOR, lw=1.8, zorder=4)
    ax.plot(xx + time_dx*q, -time_dy*q + power_scale*(pinn_power_grid[-1, time_idx]/peak), color=PINN_COLOR, lw=1.55, ls=(0,(4.0,2.5)), zorder=5)
    ax.plot(xx + time_dx*q, -time_dy*q + power_scale*(ssfm_reconstructed_output_power[time_idx]/peak), color=SSFM_RECON_COLOR, lw=1.55, ls=(0,(7.0,2.4)), zorder=5)

    # axes and ticks
    ax.plot([0.0, 0.0], [0.0, power_scale], color=AXIS_COLOR, lw=0.95, zorder=6)
    for val in (0.0, 0.5, 1.0):
        yt = val * power_scale
        ax.plot([-0.03, 0.0], [yt, yt], color=AXIS_COLOR, lw=0.75)
        ax.text(-0.055, yt, f"{val:g}", ha="right", va="center", fontsize=14, fontweight="bold")

    ax.plot([time_dx, 4.0 + time_dx], [-time_dy, -time_dy], color=AXIS_COLOR, lw=0.95, zorder=6)
    for xx in display_z:
        xt = xx + time_dx
        ax.plot([xt, xt-0.035], [-time_dy, -time_dy-0.023], color=AXIS_COLOR, lw=0.75)
        ax.text(xt, -time_dy-0.072, f"{xx:g}", ha="center", va="top", fontsize=14, fontweight="bold")
    for tv in (-20.0, 0.0, 20.0):
        qv = (tv - float(tau_d[0])) / max(float(tau_d[-1] - tau_d[0]), 1e-12)
        xt = time_dx*qv
        yt = -time_dy*qv
        ax.plot([xt, xt-0.03], [yt, yt-0.018], color=AXIS_COLOR, lw=0.75)
        ax.text(xt-0.045, yt-0.04, f"{tv:g}", ha="right", va="top", fontsize=14, fontweight="bold")

    ax.set_xlim(-0.22, 4.0 + time_dx + 0.20)
    ax.set_ylim(-0.44, 0.87)
    ax.text(0.52, 0.045, r"$z/L_D$", transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(time_dx*0.24, -time_dy-0.005, r"$t/T_0$", ha="center", va="top", fontsize=14, fontweight="bold")
    ax.text(0.002, 0.60, "Normalized Power", transform=ax.transAxes, ha="center", va="center", rotation=90, fontsize=14, fontweight="bold")
    ax.legend(handles=[
        Line2D([0],[0],color=TRUE_COLOR,lw=1.8,label="True input / true output"),
        Line2D([0],[0],color=PINN_COLOR,lw=1.55,ls=(0,(4.0,2.5)),label="Recovered input / PINN propagation"),
        Line2D([0],[0],color=SSFM_RECON_COLOR,lw=1.55,ls=(0,(7.0,2.4)),label="Recovered-input SSFM"),
    ], loc="upper right", frameon=False, bbox_to_anchor=(0.985,0.900), handlelength=1.8, handletextpad=0.55, prop={"family":"Times New Roman","weight":"bold","size":14})
    ax.text(0.045, 0.635, "(a)", ha="left", va="top", fontsize=14, fontweight="bold", zorder=20)


def draw_inverse_figure(out_stem: Path, tau: np.ndarray, zeta: np.ndarray, true_input_power: np.ndarray, recovered_input_power: np.ndarray, target_output_power: np.ndarray, pinn_power_grid: np.ndarray, ssfm_reconstructed_output_power: np.ndarray, meta: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(14.5, 7.8), facecolor="white")

    ax_top = fig.add_axes([0.055, 0.565, 0.890, 0.325])
    draw_inverse_waterfall(ax_top, tau, zeta, true_input_power, recovered_input_power, target_output_power, pinn_power_grid, ssfm_reconstructed_output_power)

    ax_b = fig.add_axes([0.055, 0.215, 0.400, 0.300])
    ax_c = fig.add_axes([0.505, 0.215, 0.455, 0.300])

    peak = max(float(np.max(true_input_power)), float(np.max(recovered_input_power)), float(np.max(target_output_power)), float(np.max(pinn_power_grid[-1])), float(np.max(ssfm_reconstructed_output_power)), 1e-12)

    line_true_input, = ax_b.plot(
        tau,
        true_input_power / peak,
        color=TRUE_COLOR,
        lw=1.8,
    )
    line_recovered_input, = ax_b.plot(
        tau,
        recovered_input_power / peak,
        color=PINN_COLOR,
        lw=1.55,
        ls=(0, (4.0, 2.5)),
    )
    ax_b.set_title(r"Input Slice at $z=0$", pad=8)
    ax_b.set_xlabel(r"$t/T_0$", labelpad=4)
    ax_b.set_ylabel("Normalized Power")
    ax_b.set_xlim(-50.0, 50.0)
    ax_b.set_ylim(0.0, 1.05)
    ax_b.set_yticks([0.0, 0.5, 1.0])
    ax_b.grid(True, color=GRID_COLOR, alpha=0.6)

    blank_input_handle = Line2D([], [], color="none", linestyle="None")
    ax_b.legend(
        handles=[
            line_true_input,
            line_recovered_input,
            blank_input_handle,
        ],
        labels=[
            "True input",
            "Recovered",
            "input",
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.915),
        borderaxespad=0.0,
        ncol=1,
        labelspacing=0.12,
        handlelength=1.8,
        handletextpad=0.55,
        prop={"family": "Times New Roman", "weight": "bold", "size": 14},
    )
    ax_b.text(0.01, 0.98, "(b)", transform=ax_b.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

    ax_c.plot(tau, target_output_power/peak, color=TRUE_COLOR, lw=1.8, label="True output")
    ax_c.plot(tau, pinn_power_grid[-1]/peak, color=PINN_COLOR, lw=1.55, ls=(0,(4.0,2.5)), label="Recovered-input PINN output")
    ax_c.plot(tau, ssfm_reconstructed_output_power/peak, color=SSFM_RECON_COLOR, lw=1.55, ls=(0,(7.0,2.4)), label="Recovered-input SSFM output")
    ax_c.set_xlim(-40.0, 40.0)
    ax_c.set_title(r"Output Slice at $z=4L_D$", pad=8)
    ax_c.set_xlabel(r"$t/T_0$", labelpad=4)
    ax_c.set_ylabel("Normalized Power")
    ax_c.set_ylim(0.0, 1.05)
    ax_c.set_yticks([0.0, 0.5, 1.0])
    ax_c.grid(True, color=GRID_COLOR, alpha=0.6)
    ax_c.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.015,0.915), borderaxespad=0.0, ncol=1, labelspacing=0.35, handlelength=1.8, handletextpad=0.55, prop={"family":"Times New Roman","weight":"bold","size":14})
    ax_c.text(0.01, 0.98, "(c)", transform=ax_c.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

    FONT_NAME = "Times New Roman"
    FONT_SIZE = 14
    for text_object in fig.findobj(matplotlib.text.Text):
        text_object.set_fontfamily(FONT_NAME)
        text_object.set_fontname(FONT_NAME)
        text_object.set_fontsize(FONT_SIZE)
        text_object.set_fontweight("bold")

    save_three_formats(fig, out_stem)
    plt.close(fig)



def _first_existing_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    for name in candidates:
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _read_scalar(row: pd.Series, candidates: Sequence[str], default: float = float("nan")) -> float:
    column = _first_existing_column(row.index, candidates)
    if column is None:
        return float(default)
    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else float(default)


def _infer_k_from_amplitudes(amplitudes: np.ndarray, threshold: float) -> int:
    values = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    return int(np.count_nonzero(np.abs(values) > float(threshold)))


def _infer_true_k(row: pd.Series, true_amp: np.ndarray, threshold: float) -> int:
    value = _read_scalar(row, [
        "true_K", "true_k", "K_true", "target_K", "target_k",
        "true_num_pulses", "target_num_pulses", "K",
    ])
    if np.isfinite(value):
        return int(round(value))
    return _infer_k_from_amplitudes(true_amp, threshold)


def _infer_predicted_k(row: pd.Series, pred_amp: np.ndarray, threshold: float) -> int:
    value = _read_scalar(row, [
        "predicted_K", "predicted_k", "pred_K", "pred_k", "K_pred",
        "recovered_K", "recovered_k", "pred_num_pulses",
    ])
    if np.isfinite(value):
        return int(round(value))
    return _infer_k_from_amplitudes(pred_amp, threshold)


def _amplitude_mae(row: pd.Series, true_amp: np.ndarray, pred_amp: np.ndarray) -> float:
    stored = _read_scalar(row, [
        "amplitude_8slot_mae", "amplitude_mae_8slot", "amplitude_mae_8slots",
        "amplitude_mae", "amp_mae", "mae_amplitude", "mae_8slot",
    ])
    if np.isfinite(stored):
        return float(stored)
    n = min(len(true_amp), len(pred_amp))
    if n <= 0:
        return float("inf")
    return float(np.mean(np.abs(np.asarray(true_amp[:n]) - np.asarray(pred_amp[:n]))))


def _ssfm_terminal_rel_l2(row: pd.Series) -> float:
    stored = _read_scalar(row, [
        "ssfm_terminal_power_rel_l2", "ssfm_output_power_rel_l2",
        "terminal_power_rel_l2", "repropagated_terminal_power_rel_l2",
        "ssfm_terminal_rel_l2",
    ])
    return float(stored) if np.isfinite(stored) else float("inf")


def collect_k_d_candidates(
    per_case_df: pd.DataFrame,
    target_k: int,
    target_d: float,
    d_tolerance: float,
    true_active_threshold: float,
    predicted_active_threshold: float,
) -> dict[str, Any]:
    required_columns = {"sample"}
    missing = sorted(required_columns.difference(per_case_df.columns))
    if missing:
        raise KeyError(f"per_case.csv is missing required columns: {missing}")

    candidates: list[dict[str, Any]] = []
    for _, row in per_case_df.iterrows():
        try:
            true_amp = find_first_available(row, [
                "true_A8", "true_amplitudes_8slots", "true_amplitudes",
                "target_amplitudes_8slots"
            ])
            pred_amp = find_first_available(row, [
                "pred_A8", "pred_amplitudes_8slots", "pred_amplitudes",
                "recovered_amplitudes_8slots"
            ])
        except (KeyError, ValueError, TypeError):
            continue

        true_k = _infer_true_k(row, true_amp, true_active_threshold)
        if int(true_k) != int(target_k):
            continue

        true_d = _read_scalar(row, [
            "true_D", "true_d", "D_true", "target_D", "target_d", "D"
        ])
        pred_d = _read_scalar(row, [
            "pred_D", "pred_d", "D_pred", "recovered_D", "estimated_D"
        ])
        if not np.isfinite(true_d) or not np.isfinite(pred_d):
            continue
        if abs(float(true_d) - float(target_d)) > float(d_tolerance):
            continue

        predicted_k = _infer_predicted_k(row, pred_amp, predicted_active_threshold)
        amp_mae = _amplitude_mae(row, true_amp, pred_amp)
        sample_value = pd.to_numeric(pd.Series([row["sample"]]), errors="coerce").iloc[0]
        if pd.isna(sample_value):
            continue

        candidates.append({
            "row": row,
            "experiment": str(row["experiment"]) if "experiment" in row.index else "",
            "source_per_case_csv": str(row["source_per_case_csv"]) if "source_per_case_csv" in row.index else "",
            "Amin": float(row["Amin_source"]) if "Amin_source" in row.index else float("nan"),
            "group": str(row["group"]) if "group" in row.index else "",
            "sample": int(sample_value),
            "true_amp": np.asarray(true_amp, dtype=np.float64),
            "pred_amp": np.asarray(pred_amp, dtype=np.float64),
            "true_K": int(true_k),
            "predicted_K": int(predicted_k),
            "true_D": float(true_d),
            "pred_D": float(pred_d),
            "D_abs_error": abs(float(pred_d) - float(true_d)),
            "amplitude_8slot_mae": float(amp_mae),
        })

    if not candidates:
        raise RuntimeError(
            f"No valid true-K={target_k}, true-D={target_d} candidates were found "
            f"among {len(per_case_df)} inverse rows."
        )

    correctly_classified = [c for c in candidates if c["predicted_K"] == int(target_k)]
    pool = correctly_classified if correctly_classified else candidates
    return {
        "all_rows_scanned": int(len(per_case_df)),
        "true_k_d_candidates": int(len(candidates)),
        "correct_predicted_k_candidates": int(len(correctly_classified)),
        "pool": pool,
    }


def load_formal_200_inverse_rows(
    run_dir: Path,
    explicit_csvs: Sequence[Path] | None,
    expected_total_rows: int,
) -> tuple[pd.DataFrame, list[Path], dict[str, int]]:
    """Load the two formal beta2 inverse experiments (100 + 100 rows)."""
    frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    row_counts: dict[str, int] = {}

    if explicit_csvs:
        path_specs = [(Path(p).expanduser().resolve().parent.name, Path(p).expanduser().resolve()) for p in explicit_csvs]
    else:
        path_specs = [
            (experiment, (run_dir / dirname / "per_case.csv").resolve())
            for experiment, dirname in BETA2_INVERSE_EXPERIMENT_DIRS.items()
        ]

    for experiment, path in path_specs:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing formal inverse result CSV: {path}\n"
                "The 200-sample visualization requires both N=100 beta2 inverse runs."
            )
        frame = pd.read_csv(path)
        frame = frame.copy()
        frame["experiment"] = str(experiment)
        frame["source_per_case_csv"] = str(path)
        lower = str(path).lower()
        frame["Amin_source"] = 0.20 if "amin0p20" in lower else 0.05
        frames.append(frame)
        source_paths.append(path)
        row_counts[str(experiment)] = int(len(frame))

    if not frames:
        raise RuntimeError("No inverse per_case.csv files were loaded.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    if int(expected_total_rows) > 0 and len(combined) != int(expected_total_rows):
        details = ", ".join(f"{name}={count}" for name, count in row_counts.items())
        raise RuntimeError(
            f"Expected exactly {expected_total_rows} inverse rows, but loaded {len(combined)}. "
            f"Per-experiment counts: {details}"
        )

    return combined, source_paths, row_counts


def terminal_power_relative_l2(pred_map: np.ndarray, true_map: np.ndarray) -> float:
    pred_power = np.sum(np.asarray(pred_map[-1], dtype=np.float64) ** 2, axis=0)
    true_power = np.sum(np.asarray(true_map[-1], dtype=np.float64) ** 2, axis=0)
    numerator = np.linalg.norm(pred_power - true_power)
    denominator = max(np.linalg.norm(true_power), 1e-300)
    return float(numerator / denominator)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(r"."))
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument(
        "--per-case-csv",
        type=Path,
        action="append",
        default=None,
        help="Optional repeated argument. By default the two formal N=100 beta2 per_case.csv files are combined.",
    )
    p.add_argument("--expected-total-rows", type=int, default=200)
    p.add_argument(
        "--selection-rank",
        type=int,
        default=1,
        help="Select the requested 1-based rank after applying the original K/D and error ranking rules; default=1.",
    )
    p.add_argument("--target-K", type=int, dest="target_K", default=8)
    p.add_argument("--target-D", type=float, dest="target_D", default=1.1)
    p.add_argument("--d-tolerance", type=float, default=1e-6)
    p.add_argument("--true-active-threshold", type=float, default=1e-8)
    p.add_argument("--predicted-active-threshold", type=float, default=0.1)
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
        else run_dir / "paper_inverse_best200_K8_D1p1_visualization"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from universal_beta2_common import (
        load_beta2_checkpoint,
        make_grid,
        run_ssfm_batch_selected_variable_d,
        safe_device,
    )

    device = safe_device(str(args.device))
    pinn_ckpt = run_dir / "sparse8_beta2_forward_pinn.pt"
    per_case_df, per_case_paths, experiment_row_counts = load_formal_200_inverse_rows(
        run_dir=run_dir,
        explicit_csvs=args.per_case_csv,
        expected_total_rows=int(args.expected_total_rows),
    )

    print("Loaded formal inverse experiments:")
    for experiment, count in experiment_row_counts.items():
        print(f"  {experiment}: {count} rows")
    print(f"  total: {len(per_case_df)} rows")

    pinn, payload = load_beta2_checkpoint(pinn_ckpt, device)
    model_cfg = dict(payload.get("model_config", {}))
    pde = dict(payload.get("pde_params", {}))

    candidate_info = collect_k_d_candidates(
        per_case_df=per_case_df,
        target_k=int(args.target_K),
        target_d=float(args.target_D),
        d_tolerance=float(args.d_tolerance),
        true_active_threshold=float(args.true_active_threshold),
        predicted_active_threshold=float(args.predicted_active_threshold),
    )

    compare_t_min = float(model_cfg.get("t_min", -44.0))
    compare_t_max = float(model_cfg.get("t_max", 44.0))
    ssfm_half_window = max(abs(compare_t_min), abs(compare_t_max))

    # Two-plane grid is used only to reproduce the original secondary ranking metric:
    # SSFM terminal power relative L2.
    selection_grid = make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=ssfm_half_window,
        n_t=2048,
        n_z=500,
        n_slices=2,
        compare_t_min=compare_t_min,
        compare_t_max=compare_t_max,
    )

    pool = list(candidate_info["pool"])
    for candidate in pool:
        true_amp_batch = candidate["true_amp"].reshape(1, -1).astype(np.float32)
        pred_amp_batch = candidate["pred_amp"].reshape(1, -1).astype(np.float32)
        true_d_batch = np.asarray([candidate["true_D"]], dtype=np.float32)
        pred_d_batch = np.asarray([candidate["pred_D"]], dtype=np.float32)

        true_terminal_map = run_ssfm_batch_selected_variable_d(
            true_amp_batch,
            true_d_batch,
            selection_grid,
            pde,
            device,
            bool(args.ssfm_complex64),
        )[0]
        pred_terminal_map = run_ssfm_batch_selected_variable_d(
            pred_amp_batch,
            pred_d_batch,
            selection_grid,
            pde,
            device,
            bool(args.ssfm_complex64),
        )[0]
        candidate["ssfm_terminal_power_rel_l2"] = terminal_power_relative_l2(
            pred_terminal_map,
            true_terminal_map,
        )

    pool.sort(
        key=lambda item: (
            float(item["amplitude_8slot_mae"])
            if np.isfinite(item["amplitude_8slot_mae"])
            else float("inf"),
            float(item["ssfm_terminal_power_rel_l2"])
            if np.isfinite(item["ssfm_terminal_power_rel_l2"])
            else float("inf"),
            str(item["experiment"]),
            str(item["group"]),
            int(item["sample"]),
        )
    )
    selection_rank = int(args.selection_rank)
    if selection_rank < 1:
        raise ValueError(f"--selection-rank must be >= 1, got {selection_rank}")
    if selection_rank > len(pool):
        raise RuntimeError(
            f"Requested selection rank {selection_rank}, but only {len(pool)} eligible candidates remain."
        )
    selected = pool[selection_rank - 1]

    true_amp = np.asarray(selected["true_amp"], dtype=np.float64)
    pred_amp = np.asarray(selected["pred_amp"], dtype=np.float64)
    true_d = float(selected["true_D"])
    pred_d = float(selected["pred_D"])

    grid_full = make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=ssfm_half_window,
        n_t=2048,
        n_z=500,
        n_slices=81,
        compare_t_min=compare_t_min,
        compare_t_max=compare_t_max,
    )
    tau = np.asarray(grid_full["tau"], dtype=np.float64)
    zeta_full = np.asarray(grid_full["zeta"], dtype=np.float64)

    true_full = run_ssfm_batch_selected_variable_d(
        true_amp.reshape(1, -1).astype(np.float32),
        np.asarray([true_d], dtype=np.float32),
        grid_full,
        pde,
        device,
        bool(args.ssfm_complex64),
    )[0]
    recovered_ssfm_full = run_ssfm_batch_selected_variable_d(
        pred_amp.reshape(1, -1).astype(np.float32),
        np.asarray([pred_d], dtype=np.float32),
        grid_full,
        pde,
        device,
        bool(args.ssfm_complex64),
    )[0]

    target_output_power = field_to_power(true_full[-1]).reshape(-1)
    ssfm_reconstructed_output_power = field_to_power(recovered_ssfm_full[-1]).reshape(-1)
    true_input_power = initial_power(tau, true_amp)
    recovered_input_power = initial_power(tau, pred_amp)

    zeta = choose_zeta_slices()
    pinn_field_grid = predict_pinn_grid(
        pinn,
        tau,
        zeta,
        pred_amp,
        pred_d,
        device,
        int(args.chunk_size),
    )
    pinn_power_grid = np.abs(pinn_field_grid) ** 2

    meta = {
        "selection_rule": (
            "combine two formal N=100 experiments (200 rows); true K=8 and true D=1.1; "
            "prefer predicted K=8; minimum 8-slot amplitude MAE; tie-break by minimum "
            "SSFM terminal power relative L2; select the requested 1-based ranked candidate"
        ),
        "selection_rank": int(selection_rank),
        "eligible_ranked_candidates": int(len(pool)),
        "all_rows_scanned": int(candidate_info["all_rows_scanned"]),
        "expected_total_rows": int(args.expected_total_rows),
        "experiment_row_counts": experiment_row_counts,
        "source_per_case_csvs": [str(p) for p in per_case_paths],
        "true_k_d_candidates": int(candidate_info["true_k_d_candidates"]),
        "correct_predicted_k_candidates": int(candidate_info["correct_predicted_k_candidates"]),
        "experiment": str(selected["experiment"]),
        "Amin": float(selected["Amin"]) if np.isfinite(selected["Amin"]) else None,
        "group": str(selected["group"]),
        "sample": int(selected["sample"]),
        "true_K": int(selected["true_K"]),
        "predicted_K": int(selected["predicted_K"]),
        "true_D": true_d,
        "pred_D": pred_d,
        "D_abs_error": float(selected["D_abs_error"]),
        "amplitude_8slot_mae": float(selected["amplitude_8slot_mae"]),
        "ssfm_terminal_power_rel_l2": float(selected["ssfm_terminal_power_rel_l2"]),
        "true_amplitudes_8slots": [float(v) for v in true_amp.tolist()],
        "pred_amplitudes_8slots": [float(v) for v in pred_amp.tolist()],
        "checkpoint": str(pinn_ckpt),
        "selected_source_per_case_csv": str(selected["source_per_case_csv"]),
    }

    (out_dir / f"selected_rank{selection_rank}_best200_k8_d11_inverse_sample.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    draw_inverse_figure(
        out_dir / f"universal_beta2_inverse_best200_K8_D11_rank{selection_rank}_paper",
        tau,
        zeta,
        np.asarray(true_input_power, dtype=np.float64),
        np.asarray(recovered_input_power, dtype=np.float64),
        np.asarray(target_output_power, dtype=np.float64),
        np.asarray(pinn_power_grid, dtype=np.float64),
        np.asarray(ssfm_reconstructed_output_power, dtype=np.float64),
        meta,
    )

    print("Selection completed.")
    print(f"Requested ranked candidate: {selection_rank}")
    print(f"Eligible ranked candidates: {len(pool)}")
    print(f"Inverse rows scanned: {candidate_info['all_rows_scanned']}")
    print(f"True K={args.target_K}, D={args.target_D} candidates: {candidate_info['true_k_d_candidates']}")
    print(f"Predicted K={args.target_K} candidates: {candidate_info['correct_predicted_k_candidates']}")
    print(f"Selected experiment: {selected['experiment']}")
    print(f"Selected group: {selected['group']}")
    print(f"Selected sample: {selected['sample']}")
    print(f"True D -> predicted D: {true_d:.8g} -> {pred_d:.8g}")
    print(f"Amplitude 8-slot MAE: {selected['amplitude_8slot_mae']:.12g}")
    print(f"SSFM terminal power relative L2: {selected['ssfm_terminal_power_rel_l2']:.12g}")
    print("Output:")
    print(out_dir / f"universal_beta2_inverse_best200_K8_D11_rank{selection_rank}_paper.png")


if __name__ == "__main__":
    main()
