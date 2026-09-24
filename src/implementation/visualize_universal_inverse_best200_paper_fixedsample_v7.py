# -*- coding: utf-8 -*-
"""
visualize_universal_inverse_best200_paper_fixedsample.py

用途：
- 不再重新扫描 200 个逆向样本。
- 直接使用已确认的最佳逆向样本：
    experiment = N100_Amin0p20_R4_seed2030
    sample     = 26
    method     = PINN
- 只对这个样本重放：
    1) 读取已保存的 true/pred amplitudes
    2) 用恢复输入经冻结通用 PINN 传播，得到瀑布图中间过程
    3) 用恢复输入经 SSFM 传播，得到绿色回代输出
    4) 读取真实目标输出
- 生成固定 M 论文风格的逆向三联图。
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

EXPERIMENT_DIR_MAP = {
    "N100_Amin0p05_R4_seed2030": "inverse_unknownK_N100_R4_B2_v4_seed2030",
    "N100_Amin0p20_R4_seed2030": "inverse_unknownK_N100_R4_B2_Amin0p20_v4_seed2030",
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


def predict_pinn_grid(model: torch.nn.Module, tau: np.ndarray, zeta: np.ndarray, amplitudes: Sequence[float], device: torch.device, chunk_size: int) -> np.ndarray:
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
                u, v = model(z_tensor, t_tensor, a_tensor)
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
    ], loc="upper right", frameon=False, bbox_to_anchor=(0.985,0.900), handlelength=1.8, handletextpad=0.55, prop={"family":"serif","weight":"bold","size":14})
    ax.text(0.045, 0.635, "(a)", ha="left", va="top", fontsize=14, fontweight="bold", zorder=20)


def draw_inverse_figure(out_stem: Path, tau: np.ndarray, zeta: np.ndarray, true_input_power: np.ndarray, recovered_input_power: np.ndarray, target_output_power: np.ndarray, pinn_power_grid: np.ndarray, ssfm_reconstructed_output_power: np.ndarray, meta: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(14.5, 7.8), facecolor="white")

    ax_top = fig.add_axes([0.055, 0.565, 0.890, 0.325])
    draw_inverse_waterfall(ax_top, tau, zeta, true_input_power, recovered_input_power, target_output_power, pinn_power_grid, ssfm_reconstructed_output_power)

    ax_b = fig.add_axes([0.055, 0.215, 0.400, 0.300])
    ax_c = fig.add_axes([0.505, 0.215, 0.455, 0.300])

    peak = max(float(np.max(true_input_power)), float(np.max(recovered_input_power)), float(np.max(target_output_power)), float(np.max(pinn_power_grid[-1])), float(np.max(ssfm_reconstructed_output_power)), 1e-12)

    ax_b.plot(tau, true_input_power/peak, color=TRUE_COLOR, lw=1.8, label="True input")
    ax_b.plot(tau, recovered_input_power/peak, color=PINN_COLOR, lw=1.55, ls=(0,(4.0,2.5)), label="Recovered input")
    ax_b.set_title(r"Input Slice at $z=0$", pad=8)
    ax_b.set_xlabel(r"$t/T_0$", labelpad=4)
    ax_b.set_ylabel("Normalized Power")
    ax_b.grid(True, color=GRID_COLOR, alpha=0.6)
    ax_b.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.985,0.985), handlelength=1.8, handletextpad=0.55, prop={"family":"serif","weight":"bold","size":14})
    ax_b.text(0.01, 0.98, "(b)", transform=ax_b.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")

    ax_c.plot(tau, target_output_power/peak, color=TRUE_COLOR, lw=1.8, label="True output")
    ax_c.plot(tau, pinn_power_grid[-1]/peak, color=PINN_COLOR, lw=1.55, ls=(0,(4.0,2.5)), label="Recovered-input PINN output")
    ax_c.plot(tau, ssfm_reconstructed_output_power/peak, color=SSFM_RECON_COLOR, lw=1.55, ls=(0,(7.0,2.4)), label="Recovered-input SSFM output")
    tau_min = float(np.min(tau))
    tau_max = float(np.max(tau))
    symmetric_limit = 1.22 * max(abs(tau_min), abs(tau_max))
    ax_c.set_xlim(-symmetric_limit, symmetric_limit)
    ax_c.set_title(r"Output Slice at $z=4L_D$", pad=8)
    ax_c.set_xlabel(r"$t/T_0$", labelpad=4)
    ax_c.set_ylabel("Normalized Power")
    ax_c.grid(True, color=GRID_COLOR, alpha=0.6)
    ax_c.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.995,0.995), borderaxespad=0.0, handlelength=1.8, handletextpad=0.55, prop={"family":"serif","weight":"bold","size":14})
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(r"."))
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--compare-dir-name", type=str, default="exact_two_PINN_unknownK_sets_CNN_DDNN_FINAL")
    p.add_argument("--experiment", type=str, default="N100_Amin0p20_R4_seed2030")
    p.add_argument("--sample", type=int, default=26)
    p.add_argument("--method", type=str, default="PINN")
    p.add_argument("--Amin", type=float, default=0.2)
    p.add_argument("--true-K", type=int, dest="true_K", default=4)
    p.add_argument("--predicted-K", type=int, dest="predicted_K", default=4)
    p.add_argument("--amplitude-mae", type=float, default=0.0009164214134216309)
    p.add_argument("--ssfm-terminal-power-rel-l2", type=float, dest="ssfm_terminal_power_rel_l2", default=0.005188286678397255)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--ssfm-batch-size", type=int, default=1)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--chunk-size", type=int, default=131072)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else project_root / "MULTIPULSE_AMPLITUDE_RUNS" / "universal_sparse8_K1to8_highK_try1"
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else run_dir / "paper_inverse_best200_visualization_fixedsample"
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    compare = load_module("compare_universal_inverse_exact_two_PINN_sets_FINAL_FIX2_fixedsample", project_root / "compare_universal_inverse_exact_two_PINN_sets_FINAL_FIX2.py")
    base = load_module("run_universal_full81_CNN_DDNN_fixedsample_inverse", project_root / "run_universal_full81_CNN_DDNN.py")
    device = base.safe_device(str(args.device))
    pinn_ckpt = run_dir / "sparse8_forward_pinn.pt"
    comparison_root = run_dir / str(args.compare_dir_name)
    combined_path = comparison_root / "combined_per_sample_results.csv"
    if not combined_path.is_file():
        raise FileNotFoundError(f"Missing combined result CSV: {combined_path}")
    combined_df = pd.read_csv(combined_path)
    rows = combined_df[(combined_df["experiment"].astype(str) == str(args.experiment)) & (combined_df["method"].astype(str) == str(args.method)) & (combined_df["sample"].astype(float).astype(int) == int(args.sample))]
    if rows.empty:
        raise RuntimeError(f"No row found for experiment={args.experiment}, method={args.method}, sample={args.sample}")
    row = rows.iloc[0]
    true_amp = find_first_available(row, ["true_amplitudes_8slots", "true_amplitudes", "target_amplitudes_8slots"])
    pred_amp = find_first_available(row, ["pred_amplitudes_8slots", "pred_amplitudes", "recovered_amplitudes_8slots"])
    exp_dir = run_dir / EXPERIMENT_DIR_MAP[str(args.experiment)]
    exp = compare.load_exact_experiment(exp_dir)
    compare.validate_formal_config(exp)
    model_cfg, pde = base.load_pinn_checkpoint_metadata(pinn_ckpt)
    grid_full = compare.build_full_endpoint_grid(base, model_cfg, exp["cfg"])
    i = int(args.sample)
    tau = np.asarray(exp["tau_input"], dtype=np.float64)
    target_output_field_1d = exp["target_field"][i]
    back_full = compare.generate_ssfm_terminal_full(base, pred_amp.reshape(1, -1).astype(np.float32), grid_full, pde, device, bool(args.ssfm_complex64), int(args.ssfm_batch_size))
    back_saved_tau = compare.interpolate_field(back_full, np.asarray(grid_full["tau"], dtype=np.float32), exp["tau_input"])[0]
    target_output_power = field_to_power(target_output_field_1d).reshape(-1)
    ssfm_reconstructed_output_power = field_to_power(back_saved_tau).reshape(-1)
    true_input_power = initial_power(tau, true_amp)
    recovered_input_power = initial_power(tau, pred_amp)
    zeta = choose_zeta_slices()
    pinn = load_pinn(pinn_ckpt, device)
    pinn_field_grid = predict_pinn_grid(pinn, tau, zeta, pred_amp, device, int(args.chunk_size))
    pinn_power_grid = np.abs(pinn_field_grid) ** 2
    meta = {
        "experiment": str(args.experiment),
        "sample": int(args.sample),
        "method": str(args.method),
        "Amin": float(args.Amin),
        "true_K": int(args.true_K),
        "predicted_K": int(args.predicted_K),
        "amplitude_8slot_mae": float(args.amplitude_mae),
        "ssfm_terminal_power_rel_l2": float(args.ssfm_terminal_power_rel_l2),
        "true_amplitudes_8slots": [float(v) for v in true_amp.tolist()],
        "pred_amplitudes_8slots": [float(v) for v in pred_amp.tolist()],
        "combined_csv": str(combined_path),
    }
    (out_dir / "selected_best_inverse_sample.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_inverse_figure(out_dir / "universal_inverse_best200_paper", tau, zeta, np.asarray(true_input_power, dtype=np.float64), np.asarray(recovered_input_power, dtype=np.float64), np.asarray(target_output_power, dtype=np.float64), np.asarray(pinn_power_grid, dtype=np.float64), np.asarray(ssfm_reconstructed_output_power, dtype=np.float64), meta)
    print("Done.")
    print(out_dir / "universal_inverse_best200_paper.png")


if __name__ == "__main__":
    main()
