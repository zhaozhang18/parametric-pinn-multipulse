# -*- coding: utf-8 -*-
"""
draw_dispersion_waterfall_EXACT_SOURCE_STYLE.py

只生成一张瀑布图。

绘图部分直接照搬论文可视化源代码中 panel (a) 的手动二维平行投影：
    - zeta = 0, 0.5, 1.0, ..., 4.0
    - 对应物理距离 0, 2.5, 5.0, ..., 20.0 km
    - time_dx = 0.72
    - time_dy = 0.28
    - power_scale = 0.68
    - 蓝色实线与红色虚线的样式、坐标、网格、右侧框、图例位置均保持一致

唯一替换的内容：
    - 蓝色实线：SSFM, D = 1.3
    - 红色虚线：SSFM, D = 0.7

不画 (a)，不画热图，不画其他子图。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# 与源瀑布图保持一致
BLUE_COLOR = "#0000FF"
RED_COLOR = "#FF0000"
GRID_COLOR = "#D0D0D0"
TEXT_COLOR = "#111111"

DEFAULT_RUN_DIR = (
    r"./MULTIPULSE_AMPLITUDE_RUNS"
    r"\universal_sparse8_K1to8_beta2_D081012_highK_transfer_from_original_v2_continue80k"
)


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 320,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "SimHei"],
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
    "mathtext.fontset": "stix",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=r".")
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--device", default="cuda")

    p.add_argument("--d-red", type=float, default=0.7)
    p.add_argument("--d-blue", type=float, default=1.3)

    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument(
        "--amplitudes",
        default="",
        help=(
            '可选：手动给出合法 K=8 振幅，例如 '
            '"1,0.75,1,1,0.75,1,0.75,1"。'
            "不填写时使用固定随机种子生成高振幅 K=8 输入。"
        ),
    )

    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--compare-t-min", type=float, default=-44.0)
    p.add_argument("--compare-t-max", type=float, default=44.0)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--n-slices", type=int, default=9)
    p.add_argument("--waterfall-max-time-points", type=int, default=700)
    p.add_argument("--ssfm-complex64", action="store_true")
    return p.parse_args()


def safe_device(text: str) -> torch.device:
    requested = str(text).strip().lower()
    if requested != "cpu" and not torch.cuda.is_available():
        print("[warning] CUDA 不可用，自动改用 CPU。", flush=True)
        return torch.device("cpu")
    return torch.device(requested)


def torch_load_full(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def load_checkpoint_metadata(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch_load_full(checkpoint)
    if not isinstance(payload, dict):
        raise RuntimeError(f"无效 checkpoint：{checkpoint}")
    model_cfg = dict(payload.get("model_config", {}))
    pde = dict(payload.get("pde_params", {}))
    if int(model_cfg.get("n_pulses", -1)) != 8:
        raise RuntimeError("该脚本要求 sparse-8 色散条件通用模型的 checkpoint。")
    return model_cfg, pde


def parse_amplitudes(text: str) -> np.ndarray:
    values = [
        float(item)
        for item in str(text).replace(";", ",").split(",")
        if item.strip()
    ]
    if len(values) != 8:
        raise ValueError("K=8 振幅向量必须恰好包含 8 个数。")

    arr = np.asarray(values, dtype=np.float32)
    allowed = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float32)

    for value in arr:
        if not np.any(np.isclose(value, allowed, atol=1e-6)):
            raise ValueError(
                f"非法振幅 {value:g}；允许值为 0.25、0.50、0.75、1.00。"
            )
    return arr


def random_high_amplitude_k8(seed: int) -> np.ndarray:
    """
    合法 K=8：8 个槽位全部有效。
    为使图中初始峰值尽可能明显，只从 0.75 和 1.00 中随机。
    """
    rng = np.random.default_rng(int(seed))

    for _ in range(1000):
        amplitudes = rng.choice(
            np.asarray([0.75, 1.00], dtype=np.float32),
            size=8,
            replace=True,
        ).astype(np.float32)

        # 至少 5 个峰值为 1，保证整体峰值较大
        if int(np.count_nonzero(np.isclose(amplitudes, 1.0))) >= 5:
            return amplitudes

    return np.ones(8, dtype=np.float32)


def reference_time_ticks(values: np.ndarray) -> list[float]:
    lo = float(np.min(values))
    hi = float(np.max(values))

    # 与源代码逻辑一致
    if lo <= -19.0 and hi >= 19.0:
        return [-20.0, 0.0, 20.0]
    if lo < 0.0 < hi:
        return [lo, 0.0, hi]
    return [lo, 0.5 * (lo + hi), hi]


def save_case_csv(
    path: Path,
    amplitudes: np.ndarray,
    d_red: float,
    d_blue: float,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["slot", "amplitude"])
        for index, value in enumerate(amplitudes.tolist(), start=1):
            writer.writerow([index, f"{float(value):g}"])

        writer.writerow([])
        writer.writerow(["red_curve_D", f"{float(d_red):g}"])
        writer.writerow(["blue_curve_D", f"{float(d_blue):g}"])


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if str(args.checkpoint).strip()
        else run_dir / "sparse8_beta2_forward_pinn.pt"
    )
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if str(args.out_dir).strip()
        else run_dir / "dispersion_waterfall_EXACT_SOURCE_STYLE"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from universal_beta2_common import (
            make_grid,
            run_ssfm_batch_selected_variable_d,
        )
    except Exception as exc:
        raise RuntimeError(
            "无法导入 universal_beta2_common.py。"
            "请把该脚本放在项目根目录运行。"
        ) from exc

    device = safe_device(args.device)
    model_cfg, pde = load_checkpoint_metadata(checkpoint)

    if str(args.amplitudes).strip():
        amplitudes = parse_amplitudes(args.amplitudes)
        amplitude_source = "user_specified"
    else:
        amplitudes = random_high_amplitude_k8(args.seed)
        amplitude_source = "seeded_random_high_amplitude"

    if int(args.n_slices) != 9:
        raise ValueError("为了完全对应源瀑布图，--n-slices 必须为 9。")

    grid = make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=float(args.ssfm_half_window),
        n_t=int(args.n_t),
        n_z=int(args.n_z),
        n_slices=9,
        compare_t_min=float(args.compare_t_min),
        compare_t_max=float(args.compare_t_max),
    )

    tau = np.asarray(grid["tau"], dtype=np.float64)
    zeta = np.asarray(grid["zeta"], dtype=np.float64)

    expected_zeta = np.arange(0.0, 4.0 + 1e-9, 0.5, dtype=np.float64)
    if not np.allclose(zeta, expected_zeta, atol=1e-12, rtol=0.0):
        raise RuntimeError(
            f"传播位置不符合源图：得到 {zeta.tolist()}，"
            f"应为 {expected_zeta.tolist()}。"
        )

    z_km = zeta * 5.0

    # 同一个合法 K=8 输入，只改变 D
    amplitude_batch = np.stack(
        [amplitudes, amplitudes],
        axis=0,
    ).astype(np.float32)

    # 第 0 个样本：D=0.7（红）
    # 第 1 个样本：D=1.3（蓝）
    d_batch = np.asarray(
        [float(args.d_red), float(args.d_blue)],
        dtype=np.float32,
    )

    print("=" * 100, flush=True)
    print("SSFM 色散影响瀑布图——完全使用源 panel (a) 绘图代码", flush=True)
    print(f"checkpoint : {checkpoint}", flush=True)
    print(f"output     : {out_dir}", flush=True)
    print(f"K          : 8", flush=True)
    print(f"A8         : {amplitudes.tolist()}", flush=True)
    print(f"A source   : {amplitude_source}", flush=True)
    print(f"red        : D = {float(args.d_red):g}", flush=True)
    print(f"blue       : D = {float(args.d_blue):g}", flush=True)
    print(f"z/L_D      : {zeta.tolist()}", flush=True)
    print(f"z_km       : {z_km.tolist()}", flush=True)
    print("=" * 100, flush=True)

    maps = run_ssfm_batch_selected_variable_d(
        amplitudes=amplitude_batch,
        d_ratio=d_batch,
        grid=grid,
        pde=pde,
        device=device,
        use_complex64=bool(args.ssfm_complex64),
    )
    maps = np.asarray(maps, dtype=np.float64)

    expected_prefix = (2, 9, 2)
    if maps.ndim != 4 or tuple(maps.shape[:3]) != expected_prefix:
        raise RuntimeError(
            f"SSFM 输出形状错误：{maps.shape}，应为 [2,9,2,T]。"
        )

    # 功率 [B,Z,T]
    power = np.sum(np.square(maps), axis=2)

    # 两组 SSFM 使用同一个归一化峰值
    reference_peak = float(np.max(power)) + 1e-300
    p_red_n = power[0] / reference_peak
    p_blue_n = power[1] / reference_peak

    # ==================================================================
    # 以下绘图部分直接照搬源代码 panel (a)，只替换两组数据及图例文本
    # ==================================================================

    # 原 composite 图宽 15.2；原 panel(a) 物理高度约 2.83 英寸。
    # 单图高度设为 3.95（较 3.30 增加约 19.7%），利于看出传播后段的差异。
    fig = plt.figure(figsize=(15.2, 3.95), facecolor="white")

    ax_a = fig.add_axes([0.055, 0.085, 0.890, 0.860])
    ax_a.set_axis_off()

    wf_z = zeta.copy()
    wf_idx = np.arange(len(zeta), dtype=int)

    # 与源代码一致：最多保留 700 个时间点并显式包含两端
    max_points = max(2, int(args.waterfall_max_time_points))
    if len(tau) > max_points:
        time_idx = np.unique(
            np.rint(
                np.linspace(0, len(tau) - 1, max_points)
            ).astype(int)
        )
    else:
        time_idx = np.arange(len(tau), dtype=int)

    tau_w = np.asarray(tau[time_idx], dtype=float)

    # 注意：先取二维数组，再按最后一维索引，保证形状为 [9,T]
    p_red_w = np.asarray(p_red_n[:, time_idx], dtype=float)
    p_blue_w = np.asarray(p_blue_n[:, time_idx], dtype=float)

    t_min = float(tau_w[0])
    t_max = float(tau_w[-1])
    t_span = max(t_max - t_min, 1e-12)
    q = (tau_w - t_min) / t_span

    z_min = float(wf_z[0])
    z_max = float(wf_z[-1])

    # 源代码原值，不修改
    time_dx = 0.72
    time_dy = 0.28
    power_scale = 0.68

    # Back power wall and vertical propagation grid
    wall_z_max = z_max
    for p_tick in (0.0, 0.5, 1.0):
        ax_a.plot(
            [z_min, wall_z_max],
            [p_tick * power_scale, p_tick * power_scale],
            color="#D0D3D6",
            lw=0.65,
            zorder=0,
        )

    for z_value in wf_z:
        ax_a.plot(
            [z_value, z_value],
            [0.0, power_scale],
            color="#D8DADC",
            lw=0.55,
            zorder=0,
        )

    # Final-position frame
    final_z = z_max
    ax_a.plot(
        [final_z, final_z + time_dx],
        [0.0, -time_dy],
        color="#D0D3D6",
        lw=0.65,
        zorder=0,
    )
    ax_a.plot(
        [final_z, final_z + time_dx],
        [power_scale, power_scale - time_dy],
        color="#D0D3D6",
        lw=0.65,
        zorder=0,
    )
    ax_a.plot(
        [final_z + time_dx, final_z + time_dx],
        [-time_dy, power_scale - time_dy],
        color="#D8DADC",
        lw=0.55,
        zorder=0,
    )

    for p_tick in (0.0, 0.5, 1.0):
        y_tick = p_tick * power_scale
        ax_a.plot(
            [final_z, final_z + time_dx],
            [y_tick, y_tick - time_dy],
            color="#D0D3D6",
            lw=0.55,
            zorder=0,
        )

    # Complete diagonal baseline
    for z_value in wf_z:
        ax_a.plot(
            z_value + time_dx * q,
            -time_dy * q,
            color="#BFC3C7",
            lw=0.65,
            zorder=1,
        )

    # Draw far slices first.
    # 蓝色 D=1.3 完全继承原 SSFM 蓝色实线样式。
    # 红色 D=0.7 完全继承原 PINN 红色虚线样式。
    for row, z_value in reversed(list(zip(wf_idx, wf_z))):
        x = z_value + time_dx * q
        baseline = -time_dy * q

        ax_a.plot(
            x,
            baseline + power_scale * p_blue_w[row],
            color=BLUE_COLOR,
            lw=2.50,
            solid_capstyle="round",
            zorder=4,
        )

        ax_a.plot(
            x,
            baseline + power_scale * p_red_w[row],
            color=RED_COLOR,
            lw=2.20,
            ls=(0, (5.0, 3.0)),
            dash_capstyle="butt",
            zorder=5,
        )

    # Front z/L_D axis
    ax_a.plot(
        [z_min + time_dx, z_max + time_dx],
        [-time_dy, -time_dy],
        color="#444444",
        lw=1.0,
        zorder=6,
    )

    for z_value in wf_z:
        x_tick = float(z_value) + time_dx
        ax_a.plot(
            [x_tick, x_tick - 0.045],
            [-time_dy, -time_dy - 0.030],
            color="#444444",
            lw=0.8,
            zorder=6,
        )
        ax_a.text(
            x_tick,
            -time_dy - 0.080,
            f"{z_value:g}",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    # t/T0 axis
    time_ticks = reference_time_ticks(tau_w)
    for value in time_ticks:
        value_for_position = min(
            max(float(value), t_min),
            t_max,
        )
        qt = (value_for_position - t_min) / t_span
        x_tick = z_min + time_dx * qt
        y_tick = -time_dy * qt

        ax_a.plot(
            [x_tick, x_tick - 0.035],
            [y_tick, y_tick - 0.020],
            color="#444444",
            lw=0.8,
            zorder=6,
        )
        ax_a.text(
            x_tick - 0.050,
            y_tick - 0.045,
            f"{value:g}",
            ha="right",
            va="top",
            fontsize=14,
            fontweight="bold",
        )

    # Normalized-power axis
    ax_a.plot(
        [z_min, z_min],
        [0.0, power_scale],
        color="#444444",
        lw=1.0,
        zorder=6,
    )

    for value in (0.0, 0.5, 1.0):
        y_tick = value * power_scale
        ax_a.plot(
            [z_min - 0.035, z_min],
            [y_tick, y_tick],
            color="#444444",
            lw=0.8,
            zorder=6,
        )
        ax_a.text(
            z_min - 0.060,
            y_tick,
            f"{value:g}",
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # 原 panel(a) 的坐标范围，不修改
    ax_a.set_xlim(
        z_min - 0.25,
        z_max + time_dx + 0.08,
    )
    ax_a.set_ylim(-0.45, 0.91)

    # 不保留 (a)
    ax_a.text(
        0.510,
        0.005,
        r"$z$",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    ax_a.text(
        z_min + time_dx * 0.24,
        -time_dy - 0.010,
        r"$t$",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    ax_a.text(
        0.012,
        0.620,
        "Normalized Power",
        transform=ax_a.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        rotation=90,
    )

    # 图例也继承源图顺序与位置：蓝实线在上，红虚线在下
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=BLUE_COLOR,
                lw=2.50,
                label=rf"$\tilde{{\beta}}_2 = {float(args.d_blue):g}$",
            ),
            Line2D(
                [0],
                [0],
                color=RED_COLOR,
                lw=2.20,
                ls=(0, (5.0, 3.0)),
                label=rf"$\tilde{{\beta}}_2 = {float(args.d_red):g}$",
            ),
        ],
        loc="upper right",
        bbox_to_anchor=(0.955, 0.945),
        frameon=False,
        prop={
            "family": "serif",
            "weight": "bold",
            "size": 14,
        },
    )

    png_path = out_dir / "dispersion_waterfall_EXACT_SOURCE_STYLE.png"
    pdf_path = out_dir / "dispersion_waterfall_EXACT_SOURCE_STYLE.pdf"
    svg_path = out_dir / "dispersion_waterfall_EXACT_SOURCE_STYLE.svg"

    fig.savefig(
        png_path,
        dpi=320,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    fig.savefig(
        pdf_path,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    fig.savefig(
        svg_path,
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close(fig)

    np.savez_compressed(
        out_dir / "cached_ssfm_fields.npz",
        tau=tau,
        zeta=zeta,
        z_km=z_km,
        amplitudes=amplitudes,
        d_values=d_batch,
        maps=maps,
        power=power,
        power_normalized=np.stack(
            [p_red_n, p_blue_n],
            axis=0,
        ),
    )

    metadata = {
        "K": 8,
        "amplitude_source": amplitude_source,
        "seed": int(args.seed),
        "amplitudes_A8": [
            float(value)
            for value in amplitudes.tolist()
        ],
        "red_curve": {
            "method": "SSFM",
            "D": float(args.d_red),
            "style": "red dashed",
        },
        "blue_curve": {
            "method": "SSFM",
            "D": float(args.d_blue),
            "style": "blue solid",
        },
        "z_over_LD": [
            float(value)
            for value in zeta.tolist()
        ],
        "z_km": [
            float(value)
            for value in z_km.tolist()
        ],
        "checkpoint_metadata_source": str(checkpoint),
        "output_png": str(png_path),
        "output_pdf": str(pdf_path),
        "output_svg": str(svg_path),
    }

    (out_dir / "selected_case.json").write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_case_csv(
        out_dir / "selected_case.csv",
        amplitudes,
        args.d_red,
        args.d_blue,
    )

    print()
    print("完成。", flush=True)
    print(f"PNG : {png_path}", flush=True)
    print(f"PDF : {pdf_path}", flush=True)
    print(f"SVG : {svg_path}", flush=True)
    print(f"A8  : {amplitudes.tolist()}", flush=True)


if __name__ == "__main__":
    main()
