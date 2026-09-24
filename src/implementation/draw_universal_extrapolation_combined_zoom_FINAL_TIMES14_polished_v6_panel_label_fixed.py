# -*- coding: utf-8 -*-
"""
draw_universal_extrapolation_best_two_waterfalls_v2.py

目的
----
只生成两张瀑布图，样本自动选择为：
    全部 unseen 样本里，0--25 km 全过程 cumulative power relative L2
    (PINN_25km) 最小的那个样本。

相比上一版的改动
----------------
1. 20--25 km 图改为 0.1 L_D 步长，共 11 个位置：
      4.0, 4.1, 4.2, ..., 5.0
2. 20--25 km 图不再直接用“真实 z 值”作为绘图横向坐标，
   而是做“显示拉伸”（display stretch）：
      仍然标注真实 z/L_D 数值，
      但把 4--5 这一小段在画面里横向展开，
      否则同样的斜投影参数会让切片严重拥挤重叠。
3. 20--25 km 图把所有中间位置都标出来。

为什么前一版一直显得怪
----------------------
因为 0--25 km 图的传播距离跨度是 5 L_D，
而 20--25 km 图的跨度只有 1 L_D。
如果还沿用同样的斜投影参数（time_dx / time_dy）和
接近真实的横坐标尺度，那么：
    - 每张切片的“时间方向投影长度”相对过长
    - 相邻切片的横向间距相对过小
于是波形会在视觉上严重压在一起，看起来就“怪”。

所以这版对 zoom 图做了两件事：
    - 切片更密：0.1 L_D 步长
    - 显示更开：横向显示坐标拉伸
"""

from __future__ import annotations

# FIXED3: y-label spacing, stable (a)/(b) anchors, matched lower t/T0 projection.

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

from train_multi_pulse_pinn import ConditionalPINN
import run_universal_full81_CNN_DDNN as base


# -----------------------------------------------------------------------------
# Plot style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman:bold",
    "mathtext.it": "Times New Roman:italic:bold",
    "mathtext.bf": "Times New Roman:bold",
    "font.size": 14.0,
    "axes.titlesize": 14.0,
    "axes.labelsize": 14.0,
    "legend.fontsize": 14.0,
    "xtick.labelsize": 14.0,
    "ytick.labelsize": 14.0,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.9,
    "axes.labelcolor": "#111111",
    "xtick.color": "#111111",
    "ytick.color": "#111111",
    "text.color": "#111111",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SSFM_COLOR = "#0000FF"
PINN_COLOR = "#FF0000"
GRID_COLOR = "#D0D3D6"
BASELINE_COLOR = "#BFC3C7"
AXIS_COLOR = "#444444"

REQUIRED_COLUMNS = [
    "idx", "K",
    "PINN_20km", "PINN_21km", "PINN_22km",
    "PINN_23km", "PINN_24km", "PINN_25km",
]


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pinn-checkpoint", default="")
    p.add_argument("--metrics-csv", default="")
    p.add_argument("--search-root", default="")
    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--model-chunk-size", type=int, default=131072)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--waterfall-max-time-points", type=int, default=700)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------
def safe_device(name: str) -> torch.device:
    req = str(name).strip().lower()
    if req != "cpu" and not torch.cuda.is_available():
        print("[warning] CUDA unavailable; fallback to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(req)


def torch_load_full(path: Path, map_location: Any) -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def load_checkpoint_metadata(checkpoint: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = torch_load_full(checkpoint, "cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid checkpoint: {checkpoint}")

    model_cfg = dict(payload.get("model_config", {}))
    if int(model_cfg.get("n_pulses", -1)) != 8:
        raise RuntimeError("Expected sparse8 universal PINN checkpoint.")

    pde_raw = dict(payload.get("pde_params", {}))
    pde = {
        "beta2_norm": float(pde_raw.get("beta2_norm", 1.0)),
        "N_sq": float(pde_raw.get("N_sq", 1.0)),
        "alpha_norm": float(pde_raw.get("alpha_norm", 0.0)),
        "beta3_norm": float(pde_raw.get("beta3_norm", 0.0)),
        "has_tod": bool(pde_raw.get("has_tod", False)),
        "has_ss": bool(pde_raw.get("has_ss", False)),
        "has_irs": bool(pde_raw.get("has_irs", False)),
        "s": float(pde_raw.get("s", 0.0)),
        "ss_coef": float(pde_raw.get("ss_coef", 1.0)),
        "tau_R": float(pde_raw.get("tau_R", 0.0)),
    }
    return model_cfg, pde


def load_pinn(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch_load_full(checkpoint, device)
    model_cfg = dict(payload.get("model_config", {}))
    state = payload.get("model_state")
    if state is None:
        raise RuntimeError(f"model_state missing in {checkpoint}")
    model = ConditionalPINN(**model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_sparse8_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    k_values: List[int] = []
    amplitudes: List[List[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        required = ["K"] + [f"A{i}" for i in range(1, 9)]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"{path} missing columns: {missing}")
        for row in reader:
            k_values.append(int(float(row["K"])))
            amplitudes.append([float(row[f"A{i}"]) for i in range(1, 9)])
    return np.asarray(k_values, dtype=np.int64), np.asarray(amplitudes, dtype=np.float64)


def exact_grid(model_cfg: Mapping[str, Any], half_window: float, n_t: int) -> Dict[str, Any]:
    n_z = 625
    n_slices = 101
    z_max_ld = 5.0

    tau_full = np.linspace(-float(half_window), float(half_window), int(n_t), endpoint=False, dtype=np.float64)
    compare_t_min = float(model_cfg.get("t_min", -44.0))
    compare_t_max = float(model_cfg.get("t_max", 44.0))
    time_mask = (tau_full >= compare_t_min - 1e-12) & (tau_full <= compare_t_max + 1e-12)

    return {
        "ssfm_half_window": float(half_window),
        "n_t": int(n_t),
        "n_z": int(n_z),
        "n_slices": int(n_slices),
        "z_max_ld": float(z_max_ld),
        "compare_t_min": compare_t_min,
        "compare_t_max": compare_t_max,
        "tau_full": tau_full,
        "time_mask": time_mask,
        "tau": tau_full[time_mask],
        "selected_steps": np.rint(np.linspace(0, n_z, n_slices)).astype(np.int64),
        "zeta": np.linspace(0.0, z_max_ld, n_slices, dtype=np.float64),
        "z_km": np.linspace(0.0, z_max_ld, n_slices, dtype=np.float64) * 5.0,
    }


# -----------------------------------------------------------------------------
# Metrics discovery and best-sample selection
# -----------------------------------------------------------------------------
def csv_header(path: Path) -> List[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            return next(reader)
    except Exception:
        return []


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        count = -1
        for count, _ in enumerate(f):
            pass
    return max(count, 0)


def candidate_score(metrics_csv: Path, run_dir: Path, expected_n: int) -> Tuple[int, Dict[str, Any]]:
    score = 0
    cfg: Dict[str, Any] = {}
    text = str(metrics_csv).lower()

    if "exact" in text:
        score += 20
    if "all_unseen" in text:
        score += 30
    if "cumulative" in text:
        score += 20
    if "0to25" in text or "0_to_25" in text:
        score += 10

    cfg_path = metrics_csv.parent / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        except Exception:
            cfg = {}
        if str(cfg.get("mode", "")).lower() == "all_unseen":
            score += 100
        if int(cfg.get("n_selected", -1)) == int(expected_n):
            score += 100
        if "EXACT" in str(cfg.get("script_version", "")).upper():
            score += 80
        if str(cfg.get("run_dir", "")).lower() == str(run_dir).lower():
            score += 80
        if bool(cfg.get("includes_z0", False)):
            score += 20

    return score, cfg


def discover_exact_metrics_csv(
    run_dir: Path,
    expected_n: int,
    explicit_metrics_csv: Optional[Path],
    extra_search_root: Optional[Path],
) -> Path:
    if explicit_metrics_csv is not None:
        candidates = [explicit_metrics_csv]
    else:
        search_roots = [run_dir, run_dir.parent]
        if extra_search_root is not None:
            search_roots.insert(0, extra_search_root)
        candidates = []
        seen = set()
        for root in search_roots:
            if not root.is_dir():
                continue
            for path in root.rglob("per_sample_cumulative_metrics.csv"):
                resolved = path.resolve()
                key = str(resolved).lower()
                if key not in seen:
                    seen.add(key)
                    candidates.append(resolved)

    valid: List[Tuple[int, Path, Dict[str, Any]]] = []
    rejected: List[str] = []

    for path in candidates:
        if not path.is_file():
            rejected.append(f"{path}: missing")
            continue

        header = csv_header(path)
        if not all(col in header for col in REQUIRED_COLUMNS):
            rejected.append(f"{path}: missing exact columns")
            continue

        n_rows = count_csv_data_rows(path)
        if int(n_rows) != int(expected_n):
            rejected.append(f"{path}: {n_rows} rows, expected {expected_n}")
            continue

        score, cfg = candidate_score(path, run_dir, expected_n)
        valid.append((score, path, cfg))

    if not valid:
        details = "\n".join(rejected[:30])
        raise FileNotFoundError(
            "Could not find exact all-unseen per_sample_cumulative_metrics.csv.\n"
            f"Rejected candidates:\n{details if details else 'none'}"
        )

    valid.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    score, best_path, cfg = valid[0]
    print(f"[metrics] selected: {best_path}", flush=True)
    print(f"[metrics] score   : {score}", flush=True)
    if cfg:
        print(
            f"[metrics] config  : mode={cfg.get('mode')}, "
            f"n_selected={cfg.get('n_selected')}, "
            f"script_version={cfg.get('script_version')}",
            flush=True,
        )
    return best_path


def select_best_sample(metrics_csv: Path, expected_n: int) -> Dict[str, Any]:
    best_row: Optional[Dict[str, Any]] = None
    best_err = float("inf")
    seen = set()
    rows = 0

    with metrics_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            if idx in seen:
                raise RuntimeError(f"duplicate idx {idx} in {metrics_csv}")
            seen.add(idx)
            rows += 1

            e25 = float(row["PINN_25km"])
            if not np.isfinite(e25):
                continue

            if e25 < best_err:
                best_err = e25
                best_row = {
                    "idx": idx,
                    "K": int(float(row["K"])),
                    "PINN_20km": float(row["PINN_20km"]),
                    "PINN_21km": float(row["PINN_21km"]),
                    "PINN_22km": float(row["PINN_22km"]),
                    "PINN_23km": float(row["PINN_23km"]),
                    "PINN_24km": float(row["PINN_24km"]),
                    "PINN_25km": e25,
                }

    if rows != int(expected_n):
        raise RuntimeError(f"row count mismatch: got {rows}, expected {expected_n}")
    if len(seen) != int(expected_n):
        raise RuntimeError(f"unique idx count mismatch: got {len(seen)}, expected {expected_n}")
    if best_row is None:
        raise RuntimeError("No valid best sample found.")
    return best_row


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def normalized_power_maps(ssfm_map: np.ndarray, pinn_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p_ssfm = np.sum(np.asarray(ssfm_map, dtype=np.float64) ** 2, axis=1)
    p_pinn = np.sum(np.asarray(pinn_map, dtype=np.float64) ** 2, axis=1)
    peak = float(np.max(p_ssfm)) + 1e-300
    return p_ssfm / peak, p_pinn / peak


def nearest_indices(values: np.ndarray, targets: Sequence[float]) -> np.ndarray:
    indices = np.asarray(
        [int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(t)))) for t in targets],
        dtype=np.int64,
    )
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("Duplicated slice indices encountered.")
    return indices


def reference_time_ticks(values: np.ndarray) -> List[float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if lo <= -19.0 and hi >= 19.0:
        return [-20.0, 0.0, 20.0]
    if lo < 0.0 < hi:
        return [lo, 0.0, hi]
    return [lo, 0.5 * (lo + hi), hi]


def save_three_formats(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(str(stem) + ".png", dpi=320, bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(str(stem) + ".svg", bbox_inches="tight", pad_inches=0.02, facecolor="white", edgecolor="white", transparent=False)


def _fmt_tick_label(value: float, force_one_decimal: bool) -> str:
    if force_one_decimal:
        return f"{value:.1f}"
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value))}"
    return f"{value:g}"


def draw_waterfall_panel(
    ax: plt.Axes,
    tau: np.ndarray,
    zeta: np.ndarray,
    p_ssfm: np.ndarray,
    p_pinn: np.ndarray,
    displayed_zeta: Sequence[float],
    display_x_positions: Sequence[float],
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    x_label: str,
    show_legend: bool,
    max_time_points: int,
    time_dx: float,
    time_dy: float,
    power_scale: float,
    x_tick_fontsize: int,
    x_label_ax_x: float,
    x_label_ax_y: float,
    y_label_ax_x: float,
    y_label_ax_y: float,
    legend_y_anchor: float,
    power_axis_max: float = 1.0,
    power_ticks: Optional[Sequence[float]] = None,
    panel_label: str = "",
    panel_label_rel_x: float = 0.020,
    panel_label_rel_y: float = 0.965,
) -> Dict[str, float]:
    ax.set_axis_off()

    displayed_zeta = np.asarray(displayed_zeta, dtype=np.float64)
    display_x_positions = np.asarray(display_x_positions, dtype=np.float64)
    tick_positions = np.asarray(tick_positions, dtype=np.float64)
    tick_labels = list(tick_labels)

    if not (len(displayed_zeta) == len(display_x_positions)):
        raise RuntimeError("displayed_zeta and display_x_positions length mismatch.")
    if not (len(tick_positions) == len(tick_labels)):
        raise RuntimeError("tick_positions and tick_labels length mismatch.")

    display_indices = nearest_indices(np.asarray(zeta, dtype=float), displayed_zeta)

    if len(tau) > int(max_time_points):
        time_indices = np.unique(np.rint(np.linspace(0, len(tau) - 1, int(max_time_points))).astype(np.int64))
    else:
        time_indices = np.arange(len(tau), dtype=np.int64)

    tau_d = np.asarray(tau[time_indices], dtype=np.float64)
    p_ssfm_d = np.take(np.asarray(p_ssfm, dtype=np.float64)[display_indices], time_indices, axis=1)
    p_pinn_d = np.take(np.asarray(p_pinn, dtype=np.float64)[display_indices], time_indices, axis=1)

    t_min = float(tau_d[0])
    t_max = float(tau_d[-1])
    t_span = max(t_max - t_min, 1e-12)
    q = (tau_d - t_min) / t_span

    x_min = float(display_x_positions[0])
    x_max = float(display_x_positions[-1])

    axis_max = float(power_axis_max)
    if not np.isfinite(axis_max) or axis_max <= 0.0:
        raise ValueError("power_axis_max must be a finite positive value.")
    if power_ticks is None:
        power_ticks_arr = np.asarray([0.0, 0.5 * axis_max, axis_max], dtype=np.float64)
    else:
        power_ticks_arr = np.asarray(list(power_ticks), dtype=np.float64)
    if power_ticks_arr.ndim != 1 or len(power_ticks_arr) == 0:
        raise ValueError("power_ticks must be a non-empty one-dimensional sequence.")
    vertical_gain = float(power_scale) / axis_max

    # background grid
    for p_tick in power_ticks_arr:
        y_grid = float(p_tick) * vertical_gain
        ax.plot([x_min, x_max], [y_grid, y_grid], color=GRID_COLOR, lw=0.65, zorder=0)

    for x0 in display_x_positions:
        ax.plot([x0, x0], [0.0, power_scale], color="#D8DADC", lw=0.55, zorder=0)

    # right wall
    final_x = x_max
    ax.plot([final_x, final_x + time_dx], [0.0, -time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([final_x, final_x + time_dx], [power_scale, power_scale - time_dy], color=GRID_COLOR, lw=0.65, zorder=0)
    ax.plot([final_x + time_dx, final_x + time_dx], [-time_dy, power_scale - time_dy], color="#D8DADC", lw=0.55, zorder=0)

    for p_tick in power_ticks_arr:
        y_tick = float(p_tick) * vertical_gain
        ax.plot([final_x, final_x + time_dx], [y_tick, y_tick - time_dy], color=GRID_COLOR, lw=0.55, zorder=0)

    # baselines
    for x0 in display_x_positions:
        ax.plot(x0 + time_dx * q, -time_dy * q, color=BASELINE_COLOR, lw=0.65, zorder=1)

    # waveforms, back to front
    for row in reversed(range(len(display_x_positions))):
        x0 = float(display_x_positions[row])
        x = x0 + time_dx * q
        baseline = -time_dy * q
        ax.plot(x, baseline + vertical_gain * p_ssfm_d[row], color=SSFM_COLOR, lw=2.50, solid_capstyle="round", zorder=4)
        ax.plot(x, baseline + vertical_gain * p_pinn_d[row], color=PINN_COLOR, lw=2.20, ls=(0, (5.0, 3.0)), dash_capstyle="butt", zorder=5)

    # front z axis
    ax.plot([x_min + time_dx, x_max + time_dx], [-time_dy, -time_dy], color=AXIS_COLOR, lw=1.0, zorder=6)
    for xpos, lbl in zip(tick_positions, tick_labels):
        x_tick = float(xpos) + time_dx
        ax.plot([x_tick, x_tick - 0.040], [-time_dy, -time_dy - 0.030], color=AXIS_COLOR, lw=0.8, zorder=6)
        ax.text(x_tick, -time_dy - 0.082, lbl, ha="center", va="top", fontsize=14, fontweight="bold", color="black")

    # time axis on first slice
    first_x = x_min
    for value in reference_time_ticks(tau_d):
        clipped = min(max(float(value), t_min), t_max)
        qt = (clipped - t_min) / t_span
        x_tick = first_x + time_dx * qt
        y_tick = -time_dy * qt
        ax.plot([x_tick, x_tick - 0.035], [y_tick, y_tick - 0.020], color=AXIS_COLOR, lw=0.8, zorder=6)
        ax.text(x_tick - 0.050, y_tick - 0.045, f"{value:g}", ha="right", va="top", fontsize=14, fontweight="bold", color="black")

    # power axis
    ax.plot([x_min, x_min], [0.0, power_scale], color=AXIS_COLOR, lw=1.0, zorder=6)
    for val in power_ticks_arr:
        y_tick = float(val) * vertical_gain
        ax.plot([x_min - 0.035, x_min], [y_tick, y_tick], color=AXIS_COLOR, lw=0.8, zorder=6)
        ax.text(x_min - 0.060, y_tick, f"{float(val):g}", ha="right", va="center", fontsize=14, fontweight="bold", color="black")

    ax.set_xlim(x_min - 0.28, x_max + time_dx + 0.10)
    ax.set_ylim(-0.47, 0.91)

    # Proper math labels and bold black y-label, matching the reference style.
    ax.text(
        float(x_label_ax_x), float(x_label_ax_y), x_label,
        transform=ax.transAxes, ha="center", va="top",
        fontsize=14, fontweight="bold", color="black",
    )
    ax.text(
        first_x + time_dx * 0.24, -time_dy - 0.012,
        r"$t/T_0$",
        ha="center", va="top", fontsize=14,
        fontweight="bold", color="black",
    )

    # Keep the y-axis title at a fixed physical distance from the custom
    # power axis.  Using an offset in points avoids overlap with tick labels
    # and also prevents different panel widths from changing the visual gap.
    ax.annotate(
        "Normalized Power",
        xy=(x_min, power_scale * 0.55),
        xycoords="data",
        xytext=(-42, 0),
        textcoords="offset points",
        ha="center", va="center", rotation=90,
        fontsize=14, fontweight="bold", color="black",
        annotation_clip=False, zorder=20,
    )

    if show_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=SSFM_COLOR, lw=2.50, label="SSFM"),
                Line2D([0], [0], color=PINN_COLOR, lw=2.20, ls=(0, (5.0, 3.0)), label="PINN"),
            ],
            loc="upper right",
            bbox_to_anchor=(0.955, float(legend_y_anchor)),
            frameon=False,
            handlelength=1.8,
            handletextpad=0.55,
            prop={"family": "Times New Roman", "weight": "bold", "size": 14},
        )

    if str(panel_label).strip():
        # Place (a)/(b) close to the upper-left corner, matching the desired
        # paper style shown by the user: slightly to the right of the custom
        # power axis and visually level with the top of that axis.
        ax.annotate(
            str(panel_label),
            xy=(x_min, power_scale),
            xycoords="data",
            xytext=(6, -2),
            textcoords="offset points",
            ha="left", va="top",
            fontsize=14, fontweight="bold", color="black",
            annotation_clip=False, zorder=30,
        )

    return {
        "x_min": x_min,
        "x_max": x_max,
        "time_dx": float(time_dx),
        "time_dy": float(time_dy),
        "power_scale": float(power_scale),
        "power_axis_max": axis_max,
    }


def draw_waterfall_figure(
    out_stem: Path,
    tau: np.ndarray,
    zeta: np.ndarray,
    p_ssfm: np.ndarray,
    p_pinn: np.ndarray,
    displayed_zeta: Sequence[float],
    display_x_positions: Sequence[float],
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
    x_label: str,
    max_time_points: int,
    time_dx: float,
    time_dy: float,
    power_scale: float,
    x_tick_fontsize: int,
    x_label_ax_x: float,
    x_label_ax_y: float,
    y_label_ax_x: float,
    y_label_ax_y: float,
    legend_y_anchor: float,
    figsize: Tuple[float, float],
) -> None:
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.055, 0.090, 0.890, 0.850])
    draw_waterfall_panel(
        ax=ax,
        tau=tau,
        zeta=zeta,
        p_ssfm=p_ssfm,
        p_pinn=p_pinn,
        displayed_zeta=displayed_zeta,
        display_x_positions=display_x_positions,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        x_label=x_label,
        show_legend=True,
        max_time_points=max_time_points,
        time_dx=time_dx,
        time_dy=time_dy,
        power_scale=power_scale,
        x_tick_fontsize=x_tick_fontsize,
        x_label_ax_x=x_label_ax_x,
        x_label_ax_y=x_label_ax_y,
        y_label_ax_x=y_label_ax_x,
        y_label_ax_y=y_label_ax_y,
        legend_y_anchor=legend_y_anchor,
    )
    # Force every visible text object, including manually drawn waterfall
    # ticks, labels, and legend entries, to Times New Roman, 14 pt.
    FONT_NAME = "Times New Roman"
    FONT_SIZE = 14
    for text_object in fig.findobj(matplotlib.text.Text):
        text_object.set_fontfamily(FONT_NAME)
        text_object.set_fontname(FONT_NAME)
        text_object.set_fontsize(FONT_SIZE)
        text_object.set_fontweight("bold")
        text_object.set_color("black")

    save_three_formats(fig, out_stem)
    plt.close(fig)


def draw_combined_waterfall_figure(
    out_stem: Path,
    tau: np.ndarray,
    zeta: np.ndarray,
    p_ssfm: np.ndarray,
    p_pinn: np.ndarray,
    displayed_full: Sequence[float],
    display_x_full: Sequence[float],
    tick_pos_full: Sequence[float],
    tick_lbl_full: Sequence[str],
    displayed_zoom: Sequence[float],
    display_x_zoom: Sequence[float],
    tick_pos_zoom: Sequence[float],
    tick_lbl_zoom: Sequence[str],
    max_time_points: int,
) -> None:
    """Draw one academic composite figure containing the full and zoom waterfalls."""
    fig = plt.figure(figsize=(15.4, 6.9), facecolor="white")

    # Upper panel: main extrapolation waterfall.
    ax_top = fig.add_axes([0.062, 0.565, 0.882, 0.315])
    top_geo = draw_waterfall_panel(
        ax=ax_top,
        tau=tau,
        zeta=zeta,
        p_ssfm=p_ssfm,
        p_pinn=p_pinn,
        displayed_zeta=displayed_full,
        display_x_positions=display_x_full,
        tick_positions=tick_pos_full,
        tick_labels=tick_lbl_full,
        x_label=r"$z/L_D$",
        show_legend=True,
        max_time_points=max_time_points,
        time_dx=0.72,
        time_dy=0.28,
        power_scale=0.68,
        x_tick_fontsize=14,
        x_label_ax_x=0.510,
        x_label_ax_y=0.020,
        y_label_ax_x=0.000,
        y_label_ax_y=0.610,
        legend_y_anchor=0.845,
        power_axis_max=1.0,
        power_ticks=(0.0, 0.5, 1.0),
        panel_label="(a)",
        panel_label_rel_x=0.025,
        panel_label_rel_y=0.915,
    )

    # Lower panel: enlarged 20--25 km window.
    ax_bottom = fig.add_axes([0.062, 0.105, 0.882, 0.335])
    bottom_geo = draw_waterfall_panel(
        ax=ax_bottom,
        tau=tau,
        zeta=zeta,
        p_ssfm=p_ssfm,
        p_pinn=p_pinn,
        displayed_zeta=displayed_zoom,
        display_x_positions=display_x_zoom,
        tick_positions=tick_pos_zoom,
        tick_labels=tick_lbl_zoom,
        x_label=r"$z/L_D$",
        show_legend=False,
        max_time_points=max_time_points,
        time_dx=0.72,
        time_dy=0.28,
        power_scale=0.68,
        x_tick_fontsize=14,
        x_label_ax_x=0.510,
        x_label_ax_y=0.020,
        y_label_ax_x=0.000,
        y_label_ax_y=0.605,
        legend_y_anchor=0.845,
        power_axis_max=0.4,
        power_ticks=(0.0, 0.2, 0.4),
        panel_label="(b)",
        panel_label_rel_x=0.025,
        panel_label_rel_y=0.915,
    )

    # Subtle academic zoom connectors. They indicate that panel (b) enlarges
    # the extrapolation interval 4 <= z/L_D <= 5 in panel (a).
    connector_color = "#8EA8DC"
    left_connector = ConnectionPatch(
        xyA=(4.0 + top_geo["time_dx"], -top_geo["time_dy"]),
        coordsA=ax_top.transData,
        # Move the lower endpoint inward so the left zoom guide is shorter
        # and does not cut across the entire blank region.
        xyB=(0.280, 1.005),
        coordsB=ax_bottom.transAxes,
        arrowstyle="-",
        linewidth=0.85,
        color=connector_color,
        alpha=0.55,
        clip_on=False,
        zorder=2,
    )
    right_connector = ConnectionPatch(
        xyA=(5.0 + top_geo["time_dx"], -top_geo["time_dy"]),
        coordsA=ax_top.transData,
        xyB=(0.982, 1.005),
        coordsB=ax_bottom.transAxes,
        arrowstyle="-",
        linewidth=0.85,
        color=connector_color,
        alpha=0.55,
        clip_on=False,
        zorder=2,
    )
    fig.add_artist(left_connector)
    fig.add_artist(right_connector)

    # Force every visible text object, including manually drawn waterfall
    # ticks, labels, legends, and panel labels, to Times New Roman, 14 pt.
    FONT_NAME = "Times New Roman"
    FONT_SIZE = 14
    for text_object in fig.findobj(matplotlib.text.Text):
        text_object.set_fontfamily(FONT_NAME)
        text_object.set_fontname(FONT_NAME)
        text_object.set_fontsize(FONT_SIZE)
        text_object.set_fontweight("bold")
        text_object.set_color("black")

    save_three_formats(fig, out_stem)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = safe_device(args.device)

    checkpoint = (
        Path(args.pinn_checkpoint).expanduser().resolve()
        if str(args.pinn_checkpoint).strip()
        else run_dir / "sparse8_forward_pinn.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"PINN checkpoint not found: {checkpoint}")

    unseen_csv = run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    if not unseen_csv.is_file():
        alt = run_dir / "dataset" / "unseen_sparse8_combinations_exact_copy.csv"
        if alt.is_file():
            unseen_csv = alt
        else:
            raise FileNotFoundError(f"Unseen CSV not found: {unseen_csv}")

    unseen_k, unseen_amp = load_sparse8_csv(unseen_csv)
    n_unseen = len(unseen_amp)

    explicit_metrics_csv = (
        Path(args.metrics_csv).expanduser().resolve()
        if str(args.metrics_csv).strip()
        else None
    )
    extra_search_root = (
        Path(args.search_root).expanduser().resolve()
        if str(args.search_root).strip()
        else None
    )

    metrics_csv = discover_exact_metrics_csv(
        run_dir=run_dir,
        expected_n=n_unseen,
        explicit_metrics_csv=explicit_metrics_csv,
        extra_search_root=extra_search_root,
    )
    best = select_best_sample(metrics_csv, n_unseen)

    sample_idx = int(best["idx"])
    if int(unseen_k[sample_idx]) != int(best["K"]):
        raise RuntimeError(
            f"K mismatch between unseen CSV and metrics CSV at idx={sample_idx}: "
            f"{int(unseen_k[sample_idx])} vs {int(best['K'])}"
        )

    amplitudes = np.asarray(unseen_amp[sample_idx:sample_idx + 1], dtype=np.float32)

    model_cfg, pde = load_checkpoint_metadata(checkpoint)
    if abs(float(model_cfg.get("z_max_ld", 4.0)) - 4.0) > 1e-12:
        raise RuntimeError("This checkpoint is not the frozen 0-20 km universal PINN.")

    grid = exact_grid(model_cfg=model_cfg, half_window=float(args.ssfm_half_window), n_t=int(args.n_t))
    tau = np.asarray(grid["tau"], dtype=np.float32)
    zeta = np.asarray(grid["zeta"], dtype=np.float32)

    print("=" * 118, flush=True)
    print("Universal PINN extrapolation waterfalls v2 (best full 0-25 km sample)", flush=True)
    print(f"run_dir     : {run_dir}", flush=True)
    print(f"checkpoint  : {checkpoint}", flush=True)
    print(f"metrics_csv : {metrics_csv}", flush=True)
    print(f"best idx    : {sample_idx}", flush=True)
    print(f"K           : {int(best['K'])}", flush=True)
    print(f"A1...A8     : {[float(x) for x in amplitudes[0].tolist()]}", flush=True)
    print(
        "cumulative PINN power relative L2 (%) : "
        f"20={100.0 * best['PINN_20km']:.6f}, "
        f"21={100.0 * best['PINN_21km']:.6f}, "
        f"22={100.0 * best['PINN_22km']:.6f}, "
        f"23={100.0 * best['PINN_23km']:.6f}, "
        f"24={100.0 * best['PINN_24km']:.6f}, "
        f"25={100.0 * best['PINN_25km']:.6f}",
        flush=True,
    )
    print(f"output      : {out_dir}", flush=True)
    print("=" * 118, flush=True)

    pinn = load_pinn(checkpoint, device)
    ssfm_map = base.run_ssfm_batch_selected(amplitudes, grid, pde, device, bool(args.ssfm_complex64))[0]
    pinn_map = base.predict_ddnn_maps(pinn, amplitudes, tau, zeta, device, int(args.model_chunk_size))[0]
    p_ssfm, p_pinn = normalized_power_maps(ssfm_map, pinn_map)

    # One combined academic figure: full range + enlarged extrapolation interval.
    displayed_full = np.arange(0.0, 5.0 + 1e-12, 0.5, dtype=np.float64)
    display_x_full = displayed_full.copy()
    tick_pos_full = displayed_full.copy()
    tick_lbl_full = [_fmt_tick_label(v, False) for v in displayed_full]

    displayed_zoom = np.arange(4.0, 5.0 + 1e-12, 0.1, dtype=np.float64)
    display_x_zoom = np.linspace(4.0, 7.2, len(displayed_zoom), dtype=np.float64)
    tick_pos_zoom = display_x_zoom
    tick_lbl_zoom = [_fmt_tick_label(v, True) for v in displayed_zoom]

    draw_combined_waterfall_figure(
        out_stem=out_dir / "extrapolation_waterfall_combined",
        tau=tau,
        zeta=zeta,
        p_ssfm=p_ssfm,
        p_pinn=p_pinn,
        displayed_full=displayed_full,
        display_x_full=display_x_full,
        tick_pos_full=tick_pos_full,
        tick_lbl_full=tick_lbl_full,
        displayed_zoom=displayed_zoom,
        display_x_zoom=display_x_zoom,
        tick_pos_zoom=tick_pos_zoom,
        tick_lbl_zoom=tick_lbl_zoom,
        max_time_points=int(args.waterfall_max_time_points),
    )

    meta = {
        "selection_definition": "minimum PINN_25km over all unseen samples (full 0-25 km cumulative power relative L2)",
        "metrics_csv": str(metrics_csv),
        "sample_idx": sample_idx,
        "K": int(best["K"]),
        "amplitudes_A1_to_A8": [float(x) for x in amplitudes[0].tolist()],
        "PINN_cumulative_power_relative_L2_percent": {
            "20km": 100.0 * float(best["PINN_20km"]),
            "21km": 100.0 * float(best["PINN_21km"]),
            "22km": 100.0 * float(best["PINN_22km"]),
            "23km": 100.0 * float(best["PINN_23km"]),
            "24km": 100.0 * float(best["PINN_24km"]),
            "25km": 100.0 * float(best["PINN_25km"]),
        },
        "displayed_full_zeta": displayed_full.tolist(),
        "displayed_zoom_zeta_true": displayed_zoom.tolist(),
        "displayed_zoom_x_for_visualization": display_x_zoom.tolist(),
        "combined_figure": "extrapolation_waterfall_combined",
        "lower_panel_power_axis_max": 0.4,
        "lower_panel_power_ticks": [0.0, 0.2, 0.4],
        "shared_legend": True,
        "panel_labels_inside_axes": True,
        "font_family": "Times New Roman",
        "font_size_pt": 14,
    }
    (out_dir / "selected_best_sample.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Done.", flush=True)
    print(f"  {out_dir / 'extrapolation_waterfall_combined.png'}", flush=True)
    print(f"  {out_dir / 'extrapolation_waterfall_combined.pdf'}", flush=True)
    print(f"  {out_dir / 'extrapolation_waterfall_combined.svg'}", flush=True)


if __name__ == "__main__":
    main()
