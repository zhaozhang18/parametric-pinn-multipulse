# -*- coding: utf-8 -*-
"""Summarize all inverse strategies for one fixed-M run.

V6 reporting convention
-----------------------
The comparison is intended to measure *problem-solving time* only.

Included in ``solution_time_sec``
    Frozen-forward inverse optimization or exhaustive search.

Excluded from runtime comparison
    1. SSFM generation of target output waveforms (question preparation).
    2. SSFM reconstruction after inversion (post-solution verification).

SSFM reconstruction error is still reported for the random-continuous method,
because it is a useful physical validation metric; only its runtime is excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create inverse summary using solution time only.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fget(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def as_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def first_existing(inverse_dir: Path, relative_paths: Iterable[str]) -> Path:
    candidates = [inverse_dir / p for p in relative_paths]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def base_row(
    method_id: str,
    method_name: str,
    target_type: str,
    inverse_information: str,
    correctness_definition: str,
) -> dict[str, Any]:
    return {
        "method_id": method_id,
        "method_name": method_name,
        "target_amplitude_type": target_type,
        "inverse_information": inverse_information,
        "n_samples": np.nan,
        "restarts_per_sample": np.nan,
        "correctness_definition": correctness_definition,
        "primary_accuracy": np.nan,
        "per_pulse_accuracy": np.nan,
        "sample_all_pulses_within_0p01_accuracy": np.nan,
        "sample_all_pulses_within_0p025_accuracy": np.nan,
        "sample_all_pulses_within_0p05_accuracy": np.nan,
        "amplitude_mae_mean": np.nan,
        "amplitude_rmse_mean": np.nan,
        "frozen_forward_terminal_rel_l2_mean": np.nan,
        "ssfm_reconstruction_rel_l2_mean": np.nan,
        "best_epoch_p50": np.nan,
        "best_epoch_p90": np.nan,
        "best_epoch_max": np.nan,
        "solution_time_sec": np.nan,
        "solution_time_per_sample_sec": np.nan,
        "runtime_definition": "仅反演优化/搜索耗时；不含目标SSFM生成和SSFM重构验证",
        "metric_note": "",
        "summary_path": "",
    }


def add_solution_time(row: dict[str, Any], seconds: Any) -> None:
    sec = as_float(seconds)
    row["solution_time_sec"] = sec
    n = as_float(row.get("n_samples"))
    row["solution_time_per_sample_sec"] = sec / n if (not np.isnan(sec) and not np.isnan(n) and n > 0) else np.nan


def collect_method_rows(inverse_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    # 1) PAM4 output targets, continuous blind inversion in [0,1]^M.
    folder = first_existing(inverse_dir, [
        "01_pam4_target_continuous_blind_0to1",
        "01_continuous_0to1_amplitude4",
    ])
    p = folder / "summary.json"
    if p.exists():
        s = load_json(p)
        r = base_row(
            "pam4_target_continuous_blind",
            "PAM4四档目标 → [0,1]连续盲反演（事后四档判定）",
            "真实振幅来自{0.25,0.5,0.75,1}，目标输出由SSFM生成",
            "优化器只知道每个振幅∈[0,1]；优化中不知道PAM4档位",
            "连续预测结束后映射到最近PAM4档位，统计整样本完全正确率",
        )
        r.update({
            "n_samples": s.get("n_samples"),
            "restarts_per_sample": s.get("restarts_per_sample"),
            "primary_accuracy": s.get("amplitude4_exact_match_accuracy_after_rounding"),
            "per_pulse_accuracy": s.get("amplitude4_per_pulse_accuracy_after_rounding"),
            "amplitude_mae_mean": s.get("mean_amplitude_mae"),
            "amplitude_rmse_mean": s.get("mean_amplitude_rmse"),
            "frozen_forward_terminal_rel_l2_mean": s.get("terminal_rel_l2_mean"),
            "best_epoch_p50": s.get("best_epoch_p50"),
            "best_epoch_p90": s.get("best_epoch_p90"),
            "best_epoch_max": s.get("best_epoch_max"),
            "metric_note": "PAM4映射只用于反演结束后的正确率评分；振幅MAE使用映射前的连续预测值。",
            "summary_path": str(p),
        })
        add_solution_time(r, s.get("inverse_core_elapsed_sec", s.get("elapsed_sec_total")))
        rows.append(r)
    else:
        missing.append({"method_id": "pam4_target_continuous_blind", "expected": str(p)})

    # 2) PAM4 exhaustive enumeration.
    folder = first_existing(inverse_dir, [
        "02_pam4_exhaustive_enumeration",
        "02_enum_amplitude4",
    ])
    p = folder / "summary.json"
    if p.exists():
        s = load_json(p)
        terminal_rel = np.sqrt(max(0.0, as_float(s.get("terminal_loss_mean"), 0.0)))
        r = base_row(
            "pam4_exhaustive_enumeration",
            "PAM4四档穷举反演（4^M候选）",
            "真实振幅来自{0.25,0.5,0.75,1}，目标输出由SSFM生成",
            "已知四档集合，枚举全部4^M个候选并选择终端误差最小者",
            "预测组合与真实组合完全相同",
        )
        r.update({
            "n_samples": s.get("n_samples"),
            "restarts_per_sample": 0,
            "primary_accuracy": s.get("exact_match_accuracy"),
            "per_pulse_accuracy": s.get("per_pulse_accuracy_mean"),
            "amplitude_mae_mean": s.get("amplitude_mae_mean"),
            "amplitude_rmse_mean": s.get("amplitude_rmse_mean"),
            "frozen_forward_terminal_rel_l2_mean": terminal_rel,
            "metric_note": "无梯度迭代、无restart；耗时为候选前向计算与最优候选搜索时间。",
            "summary_path": str(p),
        })
        add_solution_time(r, s.get("inverse_core_elapsed_sec", s.get("elapsed_sec_total")))
        rows.append(r)
    else:
        missing.append({"method_id": "pam4_exhaustive_enumeration", "expected": str(p)})

    # 3/4) 10-level and 20-level targets, each with two optimizers.
    level_specs = [
        (
            "levels10",
            "10档",
            ["03_levels10_two_optimizers", "03_fine10"],
        ),
        (
            "levels20",
            "20档",
            ["04_levels20_two_optimizers", "04_fine20"],
        ),
    ]
    for prefix, level_name, folder_candidates in level_specs:
        folder = first_existing(inverse_dir, folder_candidates)
        p = folder / "summary.json"
        if not p.exists():
            missing.append({"method_id": prefix, "expected": str(p)})
            continue
        s = load_json(p)
        for mode, method_name, inverse_info, note in [
            (
                "discrete",
                f"{level_name}目标 → 已知档位离散Logits优化",
                f"优化器知道{level_name}档位集合，直接在离散候选上优化",
                "振幅MAE按最终离散预测计算。",
            ),
            (
                "continuous_round",
                f"{level_name}目标 → 连续优化后投影到最近档位",
                "先在连续区间内优化；结束后才投影到最近目标档位",
                "正确率与离散MAE按投影后结果计算；冻结正向终端误差对应连续优化结果。",
            ),
        ]:
            ms = s.get(mode)
            if not isinstance(ms, dict):
                alt = folder / f"summary_{mode}.json"
                ms = load_json(alt) if alt.exists() else None
            if not isinstance(ms, dict):
                missing.append({"method_id": f"{prefix}_{mode}", "expected": str(folder / f"summary_{mode}.json")})
                continue
            r = base_row(
                f"{prefix}_{mode}",
                method_name,
                f"真实振幅从{level_name}集合中随机选择，目标输出由SSFM生成",
                inverse_info,
                "预测档位与真实档位完全一致的样本比例",
            )
            r.update({
                "n_samples": s.get("n_samples"),
                "restarts_per_sample": fget(s, "args", "restarts"),
                "primary_accuracy": ms.get("exact_match_accuracy"),
                "per_pulse_accuracy": ms.get("per_pulse_accuracy_mean"),
                "amplitude_mae_mean": ms.get("amplitude_mae_mean_after_rounding_or_discrete"),
                "amplitude_rmse_mean": ms.get("amplitude_rmse_mean_after_rounding_or_discrete"),
                "frozen_forward_terminal_rel_l2_mean": np.sqrt(max(0.0, as_float(ms.get("terminal_loss_mean"), 0.0))),
                "best_epoch_p50": ms.get("best_epoch_p50"),
                "best_epoch_p90": ms.get("best_epoch_p90"),
                "best_epoch_max": ms.get("best_epoch_max"),
                "metric_note": note,
                "summary_path": str(p),
            })
            add_solution_time(r, ms.get("inverse_core_elapsed_sec", ms.get("elapsed_sec")))
            rows.append(r)

    # 5) Random continuous truth.
    folder = first_existing(inverse_dir, [
        "05_random_continuous_target_blind_0to1",
        "05_random_continuous_0to1_ssfm",
    ])
    p = folder / "summary.json"
    if p.exists():
        s = load_json(p)
        r = base_row(
            "random_continuous_target_blind",
            "连续随机目标U(0,1) → [0,1]连续盲反演",
            "每个真实振幅独立采样自Uniform(0,1)，目标输出由SSFM生成",
            "优化器只知道每个振幅∈[0,1]，无离散档位",
            "连续值不使用精确相等；主正确率定义为整样本所有振幅误差≤0.05",
        )
        r.update({
            "n_samples": s.get("n_samples"),
            "restarts_per_sample": s.get("restarts_per_sample"),
            "primary_accuracy": s.get("sample_all_pulses_within_0p05_accuracy"),
            "per_pulse_accuracy": s.get("per_pulse_within_0p05_accuracy"),
            "sample_all_pulses_within_0p01_accuracy": s.get("sample_all_pulses_within_0p01_accuracy"),
            "sample_all_pulses_within_0p025_accuracy": s.get("sample_all_pulses_within_0p025_accuracy"),
            "sample_all_pulses_within_0p05_accuracy": s.get("sample_all_pulses_within_0p05_accuracy"),
            "amplitude_mae_mean": s.get("amplitude_mae_mean"),
            "amplitude_rmse_mean": s.get("amplitude_rmse_mean"),
            "frozen_forward_terminal_rel_l2_mean": s.get("best_restart_forward_inverse_rel_l2_mean"),
            "ssfm_reconstruction_rel_l2_mean": s.get("ssfm_reconstruction_output_rel_l2_mean"),
            "best_epoch_p50": s.get("best_epoch_p50_all_restarts"),
            "best_epoch_p90": s.get("best_epoch_p90_all_restarts"),
            "best_epoch_max": s.get("best_epoch_max_all_restarts"),
            "metric_note": "SSFM重构误差用于物理验证，但SSFM重构耗时不计入解题耗时。",
            "summary_path": str(p),
        })
        add_solution_time(r, s.get("inverse_core_elapsed_sec", s.get("optimization_elapsed_sec")))
        rows.append(r)
    else:
        missing.append({"method_id": "random_continuous_target_blind", "expected": str(p)})

    return rows, missing


def collect_sample_tables(inverse_dir: Path) -> pd.DataFrame:
    specs = [
        ("pam4_target_continuous_blind", ["01_pam4_target_continuous_blind_0to1/per_sample_summary.csv", "01_continuous_0to1_amplitude4/per_sample_summary.csv"]),
        ("pam4_exhaustive_enumeration", ["02_pam4_exhaustive_enumeration/per_sample_summary.csv", "02_enum_amplitude4/per_sample_summary.csv"]),
        ("levels10_discrete", ["03_levels10_two_optimizers/per_sample_summary_discrete.csv", "03_fine10/per_sample_summary_discrete.csv"]),
        ("levels10_continuous_round", ["03_levels10_two_optimizers/per_sample_summary_continuous_round.csv", "03_fine10/per_sample_summary_continuous_round.csv"]),
        ("levels20_discrete", ["04_levels20_two_optimizers/per_sample_summary_discrete.csv", "04_fine20/per_sample_summary_discrete.csv"]),
        ("levels20_continuous_round", ["04_levels20_two_optimizers/per_sample_summary_continuous_round.csv", "04_fine20/per_sample_summary_continuous_round.csv"]),
        ("random_continuous_target_blind", ["05_random_continuous_target_blind_0to1/per_sample_summary.csv", "05_random_continuous_0to1_ssfm/per_sample_summary.csv"]),
    ]
    frames: list[pd.DataFrame] = []
    for method_id, candidates in specs:
        path = first_existing(inverse_dir, candidates)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.insert(0, "method_id", method_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def fmt_pct(value: Any) -> str:
    v = as_float(value)
    return "N/A" if np.isnan(v) else f"{100.0 * v:.2f}%"


def fmt_num(value: Any, digits: int = 5) -> str:
    v = as_float(value)
    return "N/A" if np.isnan(v) else f"{v:.{digits}g}"


def fmt_epoch(value: Any) -> str:
    v = as_float(value)
    return "N/A" if np.isnan(v) else f"{v:.0f}"


def fmt_sec(value: Any) -> str:
    v = as_float(value)
    return "N/A" if np.isnan(v) else f"{v:.2f}"


def make_readable_table(methods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in methods.iterrows():
        rows.append({
            "逆向思路": r["method_name"],
            "样本数": int(r["n_samples"]) if not np.isnan(as_float(r["n_samples"])) else "N/A",
            "每样本restart": int(r["restarts_per_sample"]) if not np.isnan(as_float(r["restarts_per_sample"])) else "N/A",
            "主正确率定义": r["correctness_definition"],
            "主正确率": fmt_pct(r["primary_accuracy"]),
            "单脉冲正确率": fmt_pct(r["per_pulse_accuracy"]),
            "振幅MAE": fmt_num(r["amplitude_mae_mean"]),
            "冻结正向终端rel-L2": fmt_pct(r["frozen_forward_terminal_rel_l2_mean"]),
            "SSFM重构rel-L2": fmt_pct(r["ssfm_reconstruction_rel_l2_mean"]),
            "最优轮次P90": fmt_epoch(r["best_epoch_p90"]),
            "解题耗时(s)": fmt_sec(r["solution_time_sec"]),
            "平均每样本(s)": fmt_sec(r["solution_time_per_sample_sec"]),
        })
    return pd.DataFrame(rows)


def configure_chinese_font() -> None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Optional platform-specific font candidates; missing files are skipped.
    known = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    ]
    selected = None
    for font_file in known:
        if not font_file.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_file))
            selected = font_manager.FontProperties(fname=str(font_file)).get_name()
            break
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = [selected or "Microsoft YaHei", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def make_table_image(path: Path, table: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    view = table[[
        "逆向思路", "样本数", "主正确率", "单脉冲正确率", "振幅MAE",
        "冻结正向终端rel-L2", "SSFM重构rel-L2", "最优轮次P90",
        "解题耗时(s)", "平均每样本(s)",
    ]]
    fig_h = max(4.0, 0.58 * (len(view) + 2))
    fig, ax = plt.subplots(figsize=(20, fig_h))
    ax.axis("off")
    col_widths = [0.22, 0.05, 0.07, 0.075, 0.065, 0.095, 0.085, 0.07, 0.075, 0.08]
    tbl = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    tbl.scale(1, 1.55)
    ax.set_title(
        "逆向方法汇总：耗时仅统计反演优化/搜索（不含目标SSFM生成与SSFM重构验证）",
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax: Any, bars: Any, values: list[float], percent: bool = False) -> None:
    labels = [f"{100*v:.1f}%" if percent else f"{v:.2f}" for v in values]
    try:
        ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    except Exception:
        pass


def make_plots(out_dir: Path, methods: pd.DataFrame, readable: pd.DataFrame) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        configure_chinese_font()
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]

    plot_dir = out_dir / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Remove obsolete V4/V5 figures so users do not confuse SSFM-inclusive timing
    # with the V6 solution-time-only convention.
    for obsolete_name in [
        "end_to_end_timing_breakdown.png",
        "elapsed_time_by_method.png",
        "discrete_accuracy_by_method.png",
        "amplitude_mae_by_method.png",
        "terminal_rel_l2_by_method.png",
        "random_continuous_tolerance_accuracy.png",
    ]:
        obsolete = plot_dir / obsolete_name
        if obsolete.exists():
            try:
                obsolete.unlink()
            except OSError:
                pass
    created: list[str] = []

    table_path = plot_dir / "inverse_methods_summary_table.png"
    make_table_image(table_path, readable)
    created.append(str(table_path))

    # The only runtime plot in V6: solution/search time only.
    d = methods[methods["solution_time_sec"].notna()].copy()
    if not d.empty:
        x = np.arange(len(d))
        vals = d["solution_time_sec"].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(max(10, 1.55 * len(d)), 5.8))
        bars = ax.bar(x, vals)
        annotate_bars(ax, bars, vals.tolist(), percent=False)
        ax.set_ylabel("解题耗时 (s)")
        ax.set_title("逆向解题耗时对比（仅优化/搜索，不含任何SSFM时间）")
        ax.set_xticks(x)
        ax.set_xticklabels(d["method_name"], rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        q = plot_dir / "inverse_solution_time_only.png"
        fig.savefig(q, dpi=190)
        plt.close(fig)
        created.append(str(q))

    # Primary accuracy. The definition is stored in the table and differs for continuous truth.
    d = methods[methods["primary_accuracy"].notna()].copy()
    if not d.empty:
        x = np.arange(len(d))
        vals = d["primary_accuracy"].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(max(10, 1.55 * len(d)), 5.8))
        bars = ax.bar(x, vals)
        annotate_bars(ax, bars, vals.tolist(), percent=True)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("主正确率")
        ax.set_title("各逆向方法主正确率（具体判定标准见汇总表）")
        ax.set_xticks(x)
        ax.set_xticklabels(d["method_name"], rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        q = plot_dir / "inverse_primary_accuracy.png"
        fig.savefig(q, dpi=190)
        plt.close(fig)
        created.append(str(q))

    # Amplitude MAE.
    d = methods[methods["amplitude_mae_mean"].notna()].copy()
    if not d.empty:
        x = np.arange(len(d))
        vals = d["amplitude_mae_mean"].astype(float).to_numpy()
        fig, ax = plt.subplots(figsize=(max(10, 1.55 * len(d)), 5.8))
        bars = ax.bar(x, vals)
        labels = [f"{v:.5f}" for v in vals]
        try:
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
        except Exception:
            pass
        ax.set_ylabel("平均振幅 MAE")
        ax.set_title("各逆向方法振幅误差（0表示实际测得为0；N/A不绘制）")
        ax.set_xticks(x)
        ax.set_xticklabels(d["method_name"], rotation=24, ha="right")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        q = plot_dir / "inverse_amplitude_mae.png"
        fig.savefig(q, dpi=190)
        plt.close(fig)
        created.append(str(q))

    return created


def infer_M(run_dir: Path, inverse_dir: Path) -> int | None:
    candidates = [
        run_dir / "ssfm_eval_grid.json",
        run_dir / "run_manifest.json",
        first_existing(inverse_dir, [
            "05_random_continuous_target_blind_0to1/summary.json",
            "05_random_continuous_0to1_ssfm/summary.json",
        ]),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            js = load_json(path)
            for value in [js.get("M"), fget(js, "ssfm_eval_grid", "M")]:
                if value is not None:
                    return int(value)
        except Exception:
            pass
    return None


def write_markdown_report(path: Path, methods: pd.DataFrame, readable: pd.DataFrame, missing: list[dict[str, Any]], M: int | None) -> None:
    lines = [
        f"# M={M if M is not None else '?'} 逆向结果汇总",
        "",
        "## 耗时口径",
        "",
        "表中的‘解题耗时’只统计冻结正向模型后的反演优化或枚举搜索。",
        "目标输出波形的SSFM生成属于出题准备，不计时；预测振幅的SSFM重构属于结果验证，也不计时。",
        "",
        "## 汇总表",
        "",
        readable.to_markdown(index=False),
        "",
        "## 方法说明",
        "",
    ]
    for _, r in methods.iterrows():
        lines.extend([
            f"### {r['method_name']}",
            "",
            f"- 目标：{r['target_amplitude_type']}",
            f"- 反演已知信息：{r['inverse_information']}",
            f"- 正确率判定：{r['correctness_definition']}",
            f"- 备注：{r['metric_note']}",
            "",
        ])
    lines.extend([
        "## N/A说明",
        "",
        "N/A表示该指标对该方法不适用或未计算，不代表误差为0。",
    ])
    if missing:
        lines.extend(["", "## 未完成的方法", ""])
        lines.extend([f"- `{item['method_id']}`：缺少 `{item['expected']}`" for item in missing])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    inverse_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "inverse"
    inverse_dir.mkdir(parents=True, exist_ok=True)

    method_rows, missing = collect_method_rows(inverse_dir)
    methods = pd.DataFrame(method_rows)
    if methods.empty:
        raise RuntimeError(f"No completed inverse summaries found under {inverse_dir}")

    machine_csv = inverse_dir / "inverse_methods_summary_solution_time_only.csv"
    methods.to_csv(machine_csv, index=False, encoding="utf-8-sig")

    readable = make_readable_table(methods)
    readable_csv = inverse_dir / "inverse_methods_summary_readable.csv"
    readable.to_csv(readable_csv, index=False, encoding="utf-8-sig")

    samples = collect_sample_tables(inverse_dir)
    sample_csv = inverse_dir / "inverse_samples_summary.csv"
    if not samples.empty:
        samples.to_csv(sample_csv, index=False, encoding="utf-8-sig")

    M = infer_M(run_dir, inverse_dir)
    plots = make_plots(inverse_dir, methods, readable)
    report = inverse_dir / "inverse_summary_report.md"
    write_markdown_report(report, methods, readable, missing, M)

    payload = {
        "run_dir": str(run_dir),
        "inverse_dir": str(inverse_dir),
        "M": M,
        "completed_method_count": int(len(methods)),
        "completed_methods": json_safe(methods.to_dict(orient="records")),
        "missing_methods": missing,
        "machine_readable_table": str(machine_csv),
        "human_readable_table": str(readable_csv),
        "sample_table": str(sample_csv) if not samples.empty else None,
        "plots": plots,
        "report": str(report),
        "runtime_rule": {
            "included": "inverse optimization/search only",
            "excluded": [
                "target SSFM waveform generation",
                "SSFM reconstruction verification",
            ],
        },
        "na_rule": "N/A means not applicable or not calculated; it never means zero error.",
    }
    summary_path = inverse_dir / "inverse_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
