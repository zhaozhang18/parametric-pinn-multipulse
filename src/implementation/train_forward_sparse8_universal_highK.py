# -*- coding: utf-8 -*-
"""
train_forward_sparse8_universal_highK.py

通用正向 PINN：固定 8 个时间槽位，同时学习 K=1...8 个连续脉冲场景。

物理设定
--------
固定 8 个时间槽位，中心由 pulse_centers_t0(8) 给出：
[-28,-20,-12,-4,4,12,20,28] T0。

每个激活槽位的归一化场幅度取：
{0.25, 0.5, 0.75, 1.0}。
非激活槽位的幅度固定为 0，表示该槽位没有脉冲。

连续槽位规则（先删左端，再删右端，交替进行）
------------------------------------------------
K=8: 槽位 1..8 激活
K=7: 槽位 2..8 激活
K=6: 槽位 2..7 激活
K=5: 槽位 3..7 激活
K=4: 槽位 3..6 激活
K=3: 槽位 4..6 激活
K=2: 槽位 4..5 激活
K=1: 槽位 5 激活

本 High-K 增强尝试的关键修改
----------------------
1. K=1...8 按层划分 seen/unseen，而不是每个 K 固定抽 256 个。
2. 默认 seen 数量沿用固定 M 实验中已经验证的设置：
   K1=3, K2=9, K3=14, K4=26, K5=102,
   K6=410, K7=1638, K8=6554。
   总计 8756 / 87380 = 10.0206%。
3. 每个 K 内采用边际平衡的随机无放回抽样。
4. PDE 配点先等概率选择 K，再在该 K 的 seen 组合内抽样，防止 K=8 支配训练。
5. IC 损失和 PDE 损失都按 K 分别求均值，再对 K 等权平均。
6. 默认保存每个 K 的全部 unseen 组合用于后续泛化评估。
7. IC 默认保证每个 seen 组合至少出现多次；即使不同 K 的 IC 点数不同，
   损失仍按 K 等权平均，不会让 K=8 因样本多而支配 IC 损失。
8. Adam 阶段周期性重采样，扩大累计物理覆盖而不增加单轮显存峰值。
9. PDE/IC 时间点采用全局、脉冲局部及脉冲间区域的混合采样。
10. 每个 K 内按总输入功率分层抽取组合，并加入边界零场与能量守恒约束。
11. 保持 ConditionalPINN 的网络输出和 checkpoint 格式不变，因此原评估脚本无需修改。
12. PDE 配点不再按 K 等分，而是向 K=6,7,8 倾斜；损失也采用 K 难度权重。
13. 主训练后增加独立的 High-K 微调阶段，继续保留低 K 约束以防灾难性遗忘。

该脚本只训练一个 8 槽位通用正向模型。固定 M 正向、固定 M 逆向及其
汇总脚本互相独立，不需要因为本文件的修改而改动。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats.qmc import LatinHypercube

from train_multi_pulse_pinn import (
    ConditionalPINN,
    load_pde_params_from_nlse,
    pde_residual_cond,
    initial_condition,
)
from nlse import pulse_centers_t0


DEFAULT_TRAIN_COUNTS_BY_K = {
    1: 3,
    2: 9,
    3: 14,
    4: 26,
    5: 102,
    6: 410,
    7: 1638,
    8: 6554,
}


# 6 GB GPU-safe total: 40,000 PDE points per collocation batch.
# Main stage: moderately emphasize difficult high-K cases.
DEFAULT_MAIN_PDE_COUNTS_BY_K = {
    1: 1500,
    2: 1500,
    3: 2000,
    4: 3000,
    5: 4500,
    6: 6500,
    7: 8500,
    8: 12500,
}

# Fine-tuning stage: stronger emphasis on K=6,7,8 while retaining all K.
DEFAULT_FINETUNE_PDE_COUNTS_BY_K = {
    1: 500,
    2: 500,
    3: 1000,
    4: 2000,
    5: 3500,
    6: 6500,
    7: 10000,
    8: 16000,
}

DEFAULT_MAIN_K_LOSS_WEIGHTS = {
    1: 0.5,
    2: 0.5,
    3: 0.7,
    4: 0.9,
    5: 1.1,
    6: 1.4,
    7: 1.8,
    8: 2.4,
}

DEFAULT_FINETUNE_K_LOSS_WEIGHTS = {
    1: 0.2,
    2: 0.2,
    3: 0.3,
    4: 0.5,
    5: 0.8,
    6: 1.3,
    7: 2.0,
    8: 3.0,
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_combos_with_k_csv(path: Path, combos: np.ndarray, k_labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    combos = np.asarray(combos, dtype=np.float32)
    k_labels = np.asarray(k_labels, dtype=np.int64).reshape(-1)
    if len(combos) != len(k_labels):
        raise ValueError("combos and k_labels must have the same number of rows.")
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["K"] + [f"A{i+1}" for i in range(combos.shape[1])])
        for k, row in zip(k_labels.tolist(), combos.tolist()):
            wr.writerow([int(k)] + [f"{float(x):g}" for x in row])


def parse_levels(text: str) -> list[float]:
    vals = [
        float(x)
        for x in str(text).replace(";", ",").replace(" ", ",").split(",")
        if x.strip()
    ]
    if not vals:
        raise ValueError("At least one nonzero amplitude level is required.")
    if any(x <= 0.0 or x > 1.0 for x in vals):
        raise ValueError("All nonzero amplitude levels must lie in (0, 1].")
    if len(set(vals)) != len(vals):
        raise ValueError("Amplitude levels must be unique.")
    return vals


def parse_int_map(text: str) -> dict[int, int]:
    """Parse strings such as '1:3,2:9,3:14'."""
    out: dict[int, int] = {}
    chunks = str(text).replace(";", ",").split(",")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid train-count entry {chunk!r}; expected K:count, e.g. 2:9."
            )
        k_text, count_text = chunk.split(":", 1)
        k = int(k_text.strip())
        count = int(count_text.strip())
        if count <= 0:
            raise ValueError(f"Training count for K={k} must be positive.")
        out[k] = count
    if not out:
        raise ValueError("train-counts-by-k is empty.")
    return out


def train_count_map_to_text(mapping: Mapping[int, int]) -> str:
    return ",".join(f"{int(k)}:{int(mapping[k])}" for k in sorted(mapping))



def parse_float_map(text: str) -> dict[int, float]:
    """Parse strings such as '1:0.5,2:1.0,8:3.0'."""
    out: dict[int, float] = {}
    for chunk in str(text).replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(
                f"Invalid float-map entry {chunk!r}; expected K:value."
            )
        k_text, value_text = chunk.split(":", 1)
        k = int(k_text.strip())
        value = float(value_text.strip())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Weight for K={k} must be finite and positive.")
        out[k] = value
    if not out:
        raise ValueError("Float map is empty.")
    return out


def float_map_to_text(mapping: Mapping[int, float]) -> str:
    return ",".join(f"{int(k)}:{float(mapping[k]):g}" for k in sorted(mapping))


def validate_complete_k_map(
    name: str,
    mapping: Mapping[int, float | int],
    K_values: Sequence[int],
) -> None:
    expected = {int(k) for k in K_values}
    actual = {int(k) for k in mapping}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} K keys mismatch; missing={missing}, extra={extra}.")


def active_slots_for_k(K: int, n_slots: int = 8) -> tuple[int, ...]:
    """Return one contiguous active block by alternating left/right trimming."""
    K = int(K)
    n_slots = int(n_slots)
    if K < 1 or K > n_slots:
        raise ValueError(f"K must be in [1,{n_slots}], got {K}.")
    left, right = 0, n_slots - 1
    remove_left = True
    while right - left + 1 > K:
        if remove_left:
            left += 1
        else:
            right -= 1
        remove_left = not remove_left
    return tuple(range(left, right + 1))


def marginal_level_counts(values: np.ndarray, levels: Sequence[float]) -> dict[str, dict[str, int]]:
    values = np.asarray(values, dtype=np.float32)
    out: dict[str, dict[str, int]] = {}
    if values.ndim != 2:
        return out
    for j in range(values.shape[1]):
        counts = {f"{float(level):g}": 0 for level in levels}
        for x in values[:, j].tolist():
            key = f"{float(x):g}"
            counts[key] = counts.get(key, 0) + 1
        out[f"active_position_{j+1}"] = counts
    return out


def balanced_unique_sample(
    all_values: Sequence[tuple[float, ...]],
    n_seen: int,
    seed: int,
    levels: Sequence[float],
) -> list[tuple[float, ...]]:
    """Approximate marginal balance at every active pulse position.

    This is a reproducible random design, not a hand-selected list. Pairwise
    balance is not enforced, but every active position sees each amplitude
    level as equally often as possible.
    """
    n_seen = int(n_seen)
    if n_seen <= 0:
        return []
    if n_seen > len(all_values):
        raise ValueError(f"n_seen={n_seen} exceeds capacity={len(all_values)}.")

    rng = np.random.default_rng(int(seed))
    all_set = set(all_values)
    n_dims = len(all_values[0]) if all_values else 0
    levels = tuple(float(x) for x in levels)
    selected: list[tuple[float, ...]] = []
    selected_set: set[tuple[float, ...]] = set()

    attempts = 0
    while len(selected) < n_seen and attempts < 300:
        attempts += 1
        cols: list[np.ndarray] = []
        for _ in range(n_dims):
            base = n_seen // len(levels)
            rem = n_seen % len(levels)
            counts = np.full(len(levels), base, dtype=int)
            if rem:
                extra_order = rng.permutation(len(levels))[:rem]
                counts[extra_order] += 1
            col: list[float] = []
            for level, count in zip(levels, counts):
                col.extend([float(level)] * int(count))
            arr = np.asarray(col, dtype=np.float32)
            rng.shuffle(arr)
            cols.append(arr)
        rows = list(zip(*cols))
        rng.shuffle(rows)
        for row in rows:
            tup = tuple(float(x) for x in row)
            if tup in all_set and tup not in selected_set:
                selected.append(tup)
                selected_set.add(tup)
                if len(selected) >= n_seen:
                    break

    if len(selected) < n_seen:
        remaining = [row for row in all_values if row not in selected_set]
        need = n_seen - len(selected)
        idx = rng.choice(len(remaining), size=need, replace=False)
        selected.extend(remaining[int(i)] for i in idx.tolist())

    return selected[:n_seen]


def random_unique_sample(
    all_values: Sequence[tuple[float, ...]],
    n_seen: int,
    seed: int,
) -> list[tuple[float, ...]]:
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(all_values), size=int(n_seen), replace=False)
    return [all_values[int(i)] for i in idx.tolist()]


def embed_active_values(values: np.ndarray, active_slots: Sequence[int], n_slots: int = 8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    rows = np.zeros((len(values), int(n_slots)), dtype=np.float32)
    rows[:, list(active_slots)] = values
    return rows


def build_stratified_dataset(
    levels_nonzero: Sequence[float],
    K_values: Sequence[int],
    train_counts_by_k: Mapping[int, int],
    seed: int,
    sampling_strategy: str,
    test_per_k: int | None,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Create per-K train/unseen sets and combined 8-slot arrays."""
    levels = tuple(float(x) for x in levels_nonzero)
    strategy = str(sampling_strategy).strip().lower()
    if strategy not in {"balanced", "random"}:
        raise ValueError("sampling-strategy must be 'balanced' or 'random'.")

    train_by_k: dict[int, np.ndarray] = {}
    test_by_k: dict[int, np.ndarray] = {}
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    train_k_parts: list[np.ndarray] = []
    test_k_parts: list[np.ndarray] = []
    stats: dict[str, dict] = {}

    for K0 in K_values:
        K = int(K0)
        active = active_slots_for_k(K, 8)
        all_values = list(product(levels, repeat=K))
        capacity = len(all_values)
        if K not in train_counts_by_k:
            raise ValueError(
                f"No training count was supplied for K={K}. "
                f"Current map: {train_count_map_to_text(train_counts_by_k)}"
            )
        n_train = int(train_counts_by_k[K])
        if not 1 <= n_train < capacity:
            raise ValueError(
                f"For K={K}, training count must be in [1,{capacity-1}], got {n_train}."
            )

        if strategy == "balanced":
            seen_values = balanced_unique_sample(
                all_values,
                n_train,
                seed=int(seed) + 1009 * K,
                levels=levels,
            )
        else:
            seen_values = random_unique_sample(
                all_values,
                n_train,
                seed=int(seed) + 1009 * K,
            )

        seen_set = set(seen_values)
        unseen_values = [row for row in all_values if row not in seen_set]
        if test_per_k is not None:
            n_test = min(int(test_per_k), len(unseen_values))
            rng = np.random.default_rng(int(seed) + 3001 * K)
            idx = rng.choice(len(unseen_values), size=n_test, replace=False)
            unseen_values = [unseen_values[int(i)] for i in idx.tolist()]

        seen_active = np.asarray(seen_values, dtype=np.float32)
        unseen_active = np.asarray(unseen_values, dtype=np.float32)
        seen_rows = embed_active_values(seen_active, active, 8)
        unseen_rows = embed_active_values(unseen_active, active, 8)

        train_by_k[K] = seen_rows
        test_by_k[K] = unseen_rows
        train_parts.append(seen_rows)
        test_parts.append(unseen_rows)
        train_k_parts.append(np.full(len(seen_rows), K, dtype=np.int64))
        test_k_parts.append(np.full(len(unseen_rows), K, dtype=np.int64))

        inactive = [i for i in range(8) if i not in active]
        ideal = n_train / float(len(levels))
        counts = marginal_level_counts(seen_active, levels)
        max_dev = 0.0
        for position_counts in counts.values():
            for count in position_counts.values():
                max_dev = max(max_dev, abs(float(count) - ideal))

        stats[str(K)] = {
            "active_slots_0based": list(active),
            "active_slots_1based": [i + 1 for i in active],
            "inactive_slots_0based": inactive,
            "inactive_slots_1based": [i + 1 for i in inactive],
            "capacity": int(capacity),
            "actual_train": int(len(seen_rows)),
            "actual_unseen_saved": int(len(unseen_rows)),
            "train_fraction": float(len(seen_rows) / capacity),
            "sampling_strategy": strategy,
            "marginal_level_counts_active_positions": counts,
            "max_marginal_deviation_from_ideal": float(max_dev),
            "contiguous": True,
        }

    train_all = np.concatenate(train_parts, axis=0)
    test_all = np.concatenate(test_parts, axis=0)
    train_k_all = np.concatenate(train_k_parts, axis=0)
    test_k_all = np.concatenate(test_k_parts, axis=0)
    return (
        train_by_k,
        test_by_k,
        train_all,
        test_all,
        train_k_all,
        test_k_all,
        stats,
    )



def allocate_equal_counts(total: int, K_values: Sequence[int]) -> dict[int, int]:
    total = int(total)
    K_values = [int(k) for k in K_values]
    if total < len(K_values):
        raise ValueError(
            f"total={total} is smaller than the number of K strata={len(K_values)}."
        )
    base, rem = divmod(total, len(K_values))
    return {K: base + (i < rem) for i, K in enumerate(K_values)}


def allocate_fraction_counts(total: int, fractions: Sequence[float]) -> list[int]:
    """Convert nonnegative fractions to integer counts that sum exactly to total."""
    total = int(total)
    frac = np.asarray(fractions, dtype=np.float64)
    if np.any(frac < 0.0) or not np.isfinite(frac).all():
        raise ValueError(f"Invalid sampling fractions: {fractions}")
    s = float(frac.sum())
    if s <= 0.0:
        raise ValueError("At least one sampling fraction must be positive.")
    frac = frac / s
    raw = frac * total
    counts = np.floor(raw).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts.tolist()


def validate_three_fractions(name: str, values: Sequence[float]) -> None:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three fractions.")
    if any(float(x) < 0.0 for x in values):
        raise ValueError(f"{name} fractions must be nonnegative, got {values}.")
    if not math.isclose(sum(float(x) for x in values), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"{name} fractions must sum to 1.0, got {values}.")


def power_strata_indices(combos: np.ndarray, n_strata: int) -> list[np.ndarray]:
    """Split combinations into equal-width total-power bands.

    Equal-width power bands deliberately give the low/middle/high power ranges
    stable representation, instead of letting the dense middle of the discrete
    combination space dominate every collocation batch.
    """
    combos = np.asarray(combos, dtype=np.float32)
    n_strata = max(1, int(n_strata))
    powers = np.sum(combos * combos, axis=1)
    p_min = float(powers.min())
    p_max = float(powers.max())
    if n_strata == 1 or math.isclose(p_min, p_max):
        return [np.arange(len(combos), dtype=np.int64)]

    edges = np.linspace(p_min, p_max, n_strata + 1, dtype=np.float64)
    # np.digitize against internal edges gives labels 0...(n_strata-1).
    labels = np.digitize(powers, edges[1:-1], right=False)
    groups = [np.flatnonzero(labels == s).astype(np.int64) for s in range(n_strata)]
    groups = [g for g in groups if len(g) > 0]
    if not groups:
        raise RuntimeError("Power stratification produced no nonempty group.")
    return groups


def sample_power_stratified_indices(
    combos: np.ndarray,
    n_requested: int,
    n_strata: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample approximately equal counts from total-power bands."""
    combos = np.asarray(combos, dtype=np.float32)
    groups = power_strata_indices(combos, n_strata)
    budgets = allocate_fraction_counts(int(n_requested), [1.0] * len(groups))
    parts: list[np.ndarray] = []
    for s, group in enumerate(groups):
        n_s = int(budgets[s])
        if n_s <= 0:
            continue
        replace = n_s > len(group)
        parts.append(rng.choice(group, size=n_s, replace=replace).astype(np.int64))
    if not parts:
        raise RuntimeError("Power-stratified sampling requested zero rows.")
    idx = np.concatenate(parts, axis=0)
    rng.shuffle(idx)
    return idx


def build_ic_combo_indices(
    combos: np.ndarray,
    requested_budget: int,
    ensure_all: bool,
    points_per_seen: int,
    power_strata: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Guarantee repeated IC coverage of every seen combination when requested."""
    n_available = int(len(combos))
    if n_available <= 0:
        raise ValueError("No training combinations are available in this K stratum.")
    requested_budget = int(requested_budget)
    points_per_seen = max(1, int(points_per_seen))

    if ensure_all:
        base = np.repeat(np.arange(n_available, dtype=np.int64), points_per_seen)
        n_target = max(requested_budget, len(base))
        if n_target > len(base):
            extra = sample_power_stratified_indices(
                combos, n_target - len(base), power_strata, rng
            )
            idx = np.concatenate([base, extra])
        else:
            idx = base
    else:
        idx = sample_power_stratified_indices(
            combos, requested_budget, power_strata, rng
        )
    rng.shuffle(idx)
    return idx.astype(np.int64)


def sample_mixed_times(
    amplitude_rows: np.ndarray,
    centers: np.ndarray,
    t_min: float,
    t_max: float,
    global_fraction: float,
    pulse_fraction: float,
    midpoint_fraction: float,
    pulse_sigma: float,
    midpoint_sigma: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Sample time points from global, pulse-local, and inter-pulse regions."""
    A = np.asarray(amplitude_rows, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32).reshape(-1)
    n = len(A)
    counts = allocate_fraction_counts(
        n, [global_fraction, pulse_fraction, midpoint_fraction]
    )
    labels = np.concatenate(
        [
            np.full(counts[0], 0, dtype=np.int8),
            np.full(counts[1], 1, dtype=np.int8),
            np.full(counts[2], 2, dtype=np.int8),
        ]
    )
    rng = np.random.default_rng(int(seed))
    rng.shuffle(labels)
    t = np.empty(n, dtype=np.float32)

    global_rows = np.flatnonzero(labels == 0)
    if len(global_rows):
        lhs = LatinHypercube(d=1, seed=int(seed) + 17)
        unit = lhs.random(n=len(global_rows)).reshape(-1).astype(np.float32)
        t[global_rows] = float(t_min) + (float(t_max) - float(t_min)) * unit

    for i in np.flatnonzero(labels == 1):
        active = np.flatnonzero(A[i] > 0.0)
        if len(active) == 0:
            t[i] = rng.uniform(t_min, t_max)
            continue
        slot = int(rng.choice(active))
        t[i] = rng.normal(float(centers[slot]), float(pulse_sigma))

    for i in np.flatnonzero(labels == 2):
        active = np.flatnonzero(A[i] > 0.0)
        if len(active) >= 2:
            pair_start = int(rng.integers(0, len(active) - 1))
            c0 = float(centers[int(active[pair_start])])
            c1 = float(centers[int(active[pair_start + 1])])
            loc = 0.5 * (c0 + c1)
            t[i] = rng.normal(loc, float(midpoint_sigma))
        elif len(active) == 1:
            t[i] = rng.normal(float(centers[int(active[0])]), float(pulse_sigma))
        else:
            t[i] = rng.uniform(t_min, t_max)

    np.clip(t, float(t_min), float(t_max), out=t)
    return t.reshape(-1, 1).astype(np.float32), {
        "global": int(counts[0]),
        "pulse_local": int(counts[1]),
        "midpoint_local": int(counts[2]),
    }


def sample_collocation_stratified_k(
    train_by_k: Mapping[int, np.ndarray],
    K_values: Sequence[int],
    n_pde: int,
    n_ic: int,
    z_min: float,
    z_max: float,
    t_min: float,
    t_max: float,
    seed: int,
    device: torch.device,
    ensure_all_seen_ic: bool,
    ic_points_per_seen: int,
    power_strata: int,
    pde_time_fractions: Sequence[float],
    ic_time_fractions: Sequence[float],
    pulse_local_sigma: float,
    midpoint_local_sigma: float,
    n_boundary: int,
    conservation_combos_per_k: int,
    conservation_nt: int,
    pde_counts_by_k: Mapping[int, int],
):
    """Create one High-K-weighted, power-stratified collocation batch."""
    K_values = [int(k) for k in K_values]
    validate_complete_k_map("pde_counts_by_k", pde_counts_by_k, K_values)
    pde_budget = {int(k): int(pde_counts_by_k[int(k)]) for k in K_values}
    if any(v <= 0 for v in pde_budget.values()):
        raise ValueError(f"All pde_counts_by_k values must be positive: {pde_budget}")
    if sum(pde_budget.values()) != int(n_pde):
        raise ValueError(
            f"sum(pde_counts_by_k)={sum(pde_budget.values())} must equal n_pde={int(n_pde)}."
        )
    ic_budget_requested = allocate_equal_counts(int(n_ic), K_values)
    boundary_budget = allocate_equal_counts(int(n_boundary), K_values)
    centers = np.asarray(pulse_centers_t0(8), dtype=np.float32)

    z_pde_parts: list[np.ndarray] = []
    t_pde_parts: list[np.ndarray] = []
    A_pde_parts: list[np.ndarray] = []
    K_pde_parts: list[np.ndarray] = []
    z_ic_parts: list[np.ndarray] = []
    t_ic_parts: list[np.ndarray] = []
    A_ic_parts: list[np.ndarray] = []
    K_ic_parts: list[np.ndarray] = []
    z_b_parts: list[np.ndarray] = []
    t_b_parts: list[np.ndarray] = []
    A_b_parts: list[np.ndarray] = []
    K_b_parts: list[np.ndarray] = []
    z_cons_combo_parts: list[np.ndarray] = []
    A_cons_combo_parts: list[np.ndarray] = []
    K_cons_combo_parts: list[np.ndarray] = []

    actual_ic_budget: dict[int, int] = {}
    pde_time_counts_by_k: dict[str, dict[str, int]] = {}
    ic_time_counts_by_k: dict[str, dict[str, int]] = {}
    power_group_sizes_by_k: dict[str, list[int]] = {}

    for K in K_values:
        combos = np.asarray(train_by_k[K], dtype=np.float32)
        rng = np.random.default_rng(int(seed) + 7919 * K)
        groups = power_strata_indices(combos, power_strata)
        power_group_sizes_by_k[str(K)] = [int(len(g)) for g in groups]

        n_pde_k = int(pde_budget[K])
        pde_idx = sample_power_stratified_indices(
            combos, n_pde_k, power_strata, rng
        )
        A_pde_k = combos[pde_idx]
        lhs_z = LatinHypercube(d=1, seed=int(seed) + 104729 * K)
        z_unit = lhs_z.random(n=n_pde_k).astype(np.float32)
        z_pde_k = float(z_min) + (float(z_max) - float(z_min)) * z_unit
        t_pde_k, pde_counts = sample_mixed_times(
            A_pde_k,
            centers,
            t_min,
            t_max,
            *pde_time_fractions,
            pulse_local_sigma,
            midpoint_local_sigma,
            seed=int(seed) + 130363 * K,
        )
        pde_time_counts_by_k[str(K)] = pde_counts

        ic_idx = build_ic_combo_indices(
            combos=combos,
            requested_budget=int(ic_budget_requested[K]),
            ensure_all=bool(ensure_all_seen_ic),
            points_per_seen=int(ic_points_per_seen),
            power_strata=int(power_strata),
            rng=rng,
        )
        n_ic_k = int(len(ic_idx))
        actual_ic_budget[K] = n_ic_k
        A_ic_k = combos[ic_idx]
        t_ic_k, ic_counts = sample_mixed_times(
            A_ic_k,
            centers,
            t_min,
            t_max,
            *ic_time_fractions,
            pulse_local_sigma,
            midpoint_local_sigma,
            seed=int(seed) + 155921 * K,
        )
        ic_time_counts_by_k[str(K)] = ic_counts
        z_ic_k = np.zeros((n_ic_k, 1), dtype=np.float32)

        n_b_k = int(boundary_budget[K])
        b_idx = sample_power_stratified_indices(
            combos, n_b_k, power_strata, rng
        )
        A_b_k = combos[b_idx]
        lhs_z_b = LatinHypercube(d=1, seed=int(seed) + 180289 * K)
        z_b_k = float(z_min) + (float(z_max) - float(z_min)) * lhs_z_b.random(
            n=n_b_k
        ).astype(np.float32)
        t_b_k = np.empty((n_b_k, 1), dtype=np.float32)
        t_b_k[0::2, 0] = float(t_min)
        t_b_k[1::2, 0] = float(t_max)
        perm_b = rng.permutation(n_b_k)
        z_b_k, t_b_k, A_b_k = z_b_k[perm_b], t_b_k[perm_b], A_b_k[perm_b]

        n_cons_k = max(1, int(conservation_combos_per_k))
        cons_idx = sample_power_stratified_indices(
            combos, n_cons_k, power_strata, rng
        )
        A_cons_k = combos[cons_idx]
        lhs_z_c = LatinHypercube(d=1, seed=int(seed) + 196613 * K)
        z_cons_k = float(z_min) + (float(z_max) - float(z_min)) * lhs_z_c.random(
            n=n_cons_k
        ).astype(np.float32)

        z_pde_parts.append(z_pde_k.astype(np.float32))
        t_pde_parts.append(t_pde_k.astype(np.float32))
        A_pde_parts.append(A_pde_k.astype(np.float32))
        K_pde_parts.append(np.full((n_pde_k, 1), K, dtype=np.int64))
        z_ic_parts.append(z_ic_k)
        t_ic_parts.append(t_ic_k.astype(np.float32))
        A_ic_parts.append(A_ic_k.astype(np.float32))
        K_ic_parts.append(np.full((n_ic_k, 1), K, dtype=np.int64))
        z_b_parts.append(z_b_k.astype(np.float32))
        t_b_parts.append(t_b_k.astype(np.float32))
        A_b_parts.append(A_b_k.astype(np.float32))
        K_b_parts.append(np.full((n_b_k, 1), K, dtype=np.int64))
        z_cons_combo_parts.append(z_cons_k.astype(np.float32))
        A_cons_combo_parts.append(A_cons_k.astype(np.float32))
        K_cons_combo_parts.append(np.full((n_cons_k, 1), K, dtype=np.int64))

    arrays = {
        "z_pde": np.concatenate(z_pde_parts, axis=0),
        "t_pde": np.concatenate(t_pde_parts, axis=0),
        "A_pde": np.concatenate(A_pde_parts, axis=0),
        "K_pde": np.concatenate(K_pde_parts, axis=0),
        "z_ic": np.concatenate(z_ic_parts, axis=0),
        "t_ic": np.concatenate(t_ic_parts, axis=0),
        "A_ic": np.concatenate(A_ic_parts, axis=0),
        "K_ic": np.concatenate(K_ic_parts, axis=0),
        "z_boundary": np.concatenate(z_b_parts, axis=0),
        "t_boundary": np.concatenate(t_b_parts, axis=0),
        "A_boundary": np.concatenate(A_b_parts, axis=0),
        "K_boundary": np.concatenate(K_b_parts, axis=0),
        "z_cons_combo": np.concatenate(z_cons_combo_parts, axis=0),
        "A_cons_combo": np.concatenate(A_cons_combo_parts, axis=0),
        "K_cons_combo": np.concatenate(K_cons_combo_parts, axis=0),
    }

    rng_all = np.random.default_rng(int(seed) + 999983)
    for family, keys in (
        ("pde", ("z_pde", "t_pde", "A_pde", "K_pde")),
        ("ic", ("z_ic", "t_ic", "A_ic", "K_ic")),
        ("boundary", ("z_boundary", "t_boundary", "A_boundary", "K_boundary")),
        ("cons", ("z_cons_combo", "A_cons_combo", "K_cons_combo")),
    ):
        n_rows = len(arrays[keys[0]])
        perm = rng_all.permutation(n_rows)
        for key in keys:
            arrays[key] = arrays[key][perm]

    tensors = {
        key: torch.tensor(
            value,
            dtype=torch.long if key.startswith("K_") else torch.float32,
            device=device,
        )
        for key, value in arrays.items()
    }
    tensors["t_cons_grid"] = torch.linspace(
        float(t_min), float(t_max), int(conservation_nt), device=device
    ).reshape(-1, 1)

    diagnostics = {
        "pde_points_requested": int(n_pde),
        "pde_points_actual": int(len(arrays["z_pde"])),
        "pde_points_by_k": {str(k): int(v) for k, v in pde_budget.items()},
        "ic_points_requested": int(n_ic),
        "ic_points_actual": int(len(arrays["z_ic"])),
        "ic_points_requested_equal_budget_by_k": {
            str(k): int(v) for k, v in ic_budget_requested.items()
        },
        "ic_points_actual_by_k": {str(k): int(v) for k, v in actual_ic_budget.items()},
        "boundary_points_actual": int(len(arrays["z_boundary"])),
        "boundary_points_by_k": {str(k): int(v) for k, v in boundary_budget.items()},
        "conservation_combos_per_k": int(conservation_combos_per_k),
        "conservation_nt": int(conservation_nt),
        "ensure_all_seen_ic": bool(ensure_all_seen_ic),
        "ic_points_per_seen": int(ic_points_per_seen),
        "power_strata_requested": int(power_strata),
        "power_group_sizes_by_k": power_group_sizes_by_k,
        "pde_time_counts_by_k": pde_time_counts_by_k,
        "ic_time_counts_by_k": ic_time_counts_by_k,
        "loss_aggregation": "mean within each K, then weighted mean over K",
    }
    return tensors, diagnostics


def weighted_k_mean(
    per_row_value: torch.Tensor,
    k_labels: torch.Tensor,
    K_values: Sequence[int],
    k_weights: Mapping[int, float],
) -> torch.Tensor:
    """Mean inside each K, then combine strata using normalized difficulty weights."""
    validate_complete_k_map("k_weights", k_weights, K_values)
    values: list[torch.Tensor] = []
    weights: list[float] = []
    labels = k_labels.reshape(-1)
    row_value = per_row_value.reshape(-1)
    for K0 in K_values:
        K = int(K0)
        mask = labels == K
        if not bool(torch.any(mask)):
            raise RuntimeError(f"No collocation rows found for K={K}.")
        values.append(row_value[mask].mean())
        weights.append(float(k_weights[K]))
    w = torch.tensor(weights, dtype=values[0].dtype, device=values[0].device)
    w = w / w.sum()
    return torch.sum(torch.stack(values) * w)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a new High-K-emphasis universal 8-slot forward PINN attempt with "
            "difficulty-weighted collocation, losses and a dedicated fine-tuning stage."
        )
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--levels-nonzero", default="0.25,0.5,0.75,1.0")
    p.add_argument("--k-values", default="1,2,3,4,5,6,7,8")
    p.add_argument(
        "--train-counts-by-k",
        default=train_count_map_to_text(DEFAULT_TRAIN_COUNTS_BY_K),
    )
    p.add_argument("--train-per-k", type=int, default=None)
    p.add_argument("--test-per-k", type=int, default=None)
    p.add_argument("--sampling-strategy", choices=["balanced", "random"], default="balanced")

    # A moderate capacity increase over the 4x100 baseline. The checkpoint
    # remains a standard ConditionalPINN and is loadable by the old evaluator.
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--fourier-features", type=int, default=4)
    p.add_argument("--z-max-ld", type=float, default=4.0)
    p.add_argument("--t-min", type=float, default=-44.0)
    p.add_argument("--t-max", type=float, default=44.0)

    p.add_argument("--n-ic", type=int, default=7500)
    p.add_argument("--n-pde", type=int, default=40000)
    p.add_argument(
        "--ensure-all-seen-ic",
        dest="ensure_all_seen_ic",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-ensure-all-seen-ic",
        dest="ensure_all_seen_ic",
        action="store_false",
    )
    p.add_argument("--ic-points-per-seen", type=int, default=2)
    p.add_argument("--power-strata", type=int, default=3)
    p.add_argument(
        "--pde-counts-by-k",
        default=train_count_map_to_text(DEFAULT_MAIN_PDE_COUNTS_BY_K),
        help="Main-stage PDE allocation; counts must sum to --n-pde.",
    )
    p.add_argument(
        "--finetune-pde-counts-by-k",
        default=train_count_map_to_text(DEFAULT_FINETUNE_PDE_COUNTS_BY_K),
        help="High-K fine-tuning PDE allocation; counts must sum to --n-pde.",
    )
    p.add_argument(
        "--main-k-loss-weights",
        default=float_map_to_text(DEFAULT_MAIN_K_LOSS_WEIGHTS),
        help="Positive K difficulty weights used in the main Adam stage.",
    )
    p.add_argument(
        "--finetune-k-loss-weights",
        default=float_map_to_text(DEFAULT_FINETUNE_K_LOSS_WEIGHTS),
        help="Positive K difficulty weights used in High-K fine-tuning and L-BFGS.",
    )

    p.add_argument("--pde-global-fraction", type=float, default=0.40)
    p.add_argument("--pde-pulse-fraction", type=float, default=0.40)
    p.add_argument("--pde-midpoint-fraction", type=float, default=0.20)
    p.add_argument("--ic-global-fraction", type=float, default=0.20)
    p.add_argument("--ic-pulse-fraction", type=float, default=0.55)
    p.add_argument("--ic-midpoint-fraction", type=float, default=0.25)
    p.add_argument("--pulse-local-sigma", type=float, default=3.5)
    p.add_argument("--midpoint-local-sigma", type=float, default=2.5)

    p.add_argument("--n-boundary", type=int, default=1024)
    p.add_argument("--conservation-combos-per-k", type=int, default=2)
    p.add_argument("--conservation-nt", type=int, default=128)

    p.add_argument("--adam-steps", type=int, default=5000)
    p.add_argument("--finetune-steps", type=int, default=1500)
    p.add_argument("--finetune-lr", type=float, default=3e-4)
    p.add_argument("--finetune-resample-every", type=int, default=100)
    p.add_argument("--ic-warmup-steps", type=int, default=500)
    p.add_argument("--resample-every", type=int, default=100)
    p.add_argument("--lbfgs-epochs", type=int, default=1200)
    p.add_argument("--lbfgs-max-iter", type=int, default=20)
    p.add_argument("--min-lbfgs-epochs", type=int, default=100)
    p.add_argument("--early-stop-eps", type=float, default=1e-8)
    p.add_argument("--early-stop-patience", type=int, default=20)

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ic-weight", type=float, default=2.0)
    p.add_argument("--pde-weight", type=float, default=1.0)
    p.add_argument("--boundary-weight", type=float, default=0.05)
    p.add_argument("--conservation-weight", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    out_dir = ensure_dir(args.out_dir)
    levels_nonzero = parse_levels(args.levels_nonzero)
    K_values = sorted(
        set(int(x) for x in str(args.k_values).replace(",", " ").split() if x.strip())
    )
    if not K_values:
        raise ValueError("k-values is empty.")
    if any(K < 1 or K > 8 for K in K_values):
        raise ValueError(f"All K values must lie in [1,8], got {K_values}.")
    if args.ic_points_per_seen < 1:
        raise ValueError("ic-points-per-seen must be at least 1.")
    if args.power_strata < 1:
        raise ValueError("power-strata must be at least 1.")
    if args.conservation_nt < 16:
        raise ValueError("conservation-nt must be at least 16.")
    validate_three_fractions(
        "PDE time",
        [args.pde_global_fraction, args.pde_pulse_fraction, args.pde_midpoint_fraction],
    )
    validate_three_fractions(
        "IC time",
        [args.ic_global_fraction, args.ic_pulse_fraction, args.ic_midpoint_fraction],
    )

    train_counts_by_k = parse_int_map(args.train_counts_by_k)
    if args.train_per_k is not None:
        train_counts_by_k = {K: int(args.train_per_k) for K in K_values}

    main_pde_counts_by_k = parse_int_map(args.pde_counts_by_k)
    finetune_pde_counts_by_k = parse_int_map(args.finetune_pde_counts_by_k)
    main_k_loss_weights = parse_float_map(args.main_k_loss_weights)
    finetune_k_loss_weights = parse_float_map(args.finetune_k_loss_weights)
    for name, mapping in (
        ("main_pde_counts_by_k", main_pde_counts_by_k),
        ("finetune_pde_counts_by_k", finetune_pde_counts_by_k),
        ("main_k_loss_weights", main_k_loss_weights),
        ("finetune_k_loss_weights", finetune_k_loss_weights),
    ):
        validate_complete_k_map(name, mapping, K_values)
    if sum(main_pde_counts_by_k.values()) != int(args.n_pde):
        raise ValueError(
            f"Main PDE allocation sums to {sum(main_pde_counts_by_k.values())}, "
            f"but --n-pde={args.n_pde}."
        )
    if sum(finetune_pde_counts_by_k.values()) != int(args.n_pde):
        raise ValueError(
            f"Fine-tune PDE allocation sums to {sum(finetune_pde_counts_by_k.values())}, "
            f"but --n-pde={args.n_pde}."
        )

    (
        train_by_k,
        test_by_k,
        train_combos,
        test_combos,
        train_k_labels,
        test_k_labels,
        sample_stats,
    ) = build_stratified_dataset(
        levels_nonzero=levels_nonzero,
        K_values=K_values,
        train_counts_by_k=train_counts_by_k,
        seed=args.seed,
        sampling_strategy=args.sampling_strategy,
        test_per_k=args.test_per_k,
    )

    dataset_dir = ensure_dir(out_dir / "dataset")
    write_combos_with_k_csv(
        dataset_dir / "seen_sparse8_combinations.csv", train_combos, train_k_labels
    )
    write_combos_with_k_csv(
        dataset_dir / "unseen_sparse8_combinations.csv", test_combos, test_k_labels
    )
    for K in K_values:
        write_combos_with_k_csv(
            dataset_dir / f"K{K}_seen_sparse8_combinations.csv",
            train_by_k[K],
            np.full(len(train_by_k[K]), K, dtype=np.int64),
        )
        write_combos_with_k_csv(
            dataset_dir / f"K{K}_unseen_sparse8_combinations.csv",
            test_by_k[K],
            np.full(len(test_by_k[K]), K, dtype=np.int64),
        )

    total_capacity = int(sum(int(sample_stats[str(K)]["capacity"]) for K in K_values))
    dataset_summary = {
        "model_type": "8_slot_contiguous_variable_K_stratified_enhanced",
        "n_slots": 8,
        "centers_t0": [float(x) for x in pulse_centers_t0(8)],
        "levels": [0.0] + [float(x) for x in levels_nonzero],
        "nonzero_levels": [float(x) for x in levels_nonzero],
        "K_values": K_values,
        "train_counts_by_k": {str(K): int(len(train_by_k[K])) for K in K_values},
        "unseen_counts_by_k": {str(K): int(len(test_by_k[K])) for K in K_values},
        "total_capacity": total_capacity,
        "n_train": int(len(train_combos)),
        "n_unseen_saved": int(len(test_combos)),
        "overall_train_fraction": float(len(train_combos) / total_capacity),
        "sampling_strategy_within_k": args.sampling_strategy,
        "test_policy": "all remaining unseen combinations"
        if args.test_per_k is None
        else f"randomly subsample at most {int(args.test_per_k)} unseen combinations per K",
        "sample_stats": sample_stats,
        "slot_policy": (
            "start from K=8 and remove left, then right, alternating; "
            "all active pulses form one contiguous block"
        ),
        "physics_sampling_policy": (
            "High-K-weighted PDE budgets and K loss weights; equal-width total-power "
            "strata; mixed global, pulse-local and midpoint-local time sampling; "
            "periodic Adam resampling plus a dedicated High-K fine-tuning stage"
        ),
        "checkpoint_compatibility": (
            "standard ConditionalPINN state_dict/model_config; existing universal evaluator is compatible"
        ),
    }
    write_json(dataset_dir / "dataset_summary.json", dataset_summary)

    model = ConditionalPINN(
        n_pulses=8,
        hidden=args.hidden,
        layers=args.layers,
        z_max_ld=args.z_max_ld,
        t_min=args.t_min,
        t_max=args.t_max,
        p_min=0.0,
        p_max=1.0,
        fourier_features=args.fourier_features,
    ).to(device)
    pde_params = load_pde_params_from_nlse()
    centers = pulse_centers_t0(8)

    latest_collocation_diagnostics: dict = {}

    def resample(seed_offset: int, stage: str):
        nonlocal latest_collocation_diagnostics
        if stage == "main":
            pde_counts_by_k = main_pde_counts_by_k
        elif stage == "finetune":
            pde_counts_by_k = finetune_pde_counts_by_k
        else:
            raise ValueError(f"Unknown training stage: {stage}")
        tensors, diagnostics = sample_collocation_stratified_k(
            train_by_k=train_by_k,
            K_values=K_values,
            n_pde=args.n_pde,
            n_ic=args.n_ic,
            z_min=0.0,
            z_max=args.z_max_ld,
            t_min=args.t_min,
            t_max=args.t_max,
            seed=args.seed + int(seed_offset),
            device=device,
            ensure_all_seen_ic=args.ensure_all_seen_ic,
            ic_points_per_seen=args.ic_points_per_seen,
            power_strata=args.power_strata,
            pde_time_fractions=(
                args.pde_global_fraction,
                args.pde_pulse_fraction,
                args.pde_midpoint_fraction,
            ),
            ic_time_fractions=(
                args.ic_global_fraction,
                args.ic_pulse_fraction,
                args.ic_midpoint_fraction,
            ),
            pulse_local_sigma=args.pulse_local_sigma,
            midpoint_local_sigma=args.midpoint_local_sigma,
            n_boundary=args.n_boundary,
            conservation_combos_per_k=args.conservation_combos_per_k,
            conservation_nt=args.conservation_nt,
            pde_counts_by_k=pde_counts_by_k,
        )
        u0, v0 = initial_condition(tensors["t_ic"], tensors["A_ic"], centers)

        # Precompute the analytic initial energy for the compact conservation set.
        n_cons = tensors["A_cons_combo"].shape[0]
        nt_cons = tensors["t_cons_grid"].shape[0]
        t0_expanded = tensors["t_cons_grid"].reshape(1, nt_cons, 1).expand(
            n_cons, nt_cons, 1
        )
        A0_expanded = tensors["A_cons_combo"].reshape(n_cons, 1, 8).expand(
            n_cons, nt_cons, 8
        )
        u_cons0, v_cons0 = initial_condition(
            t0_expanded.reshape(-1, 1), A0_expanded.reshape(-1, 8), centers
        )
        p0 = (u_cons0.pow(2) + v_cons0.pow(2)).reshape(n_cons, nt_cons)
        tensors["E0_cons_combo"] = torch.trapz(
            p0, tensors["t_cons_grid"].reshape(-1), dim=1
        ).detach()

        diagnostics["training_stage"] = stage
        diagnostics["active_pde_counts_by_k"] = {
            str(k): int(v) for k, v in pde_counts_by_k.items()
        }
        diagnostics["active_k_loss_weights"] = {
            str(k): float(v)
            for k, v in (
                main_k_loss_weights.items()
                if stage == "main"
                else finetune_k_loss_weights.items()
            )
        }
        latest_collocation_diagnostics = diagnostics
        return tensors, u0.detach(), v0.detach()

    tensors, u0, v0 = resample(0, "main")
    write_json(out_dir / "collocation_summary.json", latest_collocation_diagnostics)

    def calculate_losses(k_weights: Mapping[int, float], physics_scale: float = 1.0):
        up, vp = model(tensors["z_ic"], tensors["t_ic"], tensors["A_ic"])
        ic_per_row = (up - u0).pow(2) + (vp - v0).pow(2)
        loss_ic = weighted_k_mean(ic_per_row, tensors["K_ic"], K_values, k_weights)

        f, g = pde_residual_cond(
            model,
            tensors["z_pde"],
            tensors["t_pde"],
            tensors["A_pde"],
            pde_params,
        )
        pde_per_row = f.pow(2) + g.pow(2)
        loss_pde = weighted_k_mean(pde_per_row, tensors["K_pde"], K_values, k_weights)

        ub, vb = model(
            tensors["z_boundary"],
            tensors["t_boundary"],
            tensors["A_boundary"],
        )
        boundary_per_row = ub.pow(2) + vb.pow(2)
        loss_boundary = weighted_k_mean(
            boundary_per_row, tensors["K_boundary"], K_values, k_weights
        )

        n_cons = tensors["A_cons_combo"].shape[0]
        nt_cons = tensors["t_cons_grid"].shape[0]
        zc = tensors["z_cons_combo"].reshape(n_cons, 1, 1).expand(
            n_cons, nt_cons, 1
        )
        tc = tensors["t_cons_grid"].reshape(1, nt_cons, 1).expand(
            n_cons, nt_cons, 1
        )
        Ac = tensors["A_cons_combo"].reshape(n_cons, 1, 8).expand(
            n_cons, nt_cons, 8
        )
        uc, vc = model(zc.reshape(-1, 1), tc.reshape(-1, 1), Ac.reshape(-1, 8))
        pc = (uc.pow(2) + vc.pow(2)).reshape(n_cons, nt_cons)
        Ec = torch.trapz(pc, tensors["t_cons_grid"].reshape(-1), dim=1)
        E0 = tensors["E0_cons_combo"]
        # Log-energy ratio is scale-invariant but remains numerically bounded
        # enough at random initialization; raw relative-square error can otherwise
        # dominate the entire loss for low-power K=1 cases.
        cons_per_combo = torch.log((Ec + 1e-8) / (E0 + 1e-8)).pow(2)
        loss_conservation = weighted_k_mean(
            cons_per_combo, tensors["K_cons_combo"], K_values, k_weights
        )

        physics_scale = float(physics_scale)
        loss_total = (
            args.ic_weight * loss_ic
            + physics_scale
            * (
                args.pde_weight * loss_pde
                + args.boundary_weight * loss_boundary
                + args.conservation_weight * loss_conservation
            )
        )
        return loss_total, loss_ic, loss_pde, loss_boundary, loss_conservation

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logs: list[dict] = []
    t0 = time.time()

    def save_stage_checkpoint(filename: str, stage: str) -> Path:
        path = out_dir / filename
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": model.config(),
                "pde_params": pde_params.__dict__,
                "dataset_summary": dataset_summary,
                "collocation_summary": latest_collocation_diagnostics,
                "train_args": vars(args),
                "training_stage": stage,
                "final_loss": logs[-1] if logs else None,
            },
            path,
        )
        print(f"stage checkpoint -> {path}", flush=True)
        return path

    print("=" * 100, flush=True)
    print("High-K-emphasis universal 8-slot forward PINN: new independent attempt", flush=True)
    print(f"script={Path(__file__).resolve()}", flush=True)
    print(f"device={device}", flush=True)
    print(f"K_values={K_values}", flush=True)
    print(f"train_counts_by_k={dataset_summary['train_counts_by_k']}", flush=True)
    print(f"overall_train_fraction={dataset_summary['overall_train_fraction']:.6%}", flush=True)
    print(
        f"network=8-slot input, hidden={args.hidden}, layers={args.layers}, "
        f"Fourier={args.fourier_features}",
        flush=True,
    )
    print(
        f"IC points={latest_collocation_diagnostics['ic_points_actual']} "
        f"(points_per_seen={args.ic_points_per_seen})",
        flush=True,
    )
    print(f"Main PDE points by K={main_pde_counts_by_k}", flush=True)
    print(f"Fine-tune PDE points by K={finetune_pde_counts_by_k}", flush=True)
    print(f"Main K loss weights={main_k_loss_weights}", flush=True)
    print(f"Fine-tune K loss weights={finetune_k_loss_weights}", flush=True)
    print(f"IC points by K={latest_collocation_diagnostics['ic_points_actual_by_k']}", flush=True)
    print(
        f"resample_every={args.resample_every}, power_strata={args.power_strata}, "
        f"boundary={args.n_boundary}, conservation={args.conservation_combos_per_k}xK x {args.conservation_nt}t",
        flush=True,
    )
    print("=" * 100, flush=True)

    for step in range(1, args.adam_steps + 1):
        if args.resample_every > 0 and step > 1 and step % args.resample_every == 0:
            tensors, u0, v0 = resample(step, "main")
            write_json(out_dir / "collocation_summary.json", latest_collocation_diagnostics)

        if args.ic_warmup_steps > 0:
            physics_scale = min(1.0, max(0.05, step / float(args.ic_warmup_steps)))
        else:
            physics_scale = 1.0

        opt.zero_grad(set_to_none=True)
        loss, loss_ic, loss_pde, loss_boundary, loss_conservation = calculate_losses(
            main_k_loss_weights, physics_scale
        )
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.adam_steps:
            row = {
                "phase": "adam",
                "step": step,
                "loss": float(loss.detach().cpu()),
                "ic": float(loss_ic.detach().cpu()),
                "pde": float(loss_pde.detach().cpu()),
                "boundary": float(loss_boundary.detach().cpu()),
                "conservation": float(loss_conservation.detach().cpu()),
                "physics_scale": float(physics_scale),
                "elapsed_sec": time.time() - t0,
            }
            logs.append(row)
            print(row, flush=True)

    save_stage_checkpoint(
        "sparse8_forward_pinn_after_main_adam.pt",
        "after_main_adam",
    )

    if args.finetune_steps > 0:
        print("-" * 100, flush=True)
        print(
            f"Start High-K fine-tuning: steps={args.finetune_steps}, "
            f"lr={args.finetune_lr:g}",
            flush=True,
        )
        tensors, u0, v0 = resample(args.adam_steps + 1, "finetune")
        write_json(out_dir / "collocation_summary.json", latest_collocation_diagnostics)
        opt_finetune = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
        for ft_step in range(1, args.finetune_steps + 1):
            if (
                args.finetune_resample_every > 0
                and ft_step > 1
                and ft_step % args.finetune_resample_every == 0
            ):
                tensors, u0, v0 = resample(
                    args.adam_steps + ft_step, "finetune"
                )
                write_json(
                    out_dir / "collocation_summary.json",
                    latest_collocation_diagnostics,
                )

            opt_finetune.zero_grad(set_to_none=True)
            losses = calculate_losses(finetune_k_loss_weights, 1.0)
            loss, loss_ic, loss_pde, loss_boundary, loss_conservation = losses
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip
                )
            opt_finetune.step()

            if (
                ft_step == 1
                or ft_step % args.log_every == 0
                or ft_step == args.finetune_steps
            ):
                row = {
                    "phase": "adam_highk_finetune",
                    "step": ft_step,
                    "loss": float(loss.detach().cpu()),
                    "ic": float(loss_ic.detach().cpu()),
                    "pde": float(loss_pde.detach().cpu()),
                    "boundary": float(loss_boundary.detach().cpu()),
                    "conservation": float(loss_conservation.detach().cpu()),
                    "physics_scale": 1.0,
                    "elapsed_sec": time.time() - t0,
                }
                logs.append(row)
                print(row, flush=True)

        save_stage_checkpoint(
            "sparse8_forward_pinn_after_highk_finetune.pt",
            "after_highk_finetune",
        )

    if args.lbfgs_epochs > 0:
        opt_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=int(args.lbfgs_max_iter),
            max_eval=max(int(args.lbfgs_max_iter) + 10, int(args.lbfgs_max_iter)),
            line_search_fn="strong_wolfe",
        )
        best_lbfgs = float("inf")
        stale_epochs = 0

        for ep in range(1, args.lbfgs_epochs + 1):
            vals: dict[str, float] = {}

            def closure():
                opt_lbfgs.zero_grad(set_to_none=True)
                losses = calculate_losses(finetune_k_loss_weights, 1.0)
                loss, loss_ic, loss_pde, loss_boundary, loss_conservation = losses
                loss.backward()
                vals.update(
                    total=float(loss.detach().cpu()),
                    ic=float(loss_ic.detach().cpu()),
                    pde=float(loss_pde.detach().cpu()),
                    boundary=float(loss_boundary.detach().cpu()),
                    conservation=float(loss_conservation.detach().cpu()),
                )
                return loss

            loss_out = opt_lbfgs.step(closure)
            current = vals.get(
                "total",
                float(loss_out.detach().cpu())
                if torch.is_tensor(loss_out)
                else float(loss_out),
            )
            improvement = best_lbfgs - current
            threshold = max(float(args.early_stop_eps), abs(best_lbfgs) * 1e-7) if math.isfinite(best_lbfgs) else 0.0
            if current < best_lbfgs:
                best_lbfgs = current
            if improvement > threshold:
                stale_epochs = 0
            else:
                stale_epochs += 1

            should_log = ep == 1 or ep % args.log_every == 0 or ep == args.lbfgs_epochs
            if should_log:
                row = {
                    "phase": "lbfgs",
                    "step": ep,
                    "loss": current,
                    "ic": vals.get("ic"),
                    "pde": vals.get("pde"),
                    "boundary": vals.get("boundary"),
                    "conservation": vals.get("conservation"),
                    "physics_scale": 1.0,
                    "elapsed_sec": time.time() - t0,
                }
                logs.append(row)
                print(row, flush=True)

            if (
                ep >= int(args.min_lbfgs_epochs)
                and stale_epochs >= int(args.early_stop_patience)
            ):
                row = {
                    "phase": "lbfgs_early_stop",
                    "step": ep,
                    "loss": current,
                    "ic": vals.get("ic"),
                    "pde": vals.get("pde"),
                    "boundary": vals.get("boundary"),
                    "conservation": vals.get("conservation"),
                    "physics_scale": 1.0,
                    "elapsed_sec": time.time() - t0,
                }
                logs.append(row)
                print(
                    f"L-BFGS early stop at epoch={ep}, stale_epochs={stale_epochs}",
                    flush=True,
                )
                break

    fieldnames = [
        "phase",
        "step",
        "loss",
        "ic",
        "pde",
        "boundary",
        "conservation",
        "physics_scale",
        "elapsed_sec",
    ]
    with (out_dir / "sparse8_train_log.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(logs)

    ckpt = {
        "model_state": model.state_dict(),
        "model_config": model.config(),
        "pde_params": pde_params.__dict__,
        "dataset_summary": dataset_summary,
        "collocation_summary": latest_collocation_diagnostics,
        "train_args": vars(args),
        "final_loss": logs[-1] if logs else None,
    }
    checkpoint_path = out_dir / "sparse8_forward_pinn.pt"
    torch.save(ckpt, checkpoint_path)
    write_json(
        out_dir / "run_summary.json",
        {
            "checkpoint": str(checkpoint_path),
            "elapsed_sec": time.time() - t0,
            "final_log": logs[-1] if logs else None,
            "overall_train_fraction": dataset_summary["overall_train_fraction"],
            "train_counts_by_k": dataset_summary["train_counts_by_k"],
            "main_pde_counts_by_k": {str(k): int(v) for k, v in main_pde_counts_by_k.items()},
            "finetune_pde_counts_by_k": {str(k): int(v) for k, v in finetune_pde_counts_by_k.items()},
            "main_k_loss_weights": {str(k): float(v) for k, v in main_k_loss_weights.items()},
            "finetune_k_loss_weights": {str(k): float(v) for k, v in finetune_k_loss_weights.items()},
            "loss_aggregation": latest_collocation_diagnostics["loss_aggregation"],
            "evaluation_compatible": True,
        },
    )
    print(f"checkpoint -> {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
