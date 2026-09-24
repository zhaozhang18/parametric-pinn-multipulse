# -*- coding: utf-8 -*-
"""
eval_fixedM_pinn_full_propagation.py

对固定 M 的 Fourier-PINN 做与 CNN/DDNN 相同口径的 81 面全传播评估。
不重新训练模型；自动选择每个 M 训练比例最大的正式 run，并自动寻找
Fourier forward_pinn.pt，并先用原始终端评估结果自动核验 checkpoint。

输出：
  <run_dir>/eval/pinn_full_propagation_81_exact_checkpoint/metrics_stream.csv
  <run_dir>/eval/pinn_full_propagation_81_exact_checkpoint/summary_by_model.json
  <run_dir>/eval/pinn_full_propagation_81_exact_checkpoint/run_config.json

支持断点续跑：重复执行相同命令即可跳过已经完成的 idx。
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from nlse import NLSEParams, load_combinations_csv
from train_multi_pulse_pinn import load_forward_checkpoint
from ssfm import run_ssfm
from train_fixedM_cnn_full_propagation_v2 import (
    find_run_dir,
    load_grid,
    make_tau_and_mask,
    make_z_sampling,
    run_ssfm_batch_selected,
)


METRIC_COLUMNS = [
    "full_rel_l2_field",
    "full_rel_l2_power",
    "full_e1",
    "full_e2",
    "full_mse_power",
    "full_mae_power",
    "terminal_rel_l2_field",
    "terminal_rel_l2_power",
    "terminal_e1",
    "terminal_e2",
    "terminal_mse_power",
    "terminal_mae_power",
    "ssfm_sec",
    "pinn_sec",
]

CSV_FIELDS = [
    "idx",
    "model",
    "levels",
    *METRIC_COLUMNS,
]


def safe_device(text: str) -> torch.device:
    if text != "cpu" and not torch.cuda.is_available():
        print("[warning] CUDA unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(text)


def combo_text(combo: Sequence[float]) -> str:
    return ";".join("%g" % float(x) for x in combo)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fourier_count(path: Path) -> int:
    text = str(path).lower()
    if "no_fourier" in text or "without_fourier" in text:
        return -1
    match = re.search(r"fourier[_-]?(\d+)", text)
    if match:
        return int(match.group(1))
    return 0 if "fourier" in text else -1



def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def relocate_stored_path(run_dir: Path, stored: str) -> Optional[Path]:
    """
    Resolve an absolute path saved by the original Windows run.

    First use the exact saved path. If the project was moved, rebuild the suffix
    after the fixed-M run directory name. This avoids choosing a checkpoint only
    because it is newer.
    """
    if not stored:
        return None

    direct = Path(stored)
    if direct.exists():
        return direct.resolve()

    normalized = str(stored).replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    try:
        pos = parts.index(run_dir.name)
    except ValueError:
        return None

    rebuilt = run_dir.joinpath(*parts[pos + 1 :])
    return rebuilt.resolve() if rebuilt.exists() else None


def checkpoint_fourier_features(path: Path) -> int:
    try:
        ckpt = torch.load(path, map_location="cpu")
        cfg = dict(ckpt.get("model_config", {}))
        return int(cfg.get("fourier_features", fourier_count(path)))
    except Exception:
        return fourier_count(path)


def collect_fourier_candidates(
    run_dir: Path,
    explicit_checkpoint: str = "",
) -> Tuple[List[Path], str]:
    """
    Collect candidates without using modification time as the decision rule.

    Priority:
      1. Explicit --checkpoint
      2. run_manifest.json selected_forward_checkpoint
      3. selected_forward_for_inverse.json
      4. run_manifest.json forward_checkpoints entries
      5. Other Fourier checkpoints under train/
    """
    ordered: List[Path] = []
    selected_label = "fourier4"

    def add(path: Optional[Path]) -> None:
        if path is None or not path.exists() or not path.is_file():
            return
        if checkpoint_fourier_features(path) <= 0:
            return
        resolved = path.resolve()
        if resolved not in ordered:
            ordered.append(resolved)

    if explicit_checkpoint:
        path = Path(explicit_checkpoint)
        if not path.is_absolute():
            path = run_dir / path
        add(path)

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        selected_label = str(manifest.get("selected_forward_label", selected_label))
        add(relocate_stored_path(run_dir, str(manifest.get("selected_forward_checkpoint", ""))))
        forward = manifest.get("forward_checkpoints", {})
        if isinstance(forward, dict):
            # Selected Fourier label first.
            if selected_label in forward:
                add(relocate_stored_path(run_dir, str(forward[selected_label])))
            for label, stored in forward.items():
                lower = str(label).lower()
                if "fourier" in lower and "no_fourier" not in lower and "without" not in lower:
                    add(relocate_stored_path(run_dir, str(stored)))

    selected_json = run_dir / "selected_forward_for_inverse.json"
    if selected_json.exists():
        data = read_json(selected_json)
        selected = data.get("selected", data)
        if isinstance(selected, dict):
            selected_label = str(selected.get("label", selected_label))
            add(relocate_stored_path(run_dir, str(selected.get("checkpoint", ""))))

    for path in sorted((run_dir / "train").rglob("forward_pinn.pt")):
        add(path)

    if not ordered:
        raise FileNotFoundError(
            "No Fourier forward_pinn.pt found under %s" % (run_dir / "train")
        )
    return ordered, selected_label


def find_original_terminal_metrics(
    run_dir: Path,
    selected_label: str,
) -> Tuple[Path, pd.DataFrame]:
    candidates: List[Path] = []

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        stored = str(manifest.get("forward_metrics_csv", ""))
        resolved = relocate_stored_path(run_dir, stored)
        if resolved is not None:
            candidates.append(resolved)

    for path in (run_dir / "eval").rglob("metrics_stream.csv"):
        lower_parts = [part.lower() for part in path.relative_to(run_dir / "eval").parts]
        if any(
            part.startswith("cnn_")
            or part.startswith("ddnn_")
            or part.startswith("pinn_full_propagation")
            or "backup" in part
            for part in lower_parts
        ):
            continue
        if path not in candidates:
            candidates.append(path)

    valid: List[Tuple[float, Path, pd.DataFrame]] = []
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        needed = {"idx", "model", "rel_l2_power"}
        if not needed.issubset(set(df.columns)):
            continue
        model_mask = df["model"].astype(str).str.lower() == str(selected_label).lower()
        subset = df.loc[model_mask].copy()
        if subset.empty:
            # Fallback to any non-no-Fourier model.
            lower = df["model"].astype(str).str.lower()
            subset = df[
                lower.str.contains("fourier")
                & ~lower.str.contains("no_fourier")
                & ~lower.str.contains("without")
            ].copy()
        if subset.empty:
            continue
        valid.append((path.stat().st_mtime, path, subset))

    if not valid:
        raise FileNotFoundError(
            "Could not find the original terminal PINN metrics_stream.csv for "
            f"{run_dir}. The checkpoint cannot be verified safely."
        )

    _, path, subset = max(valid, key=lambda item: item[0])
    subset["idx"] = pd.to_numeric(subset["idx"], errors="coerce")
    subset["rel_l2_power"] = pd.to_numeric(subset["rel_l2_power"], errors="coerce")
    subset = subset.dropna(subset=["idx", "rel_l2_power"])
    subset["idx"] = subset["idx"].astype(int)
    return path, subset


def predict_terminal_original_style(
    model: torch.nn.Module,
    tau: np.ndarray,
    zeta: float,
    combo: Sequence[float],
    device: torch.device,
    point_chunk: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    amplitudes_np = np.asarray(combo, dtype=np.float32).reshape(1, -1)

    with torch.no_grad():
        for start in range(0, len(tau), int(point_chunk)):
            end = min(len(tau), start + int(point_chunk))
            n = end - start
            z = torch.full((n, 1), float(zeta), dtype=torch.float32, device=device)
            t = torch.as_tensor(
                np.asarray(tau[start:end], dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32,
                device=device,
            )
            amplitudes = torch.as_tensor(
                np.repeat(amplitudes_np, n, axis=0),
                dtype=torch.float32,
                device=device,
            )
            u, v = model(z, t, amplitudes)
            pred = (
                u.detach().cpu().numpy().reshape(-1).astype(np.float64)
                + 1j * v.detach().cpu().numpy().reshape(-1).astype(np.float64)
            )
            outputs.append(pred)

    result = np.concatenate(outputs)
    if not np.isfinite(result.real).all() or not np.isfinite(result.imag).all():
        raise FloatingPointError("Terminal PINN prediction contains NaN/Inf.")
    return result


def terminal_reference_original_style(
    combo: Sequence[float],
    grid: Dict[str, Any],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    params = NLSEParams.paper_pam4(
        z_max_ld=float(grid.get("z_max_ld", 4.0)),
        t_window_t0=float(grid["eval_half_window_t0"]),
        n_t=int(grid["eval_n_t"]),
        n_z=int(grid["eval_n_z"]),
    ).with_multi_pulse(combo)

    _, t_ps, fields = run_ssfm(
        params,
        device=str(device),
        save_every=int(params.n_z),
        quiet=True,
    )
    tau = np.asarray(t_ps, dtype=np.float64) / float(params.T0_ps)
    mask = (
        (tau >= float(grid["compare_t_min"]))
        & (tau <= float(grid["compare_t_max"]))
    )
    return tau[mask], np.asarray(fields[-1], dtype=np.complex128)[mask]


def relative_l2_power(pred: np.ndarray, ref: np.ndarray) -> float:
    pred_power = np.abs(pred) ** 2
    ref_power = np.abs(ref) ** 2
    return float(
        np.linalg.norm(pred_power - ref_power)
        / (np.linalg.norm(ref_power) + 1e-300)
    )


def verification_indices(available: Sequence[int], count: int) -> List[int]:
    values = sorted(set(int(x) for x in available))
    if not values:
        return []
    count = max(1, min(int(count), len(values)))
    positions = np.linspace(0, len(values) - 1, count)
    return sorted(set(values[int(round(pos))] for pos in positions))


def select_checkpoint_by_terminal_verification(
    run_dir: Path,
    grid: Dict[str, Any],
    combos: np.ndarray,
    device: torch.device,
    point_chunk: int,
    verify_count: int,
    abs_tol: float,
    rel_tol: float,
    explicit_checkpoint: str = "",
) -> Tuple[Path, str, torch.nn.Module, Dict[str, Any]]:
    candidates, selected_label = collect_fourier_candidates(
        run_dir, explicit_checkpoint=explicit_checkpoint
    )
    original_metrics_path, original = find_original_terminal_metrics(
        run_dir, selected_label
    )

    original_by_idx = (
        original.sort_values("idx")
        .drop_duplicates(subset=["idx"], keep="last")
        .set_index("idx")
    )
    available = [
        idx for idx in original_by_idx.index.tolist()
        if 0 <= int(idx) < len(combos)
    ]
    indices = verification_indices(available, verify_count)
    if not indices:
        raise RuntimeError("No valid calibration indices were found.")

    print("\n[checkpoint verification]")
    print("  original terminal metrics =", original_metrics_path)
    print("  selected label            =", selected_label)
    print("  calibration indices       =", indices)
    print("  candidates                =", len(candidates))

    references: Dict[int, Tuple[np.ndarray, np.ndarray, float]] = {}
    for idx in indices:
        tau, ref = terminal_reference_original_style(
            combos[idx], grid, device
        )
        expected = float(original_by_idx.loc[idx, "rel_l2_power"])
        references[idx] = (tau, ref, expected)

    candidate_reports: List[Dict[str, Any]] = []
    passing: List[Tuple[int, float, Path, torch.nn.Module, Dict[str, Any]]] = []

    for priority, checkpoint in enumerate(candidates):
        model = load_forward_checkpoint(checkpoint, device)
        per_case = []
        passed = True

        for idx in indices:
            tau, ref, expected = references[idx]
            pred = predict_terminal_original_style(
                model=model,
                tau=tau,
                zeta=float(grid.get("z_max_ld", 4.0)),
                combo=combos[idx],
                device=device,
                point_chunk=point_chunk,
            )
            actual = relative_l2_power(pred, ref)
            diff = abs(actual - expected)
            tolerance = float(abs_tol) + float(rel_tol) * abs(expected)
            ok = bool(diff <= tolerance)
            passed = passed and ok
            per_case.append({
                "idx": int(idx),
                "expected": float(expected),
                "actual": float(actual),
                "abs_diff": float(diff),
                "tolerance": float(tolerance),
                "pass": ok,
            })

        max_diff = max(item["abs_diff"] for item in per_case)
        mean_diff = float(np.mean([item["abs_diff"] for item in per_case]))
        report = {
            "priority": int(priority),
            "checkpoint": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "fourier_features": checkpoint_fourier_features(checkpoint),
            "passed": bool(passed),
            "max_abs_diff": float(max_diff),
            "mean_abs_diff": float(mean_diff),
            "cases": per_case,
        }
        candidate_reports.append(report)

        print(
            "  [%s] max_diff=%.3e mean_diff=%.3e  %s"
            % (
                "PASS" if passed else "FAIL",
                max_diff,
                mean_diff,
                checkpoint,
            )
        )
        for item in per_case:
            print(
                "      idx=%d old=%.9f now=%.9f diff=%.3e tol=%.3e %s"
                % (
                    item["idx"],
                    item["expected"],
                    item["actual"],
                    item["abs_diff"],
                    item["tolerance"],
                    "PASS" if item["pass"] else "FAIL",
                )
            )

        if passed:
            passing.append((priority, max_diff, checkpoint, model, report))
        else:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not passing:
        report_path = run_dir / "eval" / "pinn_checkpoint_verification_failed.json"
        write_json(report_path, {
            "run_dir": str(run_dir),
            "selected_label": selected_label,
            "original_terminal_metrics": str(original_metrics_path),
            "reports": candidate_reports,
        })
        raise RuntimeError(
            "None of the available Fourier checkpoints reproduces the original "
            "terminal PINN metrics. Do not start the full evaluation.\n"
            f"Diagnostic report: {report_path}\n"
            "This usually means the original checkpoint was overwritten or is "
            "missing and must be restored from a backup/archive."
        )

    priority, _, checkpoint, model, chosen_report = min(
        passing, key=lambda item: (item[0], item[1])
    )
    label = (
        "fourier%d" % checkpoint_fourier_features(checkpoint)
        if checkpoint_fourier_features(checkpoint) > 0
        else selected_label
    )
    verification = {
        "status": "PASS",
        "original_terminal_metrics": str(original_metrics_path),
        "selected_label": selected_label,
        "chosen_checkpoint": str(checkpoint),
        "chosen_checkpoint_sha256": sha256_file(checkpoint),
        "calibration_indices": indices,
        "chosen_report": chosen_report,
        "all_candidates": candidate_reports,
    }
    print("  CHOSEN =", checkpoint)
    print("  SHA256 =", verification["chosen_checkpoint_sha256"])
    return checkpoint, label, model, verification


def sanitize_existing_metrics_csv(path: Path, summary_path: Path) -> None:
    """
    Keep only rows whose numerical metrics are all finite.

    The first evaluator version could write invalid rows because the prediction
    output buffer was not actually filled. The original CSV is backed up before
    any repair. Valid rows, if any, are retained; invalid rows are recomputed.
    """
    if not path.exists() or path.stat().st_size == 0:
        return

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read existing metrics CSV: {path}\n{exc}") from exc

    if df.empty:
        return

    required = [col for col in CSV_FIELDS if col not in ("model", "levels")]
    missing = [col for col in required if col not in df.columns]
    if missing:
        # Header repair runs before this function; a remaining mismatch is unsafe.
        raise RuntimeError(
            f"Existing metrics CSV is missing required columns: {missing}\nfile={path}"
        )

    metric_cols = [
        "full_rel_l2_field",
        "full_rel_l2_power",
        "full_e1",
        "full_e2",
        "full_mse_power",
        "full_mae_power",
        "terminal_rel_l2_field",
        "terminal_rel_l2_power",
        "terminal_e1",
        "terminal_e2",
        "terminal_mse_power",
        "terminal_mae_power",
    ]

    numeric = df[metric_cols].apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    n_valid = int(finite_mask.sum())
    n_total = int(len(df))

    if n_valid == n_total:
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.stem + f"_invalid_backup_{stamp}" + path.suffix)
    path.replace(backup)

    valid_df = df.loc[finite_mask, CSV_FIELDS].copy()
    if valid_df.empty:
        # Leave no active CSV, so the run starts from idx 0.
        if path.exists():
            path.unlink()
    else:
        valid_df.to_csv(path, index=False, encoding="utf-8-sig")

    if summary_path.exists():
        summary_path.unlink()

    print("[invalid metric rows removed]")
    print(f"  original rows = {n_total}")
    print(f"  valid rows    = {n_valid}")
    print(f"  recompute     = {n_total - n_valid}")
    print(f"  backup        = {backup}")
    print(f"  active CSV    = {path if path.exists() else 'will be recreated'}")


def completed_indices(path: Path) -> set:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, usecols=["idx"])
        return set(int(x) for x in df["idx"].dropna().tolist())
    except Exception:
        return set()


def predict_full_grid(
    model: torch.nn.Module,
    combos: np.ndarray,
    zeta: np.ndarray,
    tau: np.ndarray,
    device: torch.device,
    point_chunk: int,
) -> np.ndarray:
    """
    Return [B,Z,2,T] float32.

    Important:
    Real and imaginary parts are filled into two separate contiguous arrays.
    Do not flatten output[:, :, 0, :] directly: that slice is non-contiguous,
    and NumPy reshape may create a copy, leaving the original output uninitialized.
    """
    combos = np.asarray(combos, dtype=np.float32)
    zeta = np.asarray(zeta, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)

    batch = int(combos.shape[0])
    n_z = int(len(zeta))
    n_t = int(len(tau))
    total = batch * n_z * n_t

    real = np.empty((batch, n_z, n_t), dtype=np.float32)
    imag = np.empty((batch, n_z, n_t), dtype=np.float32)
    flat_real = real.reshape(-1)
    flat_imag = imag.reshape(-1)

    with torch.no_grad():
        for start in range(0, total, int(point_chunk)):
            end = min(total, start + int(point_chunk))
            flat = np.arange(start, end, dtype=np.int64)
            per_combo = n_z * n_t
            b_idx = flat // per_combo
            rem = flat % per_combo
            z_idx = rem // n_t
            t_idx = rem % n_t

            z = torch.as_tensor(
                zeta[z_idx].reshape(-1, 1), dtype=torch.float32, device=device
            )
            t = torch.as_tensor(
                tau[t_idx].reshape(-1, 1), dtype=torch.float32, device=device
            )
            amplitudes = torch.as_tensor(
                combos[b_idx], dtype=torch.float32, device=device
            )

            u, v = model(z, t, amplitudes)
            u_np = u.detach().cpu().numpy().reshape(-1).astype(np.float32)
            v_np = v.detach().cpu().numpy().reshape(-1).astype(np.float32)

            if not np.isfinite(u_np).all() or not np.isfinite(v_np).all():
                raise FloatingPointError(
                    "PINN produced NaN/Inf values while evaluating points "
                    f"{start}:{end}. Check the checkpoint and coordinate range."
                )

            flat_real[start:end] = u_np
            flat_imag[start:end] = v_np

    output = np.stack((real, imag), axis=2)  # [B,Z,2,T]
    if not np.isfinite(output).all():
        raise FloatingPointError("PINN prediction array contains NaN/Inf.")
    return output


def metric_rows(
    indices: Sequence[int],
    combos: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    label: str,
    ssfm_sec_each: float,
    pinn_sec_each: float,
) -> List[Dict[str, Any]]:
    ref = (
        reference[:, :, 0, :].astype(np.float64)
        + 1j * reference[:, :, 1, :].astype(np.float64)
    )
    pred = (
        prediction[:, :, 0, :].astype(np.float64)
        + 1j * prediction[:, :, 1, :].astype(np.float64)
    )
    rows: List[Dict[str, Any]] = []

    if not np.isfinite(reference).all():
        raise FloatingPointError(
            f"SSFM reference contains NaN/Inf for indices {list(indices)}"
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError(
            f"PINN prediction contains NaN/Inf for indices {list(indices)}"
        )

    for local, idx in enumerate(indices):
        r = ref[local]
        p = pred[local]
        power_r = np.abs(r) ** 2
        power_p = np.abs(p) ** 2

        full_field = float(
            np.linalg.norm(p - r) / (np.linalg.norm(r) + 1e-300)
        )
        full_power = float(
            np.linalg.norm(power_p - power_r) / (np.linalg.norm(power_r) + 1e-300)
        )
        terminal_field = float(
            np.linalg.norm(p[-1] - r[-1]) / (np.linalg.norm(r[-1]) + 1e-300)
        )
        terminal_power = float(
            np.linalg.norm(power_p[-1] - power_r[-1])
            / (np.linalg.norm(power_r[-1]) + 1e-300)
        )

        rows.append({
            "idx": int(idx),
            "model": str(label),
            "levels": combo_text(combos[local]),
            "full_rel_l2_field": full_field,
            "full_rel_l2_power": full_power,
            "full_e1": full_power,
            "full_e2": float(np.max(np.abs(power_p - power_r))),
            "full_mse_power": float(np.mean((power_p - power_r) ** 2)),
            "full_mae_power": float(np.mean(np.abs(power_p - power_r))),
            "terminal_rel_l2_field": terminal_field,
            "terminal_rel_l2_power": terminal_power,
            "terminal_e1": terminal_power,
            "terminal_e2": float(np.max(np.abs(power_p[-1] - power_r[-1]))),
            "terminal_mse_power": float(np.mean((power_p[-1] - power_r[-1]) ** 2)),
            "terminal_mae_power": float(np.mean(np.abs(power_p[-1] - power_r[-1]))),
            "ssfm_sec": float(ssfm_sec_each),
            "pinn_sec": float(pinn_sec_each),
        })
    return rows


def validate_or_repair_metrics_csv(path: Path) -> None:
    """
    Validate the CSV header before resume.

    A previous evaluator version may have left a header whose column order differs
    from the order used when appending rows. In that case pandas can read the file,
    but numeric values are assigned to the wrong column and the final mean is absent.

    If the width is unchanged, this function backs up the original file and rewrites
    only the header. The data rows are not recomputed.
    """
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8-sig", newline="", errors="ignore") as f:
        rows = list(csv.reader(f))

    if not rows:
        return

    header = rows[0]
    if header == CSV_FIELDS:
        return

    if len(header) != len(CSV_FIELDS):
        raise RuntimeError(
            "Existing metrics CSV has an incompatible width.\n"
            f"file={path}\n"
            f"existing columns ({len(header)}): {header}\n"
            f"expected columns ({len(CSV_FIELDS)}): {CSV_FIELDS}\n"
            "Do not delete the file yet; keep it for diagnosis."
        )

    data_rows = rows[1:]
    if not data_rows:
        # Header-only file: it is safe to replace the header.
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)
        print(f"[CSV header repaired: header-only file] {path}")
        return

    # Check whether the positional rows match the current writer order.
    sample = data_rows[: min(100, len(data_rows))]
    index_map = {name: i for i, name in enumerate(CSV_FIELDS)}

    def numeric_fraction(column: str) -> float:
        pos = index_map[column]
        valid = 0
        total = 0
        for row in sample:
            if len(row) != len(CSV_FIELDS):
                continue
            total += 1
            try:
                value = float(row[pos])
                if math.isfinite(value):
                    valid += 1
            except Exception:
                pass
        return valid / max(1, total)

    idx_fraction = numeric_fraction("idx")
    full_fraction = numeric_fraction("full_rel_l2_power")
    terminal_fraction = numeric_fraction("terminal_rel_l2_power")

    if idx_fraction < 0.9 or full_fraction < 0.9 or terminal_fraction < 0.9:
        raise RuntimeError(
            "Existing metrics CSV header/order mismatch cannot be repaired safely.\n"
            f"file={path}\n"
            f"existing header={header}\n"
            f"expected header={CSV_FIELDS}\n"
            f"finite fractions: idx={idx_fraction:.3f}, "
            f"full_power={full_fraction:.3f}, terminal_power={terminal_fraction:.3f}"
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.stem + f"_before_header_repair_{stamp}" + path.suffix)
    path.replace(backup)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        writer.writerows(data_rows)

    print("[CSV header order repaired]")
    print("  backup =", backup)
    print("  fixed  =", path)


def append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def summarize(metrics_path: Path, out_path: Path, label: str) -> Dict[str, Any]:
    df = pd.read_csv(metrics_path)

    missing_columns = [col for col in CSV_FIELDS if col not in df.columns]
    if missing_columns:
        raise RuntimeError(
            f"Metrics CSV is missing columns: {missing_columns}\nfile={metrics_path}"
        )

    record: Dict[str, Any] = {"n_rows": int(len(df))}
    valid_counts: Dict[str, int] = {}

    for col in METRIC_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")
        finite_mask = np.isfinite(values.to_numpy(dtype=np.float64, na_value=np.nan))
        finite_values = values[finite_mask].astype(float)
        valid_counts[col] = int(len(finite_values))

        if finite_values.empty:
            record[col + "_mean"] = math.nan
            record[col + "_std"] = math.nan
            record[col + "_min"] = math.nan
            record[col + "_p50"] = math.nan
            record[col + "_p95"] = math.nan
            record[col + "_max"] = math.nan
            continue

        record[col + "_mean"] = float(finite_values.mean())
        record[col + "_std"] = float(finite_values.std(ddof=0))
        record[col + "_min"] = float(finite_values.min())
        record[col + "_p50"] = float(finite_values.quantile(0.50))
        record[col + "_p95"] = float(finite_values.quantile(0.95))
        record[col + "_max"] = float(finite_values.max())

    record["valid_numeric_counts"] = valid_counts

    # Old terminal-only field names are retained for compatibility.
    record["rel_l2_field_mean"] = record.get("terminal_rel_l2_field_mean", math.nan)
    record["rel_l2_power_mean"] = record.get("terminal_rel_l2_power_mean", math.nan)
    record["e2_mean"] = record.get("terminal_e2_mean", math.nan)
    record["mse_power_mean"] = record.get("terminal_mse_power_mean", math.nan)

    summary = {
        "evaluation_scope": "all_81_propagation_planes",
        "models": {str(label): record},
        "n_metric_rows": int(len(df)),
    }
    write_json(out_path, summary)

    if valid_counts.get("full_rel_l2_power", 0) == 0:
        preview_cols = [
            "idx", "model", "levels",
            "full_rel_l2_field", "full_rel_l2_power",
            "terminal_rel_l2_field", "terminal_rel_l2_power",
        ]
        preview = df[preview_cols].head(5).to_string(index=False)
        raise RuntimeError(
            "All values in full_rel_l2_power are non-numeric/NaN.\n"
            "The evaluation rows are already saved, but the CSV contents need diagnosis.\n"
            f"file={metrics_path}\n"
            f"valid counts={valid_counts}\n"
            f"first rows:\n{preview}"
        )

    return summary


def evaluate_one(
    runs_root: Path,
    m_value: int,
    device: torch.device,
    propagation_slices: int,
    batch_size: int,
    point_chunk: int,
    max_samples: int,
    output_folder: str,
    force: bool,
    verify_only: bool,
    verify_count: int,
    verify_abs_tol: float,
    verify_rel_tol: float,
    explicit_checkpoint: str,
) -> None:
    run_dir = find_run_dir(runs_root, int(m_value))
    grid = load_grid(run_dir)
    _, time_mask, tau_compare = make_tau_and_mask(grid)
    selected_steps, zeta = make_z_sampling(
        int(grid["eval_n_z"]),
        float(grid.get("z_max_ld", 4.0)),
        int(propagation_slices),
    )

    unseen_csv = run_dir / "dataset" / "unseen_combinations.csv"
    if not unseen_csv.exists():
        raise FileNotFoundError("Missing unseen CSV: %s" % unseen_csv)
    combos = np.asarray(load_combinations_csv(unseen_csv), dtype=np.float64)
    if int(max_samples) > 0:
        combos = combos[: int(max_samples)]

    checkpoint, label, model, verification = select_checkpoint_by_terminal_verification(
        run_dir=run_dir,
        grid=grid,
        combos=combos,
        device=device,
        point_chunk=int(point_chunk),
        verify_count=int(verify_count),
        abs_tol=float(verify_abs_tol),
        rel_tol=float(verify_rel_tol),
        explicit_checkpoint=str(explicit_checkpoint),
    )

    verification_path = (
        run_dir / "eval" / "pinn_checkpoint_verification_passed.json"
    )
    write_json(verification_path, verification)

    if verify_only:
        print("[VERIFY ONLY PASS] M=%d" % int(m_value))
        print("  checkpoint =", checkpoint)
        print("  report     =", verification_path)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return

    out_dir = run_dir / "eval" / str(output_folder)
    metrics_path = out_dir / "metrics_stream.csv"
    summary_path = out_dir / "summary_by_model.json"
    config_path = out_dir / "run_config.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in (metrics_path, summary_path):
            if path.exists():
                path.unlink()

    validate_or_repair_metrics_csv(metrics_path)
    sanitize_existing_metrics_csv(metrics_path, summary_path)
    done = completed_indices(metrics_path)

    params = NLSEParams.paper_pam4(
        z_max_ld=float(grid.get("z_max_ld", 4.0)),
        t_window_t0=float(grid["eval_half_window_t0"]),
        n_t=int(grid["eval_n_t"]),
        n_z=int(grid["eval_n_z"]),
    )

    write_json(config_path, {
        "M": int(m_value),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_verification": verification,
        "model_label": str(label),
        "unseen_csv": str(unseen_csv),
        "n_test": int(len(combos)),
        "propagation_slices": int(propagation_slices),
        "selected_ssfm_steps": [int(x) for x in selected_steps.tolist()],
        "zeta_nominal": [float(x) for x in zeta.tolist()],
        "n_time_compare": int(len(tau_compare)),
        "compare_t_min": float(tau_compare[0]),
        "compare_t_max": float(tau_compare[-1]),
        "ssfm_batch_size": int(batch_size),
        "pinn_point_chunk": int(point_chunk),
        "device": str(device),
        "resume": True,
    })

    pending = [idx for idx in range(len(combos)) if idx not in done]
    print("=" * 96)
    print("[PINN full eval] M=%d" % int(m_value))
    print("run       =", run_dir)
    print("checkpoint=", checkpoint)
    print("grid      = %d planes x %d time points" % (
        int(propagation_slices), int(len(tau_compare))
    ))
    print("progress  = %d/%d already complete; %d remaining" % (
        len(done), len(combos), len(pending)
    ))
    print("output    =", out_dir)
    print("=" * 96)

    started = time.perf_counter()
    for pos in range(0, len(pending), int(batch_size)):
        indices = pending[pos:pos + int(batch_size)]
        batch_combos = combos[indices]

        t0 = time.perf_counter()
        reference = run_ssfm_batch_selected(
            params=params,
            combos=batch_combos,
            selected_steps=selected_steps,
            time_mask=time_mask,
            device=device,
        )
        ssfm_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        prediction = predict_full_grid(
            model=model,
            combos=batch_combos,
            zeta=zeta,
            tau=tau_compare,
            device=device,
            point_chunk=int(point_chunk),
        )
        pinn_sec = time.perf_counter() - t1

        rows = metric_rows(
            indices=indices,
            combos=batch_combos,
            reference=reference,
            prediction=prediction,
            label=label,
            ssfm_sec_each=ssfm_sec / max(1, len(indices)),
            pinn_sec_each=pinn_sec / max(1, len(indices)),
        )
        append_rows(metrics_path, rows)

        completed_now = len(done) + min(pos + len(indices), len(pending))
        elapsed = time.perf_counter() - started
        print(
            "[M=%d] %d/%d  batch=%d  SSFM=%.2fs  PINN=%.2fs  elapsed=%.1fmin"
            % (
                int(m_value), completed_now, len(combos), len(indices),
                ssfm_sec, pinn_sec, elapsed / 60.0,
            ),
            flush=True,
        )

        del reference, prediction, rows
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize(metrics_path, summary_path, label)
    record = summary["models"][label]
    full_mean = float(record.get("full_rel_l2_power_mean", math.nan))
    terminal_mean = float(record.get("terminal_rel_l2_power_mean", math.nan))
    print(
        "[M=%d DONE] full power mean=%s, terminal power mean=%s, n=%d"
        % (
            int(m_value),
            ("%.6f%%" % (100.0 * full_mean)) if math.isfinite(full_mean) else "NaN",
            ("%.6f%%" % (100.0 * terminal_mean)) if math.isfinite(terminal_mean) else "NaN",
            int(record["n_rows"]),
        )
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-M Fourier PINNs on all 81 propagation planes."
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--M", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--propagation-slices", type=int, default=81)
    parser.add_argument("--ssfm-batch-size", type=int, default=8)
    parser.add_argument("--pinn-point-chunk", type=int, default=131072)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--output-folder",
        default="pinn_full_propagation_81_exact_checkpoint",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the checkpoint against the original terminal metrics.",
    )
    parser.add_argument("--verify-count", type=int, default=3)
    parser.add_argument("--verify-abs-tol", type=float, default=1e-4)
    parser.add_argument("--verify-rel-tol", type=float, default=2e-3)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional explicit checkpoint path; it is still verified before use.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete this script's old metric CSV and restart. Omit for resume.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root).resolve()
    device = safe_device(str(args.device))

    for m_value in args.M:
        evaluate_one(
            runs_root=runs_root,
            m_value=int(m_value),
            device=device,
            propagation_slices=int(args.propagation_slices),
            batch_size=int(args.ssfm_batch_size),
            point_chunk=int(args.pinn_point_chunk),
            max_samples=int(args.max_samples),
            output_folder=str(args.output_folder),
            force=bool(args.force),
            verify_only=bool(args.verify_only),
            verify_count=int(args.verify_count),
            verify_abs_tol=float(args.verify_abs_tol),
            verify_rel_tol=float(args.verify_rel_tol),
            explicit_checkpoint=str(args.checkpoint),
        )

    print("\nALL REQUESTED M VALUES FINISHED.")


if __name__ == "__main__":
    main()
