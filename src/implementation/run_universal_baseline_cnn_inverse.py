# -*- coding: utf-8 -*-
"""
run_universal_baseline_cnn_inverse.py

Baseline-size universal inverse CNN for K=1,...,8.

Architecture:
    4 residual convolution blocks, hidden=64.

Task:
    complex field at z=4LD -> complex initial field at z=0.

Data:
    Reuses the already generated shared 81-plane SSFM seen-set dataset:
    <PINN run>/universal_CNN_DDNN_full81/shared_seen_SSFM_81planes

No training data are regenerated.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from run_universal_full81_CNN_DDNN import (
    combo_text,
    dataset_paths,
    ensure_dir,
    find_universal_run,
    load_pinn_checkpoint_metadata,
    load_sparse8_csv,
    load_shared_dataset_meta,
    make_grid,
    parse_done_indices,
    resolve_checkpoint,
    run_ssfm_batch_selected,
    safe_device,
    save_csv_rows,
    select_eval_positions,
    set_seed,
    write_json,
)

SCRIPT_VERSION = "universal_baseline_cnn_inverse_v2_py38_compatible_20260716"


class SharedInverseDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        shape: Sequence[int],
        indices: Sequence[int],
        scale: float,
    ) -> None:
        self.data_path = str(data_path)
        self.shape = tuple(int(x) for x in shape)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scale = float(scale)
        self._mmap: Optional[np.memmap] = None

    def _array(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(
                self.data_path,
                dtype=np.float32,
                mode="r",
                shape=self.shape,
            )
        return self._mmap

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = int(self.indices[int(item)])
        arr = self._array()[idx]  # [Z,2,T]
        x = np.array(arr[-1], dtype=np.float32, copy=True) / self.scale
        y = np.array(arr[0], dtype=np.float32, copy=True) / self.scale
        return torch.from_numpy(x), torch.from_numpy(y)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class BaselineInverseCNN(nn.Module):
    def __init__(
        self,
        hidden: int = 64,
        kernel_size: int = 11,
        dilations: Sequence[int] = (1, 4, 16, 64),
    ) -> None:
        super().__init__()
        self.input_layer = nn.Conv1d(
            2, hidden, kernel_size, padding=kernel_size // 2
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, kernel_size, int(d)) for d in dilations]
        )
        self.output_norm = nn.GroupNorm(1, hidden)
        self.output_layer = nn.Conv1d(hidden, 2, kernel_size=1)
        nn.init.zeros_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(waveform)
        for block in self.blocks:
            x = block(x)
        delta = self.output_layer(F.gelu(self.output_norm(x)))
        return waveform + delta


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stratified_split(
    k_labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: List[int] = []
    val: List[int] = []
    all_idx = np.arange(len(k_labels), dtype=np.int64)

    for k in sorted(np.unique(k_labels).tolist()):
        idx = all_idx[k_labels == int(k)]
        perm = rng.permutation(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_val = min(n_val, len(idx) - 2)
        val.extend(int(x) for x in perm[:n_val])
        train.extend(int(x) for x in perm[n_val:])

    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def make_loader(
    data_path: Path,
    shape: Sequence[int],
    indices: np.ndarray,
    k_labels: np.ndarray,
    scale: float,
    batch_size: int,
    balanced_k: bool,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    ds = SharedInverseDataset(data_path, shape, indices, scale)

    if balanced_k:
        labels = k_labels[indices]
        unique, counts = np.unique(labels, return_counts=True)
        inv = {int(k): 1.0 / float(c) for k, c in zip(unique, counts)}
        weights = torch.tensor(
            [inv[int(k)] for k in labels],
            dtype=torch.double,
        )
        gen = torch.Generator()
        gen.manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(indices),
            replacement=True,
            generator=gen,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=True,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def loss_components(
    pred: torch.Tensor,
    target: torch.Tensor,
    field_weight: float,
    power_weight: float,
) -> Dict[str, torch.Tensor]:
    eps = 1e-12

    field_num = torch.sum((pred - target) ** 2, dim=(1, 2))
    field_den = torch.sum(target ** 2, dim=(1, 2)).clamp_min(eps)
    field_rel_sq = field_num / field_den

    p_pred = torch.sum(pred ** 2, dim=1)
    p_true = torch.sum(target ** 2, dim=1)
    power_num = torch.sum((p_pred - p_true) ** 2, dim=1)
    power_den = torch.sum(p_true ** 2, dim=1).clamp_min(eps)
    power_rel_sq = power_num / power_den

    loss = (
        field_weight * torch.mean(field_rel_sq)
        + power_weight * torch.mean(power_rel_sq)
    )
    return {
        "loss": loss,
        "field_rel_sq": torch.mean(field_rel_sq),
        "power_rel_sq": torch.mean(power_rel_sq),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    amp_enabled: bool,
    field_weight: float,
    power_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "field_rel_sq": 0.0, "power_rel_sq": 0.0}
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=bool(amp_enabled and device.type == "cuda")
            ):
                parts = loss_components(
                    model(x),
                    y,
                    field_weight,
                    power_weight,
                )
                loss = parts["loss"]

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            b = x.shape[0]
            for key in totals:
                totals[key] += float(parts[key].detach().cpu()) * b
            count += b

    return {k: v / max(count, 1) for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    scale: float,
    selected_epoch: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "script_version": SCRIPT_VERSION,
            "state_dict": model.state_dict(),
            "model_config": {
                "hidden": int(args.hidden),
                "kernel_size": int(args.kernel_size),
                "dilations": [int(x) for x in args.dilations],
            },
            "field_scale": float(scale),
            "selected_epoch": int(selected_epoch),
        },
        str(path),
    )


def load_model(path: Path, device: torch.device):
    payload = torch.load(str(path), map_location=device)
    cfg = payload["model_config"]
    model = BaselineInverseCNN(
        hidden=int(cfg["hidden"]),
        kernel_size=int(cfg["kernel_size"]),
        dilations=tuple(int(x) for x in cfg["dilations"]),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def train(
    output_root: Path,
    dataset_root: Path,
    dataset_meta: Mapping[str, Any],
    seen_k: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Path:
    final_ckpt = (
        output_root
        / "final"
        / "baseline_universal_inverse_cnn.pt"
    )
    if final_ckpt.is_file() and not args.force_train:
        print(f"[inverse] existing checkpoint: {final_ckpt}", flush=True)
        return final_ckpt

    shape = tuple(int(x) for x in dataset_meta["storage_shape"])
    scale = float(dataset_meta["field_scale_max_abs_seen"])
    data_path = dataset_paths(dataset_root)["data"]

    train_idx, val_idx = stratified_split(
        seen_k,
        args.val_fraction,
        args.split_seed,
    )

    train_loader = make_loader(
        data_path,
        shape,
        train_idx,
        seen_k,
        scale,
        args.batch_size,
        args.balanced_k,
        args.seed,
        True,
    )
    val_loader = make_loader(
        data_path,
        shape,
        val_idx,
        seen_k,
        scale,
        args.batch_size,
        False,
        args.seed,
        False,
    )

    set_seed(args.seed)
    model = BaselineInverseCNN(
        hidden=args.hidden,
        kernel_size=args.kernel_size,
        dilations=args.dilations,
    ).to(device)

    print(
        f"[inverse] baseline CNN | hidden={args.hidden} | "
        f"blocks={len(args.dilations)} | params={count_parameters(model):,}",
        flush=True,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(5, args.patience // 4),
        min_lr=args.min_learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(args.amp and device.type == "cuda")
    )

    selection_dir = ensure_dir(output_root / "selection")
    best_path = selection_dir / "best_baseline_inverse_cnn.pt"

    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.max_epochs + 1):
        tr = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            args.amp,
            args.field_weight,
            args.power_weight,
        )
        va = run_epoch(
            model,
            val_loader,
            device,
            None,
            None,
            args.amp,
            args.field_weight,
            args.power_weight,
        )

        scheduler.step(va["loss"])
        lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_loss": va["loss"],
                "train_field_rel_sq": tr["field_rel_sq"],
                "val_field_rel_sq": va["field_rel_sq"],
                "train_power_rel_sq": tr["power_rel_sq"],
                "val_power_rel_sq": va["power_rel_sq"],
                "learning_rate": lr,
            }
        )

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[inverse select] epoch={epoch:4d} "
                f"train={tr['loss']:.4e} val={va['loss']:.4e} lr={lr:.2e}",
                flush=True,
            )

        threshold = (
            0.0
            if not math.isfinite(best_val)
            else max(
                args.improvement_abs_tol,
                args.improvement_rel_tol * abs(best_val),
            )
        )

        if best_val - va["loss"] > threshold:
            best_val = va["loss"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(best_path, model, scale, best_epoch, args)
        else:
            stale += 1

        if epoch >= args.min_epochs and stale >= args.patience:
            print(
                f"[inverse select] early stop at epoch={epoch}; "
                f"selected_epoch={best_epoch}",
                flush=True,
            )
            break

    save_csv_rows(selection_dir / "training_history.csv", history)

    all_idx = np.arange(shape[0], dtype=np.int64)
    final_loader = make_loader(
        data_path,
        shape,
        all_idx,
        seen_k,
        scale,
        args.batch_size,
        args.balanced_k,
        args.seed + 17,
        True,
    )

    set_seed(args.seed)
    final_model = BaselineInverseCNN(
        hidden=args.hidden,
        kernel_size=args.kernel_size,
        dilations=args.dilations,
    ).to(device)
    final_opt = torch.optim.Adam(
        final_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    final_scaler = torch.cuda.amp.GradScaler(
        enabled=bool(args.amp and device.type == "cuda")
    )

    print(
        f"[inverse final] retrain on all {shape[0]} exact PINN-seen cases "
        f"for {best_epoch} epochs",
        flush=True,
    )

    for epoch in range(1, best_epoch + 1):
        tr = run_epoch(
            final_model,
            final_loader,
            device,
            final_opt,
            final_scaler,
            args.amp,
            args.field_weight,
            args.power_weight,
        )
        if epoch == 1 or epoch % args.log_every == 0 or epoch == best_epoch:
            print(
                f"[inverse final] epoch={epoch}/{best_epoch} "
                f"loss={tr['loss']:.4e}",
                flush=True,
            )

    save_checkpoint(final_ckpt, final_model, scale, best_epoch, args)

    write_json(
        output_root / "final" / "train_config.json",
        {
            "script_version": SCRIPT_VERSION,
            "architecture": "4 residual blocks, hidden=64",
            "selected_epoch": best_epoch,
            "n_seen": int(shape[0]),
            "balanced_k": bool(args.balanced_k),
            "uses_existing_shared_81plane_data": True,
        },
    )

    print(f"[inverse] saved: {final_ckpt}", flush=True)
    return final_ckpt


def estimate_amplitudes(initial_field: np.ndarray, tau: np.ndarray) -> np.ndarray:
    centers = np.asarray(
        [-28.0, -20.0, -12.0, -4.0, 4.0, 12.0, 20.0, 28.0],
        dtype=np.float64,
    )
    basis = np.stack(
        [np.exp(-0.5 * (tau - c) ** 2) for c in centers],
        axis=1,
    )
    pinv = np.linalg.pinv(basis)
    real_part = np.asarray(initial_field[:, 0], dtype=np.float64)
    amps = real_part @ pinv.T
    return np.clip(amps, 0.0, 1.0)


def summarize_eval(metrics_path: Path, output_dir: Path) -> None:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    k_values = sorted(set(int(r["K"]) for r in rows))
    summary_rows = []
    macro_amp_mae = []
    macro_k_acc = []

    for k in k_values:
        group = [r for r in rows if int(r["K"]) == k]
        amp_mae = np.asarray(
            [float(r["amplitude_mae_active"]) for r in group]
        )
        k_acc = np.asarray(
            [float(r["K_correct"]) for r in group]
        )
        field_err = np.asarray(
            [float(r["initial_rel_l2_field"]) for r in group]
        )
        power_err = np.asarray(
            [float(r["initial_rel_l2_power"]) for r in group]
        )

        row = {
            "K": k,
            "n_samples": len(group),
            "initial_rel_l2_field_mean": float(field_err.mean()),
            "initial_rel_l2_power_mean": float(power_err.mean()),
            "amplitude_mae_active_mean": float(amp_mae.mean()),
            "K_accuracy": float(k_acc.mean()),
        }
        summary_rows.append(row)
        macro_amp_mae.append(row["amplitude_mae_active_mean"])
        macro_k_acc.append(row["K_accuracy"])

    save_csv_rows(output_dir / "summary_by_k.csv", summary_rows)

    all_amp = np.asarray(
        [float(r["amplitude_mae_active"]) for r in rows]
    )
    all_k = np.asarray(
        [float(r["K_correct"]) for r in rows]
    )

    write_json(
        output_dir / "overall_summary.json",
        {
            "n_samples": len(rows),
            "macro_equal_K_weight": {
                "amplitude_mae_active_mean": float(np.mean(macro_amp_mae)),
                "K_accuracy": float(np.mean(macro_k_acc)),
            },
            "micro_sample_weighted": {
                "amplitude_mae_active_mean": float(all_amp.mean()),
                "K_accuracy": float(all_k.mean()),
            },
        },
    )


def evaluate(
    checkpoint: Path,
    output_root: Path,
    unseen_k: np.ndarray,
    unseen_a: np.ndarray,
    grid: Mapping[str, Any],
    pde: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    out_dir = ensure_dir(output_root / "eval_all_unseen")
    metrics_path = out_dir / "metrics_stream.csv"

    if args.overwrite_eval and metrics_path.exists():
        metrics_path.unlink()

    positions = select_eval_positions(
        unseen_k,
        args.max_eval_per_k,
        args.eval_seed,
    )
    done = parse_done_indices(metrics_path)

    model, payload = load_model(checkpoint, device)
    scale = float(payload["field_scale"])
    tau = np.asarray(grid["tau"], dtype=np.float64)

    fieldnames = (
        ["idx", "K"]
        + [f"A{i}" for i in range(1, 9)]
        + ["levels"]
        + [
            "initial_rel_l2_field",
            "initial_rel_l2_power",
            "amplitude_mae_active",
            "K_correct",
        ]
    )

    new_file = not metrics_path.exists() or metrics_path.stat().st_size == 0
    f = metrics_path.open("a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        writer.writeheader()

    pending = [int(i) for i in positions if int(i) not in done]
    print(
        f"[inverse eval] selected={len(positions)} already_done={len(positions)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )

    try:
        for start in range(0, len(pending), args.ssfm_batch_size):
            batch_idx = np.asarray(
                pending[start:start + args.ssfm_batch_size],
                dtype=np.int64,
            )
            aa = unseen_a[batch_idx]
            kk = unseen_k[batch_idx]

            ref = run_ssfm_batch_selected(
                aa,
                grid,
                pde,
                device,
                bool(args.ssfm_complex64),
            )
            true_initial = ref[:, 0]
            terminal = ref[:, -1]

            x = torch.from_numpy(
                np.asarray(terminal, dtype=np.float32) / scale
            ).to(device)

            with torch.inference_mode():
                pred = model(x).float().cpu().numpy() * scale

            eps = 1e-300
            field_err = (
                np.sqrt(np.sum((pred - true_initial) ** 2, axis=(1, 2)))
                / np.maximum(
                    np.sqrt(np.sum(true_initial ** 2, axis=(1, 2))),
                    eps,
                )
            )
            p_pred = np.sum(pred ** 2, axis=1)
            p_true = np.sum(true_initial ** 2, axis=1)
            power_err = (
                np.sqrt(np.sum((p_pred - p_true) ** 2, axis=1))
                / np.maximum(
                    np.sqrt(np.sum(p_true ** 2, axis=1)),
                    eps,
                )
            )

            amp_est = estimate_amplitudes(pred, tau)
            active = aa > 1e-12
            active_count = np.maximum(active.sum(axis=1), 1)
            amp_mae = (
                np.sum(np.abs(amp_est - aa) * active, axis=1)
                / active_count
            )
            k_hat = np.sum(amp_est >= 0.125, axis=1)
            k_true = np.sum(active, axis=1)
            k_correct = (k_hat == k_true).astype(np.float64)

            for i, idx in enumerate(batch_idx):
                writer.writerow(
                    {
                        "idx": int(idx),
                        "K": int(kk[i]),
                        **{
                            f"A{j+1}": f"{float(aa[i, j]):g}"
                            for j in range(8)
                        },
                        "levels": combo_text(aa[i]),
                        "initial_rel_l2_field": float(field_err[i]),
                        "initial_rel_l2_power": float(power_err[i]),
                        "amplitude_mae_active": float(amp_mae[i]),
                        "K_correct": float(k_correct[i]),
                    }
                )

            f.flush()
            os.fsync(f.fileno())

            print(
                f"[inverse eval] {min(start + len(batch_idx), len(pending))}/{len(pending)} | "
                f"batch amp-MAE={amp_mae.mean():.6f} | "
                f"K-acc={100*k_correct.mean():.2f}%",
                flush=True,
            )

            del ref, pred
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        f.close()

    summarize_eval(metrics_path, out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--runs-root", default="./MULTIPULSE_AMPLITUDE_RUNS")
    p.add_argument("--pinn-run-dir", default="")
    p.add_argument("--stage", choices=("train", "eval", "all"), default="all")
    p.add_argument("--device", default="cuda")

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.20)

    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--kernel-size", type=int, default=11)
    p.add_argument(
        "--dilations",
        nargs=4,
        type=int,
        default=[1, 4, 16, 64],
    )

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--max-epochs", type=int, default=800)
    p.add_argument("--min-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--improvement-rel-tol", type=float, default=1e-5)
    p.add_argument("--improvement-abs-tol", type=float, default=1e-12)
    p.add_argument("--log-every", type=int, default=5)

    p.add_argument(
        "--balanced-k",
        dest="balanced_k",
        action="store_true",
        help="Enable K-balanced sampling.",
    )
    p.add_argument(
        "--no-balanced-k",
        dest="balanced_k",
        action="store_false",
        help="Disable K-balanced sampling.",
    )
    p.set_defaults(balanced_k=True)
    p.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        help="Enable CUDA AMP mixed precision.",
    )
    p.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable CUDA AMP mixed precision.",
    )
    p.set_defaults(amp=True)
    p.add_argument("--field-weight", type=float, default=1.0)
    p.add_argument("--power-weight", type=float, default=1.0)
    p.add_argument("--force-train", action="store_true")

    p.add_argument("--ssfm-half-window", type=float, default=60.0)
    p.add_argument("--n-t", type=int, default=2048)
    p.add_argument("--n-z", type=int, default=500)
    p.add_argument("--ssfm-batch-size", type=int, default=64)
    p.add_argument("--ssfm-complex64", action="store_true")
    p.add_argument("--max-eval-per-k", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=2027)
    p.add_argument("--overwrite-eval", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    set_seed(args.seed)
    device = safe_device(args.device)

    runs_root = Path(args.runs_root).expanduser().resolve()
    run_dir = find_universal_run(runs_root, args.pinn_run_dir)
    checkpoint = resolve_checkpoint(run_dir, "")
    model_cfg, pde = load_pinn_checkpoint_metadata(checkpoint)

    seen_csv = run_dir / "dataset" / "seen_sparse8_combinations.csv"
    unseen_csv = run_dir / "dataset" / "unseen_sparse8_combinations.csv"
    seen_k, seen_a = load_sparse8_csv(seen_csv)
    unseen_k, unseen_a = load_sparse8_csv(unseen_csv)

    shared_root = (
        run_dir
        / "universal_CNN_DDNN_full81"
        / "shared_seen_SSFM_81planes"
    )
    dataset_meta = load_shared_dataset_meta(shared_root)

    output_root = ensure_dir(
        run_dir
        / "universal_CNN_DDNN_full81"
        / "universal_CNN_baseline_inverse"
    )

    endpoint_grid = make_grid(
        model_cfg=model_cfg,
        ssfm_half_window=args.ssfm_half_window,
        n_t=args.n_t,
        n_z=args.n_z,
        n_slices=2,
        compare_t_min=None,
        compare_t_max=None,
    )

    print("=" * 110, flush=True)
    print(f"SCRIPT VERSION : {SCRIPT_VERSION}", flush=True)
    print(f"PINN run       : {run_dir}", flush=True)
    print(f"shared data    : {shared_root}", flush=True)
    print(f"seen/unseen    : {len(seen_a)} / {len(unseen_a)}", flush=True)
    print("inverse CNN    : 4 residual blocks, hidden=64", flush=True)
    print(f"output         : {output_root}", flush=True)
    print("=" * 110, flush=True)

    ckpt = (
        output_root
        / "final"
        / "baseline_universal_inverse_cnn.pt"
    )

    if args.stage in ("train", "all"):
        ckpt = train(
            output_root,
            shared_root,
            dataset_meta,
            seen_k,
            device,
            args,
        )

    if args.stage in ("eval", "all"):
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        evaluate(
            ckpt,
            output_root,
            unseen_k,
            unseen_a,
            endpoint_grid,
            pde,
            device,
            args,
        )


if __name__ == "__main__":
    main()
