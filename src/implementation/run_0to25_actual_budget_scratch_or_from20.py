# -*- coding: utf-8 -*-
"""
run_0to25_actual_budget_scratch_or_from20.py

Run the validated 0-25 km trainer with explicit stage budgets.
Only initialization differs between:
  scratch : random initialization
  from20  : initialize from final 0-20 km sparse8_forward_pinn.pt

This avoids defining L-BFGS budget from the nominal 1200 cap.
For the completed full 0-25 baseline, actual L-BFGS stopped at epoch 882.
Thus strict stage-count targets are:
  10%: Adam 500,  High-K 150, L-BFGS 88
  25%: Adam 1250, High-K 375, L-BFGS 221
  50%: Adam 2500, High-K 750, L-BFGS 441
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

TRAINER_NAME = "train_forward_sparse8_universal_highK_direct_0to25_density125_FIX1_RESUME.py"
FINAL_0TO20_NAME = "sparse8_forward_pinn.pt"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("_direct025_actual_budget", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import trainer: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("scratch", "from20"), required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adam-steps", type=int, required=True)
    p.add_argument("--finetune-steps", type=int, required=True)
    p.add_argument("--lbfgs-epochs", type=int, required=True)
    p.add_argument("--ic-warmup-steps", type=int, required=True)
    p.add_argument("--trainer-script", default="")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    trainer_path = (
        Path(args.trainer_script).expanduser().resolve()
        if args.trainer_script
        else (here / TRAINER_NAME).resolve()
    )
    if not trainer_path.is_file():
        raise FileNotFoundError(str(trainer_path))

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = load_module(trainer_path)

    argv = [
        str(trainer_path),
        "--out-dir", str(out_dir),
        "--device", str(args.device),
        "--seed", str(args.seed),
        "--adam-steps", str(args.adam_steps),
        "--finetune-steps", str(args.finetune_steps),
        "--lbfgs-epochs", str(args.lbfgs_epochs),
        "--ic-warmup-steps", str(args.ic_warmup_steps),
    ]

    source = None
    if args.mode == "from20":
        source = run_dir / FINAL_0TO20_NAME
        if not source.is_file():
            raise FileNotFoundError(str(source))
        argv += ["--init-checkpoint", str(source)]

    config = {
        "mode": args.mode,
        "adam_steps": args.adam_steps,
        "finetune_steps": args.finetune_steps,
        "lbfgs_epochs": args.lbfgs_epochs,
        "ic_warmup_steps": args.ic_warmup_steps,
        "source_checkpoint": str(source) if source else None,
        "trainer_script": str(trainer_path),
        "note": "Explicit stage-count budget; only initialization differs.",
    }
    (out_dir / "actual_budget_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    old_argv = list(sys.argv)
    started = time.time()
    try:
        sys.argv = argv
        trainer.main()
    finally:
        sys.argv = old_argv

    config["wrapper_wall_sec"] = float(time.time() - started)
    (out_dir / "actual_budget_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
