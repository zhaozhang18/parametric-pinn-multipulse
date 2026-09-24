"""Generate manuscript Figure 2 (fixed eight-slot representation)."""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "implementation").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root containing src/implementation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures/figure2"))
    args = parser.parse_args()
    root = _repository_root()
    implementation = root / "src" / "implementation"
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(implementation))
    old_cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        runpy.run_path(str(implementation / "draw_subpulses_grayscale.py"), run_name="__main__")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
