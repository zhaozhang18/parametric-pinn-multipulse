"""Stable public entry point for ``run_fixedM_inverse_all.py``."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src" / "implementation").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root containing src/implementation")


if __name__ == "__main__":
    root = _repository_root()
    implementation = root / "src" / "implementation"
    sys.path.insert(0, str(implementation))
    runpy.run_path(str(implementation / "run_fixedM_inverse_all.py"), run_name="__main__")
