from __future__ import annotations

import compileall
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "CITATION.cff",
    ROOT / "scripts/mp_pinn/train.py",
    ROOT / "scripts/beta2_mp_pinn/train.py",
    ROOT / "src/implementation/nlse.py",
    ROOT / "src/implementation/ssfm.py",
]

missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    raise SystemExit("Missing required release files: " + ", ".join(missing))

if not compileall.compile_dir(str(ROOT / "src"), quiet=1):
    raise SystemExit("Python compilation failed under src/")
if not compileall.compile_dir(str(ROOT / "scripts"), quiet=1):
    raise SystemExit("Python compilation failed under scripts/")

secret_pattern = re.compile(r"(?i)(sk-[A-Za-z0-9]{12,}|api[_-]?key\s*=|password\s*=|bearer\s+[A-Za-z0-9._-]{12,})")
for base in (ROOT / "src", ROOT / "scripts"):
    for p in base.rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="replace")
        if secret_pattern.search(text):
            raise SystemExit(f"Potential credential found in {p.relative_to(ROOT)}")

print("Release verification passed:")
print("- required files present")
print("- src/ and scripts/ compile successfully")
print("- no obvious credentials detected")
