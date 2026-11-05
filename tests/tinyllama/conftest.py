import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARITHMETIC_DIR = Path(__file__).resolve().parent / "arithmetic"

for _path in (str(REPOSITORY_ROOT), str(ARITHMETIC_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
