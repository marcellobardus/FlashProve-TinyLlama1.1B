import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]

for _path in (str(HERE), str(REPOSITORY_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)
