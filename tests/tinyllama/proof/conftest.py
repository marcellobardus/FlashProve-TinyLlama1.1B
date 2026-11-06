import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
