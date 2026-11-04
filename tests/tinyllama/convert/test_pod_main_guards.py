import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULES = (
    "tinyllama.convert.run",
    "tinyllama.convert.check_permutation",
)


@pytest.mark.parametrize("module", MODULES)
def test_module_help_runs_through_main_guard(module):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout
    assert "usage" in result.stdout
