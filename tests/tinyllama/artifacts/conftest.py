import itertools
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import synth_artifact


@pytest.fixture(scope="session")
def valid_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("tl-int-artifact")
    return synth_artifact.build_valid_artifact(root)


@pytest.fixture
def artifact_copy(valid_artifact, tmp_path):
    counter = itertools.count()

    def make(name=None):
        directory = tmp_path / (name or "case-%d" % next(counter))
        return synth_artifact.copy_artifact(valid_artifact, directory)

    return make
