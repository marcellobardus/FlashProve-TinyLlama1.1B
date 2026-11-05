import numpy as np
import pytest

torch = pytest.importorskip("torch")

import tiny_artifact as ta
from tinyllama.artifacts import Dims, load_artifact
from tinyllama.proof import crosscheck_full

DIMS = Dims(L=2, D=64, H=4, G=2, DH=16, F=96, V=32, T=16)
DIMS_TEXT = "2,64,4,2,16,96,32,16"
KL_ROW = [28, 28, 24, 24, 8, 16, 16, 37, 32, 37, 32, 12, 1]

_DIRECTORIES = {}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("crosscheck-full")
    path = ta.build_artifact(
        root, dims=DIMS, kl_rows=[KL_ROW] * DIMS.L, seed=20261105
    )
    loaded = load_artifact(path, dims=DIMS)
    _DIRECTORIES[loaded.id] = path
    return loaded


def _directory(loaded):
    return str(_DIRECTORIES[loaded.id])


def test_two_layers_are_equal_at_full_dimensions(artifact):
    for layer in range(DIMS.L):
        result = crosscheck_full.run(artifact, [3, 7, 11, 19], layer)
        unexpected = [
            item.line() for item in result.comparisons if item.status != "EQUAL"
        ]
        assert unexpected == [], "layer %d: %s" % (layer, unexpected)
        assert result.ok
        assert result.exit_code == 0
        by_name = result.by_name()
        assert "layer%d.attention.a" % layer in by_name
        assert "layer%d.feed_forward.x_out" % layer in by_name


def test_cli_prints_equal_lines(artifact, capsys):
    code = crosscheck_full.main(
        [
            "--artifact",
            _directory(artifact),
            "--ids",
            "3,7,11,19",
            "--layer",
            "1",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "layer1.attention.q EQUAL" in output
    assert "layer1.feed_forward.x_out EQUAL" in output
    assert "DIFFERENT" not in output


def test_layer_outside_the_model_is_refused(artifact, capsys):
    code = crosscheck_full.main(
        [
            "--artifact",
            _directory(artifact),
            "--ids",
            "1,2",
            "--layer",
            str(DIMS.L),
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    assert code == 2
    assert "crosscheck_full:" in capsys.readouterr().err
