import numpy as np
import pytest

import tiny_artifact as ta
from tinyllama.artifacts import Dims, load_artifact
from tinyllama.proof import gate_margin
from tinyllama.reference.model import forward

DIMS = Dims(L=2, D=8, H=2, G=1, DH=4, F=6, V=32, T=8)
DIMS_TEXT = "2,8,2,1,4,6,32,8"

_DIRECTORIES = {}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("gate-margin")
    path = ta.build_artifact(root, dims=DIMS, seed=20261105)
    loaded = load_artifact(path, dims=DIMS)
    _DIRECTORIES[loaded.id] = path
    return loaded


def _directory(loaded):
    return str(_DIRECTORIES[loaded.id])


def _recorded_gw(artifact, token_ids):
    recorded = {}

    def record(name, layer, values):
        recorded[(name, layer)] = np.array(values, dtype=np.int64, copy=True)

    result = forward(artifact, token_ids, record=record)
    return result, recorded


def test_collect_matches_the_reference_records(artifact):
    token_ids = [1, 2, 3]
    result, recorded = _recorded_gw(artifact, token_ids)
    report = gate_margin.collect(artifact, token_ids)

    assert result.range_failure is None
    assert report.range_failure is None
    assert [item.layer for item in report.layers] == list(range(DIMS.L))
    for layer in range(DIMS.L):
        values = recorded[("gw", layer)]
        low = int(values.min())
        high = int(values.max())
        outside = int(
            np.count_nonzero(
                (values < gate_margin.GATE_KEY_MIN)
                | (values > gate_margin.GATE_KEY_MAX)
            )
        )
        margin = report.layers[layer]
        assert margin.observed
        assert margin.minimum == low
        assert margin.maximum == high
        assert margin.count_outside == outside
        assert margin.max_abs == max(abs(low), abs(high))
        assert margin.fits_16_bits == (outside == 0)
        assert isinstance(margin.max_abs, int)
        assert isinstance(margin.count_outside, int)

    assert report.total_outside == sum(
        item.count_outside for item in report.layers
    )
    assert isinstance(report.max_abs, int)


def test_the_interval_is_flashproves_16_bit_gate_range():
    inside = gate_margin.LayerMargin.from_values(
        0, np.array([-32768, -1, 0, 1, 32639], dtype=np.int64)
    )
    assert inside.fits_16_bits
    assert inside.count_outside == 0
    assert inside.max_abs == 32768

    outside = gate_margin.LayerMargin.from_values(
        0, np.array([-32769, 0, 32640, 40000], dtype=np.int64)
    )
    assert not outside.fits_16_bits
    assert outside.count_outside == 3
    assert outside.max_abs == 40000

    unobserved = gate_margin.LayerMargin(3, None, None, 0, observed=False)
    assert not unobserved.fits_16_bits
    assert unobserved.max_abs is None


def test_main_reports_every_layer_and_integer_values(artifact, capsys):
    code = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--ids",
            "1,2,3",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    lines = output.splitlines()
    assert len([line for line in lines if line.startswith("layer ")]) == DIMS.L
    assert "." not in output
    assert lines[-1].startswith("summary windows 1 tokens 3 ")
    assert "all_layers_fit_16bit " in lines[-1]


def test_main_exits_nonzero_for_a_key_outside_16_bits(
    artifact, monkeypatch, capsys
):
    report = gate_margin.MarginReport(
        layers=[gate_margin.LayerMargin(0, -40000, 0, 1)],
        token_ids=[1],
        mode="prefix",
    )
    monkeypatch.setattr(gate_margin, "collect", lambda *args, **kwargs: report)
    code = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--ids",
            "1",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    output = capsys.readouterr().out
    assert code == 1
    assert "outside_16bit 1" in output
    assert "all_layers_fit_16bit no" in output


def test_wrong_token_id_is_refused(artifact, capsys):
    code = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--ids",
            "1,99",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    assert code == 2
    assert "gate_margin:" in capsys.readouterr().err


def test_window_file_selects_one_row(artifact, tmp_path, capsys):
    windows = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    path = tmp_path / "windows.npy"
    np.save(path, windows)

    code = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--window-file",
            str(path),
            "--window-index",
            "1",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert output.splitlines()[-1].startswith("summary windows 1 tokens 3 ")

    expected = gate_margin.collect(artifact, [4, 5, 6])
    assert gate_margin.format_report(expected) == output.splitlines()


def test_window_input_must_be_complete_and_in_range(artifact, tmp_path, capsys):
    path = tmp_path / "windows.npy"
    np.save(path, np.array([[1, 2, 3]], dtype=np.int64))

    missing_index = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--window-file",
            str(path),
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    assert missing_index == 2
    assert "needs --window-index" in capsys.readouterr().err

    outside = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--window-file",
            str(path),
            "--window-index",
            "1",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    assert outside == 2
    assert "outside 0 .. 0" in capsys.readouterr().err

    not_integer = tmp_path / "floats.npy"
    np.save(not_integer, np.array([[1.0, 2.0]], dtype=np.float64))
    wrong_dtype = gate_margin.main(
        [
            "--artifact",
            _directory(artifact),
            "--window-file",
            str(not_integer),
            "--window-index",
            "0",
            "--dims-test",
            DIMS_TEXT,
        ]
    )
    assert wrong_dtype == 2
    assert "must hold integers" in capsys.readouterr().err
