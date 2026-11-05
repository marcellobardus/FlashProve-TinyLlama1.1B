import hashlib
from pathlib import Path

import pytest

import tiny_artifact as ta
from tinyllama.artifacts import Dims
from tinyllama.proof import config_kit

KIT_DIMS = Dims(L=2, D=128, H=4, G=2, DH=32, F=256, V=128, T=64)
KL_ROW = [20, 20, 8, 8, 8, 8, 8, 37, 32, 37, 32, 8, 1]


def _lines(path):
    return (Path(path) / "policy.txt").read_text(encoding="ascii").splitlines()


def _values(lines):
    return {
        words[0]: int(words[1])
        for words in (line.split() for line in lines)
        if len(words) == 2 and words[1].isdigit()
    }


def _artifact(tmp_path):
    root = tmp_path / "artifact-root"
    return ta.build_artifact(root, dims=KIT_DIMS, kl_rows=[KL_ROW] * KIT_DIMS.L)


def test_kit_is_deterministic_and_copies_the_artifact_scales(tmp_path):
    artifact = _artifact(tmp_path)
    first = tmp_path / "kit-a"
    second = tmp_path / "kit-b"
    config_kit.build_config(artifact, first, lt=6, layout=False)
    config_kit.build_config(artifact, second, lt=6, layout=False)

    assert (first / "policy.txt").read_bytes() == (second / "policy.txt").read_bytes()
    assert (first / "KL.bin").read_bytes() == (second / "KL.bin").read_bytes()
    assert (first / "KL.bin").read_bytes() == (artifact / "KL.bin").read_bytes()

    lines = _lines(first)
    values = _values(lines)
    assert values["prof"] == 20
    assert values["lt"] == 6
    assert values["L"] == KIT_DIMS.L
    assert values["ll"] == 1
    assert values["d"] == KIT_DIMS.D
    assert values["H"] == KIT_DIMS.H
    assert values["G"] == KIT_DIMS.G
    assert values["dh"] == KIT_DIMS.DH
    assert values["f"] == KIT_DIMS.F
    assert values["fe"] == 0
    assert values["NE"] == 0
    assert values["MF"] == 37
    assert values["KF"] == 32
    assert values["eps1_lo"] == 85899
    assert values["eps2_lo"] == 85899
    assert values["epsF_lo"] == 85899
    assert values["s"] == 20
    assert values["C"] == 256
    assert values["theta"] == 256
    assert values["cap"] == 0
    assert values["sh"] == 8
    assert values["sT"] == 4
    assert values["F"] == 24
    assert values["xl"] == 5

    names = {line.split()[0] for line in lines}
    assert "gcl" not in names
    assert "rcl" not in names
    assert "eg" not in names
    assert "smax" not in names
    assert not any(name.startswith("tsilu") for name in names)
    assert not any(name.startswith("tsig") for name in names)
    assert not any(name.startswith("trope") for name in names)
    assert not any(name.startswith("texp") for name in names)

    calls = [line.split() for line in lines if line.startswith("call ")]
    assert calls == [
        ["call", "R16a", "R16", "G", "U"],
        ["call", "R16b", "R16", "N1", "N1H", "N2", "N2H", "O", "D", "V"],
        ["call", "R16c", "R16", "QK", "QKH"],
        ["call", "R8", "R8", "GH", "UH", "OH", "DH", "VH", "ROW"],
        ["call", "FC", "FC", "CN1", "CN2", "CQK", "CV"],
        ["call", "FL", "FL", "F"],
        ["call", "R16d", "R16", "UT"],
    ]
    for removed in ("XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG"):
        assert all(removed not in call[2:] for call in calls)

    call_a = [line.split() for line in lines if line.startswith("callA ")]
    assert call_a == [
        ["callA", "KR", "KR", "RM", "RD"],
        ["callA", "KFD", "KFD", "FM", "FD"],
        ["callA", "R16", "R16", "WL", "IX"],
        ["callA", "WH", "WH", "WH"],
    ]
    call_h = [line.split() for line in lines if line.startswith("callH ")]
    assert call_h == [
        ["callH", "R16a", "R16", "NF", "NFH", "HD0"],
        ["callH", "R16b", "R16", "HD1"],
        ["callH", "R8", "R8", "ROW", "HD2"],
        ["callH", "FEa", "FE", "HE0"],
        ["callH", "FEb", "FE", "HE1"],
        ["callH", "FC", "FC", "CNF"],
    ]

    table = [line.split() for line in lines if line.startswith("table ")][0]
    rows_log, log_n, data, random_entries, _e, block_log = (int(x) for x in table[1:])
    values_for_table = dict(values)
    matrix_rows = config_kit._weight_matrix_rows(values_for_table, block_log)
    assert data >= matrix_rows * (1 << (block_log - rows_log)) + 3
    assert data + random_entries < (1 << log_n)
    assert block_log >= (KIT_DIMS.V - 1).bit_length()


def test_kit_at_lt_10_and_wrong_artifact_id(tmp_path):
    artifact = _artifact(tmp_path)
    out = tmp_path / "kit-lt10"
    assert (
        config_kit.main(
            ["--artifact", str(artifact), "--out", str(out), "--lt", "10", "--no-layout"]
        )
        == 0
    )
    assert _values(_lines(out))["lt"] == 10
    assert (out / "KL.bin").read_bytes() == (artifact / "KL.bin").read_bytes()

    manifest = (artifact / "manifest.json").read_bytes()
    assert artifact.name == hashlib.sha256(manifest).hexdigest()
    with pytest.raises(config_kit.KitError):
        config_kit.build_config(
            artifact,
            tmp_path / "wrong-id",
            lt=10,
            artifact_id="0" * 64,
            layout=False,
        )

    with pytest.raises(config_kit.KitError):
        config_kit.build_config("TL-INT-1", tmp_path / "contract-only", lt=6, layout=False)
