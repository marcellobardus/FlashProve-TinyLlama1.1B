import hashlib
from pathlib import Path
import struct

import pytest

import tiny_artifact as ta
from tinyllama import contract
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


def _table_rule_errors(table, values):
    """The table_derive() rules of prover/hash_commitment/hf_table_params.h."""
    rows_log, log_n, data, random_entries, _zero, block_log = (
        int(field) for field in table[1:]
    )
    errors = []
    if rows_log > block_log:
        errors.append("rows log above block log")
    row_data = config_kit._weight_matrix_rows(values, block_log) * (
        1 << (block_log - rows_log)
    )
    if data < row_data + 3:
        errors.append("data below rows plus pads")
    if random_entries % 2 != 1:
        errors.append("random entries not odd")
    if data % (1 << config_kit._ceil_log2(random_entries)) != 0:
        errors.append("data not a multiple of the least power of two at least random entries")
    if data % 2 != 0:
        errors.append("data not even")
    if data % (1 << (block_log - rows_log)) != 0:
        errors.append("data not a multiple of 2^(block log - rows log)")
    if config_kit._ceil_log2(data + random_entries) + 1 > log_n:
        errors.append("log n below ceil_log2(data + random entries) + 1")
    return errors


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
    # The three table digests a dense model uses (O3): tsilu, trope, texp, eight words each, in that
    # order; never tsig (the FS/SIG table), which a dense model does not use.
    for table_name in ("tsilu", "trope", "texp"):
        assert all(("%s%d" % (table_name, i)) in names for i in range(8))
    assert not any(name.startswith("tsig") for name in names)

    # The call list is built from the digit forms: KL_ROW's shifts kv..kd are 8, so hv, ho, hgD,
    # huD, hdD are form 0 and the verifier does not expect GH, UH, OH, DH or VH. Only QKH (hq 2) is
    # present. A high segment named where the verifier does not expect it is the M1 policy-call-list
    # failure.
    calls = [line.split() for line in lines if line.startswith("call ")]
    assert calls == [
        ["call", "R16a", "R16", "G", "U"],
        ["call", "R16b", "R16", "N1", "N1H", "N2", "N2H", "O", "D", "V"],
        ["call", "R16c", "R16", "QK", "QKH"],
        ["call", "R8", "R8", "ROW"],
        ["call", "FC", "FC", "CN1", "CN2", "CQK", "CV"],
        ["call", "FL", "FL", "F"],
        ["call", "R16d", "R16", "UT"],
    ]
    for removed in ("GH", "UH", "OH", "DH", "VH", "XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG"):
        assert all(removed not in call[2:] for call in calls)
    assert values["kFD"] == 14  # computed from u0, dh, the KL attention shift (not the old literal 16)
    # kFM is derived the same way as kFD (M5l), from u0, dh and the KL shift. The small fixture's
    # dh 32 / shift 8 shape needs only kFM 13; full TinyLlama (dh 64, shift 7) needs 14. Either way
    # the old hard-coded literal 15 is wrong -- 2^15 is above u0 + 2 nHM, so the prover's nFM/nHM
    # bound rejects it (DIAG-M5K-attn-shift).
    expected_kfm = config_kit._kfm(
        {"S": contract.S, "rx": contract.RX, "dh": KIT_DIMS.DH, "kl_rows": [KL_ROW] * KIT_DIMS.L}
    )
    assert expected_kfm == 13
    assert values["kFM"] == expected_kfm
    assert values["kFM"] != 15

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
    assert _table_rule_errors(table, values) == []


def test_kfm_matches_the_full_tinyllama_shape():
    # Full TinyLlama is dh 64 at attention shift 7 (row word 11). u0 + nHM <= 2^kFM <= u0 + 2 nHM
    # is first satisfied at k 14, so kFM is 14 here -- below the old hard-coded 15.
    assert config_kit._kfm({"S": 12, "rx": 8, "dh": 64, "kl_rows": [[0] * 11 + [7]]}) == 14


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


def test_contract_table_line_matches_the_prover_rules(tmp_path):
    kl = tmp_path / "KL.bin"
    kl.write_bytes(struct.pack("<13I", *KL_ROW) * contract.L)
    out = tmp_path / "kit-tl-int-1"
    config_kit.build_config("TL-INT-1", out, lt=6, kl_path=kl, layout=False)

    lines = _lines(out)
    table = [line.split() for line in lines if line.startswith("table ")][0]
    assert " ".join(table) == "table 8 25 10027008 65535 0 15"
    assert _table_rule_errors(table, _values(lines)) == []


def test_lt_below_six_is_refused(tmp_path):
    artifact = _artifact(tmp_path)
    with pytest.raises(config_kit.KitError):
        config_kit.build_config(artifact, tmp_path / "kit-lt5", lt=5, layout=False)
