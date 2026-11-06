"""Independent tests for W1 follow-ups F1 and F2 (INTERFACES.md I2.4).

F2 cases pass on this branch, so they are strict.
"""

import struct

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import synth_artifact as synth

SHIFT_NAMES = ("kq", "kk", "kv", "ko", "kg", "ku", "kd")
BAD_SHIFT_VALUES = (0, 33)
GOOD_SHIFT_VALUES = (1, 32)

F2_REJECTION_CASES = [
    pytest.param(shift, value, 0, id=f"{shift}-{value}-first-layer")
    for shift in SHIFT_NAMES
    for value in BAD_SHIFT_VALUES
] + [
    pytest.param("kd", 33, synth.L - 1, id="kd-33-last-layer"),
    pytest.param("kq", 2**31, 0, id="kq-kk-2**31-first-layer"),
]

F2_ACCEPTED_CASES = [
    pytest.param(shift, value, 0, id=f"{shift}-{value}-first-layer")
    for shift in SHIFT_NAMES
    for value in GOOD_SHIFT_VALUES
] + [
    pytest.param("kd", 32, synth.L - 1, id="kd-32-last-layer"),
]


def _normalize(code):
    if isinstance(code, (tuple, list)):
        code = code[0]
    return str(code)[:3]


def _read_kl(directory):
    raw = (directory / "KL.bin").read_bytes()
    return list(struct.unpack("<%dI" % (synth.L * 13), raw))


def _write_kl(directory, words):
    synth.rewrite_file(
        directory,
        "KL.bin",
        data=synth.kl_bytes([words[i : i + 13] for i in range(0, len(words), 13)]),
        update_manifest=True,
    )
    return _retag_to_manifest_id(directory)


def _retag_to_manifest_id(directory):
    """Keep the copy's directory name equal to its (rewritten) artifact id."""
    artifact_id = synth.file_sha256(directory / "manifest.json")
    if directory.name == artifact_id:
        return directory
    target = directory.parent / artifact_id
    directory.rename(target)
    return target


def _set_shift(words, layer, shift, value):
    start = layer * 13
    index = SHIFT_NAMES.index(shift)
    if shift in ("kq", "kk"):
        words[start] = value
        words[start + 1] = value
    else:
        words[start + index] = value


def _artifact_id(directory):
    return synth.file_sha256(directory / "manifest.json")


def test_f1_directory_not_named_by_artifact_id_is_rejected(artifact_copy, valid_artifact):
    directory = artifact_copy("not-" + _artifact_id(valid_artifact))
    assert directory.name != _artifact_id(directory)

    failures = artifacts.validate_artifact(directory)
    codes = [_normalize(item) for item in failures]
    assert "A07" in codes
    assert any(
        _normalize(item) == "A07" and item[1] == "manifest.json" for item in failures
    )


def test_f1_directory_named_by_artifact_id_is_accepted(artifact_copy, valid_artifact):
    artifact_id = _artifact_id(valid_artifact)
    directory = artifact_copy(artifact_id)
    assert directory.name == artifact_id

    failures = artifacts.validate_artifact(directory)
    codes = [_normalize(item) for item in failures]
    assert "A07" not in codes
    assert failures == []


@pytest.mark.parametrize("shift, value, layer", F2_REJECTION_CASES)
def test_f2_shift_outside_one_to_thirty_two_is_rejected(
    shift, value, layer, artifact_copy
):
    directory = artifact_copy()
    words = _read_kl(directory)
    _set_shift(words, layer, shift, value)
    directory = _write_kl(directory, words)

    codes = [_normalize(item) for item in artifacts.validate_artifact(directory)]
    assert "A11" in codes


@pytest.mark.parametrize("shift, value, layer", F2_ACCEPTED_CASES)
def test_f2_shift_at_one_and_thirty_two_is_accepted(
    shift, value, layer, artifact_copy
):
    directory = artifact_copy()
    words = _read_kl(directory)
    _set_shift(words, layer, shift, value)
    directory = _write_kl(directory, words)

    failures = artifacts.validate_artifact(directory)
    codes = [_normalize(item) for item in failures]
    assert "A11" not in codes
    assert failures == []
