import struct
from pathlib import Path

import pytest

import tiny_artifact as ta
from tinyllama.artifacts import Dims
from tinyllama.proof import make_request as mr
from tinyllama.reference import run as reference_run
from tinyllama.reference.model import forward

# P0's 24 ids and its greedy next token (docs/tinyllama/ARITHMETIC_CONTRACT.md, PROVING_PLAN).
P0_IDS = [1, 529, 29989, 1792, 29989, 29958, 13, 5618, 338, 278, 7483, 310, 3444, 29973, 2, 29871, 13,
          29966, 29989, 465, 22137, 29989, 29958, 13]  # fmt: skip
P0_NEXT = 1576


def test_request_words_cover_r23_e24():
    words = mr.request_words(P0_IDS, [P0_NEXT], 64)
    assert len(words) == 66  # T + 2
    assert words[:24] == P0_IDS
    assert words[24] == P0_NEXT
    assert words[25:64] == [0] * 39  # padding to T
    assert words[-2:] == [23, 24]  # r = len(p) - 1, e = len(p) + len(a) - 1


def test_write_request_bytes_perms_and_exclusive(tmp_path):
    path = tmp_path / "p0.req"
    r, e = mr.write_request(str(path), P0_IDS, [P0_NEXT], 64)
    assert (r, e) == (23, 24)
    blob = path.read_bytes()
    assert len(blob) == 66 * 4
    assert list(struct.unpack("<66I", blob)) == mr.request_words(P0_IDS, [P0_NEXT], 64)
    assert (path.stat().st_mode & 0o777) == 0o600
    with pytest.raises(FileExistsError):  # O_EXCL: it will not overwrite
        mr.write_request(str(path), P0_IDS, [P0_NEXT], 64)


def test_p0_from_the_stored_prompt_set():
    _, ids = reference_run.load_prompt("P0")
    assert ids == P0_IDS
    words = mr.request_words(ids, [P0_NEXT], 64)
    assert words[-2:] == [23, 24]


def test_validation():
    with pytest.raises(mr.RequestError):
        mr.request_words([], [1], 64)  # empty prompt
    with pytest.raises(mr.RequestError):
        mr.request_words([1] * 40, [2] * 40, 64)  # prompt + reply above the window
    with pytest.raises(mr.RequestError):
        mr.request_words([1, -1], [], 64)  # a negative id
    with pytest.raises(mr.RequestError):
        mr.request_words([1], [1], 0)  # window not positive


def _artifact(tmp_path):
    dims = Dims(L=2, D=8, H=2, G=1, DH=4, F=6, V=32, T=16)
    return ta.build_artifact(tmp_path / "art", dims=dims), dims


def test_greedy_reply_uses_the_integer_reference(tmp_path):
    root, dims = _artifact(tmp_path)
    from tinyllama.artifacts import load_artifact

    artifact = load_artifact(root, dims=dims)
    prompt = [1, 2, 3]
    reply = mr.greedy_reply(artifact, prompt, 3, window=dims.T)
    assert len(reply) == 3
    assert all(isinstance(t, int) and 0 <= t < dims.V for t in reply)
    # Each reply token is exactly the reference's greedy choice after the tokens before it.
    sequence = list(prompt)
    for token in reply:
        assert token == int(forward(artifact, sequence, mode="prefix").next_token)
        sequence.append(token)
    # Deterministic.
    assert mr.greedy_reply(artifact, prompt, 3, window=dims.T) == reply


def test_greedy_reply_refuses_to_overflow_the_window(tmp_path):
    root, dims = _artifact(tmp_path)
    from tinyllama.artifacts import load_artifact

    artifact = load_artifact(root, dims=dims)
    with pytest.raises(mr.RequestError):
        mr.greedy_reply(artifact, list(range(dims.T)), 1, window=dims.T)


def test_main_writes_p0(tmp_path):
    out = tmp_path / "p0.req"
    rc = mr.main(["--prompt", "P0", "--reply-ids", "1576", "--window", "64", "--out", str(out)])
    assert rc == 0
    words = list(struct.unpack("<66I", out.read_bytes()))
    assert words[:24] == P0_IDS and words[24] == P0_NEXT and words[-2:] == [23, 24]


def test_main_needs_exactly_one_reply_source(tmp_path):
    out = tmp_path / "x.req"
    assert mr.main(["--prompt", "P0", "--window", "64", "--out", str(out)]) == 2  # neither reply source
    assert mr.main(["--prompt", "P0", "--reply-ids", "1", "--greedy", "1", "--window", "64", "--out", str(out)]) == 2
