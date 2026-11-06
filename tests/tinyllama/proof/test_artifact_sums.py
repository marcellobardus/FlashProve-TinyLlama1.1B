"""Lock the wsums.txt format that artifact_sums.py writes (PROVING_PLAN T4.2).

A tiny synthetic TL-INT-1 artifact is enough: the real checkpoint is never
downloaded here. The strong check reads each tensor's bytes straight from the
artifact files and hashes them, independently of the script's NumPy path, so a
line matches only when the script hashes exactly the stored integers in file
order, the two's-complement byte layout weight_sums uses.
"""

import hashlib

import pytest

import tiny_artifact as ta
from tinyllama.artifacts import Dims, load_artifact
from tinyllama.proof import artifact_sums as asums

# Several layers so the byte sort of names is exercised (blk.10 before blk.2).
DIMS = Dims(L=12, D=8, H=2, G=1, DH=4, F=6, V=32, T=16)
E_RES = 11  # tiny_artifact's params use e_res 11

# Which artifact file holds each committed tensor's integers, with its item size.
LAYER_FILES = {
    "attn_q": "wq",
    "attn_k": "wk",
    "attn_v": "wv",
    "attn_output": "wo",
    "ffn_gate": "wg",
    "ffn_up": "wu",
    "ffn_down": "wd",
}


def _file_for(name):
    if name == "token_embd.weight":
        return "emb.i32", 4
    if name == "output.weight":
        return "wh.i16", 2
    _, layer, kind, _ = name.split(".")
    return "layers/%02d.%s.i16" % (int(layer), LAYER_FILES[kind]), 2


def _exponents():
    """A full exponents.json keyed by the HuggingFace names, as the converter
    writes it. The embedding's exponent is e_res; the others are distinct so the
    test sees they come from exponents.json and not from e_res."""
    values = {
        "model.embed_tokens.weight": {"e": E_RES, "mx": 0.0},
        "lm_head.weight": {"e": 7, "mx": 0.0},
    }
    suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    for layer in range(DIMS.L):
        for index, suffix in enumerate(suffixes):
            key = "model.layers.%d.%s.weight" % (layer, suffix)
            values[key] = {"e": (layer + index) % 17, "mx": 0.0}
    return values


@pytest.fixture
def artifact_root(tmp_path):
    return ta.build_artifact(tmp_path / "art", dims=DIMS, exponents=_exponents())


def _parse(lines):
    assert lines[-1].startswith("END ")
    assert lines[-1] == "END %d\n" % (len(lines) - 1)
    rows = {}
    for line in lines[:-1]:
        name, elements, exponent, bits, digest = line.rstrip("\n").split(" ")
        rows[name] = (int(elements), int(exponent), int(bits), digest)
    return rows


def _expected_names():
    names = {"token_embd.weight", "output.weight"}
    for layer in range(DIMS.L):
        for kind in LAYER_FILES:
            names.add("blk.%d.%s.weight" % (layer, kind))
    return names


def test_lines_cover_every_committed_tensor(artifact_root):
    artifact = load_artifact(artifact_root, dims=DIMS)
    lines = asums.build_sums(artifact)
    assert len(lines) == 2 + 7 * DIMS.L
    rows = _parse(lines + ["END %d\n" % len(lines)])
    assert set(rows) == _expected_names()


def test_lines_are_sorted_by_name_as_bytes(artifact_root):
    artifact = load_artifact(artifact_root, dims=DIMS)
    lines = asums.build_sums(artifact)
    names = [line.split(" ", 1)[0] for line in lines]
    assert names == sorted(names, key=str.encode)
    # The byte sort puts blk.10 and blk.11 before blk.2, as weight_sums documents.
    assert names.index("blk.10.attn_k.weight") < names.index("blk.2.attn_k.weight")


def test_hash_is_over_the_stored_bytes_in_file_order(artifact_root):
    """The script's hash for every tensor equals the SHA-256 of that tensor's
    file, the committed integers in file order. This is the line-for-line
    comparison that T4.2 will make against bin/weight_sums."""
    artifact = load_artifact(artifact_root, dims=DIMS)
    lines = asums.build_sums(artifact)
    rows = _parse(lines + ["END %d\n" % len(lines)])
    for name, (elements, _exponent, _bits, digest) in rows.items():
        relative, item_bytes = _file_for(name)
        raw = (artifact_root / relative).read_bytes()
        assert len(raw) == elements * item_bytes
        assert digest == hashlib.sha256(raw).hexdigest()


def test_exponents_and_bits_come_from_the_artifact(artifact_root):
    artifact = load_artifact(artifact_root, dims=DIMS)
    exponents = _exponents()
    rows = _parse(asums.build_sums(artifact) + ["END %d\n" % (2 + 7 * DIMS.L)])

    # The embedding uses e_res and the four-byte layout; the head uses its own.
    assert rows["token_embd.weight"][1] == E_RES
    assert rows["token_embd.weight"][0] == DIMS.V * DIMS.D
    assert rows["output.weight"][1] == exponents["lm_head.weight"]["e"]

    # A layer matrix takes its exponent from exponents.json, not from e_res.
    name = "blk.3.ffn_gate.weight"
    assert rows[name][1] == exponents["model.layers.3.mlp.gate_proj.weight"]["e"]

    # bits is the least width that holds every integer, as weight_sums defines it.
    wq = artifact.matrix(0, "wq")
    low, high = int(wq.min()), int(wq.max())
    bits = 1
    while not -(2 ** (bits - 1)) <= low <= high < 2 ** (bits - 1):
        bits += 1
    assert rows["blk.0.attn_q.weight"][2] == bits


def test_main_writes_an_owner_only_file_that_is_not_overwritten(tmp_path, artifact_root):
    out = tmp_path / "wsums.artifact.txt"
    dims = ",".join(str(value) for value in (DIMS.L, DIMS.D, DIMS.H, DIMS.G, DIMS.DH, DIMS.F, DIMS.V, DIMS.T))
    rc = asums.main(["--artifact", str(artifact_root), "--out", str(out), "--eres", "11", "--dims-test", dims])
    assert rc == 0

    lines = out.read_text(encoding="ascii").splitlines(keepends=True)
    assert lines[-1] == "END %d\n" % (len(lines) - 1)
    assert (out.stat().st_mode & 0o777) == 0o600

    # Every written line still matches the stored bytes.
    rows = _parse(lines)
    for name, (_elements, _exponent, _bits, digest) in rows.items():
        relative, _ = _file_for(name)
        raw = (artifact_root / relative).read_bytes()
        assert digest == hashlib.sha256(raw).hexdigest()

    # O_EXCL: a second run does not overwrite the file.
    assert asums.main(["--artifact", str(artifact_root), "--out", str(out), "--eres", "11", "--dims-test", dims]) == 2


def test_eres_mismatch_is_refused(artifact_root):
    artifact = load_artifact(artifact_root, dims=DIMS)
    with pytest.raises(asums.ArtifactSumsError):
        asums.build_sums(artifact, eres=10)


def test_missing_exponent_is_refused(tmp_path):
    root = ta.build_artifact(tmp_path / "bare", dims=DIMS)  # exponents.json is empty
    artifact = load_artifact(root, dims=DIMS)
    with pytest.raises(asums.ArtifactSumsError) as error:
        asums.build_sums(artifact)
    assert "token_embd.weight" in str(error.value)
