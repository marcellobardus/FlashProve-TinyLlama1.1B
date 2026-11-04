import copy
import hashlib
import json
import shutil
import struct
from functools import lru_cache

import numpy as np
import pytest

from tinyllama import contract
from tinyllama.artifacts import (
    ArtifactRejected,
    Dims,
    load_artifact,
    validate_artifact,
    write_artifact,
)
from tinyllama.convert import tables


DIMS = Dims(L=2, D=128, H=2, G=1, DH=64, F=192, V=50, T=1024)
PROVENANCE = {
    "source": {
        "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "model_safetensors_sha256": "6e6001da" + "0" * 56,
        "flashprove_base_commit": "9e2bdd8f" + "0" * 56,
    },
    "converter": {
        "git_commit": "deadbeef",
        "command": "python -m tinyllama.convert --synthetic",
        "python": "3.13.5",
        "numpy": "2.5.4",
    },
    "calibration": {
        "set": "TL-CAL-1",
        "fit_sha256": "ab" * 32,
        "heldout_sha256": "cd" * 32,
    },
}


def _canonical(obj):
    return (
        json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


@lru_cache(maxsize=None)
def _table_arrays():
    cosines, sines = tables.rope_tables(
        DIMS.T, DIMS.DH, contract.CS, contract.THETA0
    )
    exp = tables.softmax_table(contract.EXP_S, contract.RX)
    silu = tables.silu_table(contract.EG, contract.SMAX)
    return {
        "rope_c": np.asarray(cosines, dtype=np.int32).reshape(
            DIMS.T, DIMS.DH // 2
        ),
        "rope_s": np.asarray(sines, dtype=np.int32).reshape(
            DIMS.T, DIMS.DH // 2
        ),
        "softmax_ex": np.asarray(exp, dtype=np.int32),
        "silu_ts": np.asarray(silu, dtype=np.int32),
    }, (cosines, sines, exp, silu)


def _table_words():
    _, (cosines, sines, exp, silu) = _table_arrays()
    return {
        "trope": tables.trope_words(cosines, sines),
        "texp": tables.texp_words(exp),
        "tsilu": tables.tsilu_words(silu),
    }


def _table_shas():
    arrays, _ = _table_arrays()
    return {
        "tables/rope_c.i32": hashlib.sha256(
            arrays["rope_c"].astype("<i4").tobytes()
        ).hexdigest(),
        "tables/rope_s.i32": hashlib.sha256(
            arrays["rope_s"].astype("<i4").tobytes()
        ).hexdigest(),
        "tables/softmax_ex.i32": hashlib.sha256(
            arrays["softmax_ex"].astype("<i4").tobytes()
        ).hexdigest(),
        "tables/silu_ts.i32": hashlib.sha256(
            arrays["silu_ts"].astype("<i4").tobytes()
        ).hexdigest(),
    }


def _params():
    arrays, _ = _table_arrays()
    del arrays
    digests = {
        name: tables.table_digest(words) for name, words in _table_words().items()
    }
    params = {
        "contract": contract.CONTRACT_ID,
        "L": DIMS.L,
        "d": DIMS.D,
        "H": DIMS.H,
        "G": DIMS.G,
        "dh": DIMS.DH,
        "f": DIMS.F,
        "V": DIMS.V,
        "T": DIMS.T,
        "NE": contract.NE,
        "lt": contract.LT,
        "cs": contract.CS,
        "theta0": contract.THETA0,
        "S": contract.S,
        "rx": contract.RX,
        "eg": contract.EG,
        "smax": contract.SMAX,
        "cw": contract.CW,
        "weight_bound": contract.WEIGHT_BOUND,
        "embedding_bound": contract.EMB_BOUND,
        "u0": contract.U0,
        "c0_bits": contract.C0_BITS,
        "cq_bits": contract.CQ_BITS,
        "pad_id": contract.PAD_ID,
        "gcl": contract.GCL,
        "range_conditions": {
            str(key): list(value)
            for key, value in contract.RANGE_CONDITIONS.items()
        },
        "e_res": 11,
        "eps1": [85899] * DIMS.L,
        "eps2": [85899] * DIMS.L,
        "epsF": 85899,
        "MF": 37,
        "KF": 32,
        "s": 20,
        "digests": digests,
        "tables_sha256": _table_shas(),
    }
    return params


def _matrices():
    rng = np.random.default_rng(20261104)
    matrices = {}
    matrices["wh.i16"] = rng.integers(
        -4095, 4096, size=(DIMS.V, DIMS.D), dtype=np.int16
    )
    matrices["wh.i16"].flat[0] = -4095
    matrices["wh.i16"].flat[1] = 4095
    for layer in range(DIMS.L):
        ll = "%02d" % layer
        shapes = {
            "wq": (DIMS.H * DIMS.DH, DIMS.D),
            "wk": (DIMS.G * DIMS.DH, DIMS.D),
            "wv": (DIMS.G * DIMS.DH, DIMS.D),
            "wo": (DIMS.D, DIMS.H * DIMS.DH),
            "wg": (DIMS.F, DIMS.D),
            "wu": (DIMS.F, DIMS.D),
            "wd": (DIMS.D, DIMS.F),
        }
        for which, shape in shapes.items():
            matrices["layers/%s.%s.i16" % (ll, which)] = rng.integers(
                -4095, 4096, size=shape, dtype=np.int16
            )
    matrices["layers/00.wq.i16"].flat[0] = -4095
    matrices["layers/00.wq.i16"].flat[1] = 4095
    return matrices


def _kl_words():
    words = np.zeros((DIMS.L, 13), dtype=np.uint32)
    for layer in range(DIMS.L):
        words[layer] = [20, 20, 20, 20, 20, 20, 20, 37, 32, 37, 32, 8, 1]
    return words


def _valid_data():
    arrays, _ = _table_arrays()
    return {
        "matrices": _matrices(),
        "emb": np.random.default_rng(7).integers(
            -(2**21 - 1), 2**21, size=(DIMS.V, DIMS.D), dtype=np.int32
        ),
        "kl_words": _kl_words(),
        "params": _params(),
        "tables": {name: array.copy() for name, array in arrays.items()},
        "provenance": copy.deepcopy(PROVENANCE),
    }


@pytest.fixture
def artifact(tmp_path):
    data = _valid_data()
    data["emb"].flat[0] = 2**21 - 1
    data["emb"].flat[1] = -(2**21 - 1)
    out = tmp_path / "artifacts"
    exponents = {"emb": {"e": 11, "mx": 0}}
    artifact_id = write_artifact(
        out,
        data["matrices"],
        data["emb"],
        data["kl_words"],
        data["params"],
        data["tables"],
        data["provenance"],
        dims=DIMS,
        exponents=exponents,
    )
    return {
        "id": artifact_id,
        "dir": out / artifact_id,
        "out": out,
        "data": data,
        "exponents": exponents,
    }


def _copy(artifact, tmp_path, name="mutated"):
    destination = tmp_path / name
    shutil.copytree(artifact["dir"], destination)
    return destination


def _manifest(path):
    return json.loads((path / "manifest.json").read_text(encoding="ascii"))


def _write_manifest(path, manifest):
    (path / "manifest.json").write_bytes(_canonical(manifest))


def _entry(manifest, relative_path):
    for entry in manifest["files"]:
        if entry["path"] == relative_path:
            return entry
    raise KeyError(relative_path)


def _rewrite(path, relative_path, raw, *, update_bytes=True, shape=None, dtype=None):
    (path / relative_path).write_bytes(raw)
    manifest = _manifest(path)
    entry = _entry(manifest, relative_path)
    entry["sha256"] = hashlib.sha256(raw).hexdigest()
    if update_bytes:
        entry["bytes"] = len(raw)
    if shape is not None:
        entry["shape"] = list(shape)
    if dtype is not None:
        entry["dtype"] = dtype
    _write_manifest(path, manifest)


def _rewrite_json(path, relative_path, obj):
    _rewrite(path, relative_path, _canonical(obj))


def _rewrite_kl(path, words):
    _rewrite(path, "KL.bin", np.asarray(words, dtype="<u4").tobytes())


def _assert_code(path, code):
    failures = validate_artifact(path, dims=DIMS)
    codes = [failure[0] for failure in failures]
    assert code in codes, failures
    with pytest.raises(ArtifactRejected) as caught:
        load_artifact(path, dims=DIMS)
    assert caught.value.code == code, failures


def test_round_trip(artifact):
    loaded = load_artifact(
        artifact["dir"], expected_id=artifact["id"], dims=DIMS
    )
    assert validate_artifact(artifact["dir"], dims=DIMS) == []
    assert loaded.id == artifact["id"]
    assert loaded.id == artifact["dir"].name
    raw_manifest = (artifact["dir"] / "manifest.json").read_bytes()
    assert loaded.id == hashlib.sha256(raw_manifest).hexdigest()
    assert _canonical(loaded.manifest) == raw_manifest
    paths = [entry["path"] for entry in loaded.manifest["files"]]
    assert paths == sorted(paths)

    for path, array in artifact["data"]["matrices"].items():
        np.testing.assert_array_equal(np.asarray(loaded.matrices[path]), array)
    np.testing.assert_array_equal(np.asarray(loaded.emb), artifact["data"]["emb"])
    np.testing.assert_array_equal(loaded.wh, artifact["data"]["matrices"]["wh.i16"])
    np.testing.assert_array_equal(np.asarray(loaded.kl), artifact["data"]["kl_words"])
    for name, array in artifact["data"]["tables"].items():
        np.testing.assert_array_equal(np.asarray(loaded.table(name)), array)
        np.testing.assert_array_equal(
            np.asarray(loaded.matrix(0, "wq")),
            artifact["data"]["matrices"]["layers/00.wq.i16"],
        )

    with pytest.raises(ValueError):
        loaded.wh[0, 0] = 0
    with pytest.raises(ValueError):
        loaded.kl[0, 0] = 0
    with pytest.raises(ValueError):
        loaded.table("rope_c")[0, 0] = 0

    raw = (artifact["dir"] / "wh.i16").read_bytes()
    assert struct.unpack("<h", raw[:2])[0] == int(
        artifact["data"]["matrices"]["wh.i16"].flat[0]
    )


def test_write_again_refuses_and_leaves_first(artifact):
    manifest_before = (artifact["dir"] / "manifest.json").read_bytes()
    data = artifact["data"]
    with pytest.raises(FileExistsError):
        write_artifact(
            artifact["out"],
            data["matrices"],
            data["emb"],
            data["kl_words"],
            data["params"],
            data["tables"],
            data["provenance"],
            dims=DIMS,
            exponents=artifact["exponents"],
        )
    assert (artifact["dir"] / "manifest.json").read_bytes() == manifest_before


def test_expected_id_mismatch(artifact):
    with pytest.raises(ArtifactRejected) as caught:
        load_artifact(artifact["dir"], expected_id="0" * 64, dims=DIMS)
    assert caught.value.code == "A07"
    assert caught.value.path == "manifest.json"


def test_writer_refuses_missing_params_field(artifact, tmp_path):
    params = copy.deepcopy(artifact["data"]["params"])
    del params["e_res"]
    with pytest.raises(ValueError):
        write_artifact(
            tmp_path / "bad",
            artifact["data"]["matrices"],
            artifact["data"]["emb"],
            artifact["data"]["kl_words"],
            params,
            artifact["data"]["tables"],
            artifact["data"]["provenance"],
            dims=DIMS,
        )


# A01 -----------------------------------------------------------------------


def test_a01_missing_manifest(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    (path / "manifest.json").unlink()
    _assert_code(path, "A01")


def test_a01_non_utf8(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    (path / "manifest.json").write_bytes(b"\xff\xfe\x00")
    _assert_code(path, "A01")


def test_a01_duplicate_key(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    (path / "manifest.json").write_bytes(b'{"format":"a","format":"b"}\n')
    _assert_code(path, "A01")


def test_a01_non_canonical_whitespace(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    manifest = _manifest(path)
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="ascii"
    )
    _assert_code(path, "A01")


# A02 -----------------------------------------------------------------------


def test_a02_wrong_format_version(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    manifest = _manifest(path)
    manifest["format_version"] = 2
    _write_manifest(path, manifest)
    _assert_code(path, "A02")


# A03 -----------------------------------------------------------------------


@pytest.mark.parametrize("bad_path", ["../emb.i32", "/emb.i32", "emb\\i32"])
def test_a03_bad_path(artifact, tmp_path, bad_path):
    path = _copy(artifact, tmp_path)
    manifest = _manifest(path)
    _entry(manifest, "emb.i32")["path"] = bad_path
    _write_manifest(path, manifest)
    _assert_code(path, "A03")


def test_a03_symlinked_file(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    target = path / "emb.i32"
    link = path / "wh.i16"
    link.unlink()
    link.symlink_to(target)
    _assert_code(path, "A03")


# A04 and A05 ---------------------------------------------------------------


def test_a04_missing_file(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    (path / "layers/00.wq.i16").unlink()
    _assert_code(path, "A04")


def test_a05_extra_file(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    (path / "extra.bin").write_bytes(b"extra")
    _assert_code(path, "A05")


# A06 and A07 ---------------------------------------------------------------


def test_a06_truncated_file(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    target = path / "layers/00.wq.i16"
    raw = target.read_bytes()[:-1]
    _rewrite(path, "layers/00.wq.i16", raw, update_bytes=False)
    _assert_code(path, "A06")


def test_a06_trailing_byte(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    target = path / "layers/00.wq.i16"
    raw = target.read_bytes() + b"\0"
    _rewrite(path, "layers/00.wq.i16", raw, update_bytes=False)
    _assert_code(path, "A06")


def test_a07_flipped_byte(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    target = path / "wh.i16"
    raw = bytearray(target.read_bytes())
    raw[0] ^= 1
    target.write_bytes(bytes(raw))
    _assert_code(path, "A07")


# A08 -----------------------------------------------------------------------


def test_a08_wrong_dtype(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    manifest = _manifest(path)
    _entry(manifest, "layers/00.wq.i16")["dtype"] = "i32"
    _write_manifest(path, manifest)
    _assert_code(path, "A08")


def test_a08_wrong_shape_with_same_bytes(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    manifest = _manifest(path)
    _entry(manifest, "wh.i16")["shape"] = [DIMS.V * DIMS.D]
    _write_manifest(path, manifest)
    _assert_code(path, "A08")


# A09 -----------------------------------------------------------------------


def test_a09_matrix_out_of_range(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    array = np.asarray(artifact["data"]["matrices"]["wh.i16"]).copy()
    array.flat[0] = 4096
    array.flat[1] = -4096
    _rewrite(path, "wh.i16", array.astype("<i2").tobytes())
    _assert_code(path, "A09")


def test_a09_embedding_out_of_range(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    array = np.asarray(artifact["data"]["emb"]).copy()
    array.flat[0] = 2**21
    array.flat[1] = -(2**21)
    _rewrite(path, "emb.i32", array.astype("<i4").tobytes())
    _assert_code(path, "A09")


# A10 to A14 ----------------------------------------------------------------


def test_a10_kl_extra_word(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    words = np.asarray(artifact["data"]["kl_words"])
    raw = words.astype("<u4").tobytes() + struct.pack("<I", 1)
    _rewrite(path, "KL.bin", raw)
    _assert_code(path, "A10")


@pytest.mark.parametrize(
    "mutation",
    [
        ("shift0", lambda words: words.__setitem__((0, 0), 0)),
        ("shift33", lambda words: words.__setitem__((0, 1), 33)),
        (
            "kq15kk14",
            lambda words: words.__setitem__((0, slice(0, 2)), [15, 14]),
        ),
    ],
)
def test_a11_bad_shift(artifact, tmp_path, mutation):
    path = _copy(artifact, tmp_path)
    words = np.asarray(artifact["data"]["kl_words"]).copy()
    mutation[1](words)
    _rewrite_kl(path, words)
    _assert_code(path, "A11")


@pytest.mark.parametrize(
    "values",
    [
        (23, 32, 37, 32),
        (37, 33, 37, 32),
        (37, 32, 47, 32),
        (37, 32, 24, 32),
    ],
)
def test_a12_bad_layer_norm(artifact, tmp_path, values):
    path = _copy(artifact, tmp_path)
    words = np.asarray(artifact["data"]["kl_words"]).copy()
    words[0, 7], words[0, 8], words[0, 9], words[0, 10] = values
    _rewrite_kl(path, words)
    _assert_code(path, "A12")


def test_a12_bad_mf(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    params = copy.deepcopy(artifact["data"]["params"])
    params["MF"] = 48
    _rewrite_json(path, "params.json", params)
    _assert_code(path, "A12")


def test_a13_bad_rp(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    words = np.asarray(artifact["data"]["kl_words"]).copy()
    words[0, 12] = 0
    _rewrite_kl(path, words)
    _assert_code(path, "A13")


@pytest.mark.parametrize("sh_values", [(8, 9), (3, 3), (13, 13)])
def test_a14_bad_sh(artifact, tmp_path, sh_values):
    path = _copy(artifact, tmp_path)
    words = np.asarray(artifact["data"]["kl_words"]).copy()
    words[:, 11] = sh_values
    _rewrite_kl(path, words)
    _assert_code(path, "A14")


# A15 and A16 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation",
    [
        lambda params: params.__setitem__("d", DIMS.D + 1),
        lambda params: params.__setitem__("e_res", 2),
        lambda params: params["eps1"].__setitem__(0, 85900),
        lambda params: params.__setitem__("s", 64),
    ],
)
def test_a15_bad_params(artifact, tmp_path, mutation):
    path = _copy(artifact, tmp_path)
    params = copy.deepcopy(artifact["data"]["params"])
    mutation(params)
    _rewrite_json(path, "params.json", params)
    _assert_code(path, "A15")


def test_a15_non_canonical_constant_name(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    params = copy.deepcopy(artifact["data"]["params"])
    params["D"] = params.pop("d")
    _rewrite_json(path, "params.json", params)
    _assert_code(path, "A15")
    failures = validate_artifact(path, dims=DIMS)
    assert (
        "A15",
        "params.json",
        "non canonical constant name",
    ) in failures


def test_a16_changed_table_entry(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    array = np.asarray(artifact["data"]["tables"]["rope_c"]).copy()
    array.flat[0] = 4095
    _rewrite(path, "tables/rope_c.i32", array.astype("<i4").tobytes())
    _assert_code(path, "A16")


def test_a16_wrong_digest(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    params = copy.deepcopy(artifact["data"]["params"])
    params["digests"]["trope"][0] += 1
    _rewrite_json(path, "params.json", params)
    _assert_code(path, "A16")


def test_a16_wrong_tables_sha256(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    params = copy.deepcopy(artifact["data"]["params"])
    key = sorted(params["tables_sha256"])[0]
    params["tables_sha256"][key] = "0" * 64
    _rewrite_json(path, "params.json", params)
    _assert_code(path, "A16")


def test_multiple_failures_are_reported(artifact, tmp_path):
    path = _copy(artifact, tmp_path)
    words = np.asarray(artifact["data"]["kl_words"]).copy()
    words[0, 12] = 0
    _rewrite_kl(path, words)
    target = path / "wh.i16"
    raw = bytearray(target.read_bytes())
    raw[0] ^= 1
    target.write_bytes(bytes(raw))

    failures = validate_artifact(path, dims=DIMS)
    codes = {failure[0] for failure in failures}
    assert "A07" in codes
    assert "A13" in codes
    with pytest.raises(ArtifactRejected) as caught:
        load_artifact(path, dims=DIMS)
    assert caught.value.code == "A07"
