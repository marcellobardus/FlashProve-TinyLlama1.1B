import json
import math

import synth_artifact as synth


def _manifest(valid_artifact):
    return json.loads((valid_artifact / "manifest.json").read_text(encoding="ascii"))


def _params(valid_artifact):
    return json.loads((valid_artifact / "params.json").read_text(encoding="ascii"))


def test_manifest_is_canonical_and_lists_the_directory(valid_artifact):
    raw = (valid_artifact / "manifest.json").read_bytes()
    manifest = _manifest(valid_artifact)
    assert synth.canonical_json_bytes(manifest) == raw

    listed = [entry["path"] for entry in manifest["files"]]
    assert listed == sorted(listed)
    actual = sorted(
        path.relative_to(valid_artifact).as_posix()
        for path in valid_artifact.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    assert listed == actual
    assert set(listed) == set(synth.expected_specs())


def test_sizes_equal_shape_product_times_item_size(valid_artifact):
    manifest = _manifest(valid_artifact)
    specs = synth.expected_specs()
    for entry in manifest["files"]:
        dtype, shape = specs[entry["path"]]
        assert entry["dtype"] == dtype
        assert entry["shape"] == shape
        if dtype == "json":
            assert entry["bytes"] == (valid_artifact / entry["path"]).stat().st_size
        else:
            assert entry["bytes"] == math.prod(shape) * synth.ITEMSIZE[dtype]


def test_params_eps_follow_the_formula(valid_artifact):
    params = _params(valid_artifact)
    e_res = params["e_res"]
    assert 3 <= e_res <= 14
    expected = (4096 * 4**e_res + 100000) // 200000
    assert params["eps1"] == [expected] * synth.L
    assert params["eps2"] == [expected] * synth.L
    assert params["epsF"] == expected
    assert expected == synth.expected_eps(e_res)


def test_params_digests_and_table_sha256_match_the_contract(valid_artifact):
    params = _params(valid_artifact)
    tt = synth.test_tables_module()
    for name, words in tt.CONTRACT_DIGEST_WORDS.items():
        assert tuple(params["digests"][name]) == words
        assert tt._sha256_of_words(tt._hashed_words(name)) == tt.CONTRACT_SHA256[name]

    for path, digest in params["tables_sha256"].items():
        assert synth.file_sha256(valid_artifact / path) == digest
