"""A03, A06 and A08 sub-rules for paths, sizes and shapes."""

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

from test_hardening_support import (
    DIMS,
    artifact_copy,
    assert_only,
    entry,
    manifest,
    rewrite_file,
    small_artifact,
    write_manifest,
)


def _add_entry(directory, path):
    manifest_object = manifest(directory)
    manifest_object["files"].append(
        {
            "path": path,
            "dtype": "i16",
            "shape": [1],
            "bytes": 2,
            "sha256": "0" * 64,
        }
    )
    manifest_object["files"].sort(key=lambda item: item["path"])
    return write_manifest(directory, manifest_object)


@pytest.mark.parametrize(
    "bad_path",
    ["/outside.i16", "outside\\file.i16"],
    ids=["absolute", "backslash"],
)
def test_manifest_path_is_absolute_or_has_backslash_a03(artifact_copy, bad_path):
    directory = _add_entry(artifact_copy, bad_path)
    assert_only(directory, "A03")


def test_symlinked_listed_file_is_a03(artifact_copy):
    link = artifact_copy / "emb.i32"
    link.unlink()
    link.symlink_to(artifact_copy / "wh.i16")
    assert_only(artifact_copy, "A03")


def test_listed_bytes_differ_from_shape_but_match_file_is_a06(artifact_copy):
    path = "layers/00.wk.i16"
    size = (artifact_copy / path).stat().st_size
    directory = rewrite_file(artifact_copy, path, size=size - 2)
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A06"
        and failure_path == path
        and "product(shape)" in detail
        for code, failure_path, detail in found
    ), found
    assert_only(directory, "A06")


def test_wrong_dtype_is_a08(artifact_copy):
    path = "layers/00.wk.i16"
    manifest_object = manifest(artifact_copy)
    entry(manifest_object, path)["dtype"] = "i32"
    directory = write_manifest(artifact_copy, manifest_object)
    assert_only(directory, "A08")


@pytest.mark.parametrize(
    "raw",
    [b"{not json\n", b'{"emb": {"e": 11}}\n'],
    ids=["not-json", "not-canonical"],
)
def test_malformed_exponents_json_is_a08(artifact_copy, raw):
    directory = rewrite_file(artifact_copy, "exponents.json", data=raw)
    assert_only(directory, "A08")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A08"
        and failure_path == "exponents.json"
        and "not parseable" in detail
        for code, failure_path, detail in found
    ), found
