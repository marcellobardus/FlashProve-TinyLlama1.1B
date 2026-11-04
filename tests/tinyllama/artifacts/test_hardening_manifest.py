"""Hardening tests for manifest structure and header fields (T1.6a)."""

import json
import shutil

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import synth_artifact as synth
import test_hardening_artifact_factory as factory


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory):
    return factory.build_small_artifact(tmp_path_factory.mktemp("small-artifact"))


@pytest.fixture
def artifact_copy(small_artifact, tmp_path):
    def make():
        destination = tmp_path / "case"
        shutil.copytree(small_artifact, destination)
        return destination

    return make


def _manifest(directory):
    return synth.read_manifest(directory)


def _write_manifest(directory, manifest):
    return synth.write_manifest(directory, manifest)


def _entry(manifest, path):
    return synth.manifest_entry(manifest, path)


def _codes(directory):
    return [str(item[0])[:3] for item in artifacts.validate_artifact(directory, dims=factory.DIMS)]


def _load_code(directory):
    with pytest.raises(artifacts.ArtifactRejected) as caught:
        artifacts.load_artifact(directory, dims=factory.DIMS)
    return caught.value.code


@pytest.mark.parametrize("path", ["params.json", "emb.i32"])
def test_expected_file_not_listed_is_a04(artifact_copy, path):
    directory = artifact_copy()
    manifest = _manifest(directory)
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != path
    ]
    _write_manifest(directory, manifest)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(
        code == "A04" and failure_path == path
        for code, failure_path, _ in failures
    ), failures
    assert _load_code(directory) == "A04"


@pytest.mark.parametrize("path", ["params.json", "tables/silu_ts.i32"])
def test_absent_and_unlisted_expected_file_is_a04(artifact_copy, path):
    directory = artifact_copy()
    (directory / path).unlink()
    manifest = _manifest(directory)
    manifest["files"] = [
        entry for entry in manifest["files"] if entry["path"] != path
    ]
    _write_manifest(directory, manifest)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(
        code == "A04" and failure_path == path
        for code, failure_path, _ in failures
    ), failures
    assert _load_code(directory) == "A04"


def test_duplicate_files_entry_is_a01(artifact_copy):
    directory = artifact_copy()
    manifest = _manifest(directory)
    manifest["files"].append(dict(manifest["files"][-1]))
    _write_manifest(directory, manifest)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(code == "A01" for code, _, _ in failures), failures
    assert _load_code(directory) == "A01"


def test_unsorted_files_entry_is_a01(artifact_copy):
    directory = artifact_copy()
    manifest = _manifest(directory)
    manifest["files"][0], manifest["files"][1] = (
        manifest["files"][1],
        manifest["files"][0],
    )
    _write_manifest(directory, manifest)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(code == "A01" for code, _, _ in failures), failures
    assert _load_code(directory) == "A01"


@pytest.mark.parametrize("value", [None, {}])
def test_files_not_a_list_is_a01(artifact_copy, value):
    directory = artifact_copy()
    manifest = _manifest(directory)
    manifest["files"] = value
    _write_manifest(directory, manifest)

    codes = _codes(directory)
    assert "A01" in codes
    assert "A02" not in codes
    assert _load_code(directory) == "A01"


@pytest.mark.parametrize("path", ["KL.bin", "wh.i16"])
def test_float_shape_elements_are_refused(artifact_copy, path):
    directory = artifact_copy()
    manifest = _manifest(directory)
    entry = _entry(manifest, path)
    entry["shape"] = [float(dim) for dim in entry["shape"]]
    _write_manifest(directory, manifest)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(
        code == "A08" and failure_path == path
        for code, failure_path, _ in failures
    ), failures
    assert _load_code(directory) == "A08"


@pytest.mark.parametrize("value", [1.0, True])
def test_non_integer_format_version_is_a02(artifact_copy, value):
    directory = artifact_copy()
    manifest = _manifest(directory)
    manifest["format_version"] = value
    _write_manifest(directory, manifest)

    codes = _codes(directory)
    assert "A02" in codes
    assert _load_code(directory) == "A02"


def test_two_dots_inside_a_name_is_not_a03(artifact_copy):
    directory = artifact_copy()
    path = "a..b.bin"
    (directory / path).write_bytes(b"\x00\x00")
    manifest = _manifest(directory)
    manifest["files"].append(
        {
            "path": path,
            "dtype": "i16",
            "shape": [1],
            "bytes": 2,
            "sha256": synth.file_sha256(directory / path),
        }
    )
    manifest["files"].sort(key=lambda entry: entry["path"])
    _write_manifest(directory, manifest)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    codes = [code for code, _, _ in failures]
    assert "A03" not in codes, failures
    assert any(
        code == "A08" and failure_path == path
        for code, failure_path, _ in failures
    ), failures
    assert _load_code(directory) == "A08"


@pytest.mark.parametrize(
    "alias",
    ["rope_c", "rope_c.i32", "tables/rope_c"],
)
def test_tables_sha256_alias_key_is_refused(artifact_copy, alias):
    directory = artifact_copy()
    path = "tables/rope_c.i32"
    params = json.loads((directory / "params.json").read_text(encoding="ascii"))
    params["tables_sha256"][alias] = params["tables_sha256"].pop(path)
    synth.rewrite_json(directory, "params.json", params, update_manifest=True)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(
        code == "A16"
        and failure_path == "params.json"
        and "rope_c" in detail
        for code, failure_path, detail in failures
    ), failures
    assert _load_code(directory) == "A16"
