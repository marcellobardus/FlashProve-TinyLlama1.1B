"""Validation helpers: expected id reporting and one manifest read."""

import shutil
import struct
from pathlib import Path

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import synth_artifact as synth
import test_hardening_artifact_factory as factory


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory):
    return factory.build_small_artifact(tmp_path_factory.mktemp("validation-artifact"))


@pytest.fixture
def artifact_copy(small_artifact, tmp_path):
    destination = tmp_path / small_artifact.name
    shutil.copytree(small_artifact, destination)
    return destination


def _retag(directory):
    artifact_id = synth.file_sha256(directory / "manifest.json")
    if directory.name == artifact_id:
        return directory
    target = directory.parent / artifact_id
    directory.rename(target)
    return target


def test_expected_id_is_reported_with_another_failure(artifact_copy):
    directory = artifact_copy
    raw = bytearray((directory / "KL.bin").read_bytes())
    struct.pack_into("<I", raw, 12 * 4, 0)
    synth.rewrite_file(directory, "KL.bin", data=bytes(raw), update_manifest=True)
    directory = _retag(directory)

    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(code == "A13" for code, _, _ in failures), failures

    with pytest.raises(artifacts.ArtifactRejected) as caught:
        artifacts.load_artifact(
            directory, expected_id="0" * 64, dims=factory.DIMS
        )
    assert caught.value.code == "A07"
    assert caught.value.path == "manifest.json"
    assert "expected id" in caught.value.detail


def test_load_reads_manifest_once(artifact_copy, monkeypatch):
    directory = artifact_copy
    original = artifacts._read
    manifest_reads = []

    def tracked(path):
        if Path(path).name == "manifest.json":
            manifest_reads.append(path)
            if len(manifest_reads) > 1:
                raise AssertionError("manifest.json was read again")
        return original(path)

    monkeypatch.setattr(artifacts, "_read", tracked)
    artifacts.load_artifact(directory, dims=factory.DIMS)
    assert len(manifest_reads) == 1


def test_loader_source_does_not_mention_the_float_recipe():
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    assert "convert.tables" not in source
    assert "convert import tables" not in source
