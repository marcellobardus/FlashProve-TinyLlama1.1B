"""Hardening tests for non-regular and unreadable listed files (T1.6a)."""

import os
import shutil

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import synth_artifact as synth
import test_hardening_artifact_factory as factory

PATH = "layers/00.wq.i16"


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


def _assert_rejected(directory):
    failures = artifacts.validate_artifact(directory, dims=factory.DIMS)
    assert any(
        code in ("A04", "A06") and failure_path == PATH
        for code, failure_path, _ in failures
    ), failures
    with pytest.raises(artifacts.ArtifactRejected) as caught:
        artifacts.load_artifact(directory, dims=factory.DIMS)
    assert caught.value.code in ("A04", "A06")
    assert caught.value.path == PATH


def test_listed_directory_is_rejected(artifact_copy):
    directory = artifact_copy()
    (directory / PATH).unlink()
    (directory / PATH).mkdir()
    _assert_rejected(directory)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="os.mkfifo is absent")
def test_listed_fifo_is_rejected_without_hanging(artifact_copy):
    directory = artifact_copy()
    (directory / PATH).unlink()
    os.mkfifo(directory / PATH)
    _assert_rejected(directory)


@pytest.mark.skipif(
    os.geteuid() == 0, reason="root can read a file with mode 0"
)
def test_unreadable_listed_file_is_rejected(artifact_copy):
    directory = artifact_copy()
    size = (directory / PATH).stat().st_size
    synth.rewrite_file(directory, PATH, size=size, update_manifest=True)
    os.chmod(directory / PATH, 0)
    try:
        _assert_rejected(directory)
    finally:
        os.chmod(directory / PATH, 0o600)
