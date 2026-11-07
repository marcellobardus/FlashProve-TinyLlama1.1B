"""Shared helpers for the T1.6c hardening rule tests."""

import json
import shutil
import struct
from pathlib import Path

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import synth_artifact as synth
import test_hardening_artifact_factory as factory


DIMS = factory.DIMS


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory):
    return factory.build_small_artifact(tmp_path_factory.mktemp("rules-artifact"))


@pytest.fixture
def artifact_copy(small_artifact, tmp_path):
    destination = tmp_path / small_artifact.name
    shutil.copytree(small_artifact, destination)
    return destination


def retag(directory):
    directory = Path(directory)
    artifact_id = synth.file_sha256(directory / "manifest.json")
    if directory.name == artifact_id:
        return directory
    target = directory.parent / artifact_id
    directory.rename(target)
    return target


def manifest(directory):
    return synth.read_manifest(directory)


def entry(manifest_object, path):
    return synth.manifest_entry(manifest_object, path)


def write_manifest(directory, manifest_object):
    synth.write_manifest(directory, manifest_object)
    return retag(directory)


def params(directory):
    return json.loads(
        (Path(directory) / "params.json").read_text(encoding="ascii")
    )


def write_params(directory, params_object):
    synth.rewrite_json(directory, "params.json", params_object)
    return retag(directory)


def rewrite_file(directory, relative_path, **kwargs):
    kwargs.setdefault("update_manifest", True)
    synth.rewrite_file(directory, relative_path, **kwargs)
    return retag(directory)


def read_kl(directory):
    raw = (Path(directory) / "KL.bin").read_bytes()
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def write_kl(directory, words):
    data = struct.pack("<%dI" % len(words), *words)
    synth.rewrite_file(directory, "KL.bin", data=data, update_manifest=True)
    return retag(directory)


def failures(directory):
    return artifacts.validate_artifact(directory, dims=DIMS)


def codes(directory):
    return sorted({str(item[0]) for item in failures(directory)})


def load_code(directory, expected_id=None):
    with pytest.raises(artifacts.ArtifactRejected) as caught:
        artifacts.load_artifact(
            directory, expected_id=expected_id, dims=DIMS
        )
    return caught.value


def assert_only(directory, code):
    found = failures(directory)
    assert codes(directory) == [code], found
    rejected = load_code(directory)
    assert rejected.code == code, found
