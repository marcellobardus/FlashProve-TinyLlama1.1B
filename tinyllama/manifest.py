"""Helpers for the pinned checkpoint manifest."""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "docs" / "tinyllama" / "checkpoint_manifest.json"


class ManifestMismatch(ValueError):
    """A file's digest does not match the manifest."""


def load_manifest(path=None):
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


_MANIFEST = load_manifest()

REPO_ID = _MANIFEST["repo_id"]
REVISION = _MANIFEST["revision"]
MODEL_SHA256 = _MANIFEST["model_safetensors"]["sha256_lfs_etag"]
HEADER_JSON_SHA256 = _MANIFEST["model_safetensors"]["header_json_sha256"]


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def check_file_sha256(path, expected):
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ManifestMismatch(
            "sha256 mismatch for {}: expected {}, got {}".format(path, expected, actual)
        )
    return actual


def expected_tensors(manifest):
    return {
        name: (spec["dtype"], tuple(spec["shape"]))
        for name, spec in manifest["tensors"].items()
    }
