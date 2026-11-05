"""A01 cases for manifest.json itself."""

import json

from test_hardening_support import (
    artifact_copy,
    assert_only,
    manifest,
    small_artifact,
)


def test_manifest_absent_is_a01(artifact_copy):
    (artifact_copy / "manifest.json").unlink()
    assert_only(artifact_copy, "A01")


def test_manifest_not_utf8_is_a01(artifact_copy):
    (artifact_copy / "manifest.json").write_bytes(b"\xff\xfe\x00")
    assert_only(artifact_copy, "A01")


def test_manifest_duplicate_key_is_a01(artifact_copy):
    (artifact_copy / "manifest.json").write_bytes(
        b'{"format":"tl-int-artifact","format":"tl-int-artifact"}\n'
    )
    assert_only(artifact_copy, "A01")


def test_manifest_not_canonical_is_a01(artifact_copy):
    text = json.dumps(manifest(artifact_copy), indent=2) + "\n"
    (artifact_copy / "manifest.json").write_text(text, encoding="ascii")
    assert_only(artifact_copy, "A01")
