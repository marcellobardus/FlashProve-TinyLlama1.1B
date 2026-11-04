"""A07 caller supplied ids and A09 weight range sub-rules."""

import numpy as np
import pytest

from test_hardening_support import (
    DIMS,
    artifacts,
    artifact_copy,
    assert_only,
    rewrite_file,
    small_artifact,
)


def test_expected_id_mismatch_is_a07_on_a_valid_artifact(artifact_copy):
    assert artifacts.validate_artifact(artifact_copy, dims=DIMS) == []
    with pytest.raises(artifacts.ArtifactRejected) as caught:
        artifacts.load_artifact(
            artifact_copy, expected_id="0" * 64, dims=DIMS
        )
    assert caught.value.code == "A07"
    assert caught.value.path == "manifest.json"
    assert "expected id" in caught.value.detail


@pytest.mark.parametrize("value", [2**21, -(2**21)])
def test_embedding_with_absolute_value_2_21_is_a09(artifact_copy, value):
    path = "emb.i32"
    array = np.fromfile(artifact_copy / path, dtype="<i4")
    array[0] = value
    directory = rewrite_file(
        artifact_copy, path, data=array.astype("<i4").tobytes()
    )
    assert_only(directory, "A09")


def test_weight_below_minus_4095_is_a09(artifact_copy):
    path = "wh.i16"
    array = np.fromfile(artifact_copy / path, dtype="<i2")
    array[0] = -4096
    directory = rewrite_file(
        artifact_copy, path, data=array.astype("<i2").tobytes()
    )
    assert_only(directory, "A09")
