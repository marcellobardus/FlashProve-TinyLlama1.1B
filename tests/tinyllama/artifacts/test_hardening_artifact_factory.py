"""Build a small valid artifact for the T1.6a hardening tests."""

import importlib.util
from pathlib import Path

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

CONVERT_TESTS = (
    Path(__file__).resolve().parents[1] / "convert" / "test_artifacts.py"
)


def _convert_helpers():
    spec = importlib.util.spec_from_file_location(
        "hardening_convert_test_artifacts", str(CONVERT_TESTS)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DIMS = _convert_helpers().DIMS


def build_small_artifact(root):
    helpers = _convert_helpers()
    data = helpers._valid_data()
    out = Path(root) / "artifacts"
    artifact_id = artifacts.write_artifact(
        out,
        data["matrices"],
        data["emb"],
        data["kl_words"],
        data["params"],
        data["tables"],
        data["provenance"],
        dims=helpers.DIMS,
        exponents={"emb": {"e": 11, "mx": 0}},
    )
    return out / artifact_id
