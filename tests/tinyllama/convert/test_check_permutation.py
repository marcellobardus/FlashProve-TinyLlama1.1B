import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import synthetic_checkpoint

from tinyllama import contract
from tinyllama.artifacts import Dims
from tinyllama.convert import calibrate
from tinyllama.convert import check_permutation as checks
from tinyllama.convert import fold


REPO_ROOT = Path(__file__).resolve().parents[3]


def _cos_sin(positions, depth):
    return calibrate._rotary_tables(
        contract.THETA0, np.arange(positions), depth
    )


def _hf_rotate_half(values, cosines, sines):
    """Small numpy stand in for transformers rotate_half on halves."""
    half = values.shape[-1] // 2
    first = values[..., :half]
    second = values[..., half:]
    cos = cosines[:, None, :]
    sin = sines[:, None, :]
    return np.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=-1
    )


def test_p1_scores_pass_with_permutation_and_fail_without():
    rng = np.random.default_rng(20261104)
    positions, depth, heads, groups, head_dim = 4, 10, 4, 2, 8
    a = (0.3 * rng.standard_normal((positions, depth))).astype(np.float32)
    wq = (0.2 * rng.standard_normal((heads * head_dim, depth))).astype(np.float32)
    wk = (0.2 * rng.standard_normal((groups * head_dim, depth))).astype(np.float32)
    cosines, sines = _cos_sin(positions, head_dim)

    q = (a @ wq.T).reshape(positions, heads, head_dim)
    k = (a @ wk.T).reshape(positions, groups, head_dim)
    hf_scores = checks.attention_scores(
        _hf_rotate_half(q, cosines, sines),
        _hf_rotate_half(k, cosines, sines),
    )

    wq_permuted = fold.permute_qk(wq, heads)
    wk_permuted = fold.permute_qk(wk, groups)
    q_permuted = (a @ wq_permuted.T).reshape(positions, heads, head_dim)
    k_permuted = (a @ wk_permuted.T).reshape(positions, groups, head_dim)
    neighbour_scores = checks.attention_scores(
        calibrate._rotary_neighbour(q_permuted, cosines, sines),
        calibrate._rotary_neighbour(k_permuted, cosines, sines),
    )

    good = checks.compare_score_arrays(hf_scores, neighbour_scores)
    assert good["passed"]
    assert good["max_relative_difference"] < 1e-5
    assert good["relative_difference_definition"] == "max|a-b| / max|a|"

    unpermuted_scores = checks.attention_scores(
        calibrate._rotary_neighbour(q, cosines, sines),
        calibrate._rotary_neighbour(k, cosines, sines),
    )
    bad = checks.compare_score_arrays(hf_scores, unpermuted_scores)
    assert not bad["passed"]
    assert bad["max_relative_difference"] > 1e-5


def test_relative_difference_definition():
    a = np.array([2.0, -4.0])
    b = np.array([2.5, -3.5])

    assert checks.relative_difference(a, b) == pytest.approx(0.5 / 4.0)
    assert checks.relative_difference(np.zeros(3), np.zeros(3)) == 0.0
    assert math.isinf(checks.relative_difference(np.zeros(3), np.ones(3)))
    with pytest.raises(ValueError):
        checks.relative_difference(np.zeros(2), np.zeros(3))


def test_compare_logits_reports_greedy_and_threshold():
    reference = np.array([[0.0, 1.0, 2.0], [3.0, 5.0, 4.0]], dtype=np.float32)
    candidate = reference.copy()
    candidate[-1] += np.array([0.0, 0.0, 0.0001], dtype=np.float32)

    good = checks.compare_logits(reference, candidate)
    assert good["passed"]
    assert good["same_next_token"]
    assert good["hf_next_token"] == good["numpy_next_token"] == 1

    candidate[-1, 2] = 10.0
    bad = checks.compare_logits(reference, candidate)
    assert not bad["passed"]
    assert not bad["same_next_token"]


def _write_checkpoint(root, tensors):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "model.safetensors"
    header = synthetic_checkpoint.write_safetensors(path, tensors)
    manifest = {
        "repo_id": "synthetic/check-permutation-test",
        "revision": "synthetic",
        "flashprove_base_commit": "synthetic",
        "model_safetensors": {
            "header_json_sha256": hashlib.sha256(header).hexdigest(),
            "tensor_count": len(tensors),
            "sha256_lfs_etag": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "tensors": {
            name: {"dtype": "BF16", "shape": list(values.shape)}
            for name, values in tensors.items()
        },
    }
    return path, manifest


def test_p2_passes_on_a_tiny_synthetic_checkpoint(tmp_path):
    info = synthetic_checkpoint.build_checkpoint(tmp_path / "model")
    result = checks.run_p2(
        info["dir"], info["manifest"], e_res=11, dims=info["dims"]
    )

    assert result["passed"]
    assert result["tensor_count"] == len(info["tensors"])
    assert result["checks"] == {
        "no_non_finite": True,
        "exponents_in_range": True,
        "gammas_in_range": True,
        "embedding_bound": True,
        "no_other_failures": True,
    }


def test_p2_fails_on_non_finite_tensor(tmp_path):
    tensors = {
        "model.layers.0.self_attn.o_proj.weight": np.array(
            [[1.0, np.nan], [0.5, 0.5]], dtype=np.float32
        )
    }
    path, manifest = _write_checkpoint(tmp_path / "ckpt", tensors)
    result = checks.run_p2(path, manifest, e_res=11)

    assert not result["passed"]
    assert result["checks"]["no_non_finite"] is False
    assert any(failure["reason"] == "non_finite" for failure in result["failures"])


def test_p2_fails_on_exponent_out_of_range(tmp_path):
    tensors = {
        "model.layers.0.self_attn.o_proj.weight": np.full(
            (2, 2), 5000.0, dtype=np.float32
        )
    }
    path, manifest = _write_checkpoint(tmp_path / "ckpt", tensors)
    result = checks.run_p2(path, manifest, e_res=11)

    assert not result["passed"]
    assert result["checks"]["exponents_in_range"] is False
    assert any(failure["reason"] == "exponent" for failure in result["failures"])


def test_p2_fails_on_embedding_bound(tmp_path):
    tensors = {
        "model.embed_tokens.weight": np.full(
            (2, 2), 1024.0, dtype=np.float32
        )
    }
    path, manifest = _write_checkpoint(tmp_path / "ckpt", tensors)
    result = checks.run_p2(path, manifest, e_res=11)

    assert not result["passed"]
    assert result["checks"]["embedding_bound"] is False
    assert any(
        failure["reason"] == "embedding_bound" for failure in result["failures"]
    )


def test_p2_fails_on_gain_out_of_range(tmp_path):
    tensors = {
        "model.layers.0.input_layernorm.weight": np.array(
            [1.0, 2.0**70], dtype=np.float32
        ),
        "model.layers.0.self_attn.v_proj.weight": np.ones(
            (2, 2), dtype=np.float32
        ),
    }
    path, manifest = _write_checkpoint(tmp_path / "ckpt", tensors)
    result = checks.run_p2(path, manifest, e_res=11)

    assert not result["passed"]
    assert result["checks"]["gammas_in_range"] is False
    assert any(failure["reason"] == "gamma" for failure in result["failures"])


def test_p3_passes_and_reports_a_mismatch(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"abc" * 1000)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"model_safetensors": {"sha256_lfs_etag": expected}}

    good = checks.run_p3(path, manifest)
    assert good["passed"]
    assert good["expected"] == expected
    assert good["actual"] == expected
    assert good["size_bytes"] == path.stat().st_size

    bad = checks.run_p3(
        path, {"model_safetensors": {"sha256_lfs_etag": "0" * 64}}
    )
    assert not bad["passed"]
    assert bad["actual"] == expected


def test_json_report_shape(tmp_path):
    tensors = {
        "model.embed_tokens.weight": np.full(
            (2, 2), 0.5, dtype=np.float32
        )
    }
    path, manifest = _write_checkpoint(tmp_path / "ckpt", tensors)
    report = checks.run_checks(
        "p2",
        path,
        manifest,
        manifest_source="test manifest",
        e_res=11,
        dims=Dims(L=1, D=2, H=1, G=1, DH=2, F=2, V=2, T=2),
    )
    output = tmp_path / "report.json"
    checks.write_report(output, report)
    loaded = json.loads(output.read_text(encoding="ascii"))

    assert loaded["contract"] == "TL-INT-1"
    assert loaded["mode"] == "p2"
    assert loaded["checks_requested"] == ["P2"]
    assert set(loaded["checks"]) == {"P2"}
    assert loaded["checks"]["P2"]["passed"] is True
    assert set(loaded["inputs"]) >= {
        "checkpoint",
        "safetensors",
        "manifest",
        "e_res",
        "e_res_source",
        "params",
    }
    assert loaded["inputs"]["e_res"] == 11
    assert loaded["inputs"]["manifest"] == "test manifest"
    assert "numpy" in loaded["versions"]
    assert "python" in loaded["versions"]


def test_p3_json_report_shape(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"tiny")
    manifest = {
        "model_safetensors": {
            "sha256_lfs_etag": hashlib.sha256(path.read_bytes()).hexdigest()
        }
    }
    report = checks.run_checks(
        "p3", path, manifest, manifest_source="test manifest"
    )
    output = tmp_path / "p3.json"
    checks.write_report(output, report)
    loaded = json.loads(output.read_text(encoding="ascii"))

    assert set(loaded["checks"]) == {"P3"}
    assert loaded["checks"]["P3"]["passed"] is True
    assert loaded["checks"]["P3"]["sha256_chunk_bytes"] == 1 << 20


def test_import_does_not_import_torch():
    code = (
        "import sys\n"
        "import tinyllama.convert.check_permutation\n"
        "print(int('torch' in sys.modules))\n"
        "print(int('transformers' in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["0", "0"]
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
