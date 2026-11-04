import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import synthetic_checkpoint

from tinyllama import contract
from tinyllama.artifacts import Dims
from tinyllama.convert import calibrate
from tinyllama.convert import check_permutation as checks
from tinyllama.convert.calibrate import CheckpointReader


DIMS = Dims(L=3, D=32, H=4, G=2, DH=8, F=64, V=97, T=16)
LOGIT_TOLERANCE = 1e-3


def _rms_norm(x, gain, eps):
    mean = np.mean(x * x, axis=-1, keepdims=True)
    normalized = x / np.sqrt(mean + eps)
    return normalized * gain


def _rotate_half(values):
    half = values.shape[-1] // 2
    first = values[..., :half]
    second = values[..., half:]
    return np.concatenate((-second, first), axis=-1)


def _rotary(x, cosines, sines):
    return x * cosines[:, None, :] + _rotate_half(x) * sines[:, None, :]


def _attention(q, k, v, heads, groups, dh):
    positions = q.shape[0]
    group = heads // groups
    scale = 1.0 / math.sqrt(dh)
    out = np.empty((positions, heads, dh), dtype=np.float64)
    for head in range(heads):
        key_head = head // group
        scores = (q[:, head, :] @ k[:, key_head, :].T) * scale
        for position in range(positions):
            row = scores[position, : position + 1]
            weights = np.exp(row - np.max(row))
            weights = weights / np.sum(weights)
            out[position, head] = weights @ v[: position + 1, key_head, :]
    return out.reshape(positions, heads * dh)


def _independent_forward(tensors, dims, ids, eps=1e-5, theta0=contract.THETA0):
    x = np.asarray(tensors["model.embed_tokens.weight"][ids], dtype=np.float64)
    positions = len(ids)
    frequency = 1.0 / (
        float(theta0)
        ** (np.arange(0, dims.DH, 2, dtype=np.float64) / float(dims.DH))
    )
    angle = np.arange(positions, dtype=np.float64)[:, None] * frequency[None, :]
    frequencies = np.concatenate((angle, angle), axis=-1)
    cosines = np.cos(frequencies)
    sines = np.sin(frequencies)

    for layer in range(dims.L):
        prefix = "model.layers.{}".format(layer)
        gain1 = np.asarray(
            tensors[prefix + ".input_layernorm.weight"], dtype=np.float64
        )
        gain2 = np.asarray(
            tensors[prefix + ".post_attention_layernorm.weight"], dtype=np.float64
        )
        wq = np.asarray(tensors[prefix + ".self_attn.q_proj.weight"], dtype=np.float64)
        wk = np.asarray(tensors[prefix + ".self_attn.k_proj.weight"], dtype=np.float64)
        wv = np.asarray(tensors[prefix + ".self_attn.v_proj.weight"], dtype=np.float64)
        wo = np.asarray(tensors[prefix + ".self_attn.o_proj.weight"], dtype=np.float64)
        wg = np.asarray(tensors[prefix + ".mlp.gate_proj.weight"], dtype=np.float64)
        wu = np.asarray(tensors[prefix + ".mlp.up_proj.weight"], dtype=np.float64)
        wd = np.asarray(tensors[prefix + ".mlp.down_proj.weight"], dtype=np.float64)

        normalized = _rms_norm(x, gain1, eps)
        q = (normalized @ wq.T).reshape(positions, dims.H, dims.DH)
        k = (normalized @ wk.T).reshape(positions, dims.G, dims.DH)
        v = (normalized @ wv.T).reshape(positions, dims.G, dims.DH)
        q = _rotary(q, cosines, sines)
        k = _rotary(k, cosines, sines)
        o = _attention(q, k, v, dims.H, dims.G, dims.DH)
        x = x + o @ wo.T

        normalized = _rms_norm(x, gain2, eps)
        gate = normalized @ wg.T
        up = normalized @ wu.T
        silu = gate / (1.0 + np.exp(-gate))
        x = x + (silu * up) @ wd.T

    final_gain = np.asarray(tensors["model.norm.weight"], dtype=np.float64)
    lm_head = np.asarray(tensors["lm_head.weight"], dtype=np.float64)
    return _rms_norm(x, final_gain, eps) @ lm_head.T


def _checkpoint_and_ids(tmp_path):
    info = synthetic_checkpoint.build_checkpoint(tmp_path, dims=DIMS)
    dims = info["dims"]
    ids = np.array([3, 17, 41, 5, 88, 12], dtype=np.int64)
    return info, dims, ids


def test_synthetic_checkpoint_has_the_shape_this_test_needs(tmp_path):
    info, dims, _ids = _checkpoint_and_ids(tmp_path)
    tensors = info["tensors"]

    assert dims.L >= 3
    assert dims.H > dims.G
    for layer in range(dims.L):
        gain = tensors[
            "model.layers.{}.input_layernorm.weight".format(layer)
        ]
        assert np.any(gain != np.float32(1.0))


def test_numpy_logits_matches_independent_hf_forward(tmp_path):
    info, dims, ids = _checkpoint_and_ids(tmp_path)
    reader = CheckpointReader(info["safetensors"], info["manifest"])

    reference = _independent_forward(info["tensors"], dims, ids)
    actual = checks.numpy_logits(reader, ids, dims).astype(np.float64)

    max_abs = float(np.max(np.abs(reference - actual)))
    assert max_abs < LOGIT_TOLERANCE, max_abs

    result = checks.compare_logits(reference, actual)
    assert result["passed"], result


def test_identity_permutation_fails_the_logits_check(tmp_path, monkeypatch):
    info, dims, ids = _checkpoint_and_ids(tmp_path)
    reader = CheckpointReader(info["safetensors"], info["manifest"])
    reference = _independent_forward(info["tensors"], dims, ids)

    def identity(values, n_heads):
        return np.asarray(values, dtype=np.float32).copy()

    monkeypatch.setattr(calibrate, "permute_qk", identity)
    actual = checks.numpy_logits(reader, ids, dims).astype(np.float64)

    result = checks.compare_logits(reference, actual)
    assert result["passed"] is False
