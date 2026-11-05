import struct

import numpy as np
import pytest

from tinyllama import contract
from tinyllama.convert import fold
from tinyllama.convert.fold import FoldError


C0 = 1.41421353816986083984375
CQ = 0.17677669227123260498046875


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def test_permute_qk_one_head_dh4():
    w = np.arange(4, dtype=np.float32).reshape(4, 1)
    before = w.copy()

    result = fold.permute_qk(w, 1)

    np.testing.assert_array_equal(
        result[:, 0], np.array([0, 2, 1, 3], dtype=np.float32)
    )
    assert result.dtype == np.float32
    assert result is not w
    np.testing.assert_array_equal(w, before)


def test_permute_qk_two_heads_dh8():
    w = np.arange(16, dtype=np.float32).reshape(16, 1)

    result = fold.permute_qk(w, 2)

    expected = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
    np.testing.assert_array_equal(
        result[:, 0], np.array(expected, dtype=np.float32)
    )


@pytest.mark.parametrize(
    "shape,n_heads",
    [
        ((6, 2), 4),
        ((6, 2), 2),
        ((8, 2), 0),
    ],
)
def test_permute_qk_refusals(shape, n_heads):
    with pytest.raises(ValueError):
        fold.permute_qk(np.zeros(shape, dtype=np.float32), n_heads)


def _cos_sin(position, dh):
    half = dh // 2
    index = np.arange(half, dtype=np.float64)
    inv_freq = contract.THETA0 ** (-index / half)
    angle = position * inv_freq
    return np.cos(angle), np.sin(angle)


def _hf_rotate(x, position):
    dh = x.shape[1]
    cos, sin = _cos_sin(position, dh)
    first = x[:, : dh // 2]
    second = x[:, dh // 2 :]
    return np.concatenate(
        [first * cos - second * sin, second * cos + first * sin], axis=1
    )


def _neighbour_rotate(x, position):
    dh = x.shape[1]
    cos, sin = _cos_sin(position, dh)
    a0 = x[:, 0::2]
    a1 = x[:, 1::2]
    result = np.empty_like(x)
    result[:, 0::2] = cos * a0 - sin * a1
    result[:, 1::2] = sin * a0 + cos * a1
    return result


def test_rotary_probe_matches_neighbour_pairs():
    rng = np.random.default_rng(20261104)
    n_heads, dh, position = 2, 8, 3
    q = (0.5 * rng.standard_normal((n_heads, dh))).astype(np.float32)
    k = (0.5 * rng.standard_normal((n_heads, dh))).astype(np.float32)

    q_permuted = fold.permute_qk(q.reshape(n_heads * dh, 1), n_heads).reshape(
        n_heads, dh
    )
    k_permuted = fold.permute_qk(k.reshape(n_heads * dh, 1), n_heads).reshape(
        n_heads, dh
    )

    q_permuted = q_permuted.astype(np.float64)
    k_permuted = k_permuted.astype(np.float64)
    q_neighbour = _neighbour_rotate(q_permuted, position)
    k_neighbour = _neighbour_rotate(k_permuted, position)

    q_hf = _hf_rotate(q.astype(np.float64), position)
    k_hf = _hf_rotate(k.astype(np.float64), position)
    order = np.array([0, 4, 1, 5, 2, 6, 3, 7])
    np.testing.assert_allclose(q_neighbour, q_hf[:, order], atol=1e-12, rtol=0)
    np.testing.assert_allclose(k_neighbour, k_hf[:, order], atol=1e-12, rtol=0)

    hf_scores = np.sum(q_hf * k_hf, axis=1)
    neighbour_scores = np.sum(q_neighbour * k_neighbour, axis=1)
    np.testing.assert_allclose(
        neighbour_scores, hf_scores, atol=1e-12, rtol=0
    )


def test_contract_constants():
    assert float(fold.C0) == C0
    assert float(fold.CQ) == CQ
    assert fold.c_bits_to_float32(contract.C0_BITS) == fold.C0
    assert fold.c_bits_to_float32(contract.CQ_BITS) == fold.CQ


def test_fold_and_quantize_hand_values():
    w = np.array([[1.0, -0.5], [0.25, 0.125]], dtype=np.float32)

    q, e, mx = fold.fold_and_quantize(w)

    assert e == 11
    assert isinstance(e, int)
    assert mx.dtype == np.float32
    assert float(mx) == 1.0
    assert q.dtype == np.int16
    np.testing.assert_array_equal(
        q, np.array([[2048, -1024], [512, 256]], dtype=np.int16)
    )


def test_exponent_boundary_rounding():
    w = np.array([[4095.5 * 2.0**-12]], dtype=np.float32)
    q, e, mx = fold.fold_and_quantize(w)
    assert e == 11
    assert float(mx) == 4095.5 * 2.0**-12
    np.testing.assert_array_equal(q, np.array([[2048]], dtype=np.int16))

    w = np.array([[4094.5 * 2.0**-12]], dtype=np.float32)
    q, e, mx = fold.fold_and_quantize(w)
    assert e == 12
    assert float(mx) == 4094.5 * 2.0**-12
    np.testing.assert_array_equal(q, np.array([[4094]], dtype=np.int16))


def test_ties_to_even_quantization():
    w = np.array([[0.5, 1.5, 2.5, 3000.0]], dtype=np.float32)

    q, e, mx = fold.fold_and_quantize(w)

    assert e == 0
    assert float(mx) == 3000.0
    np.testing.assert_array_equal(
        q, np.array([[0, 2, 2, 3000]], dtype=np.int16)
    )


def test_fold_gains_hand_values():
    w = np.array([[1.0, 2.0]], dtype=np.float32)
    gains = np.array([0.5, 3.0], dtype=np.float32)
    gamma0 = 0.707106769084930419921875
    gamma1 = _f32(3.0 * C0)
    pi1 = _f32(2.0 * gamma1)

    pi = fold.fold_gains(w, gains, fold.C0)
    np.testing.assert_array_equal(
        pi, np.array([[gamma0, pi1]], dtype=np.float32)
    )

    q, e, mx = fold.fold_and_quantize(w, gains, contract.C0_BITS)
    assert e == 8
    assert float(mx) == pi1
    np.testing.assert_array_equal(q, np.array([[181, 2172]], dtype=np.int16))


def test_fold_refuses_zero_mx():
    with pytest.raises(FoldError):
        fold.fold_and_quantize(np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_fold_refuses_non_finite_weight(bad):
    w = np.array([[1.0, bad]], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.fold_and_quantize(w)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_fold_refuses_non_finite_gains(bad):
    w = np.ones((1, 2), dtype=np.float32)
    gains = np.array([1.0, bad], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.fold_and_quantize(w, gains, contract.C0_BITS)


def test_fold_refuses_exponent_above_24():
    w = np.full((1, 1), 1e-10, dtype=np.float32)
    with pytest.raises(FoldError):
        fold.fold_and_quantize(w)


def test_fold_refuses_exponent_below_zero():
    w = np.array([[5000.0]], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.fold_and_quantize(w)


def test_fold_refuses_tiny_nonzero_gamma():
    w = np.ones((1, 1), dtype=np.float32)
    gains = np.array([2.0**-70], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.fold_gains(w, gains, np.float32(1.0))


def test_fold_refuses_huge_gamma():
    w = np.ones((1, 1), dtype=np.float32)
    gains = np.array([2.0**65], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.fold_gains(w, gains, np.float32(1.0))


def test_zero_gain_is_allowed():
    w = np.array([[1.0, 2.0]], dtype=np.float32)
    gains = np.array([0.0, 1.0], dtype=np.float32)

    pi = fold.fold_gains(w, gains, fold.C0)
    assert float(pi[0, 0]) == 0.0

    q, e, mx = fold.fold_and_quantize(w, gains, contract.C0_BITS)
    assert q.shape == w.shape
    assert np.all(np.abs(q.astype(np.int32)) <= contract.WEIGHT_BOUND)


def test_quantize_embedding_boundary():
    e_res = 4
    value = (2**21 - 1) * 2.0**-e_res
    v = np.array([value, 0.0, -value], dtype=np.float32)

    w = fold.quantize_embedding(v, e_res)

    assert w.dtype == np.int32
    np.testing.assert_array_equal(
        w, np.array([2**21 - 1, 0, -(2**21 - 1)], dtype=np.int32)
    )

    too_big = np.array([2.0**21 * 2.0**-e_res], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.quantize_embedding(too_big, e_res)

    too_negative = np.array([-(2.0**21) * 2.0**-e_res], dtype=np.float32)
    with pytest.raises(FoldError):
        fold.quantize_embedding(too_negative, e_res)


def test_quantize_embedding_refuses_non_finite():
    with pytest.raises(FoldError):
        fold.quantize_embedding(np.array([np.nan], dtype=np.float32), 0)


def test_random_quantization_bounded_and_deterministic():
    rng = np.random.default_rng(1234)
    w = (500.0 * rng.standard_normal((19, 31))).astype(np.float32)

    q1, e1, mx1 = fold.fold_and_quantize(w)
    q2, e2, mx2 = fold.fold_and_quantize(w)

    assert q1.shape == w.shape
    assert q1.dtype == np.int16
    assert np.array_equal(q1, q2)
    assert e1 == e2
    assert float(mx1) == float(mx2)
    assert np.max(np.abs(q1.astype(np.int32))) <= contract.WEIGHT_BOUND


def test_matrix_table():
    assert fold.MATRIX_TABLE == {
        "q_proj": ("input_layernorm", "c_q", 32),
        "k_proj": ("input_layernorm", "c_0", 4),
        "v_proj": ("input_layernorm", "c_0", None),
        "o_proj": (None, None, None),
        "gate_proj": ("post_attention_layernorm", "c_0", None),
        "up_proj": ("post_attention_layernorm", "c_0", None),
        "down_proj": (None, None, None),
        "lm_head": ("model.norm", "c_0", None),
        "embed_tokens": (None, None, None),
    }
