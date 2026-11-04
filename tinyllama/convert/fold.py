"""Folding, row permutation and quantization for TL-INT-1 (contract 4.2 to 4.5)."""

import numpy as np

from tinyllama import contract


class FoldError(ValueError):
    """A tensor cannot be folded or quantized under the contract."""


def c_bits_to_float32(bits):
    """Interpret an IEEE 754 binary32 bit pattern as a numpy float32."""
    word = np.array([bits], dtype=np.uint32)
    return word.view(np.float32)[0]


C0 = c_bits_to_float32(contract.C0_BITS)
CQ = c_bits_to_float32(contract.CQ_BITS)

_MIN_NONZERO_GAMMA = np.float32(2.0**-64)
_MAX_GAMMA = np.float32(2.0**64)


MATRIX_TABLE = {
    "q_proj": ("input_layernorm", "c_q", contract.H),
    "k_proj": ("input_layernorm", "c_0", contract.G),
    "v_proj": ("input_layernorm", "c_0", None),
    "o_proj": (None, None, None),
    "gate_proj": ("post_attention_layernorm", "c_0", None),
    "up_proj": ("post_attention_layernorm", "c_0", None),
    "down_proj": (None, None, None),
    "lm_head": ("model.norm", "c_0", None),
    "embed_tokens": (None, None, None),
}


def permute_qk(w, n_heads):
    """Return the contract 4.4 row permutation of an HF [out][in] weight."""
    w32 = np.asarray(w, dtype=np.float32)
    if w32.ndim != 2:
        raise ValueError("q/k weight must be two dimensional [out][in]")
    out = w32.shape[0]
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if out % n_heads:
        raise ValueError("output dimension is not divisible by n_heads")
    dh = out // n_heads
    if dh % 2:
        raise ValueError("head dimension must be even")

    half = dh // 2
    old = w32.reshape(n_heads, 2, half, w32.shape[1])
    return old.transpose(0, 2, 1, 3).reshape(out, w32.shape[1]).copy()


def fold_gains(w, gains, c):
    """Return pi = w * (gains * c), with each float32 operation rounded once."""
    w32 = np.asarray(w, dtype=np.float32)
    gains32 = np.asarray(gains, dtype=np.float32)
    if w32.ndim != 2:
        raise ValueError("weight must be two dimensional [out][in]")
    if gains32.ndim != 1 or gains32.shape[0] != w32.shape[1]:
        raise ValueError("gains must have one entry per input index")

    gamma = gains32 * np.float32(c)
    if not np.all(np.isfinite(gamma)):
        raise FoldError("gain product is not finite")

    magnitude = np.abs(gamma)
    bad = (gamma != 0) & (
        (magnitude < _MIN_NONZERO_GAMMA) | (magnitude > _MAX_GAMMA)
    )
    if np.any(bad):
        raise FoldError("nonzero gain product is outside 2**-64 .. 2**64")

    return w32 * gamma[None, :]


def choose_exponent(mx):
    """Return the largest e with RNE(2**e * mx) <= 4095.

    A power-of-two scale is exact in binary floating point: ldexp only changes
    the exponent, so the float32 ldexp is the exact value of 2**e * mx.  The
    frexp exponent gives a safe upper bound; e + 1 always exceeds 4095 and
    e - 1 always fits, so searching downward is exact rather than an estimate.
    """
    mx = np.float32(mx)
    if not np.isfinite(mx) or mx <= 0:
        raise FoldError("mx must be finite and positive")

    _, exponent = np.frexp(mx)
    e = int(12 - int(exponent))
    while np.rint(np.ldexp(mx, e)) > np.float32(contract.WEIGHT_BOUND):
        e -= 1
    return int(e)


def fold_and_quantize(w, gains=None, c_bits=None):
    """Fold optional gains and quantize a weight matrix by contract 4.5."""
    w32 = np.asarray(w, dtype=np.float32)
    if w32.ndim != 2:
        raise ValueError("weight must be two dimensional [out][in]")
    if not np.all(np.isfinite(w32)):
        raise FoldError("weight contains a non-finite value")

    if gains is not None:
        if c_bits is None:
            raise ValueError("c_bits is required when gains are given")
        gains32 = np.asarray(gains, dtype=np.float32)
        if gains32.ndim != 1 or gains32.shape[0] != w32.shape[1]:
            raise ValueError("gains must have one entry per input index")
        if not np.all(np.isfinite(gains32)):
            raise FoldError("gains contain a non-finite value")
        pi = fold_gains(w32, gains32, c_bits_to_float32(c_bits))
    else:
        pi = w32

    if not np.all(np.isfinite(pi)):
        raise FoldError("folded product contains a non-finite value")
    if pi.size == 0:
        raise FoldError("weight tensor is empty")

    mx = np.float32(np.max(np.abs(pi)))
    if mx == np.float32(0.0):
        raise FoldError("mx is zero")

    e = choose_exponent(mx)
    if e < 0 or e > 24:
        raise FoldError("exponent {} is outside 0 .. 24".format(e))

    q = np.rint(np.ldexp(pi, e)).astype(np.int16)
    assert np.all(np.abs(q.astype(np.int32)) <= contract.WEIGHT_BOUND)
    return q, e, mx


def quantize_embedding(v, e_res):
    """Quantize an embedding by contract 4.5 step 4, returning int32."""
    v32 = np.asarray(v, dtype=np.float32)
    if not np.all(np.isfinite(v32)):
        raise FoldError("embedding contains a non-finite value")

    w = np.rint(np.ldexp(v32, int(e_res)))
    if not np.all(np.isfinite(w)):
        raise FoldError("scaled embedding contains a non-finite value")
    if np.any(np.abs(w) >= np.float32(contract.EMB_BOUND)):
        raise FoldError("embedding entry is at least 2**21")

    return w.astype(np.int32)
