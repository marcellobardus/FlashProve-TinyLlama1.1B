import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import synthetic_checkpoint

from tinyllama.artifacts import Dims
from tinyllama.convert import calibrate as calibration
from tinyllama.convert import fit
from tinyllama.convert.fit import FitError


FIT_DIMS = Dims(L=2, D=2048, H=32, G=4, DH=64, F=5632, V=32000, T=1024)


def _base_maxima(layers=2):
    return {
        "n1": {layer: 1.0 for layer in range(layers)},
        "n2": {layer: 1.0 for layer in range(layers)},
        "q": {layer: 8.0 for layer in range(layers)},
        "k": {layer: 8.0 for layer in range(layers)},
        "v": {layer: 8.0 for layer in range(layers)},
        "u": {layer: 8.0 for layer in range(layers)},
        "nF": {layers: 8.0},
    }


def _base_exponents(layers=2, value=12):
    exponents = {
        "q": {layer: value for layer in range(layers)},
        "k": {layer: value for layer in range(layers)},
        "v": {layer: value for layer in range(layers)},
        "o": {layer: value for layer in range(layers)},
        "gate": {layer: value for layer in range(layers)},
        "up": {layer: value for layer in range(layers)},
        "down": {layer: value for layer in range(layers)},
        "lm_head": value,
    }
    return exponents


def test_load_windows_checks_digest_shape_and_dtype(tmp_path):
    dims = Dims(L=2, D=64, H=4, G=2, DH=16, F=128, V=256, T=8)
    path = tmp_path / "windows.npy"
    array, digest = synthetic_checkpoint.write_windows(path, dims, n_windows=2)

    loaded = calibration.load_windows(path, digest)

    assert loaded.dtype == np.dtype("<i4")
    np.testing.assert_array_equal(loaded, array)

    with pytest.raises(calibration.CalibrationError):
        calibration.load_windows(path, "0" * 64)

    wrong_dtype = tmp_path / "float.npy"
    np.save(wrong_dtype, array.astype(np.float32))
    import hashlib

    with pytest.raises(calibration.CalibrationError):
        calibration.load_windows(
            wrong_dtype, hashlib.sha256(wrong_dtype.read_bytes()).hexdigest()
        )

    wrong_shape = tmp_path / "one-d.npy"
    np.save(wrong_shape, array.reshape(-1))
    with pytest.raises(calibration.CalibrationError):
        calibration.load_windows(
            wrong_shape, hashlib.sha256(wrong_shape.read_bytes()).hexdigest()
        )


def test_cell_exponent_hand_values():
    assert fit.cell_exponent(2047.0) == 0
    assert fit.cell_exponent(1.0) == 10
    assert fit.cell_exponent(8.0) == 7
    assert fit.cell_exponent(0.5) == 11
    with pytest.raises(FitError):
        fit.cell_exponent(0.0)


def test_fit_hand_computed_words():
    result = fit.fit_scales(_base_maxima(), _base_exponents(), FIT_DIMS)

    expected = [26, 26, 14, 8, 13, 14, 14, 46, 32, 46, 32, 6, 1]
    assert result.kl.dtype == np.uint32
    assert result.kl.shape == (2, 13)
    np.testing.assert_array_equal(result.kl[0], expected)
    np.testing.assert_array_equal(result.kl[1], expected)
    assert result.shared["MF"] == 45
    assert result.shared["KF"] == 32
    assert result.shared["s"] == 20
    assert result.shared["epsF"] == 85899
    assert result.shared["eps1"] == [85899, 85899]


def test_epsilon_formula_hand_values():
    assert fit.epsilon_from_e_res(3) == 1
    assert fit.epsilon_from_e_res(11) == 85899
    assert fit.epsilon_from_e_res(14) == 5497558


def test_build_params_has_every_field(tmp_path):
    result = fit.fit_scales(_base_maxima(), _base_exponents(), FIT_DIMS)
    digests = {
        name: [index] * 8 for index, name in enumerate(("trope", "texp", "tsilu"))
    }
    shas = {"tables/rope_c.i32": "a" * 64}
    params = fit.build_params(result, FIT_DIMS, digests=digests, tables_sha256=shas)

    for name in ("contract", "L", "d", "H", "G", "dh", "f", "V", "T", "e_res"):
        assert name in params
    assert params["contract"] == "TL-INT-1"
    assert params["s"] == 20
    assert params["epsF"] == 85899
    assert params["MF"] == 45
    assert params["KF"] == 32
    assert params["digests"]["trope"] == [0] * 8


def _maxima_with(site, layer, value):
    maxima = _base_maxima()
    maxima[site] = dict(maxima[site])
    maxima[site][layer] = value
    return maxima


def test_refuses_sh_above_12():
    maxima = _base_maxima()
    maxima["q"] = {0: 1e-9, 1: 1e-9}
    maxima["k"] = {0: 1e-9, 1: 1e-9}
    with pytest.raises(FitError, match="sh is"):
        fit.fit_scales(maxima, _base_exponents(), FIT_DIMS)


def test_refuses_sh_below_4():
    maxima = _base_maxima()
    maxima["q"] = {0: 1e12, 1: 8.0}
    maxima["k"] = {0: 1e12, 1: 8.0}
    with pytest.raises(FitError, match="sh is"):
        fit.fit_scales(maxima, _base_exponents(), FIT_DIMS)


@pytest.mark.parametrize(
    "kind,value",
    [
        ("q", 25),
        ("k", 25),
        ("v", 40),
        ("o", 40),
        ("gate", 40),
        ("up", 40),
        ("down", 40),
    ],
)
def test_refuses_each_shift_above_32(kind, value):
    exponents = _base_exponents()
    if kind in ("q", "k", "v", "gate", "up", "down"):
        exponents[kind] = {0: value, 1: value}
    else:
        exponents[kind] = {0: value, 1: value}
    with pytest.raises(FitError, match="outside 1 .. 32"):
        fit.fit_scales(_base_maxima(), exponents, FIT_DIMS)


def test_refuses_shift_below_1():
    exponents = _base_exponents()
    exponents["q"] = {0: -20, 1: -20}
    with pytest.raises(FitError, match="outside 1 .. 32"):
        fit.fit_scales(_base_maxima(), exponents, FIT_DIMS)


def test_refuses_kq_kk_difference_below_16():
    exponents = _base_exponents()
    exponents["q"] = {0: 0, 1: 0}
    exponents["k"] = {0: 1, 1: 1}
    with pytest.raises(FitError, match="kq"):
        fit.fit_scales(_base_maxima(), exponents, FIT_DIMS)


def test_refuses_norm_m_below_24():
    maxima = _maxima_with("n1", 0, 1e20)
    with pytest.raises(FitError, match="outside 24 .. 47"):
        fit.fit_scales(maxima, _base_exponents(), FIT_DIMS)


def test_refuses_norm_k_above_m():
    maxima = _maxima_with("n1", 0, 262016.0)
    with pytest.raises(FitError, match="above M"):
        fit.fit_scales(maxima, _base_exponents(), FIT_DIMS)


@pytest.mark.parametrize(
    "m,k,message",
    [
        (47, 32, "below M - 14"),
        (30, 32, "above M"),
        (30, 16, "outside 17 .. 32"),
        (23, 23, "outside 24 .. 47"),
    ],
)
def test_norm_pair_refusals(m, k, message):
    with pytest.raises(FitError, match=message):
        fit.check_norm_pair("test", m, k)


@pytest.mark.parametrize("e_res", [2, 15])
def test_refuses_e_res_outside_range(e_res):
    with pytest.raises(FitError, match="e_res"):
        fit.fit_scales(_base_maxima(), _base_exponents(), FIT_DIMS, e_res=e_res)


def test_refuses_s_above_63():
    exponents = _base_exponents()
    exponents["lm_head"] = 60
    with pytest.raises(FitError, match="outside 0 .. 63"):
        fit.fit_scales(_base_maxima(), exponents, FIT_DIMS)


def test_refuses_missing_site():
    maxima = _base_maxima()
    del maxima["u"]
    with pytest.raises(FitError, match="site u"):
        fit.fit_scales(maxima, _base_exponents(), FIT_DIMS)


def _independent_rms(x, eps):
    values = np.asarray(x, dtype=np.float32)
    mean = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    return values / np.sqrt(mean + np.float32(eps))


def _independent_rotate_half(x, cosines, sines):
    half = x.shape[-1] // 2
    first = x[..., :half]
    second = x[..., half:]
    c = cosines[:, None, :]
    s = sines[:, None, :]
    return np.concatenate((first * c - second * s, second * c + first * s), axis=-1)


def _independent_attention(q, k, v, heads, groups):
    positions, _, dh = q.shape
    group = heads // groups
    out = np.empty((positions, heads, dh), dtype=np.float32)
    for head in range(heads):
        query = q[:, head, :]
        key = k[:, head // group, :]
        value = v[:, head // group, :]
        scores = np.matmul(query, key.T)
        for position in range(positions):
            row = scores[position, : position + 1]
            weights = np.exp(row - np.max(row)).astype(np.float32)
            out[position, head] = (
                np.matmul(weights, value[: position + 1]) / np.sum(weights, dtype=np.float32)
            )
    return out.reshape(positions, heads * dh)


def _independent_logits(tensors, dims, ids, eps=1e-5, theta0=10000.0):
    x = np.asarray(tensors["model.embed_tokens.weight"][ids], dtype=np.float32)
    positions = np.arange(ids.shape[0], dtype=np.float32)
    half = dims.DH // 2
    inv_freq = np.power(np.float32(theta0), -np.arange(half, dtype=np.float32) / np.float32(half))
    angle = positions[:, None] * inv_freq[None, :]
    cosines = np.cos(angle).astype(np.float32)
    sines = np.sin(angle).astype(np.float32)

    for layer in range(dims.L):
        prefix = "model.layers.{}".format(layer)
        gain1 = tensors[prefix + ".input_layernorm.weight"]
        gain2 = tensors[prefix + ".post_attention_layernorm.weight"]
        n1 = _independent_rms(x, eps)
        a = n1 * gain1
        q = np.matmul(a, tensors[prefix + ".self_attn.q_proj.weight"].T).reshape(
            ids.shape[0], dims.H, dims.DH
        )
        k = np.matmul(a, tensors[prefix + ".self_attn.k_proj.weight"].T).reshape(
            ids.shape[0], dims.G, dims.DH
        )
        v = np.matmul(a, tensors[prefix + ".self_attn.v_proj.weight"].T).reshape(
            ids.shape[0], dims.G, dims.DH
        )
        q = _independent_rotate_half(q, cosines, sines)
        k = _independent_rotate_half(k, cosines, sines)
        o = _independent_attention(q, k, v, dims.H, dims.G)
        x = x + np.matmul(o, tensors[prefix + ".self_attn.o_proj.weight"].T)

        n2 = _independent_rms(x, eps)
        h = n2 * gain2
        gate = np.matmul(h, tensors[prefix + ".mlp.gate_proj.weight"].T)
        up = np.matmul(h, tensors[prefix + ".mlp.up_proj.weight"].T)
        silu = gate / (np.float32(1.0) + np.exp(-gate))
        x = x + np.matmul(silu * up, tensors[prefix + ".mlp.down_proj.weight"].T)

    nF = _independent_rms(x, eps)
    return np.matmul(nF * tensors["model.norm.weight"], tensors["lm_head.weight"].T)


def test_rotary_tables_use_the_contract_frequency():
    cosines, sines = calibration._rotary_tables(10000.0, np.arange(3), 16)
    for position in range(3):
        for index in range(8):
            angle = position * 10000.0 ** (-2.0 * index / 16.0)
            assert cosines[position, index] == pytest.approx(math.cos(angle), abs=1e-7)
            assert sines[position, index] == pytest.approx(math.sin(angle), abs=1e-7)


def test_calibrate_fits_hf_rotate_half_on_one_tiny_case(tmp_path):
    checkpoint = synthetic_checkpoint.build_checkpoint(tmp_path)
    dims = checkpoint["dims"]
    reader = calibration.CheckpointReader(checkpoint["safetensors"], checkpoint["manifest"])
    model = calibration.FloatModel(reader, dims)
    rng = np.random.default_rng(99)
    ids = rng.integers(0, dims.V, size=(1, 8), dtype=np.int32)

    maxima, logits = calibration.float_forward(model, ids, return_logits=True)
    expected = _independent_logits(checkpoint["tensors"], dims, ids[0])

    assert set(maxima) == {"n1", "n2", "q", "k", "v", "u", "nF"}
    assert maxima["nF"][dims.L] > 0.0
    np.testing.assert_allclose(logits[0], expected, rtol=1e-5, atol=1e-5)
