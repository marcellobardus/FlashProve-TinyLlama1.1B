import warnings

import numpy as np
import pytest

from tinyllama.convert.calibrate import _silu


def _input_values():
    smallest = np.float32(np.finfo(np.float32).smallest_subnormal)
    largest = np.float32(np.finfo(np.float32).max)
    specials = [
        0.0,
        -0.0,
        1e-30,
        -1e-30,
        1.0,
        -1.0,
        80.0,
        -80.0,
        88.0,
        -88.0,
        88.7,
        -88.7,
        89.0,
        -89.0,
        100.0,
        -100.0,
        1e4,
        -1e4,
        1e30,
        -1e30,
        float(largest),
        float(-largest),
        float(smallest),
        float(-smallest),
    ]
    rng = np.random.default_rng(0)
    random = rng.random(10000, dtype=np.float32) * np.float32(400.0) - np.float32(200.0)
    return np.concatenate(
        (np.asarray(specials, dtype=np.float32), random.astype(np.float32))
    )


def _reference(values):
    with np.errstate(over="ignore"):
        return values / (np.float32(1.0) + np.exp(-values))


@pytest.mark.filterwarnings("error")
def test_silu_matches_old_expression_without_warnings():
    values = _input_values()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _silu(values)

    reference = _reference(values)

    assert result.dtype == np.dtype(np.float32)
    assert np.array_equal(result.view(np.uint32), reference.view(np.uint32))
    assert not np.isnan(result).any()

    negative = values <= np.float32(-89.0)
    assert negative.any()
    assert np.signbit(result[negative]).all()
    assert (result[negative] == np.float32(0.0)).all()
