import math

import pytest

import tinyllama.reference as reference

from tinyllama.reference import (
    argmax,
    clamp,
    exp_lookup,
    floor_div,
    gate_key,
    isqrt,
    low8,
    norm,
    pad,
    rq,
    softmax_div,
)


RQ_CASES = [
    (1, 1, 1),
    (-1, 1, 0),
    (3, 1, 2),
    (-3, 1, -1),
    (7, 4, 0),
    (8, 4, 1),
    (-8, 4, 0),
    (-9, 4, -1),
    (24, 4, 2),
    (-24, 4, -1),
    (-25, 4, -2),
    (2**40 - 1, 20, 1048576),
    (-(2**40), 20, -1048576),
]


def test_primitive_names_are_exported_at_package_level():
    names = [
        "rq",
        "clamp",
        "isqrt",
        "floor_div",
        "low8",
        "softmax_div",
        "exp_lookup",
        "argmax",
        "gate_key",
        "norm",
        "pad",
    ]
    for name in names:
        assert callable(getattr(reference, name))


def test_rq_matches_the_contract_table():
    for a, k, expected in RQ_CASES:
        assert rq(a, k) == expected, (a, k)


def test_rq_rejects_k_below_one():
    for k in (0, -1):
        with pytest.raises(ValueError):
            rq(0, k)


def test_clamp_matches_the_contract_examples():
    assert clamp(2048, 2047) == 2047
    assert clamp(-2048, 2047) == -2047
    assert clamp(40000, 2047) == 2047
    assert clamp(-40000, 2047) == -2047


def test_isqrt_is_exact_and_rejects_negative():
    assert isqrt(0) == 0
    assert isqrt(26) == 5
    assert isqrt(299865660046580329300225) == 547599908735
    with pytest.raises(ValueError):
        isqrt(-1)
    assert isqrt(2**62) == 2**31


def test_floor_div_floors_and_rejects_nonpositive_divisors():
    assert floor_div(7, 2) == 3
    assert floor_div(-7, 2) == -4
    assert floor_div(-9, 2) == -5
    with pytest.raises(ValueError):
        floor_div(1, 0)
    with pytest.raises(ValueError):
        floor_div(1, -2)


def test_low8_wraps_to_a_balanced_byte():
    assert low8(0) == 0
    assert low8(127) == 127
    assert low8(128) == -128
    assert low8(300) == 44
    assert low8(-300) == -44
    assert low8(-129) == 127


def test_softmax_div_matches_the_contract_examples():
    examples = [
        (3, 2, 2),
        (-3, 2, -1),
        (5, 2, 3),
        (-5, 2, -2),
        (-7, 4, -2),
    ]
    for acc, z, expected in examples:
        assert softmax_div(acc, z) == expected, (acc, z)
    for z in (0, -1):
        with pytest.raises(ValueError):
            softmax_div(1, z)


def test_exp_lookup_uses_2307_and_rejects_negative():
    table = [4096 - u for u in range(2307)]
    assert exp_lookup(0, table) == 4096
    assert exp_lookup(2306, table) == 1790
    assert exp_lookup(2307, table) == 0
    assert exp_lookup(1000000, table) == 0
    with pytest.raises(ValueError):
        exp_lookup(-1, table)


def test_argmax_is_the_first_maximum():
    assert argmax([5, 9, 9, -1]) == 1
    assert argmax([7, 7, 7]) == 0
    assert argmax([-5, -2, -9, -2]) == 1
    assert argmax([42]) == 0
    with pytest.raises(ValueError):
        argmax([])


def test_gate_key_matches_the_contract_examples():
    cases = [
        (300, (44, 1, 1, 300)),
        (-300, (-44, -1, -1, -300)),
        (128, (-128, 1, 1, 128)),
        (-129, (127, -1, -1, -129)),
        (40000, (64, 156, 127, 32576)),
        (-40000, (-64, -156, -127, -32576)),
        (0, (0, 0, 0, 0)),
    ]
    for gw, (g0, gh, gc, key) in cases:
        result = gate_key(gw)
        assert (result["g0"], result["gh"], result["gc"], result["key"]) == (
            g0,
            gh,
            gc,
            key,
        ), gw


def test_norm_matches_the_worked_example():
    result = norm([3, -4], 24, 17, 1)
    assert result["vx"] == 25
    assert result["sigma"] == 5
    assert result["rc"] == 3355443
    assert result["w"] == [77, -102]
    assert result["nr"] == [77, -102]


def test_norm_clamps_only_its_output():
    result = norm([7, 0, -7], 30, 17, 1)
    assert result["vx"] == 98
    assert result["sigma"] == 9
    assert result["rc"] == 119304647
    assert result["w"] == [6372, 0, -6372]
    assert result["nr"] == [2047, 0, -2047]


def test_pad_right_pads_to_1024_and_rejects_more():
    padded = pad([5, 6])
    assert len(padded) == 1024
    assert padded[:2] == [5, 6]
    assert padded[2:] == [0] * 1022
    assert pad([]) == [0] * 1024
    assert pad([0] * 1024) == [0] * 1024
    with pytest.raises(ValueError):
        pad([0] * 1025)


def test_norm_uses_python_integers_for_vx():
    result = norm([547599908735, 0], 24, 17, 1)
    assert result["vx"] == 547599908735**2
    assert result["vx"] > 2**64
    assert result["sigma"] == math.isqrt(547599908735**2 + 1)
