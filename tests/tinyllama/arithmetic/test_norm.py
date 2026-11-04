import math
from fractions import Fraction

import tl_ref_independent as ref


def round_half_up(numerator, denominator):
    return math.floor(Fraction(numerator, denominator) + Fraction(1, 2))


def test_vectors():
    cases = ref.load_vectors("norm.json")["cases"]
    assert len(cases) >= 8
    for case in cases:
        result = ref.norm(case["x"], case["m"], case["k"], case["eps"])
        assert result["vx"] == case["expected_vx"], case
        assert result["sigma"] == case["expected_sigma"], case
        assert result["rc"] == case["expected_rc"], case
        assert result["w"] == case["expected_w"], case
        assert result["nr"] == case["expected_nr"], case


def test_contract_worked_example():
    result = ref.norm([3, -4], 24, 17, 1)
    assert result["vx"] == 25
    assert result["sigma"] == 5
    assert result["rc"] == 3355443
    assert result["w"] == [77, -102]
    assert result["nr"] == [77, -102]


def test_vectors_recomputed_independently():
    for case in ref.load_vectors("norm.json")["cases"]:
        x, m, k, eps = case["x"], case["m"], case["k"], case["eps"]
        vx = sum(value * value for value in x)
        sigma = ref.isqrt_bisect(vx + eps)
        rc = round_half_up(1 << m, sigma)
        w = [round_half_up(value * rc, 1 << k) for value in x]
        nr = [min(max(value, -2047), 2047) for value in w]
        assert (vx, sigma, rc, w, nr) == (
            case["expected_vx"],
            case["expected_sigma"],
            case["expected_rc"],
            case["expected_w"],
            case["expected_nr"],
        ), case


def test_rc_is_two_to_the_m_over_sigma_round_half_up():
    for sigma in range(1, 200):
        for m in (24, 32, 47):
            by_contract = ((1 << m) + (sigma // 2)) // sigma
            by_fraction = round_half_up(1 << m, sigma)
            assert by_contract == by_fraction, (sigma, m)


def test_sigma_at_least_one_when_eps_at_least_one():
    for eps in range(1, 5):
        for x in ([0, 0], [0], [3, -4], [2047, -2047]):
            result = ref.norm(x, 24, 17, eps)
            assert result["sigma"] >= 1, (x, eps)


def test_output_never_exceeds_12_bits():
    for case in ref.load_vectors("norm.json")["cases"]:
        result = ref.norm(case["x"], case["m"], case["k"], case["eps"])
        for value in result["nr"]:
            assert -2047 <= value <= 2047, case
