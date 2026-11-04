import math
from fractions import Fraction

import pytest

import tl_ref_independent as ref


def round_half_up(numerator, denominator):
    return math.floor(Fraction(numerator, denominator) + Fraction(1, 2))


def rq_by_bits(a, k):
    return (a >> k) + (1 if (a & ((1 << k) - 1)) >= (1 << (k - 1)) else 0)


def test_vectors():
    cases = ref.load_vectors("rq.json")["cases"]
    assert len(cases) >= 13
    for case in cases:
        assert ref.rq(case["a"], case["k"]) == case["expected"], case


CONTRACT_TABLE_ROWS = [
    (1, 1),
    (-1, 1),
    (3, 1),
    (-3, 1),
    (7, 4),
    (8, 4),
    (-8, 4),
    (-9, 4),
    (24, 4),
    (-24, 4),
    (-25, 4),
    (1099511627775, 20),
    (-1099511627776, 20),
]


@pytest.mark.parametrize("a, k", CONTRACT_TABLE_ROWS)
def test_contract_table_rows_present_in_vectors(a, k):
    rows = [c for c in ref.load_vectors("rq.json")["cases"] if c["a"] == a and c["k"] == k]
    assert rows, (a, k)
    assert ref.rq(a, k) == rows[0]["expected"]


def test_matches_exact_rational_rounding():
    cases = ref.load_vectors("rq.json")["cases"]
    checked = 0
    for k in range(1, 41):
        magnitudes = [0, 1, -1, 2, -2]
        magnitudes += [(1 << k) - 1, -(1 << k) + 1, 1 << k, -(1 << k)]
        magnitudes += [(1 << k) * m + (1 << (k - 1)) for m in (1, -1, 5, -5, 1000, -1000)]
        magnitudes += [(1 << 62), -(1 << 62), (1 << 62) - 1, -((1 << 62) - 1)]
        for a in magnitudes:
            assert ref.rq(a, k) == round_half_up(a, 1 << k), (a, k)
            assert ref.rq(a, k) == rq_by_bits(a, k), (a, k)
            checked += 1
    for case in cases:
        a, k = case["a"], case["k"]
        assert ref.rq(a, k) == round_half_up(a, 1 << k), case
        assert ref.rq(a, k) == rq_by_bits(a, k), case
        checked += 1
    assert checked > 41 * 5


@pytest.mark.parametrize("k", [1, 2, 3, 7, 12, 17, 20, 32, 39, 40])
@pytest.mark.parametrize("m", [0, 1, -1, 5, -5, 32768, -32768, 2**40, -(2**40)])
def test_ties_go_up(k, m):
    a = (2 * m + 1) << (k - 1)
    assert ref.rq(a, k) == m + 1
    assert ref.rq(-a, k) == -m


def test_large_magnitudes_and_wide_k():
    assert ref.rq(1 << 62, 40) == 4194304
    assert ref.rq(-(1 << 62), 40) == -4194304
    assert ref.rq((1 << 62) - 1, 40) == 4194304
    assert ref.rq(-((1 << 62) - 1), 40) == -4194304
    assert ref.rq((1 << 62) + (1 << 39), 40) == 4194305
    assert ref.rq(-((1 << 62) + (1 << 39)), 40) == -4194304
    for a in range(-(1 << 12), 1 << 12):
        assert ref.rq(a, 40) == 0
