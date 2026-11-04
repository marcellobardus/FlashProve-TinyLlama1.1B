import math
from fractions import Fraction

import pytest

import tl_ref_independent as ref


def round_half_up(numerator, denominator):
    return math.floor(Fraction(numerator, denominator) + Fraction(1, 2))


def test_division_vectors():
    for case in ref.load_vectors("softmax_div.json")["cases"]:
        assert ref.softmax_div(case["acc"], case["z"]) == case["expected"], case


def test_contract_sentences():
    assert ref.softmax_div(3, 2) == 2
    assert ref.softmax_div(-3, 2) == -1
    assert ref.softmax_div(5, 2) == 3
    assert ref.softmax_div(-5, 2) == -2
    assert ref.softmax_div(-7, 4) == -2


def test_division_is_acc_over_z_round_half_up():
    for case in ref.load_vectors("softmax_div.json")["cases"]:
        acc, z = case["acc"], case["z"]
        assert ref.softmax_div(acc, z) == round_half_up(acc, z), case
    for acc in range(-50, 51):
        for z in (1, 2, 3, 4, 5, 4096, 4194304):
            assert ref.softmax_div(acc, z) == round_half_up(acc, z), (acc, z)
    for acc in (-8585740288, -2047, -1, 0, 1, 2047, 8585740288):
        for z in (4096, 8388608, 4194304):
            assert ref.softmax_div(acc, z) == round_half_up(acc, z), (acc, z)


def test_output_stays_within_12_bits_at_the_bounds():
    assert ref.softmax_div(1024 * 4096 * 2047, 4194304) == 2047
    assert ref.softmax_div(-(1024 * 4096 * 2047), 4194304) == -2047


def test_u_is_floored_difference():
    for case in ref.load_vectors("softmax_exp_index.json")["u_cases"]:
        m, s, sh = case["m"], case["s"], case["sh"]
        u = ref.floor_div(m - s, 1 << sh)
        assert u == case["expected_u"], case
        assert u >= 0, case


@pytest.mark.parametrize("case_key", ["lookup_cases", "e_cases"])
def test_exp_indexing_rule(case_key):
    table = ref.FAKE_EXP_TABLE
    assert len(table) == 2307
    for case in ref.load_vectors("softmax_exp_index.json")[case_key]:
        if case_key == "lookup_cases":
            assert ref.exp_lookup(case["u"], table) == case["expected_e"], case
        else:
            u = ref.floor_div(case["m"] - case["s"], 1 << case["sh"])
            assert ref.exp_lookup(u, table) == case["expected_e"], case


def test_exp_indexing_rule_boundary():
    table = ref.FAKE_EXP_TABLE
    assert ref.exp_lookup(2306, table) == table[2306] == 1790
    for u in (2307, 2308, 2**20, 2**31, 10**18):
        assert ref.exp_lookup(u, table) == 0
    for u in range(0, 2307, 100):
        assert ref.exp_lookup(u, table) == table[u]
