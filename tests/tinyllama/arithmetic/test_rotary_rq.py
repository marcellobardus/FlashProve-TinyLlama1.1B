import tl_ref_independent as ref


def test_rotary_pair_then_rq_and_clamp():
    cases = ref.load_vectors("rotary_rq.json")["cases"]
    assert len(cases) >= 8
    for case in cases:
        p0, p1 = ref.rotary_pair(case["a0"], case["a1"], case["c"], case["s"])
        assert p0 == case["expected_p0"], case
        assert p1 == case["expected_p1"], case
        assert ref.rq(p0, case["k"]) == case["expected_rq_p0"], case
        assert ref.rq(p1, case["k"]) == case["expected_rq_p1"], case
        assert ref.clamp(ref.rq(p0, case["k"]), 2047) == case["expected_clamped_p0"], case
        assert ref.clamp(ref.rq(p1, case["k"]), 2047) == case["expected_clamped_p1"], case


def test_hand_derivations_are_present():
    for case in ref.load_vectors("rotary_rq.json")["cases"]:
        assert case["derivation"].strip(), case


def test_negative_exact_ties_round_up():
    assert ref.rotary_pair(2, 0, 2, 1) == (4, 2)
    assert ref.rq(-4, 3) == 0
    assert ref.rq(-2, 3) == 0
    assert ref.rq(4, 3) == 1


def test_rotary_pair_is_linear_in_the_input_pair():
    assert ref.rotary_pair(0, 0, 7, -3) == (0, 0)
    assert ref.rotary_pair(1, 0, 7, -3) == (7, -3)
    assert ref.rotary_pair(0, 1, 7, -3) == (3, 7)
