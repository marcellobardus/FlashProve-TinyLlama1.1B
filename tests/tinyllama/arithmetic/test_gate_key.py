import tl_ref_independent as ref


def test_vectors():
    cases = ref.load_vectors("gate_key.json")["cases"]
    assert len(cases) >= 5
    for case in cases:
        result = ref.gate_key(case["gw"])
        assert result["g0"] == case["g0"], case
        assert result["gh"] == case["gh"], case
        assert result["gc"] == case["gc"], case
        assert result["key"] == case["key"], case


def test_contract_examples():
    assert ref.gate_key(300) == {"g0": 44, "gh": 1, "gc": 1, "key": 300}
    assert ref.gate_key(-300) == {"g0": -44, "gh": -1, "gc": -1, "key": -300}
    assert ref.gate_key(128) == {"g0": -128, "gh": 1, "gc": 1, "key": 128}
    assert ref.gate_key(-129) == {"g0": 127, "gh": -1, "gc": -1, "key": -129}
    assert ref.gate_key(40000) == {"g0": 64, "gh": 156, "gc": 127, "key": 32576}


def test_g0_is_the_balanced_low_byte_and_split_is_exact():
    for case in ref.load_vectors("gate_key.json")["cases"]:
        gw = case["gw"]
        assert ref.low8(gw) == case["g0"], case
        assert (gw - case["g0"]) % 256 == 0, case
        assert (gw - case["g0"]) // 256 == case["gh"], case
        assert -128 <= case["g0"] <= 127, case


def test_key_reconstruction_and_gate_clamp():
    for case in ref.load_vectors("gate_key.json")["cases"]:
        assert 256 * case["gc"] + case["g0"] == case["key"], case
        assert case["gc"] == ref.clamp(case["gh"], 127), case


def test_key_is_period_256_in_gw_where_gc_equals_gh():
    for gw in range(-1000, 1001):
        result = ref.gate_key(gw)
        if result["gc"] == result["gh"]:
            assert result["key"] == gw, gw
