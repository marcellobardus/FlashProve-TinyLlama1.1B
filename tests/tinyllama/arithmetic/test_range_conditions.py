import ref_independent as ref

VALUE_CONDITIONS = (2, 3, 4, 6, 7, 8)


def test_value_condition_vectors():
    cases = ref.load_vectors("range_conditions.json")["value_cases"]
    assert {case["cond"] for case in cases} == set(VALUE_CONDITIONS)
    for case in cases:
        assert ref.value_condition_ok(case["cond"], case["value"]) is case["accepted"], case


def test_condition1_vectors():
    for case in ref.load_vectors("range_conditions.json")["condition1_cases"]:
        assert ref.condition1_ok(case["vx"], case["eps"]) is case["accepted"], case


def test_32768_32639_edges():
    for cond in (2, 3, 4, 7):
        assert ref.value_condition_ok(cond, -32768)
        assert ref.value_condition_ok(cond, 32639)
        assert not ref.value_condition_ok(cond, -32769)
        assert not ref.value_condition_ok(cond, 32640)


def test_16384_16383_edges():
    assert ref.value_condition_ok(6, -16384)
    assert ref.value_condition_ok(6, 16383)
    assert not ref.value_condition_ok(6, -16385)
    assert not ref.value_condition_ok(6, 16384)


def test_stream_value_edge():
    bound = 547599908735
    assert ref.value_condition_ok(8, bound)
    assert ref.value_condition_ok(8, -bound)
    assert not ref.value_condition_ok(8, bound + 1)
    assert not ref.value_condition_ok(8, -(bound + 1))


def test_condition1_edge():
    assert ref.condition1_ok((1 << 78) - 2, 1)
    assert not ref.condition1_ok((1 << 78) - 1, 1)
    assert not ref.condition1_ok(1 << 78, 1)
    assert not ref.condition1_ok(2048 * 547599908735 ** 2, 85899)


def test_failure_is_reported_not_saturated():
    for cond in (2, 3, 4, 6, 7, 8):
        assert not ref.value_condition_ok(cond, 1 << 62)
        assert not ref.value_condition_ok(cond, -(1 << 62))
