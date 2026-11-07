import tl_ref_independent as ref


def test_floor_div_vectors():
    for case in ref.load_vectors("floor_div.json")["cases"]:
        assert ref.floor_div(case["a"], case["b"]) == case["expected"], case


def test_floor_div_floors_for_negatives():
    for a in range(-100, 101):
        for b in (1, 2, 3, 7, 16, 256):
            q = ref.floor_div(a, b)
            assert q * b <= a < (q + 1) * b, (a, b)
            assert q == a // b


def test_low8_vectors():
    for case in ref.load_vectors("low8.json")["cases"]:
        assert ref.low8(case["a"]) == case["expected"], case


def test_low8_range_and_lattice_properties():
    for a in range(-1000, 1001):
        value = ref.low8(a)
        assert -128 <= value <= 127, a
        assert (a - value) % 256 == 0, a
        assert ref.low8(a + 256) == value, a
        assert ref.low8(a - 256) == value, a
    assert ref.low8((1 << 62) + 300) == 44
    assert ref.low8(-(1 << 62) - 300) == -44
