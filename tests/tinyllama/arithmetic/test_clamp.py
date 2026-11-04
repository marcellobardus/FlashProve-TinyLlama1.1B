import ref_independent as ref


def test_vectors():
    for case in ref.load_vectors("clamp.json")["cases"]:
        assert ref.clamp(case["a"], case["bound"]) == case["expected"], case


def test_contract_sentences():
    assert ref.clamp(2048, 2047) == 2047
    assert ref.clamp(-2048, 2047) == -2047
    assert ref.clamp(40000, 2047) == 2047


def test_twelve_bit_cell_has_no_minus_2048():
    assert ref.clamp(-2048, 2047) == -2047
    assert ref.clamp(2048, 2047) == 2047


def test_activation_range_is_untouched():
    for value in range(-2047, 2048):
        assert ref.clamp(value, 2047) == value


def test_weight_range_is_untouched():
    for value in range(-4095, 4096):
        assert ref.clamp(value, 4095) == value


def test_saturation_is_symmetric():
    for bound in (1, 3, 127, 2047, 4095):
        assert ref.clamp(bound + 1, bound) == bound
        assert ref.clamp(-(bound + 1), bound) == -bound
        assert ref.clamp(bound * 1000, bound) == bound
        assert ref.clamp(-(bound * 1000), bound) == -bound
