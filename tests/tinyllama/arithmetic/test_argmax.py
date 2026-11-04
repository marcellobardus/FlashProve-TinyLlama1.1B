import ref_independent as ref


def test_vectors():
    for case in ref.load_vectors("argmax.json")["cases"]:
        values = case["values"]
        assert ref.argmax(values) == case["expected"], case


def test_contract_example():
    assert ref.argmax([5, 9, 9, -1]) == 1


def test_least_index_of_maximum_property():
    for case in ref.load_vectors("argmax.json")["cases"]:
        values = case["values"]
        result = ref.argmax(values)
        assert values[result] == max(values), case
        assert result == values.index(max(values)), case


def test_tie_goes_to_the_lowest_index():
    assert ref.argmax([7, 7, 7]) == 0
    assert ref.argmax([-1, -1]) == 0
    assert ref.argmax([1, 2, 3, 3]) == 2
    assert ref.argmax([5, 9, 9, -1, 9]) == 1
    for size in range(1, 30):
        values = [-(size + 1)] * size
        assert ref.argmax(values) == 0
        values[size // 2] = 0
        assert ref.argmax(values) == size // 2


def test_single_element_and_extremes():
    assert ref.argmax([42]) == 0
    assert ref.argmax([0]) == 0
    assert ref.argmax([2**62, -(2**62)]) == 0
    assert ref.argmax([-(2**62), 2**62]) == 1
