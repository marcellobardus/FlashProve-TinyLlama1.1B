import pytest

import tl_ref_independent as ref


def test_vectors():
    cases = ref.load_vectors("padding.json")["cases"]
    assert len(cases) >= 3
    for case in cases:
        ids = case["ids"]
        expected = case["expected"]
        assert len(expected) == 1024, case
        assert expected[: len(ids)] == ids, case
        assert expected[len(ids) :] == [0] * (1024 - len(ids)), case
        assert ref.pad(ids) == expected, case


def test_appends_token_id_zero_to_the_right():
    assert ref.pad([5]) == [5] + [0] * 1023
    assert ref.pad([1, 2, 3]) == [1, 2, 3] + [0] * 1021
    assert ref.pad([0, 1, 0, 31999, 5, 0, 7, 9]) == [0, 1, 0, 31999, 5, 0, 7, 9] + [0] * 1016


def test_window_is_always_1024_positions():
    for n in (0, 1, 2, 5, 512, 1023, 1024):
        ids = list(range(n))
        padded = ref.pad(ids)
        assert len(padded) == 1024, n
        assert padded[:n] == ids, n
        assert padded[n:] == [0] * (1024 - n), n
    assert ref.pad([]) == [0] * 1024


def test_a_full_window_is_unchanged():
    ids = [31999] * 1024
    assert ref.pad(ids) == ids
    assert ref.pad(list(range(1024))) == list(range(1024))


def test_more_than_1024_tokens_is_an_error():
    with pytest.raises(ValueError):
        ref.pad([0] * 1025)
    with pytest.raises(ValueError):
        ref.pad(list(range(1025)))
    with pytest.raises(ValueError):
        ref.pad([7] * 2000)
    with pytest.raises(ValueError):
        ref.pad([1] * (1 << 20))
