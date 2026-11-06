from collections import UserDict

import numpy as np
import pytest

import check_inputs


class Encoding:
    def __init__(self, ids):
        self.ids = list(ids)


class BatchEncoding(UserDict):
    pass


class TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_normalize_list():
    assert check_inputs.normalize_ids([1, 2, 3]) == [1, 2, 3]
    assert all(type(value) is int for value in check_inputs.normalize_ids([1, 2, 3]))


def test_normalize_tuple():
    assert check_inputs.normalize_ids((4, 5, 6)) == [4, 5, 6]


def test_normalize_single_row_nested_list():
    assert check_inputs.normalize_ids([[7, 8, 9]]) == [7, 8, 9]


def test_normalize_mapping():
    assert check_inputs.normalize_ids({"input_ids": [10, 11]}) == [10, 11]


def test_normalize_user_dict_batch_encoding():
    assert check_inputs.normalize_ids(BatchEncoding({"input_ids": [12, 13]})) == [
        12,
        13,
    ]


def test_normalize_object_with_tolist():
    assert check_inputs.normalize_ids(TensorLike([[14, 15]])) == [14, 15]


def test_normalize_two_dimensional_numpy_array():
    array = np.array([[16, 17, 18]], dtype=np.int64)
    assert check_inputs.normalize_ids(array) == [16, 17, 18]


def test_normalize_numpy_integer_elements_become_python_ints():
    values = check_inputs.normalize_ids(np.array([np.int64(19), np.int64(20)]))
    assert values == [19, 20]
    assert all(type(value) is int for value in values)


def test_normalize_object_with_ids():
    assert check_inputs.normalize_ids(Encoding([21, 22])) == [21, 22]


def test_normalize_rejects_two_batch_rows():
    with pytest.raises(ValueError):
        check_inputs.normalize_ids([[1, 2], [3, 4]])


def test_normalize_rejects_string():
    with pytest.raises(ValueError):
        check_inputs.normalize_ids("1 2 3")
