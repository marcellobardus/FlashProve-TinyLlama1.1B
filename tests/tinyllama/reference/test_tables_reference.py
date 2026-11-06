import hashlib
import struct
from functools import lru_cache

import numpy as np

from tinyllama.reference.tables import (
    CONTRACT_SHA256,
    exp_table,
    rope_tables,
    sha256_words,
    silu_table,
    verify_contract_tables,
)


@lru_cache(maxsize=None)
def _rope():
    return rope_tables(1024, 64, 10000)


@lru_cache(maxsize=None)
def _exp():
    return tuple(exp_table())


@lru_cache(maxsize=None)
def _silu():
    return silu_table()


def test_rope_table_shape_and_samples():
    cosines, sines = _rope()
    assert cosines.shape == (1024, 32)
    assert sines.shape == (1024, 32)
    assert cosines.dtype == np.int64
    assert sines.dtype == np.int64
    assert cosines[0, 0] == 4096
    assert list(cosines[0, :3]) == [4096, 4096, 4096]
    assert list(sines[0, :3]) == [0, 0, 0]
    assert list(cosines[1, 0:3]) == [2213, 2997, 3465]
    assert list(sines[1, 0:3]) == [3447, 2792, 2184]
    assert cosines[1023, 31] == 4058
    assert sines[1023, 31] == 557


def test_rope_table_small_dimensions():
    cosines, sines = rope_tables(3, 4, 10000)
    assert cosines.shape == (3, 2)
    assert sines.shape == (3, 2)
    assert list(cosines[0]) == [4096, 4096]
    assert list(sines[0]) == [0, 0]


def test_exp_table_shape_and_first_entries():
    table = _exp()
    assert len(table) == 2307
    assert table[0] == 4096
    assert all(1 <= value <= 4096 for value in table)


def test_silu_table_shape_samples_and_range():
    table = _silu()
    assert table.shape == (65536,)
    assert table.dtype == np.int64
    assert table[32768] == 0
    assert table[32768 + 256] == 47
    assert table[32768 - 256] == -17
    assert int(table.min()) == -18
    assert int(table.max()) == 4095


def test_sha256_words_is_little_endian():
    words = [0x04030201, 0x08070605]
    expected = hashlib.sha256(struct.pack("<2I", *words)).hexdigest()
    assert sha256_words(words) == expected
    assert sha256_words(words) != sha256_words(words[::-1])


def test_full_size_tables_match_the_contract_sha256_values():
    computed = verify_contract_tables()
    assert computed == CONTRACT_SHA256
