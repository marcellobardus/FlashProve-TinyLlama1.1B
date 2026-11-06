import hashlib
import struct

import pytest

from tinyllama import contract
from tinyllama.convert import tables


CONTRACT_DIGEST_WORDS = {
    "trope": (
        2141436500,
        1986672181,
        1784239491,
        223369813,
        3762430148,
        1100108057,
        3971569448,
        1221024000,
    ),
    "texp": (
        3648852807,
        2402568834,
        4186852988,
        523559909,
        1873895626,
        3042864545,
        1726265496,
        165292856,
    ),
    "tsilu": (
        2048398297,
        192427613,
        2862900154,
        2853881485,
        2797826050,
        1420723544,
        3686369553,
        2593882461,
    ),
}

CONTRACT_SHA256 = {
    "trope": "973450425e0f39b307e1561d7aaa70904b0fab3d5862e453b70ccfd7c60915c1",
    "texp": "e27fdb2a6e9b273138ee66a6729de3c2c186ac1e81d24e5514ad29c155221b63",
    "tsilu": "0c26f9b15535c1bb9d1878552b5122158670d6eadd8e9f3ba988833e87e7a442",
}


def _sha256_words(words):
    return hashlib.sha256(
        struct.pack("<%dI" % len(words), *words)
    ).hexdigest()


def _tables():
    cosines, sines = tables.rope_tables(
        contract.T, contract.DH, contract.CS, contract.THETA0
    )
    exp = tables.softmax_table(contract.EXP_S, contract.RX)
    silu = tables.silu_table(contract.EG, contract.SMAX)
    return cosines, sines, exp, silu


def _words():
    cosines, sines, exp, silu = _tables()
    return {
        "trope": tables.trope_words(cosines, sines),
        "texp": tables.texp_words(exp),
        "tsilu": tables.tsilu_words(silu),
    }


def test_rotary_sample_entries():
    cosines, sines, _, _ = _tables()
    assert cosines[:32] == [4096] * 32
    assert sines[:32] == [0] * 32
    assert cosines[32:35] == [2213, 2997, 3465]
    assert sines[32:35] == [3447, 2792, 2184]
    assert cosines[1023 * 32 + 31] == 4058
    assert sines[1023 * 32 + 31] == 557


def test_softmax_and_silu_samples():
    _, _, exp, silu = _tables()
    assert len(exp) == 2307
    assert exp[0] == 4096
    assert all(1 <= entry <= 4096 for entry in exp)
    assert len(silu) == 65536
    assert silu[32768] == 0
    assert silu[32768 + 256] == 47
    assert silu[32768 - 256] == -17


@pytest.mark.parametrize("name", ["trope", "texp", "tsilu"])
def test_digest_words_match_the_contract(name):
    assert tables.table_digest(_words()[name]) == list(
        CONTRACT_DIGEST_WORDS[name]
    )


@pytest.mark.parametrize("name", ["trope", "texp", "tsilu"])
def test_sha256_of_the_table_words_matches_the_contract(name):
    assert _sha256_words(_words()[name]) == CONTRACT_SHA256[name]
