import hashlib
import importlib.util
import math
import re
import struct
from functools import lru_cache
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

TABLE_NUMBER = 72
MASK64 = (1 << 64) - 1

BLAKE2B_IV = (
    0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1, 0x510E527FADE682D1, 0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179,
)

BLAKE2B_SIGMA = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
    (11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
    (7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
    (9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
    (2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
    (12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
    (13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
    (6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
    (10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
)

CONTRACT_DIGEST_WORDS = {
    "trope": (2141436500, 1986672181, 1784239491, 223369813, 3762430148, 1100108057, 3971569448, 1221024000),
    "texp": (3648852807, 2402568834, 4186852988, 523559909, 1873895626, 3042864545, 1726265496, 165292856),
    "tsilu": (2048398297, 192427613, 2862900154, 2853881485, 2797826050, 1420723544, 3686369553, 2593882461),
}

CONTRACT_SHA256 = {
    "trope": "973450425e0f39b307e1561d7aaa70904b0fab3d5862e453b70ccfd7c60915c1",
    "texp": "e27fdb2a6e9b273138ee66a6729de3c2c186ac1e81d24e5514ad29c155221b63",
    "tsilu": "0c26f9b15535c1bb9d1878552b5122158670d6eadd8e9f3ba988833e87e7a442",
}

PINNED_TEXP_POLICY = REPOSITORY_ROOT / "prover" / "tests" / "configurations" / "pinned_no_proof_of_work_roomy" / "policy.txt"
PINNED_TSILU_POLICY = (
    REPOSITORY_ROOT
    / "integrations"
    / "vllm"
    / "integer_model"
    / "tests"
    / "configurations"
    / "tiny_without_clamps"
    / "policy.txt"
)


def rha(v):
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def _rotr64(word, count):
    return ((word >> count) | (word << (64 - count))) & MASK64


def _g(a, b, c, d, x, y):
    a = (a + b + x) & MASK64
    d = _rotr64(d ^ a, 32)
    c = (c + d) & MASK64
    b = _rotr64(b ^ c, 24)
    a = (a + b + y) & MASK64
    d = _rotr64(d ^ a, 16)
    c = (c + d) & MASK64
    b = _rotr64(b ^ c, 63)
    return a, b, c, d


def _compress(h, m, t_low, t_high, last):
    v = list(h) + list(BLAKE2B_IV)
    v[12] ^= t_low
    v[13] ^= t_high
    if last:
        v[14] ^= MASK64
    for r in range(12):
        s = BLAKE2B_SIGMA[r]
        v[0], v[4], v[8], v[12] = _g(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]])
        v[1], v[5], v[9], v[13] = _g(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]])
        v[2], v[6], v[10], v[14] = _g(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]])
        v[3], v[7], v[11], v[15] = _g(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]])
        v[0], v[5], v[10], v[15] = _g(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]])
        v[1], v[6], v[11], v[12] = _g(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]])
        v[2], v[7], v[8], v[13] = _g(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]])
        v[3], v[4], v[9], v[14] = _g(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]])
    return [h[i] ^ v[i] ^ v[i + 8] for i in range(8)]


def _blake2b(data, counter_high):
    h = list(BLAKE2B_IV)
    h[0] ^= 0x01010040
    blocks = max(1, -(-len(data) // 128))
    for k in range(blocks):
        m = struct.unpack("<16Q", data[128 * k : 128 * (k + 1)].ljust(128, b"\x00"))
        last = k == blocks - 1
        t_low = len(data) if last else 128 * (k + 1)
        h = _compress(h, m, t_low, counter_high, last)
    return struct.pack("<8Q", *h)


def _words_bytes(words):
    return struct.pack("<%dI" % len(words), *words)


@lru_cache(maxsize=None)
def _rotary_tables():
    cosines = []
    sines = []
    for t in range(1024):
        for i in range(32):
            b = (-2 * i) / 64
            w = math.pow(10000.0, b)
            x = t * w
            cosines.append(rha(4096 * math.cos(x)))
            sines.append(rha(4096 * math.sin(x)))
    return cosines, sines


@lru_cache(maxsize=None)
def _exp_table():
    table = []
    u = 0
    while True:
        entry = round(4096 * math.exp(-u / 256))
        if entry == 0:
            return table
        table.append(entry)
        u += 1


@lru_cache(maxsize=None)
def _silu_table():
    table = []
    for c in range(-32768, 32768):
        c_clamped = min(max(c, -64 * 256), 64 * 256 - 1)
        z = c_clamped / 256
        g = 1 + math.exp(-z)
        s = z / g
        table.append(min(max(rha(64 * s), -4095), 4095))
    return table


@lru_cache(maxsize=None)
def _hashed_words(name):
    if name == "trope":
        cosines, sines = _rotary_tables()
        return tuple(v + 4096 for v in cosines) + tuple(v + 4096 for v in sines)
    if name == "texp":
        return tuple(_exp_table())
    if name == "tsilu":
        return tuple(v + 4096 for v in _silu_table())
    raise ValueError("no table named %s" % name)


@lru_cache(maxsize=None)
def _digest_words(name):
    words = _hashed_words(name)
    counter_high = TABLE_NUMBER + (len(words) << 32)
    return struct.unpack("<8I", _blake2b(_words_bytes(words), counter_high)[:32])


def _sha256_of_words(words):
    return hashlib.sha256(_words_bytes(words)).hexdigest()


@lru_cache(maxsize=None)
def _policy_tables():
    path = REPOSITORY_ROOT / "scales" / "policy_tables.py"
    if not path.is_file():
        pytest.fail("scales/policy_tables.py is missing")
    spec = importlib.util.spec_from_file_location("policy_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pinned_digest_words(path, table):
    if not path.is_file():
        pytest.skip("%s is missing" % path)
    found = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"%s([0-9]+) ([0-9]+)" % table, line)
        if match:
            found[int(match.group(1))] = int(match.group(2))
    indices = sorted(found)
    assert indices == list(range(8)), "%s: %s lines at indices %s" % (path, table, indices)
    return [found[i] for i in range(8)]


def _assert_same_entries(one, two, label):
    one = list(one)
    two = list(two)
    if one == two:
        return
    if len(one) != len(two):
        pytest.fail("%s: %d entries against %d" % (label, len(one), len(two)))
    for index in range(len(one)):
        if one[index] != two[index]:
            pytest.fail("%s: first difference at index %d: %r against %r" % (label, index, one[index], two[index]))
    pytest.fail("%s: no single entry differs" % label)


def test_rounding_primitives():
    assert [rha(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5)] == [1, 2, 3, -1, -2, -3]
    assert [round(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5, -2.5)] == [0, 2, 2, 0, -2, -2]


@pytest.mark.parametrize("size", [0, 3, 128, 129, 1000])
def test_own_blake2b_matches_hashlib(size):
    data = (bytes(range(256)) * 4)[:size]
    assert _blake2b(data, 0) == hashlib.blake2b(data, digest_size=64).digest()


def test_rotary_sample_entries():
    cosines, sines = _rotary_tables()
    assert cosines[:32] == [4096] * 32
    assert sines[:32] == [0] * 32
    assert cosines[32:35] == [2213, 2997, 3465]
    assert sines[32:35] == [3447, 2792, 2184]
    assert cosines[1023 * 32 + 31] == 4058
    assert sines[1023 * 32 + 31] == 557


def test_rotary_sizes_and_ranges():
    cosines, sines = _rotary_tables()
    assert len(cosines) == 32768
    assert len(sines) == 32768
    assert all(-4096 <= v <= 4096 for v in cosines)
    assert all(-4096 <= v <= 4096 for v in sines)


def test_exp_table_shape():
    table = _exp_table()
    assert len(table) == 2307
    assert table[0] == 4096
    assert all(1 <= v <= 4096 for v in table)
    assert round(4096 * math.exp(-len(table) / 256)) == 0


def test_silu_table_samples_and_range():
    table = _silu_table()
    assert len(table) == 65536
    assert all(-18 <= v <= 4095 for v in table)
    assert min(table) == -18
    assert max(table) == 4095
    assert table[32768] == 0
    assert table[32768 + 256] == 47
    assert table[32768 - 256] == -17


@pytest.mark.parametrize("name", ["trope", "texp", "tsilu"])
def test_digest_words_match_the_contract(name):
    assert _digest_words(name) == CONTRACT_DIGEST_WORDS[name]


@pytest.mark.parametrize("name", ["trope", "texp", "tsilu"])
def test_sha256_of_the_table_words_matches_the_contract(name):
    of_table_words = _sha256_of_words(_hashed_words(name))
    of_digest_words = _sha256_of_words(CONTRACT_DIGEST_WORDS[name])
    assert of_table_words == CONTRACT_SHA256[name], (
        "%s: sha256 of the table words is %s, sha256 of the eight digest words is %s, "
        "the contract's column says %s"
        % (name, of_table_words, of_digest_words, CONTRACT_SHA256[name])
    )


def test_policy_tables_entries_agree():
    pt = _policy_tables()
    _assert_same_entries(
        pt.silu_table(8, 4095), _silu_table(), "scales/policy_tables.py silu_table(8, 4095) against the Ts built here"
    )
    _assert_same_entries(
        pt.softmax_table(12, 8), _exp_table(), "scales/policy_tables.py softmax_table(12, 8) against the EX built here"
    )
    flash_cosines, flash_sines = pt.rotary_tables(1024, 64, 12, 10000)
    cosines, sines = _rotary_tables()
    _assert_same_entries(
        flash_cosines, cosines, "scales/policy_tables.py rotary_tables(1024, 64, 12, 10000) against the C built here"
    )
    _assert_same_entries(
        flash_sines, sines, "scales/policy_tables.py rotary_tables(1024, 64, 12, 10000) against the S built here"
    )


def test_policy_tables_hash_words_agree():
    pt = _policy_tables()
    for name in ("trope", "texp", "tsilu"):
        flash = pt.hash_words(TABLE_NUMBER, list(_hashed_words(name)))
        _assert_same_entries(
            flash,
            _digest_words(name),
            "scales/policy_tables.py hash_words(72, words) against the %s digest built here" % name,
        )


def test_pinned_policy_texp_lines():
    stated = _pinned_digest_words(PINNED_TEXP_POLICY, "texp")
    _assert_same_entries(
        stated,
        _digest_words("texp"),
        "the texp lines of prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt "
        "against the texp digest built here",
    )


def test_pinned_policy_tsilu_lines():
    stated = _pinned_digest_words(PINNED_TSILU_POLICY, "tsilu")
    _assert_same_entries(
        stated,
        _digest_words("tsilu"),
        "the tsilu lines of integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/policy.txt "
        "against the tsilu digest built here",
    )
