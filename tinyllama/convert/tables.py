"""The public lookup tables and their FlashProve digests.

This is a torch free port of the pure Python parts of FlashProve's
``integrations/vllm/integer_model/integer_model/tables.py``.  The recipes are
the ones of docs/model.md, "4 Public tables".
"""

import math
import struct

# The SiLU table has the signed keys -32,768 .. 32,767.
KEY_OFFSET_16 = 1 << 15

_TABLE_NUMBER = 72
DW, SW, BLKW = 8, 16, 32


def _round_half_away(value):
    """rha(v): floor(v + 0.5) if v >= 0, else -floor(-v + 0.5)."""
    return (
        math.floor(value + 0.5)
        if value >= 0
        else -math.floor(-value + 0.5)
    )


def rope_tables(T, dh, cs, theta0):
    """Return (C, S), each a flat list over t < T and i < dh / 2.

    Entry (t, i) is at index t * (dh / 2) + i.
    """
    angles = [
        float(t) * math.pow(float(theta0), (-2.0 * i) / dh)
        for t in range(T)
        for i in range(dh // 2)
    ]
    scale = float(1 << cs)
    return (
        [_round_half_away(scale * math.cos(x)) for x in angles],
        [_round_half_away(scale * math.sin(x)) for x in angles],
    )


def softmax_table(exp_s, rx):
    """EX[u] for u = 0 .. u0 - 1, where u0 is the least u with EX[u] = 0."""
    table = []
    while True:
        entry = round(float(1 << exp_s) * math.exp(-len(table) / float(1 << rx)))
        if entry == 0:
            return table
        table.append(entry)


def silu_table(eg, smax):
    """Ts[c] for the keys c = -32,768 .. 32,767."""
    table = []
    for c in range(-KEY_OFFSET_16, KEY_OFFSET_16):
        z = min(max(c, -(64 << eg)), (64 << eg) - 1) / float(1 << eg)
        s = z / (1.0 + math.exp(-z))
        table.append(min(max(_round_half_away(64.0 * s), -smax), smax))
    return table


_MASK64 = 2**64 - 1


def _rotr64(word, count):
    return ((word >> count) | (word << (64 - count))) & _MASK64


_IV = (
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
)

_blake2b_sigma = (
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


def _blake2b_init_param(outlen, keylen):
    param = bytes([outlen, keylen, 1, 1]) + bytes(60)
    return [
        _IV[i] ^ int.from_bytes(param[8 * i : 8 * i + 8], "little")
        for i in range(8)
    ]


def _G(r, i, a, b, c, d, m):
    a = (a + b + m[_blake2b_sigma[r][2 * i + 0]]) & _MASK64
    d = _rotr64(d ^ a, 32)
    c = (c + d) & _MASK64
    b = _rotr64(b ^ c, 24)
    a = (a + b + m[_blake2b_sigma[r][2 * i + 1]]) & _MASK64
    d = _rotr64(d ^ a, 16)
    c = (c + d) & _MASK64
    b = _rotr64(b ^ c, 63)
    return a, b, c, d


def _ROUND(r, v, m):
    v[0], v[4], v[8], v[12] = _G(r, 0, v[0], v[4], v[8], v[12], m)
    v[1], v[5], v[9], v[13] = _G(r, 1, v[1], v[5], v[9], v[13], m)
    v[2], v[6], v[10], v[14] = _G(r, 2, v[2], v[6], v[10], v[14], m)
    v[3], v[7], v[11], v[15] = _G(r, 3, v[3], v[7], v[11], v[15], m)
    v[0], v[5], v[10], v[15] = _G(r, 4, v[0], v[5], v[10], v[15], m)
    v[1], v[6], v[11], v[12] = _G(r, 5, v[1], v[6], v[11], v[12], m)
    v[2], v[7], v[8], v[13] = _G(r, 6, v[2], v[7], v[8], v[13], m)
    v[3], v[4], v[9], v[14] = _G(r, 7, v[3], v[4], v[9], v[14], m)


def _compress(h, m, t_low, t_high, last):
    v = [0] * 16
    for i in range(8):
        v[i] = h[i]
    v[8] = _IV[0]
    v[9] = _IV[1]
    v[10] = _IV[2]
    v[11] = _IV[3]
    v[12] = _IV[4] ^ t_low
    v[13] = _IV[5] ^ t_high
    v[14] = _IV[6] ^ (_MASK64 if last else 0)
    v[15] = _IV[7]
    for r in range(12):
        _ROUND(r, v, m)
    return [h[i] ^ v[i] ^ v[i + 8] for i in range(8)]


def hash_words(num, words):
    """The digest H(num; words), eight 32-bit words.

    BLAKE2b-512 without a key over the words as little-endian bytes, with the
    high half of the byte counter set to num + 2**32 * len(words).
    """
    count = len(words)
    if (
        not 0 <= num < 2**32
        or count >= 2**32
        or not all(0 <= word < 2**32 for word in words)
    ):
        raise ValueError(
            "H is defined for a number, a list length and words from 0 to 2^32 - 1"
        )
    blocks = max(1, -(-count // BLKW))
    high = num + (count << 32)
    h = _blake2b_init_param(64, 0)
    for k in range(blocks):
        block = words[BLKW * k : BLKW * (k + 1)]
        m = struct.unpack(
            "<16Q",
            struct.pack("<%dI" % len(block), *block).ljust(4 * BLKW, b"\0"),
        )
        last = k == blocks - 1
        h = _compress(
            h,
            m,
            4 * count if last else 4 * BLKW * (k + 1),
            high,
            last,
        )
    return list(struct.unpack("<%dI" % SW, struct.pack("<8Q", *h))[:DW])


def table_digest(words):
    """The digest of a public table, as eight 32-bit words."""
    return hash_words(_TABLE_NUMBER, words)


def trope_words(cosines, sines):
    """The words that ``trope`` hashes: C + 4096, then S + 4096."""
    return [entry + 4096 for entry in cosines] + [
        entry + 4096 for entry in sines
    ]


def texp_words(exp_table):
    """The words that ``texp`` hashes: EX itself."""
    return list(exp_table)


def tsilu_words(silu):
    """The words that ``tsilu`` hashes: Ts + 4096."""
    return [entry + 4096 for entry in silu]
