"""Offline construction of the public lookup tables (contract section 6).

Floating point is used in this module only to build the tables offline, as
the contract permits.  The layer receives integer tables only and never
evaluates exp, cos or sin.

The recipes follow docs/model.md section 4:

* C, S: x = t * theta0^(-2 i / dh), rha(2^12 cos x), rha(2^12 sin x).
* EX: rne(2^12 exp(-u / 2^8)) until the first zero.
* Ts: c' = clamp(c, -16384, 16383), z = c' / 256,
  rha(64 z / (1 + exp(-z))), clamped to -4095 .. 4095.
"""

import hashlib
import math
import struct

import numpy as np

from ..contract import DH, EG, EXP_S, RX, SMAX, T as WINDOW_TOKENS, THETA0, U0

CONTRACT_SHA256 = {
    "trope": "973450425e0f39b307e1561d7aaa70904b0fab3d5862e453b70ccfd7c60915c1",
    "texp": "e27fdb2a6e9b273138ee66a6729de3c2c186ac1e81d24e5514ad29c155221b63",
    "tsilu": "0c26f9b15535c1bb9d1878552b5122158670d6eadd8e9f3ba988833e87e7a442",
}

__all__ = [
    "CONTRACT_SHA256",
    "rha",
    "rne",
    "rope_tables",
    "exp_table",
    "silu_table",
    "sha256_words",
    "verify_contract_tables",
]


def rha(value):
    """Round half away from zero (the float is only used offline)."""
    return math.floor(value + 0.5) if value >= 0 else -math.floor(-value + 0.5)


def rne(value):
    """Round to nearest, ties to even (the float is only used offline)."""
    return int(round(value))


def rope_tables(T, dh, theta0):
    """Return (C, S), each an int64 array of shape [T][dh // 2]."""
    if T < 0 or dh < 2 or dh % 2:
        raise ValueError("rope_tables needs T >= 0 and an even dh >= 2")
    pairs = dh // 2
    cosines = np.empty((T, pairs), dtype=np.int64)
    sines = np.empty((T, pairs), dtype=np.int64)
    for t in range(T):
        for i in range(pairs):
            b = (-2 * i) / dh
            weight = math.pow(theta0, b)
            x = t * weight
            cosines[t, i] = rha(4096 * math.cos(x))
            sines[t, i] = rha(4096 * math.sin(x))
    return cosines, sines


def exp_table():
    """EX[u] = rne(2^12 exp(-u / 2^8)), ending before the first zero."""
    table = []
    while True:
        entry = rne((1 << EXP_S) * math.exp(-len(table) / float(1 << RX)))
        if entry == 0:
            return table
        table.append(entry)


def silu_table():
    """Ts indexed by key + 32768, an int64 array of length 65536."""
    table = np.empty(1 << 16, dtype=np.int64)
    one = 1 << EG
    for c in range(-(1 << 15), 1 << 15):
        c_clamped = min(max(c, -64 * one), 64 * one - 1)
        z = c_clamped / one
        value = rha(64 * z / (1 + math.exp(-z)))
        table[c + (1 << 15)] = min(max(value, -SMAX), SMAX)
    return table


def sha256_words(words):
    """sha256 of 32-bit little-endian words (the +4096 offsets are the caller's)."""
    values = [int(word) for word in words]
    if not all(0 <= word < (1 << 32) for word in values):
        raise ValueError("table words must be unsigned 32-bit integers")
    return hashlib.sha256(struct.pack("<%dI" % len(values), *values)).hexdigest()


def verify_contract_tables():
    """Build the full-size tables and check the three sha256 values of section 6.

    Returns a dictionary from table name to the computed digest.  Raises
    AssertionError on the first mismatch.
    """
    cosines, sines = rope_tables(WINDOW_TOKENS, DH, THETA0)
    expo = exp_table()
    silu = silu_table()
    words = {
        "trope": [int(v) + 4096 for v in cosines.ravel()]
        + [int(v) + 4096 for v in sines.ravel()],
        "texp": expo,
        "tsilu": [int(v) + 4096 for v in silu],
    }
    computed = {name: sha256_words(value) for name, value in words.items()}
    for name, expected in CONTRACT_SHA256.items():
        if computed[name] != expected:
            raise AssertionError(
                "%s sha256 is %s, expected %s" % (name, computed[name], expected)
            )
    if len(expo) != U0:
        raise AssertionError("exp_table length is %d, expected %d" % (len(expo), U0))
    return computed
