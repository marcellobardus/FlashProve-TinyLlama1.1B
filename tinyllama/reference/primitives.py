"""Integer primitives of contract TL-INT-1 (ARITHMETIC_CONTRACT.md section 2).

Every function here takes and returns Python integers (or lists of Python
integers).  No floating-point value is used anywhere in this module.
"""

import math

from ..contract import CW, EG, PAD_ID, T as WINDOW_TOKENS, U0

ACTIVATION_BOUND = (1 << (CW - 1)) - 1
GATE_CLAMP_BOUND = (1 << (EG - 1)) - 1

__all__ = [
    "rq",
    "clamp",
    "isqrt",
    "floor_div",
    "low8",
    "softmax_div",
    "exp_lookup",
    "argmax",
    "gate_key",
    "norm",
    "pad",
]


def rq(a, k):
    """floor((a + 2^(k-1)) / 2^k); a signed right shift with ties up."""
    if k < 1:
        raise ValueError("rq needs k >= 1")
    return (a + (1 << (k - 1))) >> k


def clamp(a, bound):
    """Symmetric clamp: min(max(a, -bound), bound)."""
    return min(max(a, -bound), bound)


def isqrt(a):
    """Largest r with r^2 <= a; ValueError for a < 0."""
    return math.isqrt(a)


def floor_div(a, b):
    """floor(a / b) for b > 0."""
    if b <= 0:
        raise ValueError("floor_div needs b > 0")
    return a // b


def low8(a):
    """The balanced low byte: ((a + 128) mod 256) - 128."""
    return ((a + 128) % 256) - 128


def softmax_div(acc, z):
    """acc / z rounded to nearest with ties up, i.e. (2 acc + z) // (2 z)."""
    if z <= 0:
        raise ValueError("softmax_div needs z > 0")
    return (2 * acc + z) // (2 * z)


def exp_lookup(u, exp_table):
    """EX[u] for u < 2307, otherwise 0; ValueError for u < 0."""
    if u < 0:
        raise ValueError("exp_lookup needs u >= 0")
    if u < U0:
        return exp_table[u]
    return 0


def argmax(values):
    """Index of the first maximum, found with an explicit loop."""
    if not values:
        raise ValueError("argmax needs at least one value")
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def gate_key(gw):
    """The gcl 1 gate key: g0, gh, gc and key = 256 gc + g0."""
    g0 = low8(gw)
    gh = (gw - g0) // 256
    gc = clamp(gh, GATE_CLAMP_BOUND)
    key = 256 * gc + g0
    return {"g0": g0, "gh": gh, "gc": gc, "key": key}


def norm(x, m, k, eps):
    """N(x; M, K, eps) from contract section 8."""
    values = [int(value) for value in x]
    vx = sum(value * value for value in values)
    sigma = isqrt(vx + int(eps))
    rc = ((1 << int(m)) + (sigma // 2)) // sigma
    w = [rq(value * rc, int(k)) for value in values]
    nr = [clamp(value, ACTIVATION_BOUND) for value in w]
    return {"vx": vx, "sigma": sigma, "rc": rc, "w": w, "nr": nr}


def pad(ids):
    """Right-pad token ids with PAD_ID to the 1024-token window."""
    values = list(ids)
    if len(values) > WINDOW_TOKENS:
        raise ValueError("more than 1024 tokens")
    return values + [PAD_ID] * (WINDOW_TOKENS - len(values))
