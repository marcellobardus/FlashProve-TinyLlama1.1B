import json
import math
from pathlib import Path

VECTORS_DIR = Path(__file__).resolve().parent / "vectors"

ACTIVATION_BOUND = 2047
GATE_CLAMP_BOUND = 127
WINDOW_TOKENS = 1024
PAD_TOKEN_ID = 0
EXP_INDEX_LIMIT = 2307
STREAM_VALUE_BOUND = 547599908735
CONDITION1_LIMIT = 1 << 78


def rq(a, k):
    return (a + (1 << (k - 1))) // (1 << k)


def clamp(a, bound):
    return min(max(a, -bound), bound)


def isqrt(a):
    return math.isqrt(a)


def isqrt_bisect(a):
    low, high = 0, 1 << (a.bit_length() // 2 + 1)
    while low < high:
        mid = (low + high + 1) // 2
        if mid * mid <= a:
            low = mid
        else:
            high = mid - 1
    return low


def floor_div(a, b):
    return a // b


def low8(a):
    return ((a + 128) % 256) - 128


def softmax_div(acc, z):
    return (2 * acc + z) // (2 * z)


def exp_lookup(u, exp_table):
    if u < EXP_INDEX_LIMIT:
        return exp_table[u]
    return 0


def norm(x, m, k, eps):
    vx = sum(value * value for value in x)
    sigma = isqrt(vx + eps)
    rc = ((1 << m) + (sigma // 2)) // sigma
    w = [rq(value * rc, k) for value in x]
    nr = [clamp(value, ACTIVATION_BOUND) for value in w]
    return {"vx": vx, "sigma": sigma, "rc": rc, "w": w, "nr": nr}


def gate_key(gw):
    g0 = low8(gw)
    gh = (gw - g0) // 256
    gc = clamp(gh, GATE_CLAMP_BOUND)
    key = 256 * gc + g0
    return {"g0": g0, "gh": gh, "gc": gc, "key": key}


def argmax(values):
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


VALUE_CONDITION_INTERVALS = {
    2: (-32768, 32639),
    3: (-32768, 32639),
    4: (-32768, 32639),
    6: (-16384, 16383),
    7: (-32768, 32639),
}


def value_condition_ok(number, value):
    if number == 8:
        return abs(value) <= STREAM_VALUE_BOUND
    low, high = VALUE_CONDITION_INTERVALS[number]
    return low <= value <= high


def condition1_ok(vx, eps):
    return vx + eps < CONDITION1_LIMIT


FAKE_EXP_TABLE = [4096 - u for u in range(EXP_INDEX_LIMIT)]


def pad(ids):
    if len(ids) > WINDOW_TOKENS:
        raise ValueError("more than 1024 tokens")
    return list(ids) + [PAD_TOKEN_ID] * (WINDOW_TOKENS - len(ids))


def load_vectors(name):
    return json.loads((VECTORS_DIR / name).read_text())
