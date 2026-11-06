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


def rotary_pair(a0, a1, c, s):
    return c * a0 - s * a1, s * a0 + c * a1


def _matmul_row(x, matrix):
    out = [0] * len(matrix[0])
    for i, value in enumerate(x):
        if value:
            row = matrix[i]
            for j in range(len(out)):
                out[j] += value * row[j]
    return out


def _rotary_value(tables, name, t, i):
    table = tables[name]
    if callable(table):
        return table(t, i)
    return table[t][i]


def _exp_value(tables, u):
    table = tables["EX"]
    if callable(table):
        return table(u)
    if isinstance(table, dict):
        return table.get(u, 0)
    if u < len(table):
        return table[u]
    return 0


def _silu_value(tables, key):
    table = tables["Ts"]
    if callable(table):
        return table(key)
    if isinstance(table, dict):
        return table[key]
    index = key + tables.get("ts_offset", 0)
    if not 0 <= index < len(table):
        raise IndexError("Ts key outside the passed table: " + repr(key))
    return table[index]


def layer(x_rows, weights, scales, tables):
    n = len(x_rows)
    d = len(x_rows[0])
    heads = scales["H"]
    kv_heads = scales["G"]
    head_dim = scales["dh"]
    ffn = scales["f"]
    if scales.get("d", d) != d:
        raise ValueError("scales d differs from the rows")
    if heads % kv_heads:
        raise ValueError("H must be a multiple of G")
    if head_dim % 2:
        raise ValueError("dh must be even")
    dq = heads * head_dim
    dkv = kv_heads * head_dim
    group = heads // kv_heads

    def apply_norm(rows, m, k, eps):
        return [norm(row, m, k, eps)["nr"] for row in rows]

    a = apply_norm(x_rows, scales["M1"], scales["K1"], scales["eps1"])
    aq = [_matmul_row(row, weights["wq"]) for row in a]
    ak = [_matmul_row(row, weights["wk"]) for row in a]

    qw = [[[0] * head_dim for _ in range(heads)] for _ in range(n)]
    q = [[[0] * head_dim for _ in range(heads)] for _ in range(n)]
    kw = [[[0] * head_dim for _ in range(kv_heads)] for _ in range(n)]
    k = [[[0] * head_dim for _ in range(kv_heads)] for _ in range(n)]

    for t in range(n):
        for h in range(heads):
            base = h * head_dim
            for i in range(head_dim // 2):
                c = _rotary_value(tables, "C", t, i)
                s = _rotary_value(tables, "S", t, i)
                p0, p1 = rotary_pair(aq[t][base + 2 * i], aq[t][base + 2 * i + 1], c, s)
                qw[t][h][2 * i] = rq(p0, scales["kq"])
                qw[t][h][2 * i + 1] = rq(p1, scales["kq"])
                q[t][h][2 * i] = clamp(qw[t][h][2 * i], ACTIVATION_BOUND)
                q[t][h][2 * i + 1] = clamp(qw[t][h][2 * i + 1], ACTIVATION_BOUND)
        for g in range(kv_heads):
            base = g * head_dim
            for i in range(head_dim // 2):
                c = _rotary_value(tables, "C", t, i)
                s = _rotary_value(tables, "S", t, i)
                p0, p1 = rotary_pair(ak[t][base + 2 * i], ak[t][base + 2 * i + 1], c, s)
                kw[t][g][2 * i] = rq(p0, scales["kk"])
                kw[t][g][2 * i + 1] = rq(p1, scales["kk"])
                k[t][g][2 * i] = clamp(kw[t][g][2 * i], ACTIVATION_BOUND)
                k[t][g][2 * i + 1] = clamp(kw[t][g][2 * i + 1], ACTIVATION_BOUND)

    vw = [[[0] * head_dim for _ in range(kv_heads)] for _ in range(n)]
    v = [[[0] * head_dim for _ in range(kv_heads)] for _ in range(n)]
    for t in range(n):
        av = _matmul_row(a[t], weights["wv"])
        for g in range(kv_heads):
            for c in range(head_dim):
                vw[t][g][c] = rq(av[g * head_dim + c], scales["kv"])
                v[t][g][c] = clamp(vw[t][g][c], ACTIVATION_BOUND)

    scores = [[None] * heads for _ in range(n)]
    u_softmax = [[None] * heads for _ in range(n)]
    e_weights = [[None] * heads for _ in range(n)]
    z_values = [[0] * heads for _ in range(n)]
    o = [[[0] * head_dim for _ in range(heads)] for _ in range(n)]

    for t in range(n):
        for h in range(heads):
            g = h // group
            qv = q[t][h]
            row_scores = []
            for j in range(t + 1):
                kv = k[j][g]
                row_scores.append(sum(qv[c] * kv[c] for c in range(head_dim)))
            row_max = max(row_scores)
            row_u = []
            row_e = []
            for score in row_scores:
                u = floor_div(row_max - score, 1 << scales["sh"])
                row_u.append(u)
                row_e.append(_exp_value(tables, u))
            z = sum(row_e)
            acc = [0] * head_dim
            for j in range(t + 1):
                weight = row_e[j]
                for c in range(head_dim):
                    acc[c] += weight * v[j][g][c]
            scores[t][h] = row_scores
            u_softmax[t][h] = row_u
            e_weights[t][h] = row_e
            z_values[t][h] = z
            o[t][h] = [softmax_div(acc[c], z) for c in range(head_dim)]

    x_mid = []
    for t in range(n):
        flat_o = [o[t][h][c] for h in range(heads) for c in range(head_dim)]
        projected = _matmul_row(flat_o, weights["wo"])
        ad = [rq(value, scales["ko"]) for value in projected]
        x_mid.append([x_rows[t][k] + ad[k] for k in range(d)])

    m = apply_norm(x_mid, scales["M2"], scales["K2"], scales["eps2"])
    gw = [[0] * ffn for _ in range(n)]
    key = [[0] * ffn for _ in range(n)]
    s = [[0] * ffn for _ in range(n)]
    u = [[0] * ffn for _ in range(n)]

    for t in range(n):
        gate_raw = _matmul_row(m[t], weights["wg"])
        up_raw = _matmul_row(m[t], weights["wu"])
        for j in range(ffn):
            gw[t][j] = rq(gate_raw[j], scales["kg"])
            gate = gate_key(gw[t][j])
            key[t][j] = gate["key"]
            s[t][j] = _silu_value(tables, gate["key"])
            u[t][j] = rq(up_raw[j], scales["ku"])

    x_out = []
    for t in range(n):
        projected = [0] * d
        for j in range(ffn):
            product = s[t][j] * u[t][j]
            if product:
                row = weights["wd"][j]
                for col in range(d):
                    projected[col] += product * row[col]
        ad = [rq(value, scales["kd"]) for value in projected]
        x_out.append([x_mid[t][k] + ad[k] for k in range(d)])

    return {
        "a": a,
        "qw": qw,
        "q": q,
        "kw": kw,
        "k": k,
        "vw": vw,
        "v": v,
        "scores": scores,
        "u_softmax": u_softmax,
        "e": e_weights,
        "Z": z_values,
        "o": o,
        "x_mid": x_mid,
        "m": m,
        "gw": gw,
        "key": key,
        "s": s,
        "u": u,
        "x_out": x_out,
    }


def load_vectors(name):
    return json.loads((VECTORS_DIR / name).read_text())
