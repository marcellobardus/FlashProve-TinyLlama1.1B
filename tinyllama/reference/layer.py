"""One integer transformer layer of contract TL-INT-1 (sections 8 and 9).

The path is integer only: every product and recorded value is int64, and the
norm's Vx = sum x[k]^2 is accumulated with Python integers so that it stays
exact beyond 64 bits.  The tables are already integer arrays; no float value
is read or produced here.
"""

import numpy as np

from ..contract import CW, RANGE_CONDITIONS, U0
from .primitives import isqrt, rq

ACTIVATION_BOUND = (1 << (CW - 1)) - 1
GATE_CLAMP_BOUND = (1 << 7) - 1
WINDOW_TABLE_SIZE = 1 << 16
GATE_TABLE_OFFSET = 1 << 15

__all__ = ["RangeFailure", "layer"]


class RangeFailure(Exception):
    """A contract range condition failed; it is never replaced by saturation."""

    def __init__(self, condition, name, layer, position, index, value):
        self.condition = int(condition)
        self.name = name
        self.layer = int(layer)
        self.position = int(position)
        self.index = int(index)
        self.value = value
        super().__init__(
            "range condition %d (%s) failed at layer %d position %d index %d: %r"
            % (self.condition, self.name, self.layer, self.position, self.index, self.value)
        )


def _as_int64(value, name):
    array = np.asarray(value)
    assert array.dtype == np.int64, "%s must have dtype int64, got %s" % (name, array.dtype)
    return array


def _interval(condition):
    entry = RANGE_CONDITIONS[condition]
    return entry[1], entry[2]


def _check_scalar_vector(condition, name, values, layer):
    low, high = _interval(condition)
    for position, value in enumerate(values):
        value = int(value)
        if value < low or value > high:
            raise RangeFailure(condition, name, layer, position, 0, value)


def _check_matrix(condition, name, values, layer):
    low, high = _interval(condition)
    rows = values.reshape(values.shape[0], -1)
    for position in range(rows.shape[0]):
        row = rows[position]
        bad = np.flatnonzero((row < low) | (row > high))
        if bad.size:
            index = int(bad[0])
            raise RangeFailure(condition, name, layer, position, index, int(row[index]))


def _clamp_array(values, bound):
    return np.minimum(np.maximum(values, -bound), bound)


def _vx_exact(row):
    """Exact sum of squares, using int64 only when that cannot overflow."""
    if row.size == 0:
        return 0
    largest = max(-int(row.min()), int(row.max()))
    if largest * largest * row.size <= (1 << 63) - 1:
        return int(np.dot(row, row))
    total = 0
    for value in row.tolist():
        total += value * value
    return total


def _norm_rows(x, m, k, eps, layer, x_name, sigma_name, rc_name, w_name, condition1_name):
    """N(x; M, K, eps) per row, with the contract's range checks."""
    n = x.shape[0]
    _check_matrix(8, x_name, x, layer)

    vx_plus_eps = [_vx_exact(x[position]) + int(eps) for position in range(n)]
    _check_scalar_vector(1, condition1_name, vx_plus_eps, layer)

    sigma = np.empty(n, dtype=np.int64)
    rc = np.empty(n, dtype=np.int64)
    for position, value in enumerate(vx_plus_eps):
        root = isqrt(value)
        sigma[position] = root
        rc[position] = ((1 << int(m)) + (root // 2)) // root

    w = rq(x * rc.reshape(n, 1), int(k))
    _check_matrix(2, w_name, w, layer)
    nr = _clamp_array(w, ACTIVATION_BOUND)
    return sigma, rc, w, nr


def _infer_shape(weights, tables, n):
    wq = weights["wq"]
    wk = weights["wk"]
    wv = weights["wv"]
    wo = weights["wo"]
    wg = weights["wg"]
    wu = weights["wu"]
    wd = weights["wd"]
    cosines = _as_int64(tables["C"], "C")
    sines = _as_int64(tables["S"], "S")
    tsilu = _as_int64(tables["Ts"], "Ts")

    if cosines.ndim != 2 or sines.shape != cosines.shape:
        raise ValueError("C and S must be int64 arrays of the same 2-D shape")
    if tsilu.ndim != 1 or tsilu.shape[0] != WINDOW_TABLE_SIZE:
        raise ValueError("Ts must be one-dimensional with 65536 entries")

    dh = 2 * cosines.shape[1]
    if dh <= 0 or dh % 2:
        raise ValueError("dh must be a positive even number")
    if wq.ndim != 2:
        raise ValueError("wq must be two-dimensional")
    d = wq.shape[1]
    if wq.shape[0] % dh:
        raise ValueError("wq rows are not a multiple of dh")
    H = wq.shape[0] // dh
    if wk.shape[0] % dh:
        raise ValueError("wk rows are not a multiple of dh")
    G = wk.shape[0] // dh
    if H < 1 or G < 1 or d < 1:
        raise ValueError("empty model dimensions")
    if H % G != 0:
        raise ValueError("H must be divisible by G")
    f = wg.shape[0]
    if f < 1:
        raise ValueError("f must be positive")

    expected = {
        "wq": (H * dh, d),
        "wk": (G * dh, d),
        "wv": (G * dh, d),
        "wo": (d, H * dh),
        "wg": (f, d),
        "wu": (f, d),
        "wd": (d, f),
    }
    for name, shape in expected.items():
        if weights[name].shape != shape:
            raise ValueError(
                "%s has shape %r, expected %r" % (name, weights[name].shape, shape)
            )
    if cosines.shape[0] < n:
        raise ValueError("the rotary tables have fewer than n positions")
    return d, H, G, dh, f


def _rotary(values, cosines, sines):
    """Apply neighbouring-pair rotary to [n][heads][dh] integer values."""
    n, heads, dh = values.shape
    pairs = dh // 2
    even = values[:, :, 0::2]
    odd = values[:, :, 1::2]
    c = cosines[:n].reshape(n, 1, pairs)
    s = sines[:n].reshape(n, 1, pairs)
    rotated = np.empty_like(values)
    rotated[:, :, 0::2] = c * even - s * odd
    rotated[:, :, 1::2] = s * even + c * odd
    return rotated


def layer(x, weights, scales, tables, record=None, layer_index=0):
    """Apply one TL-INT-1 transformer layer to n prefix positions.

    x has shape [n][d] and dtype int64.  The weights use the stored HF layout
    [out][in].  The layer's shape (d, H, G, dh, f) is inferred from the weight
    matrices and from the rotary tables, whose width is dh // 2.  Returns the
    next stream, an int64 array of shape [n][d].
    """
    x = _as_int64(x, "x")
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    n = x.shape[0]
    if n < 1:
        raise ValueError("x must have at least one position")

    wq = _as_int64(weights["wq"], "wq")
    wk = _as_int64(weights["wk"], "wk")
    wv = _as_int64(weights["wv"], "wv")
    wo = _as_int64(weights["wo"], "wo")
    wg = _as_int64(weights["wg"], "wg")
    wu = _as_int64(weights["wu"], "wu")
    wd = _as_int64(weights["wd"], "wd")
    matrices = {"wq": wq, "wk": wk, "wv": wv, "wo": wo, "wg": wg, "wu": wu, "wd": wd}

    cosines = _as_int64(tables["C"], "C")
    sines = _as_int64(tables["S"], "S")
    tsilu = _as_int64(tables["Ts"], "Ts")
    exp_values = np.asarray(tables["EX"], dtype=np.int64)

    d, H, G, dh, f = _infer_shape(matrices, tables, n)
    if x.shape[1] != d:
        raise ValueError("x has %d columns, the weights expect %d" % (x.shape[1], d))

    M1 = int(scales["M1"])
    K1 = int(scales["K1"])
    M2 = int(scales["M2"])
    K2 = int(scales["K2"])
    kq = int(scales["kq"])
    kk = int(scales["kk"])
    kv = int(scales["kv"])
    ko = int(scales["ko"])
    kg = int(scales["kg"])
    ku = int(scales["ku"])
    kd = int(scales["kd"])
    sh = int(scales["sh"])
    eps1 = int(scales["eps1"])
    eps2 = int(scales["eps2"])
    group = H // G

    if record is not None:
        record("x", layer_index, x)

    a_sigma, a_rc, a_w, a = _norm_rows(
        x, M1, K1, eps1, layer_index, "x", "a_sigma", "a_Rc", "a_w", "a_sigma"
    )
    if record is not None:
        record("a_sigma", layer_index, a_sigma)
        record("a_Rc", layer_index, a_rc)
        record("a_w", layer_index, a_w)
        record("a", layer_index, a)

    aq = (a @ wq.T).reshape(n, H, dh)
    q_wide = rq(_rotary(aq, cosines, sines), kq)
    _check_matrix(3, "qw", q_wide, layer_index)
    q = _clamp_array(q_wide, ACTIVATION_BOUND)

    ak = (a @ wk.T).reshape(n, G, dh)
    k_wide = rq(_rotary(ak, cosines, sines), kk)
    _check_matrix(3, "kw", k_wide, layer_index)
    k = _clamp_array(k_wide, ACTIVATION_BOUND)

    av = (a @ wv.T).reshape(n, G, dh)
    v_wide = rq(av, kv)
    _check_matrix(4, "vw", v_wide, layer_index)
    v = _clamp_array(v_wide, ACTIVATION_BOUND)

    if record is not None:
        record("qw", layer_index, q_wide)
        record("q", layer_index, q)
        record("kw", layer_index, k_wide)
        record("k", layer_index, k)
        record("vw", layer_index, v_wide)
        record("v", layer_index, v)

    o = np.empty((n, H, dh), dtype=np.int64)
    for head in range(H):
        key_head = head // group
        query = q[:, head, :]
        key = k[:, key_head, :]
        scores = query @ key.T
        for position in range(n):
            row = scores[position, : position + 1]
            maximum = int(row.max())
            u = (maximum - row) >> sh
            e = np.zeros(position + 1, dtype=np.int64)
            inside = u < U0
            e[inside] = exp_values[u[inside]]
            z = int(e.sum())
            acc = e @ v[: position + 1, key_head, :]
            o[position, head, :] = (2 * acc + z) // (2 * z)

    if record is not None:
        record("o", layer_index, o)

    o_flat = o.reshape(n, H * dh)
    ad = rq(o_flat @ wo.T, ko)
    x_mid = x + ad
    if record is not None:
        record("x_mid", layer_index, x_mid)

    m_sigma, m_rc, m_w, m = _norm_rows(
        x_mid, M2, K2, eps2, layer_index, "x_mid", "m_sigma", "m_Rc", "m_w", "m_sigma"
    )
    if record is not None:
        record("m_sigma", layer_index, m_sigma)
        record("m_Rc", layer_index, m_rc)
        record("m_w", layer_index, m_w)
        record("m", layer_index, m)

    gate_wide = rq(m @ wg.T, kg)
    g0 = ((gate_wide + 128) % 256) - 128
    gh = (gate_wide - g0) // 256
    _check_matrix(6, "gh", gh, layer_index)
    gc = _clamp_array(gh, GATE_CLAMP_BOUND)
    key = 256 * gc + g0
    s = tsilu[key + GATE_TABLE_OFFSET]

    u = rq(m @ wu.T, ku)
    _check_matrix(7, "u", u, layer_index)

    if record is not None:
        record("gw", layer_index, gate_wide)
        record("key", layer_index, key)
        record("s", layer_index, s)
        record("u", layer_index, u)

    down = (s * u) @ wd.T
    ad2 = rq(down, kd)
    x_out = x_mid + ad2
    return x_out
