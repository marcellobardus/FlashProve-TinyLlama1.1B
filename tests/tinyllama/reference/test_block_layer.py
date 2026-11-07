import math
from functools import lru_cache

import numpy as np
import pytest

from tinyllama.reference.layer import RangeFailure, layer
from tinyllama.reference.tables import exp_table, rope_tables, silu_table

D = 8
H = 2
G = 1
DH = 4
F = 6
N = 3
T = 4

RECORD_NAMES = [
    "x",
    "a_sigma",
    "a_Rc",
    "a_w",
    "a",
    "qw",
    "q",
    "kw",
    "k",
    "vw",
    "v",
    "o",
    "x_mid",
    "m_sigma",
    "m_Rc",
    "m_w",
    "m",
    "gw",
    "key",
    "s",
    "u",
]

RECORD_SHAPES = {
    "x": (N, D),
    "a_sigma": (N,),
    "a_Rc": (N,),
    "a_w": (N, D),
    "a": (N, D),
    "qw": (N, H, DH),
    "q": (N, H, DH),
    "kw": (N, G, DH),
    "k": (N, G, DH),
    "vw": (N, G, DH),
    "v": (N, G, DH),
    "o": (N, H, DH),
    "x_mid": (N, D),
    "m_sigma": (N,),
    "m_Rc": (N,),
    "m_w": (N, D),
    "m": (N, D),
    "gw": (N, F),
    "key": (N, F),
    "s": (N, F),
    "u": (N, F),
}


@lru_cache(maxsize=None)
def _tables():
    cosines, sines = rope_tables(T, DH, 10000)
    return {
        "C": cosines,
        "S": sines,
        "EX": exp_table(),
        "Ts": silu_table(),
    }


@lru_cache(maxsize=None)
def _weights():
    rng = np.random.default_rng(20261104)
    return {
        "wq": rng.integers(-4095, 4096, size=(H * DH, D), dtype=np.int64),
        "wk": rng.integers(-4095, 4096, size=(G * DH, D), dtype=np.int64),
        "wv": rng.integers(-4095, 4096, size=(G * DH, D), dtype=np.int64),
        "wo": rng.integers(-4095, 4096, size=(D, H * DH), dtype=np.int64),
        "wg": rng.integers(-4095, 4096, size=(F, D), dtype=np.int64),
        "wu": rng.integers(-4095, 4096, size=(F, D), dtype=np.int64),
        "wd": rng.integers(-4095, 4096, size=(D, F), dtype=np.int64),
    }


@lru_cache(maxsize=None)
def _x():
    rng = np.random.default_rng(77123)
    return rng.integers(-100, 101, size=(N, D), dtype=np.int64)


def _scales():
    return {
        "kq": 25,
        "kk": 25,
        "kv": 12,
        "ko": 12,
        "kg": 12,
        "ku": 12,
        "kd": 12,
        "M1": 24,
        "K1": 17,
        "M2": 24,
        "K2": 17,
        "sh": 8,
        "eps1": 1,
        "eps2": 1,
        "rp": 1,
    }


def _run(x=None, weights=None, scales=None, tables=None, layer_index=7):
    x = _x() if x is None else x
    weights = _weights() if weights is None else weights
    scales = _scales() if scales is None else scales
    tables = _tables() if tables is None else tables
    records = []

    def record(name, index, values):
        records.append((name, index, np.array(values, dtype=np.int64, copy=True)))

    result = layer(x, weights, scales, tables, record=record, layer_index=layer_index)
    return result, records


def test_block_output_shapes_dtypes_and_record_order():
    result, records = _run(layer_index=7)
    assert result.shape == (N, D)
    assert result.dtype == np.int64
    assert [name for name, _, _ in records] == RECORD_NAMES
    for name, index, values in records:
        assert index == 7
        assert values.dtype == np.int64
        assert values.shape == RECORD_SHAPES[name], name


def test_block_is_deterministic():
    first, first_records = _run()
    second, second_records = _run()
    assert np.array_equal(first, second)
    assert len(first_records) == len(second_records)
    for (name_a, index_a, value_a), (name_b, index_b, value_b) in zip(
        first_records, second_records
    ):
        assert (name_a, index_a) == (name_b, index_b)
        assert np.array_equal(value_a, value_b), name_a


def test_block_recorded_activations_stay_in_their_contract_intervals():
    _, records = _run()
    by_name = {name: values for name, _, values in records}
    for name in ("q", "k", "v", "a", "m"):
        assert int(np.max(np.abs(by_name[name]))) <= 2047, name
    keys = by_name["key"]
    assert int(keys.min()) >= -32768
    assert int(keys.max()) <= 32767


def test_block_condition_8_fails_on_a_huge_stream_value():
    x = _x().copy()
    x[0, 0] = 547599908736
    with pytest.raises(RangeFailure) as info:
        layer(x, _weights(), _scales(), _tables(), layer_index=4)
    failure = info.value
    assert failure.condition == 8
    assert failure.layer == 4
    assert failure.position == 0
    assert failure.index == 0
    assert failure.value == 547599908736


def test_block_condition_3_fails_when_a_query_shift_is_too_small():
    scales = _scales()
    scales["kq"] = 1
    scales["kk"] = 1
    with pytest.raises(RangeFailure) as info:
        layer(_x(), _weights(), scales, _tables(), layer_index=2)
    assert info.value.condition == 3
    assert info.value.layer == 2


def _as_int64(value):
    return np.array(value, dtype=np.int64)


def test_block_hand_checked_tiny_case():
    # Hand computation, n = 1, d = 2, H = 1, G = 1, dh = 2, f = 1.
    # x = [10, -20], first norm M1 = 24, K1 = 17, eps1 = 1:
    #   Vx = 100 + 400 = 500, sigma = isqrt(501) = 22,
    #   Rc = (2^24 + 11) // 22 = 762601,
    #   w = [rq(10*762601, 17), rq(-20*762601, 17)] = [58, -116].
    # With wq = wk = wv = identity, C = 4096, S = 0 and kq = kk = 12:
    #   p = [4096*58, 4096*(-116)] and q = k = [58, -116].
    # With kv = 1: vw = [rq(58, 1), rq(-116, 1)] = [29, -58], v = same.
    # One position gives score 58^2 + 116^2 = 16820, u = 0, e = EX[0] = 4096,
    # Z = 4096, so o = v = [29, -58].
    # wo = identity and ko = 1 give ad = [rq(29, 1), rq(-58, 1)] = [15, -29],
    # hence x_mid = [10 + 15, -20 - 29] = [25, -49].
    # Second norm, same M and K:
    #   Vx = 625 + 2401 = 3026, sigma = isqrt(3027) = 55,
    #   Rc = (2^24 + 27) // 55 = 305040,
    #   m_w = [rq(25*305040, 17), rq(-49*305040, 17)] = [58, -114].
    # wg = wu = [[1, 0]], kg = ku = 1: gw = [rq(58, 1)] = [29], u = [29].
    # key = 29 (g0 = 29, gh = 0, gc = 0) and we set Ts[32768 + 29] = 64.
    # wd = [[1], [1]], kd = 6: ad = [rq(64*29, 6), rq(64*29, 6)] = [29, 29],
    # so x_out = [25 + 29, -49 + 29] = [54, -20].

    x = np.array([[10, -20]], dtype=np.int64)
    weights = {
        "wq": np.array([[1, 0], [0, 1]], dtype=np.int64),
        "wk": np.array([[1, 0], [0, 1]], dtype=np.int64),
        "wv": np.array([[1, 0], [0, 1]], dtype=np.int64),
        "wo": np.array([[1, 0], [0, 1]], dtype=np.int64),
        "wg": np.array([[1, 0]], dtype=np.int64),
        "wu": np.array([[1, 0]], dtype=np.int64),
        "wd": np.array([[1], [1]], dtype=np.int64),
    }
    tables = {
        "C": np.array([[4096]], dtype=np.int64),
        "S": np.array([[0]], dtype=np.int64),
        "EX": [4096] + [0] * 2306,
        "Ts": np.zeros(65536, dtype=np.int64),
    }
    tables["Ts"][32768 + 29] = 64
    scales = {
        "kq": 12,
        "kk": 12,
        "kv": 1,
        "ko": 1,
        "kg": 1,
        "ku": 1,
        "kd": 6,
        "M1": 24,
        "K1": 17,
        "M2": 24,
        "K2": 17,
        "sh": 8,
        "eps1": 1,
        "eps2": 1,
        "rp": 1,
    }

    output, records = _run(
        x=x, weights=weights, scales=scales, tables=tables, layer_index=0
    )
    by_name = {name: values for name, _, values in records}

    assert output.tolist() == [[54, -20]]
    assert by_name["x"].tolist() == [[10, -20]]
    assert by_name["a_sigma"].tolist() == [22]
    assert by_name["a_Rc"].tolist() == [762601]
    assert by_name["a_w"].tolist() == [[58, -116]]
    assert by_name["a"].tolist() == [[58, -116]]
    assert by_name["qw"].tolist() == [[[58, -116]]]
    assert by_name["q"].tolist() == [[[58, -116]]]
    assert by_name["kw"].tolist() == [[[58, -116]]]
    assert by_name["k"].tolist() == [[[58, -116]]]
    assert by_name["vw"].tolist() == [[[29, -58]]]
    assert by_name["v"].tolist() == [[[29, -58]]]
    assert by_name["o"].tolist() == [[[29, -58]]]
    assert by_name["x_mid"].tolist() == [[25, -49]]
    assert by_name["m_sigma"].tolist() == [55]
    assert by_name["m_Rc"].tolist() == [305040]
    assert by_name["m_w"].tolist() == [[58, -114]]
    assert by_name["m"].tolist() == [[58, -114]]
    assert by_name["gw"].tolist() == [[29]]
    assert by_name["key"].tolist() == [[29]]
    assert by_name["s"].tolist() == [[64]]
    assert by_name["u"].tolist() == [[29]]


def _slow_rq(a, k):
    return (a + (1 << (k - 1))) >> k


def _slow_clamp(a, bound):
    return min(max(a, -bound), bound)


def _slow_low8(a):
    return ((a + 128) % 256) - 128


def _slow_softmax_div(acc, z):
    return (2 * acc + z) // (2 * z)


def _slow_matvec(vector, matrix):
    return [
        sum(vector[column] * matrix[row][column] for column in range(len(vector)))
        for row in range(len(matrix))
    ]


def _slow_layer(x, weights, scales, tables, record=None, layer_index=0):
    """A scalar, pure-Python-integer copy of the layer, written with loops only."""
    n = len(x)
    d = len(x[0])
    cosines = tables["C"]
    sines = tables["S"]
    expo = tables["EX"]
    tsilu = tables["Ts"]
    dh = 2 * len(cosines[0])
    heads = len(weights["wq"]) // dh
    groups = len(weights["wk"]) // dh
    hidden = len(weights["wg"])
    group = heads // groups

    def norm(vector, m, k, eps):
        vx = sum(value * value for value in vector)
        sigma = math.isqrt(vx + eps)
        rc = ((1 << m) + (sigma // 2)) // sigma
        w = [_slow_rq(value * rc, k) for value in vector]
        nr = [_slow_clamp(value, 2047) for value in w]
        return sigma, rc, w, nr

    a_sigmas = []
    a_rcs = []
    a_ws = []
    a_values = []
    qw_values = []
    q_values = []
    kw_values = []
    k_values = []
    vw_values = []
    v_values = []

    for t in range(n):
        sigma, rc, w, a = norm(x[t], scales["M1"], scales["K1"], scales["eps1"])
        a_sigmas.append(sigma)
        a_rcs.append(rc)
        a_ws.append(w)
        a_values.append(a)

        aq = _slow_matvec(a, weights["wq"])
        ak = _slow_matvec(a, weights["wk"])
        av = _slow_matvec(a, weights["wv"])
        qw_head = []
        q_head = []
        for head in range(heads):
            row = [
                _slow_rq(
                    cosines[t][i] * aq[head * dh + 2 * i]
                    - sines[t][i] * aq[head * dh + 2 * i + 1],
                    scales["kq"],
                )
                for i in range(dh // 2)
            ]
            row1 = [
                _slow_rq(
                    sines[t][i] * aq[head * dh + 2 * i]
                    + cosines[t][i] * aq[head * dh + 2 * i + 1],
                    scales["kq"],
                )
                for i in range(dh // 2)
            ]
            wide = []
            for i in range(dh // 2):
                wide.extend([row[i], row1[i]])
            qw_head.append(wide)
            q_head.append([_slow_clamp(value, 2047) for value in wide])
        qw_values.append(qw_head)
        q_values.append(q_head)

        kw_head = []
        k_head = []
        for head in range(groups):
            row = [
                _slow_rq(
                    cosines[t][i] * ak[head * dh + 2 * i]
                    - sines[t][i] * ak[head * dh + 2 * i + 1],
                    scales["kk"],
                )
                for i in range(dh // 2)
            ]
            row1 = [
                _slow_rq(
                    sines[t][i] * ak[head * dh + 2 * i]
                    + cosines[t][i] * ak[head * dh + 2 * i + 1],
                    scales["kk"],
                )
                for i in range(dh // 2)
            ]
            wide = []
            for i in range(dh // 2):
                wide.extend([row[i], row1[i]])
            kw_head.append(wide)
            k_head.append([_slow_clamp(value, 2047) for value in wide])
        kw_values.append(kw_head)
        k_values.append(k_head)

        vw_head = []
        v_head = []
        for head in range(groups):
            wide = [
                _slow_rq(av[head * dh + c], scales["kv"]) for c in range(dh)
            ]
            vw_head.append(wide)
            v_head.append([_slow_clamp(value, 2047) for value in wide])
        vw_values.append(vw_head)
        v_values.append(v_head)

    o_values = []
    for position in range(n):
        o_position = [[0] * dh for _ in range(heads)]
        for head in range(heads):
            key_head = head // group
            scores = []
            for j in range(position + 1):
                scores.append(
                    sum(
                        q_values[position][head][c] * k_values[j][key_head][c]
                        for c in range(dh)
                    )
                )
            maximum = max(scores)
            weights_exp = []
            for score in scores:
                u = (maximum - score) >> scales["sh"]
                weights_exp.append(expo[u] if u < 2307 else 0)
            z = sum(weights_exp)
            for c in range(dh):
                acc = sum(
                    weights_exp[j] * v_values[j][key_head][c]
                    for j in range(position + 1)
                )
                o_position[head][c] = _slow_softmax_div(acc, z)
        o_values.append(o_position)

    x_mid_values = []
    m_sigmas = []
    m_rcs = []
    m_ws = []
    m_values = []
    gw_values = []
    key_values = []
    s_values = []
    u_values = []

    for t in range(n):
        o_flat = [
            o_values[t][head][c] for head in range(heads) for c in range(dh)
        ]
        projected = _slow_matvec(o_flat, weights["wo"])
        attention_ad = [_slow_rq(value, scales["ko"]) for value in projected]
        x_mid = [x[t][k] + attention_ad[k] for k in range(d)]
        x_mid_values.append(x_mid)

        sigma, rc, w, m = norm(x_mid, scales["M2"], scales["K2"], scales["eps2"])
        m_sigmas.append(sigma)
        m_rcs.append(rc)
        m_ws.append(w)
        m_values.append(m)

        gate_projected = _slow_matvec(m, weights["wg"])
        up_projected = _slow_matvec(m, weights["wu"])
        gw_row = []
        key_row = []
        s_row = []
        u_row = []
        for j in range(hidden):
            gw = _slow_rq(gate_projected[j], scales["kg"])
            g0 = _slow_low8(gw)
            gh = (gw - g0) // 256
            gc = _slow_clamp(gh, 127)
            key = 256 * gc + g0
            gw_row.append(gw)
            key_row.append(key)
            s_row.append(tsilu[key + 32768])
            u_row.append(_slow_rq(up_projected[j], scales["ku"]))
        gw_values.append(gw_row)
        key_values.append(key_row)
        s_values.append(s_row)
        u_values.append(u_row)

    if record is not None:
        record("x", layer_index, [row[:] for row in x])
        record("a_sigma", layer_index, a_sigmas)
        record("a_Rc", layer_index, a_rcs)
        record("a_w", layer_index, a_ws)
        record("a", layer_index, a_values)
        record("qw", layer_index, qw_values)
        record("q", layer_index, q_values)
        record("kw", layer_index, kw_values)
        record("k", layer_index, k_values)
        record("vw", layer_index, vw_values)
        record("v", layer_index, v_values)
        record("o", layer_index, o_values)
        record("x_mid", layer_index, x_mid_values)
        record("m_sigma", layer_index, m_sigmas)
        record("m_Rc", layer_index, m_rcs)
        record("m_w", layer_index, m_ws)
        record("m", layer_index, m_values)
        record("gw", layer_index, gw_values)
        record("key", layer_index, key_values)
        record("s", layer_index, s_values)
        record("u", layer_index, u_values)

    output = []
    for t in range(n):
        down = [
            sum(
                s_values[t][j] * u_values[t][j] * weights["wd"][k][j]
                for j in range(hidden)
            )
            for k in range(d)
        ]
        delta = [_slow_rq(value, scales["kd"]) for value in down]
        output.append(
            [x_mid_values[t][k] + delta[k] for k in range(d)]
        )
    return output


def test_block_matches_a_slow_scalar_pure_python_reference():
    weights = _weights()
    tables = _tables()
    x = _x()
    scales = _scales()

    fast_records = []

    def fast_record(name, index, values):
        fast_records.append((name, index, np.array(values, copy=True)))

    fast_output = layer(
        x, weights, scales, tables, record=fast_record, layer_index=5
    )

    slow_records = []
    slow_output = _slow_layer(
        x.tolist(),
        {name: value.tolist() for name, value in weights.items()},
        scales,
        {
            "C": tables["C"].tolist(),
            "S": tables["S"].tolist(),
            "EX": list(tables["EX"]),
            "Ts": tables["Ts"].tolist(),
        },
        record=lambda name, index, values: slow_records.append(
            (name, index, values)
        ),
        layer_index=5,
    )

    assert fast_output.tolist() == slow_output
    assert [name for name, _, _ in fast_records] == [
        name for name, _, _ in slow_records
    ]
    for (fast_name, fast_index, fast_value), (
        slow_name,
        slow_index,
        slow_value,
    ) in zip(fast_records, slow_records):
        assert fast_name == slow_name
        assert fast_index == slow_index == 5
        assert fast_value.tolist() == slow_value, fast_name


def test_block_records_contain_no_float_arrays():
    _, records = _run()
    for name, _, values in records:
        assert not np.issubdtype(values.dtype, np.floating), name
        assert values.dtype == np.int64, name
