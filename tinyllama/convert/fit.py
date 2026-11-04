# Run on the pod through the converter (repository at /workspace/repo,
# checkpoint at /workspace/ckpt, calibration npy files at /workspace/calib):
#   cd /workspace/repo && PYTHONPATH=. python3 -m tinyllama.convert.run \
#       --checkpoint /workspace/ckpt --out /workspace/repo/artifacts/TL-INT-1 \
#       --calib /workspace/calib/fit.npy --calib-sha256 <FIT_SHA256> \
#       --heldout /workspace/calib/heldout.npy --heldout-sha256 <HELDOUT_SHA256> \
#       --report /workspace/repo/artifacts/TL-INT-1/range-report.json
# This module has no command line of its own.

"""The TL-INT-1 scale fit of contract 5.2, in pure Python integers.

The maxima come from tinyllama.convert.calibrate. The exponents come from
tinyllama.convert.fold. The formulas and the refusal rules below are the ones
of ARITHMETIC_CONTRACT.md section 5.2, with t = 5 and c = sqrt(2), the d = 2048
values of contract 4.3. The artifact format fixes c0_bits, so these constants
are used for a test-only small shape too.

The default e_res is 11, the FlashProve default, and must be in 3 .. 14.
"""

import math

import numpy as np

from tinyllama import contract


CELL_BITS = 12
CELL_MAX = 2047
NORM_K = 32
NORM_SPAN = 14
T_CONTRACT = 5
C_CONTRACT = math.sqrt(2.0)
SILU_OUT_EXPONENT = 6
TOKEN_SITE_EXPONENT = 8
SH_LEAST = 4
SH_MOST = 12
SHIFT_LEAST = 1
SHIFT_MOST = 32

SHIFT_NAMES = ("kq", "kk", "kv", "ko", "kg", "ku", "kd")
KL_ORDER = (
    "kq",
    "kk",
    "kv",
    "ko",
    "kg",
    "ku",
    "kd",
    "M1",
    "K1",
    "M2",
    "K2",
    "sh",
    "rp",
)


class FitError(ValueError):
    """The maxima, the exponents or a fitted word violate contract 5.2."""


def cell_exponent(mx):
    """The largest integer E with mx * 2^E <= 2047, for a positive finite mx."""
    if isinstance(mx, bool) or not isinstance(mx, (int, float)):
        raise FitError("a maximum must be a number")
    value = float(mx)
    if not math.isfinite(value) or value <= 0.0:
        raise FitError("a maximum must be finite and positive")
    if value < 1e-300:
        raise FitError("a maximum is below 1e-300")
    mantissa, exponent = math.frexp(value)
    if mantissa * 2 ** (CELL_BITS - 1) <= CELL_MAX:
        return CELL_BITS - 1 - exponent
    return CELL_BITS - 2 - exponent


def _layer_maxima(values, site, layers):
    if not isinstance(values, dict):
        raise FitError("the maxima of site {} are not an object".format(site))
    result = {}
    for layer in range(layers):
        if layer in values:
            result[layer] = values[layer]
        elif str(layer) in values:
            result[layer] = values[str(layer)]
        else:
            raise FitError(
                "the maximum of site {}, layer {} is missing".format(site, layer)
            )
    return result


def _require_int(value, where):
    if isinstance(value, bool) or not isinstance(value, int):
        raise FitError("{} is not an integer".format(where))
    return value


def _exponent_layer(exponents, kind, layer):
    table = exponents.get(kind)
    if not isinstance(table, dict):
        raise FitError("the exponents of {} are missing".format(kind))
    if str(layer) in table:
        return _require_int(table[str(layer)], "the exponent of {} layer {}".format(kind, layer))
    if layer in table:
        return _require_int(table[layer], "the exponent of {} layer {}".format(kind, layer))
    raise FitError("the exponent of {} layer {} is missing".format(kind, layer))


def _exponent_one(exponents, kind):
    if kind not in exponents:
        raise FitError("the exponent of {} is missing".format(kind))
    return _require_int(exponents[kind], "the exponent of {}".format(kind))


def check_norm_pair(label, m, k):
    """Refuse a norm pair outside the contract 5.2 and verifier ranges."""
    m = _require_int(m, "{} M".format(label))
    k = _require_int(k, "{} K".format(label))
    if not 24 <= m <= 47:
        raise FitError("{} M is {}, outside 24 .. 47".format(label, m))
    if not 17 <= k <= 32:
        raise FitError("{} K is {}, outside 17 .. 32".format(label, k))
    if k < m - NORM_SPAN:
        raise FitError(
            "{} K {} is below M - 14 = {}".format(label, k, m - NORM_SPAN)
        )
    if k > m:
        raise FitError("{} K {} is above M {}".format(label, k, m))
    return m, k


def epsilon_from_e_res(e_res):
    """floor((4096 * 4^e_res + 100000) / 200000) of contract 5.2."""
    e_res = _require_int(e_res, "e_res")
    if not 3 <= e_res <= 14:
        raise FitError("e_res is {}, outside 3 .. 14".format(e_res))
    eps = (4096 * (4**e_res) + 100000) // 200000
    if eps == 0:
        raise FitError("e_res {} gives eps 0, which is refused".format(e_res))
    return eps


def _check_shift(name, value):
    value = _require_int(value, "shift {}".format(name))
    if not SHIFT_LEAST <= value <= SHIFT_MOST:
        raise FitError(
            "shift {} is {}, outside 1 .. 32".format(name, value)
        )
    return value


def fit_scales(maxima, exponents, dims, *, e_res=11, t=T_CONTRACT, c=C_CONTRACT):
    """Fit the KL words and the shared params by contract 5.2.

    maxima is {site: {layer: maximum}} with sites n1, n2, q, k, v, u and nF.
    nF is keyed by dims.L. exponents is {"q", "k", "v", "o", "gate", "up",
    "down"} each mapping layer to the tensor exponent, plus "lm_head" (or
    "output") for the logit matrix. The result is a FitResult.
    """
    eps = epsilon_from_e_res(e_res)
    layers = int(dims.L)
    if layers < 1:
        raise FitError("the shape has no layers")
    if not 1 <= int(t) <= 20:
        raise FitError("t is {}; it must be a small positive integer".format(t))
    c = float(c)
    if not math.isfinite(c) or c <= 0.0:
        raise FitError("c must be finite and positive")

    sites = ("n1", "n2", "q", "k", "v", "u")
    measured = {
        site: _layer_maxima(maxima.get(site), site, layers) for site in sites
    }
    if "nF" not in maxima:
        raise FitError("the maximum of site nF is missing")
    nF_values = maxima["nF"]
    if not isinstance(nF_values, dict):
        raise FitError("the maxima of site nF are not an object")
    if layers in nF_values:
        nF_max = nF_values[layers]
    elif str(layers) in nF_values:
        nF_max = nF_values[str(layers)]
    elif len(nF_values) == 1:
        nF_max = next(iter(nF_values.values()))
    else:
        raise FitError("the maximum of site nF, layer {} is missing".format(layers))

    cell_es = {}
    for layer in range(layers):
        cell_es[layer] = {
            "n1": min(cell_exponent(measured["n1"][layer] / c), NORM_SPAN - t),
            "n2": min(cell_exponent(measured["n2"][layer] / c), NORM_SPAN - t),
            "q": cell_exponent(measured["q"][layer]),
            "k": cell_exponent(measured["k"][layer]),
            "v": cell_exponent(measured["v"][layer]),
            "u": cell_exponent(measured["u"][layer]),
        }
    final_es = min(cell_exponent(nF_max / c), NORM_SPAN - t)

    sh = min(cell_es[layer]["q"] + cell_es[layer]["k"] - TOKEN_SITE_EXPONENT for layer in range(layers))
    if not SH_LEAST <= sh <= SH_MOST:
        raise FitError(
            "sh is {}, outside 4 .. 12; check the q and k maxima".format(sh)
        )
    for layer in range(layers):
        cell_es[layer]["q"] = TOKEN_SITE_EXPONENT + sh - cell_es[layer]["k"]

    rows = []
    for layer in range(layers):
        es = cell_es[layer]
        m1, k1 = check_norm_pair("M1/K1 layer {}".format(layer), NORM_K + es["n1"] + t, NORM_K)
        m2, k2 = check_norm_pair("M2/K2 layer {}".format(layer), NORM_K + es["n2"] + t, NORM_K)

        eq = _exponent_layer(exponents, "q", layer)
        ek = _exponent_layer(exponents, "k", layer)
        ev = _exponent_layer(exponents, "v", layer)
        eo = _exponent_layer(exponents, "o", layer)
        eg = _exponent_layer(exponents, "gate", layer)
        eu = _exponent_layer(exponents, "up", layer)
        ed = _exponent_layer(exponents, "down", layer)

        row = {
            "kq": _check_shift("kq layer {}".format(layer), es["n1"] + eq - es["q"] + contract.CS),
            "kk": _check_shift("kk layer {}".format(layer), es["n1"] + ek - es["k"] + contract.CS),
            "kv": _check_shift("kv layer {}".format(layer), es["n1"] + ev - es["v"]),
            "ko": _check_shift("ko layer {}".format(layer), es["v"] + eo - int(e_res)),
            "kg": _check_shift("kg layer {}".format(layer), es["n2"] + eg - contract.EG),
            "ku": _check_shift("ku layer {}".format(layer), es["n2"] + eu - es["u"]),
            "kd": _check_shift(
                "kd layer {}".format(layer),
                SILU_OUT_EXPONENT + es["u"] + ed - int(e_res),
            ),
            "M1": m1,
            "K1": k1,
            "M2": m2,
            "K2": k2,
            "sh": sh,
            "rp": 1,
        }
        if row["kq"] != row["kk"] and (row["kq"] < 16 or row["kk"] < 16):
            raise FitError(
                "layer {} has kq {} != kk {} while one is below 16".format(
                    layer, row["kq"], row["kk"]
                )
            )
        rows.append(row)

    mf, kf = check_norm_pair("MF/KF", NORM_K + final_es + t, NORM_K)
    if "lm_head" in exponents:
        e_logit = _exponent_one(exponents, "lm_head")
    else:
        e_logit = _exponent_one(exponents, "output")
    s = final_es + e_logit
    if not 0 <= s <= 63:
        raise FitError("s = Es(nF) + e(lm_head) is {}, outside 0 .. 63".format(s))

    kl = np.asarray(
        [[row[name] for name in KL_ORDER] for row in rows], dtype=np.uint32
    )
    shared = {
        "eps1": [eps] * layers,
        "eps2": [eps] * layers,
        "epsF": eps,
        "MF": mf,
        "KF": kf,
        "s": s,
        "sh": sh,
        "e_res": int(e_res),
        "t": int(t),
        "c": c,
        "Es_nF": final_es,
    }
    return FitResult(kl=kl, rows=rows, shared=shared, cells=cell_es)


class FitResult:
    """The KL array and the fitted shared values."""

    def __init__(self, kl, rows, shared, cells):
        self.kl = kl
        self.rows = rows
        self.shared = shared
        self.cells = cells

    def __repr__(self):
        return "FitResult(kl={!r}, shared={!r})".format(self.kl.shape, self.shared)


def _dims_constants(dims):
    return [
        ("L", dims.L),
        ("d", dims.D),
        ("H", dims.H),
        ("G", dims.G),
        ("dh", dims.DH),
        ("f", dims.F),
        ("V", dims.V),
        ("T", dims.T),
    ]


def _other_constants():
    return [
        ("NE", contract.NE),
        ("lt", contract.LT),
        ("cs", contract.CS),
        ("theta0", contract.THETA0),
        ("S", contract.S),
        ("rx", contract.RX),
        ("eg", contract.EG),
        ("smax", contract.SMAX),
        ("cw", contract.CW),
        ("weight_bound", contract.WEIGHT_BOUND),
        ("embedding_bound", contract.EMB_BOUND),
        ("u0", contract.U0),
        ("c0_bits", contract.C0_BITS),
        ("cq_bits", contract.CQ_BITS),
        ("pad_id", contract.PAD_ID),
        ("gcl", contract.GCL),
    ]


def build_params(result, dims, *, digests, tables_sha256):
    """Assemble the params.json object that write_artifact expects."""
    if not isinstance(digests, dict):
        raise FitError("digests must be an object")
    for name in ("trope", "texp", "tsilu"):
        words = digests.get(name)
        if (
            not isinstance(words, (list, tuple))
            or len(words) != 8
            or not all(isinstance(word, int) and not isinstance(word, bool) for word in words)
        ):
            raise FitError("digests.{} is not a list of eight integers".format(name))
    if not isinstance(tables_sha256, dict):
        raise FitError("tables_sha256 must be an object")

    params = {"contract": contract.CONTRACT_ID}
    for name, value in _dims_constants(dims) + _other_constants():
        params[name] = int(value)
    params["range_conditions"] = {
        str(key): list(value) for key, value in contract.RANGE_CONDITIONS.items()
    }
    params.update(
        {
            "e_res": result.shared["e_res"],
            "eps1": list(result.shared["eps1"]),
            "eps2": list(result.shared["eps2"]),
            "epsF": result.shared["epsF"],
            "MF": result.shared["MF"],
            "KF": result.shared["KF"],
            "s": result.shared["s"],
            "digests": {name: list(digests[name]) for name in ("trope", "texp", "tsilu")},
            "tables_sha256": dict(tables_sha256),
        }
    )
    return params
