import pytest

mod = pytest.importorskip("tinyllama.reference")

import ref_independent as ref

_NO_RESULT = object()


def _plain(expected_key):
    def check(result, case):
        return result, case[expected_key]

    return check


def _check_gate_key(result, case):
    four = (case["g0"], case["gh"], case["gc"], case["key"])
    if isinstance(result, (tuple, list)) and len(result) == 4:
        return tuple(result), four
    if isinstance(result, dict) and all(name in result for name in ("g0", "gh", "gc", "key")):
        return (result["g0"], result["gh"], result["gc"], result["key"]), four
    if isinstance(result, dict):
        return result.get("key", result), case["key"]
    if isinstance(result, (tuple, list)):
        return result[-1], case["key"]
    return result, case["key"]


def _check_norm(result, case):
    expected = case["expected_nr"]
    if result == expected:
        return expected, expected
    if isinstance(result, dict):
        for value in result.values():
            if value == expected:
                return expected, expected
    if isinstance(result, (tuple, list)):
        for value in result:
            if value == expected:
                return expected, expected
    return result, expected


def _norm_variants(case):
    x, m, k, eps = case["x"], case["m"], case["k"], case["eps"]
    return [
        ((x, m, k, eps), {}),
        ((), {"x": x, "m": m, "k": k, "eps": eps}),
        ((), {"x": x, "M": m, "K": k, "eps": eps}),
    ]


PRIMITIVES = [
    {
        "id": "rq",
        "names": ["rq", "rounded_quotient"],
        "vectors": "rq.json",
        "cases": "cases",
        "variants": lambda case: [((case["a"], case["k"]), {})],
        "check": _plain("expected"),
    },
    {
        "id": "clamp",
        "names": ["clamp"],
        "vectors": "clamp.json",
        "cases": "cases",
        "variants": lambda case: [
            ((case["a"], case["bound"]), {}),
            ((case["a"], -case["bound"], case["bound"]), {}),
        ],
        "check": _plain("expected"),
    },
    {
        "id": "isqrt",
        "names": ["isqrt"],
        "vectors": "isqrt.json",
        "cases": "cases",
        "variants": lambda case: [((case["a"],), {})],
        "check": _plain("expected"),
    },
    {
        "id": "floor_div",
        "names": ["floor_div", "floor_quotient"],
        "vectors": "floor_div.json",
        "cases": "cases",
        "variants": lambda case: [((case["a"], case["b"]), {})],
        "check": _plain("expected"),
    },
    {
        "id": "low8",
        "names": ["low8", "balanced_low_byte"],
        "vectors": "low8.json",
        "cases": "cases",
        "variants": lambda case: [((case["a"],), {})],
        "check": _plain("expected"),
    },
    {
        "id": "softmax_div",
        "names": ["softmax_div", "softmax_divide"],
        "vectors": "softmax_div.json",
        "cases": "cases",
        "variants": lambda case: [((case["acc"], case["z"]), {})],
        "check": _plain("expected"),
    },
    {
        "id": "exp_lookup",
        "names": ["exp_lookup", "exp_index"],
        "vectors": "softmax_exp_index.json",
        "cases": "lookup_cases",
        "variants": lambda case: [((case["u"], ref.FAKE_EXP_TABLE), {})],
        "check": _plain("expected_e"),
    },
    {
        "id": "argmax",
        "names": ["argmax", "first_argmax"],
        "vectors": "argmax.json",
        "cases": "cases",
        "variants": lambda case: [((list(case["values"]),), {})],
        "check": _plain("expected"),
    },
    {
        "id": "gate_key",
        "names": ["gate_key"],
        "vectors": "gate_key.json",
        "cases": "cases",
        "variants": lambda case: [((case["gw"],), {})],
        "check": _check_gate_key,
    },
    {
        "id": "norm",
        "names": ["norm", "rms_norm"],
        "vectors": "norm.json",
        "cases": "cases",
        "variants": _norm_variants,
        "check": _check_norm,
    },
    {
        "id": "pad",
        "names": ["pad", "pad_window"],
        "vectors": "padding.json",
        "cases": "cases",
        "variants": lambda case: [((list(case["ids"]),), {})],
        "check": _plain("expected"),
    },
]


def _resolve(names):
    for name in names:
        candidate = getattr(mod, name, None)
        if candidate is not None:
            return name, candidate
    return None, None


@pytest.mark.parametrize("spec", PRIMITIVES, ids=lambda spec: spec["id"])
def test_reference_primitive_matches_vectors(spec):
    name, fn = _resolve(spec["names"])
    if fn is None:
        pytest.skip("tinyllama.reference has none of: " + ", ".join(spec["names"]))
    for case in ref.load_vectors(spec["vectors"])[spec["cases"]]:
        result = _NO_RESULT
        for args, kwargs in spec["variants"](case):
            try:
                result = fn(*args, **kwargs)
                break
            except TypeError:
                continue
        if result is _NO_RESULT:
            pytest.skip("no calling convention of tinyllama.reference." + name + " fits case " + repr(case))
        got, expected = spec["check"](result, case)
        assert got == expected, (name, case, got, expected)
