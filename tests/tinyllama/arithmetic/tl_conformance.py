"""Checking helpers for the INTERFACES.md I7 reference primitives.

This module has no dependency on ``tinyllama.reference`` so that
``test_conformance_guard.py`` can exercise the failure paths with a fake module.
"""

import pytest

import tl_ref_independent as ref

I7_FIRST_NAMES = [
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

SPEC_BY_ID = {spec["id"]: spec for spec in PRIMITIVES}


def resolve(module, names):
    for name in names:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return name, candidate
    return None, None


def missing_i7_names(module):
    return [name for name in I7_FIRST_NAMES if getattr(module, name, None) is None]


def require_all_i7_names(module):
    missing = missing_i7_names(module)
    if missing:
        pytest.fail(
            "tinyllama.reference is missing I7 package-level name(s): "
            + ", ".join(missing)
        )


def require_primitive(module, spec):
    name, fn = resolve(module, spec["names"])
    if fn is None:
        pytest.fail(
            "tinyllama.reference has none of the I7 names for "
            + spec["id"]
            + ": "
            + ", ".join(spec["names"])
        )
    return name, fn


def call_primitive(fn, name, spec, case):
    last_error = None
    for args, kwargs in spec["variants"](case):
        try:
            return fn(*args, **kwargs)
        except TypeError as error:
            last_error = error
    pytest.fail(
        "no calling convention of tinyllama.reference."
        + name
        + " fits case "
        + repr(case)
        + " (last TypeError: "
        + repr(last_error)
        + ")"
    )
