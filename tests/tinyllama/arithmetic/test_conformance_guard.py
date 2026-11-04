import types

import pytest

import tl_conformance as guard


def _fake_module(**attributes):
    module = types.ModuleType("fake_tinyllama_reference")
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


def test_missing_primitive_fails_with_the_i7_names():
    module = _fake_module()
    with pytest.raises(pytest.fail.Exception) as caught:
        guard.require_primitive(module, guard.SPEC_BY_ID["rq"])
    message = str(caught.value)
    assert "rq" in message
    assert "rounded_quotient" in message


def test_alias_is_accepted_when_the_first_name_is_missing():
    module = _fake_module(rounded_quotient=lambda a, k: 0)
    name, fn = guard.require_primitive(module, guard.SPEC_BY_ID["rq"])
    assert name == "rounded_quotient"
    assert fn(1, 2) == 0


def test_missing_first_name_is_listed_even_when_an_alias_exists():
    module = _fake_module(rounded_quotient=lambda a, k: 0)
    with pytest.raises(pytest.fail.Exception) as caught:
        guard.require_all_i7_names(module)
    message = str(caught.value)
    assert "rq" in message
    assert "clamp" in message
    assert "pad" in message


def test_all_first_names_present_passes():
    module = _fake_module(**{name: (lambda *args: 0) for name in guard.I7_FIRST_NAMES})
    guard.require_all_i7_names(module)


def test_calling_convention_mismatch_fails_instead_of_skipping():
    def wrong(x, y, z):
        return x + y + z

    spec = guard.SPEC_BY_ID["rq"]
    name, fn = guard.require_primitive(_fake_module(rq=wrong), spec)
    with pytest.raises(pytest.fail.Exception) as caught:
        guard.call_primitive(fn, name, spec, {"a": 1, "k": 2})
    assert "no calling convention" in str(caught.value)
    assert "rq" in str(caught.value)


def test_call_primitive_reports_the_result_of_the_first_fitting_variant():
    spec = guard.SPEC_BY_ID["clamp"]
    name, fn = guard.require_primitive(_fake_module(clamp=lambda a, bound: a), spec)
    result = guard.call_primitive(fn, name, spec, {"a": 9, "bound": 3})
    assert result == 9
