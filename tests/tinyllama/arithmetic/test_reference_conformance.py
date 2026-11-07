import pytest

mod = pytest.importorskip("tinyllama.reference")

import tl_conformance as guard
import tl_ref_independent as ref


def test_every_i7_first_name_is_exported():
    guard.require_all_i7_names(mod)


@pytest.mark.parametrize("spec", guard.PRIMITIVES, ids=lambda spec: spec["id"])
def test_reference_primitive_matches_vectors(spec):
    name, fn = guard.require_primitive(mod, spec)
    for case in ref.load_vectors(spec["vectors"])[spec["cases"]]:
        result = guard.call_primitive(fn, name, spec, case)
        got, expected = spec["check"](result, case)
        assert got == expected, (name, case, got, expected)
