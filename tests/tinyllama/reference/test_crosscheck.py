import pytest

torch = pytest.importorskip("torch")

from tinyllama.reference import crosscheck


@pytest.fixture(scope="module")
def result():
    return crosscheck.run()


# These primitives have no intentional deviation at the tiny shape: both sides
# use the same weights, scales, exponent values and adjacent-pair rotary.  The
# tabulated digests are built from FlashProve's own table functions and hashed
# with table_digest(), so make_tables() accepts them honestly.
KNOWN_EQUAL = (
    "rq.k1",
    "rq.k4",
    "rq.k32",
    "clamp.b2047",
    "floor_div.d3",
    "floor_div.d256",
    "low8",
    "gate_key.g0",
    "gate_key.gh",
    "gate_key.gc",
    "gate_key.key",
    "rope_c",
    "rope_s",
    "softmax_table",
    "silu_table",
    "norm.sigma",
    "norm.Rc",
    "norm.w",
    "norm.nr",
    "rotary",
    "attention.a",
    "attention.qw",
    "attention.q",
    "attention.kw",
    "attention.k",
    "attention.vw",
    "attention.v",
    "attention.o",
    "attention.x_mid",
    "feed_forward.m",
    "feed_forward.gw",
    "feed_forward.key",
    "feed_forward.s",
    "feed_forward.u",
    "feed_forward.x_out",
    "extend.x_last",
    "extend.aF",
    "extend.lam",
    "extend.first_choice",
    "extend.next_token",
)


def test_comparison_runs_and_returns_per_primitive_results(result):
    assert isinstance(result, crosscheck.CrosscheckResult)
    assert result.comparisons
    assert set(result.by_name()) >= set(KNOWN_EQUAL)
    assert len(result.lines()) == len(result.comparisons)
    for item in result.comparisons:
        assert isinstance(item, crosscheck.Comparison)
        assert item.status in ("EQUAL", "DIFFERENT", "NOT-COMPARABLE")
        assert item.line().startswith(item.name + " ")


def test_known_equal_primitives_are_equal(result):
    by_name = result.by_name()
    for name in KNOWN_EQUAL:
        assert by_name[name].status == "EQUAL", by_name[name].line()


def test_differences_and_refusals_are_reported_not_hidden(result):
    for item in result.comparisons:
        if item.status == "DIFFERENT":
            assert item.first_index is not None
            assert item.ours is not None
            assert item.flashprove is not None
            assert item.count >= 1
            assert "first_index=" in item.line()
            assert "ours=" in item.line()
            assert "flashprove=" in item.line()
            assert "count=" in item.line()
        elif item.status == "NOT-COMPARABLE":
            assert item.reason
            assert "reason=" in item.line()


def test_every_comparison_equal_at_the_tiny_shape(result):
    unexpected = [
        item.line() for item in result.comparisons if item.status != "EQUAL"
    ]
    assert unexpected == []
    assert result.ok
    assert result.exit_code == 0
