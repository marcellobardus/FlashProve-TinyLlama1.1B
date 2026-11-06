"""Tests for tinyllama.proof.make_dense_kit.

The dense (NE 0) transform must match verifier/tests/served_form_test.py's no_experts case: NE 0,
no router digit form, no expert weight-bit-width lines, no clamps, no table digest lines, and the
router and clamp lookup segments removed from the layer call lines. The result derives from the
pinned MoE kit (prover/tests/configurations/pinned) and keeps its shape (L 8).
"""

from pathlib import Path

from tinyllama.proof import make_dense_kit

REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED = REPO_ROOT / "prover" / "tests" / "configurations" / "pinned"
MOE_SEGMENTS = {"XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG"}


def _values(lines):
    return {w[0]: int(w[1]) for w in (x.split(" ") for x in lines) if len(w) == 2 and w[1].lstrip("-").isdigit()}


def _dense_lines():
    return make_dense_kit.no_experts(make_dense_kit.read_lines(PINNED / "policy.txt"))


def test_ne_zero_and_router_off():
    values = _values(_dense_lines())
    assert values["NE"] == 0
    assert values["hr"] == 0


def test_expert_weight_widths_are_zero():
    values = _values(_dense_lines())
    for name in make_dense_kit.EXPERT_WB:
        assert values[name] == 0, name


def test_dense_weight_widths_are_kept():
    # The dense feed-forward widths (gate, up, down) stay as the base kit set them.
    base = _values(make_dense_kit.read_lines(PINNED / "policy.txt"))
    values = _values(_dense_lines())
    for name in ("wb_g", "wb_u", "wb_dn"):
        assert values[name] == base[name], name


def test_no_clamps_or_digest_lines():
    lines = _dense_lines()
    names = {ln.split(" ")[0] for ln in lines}
    assert "rcl" not in names
    assert "gcl" not in names
    assert not any(make_dense_kit.DIGEST.match(ln) for ln in lines)


def test_moe_segments_removed_and_no_empty_call():
    for ln in _dense_lines():
        if not ln.startswith("call "):
            continue
        segments = ln.split(" ")[3:]
        assert segments, "a call line was left with no segments: " + ln
        assert MOE_SEGMENTS.isdisjoint(segments), ln


def test_main_writes_kit(tmp_path, monkeypatch):
    out = tmp_path / "dense"
    monkeypatch.setattr(
        make_dense_kit.sys,
        "argv",
        ["make_dense_kit", "--base", str(PINNED), "--out", str(out)],
    )
    assert make_dense_kit.main() == 0
    values = _values(make_dense_kit.read_lines(out / "policy.txt"))
    assert values["NE"] == 0
    # KL.bin keeps one 13-word (52-byte) record per layer and no expert words.
    layers = values["L"]
    assert (out / "KL.bin").stat().st_size == 4 * make_dense_kit.KL_WORDS_PER_LAYER * layers
