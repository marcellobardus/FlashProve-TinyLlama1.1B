#!/usr/bin/env python3
"""Build a dense (NE 0) FlashProve kit from a base MoE kit, for PROVING_PLAN M2.

The transform matches verifier/tests/served_form_test.py's ``no_experts`` case exactly: it turns a
model with routed experts into an all-dense model (NE 0), which docs/model.md and PROVING_PLAN M2
define as "every layer dense". served_form_test only runs the verifier's planning calls on such a
kit; this script also writes layout.txt so cp_commit and cp_prove can run it on a GPU (see
docs/tinyllama/PROOF_PREP.md for the H100 commands).

The dense standard form is deliberately *not* a verification: cp_verify reports it is "not the
standard parameter form (routed-experts, both-clamps, table-digests)" and ends at status 14. That is
the expected and only acceptable verifier outcome until the dense standard form (O1 / T4.3) lands.

The M2 stand-in derives from prover/tests/configurations/pinned (L 8, ll 2, lt 6 -> T 64, d 512,
NE 4). The result is the same shape with NE 0.

Usage, from the repository root:
  python -m tinyllama.proof.make_dense_kit \
    --base prover/tests/configurations/pinned \
    --out  <kit dir> \
    --cp-verify <path to cp_verify>

cp_verify and prover/tools/pack_layout.py build the layout file; without --cp-verify the script
writes only policy.txt and KL.bin and skips the planning and layout steps.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# forest_lines.layer() computes the iota_q lines (log2 of each lookup call's trees) the way the
# verifier does; it is the same module served_form_test.py uses.
sys.path.insert(0, str(REPO_ROOT / "verifier" / "tests"))
import forest_lines  # noqa: E402

PACK_LAYOUT = REPO_ROOT / "prover" / "tools" / "pack_layout.py"
# A policy.txt digest line: 8 words for each of the four public tables.
DIGEST = re.compile(r"(tsilu|tsig|trope|texp)\d+ ")
# The lookup segments of the router and the two clamps, removed with the experts.
MOE_SEGMENTS = ("XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG")
# The weight-bit-width lines of the router and the shared and routed experts.
EXPERT_WB = ("wb_gs", "wb_us", "wb_gr", "wb_ur", "wb_dnr", "wb_r")
KL_WORDS_PER_LAYER = 13  # KLW: a dense layer's quantization-scale record


def read_lines(path):
    with open(path, encoding="ascii") as f:
        return f.read().split("\n")


def write_lines(path, lines):
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def put(name, value, lines):
    """policy.txt's lines with the line <name> set to <value>, or removed if value is None."""
    out, found = [], False
    for ln in lines:
        if ln.split(" ", 1)[0] == name:
            found = True
            if value is not None:
                out.append("%s %s" % (name, value))
        else:
            out.append(ln)
    if value is not None and not found:
        at = next(i for i, x in enumerate(out) if x.startswith("call"))
        out = out[:at] + ["%s %s" % (name, value)] + out[at:]
    return out


def segments_out(lines, *segments):
    """The lines without these lookup segments in the layer sub-proof's lines "call", and without a
    line "call" left empty (matches served_form_test.segments_out)."""
    new = [" ".join(w for i, w in enumerate(x.split(" ")) if i < 3 or w not in segments) if x.startswith("call ") else x for x in lines]
    return [x for x in new if not (x.startswith("call ") and len(x.split(" ")) < 4)]


def forest_fitted(lines):
    """The lines with iota_0 .. iota_7 set to forest_lines.layer()'s values (matches
    served_form_test.forest_fitted)."""
    value = {w[0]: int(w[1]) for w in (x.split(" ") for x in lines) if len(w) == 2 and w[1].isdigit()}
    fit = forest_lines.layer(value, forest_lines.calls_of("\n".join(lines)))
    return ["%s %d" % (x.split(" ")[0], fit[x.split(" ")[0]]) if x.split(" ")[0] in fit else x for x in lines]


def no_experts(lines):
    """served_form_test.no_experts on policy.txt's lines: NE 0, every layer dense."""
    lines = put("hr", 0, put("NE", 0, lines))
    lines = [x for x in lines if x.split(" ")[0] not in ("rcl", "gcl")]
    lines = [x for x in lines if not DIGEST.match(x)]
    for name in EXPERT_WB:
        lines = put(name, 0, lines)
    return forest_fitted(segments_out(lines, *MOE_SEGMENTS))


def number_of(lines, name):
    for ln in lines:
        w = ln.split(" ")
        if len(w) == 2 and w[0] == name:
            return int(w[1])
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="a MoE kit directory with policy.txt and KL.bin")
    ap.add_argument("--out", required=True, help="the dense kit directory to write")
    ap.add_argument("--cp-verify", help="a cp_verify binary; without it layout.txt is not written")
    ap.add_argument("--pack-layout", default=str(PACK_LAYOUT))
    args = ap.parse_args()

    base = Path(args.base)
    lines = no_experts(read_lines(base / "policy.txt"))
    with open(base / "KL.bin", "rb") as f:
        scales = f.read()
    # KL.bin keeps one 13-word record per layer and no expert words (served_form_test writes
    # scales[: 4 * 13 * layers]).
    layers = number_of(lines, "L")
    scales = scales[: 4 * KL_WORDS_PER_LAYER * layers]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_lines(out / "policy.txt", lines)
    with open(out / "KL.bin", "wb") as f:
        f.write(scales)
    print("make_dense_kit: wrote %s/policy.txt and KL.bin (NE 0, L %d)" % (out, layers))

    if not args.cp_verify:
        print("make_dense_kit: no --cp-verify; skipped the planning call and layout.txt")
        return 0

    r = subprocess.run([args.cp_verify, "-plan", "arrays", str(out), "0"], capture_output=True, text=True)
    (out / "arrays.txt").write_text(r.stdout)
    if r.returncode != 4:
        print("make_dense_kit: cp_verify -plan arrays ended with status %d (expected 4)\n%s" % (r.returncode, r.stdout[-800:]), file=sys.stderr)
        return 1
    row_log = next(ln.split()[4] for ln in lines if ln.startswith("group link "))
    made = subprocess.run([sys.executable, "-B", args.pack_layout, "--arrays", str(out / "arrays.txt"), row_log, "4"], capture_output=True, text=True)
    if made.returncode:
        print("make_dense_kit: pack_layout.py failed: %s" % made.stderr.strip(), file=sys.stderr)
        return 1
    (out / "layout.txt").write_text(made.stdout)
    print("make_dense_kit: wrote %s/arrays.txt and layout.txt (cp_verify -plan arrays status 4)" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
