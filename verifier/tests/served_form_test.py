#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Started by the runner; by hand: below a directory of your own. (Runner: run_suite.py. The work folder goes there.)
"""Checks, condition by condition, that cp_verify reports inputs not in the standard parameter form.

The standard parameter form (`served form` in names) is a list of conditions that a call must meet
to end with YES. They concern the call and the three configuration files: policy.txt (the parameter
file), KL.bin (the layers' quantization scales) and layout.txt (where a proof places each array).
The verifier cp_verify checks in three stages: key (the file key.bin), leaf (a file
leaf/<k>.bin) and request (the proof). A whole call, or a part-call (-key, -leaf: that stage only),
that misses a condition prints the condition's name and ends at best with NOT-A-VERIFICATION (exit
status 14). servedForm() (verifier/src/vf_common.h) lists the names of the conditions that the
request stage evaluates. The stages key and leaf evaluate one, two-codes, which the request stage
evaluates too. The stages key and leaf and the planning calls "request" and "accounting" print on
which bounds the proximity rounds (`closeness` in output) of a key file's and a leaf file's plan
rest ("accounting: one-time files: the plan of a key: .."). In every planning
case and every set of files the test checks that these lines are there. A whole call also evaluates id-named. In every mode the test first
compares SHORT below with the list in vf_common.h, so a condition that is added there fails this test.

Planning calls ("cp_verify -plan request") read no proof. They print the unmet conditions that the
configuration files decide: the request stage's. The cases change a configuration (`kit` in names and
messages; prover/tests/configurations/pinned meets every condition) so that one condition is not
met. "--list" prints every case with what it changes and what is expected, one of three results:
  - "not met: <names>": exit status 4 and the line "plan: not the standard parameter form (..): .."
    with those names and no other;
  - "the refusal <name>": exit status 1 and one line "NO: <name>": the verifier rejects the files;
  - "met": exit status 4 and no line "plan: not ..": a control, which changes the same values and
    violates nothing (the case none.kit changes nothing).
Conditions without a case in which that condition alone is not met:
  - routed-experts: a dense model (NE 0) with neither clamp line meets routed-experts and
    both-clamps (the standard dense parameter form), so neither can be the only unmet condition on a
    dense kit. The case "dense" (no digest lines) misses table-digests alone; "dense-with-three-digests"
    (the three tables a dense model uses) meets every condition; "dense-with-digests" (all four
    digest lines) is rejected (policy-tables), because a dense model does not use the FS/SIG table;
    a clamp line on a dense model is rejected too (policy-gcl, policy-rcl);
  - prompt-source, three-bytes: the verifier rejects the files of their cases;
  - placement-table: without a layout file the planning call is rejected (plan-layout).
Two parts of rotary-table have no case: lt (log2 of a request's tokens) at most 10, dh (a head's
dimension) at most 128. one-placement is tested on the residual stream only: in
prover/tests/configurations/pinned/layout.txt the first block of attention's input spans two rows,
and no two of its blocks have the same first row and length, so exchanged() and halved() do not
apply to it.

A planning call does not show what the stages key and leaf themselves print. A part-call prints the
causes of NOT-A-VERIFICATION only after its file has passed, so this takes honest files made under
changed parameters: the sets (`world` in names and messages) that the_worlds() lists.
  --bend writes the three configuration files of such a set, for a committer and a prover to work
    under; "cp_verify -plan accounting" on them must end with status 4.
  --world checks the files that they made; <ivk folder> is the commitment's public key directory.
    The part-calls -key and -leaf (on leaf/0.bin), and the whole call if the last three arguments
    are given, must each end NOT-A-VERIFICATION with exactly the set's conditions as the stage's
    causes, or pass if the set has none. The two part-calls must also print their line
    "accounting: one-time files:".
    The whole call's line "commitment:" must end with the clause on a record (RECORDCLAUSE in
    verifier/src/vf_common.h), and its line "result-v1" have the field record=1.
    The set `kit` changes nothing and must pass: the positive control.
The other sets miss two-codes. SUITE.md runs --world on the set `kit` only.
The set `record` is a second run of the control, on the files of the row that names it.
The conditions on a whole call's request and options (next-token, exact-match, policy-named,
id-named) have no case here: none of this test's planning calls can miss them.

usage: served_form_test.py <cp_verify> <kit folder> <new work folder> [--condition <name>]
       served_form_test.py --list <cp_verify> <kit folder>
       served_form_test.py --bend <name> <cp_verify> <kit folder> <new folder>
       served_form_test.py --world <name> <cp_verify> <config folder> <ivk folder> [<request file> <y file> <proof file>]
Exit status 0: all as expected. 1: something was not. 2: a usage error, or the path given for a new
folder exists. 3: a call passed where NOT-A-VERIFICATION was expected. The script creates each new
folder itself, with any missing parents, and removes only the temporary directory of "--list".
"""

import os
import re
import struct
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True  # the imports below must leave no folder __pycache__ in the tree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "scales"), HERE]
import forest_lines  # noqa: E402
import policy_tables  # noqa: E402
from policy_tables import BAR  # noqa: E402

PACK = os.path.join(HERE, "..", "..", "prover", "tools", "pack_layout.py")
# The conditions that the request stage can list, each by the first word of its name, in the order
# in which it lists them.
SHORT = ("routed-experts", "five-limbs", "both-clamps", "prompt-source", "placement-table", "table-digests",
         "no-range-words", "three-bytes", "ring-degree", "one-shift", "one-placement", "softmax-table", "rotary-table",
         "norm-shifts", "two-codes", "next-token", "exact-match", "policy-named")  # fmt: skip
ONE_TIME_LINE = r"^accounting: one-time files: the plan of (a key|an entry): (\d+ closeness rounds rest on the theorem on codes \(ePrint 2025/2055, Thm 4\.6\), taken as given, \d+ on the bound within a third of the distance); the table has a record$"
RECORD = "a record"
REC_REC = (RECORD, RECORD)  # (a key file's plan, a leaf file's)
# Matches one of policy.txt's 32 digest lines: 8 for each public table.
DIGEST = re.compile(r"(tsilu|tsig|trope|texp)\d+ ")
LINE = r"^plan: not the standard parameter form \((.*)\): no run with these public inputs can be a verification \(at best status 14\)$"
FORM = "not the standard parameter form: "


def order_in_the_source():
    """The conditions' names as servedForm() lists them, read from the verifier's source."""
    with open(os.path.join(HERE, "..", "src", "vf_common.h"), encoding="utf-8") as f:
        text = f.read()
    constant = {name: "".join(re.findall(r'"([^"]*)"', pieces))  # a constant may be written as several pieces, one after the other
                for name, pieces in re.findall(r'static constexpr const char \*(SERVED_\w+) =\s*((?:"[^"]*"\s*)+);', text)}  # fmt: skip
    body = text[text.index("static string servedForm(") :]
    body = body[: body.index("};")]
    return tuple(a or constant[b] for a, b in re.findall(r'\{s\.\w+,\s*(?:"([^"]*)"|(SERVED_\w+))\}', body))


ORDER = order_in_the_source()  # the full names: some add, after a space, a note in brackets


def rests(text, plans=("a key", "an entry")):
    """What an output prints for these plans: RECORD for each. None if it has no such line, or not
    exactly one for each plan and none for another, or one in another form."""
    found = [p for p, words in re.findall(ONE_TIME_LINE, text, re.M)]
    if len(found) != len(re.findall(r"(?m)^accounting: one-time files:", text)):
        return None  # a line with this beginning in another form
    return (RECORD,) * len(plans) if sorted(found) == sorted(plans) else None


def without_the_lines(lines):
    return [x for x in lines if not DIGEST.match(x)]


def read(path):
    with open(path, encoding="ascii") as f:
        return f.read().split("\n")


def write(path, lines):
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def call(words):
    return subprocess.run(words, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", check=False)


def refusals(text):
    return [x[4:].strip() for x in text.splitlines() if x.startswith("NO: ")]


def shifted(scales, layers, sh):
    """KL.bin's bytes with the softmax shift of these layers set to sh: word 11 of a layer's 13
    (`klSh` in verifier/src/cp_verify.cpp)."""
    words = list(struct.unpack("<%dI" % (len(scales) // 4), scales))
    for layer in layers:
        words[13 * layer + 11] = sh
    return struct.pack("<%dI" % len(words), *words)


def with_word(scales, layer, index, value):
    """KL.bin's bytes with one of a layer's 13 words set: 7 and 8 are M and K (scale and shift) of
    the layer's first norm, 9 and 10 those of its second."""
    words = list(struct.unpack("<%dI" % (len(scales) // 4), scales))
    words[13 * layer + index] = value
    return struct.pack("<%dI" % len(words), *words)


def layer_links(table):
    """The layer groups' segments of a request's proof (`link` in names and messages): those with a
    line "<link> 1 2 B ..", which heads the border group of attention's input."""
    return sorted(int(x.split()[0]) for x in table if x.split()[1:4] == ["1", "2", "B"])


def exchanged(table, links, group):
    """layout.txt's lines with the first places of two blocks exchanged, in the group
    "<link> 1 <group>" of each of these links: the first two blocks that have the same first row
    (r0) and length (2^bj). In a block's line (below): Ru, au: the array's first row in the
    block, log2 of its rows there; Kw, ew: the same for columns; after r0: the first place in that
    row; br, bj: log2 of the rows and places filled; sj is 0.
    A block's line: link, tree, group, array, Ru, au, Kw, ew,
    r or c, r0, j0, br, bj, sj."""
    new = [x.split() for x in table]
    for link in links:
        at = [i for i, f in enumerate(new) if f[:3] == [str(link), "1", str(group)] and len(f) == 14]
        pair = [(a, b) for k, a in enumerate(at) for b in at[k + 1 :] if (new[a][9], new[a][12]) == (new[b][9], new[b][12])]
        if not pair:
            sys.exit("the kit's layout file has no two blocks in the same row and of the same length in the group %d 1 %d" % (link, group))
        a, b = pair[0]
        new[a][10], new[b][10] = new[b][10], new[a][10]
    return [" ".join(f) for f in new]


def halved(table, links, group):
    """layout.txt's lines with the first block written as two blocks, its halves, in the group
    "<link> 1 <group>" of each of these links. The block is cut along the index that forms the high
    bits of a cell's position in the block (the column for "r", the row for "c"; the other index if
    that one has a single value), so no cell moves."""
    new, done = [], set()
    for x in table:
        f = x.split()
        if len(f) != 14 or f[1:3] != ["1", str(group)] or int(f[0]) not in links or f[0] in done:
            new.append(x)
            continue
        done.add(f[0])
        if f[11] != "0" or f[12] == "0":
            sys.exit("the first block of the group %s 1 %d does not lie in one row, or has one cell only" % (f[0], group))
        start, log = (6, 7) if (f[7] != "0" if f[8] == "r" else f[5] == "0") else (4, 5)  # cut along Kw, ew or along Ru, au
        for half in (0, 1):
            h = list(f)
            h[log], h[12] = str(int(f[log]) - 1), str(int(f[12]) - 1)
            h[start] = str(int(f[start]) + (half << int(h[log])))
            h[10] = str(int(f[10]) + (half << int(h[12])))
            new.append(" ".join(h))
    if len(done) != len(links):
        sys.exit("the kit's layout file has no block in the group %d of some layer link" % group)
    return new


def plan(verifier, policy, scales, asked, layout, request, y, more=()):
    words = [verifier, "-plan", "request", policy, scales, str(asked)] + (["-layout", layout] if layout else [])
    return call(words + ["-prompt", request, "-reply", y, "-q"] + list(more))


def unmet(text):
    """A stage's causes, as (the short names of the conditions that it lists, its other causes).
    The list of conditions is the last cause. It must be a non-empty selection, in order, from
    ORDER's names, separated by ", "; otherwise the first value is None."""
    at = text.find(FORM)
    rest = [p for p in (text if at < 0 else text[:at]).split("; ") if p]
    if at < 0:
        return (), rest
    left, found = text[at + len(FORM) :], []
    if not left.endswith("; "):
        return None, rest
    left = left[:-2]
    for x, s in zip(ORDER, SHORT):
        if left == x or left.startswith(x + ", "):
            found.append(s)
            left = left[len(x) + 2 :]
    return (None if left or not found else tuple(found)), rest


def planned(text):
    """The conditions (short names) that a planning call's output lists as not met, or None if
    its lines cannot be read."""
    form = re.findall(LINE, text, re.M)
    names = unmet(FORM + form[0] + "; ")[0] if form else ()
    if len(form) > 1 or names is None:
        return None
    if len(re.findall(r"^plan: not the standard parameter form", text, re.M)) != len(form):
        return None  # a line that begins like `LINE` but does not match it
    return names


def the_cases(verifier, kit, work):
    """(cases, request file, token file `y`). A case: (condition, case's name, what is changed, a
    function that returns policy.txt's lines, KL.bin's bytes and layout.txt's lines or None, the
    requested bits (`BAR` is 128, the target), the expected result: "met", a tuple of the unmet
    conditions, or a rejection name). `work` is a directory that the caller has created."""
    good = read(os.path.join(kit, "policy.txt"))
    number = {x.split()[0]: x.split()[1] for x in good if len(x.split()) == 2}
    tokens = 1 << int(number["lt"])
    request, y = os.path.join(work, "request.bin"), os.path.join(work, "y.bin")
    with open(request, "wb") as f:
        f.write(struct.pack("<%dI" % (tokens + 2), *([7] * tokens + [tokens - 1, tokens])))
    with open(y, "wb") as f:
        f.write(struct.pack("<I", 5))

    def put(name, value, lines=None):
        """policy.txt's lines with the line <name> set to <value>, or removed if value is None."""
        lines = good if lines is None else lines
        at = [i for i, x in enumerate(lines) if x.startswith(name + " ")]
        if len(at) != 1:
            sys.exit("the parameter file has %d lines %s" % (len(at), name))
        return lines[: at[0]] + ([] if value is None else ["%s %s" % (name, value)]) + lines[at[0] + 1 :]

    def behind(name, line):
        at = [i for i, x in enumerate(good) if x.startswith(name + " ")]
        return good[: at[0] + 1] + [line] + good[at[0] + 1 :]

    # The line eg comes directly after lidb, or after nminH where the parameter file has that line.
    before_eg = "nminH" if any(x.startswith("nminH ") for x in good) else "lidb"

    def tables_anew(lines, _):
        """The lines with the tables' digest lines computed again by scales/policy_tables.py, which
        shares no code with the verifier, and put directly before the first line "call"."""
        rest = without_the_lines(lines)
        at = [i for i, x in enumerate(rest) if x.startswith("call ")][0]
        return rest[:at] + policy_tables.table_lines(policy_tables.digests([x for x in rest if x])) + rest[at:]

    def less(lines, *names):
        return [x for x in lines if x.split(" ")[0] not in names]

    def segments_out(lines, *segments):
        """The lines without these lookup segments in the layer sub-proof's lines "call", and
        without a line "call" that is left empty."""
        new = [" ".join(w for i, w in enumerate(x.split(" ")) if i < 3 or w not in segments) if x.startswith("call ") else x for x in lines]
        return [x for x in new if not (x.startswith("call ") and len(x.split(" ")) < 4)]

    def forest_fitted(lines):
        """The lines with iota_0 .. iota_7 (log2 of each lookup call's number of trees) set to the
        values that forest_lines.layer() computes. The verifier checks them directly after it has
        scanned the file (shape-call-<call>): before the check that digest lines come only with
        routed experts (one of those named policy-tables), and before it evaluates any condition."""
        value = {w[0]: int(w[1]) for w in (x.split(" ") for x in lines) if len(w) == 2 and w[1].isdigit()}
        fit = forest_lines.layer(value, forest_lines.calls_of("\n".join(lines)))
        return ["%s %d" % (x.split(" ")[0], fit[x.split(" ")[0]]) if x.split(" ")[0] in fit else x for x in lines]

    with open(os.path.join(kit, "KL.bin"), "rb") as f:
        scales = f.read()
    placed = read(os.path.join(kit, "layout.txt"))
    layers = int(number["L"])
    other_theta = 10000 if number["theta0_lo"] == "500000" else 500000  # the condition rotary-table allows only these two values: its control case takes the one that the parameter file does not have
    group = 1 << int(number["ll"])  # the number of layers in a group (a block of layers)
    final = layers - group  # the last group's first layer
    other = int(struct.unpack_from("<I", scales, 4 * 11)[0]) - 1  # layer 0's softmax shift, less 1
    links = layer_links(placed)
    # layer_links() names border group 2 by number, and the cases on one-placement name group 3.
    # A border group's line: link, tree, group, "B", rows, later segments they are kept for.
    groups = [f for f in (x.split() for x in placed) if len(f) == 6 and f[1] == "1" and f[3] == "B"]
    borders = {int(f[0]): [(g[2], g[5]) for g in groups if g[0] == f[0]] for f in groups}
    if not borders or any(found != [("2", "1"), ("3", "2")] for found in borders.values()):
        sys.exit("served_form_test.py: this test needs a layout file in which each layer link has two border groups: 2, read by 1 later link (attention's input),"
                 " and 3, read by 2 (the residual stream). Found, by link, as (group, later links): %s" % (borders or "none"))
    k_of = struct.unpack_from("<I", scales, 4 * (13 + 8))[0], struct.unpack_from("<I", scales, 4 * (13 * (layers - 1) + 10))[0]  # K of the two norms that cases change
    row_log = [x.split()[4] for x in good if x.startswith("group link ")][0]  # log2 of the message length of a link's rows

    def anew(tag, lines, words_kl, free="4"):
        """The three files' contents after a change that moves the arrays of a request's proof:
        layout.txt is generated for these lines and scale words as the unchanged one was. free: the
        places that stay empty at the end of every row; an out-of-domain sample with two answers
        (line answers 2) needs more than four."""
        folder = os.path.join(work, tag + ".plan")
        os.makedirs(folder, exist_ok=True)
        write(os.path.join(folder, "policy.txt"), lines)
        with open(os.path.join(folder, "KL.bin"), "wb") as f:
            f.write(words_kl)
        r = call([verifier, "-plan", "arrays", folder, "0"])
        if r.returncode != 4:
            sys.exit("the planning call on the arrays for %s: status %d, %s" % (tag, r.returncode, r.stdout.strip()[-200:]))
        write(os.path.join(folder, "arrays.txt"), r.stdout.split("\n"))
        made = subprocess.run([sys.executable, "-B", PACK, "--arrays", os.path.join(folder, "arrays.txt"), row_log, free], capture_output=True, text=True, check=False)
        if made.returncode:
            sys.exit("pack_layout.py: " + made.stderr.strip())
        return lines, words_kl, made.stdout.split("\n")

    def no_experts():
        """NE 0: every layer is dense. Sets to 0 the router's digit form (hr) and the bit widths of
        the router's and the shared and routed experts' weights (wb_..). Removes the two clamps'
        lines, the lookup segments of the router and of the clamps, the tables' digest lines and the
        MoE layers' scale words."""
        lines = less(without_the_lines(put("hr", 0, put("NE", 0))), "rcl", "gcl")
        for name in ("wb_gs", "wb_us", "wb_gr", "wb_ur", "wb_dnr", "wb_r"):
            lines = put(name, 0, lines)
        lines = forest_fitted(segments_out(lines, "XR", "LR", "LT", "DD", "LH", "SEL", "L3", "CLR", "MR", "G0", "CG"))
        return anew("routed-experts", lines, scales[: 4 * 13 * layers])

    def and_lines(make, new, before):
        """A case's three files with these lines put into policy.txt directly before the first
        line that begins with <before>."""
        lines, words_kl, table = make()
        at = [i for i, x in enumerate(lines) if x.startswith(before)][0]
        return lines[:at] + new + lines[at:], words_kl, table

    digests = [x for x in good if DIGEST.match(x)]
    # The three digest lines a dense model uses (SILU, RoPE, softmax; not the FS/SIG table), in the
    # ascending table order the scan requires. digests is in the file's order tsilu, tsig, trope,
    # texp, so dropping tsig leaves tsilu, trope, texp.
    three_digests = [x for x in digests if not x.startswith("tsig")]
    behind_digests = good[max(i for i, x in enumerate(good) if DIGEST.match(x)) + 1]

    def with_digests(new):
        """policy.txt's lines with these lines in place of the digest lines."""
        first = good.index(digests[0])
        return good[:first] + new + good[first + len(digests) :]

    def altered(i):
        name, value = digests[i].split(" ")
        return with_digests(digests[:i] + ["%s %d" % (name, int(value) ^ 1)] + digests[i + 1 :])

    def kit_with(lines=None, words_kl=None, table=placed):
        return lambda: (good if lines is None else lines(), scales if words_kl is None else words_kl, table() if callable(table) else table)

    worlds = the_worlds(good)

    def values_of(name):
        return [x for x in good if x.startswith("group %s " % name)][0].split(" ")[2:]

    cases = [
        ("none", "kit", "the kit unchanged", kit_with(), BAR, "met"),
        ("none", "asked-above", "the kit unchanged, in a call that requests %d bits, one above its line sec" % (BAR + 1), kit_with(), BAR + 1, "security-bits"),
        ("profile", "prof-19", "prof 19 and nothing else", kit_with(lambda: put("prof", 19)), 0, "policy-profile"),
        ("routed-experts", "NE-0", "NE 0 and nothing else", kit_with(lambda: put("NE", 0)), 0, "header-absent-class"),
        ("routed-experts", "dense", "NE 0 with all that follows from it: no router, no experts' weight kinds, no clamps, no digest lines; the layout file made again. A dense model with neither clamp meets routed-experts and both-clamps; with no digest lines it still misses table-digests",
         no_experts, 0, ("table-digests",)),
        ("routed-experts", "dense-with-digests", "the same with the kit's four digest lines: a dense model uses three tables, not the FS/SIG table, so the four-table set is refused", lambda: and_lines(no_experts, digests, behind_digests), 0, "policy-tables"),
        ("routed-experts", "dense-with-three-digests", "the same with the three digest lines a dense model uses (tsilu, trope, texp): the standard dense parameter form",
         lambda: and_lines(no_experts, three_digests, behind_digests), 0, "met"),
        ("routed-experts", "dense-with-gcl", "the same with the line gcl 1", lambda: and_lines(no_experts, ["gcl 1"], "xl "), 0, "policy-gcl"),
        ("routed-experts", "dense-with-rcl", "the same with the line rcl 1", lambda: and_lines(no_experts, ["rcl 1"], "xl "), 0, "policy-rcl"),
        ("five-limbs", "no-xl", "no line xl; the layout file made again", lambda: anew("five-limbs", put("xl", None), scales), 0, ("five-limbs",)),
        ("both-clamps", "no-gcl", "no line gcl, its two segments removed from the call lines, the lines iota_q fitted, the table digest lines computed again; the layout file made again",
         lambda: anew("both-clamps", tables_anew(forest_fitted(segments_out(put("gcl", None), "G0", "CG")), "gcl"), scales), 0, ("both-clamps",)),
        ("both-clamps", "no-rcl", "no line rcl, its two segments removed from the call lines, the lines iota_q fitted, the table digest lines computed again; the layout file made again",
         lambda: anew("both-clamps.rcl", tables_anew(forest_fitted(segments_out(put("rcl", None), "L3", "CLR")), "rcl"), scales), 0, ("both-clamps",)),
        ("prompt-source", "toksrc-0", "toksrc 0", kit_with(lambda: put("toksrc", 0)), 0, "statement-required"),
        ("prompt-source", "toksrc-2", "toksrc 2", kit_with(lambda: put("toksrc", 2)), 0, "statement-required"),
        ("placement-table", "no-layout", "no layout file", kit_with(table=None), 0, "plan-layout"),
        ("table-digests", "no-lines", "none of the tables' digest lines", kit_with(lambda: without_the_lines(good)), 0, ("table-digests",)),
        ("table-digests", "silu-altered", "one bit of the last word of the SiLU table's digest", kit_with(lambda: altered(7)), 0, "table-digest-silu"),
        ("table-digests", "sigmoid-altered", "one bit of a word of the second table's digest", kit_with(lambda: altered(11)), 0, "table-digest-sigmoid"),
        ("table-digests", "rope-altered", "one bit of a word of the rotary tables' digest", kit_with(lambda: altered(21)), 0, "table-digest-rope"),
        ("table-digests", "exp-altered", "one bit of the last word of the softmax table's digest", kit_with(lambda: altered(31)), 0, "table-digest-exp"),
        ("table-digests", "last-missing", "the last of the digest lines missing", kit_with(lambda: with_digests(digests[:-1])), 0, "policy-tables"),
        ("table-digests", "two-swapped", "two of the digest lines in the other order", kit_with(lambda: with_digests(digests[:4] + [digests[5], digests[4]] + digests[6:])), 0, "policy-tables"),
        ("table-digests", "behind-a-call", "the digest lines behind the first line call",
         kit_with(lambda: with_digests([behind_digests] + digests)[: good.index(digests[0]) + 1 + len(digests)] + good[good.index(behind_digests) + 1 :]), 0, "policy-tables"),
        ("no-range-words", "eg-7", "a line eg 7, the table digest lines computed again", kit_with(lambda: tables_anew(behind(before_eg, "eg 7"), "eg")), 0, ("no-range-words",)),
        ("no-range-words", "smax-2047", "a line smax 2047, the table digest lines computed again", kit_with(lambda: tables_anew(behind(before_eg, "smax 2047"), "smax")), 0,
         ("no-range-words",)),
        ("three-bytes", "S-32", "S 32", kit_with(lambda: put("S", 32)), 0, "af-policy-wd"),
        ("three-bytes", "S-44", "S 44", kit_with(lambda: put("S", 44)), 0, "af-policy-wd"),
        ("ring-degree", "D-32", "D 32", kit_with(lambda: put("D", 32)), 0, ("ring-degree",)),
        ("one-placement", "stream", "two blocks of the residual stream exchanged in the last layer link", kit_with(table=lambda: exchanged(placed, links[-1:], 3)), 0, ("one-placement",)),
        ("one-placement", "control", "two blocks of the residual stream exchanged in every layer link", kit_with(table=lambda: exchanged(placed, links, 3)), 0, "met"),
        ("one-placement", "stream-halves", "the first block of the residual stream written as its two halves in the last layer link",
         kit_with(table=lambda: halved(placed, links[-1:], 3)), 0, ("one-placement",)),
        ("one-placement", "halves-control", "the first block of the residual stream written as its two halves in every layer link", kit_with(table=lambda: halved(placed, links, 3)), 0, "met"),
        ("softmax-table", "S-11", "S 11, the table digest lines computed again", kit_with(lambda: tables_anew(put("S", 11), "S-11")), 0, ("softmax-table",)),
        ("softmax-table", "S-13", "S 13, the table digest lines computed again", kit_with(lambda: tables_anew(put("S", 13), "S-13")), 0, ("softmax-table",)),
        ("softmax-table", "rx-7", "rx 7, the table digest lines computed again", kit_with(lambda: tables_anew(put("rx", 7), "rx-7")), 0, ("softmax-table",)),
        ("rotary-table", "theta", "theta_0 3,675,548,936, the table digest lines computed again", kit_with(lambda: tables_anew(put("theta0_lo", 3675548936), "theta")), 0, ("rotary-table",)),
        ("rotary-table", "control", "theta_0 %s, the table digest lines computed again" % format(other_theta, ","), kit_with(lambda: tables_anew(put("theta0_lo", other_theta), "theta-control")), 0, "met"),
        ("norm-shifts", "first-norm", "M = K - 1 in the first norm of layer 1", kit_with(words_kl=with_word(scales, 1, 7, k_of[0] - 1)), BAR, ("norm-shifts",)),
        ("norm-shifts", "second-norm", "M = K - 1 in the second norm of the last layer", kit_with(words_kl=with_word(scales, layers - 1, 9, k_of[1] - 1)), BAR, ("norm-shifts",)),
        ("norm-shifts", "last-norm", "MF = KF - 1, the norm before the head", kit_with(lambda: put("MF", int(number["KF"]) - 1)), BAR, ("norm-shifts",)),
        ("norm-shifts", "control", "M = K in the first norm of layer 1", kit_with(words_kl=with_word(scales, 1, 7, k_of[0])), BAR, "met"),
        ("two-codes", "batch", worlds["one-code"][0], kit_with(lambda: worlds["one-code"][1]), BAR, ("two-codes",)),
        ("two-codes", "one-time", worlds["once-one-code"][0], kit_with(lambda: worlds["once-one-code"][1]), BAR, ("two-codes",)),
        ("two-codes", "both", "both of these", kit_with(lambda: one_code("once-", worlds["one-code"][1])), BAR, ("two-codes",)),
        ("two-codes", "control", "the group folded has omega's code length and its own message length: still two codes",
         kit_with(lambda: put("group folded", " ".join(values_of("folded")[:1] + values_of("omega")[1:2] + values_of("folded")[2:]))), BAR, "met"),
    ]  # fmt: skip
    for tag, what, bent, end in (
        ("first-group", "sh %d in layer 1 only" % other, [1], ("one-shift",)),
        ("last-group", "sh %d in layer %d only" % (other, final + 1), [final + 1], ("one-shift",)),
        ("control", "sh %d in all layers of the last block of layers" % other, range(final, final + group), "met"),
    ):  # fmt: skip
        cases.append(("one-shift", tag, what + "; the layout file made again",
                      lambda tag=tag, bent=bent: anew("one-shift." + tag, good, shifted(scales, bent, other)), 0, end))  # fmt: skip
    return cases, request, y


def lay(folder, lines, words_kl, table):
    os.makedirs(os.path.dirname(os.path.abspath(folder)), exist_ok=True)
    os.mkdir(folder)  # fails if the name is taken
    write(os.path.join(folder, "policy.txt"), lines)
    with open(os.path.join(folder, "KL.bin"), "wb") as f:
        f.write(words_kl)
    if table is not None:
        write(os.path.join(folder, "layout.txt"), table)


def one_code(prefix, lines):
    """policy.txt's lines with the values of the lines "group <prefix>folded" and
    "stages <prefix>folded" replaced by those of <prefix>omega's lines. The two groups then have
    the same code, which the condition two-codes forbids."""
    value = {x.split(" ")[0] + " " + x.split(" ")[1]: x.split(" ")[2:] for x in lines if x.startswith(("group ", "stages "))}
    return [" ".join(x.split(" ")[:2] + value[x.split(" ")[0] + " " + prefix + "omega"]) if x.startswith(("group %sfolded " % prefix, "stages %sfolded " % prefix)) else x for x in lines]


# The sets of changed parameters: name: (what is changed in policy.txt, its lines after the change,
# the conditions that the stages key and leaf must print, the request stage's).
def the_worlds(good):
    return {
        "kit": ("nothing", good, (), ()),
        "one-code": ("the lines group folded and stages folded have the values of omega's: a batch's test on the table has one code",
                     one_code("", good), ("two-codes",), ("two-codes",)),
        "once-one-code": ("the lines group once-folded and stages once-folded have the values of once-omega's: the key's test on the table has one code",
                          one_code("once-", good), ("two-codes",), ("two-codes",)),
        "record": ("nothing", good, (), ()),
    }  # fmt: skip


def stage_causes(text):
    """The causes in a part-call's line "result:", as (conditions by their first words, other
    causes); None unless the output has exactly one such line."""
    found = re.findall(r"^result:.*\(not a verification: (.*?)research code", text, re.M)
    if len(found) != 1:
        return None
    at = found[0].find(FORM)
    rest = [p for p in (found[0] if at < 0 else found[0][:at]).split("; ") if p]
    names = [] if at < 0 else [n.split(" ")[0] for n in re.split(r", (?=[a-z-]+(?: \(|$|,|;))", found[0][at + len(FORM) :].rstrip("; "))]
    return tuple(names), rest


def world(name, verifier, config, ivk, files):
    lines = read(os.path.join(config, "policy.txt"))
    expected = the_worlds(lines)  # only the names and the expectations are read here
    if name not in expected:
        return usage()
    _, _, of_key, of_request = expected[name]
    own = re.findall(r"^commitment ID: ([0-9a-f]{64})$", call([verifier, "-plan", "id", config, os.path.join(ivk, "key.bin")]).stdout, re.M)
    if len(own) != 1:
        sys.exit("cp_verify -plan id printed no ID")
    bad = yes = 0
    calls = [("key", [verifier, "-key", config, os.path.join(ivk, "key.bin")], of_key),
             ("leaf", [verifier, "-leaf", config, os.path.join(ivk, "key.bin"), os.path.join(ivk, "leaf", "0.bin")], of_key)]  # fmt: skip
    for stage, words, names in calls:
        r = call(words + ["-id", own[0], "-bits", str(BAR), "-q", "-show-stages"])  # with -show-stages, a call that ends NOT-A-VERIFICATION also prints the stage's line "result:", which `stage_causes` reads
        last = r.stdout.strip().split("\n")[-1]
        said, owed = rests(r.stdout, ("a key" if stage == "key" else "an entry",)), (RECORD,)
        if said != owed:
            bad += 1
            print("%s, the part-call -%s: its line \"accounting: one-time files:\": %s; expected: %s: not as expected" % (name, stage, said, owed))
        if names:
            ok = r.returncode == 14 and last == "NOT-A-VERIFICATION" and stage_causes(r.stdout) == (names, [])
            yes += r.returncode == 3
        else:
            ok = r.returncode == 3 and last == "PART PASSED: " + stage
        bad += not ok
        print("%s, the part-call -%s: status %d, %s; causes: %s; expected: %s: %s" % (name, stage, r.returncode, last[:40], stage_causes(r.stdout), names or "none", "as expected" if ok else "not as expected"))
    if files:
        r = call([verifier, config, ivk] + files + ["-id", own[0], "-bits", str(BAR), "-q"])
        last = r.stdout.strip().split("\n")[-1]
        got = {s: tuple(n.split(" ")[0] for n in re.split(r", (?=[a-z-]+(?: \(|$|,|;))", c.rstrip("; ")))
               for s, c in re.findall(r"^stage (key|leaf|request): not a verification: not the standard parameter form: (.*?)research code", r.stdout, re.M)}  # fmt: skip
        want = {s: n for s, n in (("key", of_key), ("leaf", of_key), ("request", of_request)) if n}
        # The closing line "commitment:" and the line for programs name a record, and only a record.
        named = (len(re.findall(r"^commitment: .*; the table has a record: the binding of the weights rests on the theorem on codes \(ePrint 2025/2055, Theorem 4\.6\), taken as given$", r.stdout, re.M)), len(re.findall(r"^result-v1 .* commitment-assumed-rounds=\d+ record=1 r=", r.stdout, re.M)))
        named_as_owed = named == (1, 1)
        ok = got == want and named_as_owed and ((r.returncode == 14 and last == "NOT-A-VERIFICATION") if want else (r.returncode, last) == (0, "YES"))
        yes += bool(want) and r.returncode == 0
        bad += not ok
        print("%s, the whole call: status %d, %s; causes: %s; expected: %s; closing lines that name a record: %s: %s" % (name, r.returncode, last[:40], got, want, named, "as expected" if ok else "not as expected"))
    print("served_form_test.py: world %s: %s" % (name, "all as expected" if not bad else "%d not as expected" % bad))
    return 3 if yes else 1 if bad else 0


def result(expected):
    return expected if expected == "met" else "the refusal " + expected if isinstance(expected, str) else "not met: " + ", ".join(expected)


def usage():
    print(__doc__[__doc__.index("usage: ") : __doc__.index("Exit status")].rstrip(), file=sys.stderr)
    return 2


def there_already(folder):
    print("served_form_test.py: %s exists already; name a directory that does not exist yet" % folder, file=sys.stderr)
    return 2


def main():
    a = sys.argv[1:]
    if tuple(x.split(" ")[0] for x in ORDER) != SHORT:
        print("served_form_test.py: the conditions in verifier/src/vf_common.h are not this test's list: %s" % ", ".join(x.split(" ")[0] for x in ORDER))
        return 1
    if len(a) == 3 and a[0] == "--list":
        with tempfile.TemporaryDirectory() as work:
            for condition, name, what, _, asked, expected in the_cases(os.path.abspath(a[1]), a[2], work)[0]:
                print("%s.%s: %s; %d bits requested; expected: %s" % (condition, name, what, asked, result(expected)))
        for name, (what, _, of_key, of_request) in the_worlds(read(os.path.join(a[2], "policy.txt"))).items():
            print("world %s: %s; expected: key and leaf: %s; request: %s" % (name, what, ", ".join(of_key) or "met", ", ".join(of_request) or "met"))
        return 0
    if len(a) == 5 and a[0] == "--bend":
        worlds = the_worlds(read(os.path.join(a[3], "policy.txt")))
        if a[1] in worlds and os.path.lexists(a[4]):
            return there_already(a[4])
        if a[1] not in worlds:
            return usage()
        with open(os.path.join(a[3], "KL.bin"), "rb") as f:
            lay(a[4], worlds[a[1]][1], f.read(), read(os.path.join(a[3], "layout.txt")))
        r = call([os.path.abspath(a[2]), "-plan", "accounting", a[4], str(BAR)])
        print("\n".join(x for x in r.stdout.split("\n") if x.startswith(("footing:", "plan: not", "NO: "))))
        print("served_form_test.py: bend %s: the planning call ended with status %d" % (a[1], r.returncode))
        return 0 if r.returncode == 4 else 1
    if len(a) in (5, 8) and a[0] == "--world":
        return world(a[1], os.path.abspath(a[2]), a[3], a[4], a[5:])
    only = None
    if len(a) == 5 and a[3] == "--condition":
        only, a = a[4], a[:3]
    if len(a) != 3 or a[0].startswith("-") or (only is not None and only not in SHORT + ("none", "profile")):
        return usage()
    if os.path.lexists(a[2]):
        return there_already(a[2])
    verifier, kit, work = os.path.abspath(a[0]), a[1], a[2]  # absolute, here and in the three modes above, because subprocess looks a bare name up on PATH, not in the current directory
    os.makedirs(os.path.dirname(os.path.abspath(work)), exist_ok=True)
    os.mkdir(work, 0o700)  # fails if the name is taken; closed to other users
    cases, request, y = the_cases(verifier, kit, work)
    chosen = [c for c in cases if only is None or c[0] == only]
    if not chosen:
        print("served_form_test.py: %s: no planning case (the head of this file says why)" % only)
        return 1
    with open(os.path.join(kit, "KL.bin"), "rb") as f:
        own = (read(os.path.join(kit, "policy.txt")), f.read(), read(os.path.join(kit, "layout.txt")))
    bad = 0
    for condition, name, what, make, asked, expected in chosen:
        folder = os.path.join(work, condition + "." + name)
        lines, words_kl, table = make()
        if ((lines, words_kl, table) == own) != (condition == "none"):  # only the cases of "none" run on the configuration's own files; another case that changes nothing would show nothing
            bad += 1
            print("%s.%s: %s: its three files are %s the configuration's own: not as expected" % (condition, name, what, "not" if condition == "none" else "the same as"))
            continue
        lay(folder, lines, words_kl, table)
        r = plan(verifier, os.path.join(folder, "policy.txt"), os.path.join(folder, "KL.bin"), asked, os.path.join(folder, "layout.txt") if table is not None else None, request, y)
        if isinstance(expected, str) and expected != "met":
            got, ok = "status %d, %s" % (r.returncode, refusals(r.stdout)), r.returncode == 1 and refusals(r.stdout) == [expected]
        else:
            names = planned(r.stdout)
            said, owed = rests(r.stdout), REC_REC
            got, ok = "status %d, not met: %s, the one-time files: %s" % (r.returncode, names, said), r.returncode == 4 and names == (() if expected == "met" else expected) and said == owed
        bad += not ok
        print("%s.%s: %s: %s; expected: %s: %s" % (condition, name, what, got, result(expected), "as expected" if ok else "not as expected"))
    print("served_form_test.py: %s%s" % (only + ": " if only else "", "all as expected" if not bad else "%d not as expected" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
