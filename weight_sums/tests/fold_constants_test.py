#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Started by weight_sums/tests/run_tests.sh, which the runner starts; by hand: from any folder.
"""Check weight_sums.fold_constants: the two gain-folding constants c_0 and c_q of the conversion
rule, as binary32 bit patterns, for a model of a given width and head dimension.

The rule (scales/README.md, "The constants of a norm"): with t the largest integer with 4^t <=
width, c = sqrt(width) / 2^t, so sqrt(width) = c * 2^t and 1 <= c < 2; c_0 = RN32(c) and
c_q = RN32(c / sqrt(head_dimension)). The former literals 0x3F8F1BBD and 0x3DCA62C2 were right for
width 5,120 and head 128 only; this test pins those (unchanged) and TinyLlama's width 2,048 and
head 64, whose values docs/tinyllama/ARITHMETIC_CONTRACT.md section 4.3 fixes as 0x3FB504F3 and
0x3E3504F3 (sqrt(2) and sqrt(2) / 8). It also checks c_0 is in [1, 2) for a few other widths.

usage: fold_constants_test.py

It needs numpy and gguf-py 0.19.0 (weight_sums.py imports gguf at module load). Exit status 0 if
every case is as expected, 1 if one is not, 2 if weight_sums cannot be imported (its packages are
missing); the last line gives the counts.
"""

import math
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

try:
    import numpy as np

    import weight_sums
except Exception as error:  # numpy or gguf missing: the row counts as not run
    print("fold_constants_test.py: cannot import weight_sums (%s: %s)" % (type(error).__name__, error))
    sys.exit(2)


def bits(value):
    return int(np.float32(value).view(np.uint32))


# (width, head dimension, expected c_0 bits, expected c_q bits).
PINNED = [
    (5120, 128, 0x3F8F1BBD, 0x3DCA62C2),  # Maverick's width and head: unchanged from the old literals
    (2048, 64, 0x3FB504F3, 0x3E3504F3),  # TinyLlama: ARITHMETIC_CONTRACT.md 4.3, sqrt(2) and sqrt(2) / 8
]


def main():
    bad = 0
    for width, head, want_c0, want_cq in PINNED:
        c0, cq = weight_sums.fold_constants(width, head)
        got_c0, got_cq = bits(c0), bits(cq)
        ok = got_c0 == want_c0 and got_cq == want_cq
        bad += not ok
        print(
            "width %d head %d: c_0 0x%08X (want 0x%08X), c_q 0x%08X (want 0x%08X): %s"
            % (width, head, got_c0, want_c0, got_cq, want_cq, "as expected" if ok else "NOT as expected")
        )
    # The general rule keeps c = sqrt(width) / 2^t in [1, 2), so c_0 (as a float) is too.
    for width in (64, 256, 1024, 2048, 4096, 5120, 8192, 16384):
        c0, _ = weight_sums.fold_constants(width, 64)
        ok = 1.0 <= float(c0) < 2.0
        bad += not ok
        if not ok:
            print("width %d: c_0 %r is not in [1, 2): NOT as expected" % (width, float(c0)))
    # The two shapes above, computed here with Python's math, independently of weight_sums.
    for width, head, want_c0, want_cq in PINNED:
        t = 0
        while 4 ** (t + 1) <= width:
            t += 1
        c = math.sqrt(width) / (2**t)
        if bits(c) != want_c0 or bits(c / math.sqrt(head)) != want_cq:
            bad += 1
            print("width %d head %d: the independent check disagrees: NOT as expected" % (width, head))
    print("fold_constants_test.py: %s" % ("all as expected" if not bad else "%d not as expected" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
