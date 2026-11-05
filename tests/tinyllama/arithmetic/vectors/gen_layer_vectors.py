#!/usr/bin/env python3
"""Regenerate layer.json from the independent contract reference.

Layout used by every matrix: W[input][output].  The small tables are:

* C and S: C[t][i], S[t][i] for t < n and i < dh / 2.
* EX: a short list; a lookup at u >= len(EX) returns 0.
* Ts: a list indexed at key + ts_offset (the contract's frozen table uses
  offset 32768; this vector uses 128 to keep the file small).

Run from any directory with: python3 gen_layer_vectors.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARITHMETIC_DIR = HERE.parent
if str(ARITHMETIC_DIR) not in sys.path:
    sys.path.insert(0, str(ARITHMETIC_DIR))

import tl_ref_independent as ref

TS_OFFSET = 128


def vector_inputs():
    return {
        "x_rows": [
            [3, -4, 0, 0],
            [0, 3, -4, 0],
        ],
        "weights": {
            "wq": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "wk": [
                [1, 0],
                [0, 1],
                [1, 1],
                [1, -1],
            ],
            "wv": [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            "wo": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "wg": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
            ],
            "wu": [
                [1, 0, 0],
                [0, 2, 0],
                [0, 0, 1],
                [1, 1, 1],
            ],
            "wd": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
            ],
        },
        "scales": {
            "d": 4,
            "H": 2,
            "G": 1,
            "dh": 2,
            "f": 3,
            "kq": 3,
            "kk": 3,
            "kv": 2,
            "ko": 1,
            "kg": 1,
            "ku": 2,
            "kd": 2,
            "sh": 2,
            "M1": 24,
            "K1": 17,
            "eps1": 1,
            "M2": 24,
            "K2": 17,
            "eps2": 1,
        },
        "tables": {
            "C": [[2], [3]],
            "S": [[1], [-2]],
            "EX": [4096, 2000, 1000, 500, 0],
            "Ts": [(key + 10) for key in range(-TS_OFFSET, TS_OFFSET)],
            "ts_offset": TS_OFFSET,
        },
    }


def build():
    inputs = vector_inputs()
    expected = ref.layer(
        inputs["x_rows"],
        inputs["weights"],
        inputs["scales"],
        inputs["tables"],
    )
    return {
        "description": (
            "One full contract section 9 layer with d=4, H=2, G=1, dh=2, "
            "f=3 and n=2. Matrices are W[input][output]. C/S are indexed "
            "[t][i]; EX is cut at len(EX); Ts is indexed at key + ts_offset."
        ),
        "dims": {"d": 4, "H": 2, "G": 1, "dh": 2, "f": 3, "n": 2},
        "inputs": inputs,
        "expected": expected,
        "hand_checks": {
            "norm": (
                "x[0]=[3,-4,0,0] is the contract section 8 worked example: "
                "Vx=25, sigma=5, Rc=3355443, a[0]=[77,-102,0,0]."
            ),
            "rotary": (
                "With C(0,0)=2, S(0,0)=1 and A=[77,-102], p=(256,-127); "
                "rq(...,3)=(32,-16).  With C(1,0)=3, S(1,0)=-2 and "
                "A=[0,77], p=(154,231); rq(...,3)=(19,29)."
            ),
        },
    }


def main():
    path = HERE / "layer.json"
    path.write_text(json.dumps(build(), indent=2) + "\n")


if __name__ == "__main__":
    main()
