"""Benchmark one full-size TL-INT-1 layer, on the pod only.

Run this on the RunPod cpu3g-8-32 pod (D7), not on the shared box: it builds
random full-dimension weights (d = 2048, f = 5632, H = 32, G = 4, dh = 64) in
RAM and then times one call to :func:`tinyllama.reference.layer.layer`.

    python -m tinyllama.reference.bench_layer [--n 24] [--seed 0]

The tables come from :mod:`tinyllama.reference.tables`; that is allowed here
because this script is not part of the inference path.  Peak resident set size
is read from the ``resource`` module (KiB on Linux).
"""

import argparse
import resource
import time

import numpy as np

from ..contract import D, DH, F, G, H, T
from .layer import layer
from .tables import exp_table, rope_tables, silu_table

__all__ = ["build_weights", "build_scales", "main"]


def build_weights(seed):
    rng = np.random.default_rng(seed)
    d = D
    dh = DH
    h = H
    g = G
    f = F
    shapes = {
        "wq": (h * dh, d),
        "wk": (g * dh, d),
        "wv": (g * dh, d),
        "wo": (d, h * dh),
        "wg": (f, d),
        "wu": (f, d),
        "wd": (d, f),
    }
    return {
        name: rng.integers(-4095, 4096, size=shape, dtype=np.int16)
        for name, shape in shapes.items()
    }


def build_scales():
    return {
        "kq": 16,
        "kk": 16,
        "kv": 16,
        "ko": 16,
        "kg": 16,
        "ku": 16,
        "kd": 16,
        "M1": 37,
        "K1": 32,
        "M2": 37,
        "K2": 32,
        "sh": 8,
        "eps1": 85899,
        "eps2": 85899,
        "rp": 1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="positions (default 24)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed + 1)
    x = rng.integers(-100, 101, size=(args.n, D), dtype=np.int64)
    weights = {
        name: array.astype(np.int64)
        for name, array in build_weights(args.seed).items()
    }
    cosines, sines = rope_tables(T, DH, 10000)
    tables = {
        "C": cosines,
        "S": sines,
        "EX": np.asarray(exp_table(), dtype=np.int64),
        "Ts": silu_table(),
    }
    scales = build_scales()

    start = time.perf_counter()
    output = layer(x, weights, scales, tables, layer_index=0)
    elapsed = time.perf_counter() - start

    rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("n: %d" % args.n)
    print("seed: %d" % args.seed)
    print("layer_seconds: %.6f" % elapsed)
    print("output_shape: %r" % (tuple(output.shape),))
    print("peak_rss_kib: %d" % rss_kib)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
