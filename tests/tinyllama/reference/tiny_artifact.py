"""Small synthetic TL-INT-1 artifacts for the reference model tests.

The full artifact is about 2.33 GB.  These helpers use the I2.4 test keyword
``dims=`` and ``write_artifact`` so that a forward pass, a trace and the run
command can be exercised in a few milliseconds of work and a few MB of RAM.
"""

import hashlib
from pathlib import Path

import numpy as np

from tinyllama import contract
from tinyllama.artifacts import FULL, TABLE_PATHS, Dims, write_artifact
from tinyllama.convert import tables


def _recipe_tables(dims):
    cosines, sines = tables.rope_tables(
        dims.T, dims.DH, contract.CS, contract.THETA0
    )
    exp = tables.softmax_table(contract.EXP_S, contract.RX)
    silu = tables.silu_table(contract.EG, contract.SMAX)
    return {
        "rope_c": np.asarray(cosines, dtype=np.int32).reshape(
            dims.T, dims.DH // 2
        ),
        "rope_s": np.asarray(sines, dtype=np.int32).reshape(
            dims.T, dims.DH // 2
        ),
        "softmax_ex": np.asarray(exp, dtype=np.int32),
        "silu_ts": np.asarray(silu, dtype=np.int32),
    }


def _recipe_digests(dims):
    cosines, sines = tables.rope_tables(
        dims.T, dims.DH, contract.CS, contract.THETA0
    )
    exp = tables.softmax_table(contract.EXP_S, contract.RX)
    silu = tables.silu_table(contract.EG, contract.SMAX)
    return {
        "trope": tables.table_digest(tables.trope_words(cosines, sines)),
        "texp": tables.table_digest(tables.texp_words(exp)),
        "tsilu": tables.table_digest(tables.tsilu_words(silu)),
    }

DEFAULT_DIMS = Dims(L=2, D=8, H=2, G=1, DH=4, F=6, V=32, T=8)
RUN_DIMS = Dims(L=1, D=8, H=2, G=1, DH=4, F=6, V=32000, T=32)
DEFAULT_KL_ROW = [20, 20, 8, 8, 8, 8, 8, 37, 32, 37, 32, 8, 1]
EPS_E_RES = 11
EPS = (4096 * (4**EPS_E_RES) + 100000) // 200000

MATRIX_NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


def _table_sha256(tables):
    return {
        path: hashlib.sha256(
            np.ascontiguousarray(tables[name], dtype="<i4").tobytes(order="C")
        ).hexdigest()
        for name, path in TABLE_PATHS.items()
    }


def _params(dims, kl_rows, tables):
    digests = _recipe_digests(dims)
    values = {
        "contract": contract.CONTRACT_ID,
        "L": dims.L,
        "d": dims.D,
        "H": dims.H,
        "G": dims.G,
        "dh": dims.DH,
        "f": dims.F,
        "V": dims.V,
        "T": dims.T,
        "NE": contract.NE,
        "lt": contract.LT,
        "cs": contract.CS,
        "theta0": contract.THETA0,
        "S": contract.S,
        "rx": contract.RX,
        "eg": contract.EG,
        "smax": contract.SMAX,
        "cw": contract.CW,
        "weight_bound": contract.WEIGHT_BOUND,
        "embedding_bound": contract.EMB_BOUND,
        "u0": contract.U0,
        "c0_bits": contract.C0_BITS,
        "cq_bits": contract.CQ_BITS,
        "pad_id": contract.PAD_ID,
        "gcl": contract.GCL,
        "range_conditions": {
            str(key): list(value) for key, value in contract.RANGE_CONDITIONS.items()
        },
        "e_res": EPS_E_RES,
        "eps1": [EPS] * dims.L,
        "eps2": [EPS] * dims.L,
        "epsF": EPS,
        "MF": 37,
        "KF": 32,
        "s": 20,
        "digests": digests,
        "tables_sha256": _table_sha256(tables),
    }
    return values


def _provenance():
    return {
        "source": {
            "repo_id": "synthetic",
            "revision": "synthetic",
            "model_safetensors_sha256": "synthetic",
            "flashprove_base_commit": "synthetic",
        },
        "converter": {
            "git_commit": "synthetic",
            "command": "tests/tinyllama/reference/tiny_artifact.py",
            "python": "synthetic",
            "numpy": "synthetic",
        },
        "calibration": {
            "set": "synthetic",
            "fit_sha256": "synthetic",
            "heldout_sha256": "synthetic",
        },
    }


def build_artifact(
    root,
    *,
    dims=DEFAULT_DIMS,
    seed=20261104,
    kl_rows=None,
    matrix_scale=1000,
    matrix_values=None,
    wh_scale=1000,
    wh_values=None,
    emb_scale=100,
    emb_values=None,
    exponents=None,
):
    """Write one tiny valid artifact below ``root`` and return its path.

    ``exponents`` becomes exponents.json; the default is empty, as the reference
    tests do not read it.  Tests that need per-tensor exponents (for example the
    sums file of PROVING_PLAN T4.2) pass a dict keyed by the HuggingFace tensor
    names, as the converter writes it (tinyllama/convert/run.py).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    shapes = {
        "wq": (dims.H * dims.DH, dims.D),
        "wk": (dims.G * dims.DH, dims.D),
        "wv": (dims.G * dims.DH, dims.D),
        "wo": (dims.D, dims.H * dims.DH),
        "wg": (dims.F, dims.D),
        "wu": (dims.F, dims.D),
        "wd": (dims.D, dims.F),
    }
    matrices = {}
    for layer_index in range(dims.L):
        for name in MATRIX_NAMES:
            shape = shapes[name]
            path = "layers/%02d.%s.i16" % (layer_index, name)
            if matrix_values is not None:
                array = matrix_values(layer_index, name, shape, rng)
            else:
                array = rng.integers(
                    -matrix_scale, matrix_scale + 1, size=shape, dtype=np.int16
                )
            matrices[path] = np.asarray(array, dtype=np.int16)

    if emb_values is not None:
        emb = np.asarray(emb_values, dtype=np.int32)
    else:
        emb = rng.integers(
            -emb_scale, emb_scale + 1, size=(dims.V, dims.D), dtype=np.int32
        )
    if wh_values is not None:
        wh = np.asarray(wh_values, dtype=np.int16)
    else:
        wh = rng.integers(
            -wh_scale, wh_scale + 1, size=(dims.V, dims.D), dtype=np.int16
        )
    matrices["wh.i16"] = wh

    if kl_rows is None:
        kl_rows = [DEFAULT_KL_ROW] * dims.L
    kl = np.asarray(kl_rows, dtype=np.uint32)
    tables = _recipe_tables(dims)
    params = _params(dims, kl_rows, tables)

    artifact_id = write_artifact(
        root,
        matrices,
        emb,
        kl,
        params,
        tables,
        _provenance(),
        dims=dims,
        exponents=exponents,
    )
    return root / artifact_id
