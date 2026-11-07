"""Cross-check the CPU integer reference against FlashProve's arithmetic.

Run from the repository root::

    python -m tinyllama.reference.crosscheck

The command builds a very small integer model (d = 64, n_h = 4, n_g = 2,
dh = 16, f = 96, T = 16, V = 32, L = 1) from fixed-seed synthetic weights in
-4095 .. 4095 and drives the methods of
integrations/vllm/integer_model/integer_model/model.py directly.  It prints one
line for every compared primitive::

    NAME EQUAL
    NAME DIFFERENT first_index=... ours=... flashprove=... count=...
    NAME NOT-COMPARABLE reason=...

It exits 0 only when every compared primitive is EQUAL, otherwise 1.  If torch
is missing it prints a message and exits 2.

No file is read or written.  The parameter file reader is not used: Parameters
and LayerScales are built in memory.  FlashProve's own table functions compute
the words that make_tables() hashes, and integer_model.tables.table_digest()
computes the digests, so make_tables() passes its honest digest check.  The
only validation bypassed is Parameters.__post_init__'s refusal of gate_clamp
without routed experts (contract section 13, O2); the arithmetic is unchanged.

Both sides rotate neighbouring entries (2 i, 2 i + 1) of each head, and both
receive the same stored-layout [out][in] matrices, so no q/k row permutation is
applied between them.  Tables, scales and exponents use the contract's values
(TL-INT-1: cs 12, theta0 10000, expS 12, rx 8, eg 8, smax 4095, M/K from the
contract).  torch is imported lazily, inside the FlashProve side.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import sys
import types

import numpy as np

from . import primitives
from . import tables as reference_tables
from .layer import RangeFailure, _rotary as reference_rotary
from .layer import layer as reference_layer
from .model import forward as reference_forward

__all__ = [
    "Comparison",
    "CrosscheckResult",
    "TINY_DIMS",
    "run",
    "main",
]

# The tiny shape from the task: d, n_h, n_g, dh, f, T, V, L.
TINY_DIMS = {
    "d": 64,
    "n_h": 4,
    "n_g": 2,
    "dh": 16,
    "f": 96,
    "T": 16,
    "V": 32,
    "L": 1,
}

# The contract's table parameters.
CS = 12
THETA0 = 10000
EXP_S = 12
RX = 8
EG = 8
SMAX = 4095

# Norm exponents and regularizer from the contract (e_res = 11).
M = 37
K = 32
EPS = (4096 * (4**11) + 100000) // 200000

# Shifts chosen so that every intermediate of the synthetic model stays inside
# the contract's range conditions at the tiny shape and full 13-bit weights.
# They are the same on both sides; no value is changed to hide a difference.
LAYER_SCALES = {
    "kq": 28,
    "kk": 28,
    "kv": 24,
    "ko": 24,
    "kg": 8,
    "ku": 16,
    "kd": 16,
    "M1": M,
    "K1": K,
    "M2": M,
    "K2": K,
    "sh": 12,
    "rp": 1,
}

MATRIX_SHAPES = {
    "wq": (TINY_DIMS["n_h"] * TINY_DIMS["dh"], TINY_DIMS["d"]),
    "wk": (TINY_DIMS["n_g"] * TINY_DIMS["dh"], TINY_DIMS["d"]),
    "wv": (TINY_DIMS["n_g"] * TINY_DIMS["dh"], TINY_DIMS["d"]),
    "wo": (TINY_DIMS["d"], TINY_DIMS["n_h"] * TINY_DIMS["dh"]),
    "wg": (TINY_DIMS["f"], TINY_DIMS["d"]),
    "wu": (TINY_DIMS["f"], TINY_DIMS["d"]),
    "wd": (TINY_DIMS["d"], TINY_DIMS["f"]),
}

FLASHPROVE_TO_REFERENCE = {
    "Wq": "wq",
    "Wk": "wk",
    "Wv": "wv",
    "Wo": "wo",
    "Wg": "wg",
    "Wu": "wu",
    "Wd": "wd",
    "Wh": "wh",
}

ATTENTION_NAMES = ("a", "qw", "q", "kw", "k", "vw", "v", "o", "x_mid")
FEED_FORWARD_NAMES = ("m", "gw", "key", "s", "u")


@dataclass(frozen=True)
class Comparison:
    """The result of comparing one primitive or one recorded intermediate."""

    name: str
    status: str  # "EQUAL", "DIFFERENT" or "NOT-COMPARABLE"
    first_index: int | None = None
    ours: object = None
    flashprove: object = None
    count: int = 0
    reason: str = ""

    def line(self) -> str:
        if self.status == "EQUAL":
            return f"{self.name} EQUAL"
        if self.status == "DIFFERENT":
            return (
                f"{self.name} DIFFERENT first_index={self.first_index} "
                f"ours={self.ours} flashprove={self.flashprove} count={self.count}"
            )
        return f"{self.name} NOT-COMPARABLE reason={self.reason}"


@dataclass
class CrosscheckResult:
    """All per-primitive comparisons, plus notes that are not comparisons."""

    comparisons: list[Comparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if not self.comparisons:
            return 1
        return 0 if all(item.status == "EQUAL" for item in self.comparisons) else 1

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def lines(self) -> list[str]:
        return [item.line() for item in self.comparisons]

    def by_name(self) -> dict[str, Comparison]:
        return {item.name: item for item in self.comparisons}


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def _compare(name, ours, flashprove, reason=None) -> Comparison:
    if reason is not None:
        return Comparison(name, "NOT-COMPARABLE", reason=reason)
    left = _to_numpy(ours).reshape(-1)
    right = _to_numpy(flashprove).reshape(-1)
    if left.shape != right.shape:
        return Comparison(
            name,
            "DIFFERENT",
            first_index=0,
            ours=f"shape {left.shape}",
            flashprove=f"shape {right.shape}",
            count=int(left.size + right.size),
        )
    different = np.flatnonzero(left != right)
    if different.size == 0:
        return Comparison(name, "EQUAL")
    first = int(different[0])
    return Comparison(
        name,
        "DIFFERENT",
        first_index=first,
        ours=int(left[first]),
        flashprove=int(right[first]),
        count=int(different.size),
    )


def _import_flashprove():
    """Import torch and the integration lazily, with a clear missing-torch error."""
    repository = Path(__file__).resolve().parents[2]
    integration = repository / "integrations" / "vllm" / "integer_model"
    if str(integration) not in sys.path:
        sys.path.insert(0, str(integration))
    try:
        import torch  # noqa: F401
        from integer_model.arithmetic import (  # noqa: F401
            balanced_low_byte,
            clamp,
            floor_quotient,
            rounded_quotient,
        )
        from integer_model.model import (  # noqa: F401
            IntegerModel,
            RangeConditionFailed,
            _RangeChecks,
        )
        from integer_model.parameters import (  # noqa: F401
            LayerScales,
            ParameterError,
            Parameters,
        )
        from integer_model.tables import (  # noqa: F401
            KEY_OFFSET_15,
            KEY_OFFSET_16,
            rotary_tables,
            sigmoid_table,
            silu_table,
            softmax_table,
            table_digest,
        )
    except ImportError as exc:  # torch or the integration is missing
        raise ImportError(
            "crosscheck needs torch and integrations/vllm/integer_model; "
            "run it with /workspace/tl-torch-venv/bin/python"
        ) from exc

    return types.SimpleNamespace(
        torch=torch,
        balanced_low_byte=balanced_low_byte,
        clamp=clamp,
        floor_quotient=floor_quotient,
        rounded_quotient=rounded_quotient,
        IntegerModel=IntegerModel,
        RangeConditionFailed=RangeConditionFailed,
        RangeChecks=_RangeChecks,
        LayerScales=LayerScales,
        ParameterError=ParameterError,
        Parameters=Parameters,
        KEY_OFFSET_15=KEY_OFFSET_15,
        KEY_OFFSET_16=KEY_OFFSET_16,
        rotary_tables=rotary_tables,
        sigmoid_table=sigmoid_table,
        silu_table=silu_table,
        softmax_table=softmax_table,
        table_digest=table_digest,
    )


def _build_parameters(fp):
    """Parameters and LayerScales in memory, with honest FlashProve table digests."""
    gate_clamp = True
    layer = fp.LayerScales(**LAYER_SCALES)

    cosines, sines = fp.rotary_tables(TINY_DIMS["T"], TINY_DIMS["dh"], CS, THETA0)
    softmax = fp.softmax_table(EXP_S, RX)
    silu = fp.silu_table(EG, SMAX)
    sigmoid = fp.sigmoid_table()
    router_sigmoid = sigmoid[
        fp.KEY_OFFSET_16 - fp.KEY_OFFSET_15 : fp.KEY_OFFSET_16 + fp.KEY_OFFSET_15
    ]
    clamp_rows = [
        min(max(key, -127), 127) + 128
        for key in range(-fp.KEY_OFFSET_15, fp.KEY_OFFSET_15)
    ]
    sigmoid_words = router_sigmoid + clamp_rows if gate_clamp else sigmoid
    digests = {
        "tsilu": fp.table_digest([entry + 4096 for entry in silu]),
        "tsig": fp.table_digest(sigmoid_words),
        "trope": fp.table_digest([entry + 4096 for entry in cosines + sines]),
        "texp": fp.table_digest(softmax),
    }

    values = {
        "T": TINY_DIMS["T"],
        "L": TINY_DIMS["L"],
        "d": TINY_DIMS["d"],
        "n_h": TINY_DIMS["n_h"],
        "n_g": TINY_DIMS["n_g"],
        "dh": TINY_DIMS["dh"],
        "f": TINY_DIMS["f"],
        "fe": 0,
        "NE": 0,
        "V": TINY_DIMS["V"],
        "cs": CS,
        "theta0": THETA0,
        "expS": EXP_S,
        "rx": RX,
        "eg": EG,
        "smax": SMAX,
        "NX": 5,
        "router_clamp": False,
        "gate_clamp": gate_clamp,
        "eps1": EPS,
        "eps2": EPS,
        "epsF": EPS,
        "MF": M,
        "KF": K,
        "seed": 0,
        "layers": (layer,),
        "table_digests": digests,
        "s": 20,
    }

    note = None
    try:
        parameters = fp.Parameters(**values)
    except fp.ParameterError as exc:
        # Contract section 13, O2: Parameters refuses gcl without experts.
        # Bypass only that validation; every arithmetic parameter is unchanged.
        parameters = object.__new__(fp.Parameters)
        for item in fields(fp.Parameters):
            object.__setattr__(parameters, item.name, values[item.name])
        note = (
            "Parameters(...) refused this shape (%s); constructed with "
            "object.__new__ to bypass only the O2 gate_clamp/NE=0 validation."
            % exc
        )
    return parameters, layer, note, (cosines, sines, softmax, silu)


def _synthetic_weights(seed=20261105):
    rng = np.random.default_rng(seed)
    matrices = {
        name: rng.integers(-4096, 4096, size=shape, dtype=np.int64)
        for name, shape in MATRIX_SHAPES.items()
    }
    embedding = rng.integers(
        -1000, 1001, size=(TINY_DIMS["V"], TINY_DIMS["d"]), dtype=np.int64
    )
    head = rng.integers(
        -4096, 4096, size=(TINY_DIMS["V"], TINY_DIMS["d"]), dtype=np.int64
    )
    return matrices, embedding, head


class _InMemoryWeights:
    """A tiny WeightSource: fixed tensors, stored layout [out][in]."""

    def __init__(self, torch, matrices, embedding, head, device="cpu"):
        self.device = torch.device(device)
        self._matrices = {
            name: torch.from_numpy(value).to(self.device)
            for name, value in matrices.items()
        }
        self._head = torch.from_numpy(head).to(self.device)
        self._embedding = torch.from_numpy(embedding).to(self.device)

    def matrix(self, name, layer=None, expert=None):
        if name == "Wh":
            return self._head
        return self._matrices[FLASHPROVE_TO_REFERENCE[name]]

    def embedding_rows(self, token_ids):
        return self._embedding[token_ids.to(self.device)]


class _ReferenceArtifact:
    """The small adapter that tinyllama.reference.model.forward needs."""

    def __init__(self, parameters, layer, matrices, embedding, head, table_map):
        self.dims = types.SimpleNamespace(
            L=parameters.L, T=parameters.T, V=parameters.V
        )
        self.emb = embedding
        self.wh = head
        self.params = {
            "MF": parameters.MF,
            "KF": parameters.KF,
            "epsF": parameters.epsF,
            "eps1": np.asarray([parameters.eps1] * parameters.L, dtype=np.int64),
            "eps2": np.asarray([parameters.eps2] * parameters.L, dtype=np.int64),
        }
        self.kl = np.asarray(
            [[layer.kq, layer.kk, layer.kv, layer.ko, layer.kg, layer.ku, layer.kd,
              layer.M1, layer.K1, layer.M2, layer.K2, layer.sh, layer.rp]],
            dtype=np.int64,
        )
        self._matrices = matrices
        self._tables = table_map

    def matrix(self, layer_index, name):
        return self._matrices[name]

    def table(self, name):
        return self._tables[name]


def _compare_arithmetic(fp, result):
    torch = fp.torch
    edges = [
        -(1 << 40),
        -32769,
        -32768,
        -256,
        -129,
        -128,
        -127,
        -25,
        -24,
        -9,
        -8,
        -3,
        -1,
        0,
        1,
        3,
        7,
        8,
        24,
        (1 << 40) - 1,
    ]
    values = torch.tensor(edges, dtype=torch.int64)

    for shift in (1, 2, 4, 8, 16, 20, 32):
        ours = [primitives.rq(value, shift) for value in edges]
        theirs = fp.rounded_quotient(values, shift)
        result.comparisons.append(_compare(f"rq.k{shift}", ours, theirs))

    for bound in (1, 127, 2047, 4095):
        ours = [primitives.clamp(value, bound) for value in edges]
        theirs = fp.clamp(values, bound)
        result.comparisons.append(_compare(f"clamp.b{bound}", ours, theirs))

    for divisor in (1, 2, 3, 7, 255, 256, 1 << 20):
        ours = [primitives.floor_div(value, divisor) for value in edges]
        theirs = fp.floor_quotient(values, torch.tensor(divisor, dtype=torch.int64))
        result.comparisons.append(_compare(f"floor_div.d{divisor}", ours, theirs))

    ours_low = [primitives.low8(value) for value in edges]
    theirs_low = fp.balanced_low_byte(values)
    result.comparisons.append(_compare("low8", ours_low, theirs_low))

    gate_values = edges + [300, -300, 128, -129, 40000, -40000, 32768, -32768]
    gate_tensor = torch.tensor(gate_values, dtype=torch.int64)
    g0 = fp.balanced_low_byte(gate_tensor)
    gh = (gate_tensor - g0) >> 8
    gc = fp.clamp(gh, 127)
    key = 256 * gc + g0
    ours = [primitives.gate_key(value) for value in gate_values]
    result.comparisons.append(
        _compare("gate_key.g0", [item["g0"] for item in ours], g0)
    )
    result.comparisons.append(
        _compare("gate_key.gh", [item["gh"] for item in ours], gh)
    )
    result.comparisons.append(
        _compare("gate_key.gc", [item["gc"] for item in ours], gc)
    )
    result.comparisons.append(
        _compare("gate_key.key", [item["key"] for item in ours], key)
    )


def _compare_tables(model):
    ours_c, ours_s = reference_tables.rope_tables(
        TINY_DIMS["T"], TINY_DIMS["dh"], THETA0
    )
    ours_ex = np.asarray(reference_tables.exp_table(), dtype=np.int64)
    ours_ts = np.asarray(reference_tables.silu_table(), dtype=np.int64)

    flashprove_c = _to_numpy(model.tables.cos)
    flashprove_s = _to_numpy(model.tables.sin)
    flashprove_ex = _to_numpy(model.tables.softmax)
    flashprove_ts = _to_numpy(model.tables.silu)

    # make_tables() appends one zero entry to EX for keys from u0 on.
    if flashprove_ex.size != ours_ex.size + 1 or int(flashprove_ex[-1]) != 0:
        trailing = Comparison(
            "softmax_table",
            "DIFFERENT",
            first_index=0,
            ours=f"length {ours_ex.size}",
            flashprove=f"length {flashprove_ex.size}, last={int(flashprove_ex[-1])}",
            count=1,
        )
    else:
        trailing = _compare("softmax_table", ours_ex, flashprove_ex[:-1])

    return [
        _compare("rope_c", ours_c, flashprove_c),
        _compare("rope_s", ours_s, flashprove_s),
        trailing,
        _compare("silu_table", ours_ts, flashprove_ts),
    ]


def _compare_norm(fp, model):
    torch = fp.torch
    rng = np.random.default_rng(17)
    x = rng.integers(-1500, 1501, size=(4, TINY_DIMS["d"]), dtype=np.int64)
    checks = fp.RangeChecks(x.shape[0], model.device)
    captured = {}

    def note(name, layer, values):
        captured[name] = _to_numpy(values)

    model._norm(
        torch.from_numpy(x), M, K, EPS, checks, note, None, "aF"
    )
    failure = checks.first_failure()
    if failure is not None:
        reason = "FlashProve range condition %d failed at position %d" % failure
        return [
            _compare(name, None, None, reason=reason)
            for name in ("norm.sigma", "norm.Rc", "norm.w", "norm.nr")
        ]

    ours = [primitives.norm(row, M, K, EPS) for row in x]
    return [
        _compare("norm.sigma", [item["sigma"] for item in ours], captured["aF_sigma"]),
        _compare("norm.Rc", [item["rc"] for item in ours], captured["aF_Rc"]),
        _compare("norm.w", [item["w"] for item in ours], captured["aF_w"]),
        _compare("norm.nr", [item["nr"] for item in ours], captured["aF"]),
    ]


def _compare_rotary(fp, model):
    torch = fp.torch
    rng = np.random.default_rng(23)
    n = 3
    start = 2
    values = rng.integers(
        -(5 * 10**8), 5 * 10**8, size=(n, TINY_DIMS["n_h"] * TINY_DIMS["dh"]),
        dtype=np.int64,
    )
    flashprove = _to_numpy(
        model._rotary(torch.from_numpy(values), 0, start)
    )
    cosines = _to_numpy(model.tables.cos)
    sines = _to_numpy(model.tables.sin)
    ours = _to_numpy(
        reference_rotary(
            values.reshape(n, TINY_DIMS["n_h"], TINY_DIMS["dh"]),
            cosines[start : start + n],
            sines[start : start + n],
        ).reshape(n, -1)
    )
    return [_compare("rotary", ours, flashprove)]


def _compare_layer(fp, model, parameters, matrices, embedding, tables, tokens):
    x = embedding[np.asarray(tokens, dtype=np.int64)]
    checks = fp.RangeChecks(len(tokens), model.device)
    captured = {}

    def note(name, layer, values):
        captured[name] = _to_numpy(values)

    try:
        x_mid = model._attention(
            0, fp.torch.from_numpy(x), model.new_sequence(), checks, note
        )
        captured["x_mid"] = _to_numpy(x_mid)
        x_out = model._feed_forward(0, x_mid, checks, note)
    except Exception as exc:  # keep a refusal as a reported result
        reason = "FlashProve refused the layer input: %s" % exc
        names = [f"attention.{name}" for name in ATTENTION_NAMES]
        names += [f"feed_forward.{name}" for name in FEED_FORWARD_NAMES]
        names.append("feed_forward.x_out")
        return [_compare(name, None, None, reason=reason) for name in names]

    failure = checks.first_failure()
    if failure is not None:
        reason = "FlashProve range condition %d failed at position %d" % failure
        names = [f"attention.{name}" for name in ATTENTION_NAMES]
        names += [f"feed_forward.{name}" for name in FEED_FORWARD_NAMES]
        names.append("feed_forward.x_out")
        return [_compare(name, None, None, reason=reason) for name in names]

    scales = dict(LAYER_SCALES)
    scales["eps1"] = parameters.eps1
    scales["eps2"] = parameters.eps2
    recorded = {}
    try:
        our_out = reference_layer(
            x,
            matrices,
            scales,
            tables,
            record=lambda name, layer, values: recorded.__setitem__(
                name, _to_numpy(values)
            ),
            layer_index=0,
        )
    except RangeFailure as exc:
        reason = "our reference range failure: %s" % exc
        names = [f"attention.{name}" for name in ATTENTION_NAMES]
        names += [f"feed_forward.{name}" for name in FEED_FORWARD_NAMES]
        names.append("feed_forward.x_out")
        return [_compare(name, None, None, reason=reason) for name in names]

    comparisons = [
        _compare(f"attention.{name}", recorded[name], captured[name])
        for name in ATTENTION_NAMES
    ]
    comparisons += [
        _compare(f"feed_forward.{name}", recorded[name], captured[name])
        for name in FEED_FORWARD_NAMES
    ]
    comparisons.append(_compare("feed_forward.x_out", our_out, x_out))
    return comparisons


def _compare_extend(
    fp, model, parameters, matrices, embedding, head, tables, tokens
):
    note = {}
    try:
        sequence = model.new_sequence()
        choices = model.extend(
            sequence,
            tokens,
            record=lambda name, layer, values: note.__setitem__(
                name, _to_numpy(values)
            ),
        )
    except fp.RangeConditionFailed as exc:
        reason = "FlashProve extend raised RangeConditionFailed: %s" % exc
        return [
            _compare(name, None, None, reason=reason)
            for name in ("extend.x_last", "extend.aF", "extend.lam",
                         "extend.first_choice", "extend.next_token")
        ]
    except Exception as exc:
        reason = "FlashProve extend refused: %s" % exc
        return [
            _compare(name, None, None, reason=reason)
            for name in ("extend.x_last", "extend.aF", "extend.lam",
                         "extend.first_choice", "extend.next_token")
        ]

    table_map = {
        "rope_c": tables["C"],
        "rope_s": tables["S"],
        "softmax_ex": tables["EX"],
        "silu_ts": tables["Ts"],
    }
    artifact = _ReferenceArtifact(
        parameters, parameters.layers[0], matrices, embedding, head, table_map
    )
    recorded = {}
    result = reference_forward(
        artifact,
        tokens,
        mode="prefix",
        record=lambda name, layer, values: recorded.__setitem__(
            name, _to_numpy(values)
        ),
    )
    if result.range_failure is not None:
        reason = "our reference range failure: %s" % result.range_failure
        return [
            _compare(name, None, None, reason=reason)
            for name in ("extend.x_last", "extend.aF", "extend.lam",
                         "extend.first_choice", "extend.next_token")
        ]

    flashprove_next = int(choices[-1])
    return [
        _compare("extend.x_last", recorded["x_last"], note["x_last"]),
        _compare("extend.aF", recorded["aF"], note["aF"]),
        _compare("extend.lam", recorded["lam"], note["lam"]),
        _compare(
            "extend.first_choice", recorded["first_choice"], note["first_choice"]
        ),
        _compare("extend.next_token", result.next_token, flashprove_next),
    ]


def run() -> CrosscheckResult:
    """Build the tiny model, compare every primitive, and return the results."""
    fp = _import_flashprove()
    torch = fp.torch

    parameters, layer, parameter_note, built = _build_parameters(fp)
    matrices, embedding, head = _synthetic_weights()
    weights = _InMemoryWeights(torch, matrices, embedding, head)
    model = fp.IntegerModel(parameters, weights, "cpu")

    result = CrosscheckResult()
    if parameter_note is not None:
        result.notes.append(parameter_note)
    result.notes.append(
        "table digests: trope, texp, tsilu and tsig were computed with "
        "integer_model.tables.table_digest over the same words make_tables hashes"
    )
    result.notes.append(
        "rotary pairing: both sides rotate adjacent entries (2i, 2i+1) of each "
        "head; the shared weights are not permuted between the two sides"
    )

    _compare_arithmetic(fp, result)
    result.comparisons.extend(_compare_tables(model))
    result.comparisons.extend(_compare_norm(fp, model))
    result.comparisons.extend(_compare_rotary(fp, model))

    tables = {
        "C": np.asarray(built[0], dtype=np.int64).reshape(
            TINY_DIMS["T"], TINY_DIMS["dh"] // 2
        ),
        "S": np.asarray(built[1], dtype=np.int64).reshape(
            TINY_DIMS["T"], TINY_DIMS["dh"] // 2
        ),
        "EX": np.asarray(built[2], dtype=np.int64),
        "Ts": np.asarray(built[3], dtype=np.int64),
    }
    tokens = [3, 7, 11, 19]
    result.comparisons.extend(
        _compare_layer(fp, model, parameters, matrices, embedding, tables, tokens)
    )
    result.comparisons.extend(
        _compare_extend(
            fp, model, parameters, matrices, embedding, head, tables, tokens
        )
    )
    return result


def main(argv=None) -> int:
    """Print one line per comparison and return the process exit code."""
    try:
        result = run()
    except ImportError as exc:
        print(f"crosscheck: {exc}", file=sys.stderr)
        return 2
    for line in result.lines():
        print(line)
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
