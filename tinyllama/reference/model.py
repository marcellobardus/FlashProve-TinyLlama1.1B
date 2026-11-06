"""The full TL-INT-1 integer model: token ids to integer logits.

The inference path here is integer only.  Every table and every scale is read
from the loaded artifact; no float value is produced, and no module that uses
float on the inference path is imported.  Weight matrices are mapped one layer
at a time, so the whole artifact is never copied into RAM.

The final norm uses the same norm helper as :mod:`tinyllama.reference.layer`.
It is outside the layers, so its :class:`RangeFailure` carries the layer number
``L`` (one past the last layer, 0 .. L-1) as a sentinel.

QUESTION: INTERFACES.md I3.2 says "21 per layer and 8 after the layers" (about
470 records), but the I3.1 table has seven names after the layers (x_last,
aF_sigma, aF_Rc, aF_w, aF, lam, first_choice).  This module emits those seven
and does not invent an eighth name.

QUESTION: I3.1 fixes the condition order within a layer as 1, 2, 3, 4, 6, 7, 8,
but the existing layer.py norm helper checks 8 before 1 and 2, and
test_block_layer.py locks that behaviour.  The model reuses that helper for the
final norm rather than reordering it.
"""

from dataclasses import dataclass

import numpy as np

from ..contract import PAD_ID
from .layer import RangeFailure, _norm_rows, layer

__all__ = [
    "Result",
    "forward",
    "first_argmax",
    "LAYER_RECORD_NAMES",
    "FINAL_RECORD_NAMES",
]

MATRIX_NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")

LAYER_RECORD_NAMES = (
    "x",
    "a_sigma",
    "a_Rc",
    "a_w",
    "a",
    "qw",
    "q",
    "kw",
    "k",
    "vw",
    "v",
    "o",
    "x_mid",
    "m_sigma",
    "m_Rc",
    "m_w",
    "m",
    "gw",
    "key",
    "s",
    "u",
)

FINAL_RECORD_NAMES = (
    "x_last",
    "aF_sigma",
    "aF_Rc",
    "aF_w",
    "aF",
    "lam",
    "first_choice",
)

SCALE_NAMES = (
    "kq",
    "kk",
    "kv",
    "ko",
    "kg",
    "ku",
    "kd",
    "M1",
    "K1",
    "M2",
    "K2",
    "sh",
    "rp",
)

INT64_MAX = (1 << 63) - 1
ACTIVATION_BOUND = 2047
WEIGHT_BOUND = 4095


@dataclass
class Result:
    """The forward result; on a range failure the three value fields are None."""

    lam_last: object
    next_token: object
    range_failure: object
    n: int


def first_argmax(values):
    """Return the least index of the maximum; integer only, no library argmax."""
    length = len(values)
    if length == 0:
        raise ValueError("first_argmax needs at least one value")
    best_index = 0
    best_value = int(values[0])
    for index in range(1, length):
        value = int(values[index])
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


def _check_token_ids(artifact, token_ids):
    try:
        ids = list(token_ids)
    except TypeError as exc:
        raise ValueError("token_ids must be a sequence of integers") from exc
    window = int(artifact.dims.T)
    vocab = int(artifact.dims.V)
    if not ids:
        raise ValueError("token_ids must not be empty")
    if len(ids) > window:
        raise ValueError(
            "token_ids has %d entries, more than the window %d"
            % (len(ids), window)
        )
    checked = []
    for token in ids:
        if isinstance(token, (bool, np.bool_)):
            raise ValueError("token ids must be integers")
        if not isinstance(token, (int, np.integer)):
            raise ValueError("token ids must be integers")
        value = int(token)
        if value < 0 or value >= vocab:
            raise ValueError(
                "token id %d is outside 0 .. %d" % (value, vocab - 1)
            )
        checked.append(value)
    return checked


def _tables(artifact):
    return {
        "C": np.asarray(artifact.table("rope_c"), dtype=np.int64),
        "S": np.asarray(artifact.table("rope_s"), dtype=np.int64),
        "EX": np.asarray(artifact.table("softmax_ex"), dtype=np.int64),
        "Ts": np.asarray(artifact.table("silu_ts"), dtype=np.int64),
    }


def _scales(artifact, layer_index):
    params = artifact.params
    row = artifact.kl[layer_index]
    scales = {
        name: int(row[index]) for index, name in enumerate(SCALE_NAMES)
    }
    scales["eps1"] = int(params["eps1"][layer_index])
    scales["eps2"] = int(params["eps2"][layer_index])
    return scales


def _logits(activations, wh):
    """aF Wh^T, checked against the contract's int64 bound of section 3.1."""
    d = int(activations.shape[1])
    if d * ACTIVATION_BOUND * WEIGHT_BOUND > INT64_MAX:
        raise OverflowError(
            "the %d-term logit product can leave int64" % d
        )
    product = np.matmul(activations, np.asarray(wh).T)
    if product.dtype != np.int64:
        product = product.astype(np.int64)
    return product


def _first_choice_rows(lam):
    choices = np.empty(lam.shape[0], dtype=np.int64)
    for position in range(lam.shape[0]):
        choices[position] = first_argmax(lam[position])
    return choices


def forward(artifact, token_ids, mode="prefix", record=None):
    """Run the integer model on 1 .. T token ids.

    ``mode`` is ``"prefix"`` (no padding) or ``"window"`` (right padding with
    id 0 to the artifact's T).  ``record``, if given, is called as
    ``record(name, layer, values)`` in the I3.1 order.  The next token always
    comes from the real position ``n - 1``, in both modes.
    """
    if mode not in ("prefix", "window"):
        raise ValueError("mode must be 'prefix' or 'window'")
    ids = _check_token_ids(artifact, token_ids)
    n = len(ids)
    window = int(artifact.dims.T)
    layers = int(artifact.dims.L)

    sequence = ids
    if mode == "window":
        sequence = ids + [PAD_ID] * (window - n)

    tokens = np.asarray(sequence, dtype=np.int64)
    x = np.asarray(artifact.emb[tokens], dtype=np.int64)
    tables = _tables(artifact)

    try:
        for layer_index in range(layers):
            weights = {
                name: np.asarray(
                    artifact.matrix(layer_index, name), dtype=np.int64
                )
                for name in MATRIX_NAMES
            }
            x = layer(
                x,
                weights,
                _scales(artifact, layer_index),
                tables,
                record=record,
                layer_index=layer_index,
            )
        x_last = x
        if record is not None:
            record("x_last", None, x_last)

        params = artifact.params
        aF_sigma, aF_rc, aF_w, aF = _norm_rows(
            x_last,
            int(params["MF"]),
            int(params["KF"]),
            int(params["epsF"]),
            layers,
            "x_last",
            "aF_sigma",
            "aF_Rc",
            "aF_w",
            "aF_sigma",
        )
        if record is not None:
            record("aF_sigma", None, aF_sigma)
            record("aF_Rc", None, aF_rc)
            record("aF_w", None, aF_w)
            record("aF", None, aF)
    except RangeFailure as failure:
        return Result(None, None, failure, n)

    if record is None:
        lam_last = _logits(aF[n - 1 : n], artifact.wh)[0].copy()
    else:
        lam = _logits(aF, artifact.wh)
        choices = _first_choice_rows(lam)
        record("lam", None, lam)
        record("first_choice", None, choices)
        lam_last = lam[n - 1].copy()

    next_token = first_argmax(lam_last)
    return Result(lam_last, int(next_token), None, n)
