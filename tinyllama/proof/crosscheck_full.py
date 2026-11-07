"""Full-dimension cross check of one layer (PROVING_PLAN M3, O5).

The tiny cross check in :mod:`tinyllama.reference.crosscheck` compares the CPU
integer reference with FlashProve's ``model.py`` at d = 64.  This module does
the same for one layer of a real artifact: it reads the layer's weights,
scales and tables from the artifact, feeds the prompt's embedding rows into
both implementations on the CPU, and compares the recorded intermediates and
the layer output.

Run from the repository root with a torch CPU environment::

    python -m tinyllama.proof.crosscheck_full --artifact DIR --prompt P0 --layer 0

The comparison names are prefixed with ``layerN.``.  The exit code is 0 only
when every comparison is ``EQUAL``; it is 2 when torch or the integration is
missing, and 1 for any difference or refusal.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import sys

import numpy as np

from tinyllama.artifacts import FULL, load_artifact
from tinyllama.reference import crosscheck as tiny
from tinyllama.reference.layer import RangeFailure
from tinyllama.reference.layer import layer as reference_layer
from tinyllama.reference.model import _scales, _tables
from tinyllama.reference.run import _parse_dims_test, load_prompt

__all__ = [
    "ATTENTION_NAMES",
    "FEED_FORWARD_NAMES",
    "MATRIX_NAMES",
    "build_parameters",
    "main",
    "run",
]

ATTENTION_NAMES = tiny.ATTENTION_NAMES
FEED_FORWARD_NAMES = tiny.FEED_FORWARD_NAMES
MATRIX_NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")


class _ArtifactWeights:
    """A ``WeightSource`` backed by one layer of a TL-INT-1 artifact."""

    def __init__(self, torch, artifact, layer_index):
        self.torch = torch
        self.artifact = artifact
        self.layer_index = int(layer_index)
        self.device = torch.device("cpu")

    def matrix(self, name, layer=None, expert=None):
        if name == "Wh":
            array = self.artifact.wh
        else:
            which = tiny.FLASHPROVE_TO_REFERENCE[name]
            index = self.layer_index if layer is None else int(layer)
            array = self.artifact.matrix(index, which)
        return self.torch.from_numpy(
            np.array(array, dtype=np.int64, copy=True)
        )

    def embedding_rows(self, token_ids):
        if hasattr(token_ids, "detach"):
            ids = token_ids.detach().cpu().numpy()
        else:
            ids = np.asarray(token_ids)
        rows = self.artifact.emb[np.asarray(ids, dtype=np.int64)]
        return self.torch.from_numpy(np.array(rows, dtype=np.int64, copy=True))


def _table_digests(fp, artifact):
    """The artifact's three digests plus the sigmoid digest make_tables wants."""
    params = artifact.params
    digests = {
        name: [int(word) for word in params["digests"][name]]
        for name in ("trope", "texp", "tsilu")
    }
    sigmoid = fp.sigmoid_table()
    if params.get("gcl", 1):
        router = sigmoid[
            fp.KEY_OFFSET_16 - fp.KEY_OFFSET_15 :
            fp.KEY_OFFSET_16 + fp.KEY_OFFSET_15
        ]
        clamp_rows = [
            min(max(key, -127), 127) + 128
            for key in range(-fp.KEY_OFFSET_15, fp.KEY_OFFSET_15)
        ]
        words = router + clamp_rows
    else:
        words = sigmoid
    digests["tsig"] = fp.table_digest(words)
    return digests


def build_parameters(fp, artifact, layer_index):
    """Parameters for FlashProve's model.py from the artifact, without a policy file.

    ``Parameters.__post_init__`` refuses table digests when NE = 0 (O2).  A dense
    artifact cannot carry the sigmoid digest either.  Like the tiny cross check,
    this builds the object with ``object.__new__`` and sets every field, so only
    that validation is bypassed; every arithmetic value comes from the artifact.
    """
    params = artifact.params
    dims = artifact.dims
    layers = tuple(
        fp.LayerScales(*[int(value) for value in artifact.kl[index][:13]])
        for index in range(dims.L)
    )
    values = {
        "T": int(dims.T),
        "L": int(dims.L),
        "d": int(dims.D),
        "n_h": int(dims.H),
        "n_g": int(dims.G),
        "dh": int(dims.DH),
        "f": int(dims.F),
        "fe": 0,
        "NE": 0,
        "V": int(dims.V),
        "cs": int(params["cs"]),
        "theta0": int(params["theta0"]),
        "expS": int(params["S"]),
        "rx": int(params["rx"]),
        "eg": int(params["eg"]),
        "smax": int(params["smax"]),
        "NX": 5,
        "router_clamp": False,
        "gate_clamp": bool(params.get("gcl", 1)),
        "eps1": int(params["eps1"][layer_index]),
        "eps2": int(params["eps2"][layer_index]),
        "epsF": int(params["epsF"]),
        "MF": int(params["MF"]),
        "KF": int(params["KF"]),
        "seed": 0,
        "layers": layers,
        "table_digests": _table_digests(fp, artifact),
        "s": int(params["s"]),
    }
    try:
        return fp.Parameters(**values), None
    except fp.ParameterError as exc:
        parameters = object.__new__(fp.Parameters)
        for item in fields(fp.Parameters):
            object.__setattr__(parameters, item.name, values[item.name])
        return parameters, (
            "Parameters(...) refused this artifact (%s); built with "
            "object.__new__ to bypass that validation. For a valid TL-INT-1 "
            "artifact the refused rule is the NE=0/gcl table-digest rule."
            % exc
        )


def _failure_names(layer_index):
    prefix = "layer%d." % layer_index
    names = [prefix + "attention." + name for name in ATTENTION_NAMES]
    names += [prefix + "feed_forward." + name for name in FEED_FORWARD_NAMES]
    names.append(prefix + "feed_forward.x_out")
    return names


def _not_comparable(names, reason):
    return [tiny._compare(name, None, None, reason=reason) for name in names]


def run(artifact, token_ids, layer_index=0):
    """Compare one full-dimension layer and return a ``CrosscheckResult``."""
    fp = tiny._import_flashprove()
    torch = fp.torch
    layer_index = int(layer_index)
    if not 0 <= layer_index < int(artifact.dims.L):
        raise ValueError(
            "layer %d is outside 0 .. %d"
            % (layer_index, int(artifact.dims.L) - 1)
        )

    parameters, parameter_note = build_parameters(fp, artifact, layer_index)
    weights = _ArtifactWeights(torch, artifact, layer_index)
    model = fp.IntegerModel(parameters, weights, "cpu")

    ids = [int(token) for token in token_ids]
    x = np.asarray(artifact.emb[np.asarray(ids, dtype=np.int64)], dtype=np.int64)
    checks = fp.RangeChecks(len(ids), model.device)
    captured = {}

    def note(name, layer, values):
        captured[name] = tiny._to_numpy(values)

    result = tiny.CrosscheckResult()
    if parameter_note is not None:
        result.notes.append(parameter_note)
    result.notes.append(
        "layer %d, token ids %d; both sides read the same artifact weights"
        % (layer_index, len(ids))
    )

    names = _failure_names(layer_index)
    try:
        x_mid = model._attention(
            layer_index,
            torch.from_numpy(x),
            model.new_sequence(),
            checks,
            note,
        )
        captured["x_mid"] = tiny._to_numpy(x_mid)
        x_out = model._feed_forward(layer_index, x_mid, checks, note)
    except Exception as exc:  # keep a refusal as a reported result
        result.comparisons.extend(
            _not_comparable(names, "FlashProve refused the layer input: %s" % exc)
        )
        return result

    failure = checks.first_failure()
    if failure is not None:
        position, condition = failure
        result.comparisons.extend(
            _not_comparable(
                names,
                "FlashProve range condition %d failed at position %d"
                % (condition, position),
            )
        )
        return result

    matrices = {
        name: np.asarray(artifact.matrix(layer_index, name), dtype=np.int64)
        for name in MATRIX_NAMES
    }
    recorded = {}

    def record(name, layer, values):
        recorded[name] = tiny._to_numpy(values)

    try:
        our_out = reference_layer(
            x,
            matrices,
            _scales(artifact, layer_index),
            _tables(artifact),
            record=record,
            layer_index=layer_index,
        )
    except RangeFailure as exc:
        result.comparisons.extend(
            _not_comparable(names, "our reference range failure: %s" % exc)
        )
        return result

    prefix = "layer%d." % layer_index
    result.comparisons.extend(
        tiny._compare(prefix + "attention." + name, recorded[name], captured[name])
        for name in ATTENTION_NAMES
    )
    result.comparisons.extend(
        tiny._compare(
            prefix + "feed_forward." + name, recorded[name], captured[name]
        )
        for name in FEED_FORWARD_NAMES
    )
    result.comparisons.append(
        tiny._compare(prefix + "feed_forward.x_out", our_out, x_out)
    )
    return result


def _parse_ids(text):
    fields = text.replace(",", " ").split()
    if not fields:
        raise argparse.ArgumentTypeError("--ids needs at least one token id")
    try:
        return [int(field) for field in fields]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--ids needs comma or space separated integers"
        ) from exc


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare one full-dimension artifact layer between the integer "
            "reference and FlashProve model.py on the CPU."
        )
    )
    parser.add_argument("--artifact", required=True, help="artifact directory")
    parser.add_argument("--layer", required=True, type=int, help="layer index")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="prompt id in prompts_v1.json")
    source.add_argument(
        "--ids",
        type=_parse_ids,
        help="comma or space separated token ids for one window",
    )
    parser.add_argument(
        "--dims-test",
        type=_parse_dims_test,
        default=None,
        help=(
            "test only: load a tiny artifact with dims L,D,H,G,DH,F,V,T "
            "(never for a real artifact)"
        ),
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    dims = FULL if args.dims_test is None else args.dims_test
    try:
        artifact = load_artifact(args.artifact, dims=dims)
        if args.prompt is not None:
            _prompt, ids = load_prompt(args.prompt)
        else:
            ids = list(args.ids)
        result = run(artifact, ids, layer_index=args.layer)
    except ImportError as exc:
        print("crosscheck_full: %s" % exc, file=sys.stderr)
        return 2
    except Exception as exc:
        print("crosscheck_full: %s" % exc, file=sys.stderr)
        return 2

    for line in result.lines():
        print(line)
    for line in result.notes:
        print("note: %s" % line, file=sys.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
