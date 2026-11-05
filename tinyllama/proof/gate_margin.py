"""Measure the gate-key margin of the integer reference (PROVING_PLAN M3).

The M3 question for O2 option B is whether every ``gw`` that the reference
computes stays in the interval that FlashProve can write as a 16-bit key,
-32,768 .. 32,639.  This module runs :func:`tinyllama.reference.model.forward`
with a ``record`` callable and keeps, for each layer, the minimum and maximum
``gw``, the largest absolute ``gw`` and the count of entries outside that
interval.  The reference results are integers throughout; this module never
converts them to float.

Run from the repository root::

    python -m tinyllama.proof.gate_margin --artifact DIR --prompt P0
    python -m tinyllama.proof.gate_margin --artifact DIR --ids 1,2,3
    python -m tinyllama.proof.gate_margin --artifact DIR \\
        --window-file heldout.npy --window-index 0

The window form reads one row of a two-dimensional ``.npy`` array of token ids,
so a shell loop or ``xargs -P`` can run the calibration windows in parallel
without loading the whole array as text on the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import sys

import numpy as np

from tinyllama.artifacts import FULL, load_artifact
from tinyllama.reference.model import forward
from tinyllama.reference.run import _parse_dims_test, load_prompt

__all__ = [
    "GATE_KEY_MIN",
    "GATE_KEY_MAX",
    "GATE_KEY_MAX_ABS",
    "LayerMargin",
    "MarginReport",
    "collect",
    "format_report",
    "main",
]

# The interval in which FlashProve can write a gate key (docs/model.md:338,
# parameter-file.md:269).  Its absolute maximum is larger than the upper bound.
GATE_KEY_MIN = -32768
GATE_KEY_MAX = 32639
GATE_KEY_MAX_ABS = 32768


@dataclass(frozen=True)
class LayerMargin:
    """The ``gw`` margin of one layer over one or more windows."""

    layer: int
    minimum: int | None
    maximum: int | None
    count_outside: int
    observed: bool = True

    @classmethod
    def from_values(cls, layer, values):
        """Build one layer's margin from an integer array of ``gw`` values."""
        array = np.asarray(values)
        if array.dtype != np.int64:
            array = array.astype(np.int64)
        if array.size == 0:
            return cls(int(layer), None, None, 0, observed=False)
        low = int(array.min())
        high = int(array.max())
        outside = int(
            np.count_nonzero((array < GATE_KEY_MIN) | (array > GATE_KEY_MAX))
        )
        return cls(int(layer), low, high, outside, observed=True)

    @property
    def max_abs(self) -> int | None:
        if self.minimum is None or self.maximum is None:
            return None
        return max(abs(self.minimum), abs(self.maximum))

    @property
    def fits_16_bits(self) -> bool:
        """True when the layer was observed and every ``gw`` fits the interval."""
        return self.observed and self.count_outside == 0


@dataclass
class MarginReport:
    """Per-layer margins, the token ids, and an optional reference range failure."""

    layers: list[LayerMargin] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    prompt_id: str | None = None
    mode: str = "prefix"
    range_failure: object = None

    @property
    def total_outside(self) -> int:
        return sum(item.count_outside for item in self.layers)

    @property
    def max_abs(self) -> int | None:
        values = [item.max_abs for item in self.layers if item.max_abs is not None]
        return max(values) if values else None

    @property
    def all_layers_fit_16_bits(self) -> bool:
        return (
            bool(self.layers)
            and self.range_failure is None
            and all(item.fits_16_bits for item in self.layers)
        )


class _GateCollector:
    """The ``record`` callable: keep ``gw`` statistics only."""

    def __init__(self):
        self.values = {}

    def __call__(self, name, layer, values):
        if name != "gw" or layer is None:
            return
        margin = LayerMargin.from_values(layer, values)
        if not margin.observed:
            return
        current = self.values.get(margin.layer)
        if current is None:
            self.values[margin.layer] = margin
            return
        low = min(current.minimum, margin.minimum)
        high = max(current.maximum, margin.maximum)
        self.values[margin.layer] = LayerMargin(
            margin.layer,
            low,
            high,
            current.count_outside + margin.count_outside,
            observed=True,
        )


def collect(artifact, token_ids, *, mode="prefix", prompt_id=None):
    """Run the integer reference and return the per-layer gate-key margin."""
    collector = _GateCollector()
    result = forward(artifact, token_ids, mode=mode, record=collector)
    layers = int(artifact.dims.L)
    margins = [
        collector.values.get(index, LayerMargin(index, None, None, 0, False))
        for index in range(layers)
    ]
    return MarginReport(
        layers=margins,
        token_ids=[int(token) for token in token_ids],
        prompt_id=prompt_id,
        mode=mode,
        range_failure=result.range_failure,
    )


def _yes(flag):
    return "yes" if flag else "no"


def format_report(report):
    """Return the report as text lines; all values are decimal integers."""
    lines = []
    for item in report.layers:
        if not item.observed:
            lines.append("layer %02d unobserved" % item.layer)
            continue
        lines.append(
            "layer %02d max_abs_gw %d min_gw %d max_gw %d outside_16bit %d "
            "fits_16bit %s"
            % (
                item.layer,
                item.max_abs,
                item.minimum,
                item.maximum,
                item.count_outside,
                _yes(item.fits_16_bits),
            )
        )
    lines.append(
        "summary windows 1 tokens %d max_abs_gw %s outside_16bit %d "
        "all_layers_fit_16bit %s"
        % (
            len(report.token_ids),
            "none" if report.max_abs is None else str(report.max_abs),
            report.total_outside,
            _yes(report.all_layers_fit_16_bits),
        )
    )
    failure = report.range_failure
    if failure is not None:
        lines.append(
            "range_failure condition %d (%s) layer %s position %d index %d"
            % (
                failure.condition,
                failure.name,
                failure.layer,
                failure.position,
                failure.index,
            )
        )
    return lines


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


def _load_window(path, index):
    """Read one row of a two-dimensional integer ``.npy`` array of token ids."""
    try:
        windows = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("%s is not a readable npy file: %s" % (path, exc))
    if not isinstance(windows, np.ndarray) or windows.ndim != 2:
        raise ValueError(
            "%s must hold a two-dimensional array of token ids" % path
        )
    if windows.dtype.kind not in "iu":
        raise ValueError(
            "%s must hold integers, not %s" % (path, windows.dtype)
        )
    if not 0 <= index < windows.shape[0]:
        raise ValueError(
            "window index %d is outside 0 .. %d"
            % (index, windows.shape[0] - 1)
        )
    return [int(token) for token in windows[index]]


def _parser():
    parser = argparse.ArgumentParser(
        description=(
            "Measure the per-layer max |gw| and the count outside "
            "-32768 .. 32639 (PROVING_PLAN M3, O2 option B)."
        )
    )
    parser.add_argument("--artifact", required=True, help="artifact directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="prompt id in prompts_v1.json")
    source.add_argument(
        "--ids",
        type=_parse_ids,
        help="comma or space separated token ids for one window",
    )
    source.add_argument(
        "--window-file",
        metavar="FILE.npy",
        help=(
            "two-dimensional npy array of token ids; select one with "
            "--window-index"
        ),
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=None,
        metavar="N",
        help="row of --window-file to run (0-based)",
    )
    parser.add_argument(
        "--mode", choices=("prefix", "window"), default="prefix"
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
    if args.window_file is None and args.window_index is not None:
        print(
            "gate_margin: --window-index needs --window-file",
            file=sys.stderr,
        )
        return 2
    if args.window_file is not None and args.window_index is None:
        print(
            "gate_margin: --window-file needs --window-index",
            file=sys.stderr,
        )
        return 2
    dims = FULL if args.dims_test is None else args.dims_test
    try:
        artifact = load_artifact(args.artifact, dims=dims)
        if args.prompt is not None:
            _prompt, ids = load_prompt(args.prompt)
            prompt_id = args.prompt
        elif args.window_file is not None:
            ids = _load_window(args.window_file, args.window_index)
            prompt_id = None
        else:
            ids = list(args.ids)
            prompt_id = None
        report = collect(
            artifact, ids, mode=args.mode, prompt_id=prompt_id
        )
    except Exception as exc:  # artifact, prompt, window or token-id problem
        print("gate_margin: %s" % exc, file=sys.stderr)
        return 2

    for line in format_report(report):
        print(line)
    return 1 if report.range_failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
