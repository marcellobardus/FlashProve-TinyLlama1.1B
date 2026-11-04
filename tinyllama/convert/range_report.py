# Run on the pod (repository at /workspace/repo, checkpoint at /workspace/ckpt,
# calibration npy files at /workspace/calib):
#   cd /workspace/repo && PYTHONPATH=. python3 -m tinyllama.convert.range_report \
#       --artifact /workspace/repo/artifacts/TL-INT-1/<ARTIFACT_ID> \
#       --calib /workspace/calib/fit.npy --heldout /workspace/calib/heldout.npy \
#       --checkpoint /workspace/ckpt --workers 8 \
#       --report /workspace/repo/artifacts/TL-INT-1/range-report.json
#
# --sets both adds the fit windows to the report.  --max-windows K keeps the
# first K windows of each selected set.  --checkpoint is optional; without it
# the report has float_maxima null and float_over_limit [].
#
# Add --test-dims '{"L":2,"D":64,"H":4,"G":2,"DH":16,"F":128,"V":256,"T":32}'
# and --manifest PATH only for the synthetic tests of this repository.

"""Run the P4 integer range report over heldout, fit or both calibration sets.

The integer pass is the exact integer reference from tinyllama.reference; this
module only loads the npy windows, optionally builds the float model, calls
tinyllama.convert.calibrate.range_report and writes its JSON.  Workers load
the artifact themselves with read only memmaps, so no weight array is pickled.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from tinyllama.artifacts import FULL
from tinyllama.convert.calibrate import (
    CalibrationError,
    CheckpointReader,
    FloatModel,
    range_report,
)
from tinyllama.convert.run import _parse_test_dims, _safetensors_path


class ReportError(ValueError):
    """The range report cannot run with the arguments or files that it has."""


def _load_windows(path, dims):
    if path is None:
        raise ReportError("a calibration npy path is required for this set")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReportError("{} is not a readable npy file: {}".format(path, exc))
    if not isinstance(array, np.ndarray):
        raise ReportError("{} does not contain one array".format(path))
    if array.dtype.kind not in "iu":
        raise ReportError(
            "{} has dtype {}, expected integers".format(path, array.dtype)
        )
    if array.ndim != 2:
        raise ReportError(
            "{} has shape {}, expected two dimensions".format(path, array.shape)
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ReportError("{} is empty".format(path))
    if array.shape[1] > dims.T:
        raise ReportError(
            "{} has {} positions in a window, the shape admits {}".format(
                path, array.shape[1], dims.T
            )
        )
    if array.size and (int(array.min()) < 0 or int(array.max()) >= dims.V):
        raise ReportError(
            "{} has a token id outside 0 .. {}".format(path, dims.V - 1)
        )
    return np.ascontiguousarray(array, dtype=np.int32)


def _selected_sets(name):
    if name == "both":
        return ["heldout", "fit"]
    return [name]


def _load_manifest(path):
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    from tinyllama import manifest as manifest_module

    return manifest_module.load_manifest()


def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m tinyllama.convert.range_report",
        description="Run the integer P4 range report on artifact windows.",
    )
    parser.add_argument("--artifact", required=True, metavar="DIR")
    parser.add_argument("--calib", default=None, metavar="FIT.npy")
    parser.add_argument("--heldout", default=None, metavar="HELDOUT.npy")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        metavar="N",
        help="worker processes (default: min(8, cpu count))",
    )
    parser.add_argument("--report", required=True, metavar="OUT.json")
    parser.add_argument(
        "--sets",
        choices=("heldout", "fit", "both"),
        default="heldout",
        metavar="SET",
        help="heldout, fit or both (default: heldout)",
    )
    parser.add_argument("--max-windows", type=int, default=None, metavar="K")
    parser.add_argument("--checkpoint", default=None, metavar="DIR")
    parser.add_argument("--manifest", default=None, metavar="PATH")
    parser.add_argument("--test-dims", default=None, metavar="JSON")
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="seconds between layer progress lines (default: 60)",
    )
    args = parser.parse_args(argv)

    if args.workers < 1:
        raise ReportError("--workers must be at least 1")
    if args.max_windows is not None and args.max_windows < 1:
        raise ReportError("--max-windows must be at least 1")

    dims = FULL if args.test_dims is None else _parse_test_dims(args.test_dims)

    paths = {"heldout": args.heldout, "fit": args.calib}
    selected = _selected_sets(args.sets)
    windows = {}
    for name in selected:
        array = _load_windows(paths[name], dims)
        if args.max_windows is not None:
            array = np.ascontiguousarray(
                array[: args.max_windows], dtype=np.int32
            )
        windows[name] = array

    float_model = None
    if args.checkpoint is not None:
        manifest = _load_manifest(args.manifest)
        reader = CheckpointReader(_safetensors_path(args.checkpoint), manifest)
        float_model = FloatModel(reader, dims)

    report = range_report(
        args.artifact,
        windows,
        dims=dims,
        float_model=float_model,
        workers=args.workers,
        progress_interval=args.progress_interval,
    )

    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="ascii"
    )
    return 1 if report["range_failure_count"] else 0


def main(argv=None):
    """Run the range report. Return 0 without failures, 1 with, 2 on errors."""
    try:
        return _main(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except (ReportError, CalibrationError, ValueError, OSError) as exc:
        print(
            "tinyllama.convert.range_report: {}".format(exc), file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
