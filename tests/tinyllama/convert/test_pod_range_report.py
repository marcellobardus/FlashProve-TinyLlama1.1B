import json
import multiprocessing
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import synthetic_checkpoint

from tinyllama.artifacts import load_artifact, validate_artifact
from tinyllama.convert import range_report as range_report_module
from tinyllama.convert import run
from tinyllama.convert.calibrate import CalibrationError, range_report

REPO_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_KEYS = {
    "contract",
    "artifact_id",
    "windows",
    "integer_reference",
    "range_failures",
    "range_failure_count",
    "per_window",
    "float_maxima",
    "float_limit",
    "float_over_limit",
}


def _dims_json(dims):
    return json.dumps(
        {
            "L": dims.L,
            "D": dims.D,
            "H": dims.H,
            "G": dims.G,
            "DH": dims.DH,
            "F": dims.F,
            "V": dims.V,
            "T": dims.T,
        },
        sort_keys=True,
    )


def _run_arguments(pod, out, report=None, skip=False):
    arguments = [
        "--checkpoint",
        str(pod["checkpoint_dir"]),
        "--out",
        str(out),
        "--calib",
        str(pod["calib_path"]),
        "--calib-sha256",
        pod["calib_sha"],
        "--heldout",
        str(pod["heldout_path"]),
        "--heldout-sha256",
        pod["heldout_sha"],
        "--test-dims",
        _dims_json(pod["dims"]),
        "--manifest",
        str(pod["manifest_path"]),
    ]
    if report is not None:
        arguments.extend(["--report", str(report)])
    if skip:
        arguments.append("--skip-range-report")
    return arguments


@pytest.fixture(scope="module")
def pod(tmp_path_factory):
    root = tmp_path_factory.mktemp("pod-range-report")
    checkpoint = synthetic_checkpoint.build_checkpoint(root / "checkpoint")
    dims = checkpoint["dims"]
    calib_dir = root / "calib"
    calib_dir.mkdir()
    calib_path = calib_dir / "fit.npy"
    fit, calib_sha = synthetic_checkpoint.write_windows(
        calib_path, dims, n_windows=5, seed=7
    )
    heldout_path = calib_dir / "heldout.npy"
    heldout, heldout_sha = synthetic_checkpoint.write_windows(
        heldout_path, dims, n_windows=6, seed=8
    )
    out = root / "artifacts"
    info = {
        "root": root,
        "checkpoint": checkpoint,
        "checkpoint_dir": checkpoint["dir"],
        "manifest_path": checkpoint["manifest_path"],
        "dims": dims,
        "calib_path": calib_path,
        "calib_sha": calib_sha,
        "fit": fit,
        "heldout_path": heldout_path,
        "heldout_sha": heldout_sha,
        "heldout": heldout,
    }
    assert run.main(_run_arguments(info, out, skip=True)) == 0
    artifact_dirs = [
        path
        for path in out.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ]
    assert len(artifact_dirs) == 1
    info["artifact_dir"] = artifact_dirs[0]
    info["out"] = out
    return info


def _without_seconds(report):
    report = deepcopy(report)
    for entry in report["per_window"]:
        entry.pop("seconds", None)
    return report


def _window_progress_count(stderr):
    return sum(
        1
        for line in stderr.splitlines()
        if line.startswith("range report: window ")
        and "range failures so far" in line
    )


def test_parallel_reports_match_sequential_and_print_progress(pod, capsys):
    windows = {"heldout": pod["heldout"]}
    sequential = range_report(
        pod["artifact_dir"],
        windows,
        dims=pod["dims"],
        workers=1,
        progress_interval=3600.0,
    )
    sequential_stderr = capsys.readouterr().err
    assert _window_progress_count(sequential_stderr) == len(windows["heldout"])

    methods = [
        method
        for method in ("fork", "spawn")
        if method in multiprocessing.get_all_start_methods()
    ]
    assert methods
    for method in methods:
        parallel = range_report(
            pod["artifact_dir"],
            windows,
            dims=pod["dims"],
            workers=3,
            progress_interval=3600.0,
            start_method=method,
        )
        parallel_stderr = capsys.readouterr().err
        assert _window_progress_count(parallel_stderr) == len(windows["heldout"])
        assert _without_seconds(parallel) == _without_seconds(sequential)


def test_layer_progress_interval_zero(pod, capsys):
    report = range_report(
        pod["artifact_dir"],
        {"heldout": pod["heldout"][:1]},
        dims=pod["dims"],
        workers=1,
        progress_interval=0.0,
    )
    stderr = capsys.readouterr().err
    layer_lines = [line for line in stderr.splitlines() if " layer " in line]
    assert len(layer_lines) == pod["dims"].L
    assert report["windows"] == 1


def test_parallel_needs_a_directory_for_an_artifact_object(pod):
    artifact = load_artifact(pod["artifact_dir"], dims=pod["dims"])
    with pytest.raises(CalibrationError):
        range_report(
            artifact,
            {"heldout": pod["heldout"][:1]},
            dims=pod["dims"],
            workers=2,
        )


def test_module_subprocess_workers_two(pod, tmp_path):
    report_path = tmp_path / "subprocess-report.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    command = [
        sys.executable,
        "-m",
        "tinyllama.convert.range_report",
        "--artifact",
        str(pod["artifact_dir"]),
        "--calib",
        str(pod["calib_path"]),
        "--heldout",
        str(pod["heldout_path"]),
        "--workers",
        "2",
        "--report",
        str(report_path),
        "--test-dims",
        _dims_json(pod["dims"]),
        "--progress-interval",
        "0",
    ]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="ascii"))
    assert REQUIRED_KEYS <= set(report)
    assert set(report) - REQUIRED_KEYS <= {"sets", "workers"}
    assert report["artifact_id"] == pod["artifact_dir"].name
    n_heldout = len(pod["heldout"])
    n_fit = len(pod["fit"])
    assert report["sets"] == ["heldout", "fit"]
    assert report["windows"] == n_heldout + n_fit
    assert report["range_failure_count"] == 0
    assert report["float_maxima"] is None
    assert report["float_over_limit"] == []
    assert [entry["set"] for entry in report["per_window"]] == [
        "heldout"
    ] * n_heldout + ["fit"] * n_fit
    assert [entry["index"] for entry in report["per_window"]] == list(
        range(n_heldout)
    ) + list(range(n_fit))
    assert [entry["window"] for entry in report["per_window"]] == list(
        range(n_heldout + n_fit)
    )
    assert _window_progress_count(result.stderr) == n_heldout + n_fit
    layer_lines = [
        line
        for line in result.stderr.splitlines()
        if line.startswith("range report: window ")
        and " layer " in line
        and "(worker pid " in line
    ]
    assert len(layer_lines) == (n_heldout + n_fit) * pod["dims"].L


def test_importing_these_modules_does_not_load_torch_or_transformers():
    code = (
        "import sys\n"
        "import tinyllama.convert.range_report\n"
        "import tinyllama.convert.run\n"
        "import tinyllama.convert.calibrate\n"
        "bad = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name == 'torch' or name.startswith('torch.')\n"
        "    or name == 'transformers' or name.startswith('transformers.')\n"
        ")\n"
        "print(','.join(bad))\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == ""


def test_sets_both_and_max_windows(pod, tmp_path):
    report_path = tmp_path / "both-report.json"
    code = range_report_module.main(
        [
            "--artifact",
            str(pod["artifact_dir"]),
            "--calib",
            str(pod["calib_path"]),
            "--heldout",
            str(pod["heldout_path"]),
            "--sets",
            "both",
            "--max-windows",
            "2",
            "--workers",
            "1",
            "--report",
            str(report_path),
            "--test-dims",
            _dims_json(pod["dims"]),
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="ascii"))
    assert report["windows"] == 4
    assert report["sets"] == ["heldout", "fit"]
    assert [entry["set"] for entry in report["per_window"]] == [
        "heldout",
        "heldout",
        "fit",
        "fit",
    ]
    assert [entry["index"] for entry in report["per_window"]] == [0, 1, 0, 1]
    assert [entry["window"] for entry in report["per_window"]] == [0, 1, 2, 3]


def test_sets_fit_needs_calib(pod, tmp_path):
    code = range_report_module.main(
        [
            "--artifact",
            str(pod["artifact_dir"]),
            "--heldout",
            str(pod["heldout_path"]),
            "--sets",
            "fit",
            "--workers",
            "1",
            "--report",
            str(tmp_path / "missing-calib.json"),
            "--test-dims",
            _dims_json(pod["dims"]),
        ]
    )
    assert code == 2


def test_run_skip_range_report(pod, tmp_path, capsys):
    out = tmp_path / "skip-artifacts"
    report_path = tmp_path / "must-not-be-written.json"
    code = run.main(_run_arguments(pod, out, report_path, skip=True))
    captured = capsys.readouterr()
    assert code == 0

    artifact_id = captured.out.strip().splitlines()[-1]
    artifact_dir = out / artifact_id
    assert artifact_dir.is_dir()
    assert validate_artifact(artifact_dir, dims=pod["dims"]) == []
    assert not report_path.exists()

    command_lines = [
        line
        for line in captured.err.splitlines()
        if "tinyllama.convert.range_report" in line
    ]
    assert command_lines
    command = command_lines[-1]
    assert "--artifact {}".format(artifact_dir) in command
    assert "--calib {}".format(pod["calib_path"]) in command
    assert "--heldout {}".format(pod["heldout_path"]) in command
    assert "--checkpoint {}".format(pod["checkpoint_dir"]) in command
    assert "--workers {}".format(min(8, os.cpu_count() or 1)) in command
    assert "--report {}".format(report_path) in command
    assert "--sets both" in command


def test_load_artifact_arrays_are_read_only_memmaps(pod):
    artifact = load_artifact(pod["artifact_dir"], dims=pod["dims"])
    arrays = (
        [artifact.emb, artifact.kl, artifact.wh]
        + list(artifact.matrices.values())
        + list(artifact.tables.values())
    )
    for array in arrays:
        assert isinstance(array, np.memmap)
        assert array.mode == "r"
        assert array.flags.writeable is False
