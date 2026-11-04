import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import synthetic_checkpoint

from tinyllama.artifacts import load_artifact, validate_artifact
from tinyllama.convert import run


@pytest.fixture
def pipeline(tmp_path):
    checkpoint = synthetic_checkpoint.build_checkpoint(tmp_path / "model")
    dims = checkpoint["dims"]
    calib_path = tmp_path / "calib" / "fit.npy"
    calib_path.parent.mkdir(parents=True)
    _, calib_sha = synthetic_checkpoint.write_windows(
        calib_path, dims, n_windows=2, seed=7
    )
    heldout_path = tmp_path / "calib" / "heldout.npy"
    _, heldout_sha = synthetic_checkpoint.write_windows(
        heldout_path, dims, n_windows=1, seed=8
    )
    return {
        "checkpoint": checkpoint,
        "dims": dims,
        "calib": calib_path,
        "calib_sha": calib_sha,
        "heldout": heldout_path,
        "heldout_sha": heldout_sha,
    }


def _arguments(pipeline, out, report=None):
    arguments = [
        "--checkpoint",
        str(pipeline["checkpoint"]["dir"]),
        "--out",
        str(out),
        "--calib",
        str(pipeline["calib"]),
        "--calib-sha256",
        pipeline["calib_sha"],
        "--heldout",
        str(pipeline["heldout"]),
        "--heldout-sha256",
        pipeline["heldout_sha"],
        "--test-dims",
        json.dumps(
            {
                "L": pipeline["dims"].L,
                "D": pipeline["dims"].D,
                "H": pipeline["dims"].H,
                "G": pipeline["dims"].G,
                "DH": pipeline["dims"].DH,
                "F": pipeline["dims"].F,
                "V": pipeline["dims"].V,
                "T": pipeline["dims"].T,
            },
            sort_keys=True,
        ),
        "--manifest",
        str(pipeline["checkpoint"]["manifest_path"]),
    ]
    if report is not None:
        arguments.extend(["--report", str(report)])
    return arguments


def test_pipeline_end_to_end_and_deterministic(pipeline, tmp_path, capsys):
    out_one = tmp_path / "out-one"
    out_two = tmp_path / "out-two"
    report_path = tmp_path / "report.json"

    assert run.main(_arguments(pipeline, out_one, report_path)) == 0
    first_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert run.main(_arguments(pipeline, out_two)) == 0
    second_line = capsys.readouterr().out.strip().splitlines()[-1]

    assert first_line == second_line
    first_dir = Path(out_one) / first_line
    second_dir = Path(out_two) / second_line
    assert first_dir.is_dir()
    assert second_dir.is_dir()
    assert validate_artifact(first_dir, dims=pipeline["dims"]) == []
    assert validate_artifact(second_dir, dims=pipeline["dims"]) == []

    artifact = load_artifact(
        first_dir, expected_id=first_line, dims=pipeline["dims"]
    )
    assert artifact.id == first_line
    assert artifact.kl.shape == (pipeline["dims"].L, 13)
    assert artifact.params["contract"] == "TL-INT-1"
    assert artifact.params["e_res"] == 11

    report = json.loads(report_path.read_text(encoding="ascii"))
    assert report["contract"] == "TL-INT-1"
    assert report["windows"] == 1
    assert report["range_failure_count"] == 0
    assert report["integer_reference"] == "ran"
    assert report["float_maxima"] is not None


def test_pipeline_refuses_a_bad_calibration_digest(pipeline, tmp_path):
    arguments = _arguments(pipeline, tmp_path / "out")
    index = arguments.index("--calib-sha256") + 1
    arguments[index] = "0" * 64
    assert run.main(arguments) == 2


def test_output_location_rule(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    run._check_output_location(repo / "artifacts" / "x", repo_root=repo)
    run._check_output_location(tmp_path / "elsewhere" / "x", repo_root=repo)
    with pytest.raises(run.RunError):
        run._check_output_location(repo / "not-artifacts" / "x", repo_root=repo)
    with pytest.raises(run.RunError):
        run._check_output_location(repo, repo_root=repo)


def test_test_dims_parser_refuses_inconsistent_shapes():
    with pytest.raises(run.RunError):
        run._parse_test_dims(
            '{"L":1,"D":64,"H":4,"G":2,"DH":8,"F":128,"V":256,"T":8}'
        )
    with pytest.raises(run.RunError):
        run._parse_test_dims('{"L":1}')
