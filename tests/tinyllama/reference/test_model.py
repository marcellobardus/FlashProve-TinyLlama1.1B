import contextlib
import io
import json
import shutil
import numpy as np
import pytest

import tiny_artifact as ta
from tinyllama.artifacts import Dims, ArtifactRejected, load_artifact
from tinyllama.reference import (
    TraceWriter,
    forward,
    trace_digest,
)
from tinyllama.reference.model import (
    FINAL_RECORD_NAMES,
    LAYER_RECORD_NAMES,
    Result,
    first_argmax,
)
from tinyllama.reference.run import main as run_main
from tinyllama.reference.trace import canonical_json_bytes, int64_sha256, token_ids_sha256


_ARTIFACT_DIRECTORIES = {}


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-model")
    path = ta.build_artifact(root, dims=ta.DEFAULT_DIMS)
    loaded = load_artifact(path, dims=ta.DEFAULT_DIMS)
    _ARTIFACT_DIRECTORIES[loaded.id] = path
    return loaded


@pytest.fixture(scope="module")
def run_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny-run")
    path = ta.build_artifact(root, dims=ta.RUN_DIMS, seed=7)
    loaded = load_artifact(path, dims=ta.RUN_DIMS)
    _ARTIFACT_DIRECTORIES[loaded.id] = path
    return loaded


class Recorder:
    def __init__(self):
        self.items = []

    def __call__(self, name, layer, values):
        self.items.append((name, layer, np.array(values, dtype=np.int64, copy=True)))

    def by_key(self):
        return {(name, layer): values for name, layer, values in self.items}


def test_result_fields_and_dtypes(artifact):
    token_ids = [1, 2, 3]
    result = forward(artifact, token_ids)
    assert isinstance(result, Result)
    assert result.n == len(token_ids)
    assert result.range_failure is None
    assert result.next_token == first_argmax(result.lam_last)
    assert result.lam_last.dtype == np.int64
    assert result.lam_last.shape == (artifact.dims.V,)
    assert np.unique(result.lam_last).size > 1


def test_forward_is_deterministic(artifact):
    first = forward(artifact, [1, 2, 3])
    second = forward(artifact, [1, 2, 3])
    assert first.next_token == second.next_token
    assert int64_sha256(first.lam_last) == int64_sha256(second.lam_last)


def test_run_command_is_deterministic(tmp_path, run_artifact):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    dims_text = "1,8,2,1,4,6,32000,32"
    artifact_directory = str(_artifact_directory(run_artifact))
    with contextlib.redirect_stdout(io.StringIO()) as first_output:
        assert run_main(
            [
                "--artifact",
                artifact_directory,
                "--prompt",
                "P0",
                "--trace",
                str(first_path),
                "--dims-test",
                dims_text,
            ]
        ) == 0
    with contextlib.redirect_stdout(io.StringIO()):
        assert run_main(
            [
                "--artifact",
                artifact_directory,
                "--prompt",
                "P0",
                "--trace",
                str(second_path),
                "--dims-test",
                dims_text,
            ]
        ) == 0
    first = trace_digest(first_path)
    second = trace_digest(second_path)
    assert first["lam_last_sha256"] == second["lam_last_sha256"]
    assert first["next_token"] == second["next_token"]
    assert "lam_last_sha256: " + first["lam_last_sha256"] in first_output.getvalue()
    assert first["prompt_id"] == "P0"
    assert first["mode"] == "prefix"


def _artifact_directory(artifact):
    # load_artifact does not keep the directory; the test module records it in
    # the fixture below instead of guessing from the id.
    return _ARTIFACT_DIRECTORIES[artifact.id]


def test_prefix_and_window_agree_at_real_positions(artifact):
    token_ids = [1, 2, 3]
    prefix_records = Recorder()
    window_records = Recorder()
    prefix = forward(artifact, token_ids, mode="prefix", record=prefix_records)
    window = forward(artifact, token_ids, mode="window", record=window_records)

    assert prefix.range_failure is None
    assert window.range_failure is None
    assert prefix.next_token == window.next_token
    assert np.array_equal(prefix.lam_last, window.lam_last)

    prefix_map = prefix_records.by_key()
    window_map = window_records.by_key()
    assert set(prefix_map) == set(window_map)
    n = len(token_ids)
    for key, values in prefix_map.items():
        assert np.array_equal(values, window_map[key][:n]), key
    assert prefix_map[("first_choice", None)].tolist() == window_map[
        ("first_choice", None)
    ][:n].tolist()


def test_first_argmax_tie_rule():
    assert first_argmax([5, 9, 9, -1]) == 1
    assert first_argmax([7, 7, 7]) == 0
    assert first_argmax([-3, -3, -4]) == 0
    assert first_argmax([-10, -2, -2]) == 1


def test_model_first_choice_uses_lowest_id_on_ties(tmp_path):
    dims = Dims(L=1, D=8, H=2, G=1, DH=4, F=6, V=32, T=8)
    zero_wh = np.zeros((dims.V, dims.D), dtype=np.int16)
    path = ta.build_artifact(
        tmp_path, dims=dims, wh_values=zero_wh, seed=11
    )
    tiny = load_artifact(path, dims=dims)
    records = Recorder()
    result = forward(tiny, [1, 2], record=records)
    assert result.range_failure is None
    assert result.next_token == 0
    assert records.by_key()[("first_choice", None)].tolist() == [0, 0]
    for row in records.by_key()[("lam", None)]:
        assert first_argmax(row) == 0


def _two_failure_artifact(tmp_path):
    dims = Dims(L=2, D=8, H=2, G=1, DH=4, F=6, V=32, T=8)
    row = [1, 1, 1, 1, 8, 8, 8, 37, 32, 37, 32, 8, 1]
    path = ta.build_artifact(
        tmp_path, dims=dims, kl_rows=[row, row], seed=4
    )
    return load_artifact(path, dims=dims)


def test_range_failure_reports_the_earlier_condition(tmp_path):
    tiny = _two_failure_artifact(tmp_path)
    result = forward(tiny, [1, 2])
    failure = result.range_failure
    assert failure is not None
    assert result.lam_last is None
    assert result.next_token is None
    assert failure.condition == 3
    assert failure.name == "qw"
    assert failure.layer == 0
    assert (failure.position, failure.index) == (0, 0)


def _final_norm_failure_artifact(tmp_path):
    dims = Dims(L=1, D=8, H=2, G=1, DH=4, F=6, V=32, T=8)

    def matrices(layer, name, shape, rng):
        if name in ("wq", "wk", "wv", "wo"):
            return np.zeros(shape, dtype=np.int16)
        if name == "wg":
            return np.full(shape, 4095, dtype=np.int16)
        if name == "wu":
            return np.full(shape, 341, dtype=np.int16)
        if name == "wd":
            array = np.zeros(shape, dtype=np.int16)
            array[0, :] = 4095
            return array
        raise AssertionError(name)

    row = [8, 8, 8, 8, 1, 1, 1, 37, 32, 37, 32, 8, 1]
    emb = np.full((dims.V, dims.D), 100, dtype=np.int32)
    path = ta.build_artifact(
        tmp_path,
        dims=dims,
        kl_rows=[row],
        matrix_values=matrices,
        emb_values=emb,
        seed=3,
    )
    return load_artifact(path, dims=dims), dims


def test_final_norm_failure_is_reported_after_all_layers_pass(tmp_path):
    tiny, dims = _final_norm_failure_artifact(tmp_path)
    records = Recorder()
    result = forward(tiny, [1, 2], record=records)
    failure = result.range_failure
    assert failure is not None
    # Every layer record was produced; the failure is the final norm's.
    names = [name for name, _, _ in records.items]
    assert names[: len(LAYER_RECORD_NAMES)] == list(LAYER_RECORD_NAMES)
    assert "x_last" in names
    assert "aF_sigma" not in names
    assert failure.condition == 8
    assert failure.name == "x_last"
    assert failure.layer == dims.L
    assert failure.index == 0


def test_malformed_artifact_is_rejected_before_forward(artifact, tmp_path):
    source = _artifact_directory(artifact)
    table_copy = tmp_path / "table-broken"
    shutil.copytree(source, table_copy)
    table = table_copy / "tables" / "rope_c.i32"
    raw = bytearray(table.read_bytes())
    raw[0] ^= 1
    table.write_bytes(bytes(raw))
    with pytest.raises(ArtifactRejected):
        load_artifact(table_copy, dims=ta.DEFAULT_DIMS)

    manifest_copy = tmp_path / "manifest-broken"
    shutil.copytree(source, manifest_copy)
    (manifest_copy / "manifest.json").write_bytes(b"{")
    with pytest.raises(ArtifactRejected):
        load_artifact(manifest_copy, dims=ta.DEFAULT_DIMS)


def test_bad_token_ids_raise_value_error(artifact):
    with pytest.raises(ValueError):
        forward(artifact, [])
    with pytest.raises(ValueError):
        forward(artifact, [0] * (artifact.dims.T + 1))
    with pytest.raises(ValueError):
        forward(artifact, [artifact.dims.V])
    with pytest.raises(ValueError):
        forward(artifact, [-1])
    with pytest.raises(ValueError):
        forward(artifact, [0], mode="nope")


def test_trace_schema_order_and_canonical_json(artifact, tmp_path):
    token_ids = [1, 2, 3]
    result = forward(artifact, token_ids)
    writer = TraceWriter(tmp_path / "P0.prefix.json", full=True)
    result = forward(artifact, token_ids, record=writer)
    trace = writer.finish(
        artifact, prompt_id="P0", mode="prefix", token_ids=token_ids, result=result
    )

    assert set(trace) == {
        "contract",
        "artifact_id",
        "prompt_id",
        "mode",
        "n",
        "token_ids_sha256_u32le",
        "records",
        "range_failure",
        "next_token",
        "lam_last_sha256",
        "top16",
    }
    assert trace["contract"] == "TL-INT-1"
    assert trace["artifact_id"] == artifact.id
    assert trace["n"] == len(token_ids)
    assert trace["token_ids_sha256_u32le"] == token_ids_sha256(token_ids)
    assert trace["range_failure"] is None
    assert trace["next_token"] == result.next_token
    assert trace["lam_last_sha256"] == int64_sha256(result.lam_last)

    expected_names = []
    for layer in range(artifact.dims.L):
        expected_names.extend(LAYER_RECORD_NAMES)
    expected_names.extend(FINAL_RECORD_NAMES)
    assert [entry["name"] for entry in trace["records"]] == expected_names
    # I3.1 has seven names after the layers, so the count is 21 L + 7.
    assert len(trace["records"]) == 21 * artifact.dims.L + 7

    for entry in trace["records"]:
        assert set(entry) == {"name", "layer", "shape", "sha256", "min", "max"}
        assert isinstance(entry["shape"], list)
        assert all(isinstance(size, int) for size in entry["shape"])
        assert len(entry["sha256"]) == 64
        assert isinstance(entry["min"], int)
        assert isinstance(entry["max"], int)
        if entry["name"] in ("a_sigma", "a_Rc", "m_sigma", "m_Rc"):
            assert entry["shape"] == [len(token_ids)]
        assert entry["layer"] is None or 0 <= entry["layer"] < artifact.dims.L

    top = trace["top16"]
    assert len(top) == min(16, artifact.dims.V)
    ordered = sorted(top, key=lambda pair: (-pair[1], pair[0]))
    assert top == ordered
    assert all(isinstance(pair[0], int) and isinstance(pair[1], int) for pair in top)

    raw = (tmp_path / "P0.prefix.json").read_bytes()
    assert raw == canonical_json_bytes(json.loads(raw))
    assert trace_digest(tmp_path / "P0.prefix.json") == trace

    full_dir = tmp_path / "P0.prefix"
    assert (full_dir / "x.00.npy").is_file()
    assert (full_dir / "x_last.F.npy").is_file()
    assert (full_dir / "lam.F.npy").is_file()
    saved = np.load(full_dir / "lam.F.npy")
    assert saved.dtype == np.int64
    assert saved.shape == (len(token_ids), artifact.dims.V)
