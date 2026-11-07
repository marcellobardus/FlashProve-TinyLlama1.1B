"""Tests for the float-versus-integer quality harness (T3.3).

The runners here are small synthetic doubles.  No torch, transformers or real
model is used, and no expected value is taken from tinyllama.* output.
"""

import json
import math
import sys

import numpy as np
import pytest

import metrics
import quality_eval

SCALE = 8
VOCABULARY = 5


class FakeFloatRunner:
    def __init__(self, logits_by_prompt, revision="test-revision"):
        self.logits_by_prompt = logits_by_prompt
        self.revision = revision

    def logits(self, token_ids):
        key = tuple(int(value) for value in token_ids)
        return np.asarray(self.logits_by_prompt[key], dtype=np.float32)


class FakeIntegerRunner:
    def __init__(self, lam_by_prompt, failures=None, artifact_id="test-artifact"):
        self.lam_by_prompt = lam_by_prompt
        self.failures = dict(failures or {})
        self.s = SCALE
        self.artifact_id = artifact_id

    def run(self, token_ids):
        key = tuple(int(value) for value in token_ids)
        lam = np.asarray(self.lam_by_prompt[key], dtype=np.int64)
        failure = self.failures.get(key)
        failures = [] if failure is None else [failure]
        return quality_eval.IntegerOutput(
            lam=lam,
            next_token=int(np.argmax(lam[-1])),
            range_failures=failures,
        )


def rounded_lam(float_logits):
    return np.rint(np.asarray(float_logits, dtype=np.float64) * 2**SCALE).astype(
        np.int64
    )


def independent_log_softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def independent_kl(p_logits, q_logits):
    log_p = independent_log_softmax(p_logits)
    log_q = independent_log_softmax(q_logits)
    return np.sum(np.exp(log_p) * (log_p - log_q), axis=-1)


def independent_ppl(logits, targets):
    log_p = independent_log_softmax(logits)
    targets = np.asarray(targets, dtype=np.int64)
    nll = -log_p[np.arange(targets.size), targets]
    return math.exp(float(np.sum(nll)) / targets.size)


def make_case():
    prompt_a_ids = [101, 102, 103]
    prompt_b_ids = [201, 202]
    # Token ids double as the next-token targets, so keep them inside the
    # synthetic vocabulary.
    window_a_ids = [0, 1, 2, 3]
    window_b_ids = [4, 3, 2]

    float_map = {
        tuple(prompt_a_ids): [
            [0.1, -0.2, 0.3, -0.4, 0.5],
            [1.0, 0.0, -1.0, 0.5, 0.25],
            [-0.5, 0.2, 0.1, 0.9, -0.3],
        ],
        tuple(prompt_b_ids): [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.4, 0.3, 0.2],
        ],
        tuple(window_a_ids): [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.5, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ],
        tuple(window_b_ids): [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
    }
    # The float runner returns float32 logits, so round those same values.
    float_map = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in float_map.items()
    }
    int_map = {
        key: rounded_lam(value) for key, value in float_map.items()
    }

    failure = {
        "condition": 7,
        "name": "u",
        "layer": 3,
        "position": 1,
        "index": 2,
        "value": 123,
    }
    float_runner = FakeFloatRunner(float_map)
    int_runner = FakeIntegerRunner(
        int_map, failures={tuple(prompt_b_ids): failure}
    )
    prompts = [
        ("PA", prompt_a_ids),
        ("PB", prompt_b_ids),
    ]
    windows = [
        np.asarray(window_a_ids, dtype=np.int32),
        np.asarray(window_b_ids, dtype=np.int32),
    ]
    return (
        float_runner,
        int_runner,
        prompts,
        windows,
        float_map,
        int_map,
        failure,
    )


def test_evaluate_fields_and_independent_metrics():
    (
        float_runner,
        int_runner,
        prompts,
        windows,
        float_map,
        int_map,
        failure,
    ) = make_case()

    result = quality_eval.evaluate(float_runner, int_runner, prompts, windows)

    assert result["contract"] == "TL-INT-1"
    assert result["artifact_id"] == "test-artifact"
    assert result["revision"] == "test-revision"
    assert result["s"] == SCALE
    assert result["metadata"]["counts"]["prompts"] == 2
    assert result["metadata"]["counts"]["heldout_windows"] == 2
    assert result["range_failure_count"] == 1
    assert result["range_failures"] == [failure]

    assert [row["id"] for row in result["prompts"]] == ["PA", "PB"]

    for row, (prompt_id, token_ids) in zip(result["prompts"], prompts):
        key = tuple(token_ids)
        float_logits = np.asarray(float_map[key], dtype=np.float64)
        int_logits = np.ldexp(
            np.asarray(int_map[key], dtype=np.float64), -SCALE
        )
        agreement = np.argmax(float_logits, axis=-1) == np.argmax(
            int_logits, axis=-1
        )
        kl = independent_kl(float_logits, int_logits)
        assert row["n"] == len(token_ids)
        assert row["top1_agreement"] == pytest.approx(
            float(np.mean(agreement)), rel=0.0, abs=1e-15
        )
        assert row["top1"] == pytest.approx(
            float(np.mean(agreement)), rel=0.0, abs=1e-15
        )
        assert row["mean_kl"] == pytest.approx(
            float(np.mean(kl)), rel=1e-12, abs=1e-15
        )
        assert row["last_position_agreement"] == bool(agreement[-1])
        assert row["next_token_float"] == int(
            np.argmax(float_logits[-1])
        )
        assert row["next_token_int"] == int(np.argmax(int_logits[-1]))
        expected_failures = 1 if prompt_id == "PB" else 0
        assert row["range_failure_count"] == expected_failures

    assert result["prompts"][1]["range_failures"] == [failure]

    all_float = []
    all_int = []
    nll_float = 0.0
    nll_int = 0.0
    target_count = 0
    for token_ids in windows:
        key = tuple(int(value) for value in token_ids)
        float_logits = np.asarray(float_map[key], dtype=np.float64)
        int_logits = np.ldexp(
            np.asarray(int_map[key], dtype=np.float64), -SCALE
        )
        all_float.append(float_logits)
        all_int.append(int_logits)
        targets = np.asarray(key[1:], dtype=np.int64)
        float_log_p = independent_log_softmax(float_logits[:-1])
        int_log_p = independent_log_softmax(int_logits[:-1])
        positions = np.arange(targets.size)
        nll_float += float(np.sum(-float_log_p[positions, targets]))
        nll_int += float(np.sum(-int_log_p[positions, targets]))
        target_count += int(targets.size)

    concat_float = np.concatenate(all_float, axis=0)
    concat_int = np.concatenate(all_int, axis=0)
    expected_top1 = float(
        np.mean(
            np.argmax(concat_float, axis=-1)
            == np.argmax(concat_int, axis=-1)
        )
    )
    expected_kl = float(np.mean(independent_kl(concat_float, concat_int)))
    expected_ppl_float = math.exp(nll_float / target_count)
    expected_ppl_int = math.exp(nll_int / target_count)

    heldout = result["heldout"]
    assert heldout["windows"] == 2
    assert heldout["positions"] == 7
    assert heldout["tokens"] == 5
    assert heldout["top1_agreement"] == pytest.approx(
        expected_top1, rel=0.0, abs=1e-15
    )
    assert heldout["mean_kl"] == pytest.approx(
        expected_kl, rel=1e-12, abs=1e-15
    )
    assert heldout["ppl_float"] == pytest.approx(
        expected_ppl_float, rel=1e-12
    )
    assert heldout["ppl_int"] == pytest.approx(
        expected_ppl_int, rel=1e-12
    )
    assert heldout["range_failure_count"] == 0


def test_write_json_and_markdown(tmp_path):
    (
        float_runner,
        int_runner,
        prompts,
        windows,
        _float_map,
        _int_map,
        _failure,
    ) = make_case()
    result = quality_eval.evaluate(float_runner, int_runner, prompts, windows)

    json_path = tmp_path / "quality.json"
    markdown_path = tmp_path / "quality.md"
    quality_eval.write_json(result, json_path)
    quality_eval.write_markdown(result, markdown_path)

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["contract"] == "TL-INT-1"
    assert [row["id"] for row in loaded["prompts"]] == ["PA", "PB"]
    assert loaded["range_failure_count"] == 1
    assert loaded["heldout"]["windows"] == 2
    json_text = json_path.read_text(encoding="utf-8")
    assert json_text.index('"artifact_id"') < json_text.index('"contract"')

    markdown = markdown_path.read_text(encoding="utf-8")
    assert (
        "| Row | Top-1 agreement | Mean KL | PPL float | PPL int | "
        "Range failures |"
    ) in markdown
    assert "| --- |" in markdown
    assert "| PA |" in markdown
    assert "| PB |" in markdown
    assert "| held-out aggregate |" in markdown


def test_integer_runner_raises_reference_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "tinyllama.reference", None)
    with pytest.raises(quality_eval.ReferenceUnavailable):
        quality_eval.IntegerRunner("artifact-path-does-not-matter")


def test_main_returns_3_when_reference_is_missing(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "tinyllama.reference", None)
    out_json = tmp_path / "quality.json"
    out_md = tmp_path / "quality.md"

    code = quality_eval.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--artifact",
            str(tmp_path / "artifact"),
            "--heldout",
            str(tmp_path / "heldout.npy"),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )

    assert code == 3
    captured = capsys.readouterr()
    assert "tinyllama.reference" in captured.err
    assert "SKIP" in captured.err
    assert not out_json.exists()
    assert not out_md.exists()


def test_load_prompts_reads_real_prompt_set():
    prompts = quality_eval.load_prompts()
    assert [prompt_id for prompt_id, _ids in prompts] == [
        "P0",
        "P1",
        "P2",
        "P3",
    ]
    assert [len(ids) for _prompt_id, ids in prompts] == [24, 37, 57, 25]
    assert prompts[0][1][:3] == [1, 529, 29989]


def test_load_heldout_reads_int32_windows(tmp_path):
    path = tmp_path / "heldout.npy"
    array = np.arange(8 * 1024, dtype=np.int32).reshape(8, 1024)
    np.save(path, array)
    loaded = quality_eval.load_heldout(path)
    assert loaded.shape == (8, 1024)
    assert loaded.dtype == np.int32
    np.testing.assert_array_equal(loaded, array)
