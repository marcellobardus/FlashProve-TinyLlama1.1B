"""Hand-worked tests for the pure numpy quality metrics (T3.3)."""

import math

import numpy as np
import pytest

import metrics


def test_kl_of_identical_logits_is_zero():
    rng = np.random.default_rng(1234)
    logits = rng.normal(size=(6, 9))
    kl = metrics.kl_divergence(logits, logits)
    np.testing.assert_allclose(kl, 0.0, atol=1e-12, rtol=0.0)
    assert metrics.mean_kl(logits, logits) == pytest.approx(0.0, abs=1e-12)


def test_kl_hand_worked_two_class_case():
    p_logits = np.array([[0.0, 0.0]])
    q_logits = np.array([[math.log(0.25), math.log(0.75)]])
    expected = (
        0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    )
    kl = metrics.kl_divergence(p_logits, q_logits)
    assert kl.shape == (1,)
    assert kl[0] == pytest.approx(expected, rel=1e-15, abs=0.0)
    assert metrics.mean_kl(p_logits, q_logits) == pytest.approx(
        expected, rel=1e-15, abs=0.0
    )


def test_kl_is_invariant_to_adding_a_constant_to_logits():
    rng = np.random.default_rng(7)
    p_logits = rng.normal(size=(4, 5))
    q_logits = rng.normal(size=(4, 5))
    base = metrics.kl_divergence(p_logits, q_logits)
    shifted = metrics.kl_divergence(
        p_logits + 1234.5, q_logits - 987.25
    )
    np.testing.assert_allclose(shifted, base, rtol=1e-12, atol=1e-12)


def test_kl_is_non_negative_on_random_arrays():
    rng = np.random.default_rng(99)
    for _ in range(20):
        p_logits = rng.normal(size=(3, 11)) * 5.0
        q_logits = rng.normal(size=(3, 11)) * 5.0
        kl = metrics.kl_divergence(p_logits, q_logits)
        assert np.all(kl >= -1e-12)
        assert metrics.mean_kl(p_logits, q_logits) >= -1e-12


def test_perplexity_of_uniform_logits_is_vocabulary_size():
    vocabulary = 17
    logits = np.zeros((5, vocabulary))
    targets = np.array([0, 3, 11, 16, 2])
    result = metrics.perplexity(logits, targets)
    assert float(result) == pytest.approx(vocabulary, abs=1e-9)
    assert result.perplexity == pytest.approx(vocabulary, abs=1e-9)
    assert result.count == 5
    assert result.nll_sum == pytest.approx(5 * math.log(vocabulary))
    ppl, nll_sum, count = result
    assert ppl == pytest.approx(vocabulary, abs=1e-9)
    assert nll_sum == pytest.approx(5 * math.log(vocabulary))
    assert count == 5


def test_perplexity_hand_worked_two_positions():
    logits = np.array(
        [
            [math.log(0.5), math.log(0.5)],
            [math.log(0.25), math.log(0.75)],
        ]
    )
    targets = np.array([0, 0])
    expected = math.exp((math.log(2.0) + math.log(4.0)) / 2.0)
    assert expected == pytest.approx(math.sqrt(8.0))
    assert float(metrics.perplexity(logits, targets)) == pytest.approx(
        math.sqrt(8.0), rel=1e-15, abs=0.0
    )


def test_top1_agreement_hand_worked_and_ties_choose_lowest_index():
    float_logits = np.array(
        [
            [0.0, 1.0, 1.0],
            [3.0, 2.0, 1.0],
            [1.0, 5.0, 2.0],
        ]
    )
    int_logits = np.array(
        [
            [1.0, 1.0, 0.0],
            [4.0, 2.0, 1.0],
            [1.0, 4.0, 2.0],
        ]
    )
    assert metrics.top1_agreement(float_logits, int_logits) == pytest.approx(
        2.0 / 3.0
    )

    # numpy argmax, like the contract, chooses index 0 for both ties.
    tie_float = np.array([[1.0, 1.0, 0.0]])
    tie_int = np.array([[1.0, 1.0, 0.0]])
    assert np.argmax(tie_float[0]) == 0
    assert np.argmax(tie_int[0]) == 0
    assert metrics.top1_agreement(tie_float, tie_int) == 1.0


def test_int_logits_to_float_hand_worked_and_exact():
    lam = np.array(
        [[3 * 2**10, -3 * 2**10], [0, -2**10]], dtype=np.int64
    )
    converted = metrics.int_logits_to_float(lam, 10)
    expected = np.array([[3.0, -3.0], [0.0, -1.0]])
    np.testing.assert_array_equal(converted, expected)
    assert converted.dtype == np.float64


def test_log_softmax_on_large_logits_is_finite():
    logits = np.array([[1e4, 0.0, -1e4], [2e4, 2e4, 2e4]])
    log_p = metrics.log_softmax(logits)
    assert np.all(np.isfinite(log_p))
    np.testing.assert_allclose(
        np.sum(np.exp(log_p), axis=-1), np.ones(2), rtol=0.0, atol=1e-12
    )


def test_perplexity_aggregation_over_two_windows_equals_concatenation():
    rng = np.random.default_rng(2024)
    logits_a = rng.normal(size=(3, 6))
    logits_b = rng.normal(size=(4, 6))
    targets_a = rng.integers(0, 6, size=3)
    targets_b = rng.integers(0, 6, size=4)

    result_a = metrics.perplexity(logits_a, targets_a)
    result_b = metrics.perplexity(logits_b, targets_b)
    aggregated = math.exp(
        (result_a.nll_sum + result_b.nll_sum)
        / (result_a.count + result_b.count)
    )
    concatenated = metrics.perplexity(
        np.concatenate([logits_a, logits_b], axis=0),
        np.concatenate([targets_a, targets_b]),
    )
    assert float(concatenated) == pytest.approx(aggregated, rel=1e-15)
    assert concatenated.count == result_a.count + result_b.count


def test_count_range_failures_counts_only_non_none_values():
    class Result:
        def __init__(self, failure=None, failures=None):
            if failures is not None:
                self.range_failures = failures
            self.range_failure = failure

    results = [
        Result(failure=None),
        Result(failure={"condition": 1}),
        {"range_failure": None},
        {"range_failure": {"condition": 2}},
        {"range_failures": [None, {"condition": 3}]},
        None,
    ]
    assert metrics.count_range_failures(results) == 3
