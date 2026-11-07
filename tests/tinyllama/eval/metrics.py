"""Quality metrics for the float32 and integer TinyLlama comparison (T3.3).

This module deliberately depends only on numpy.  It contains no torch,
transformers, tokenizers or tinyllama imports, so it can be imported and
tested on a machine that has only numpy and pytest.

All logits are converted to float64 before any arithmetic.  Integer logits
``lam`` are int64 values in units of 2**-s; ``int_logits_to_float`` converts
them exactly with ``np.ldexp``.

The argmax rule is the one of the arithmetic contract (section 10) and of
numpy: ties go to the lowest index.
"""

from typing import NamedTuple

import numpy as np


def _float64(values):
    """Return values as a C-contiguous float64 array."""
    return np.asarray(values, dtype=np.float64)


def log_softmax(logits, axis=-1):
    """Return the natural-log softmax, computed stably along ``axis``.

    The maximum is subtracted before exponentiation, so large logits (for
    example 1e4) do not overflow.
    """
    values = _float64(logits)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    return shifted - np.log(
        np.sum(np.exp(shifted), axis=axis, keepdims=True)
    )


def softmax(logits, axis=-1):
    """Return the softmax along ``axis`` (a convenience used by the tests)."""
    return np.exp(log_softmax(logits, axis=axis))


def top1_agreement(float_logits, int_logits):
    """Return the fraction of positions where the two argmaxes agree.

    ``float_logits`` and ``int_logits`` have the same shape.  The last axis is
    the vocabulary axis.  numpy's ``argmax`` returns the lowest index on a tie,
    which is the rule the integer contract also uses.
    """
    float_values = _float64(float_logits)
    int_values = _float64(int_logits)
    if float_values.shape != int_values.shape:
        raise ValueError(
            "logit shapes differ: %s versus %s"
            % (float_values.shape, int_values.shape)
        )
    if float_values.size == 0:
        raise ValueError("top1_agreement needs at least one position")
    equal = np.argmax(float_values, axis=-1) == np.argmax(int_values, axis=-1)
    return float(np.mean(equal))


def kl_divergence(p_logits, q_logits):
    """Return per-position KL(P || Q) in nats.

    ``P`` is the float baseline and ``Q`` is the integer model.  Both use the
    natural logarithm, so the result is in nats.  The last axis is the
    vocabulary axis; the returned array has the leading (position) shape.
    """
    p_values = _float64(p_logits)
    q_values = _float64(q_logits)
    if p_values.shape != q_values.shape:
        raise ValueError(
            "logit shapes differ: %s versus %s"
            % (p_values.shape, q_values.shape)
        )
    log_p = log_softmax(p_values, axis=-1)
    log_q = log_softmax(q_values, axis=-1)
    p = np.exp(log_p)
    return np.sum(p * (log_p - log_q), axis=-1)


def mean_kl(p_logits, q_logits, axis=None):
    """Return the mean of the per-position KL values (default: all)."""
    return float(np.mean(kl_divergence(p_logits, q_logits), axis=axis))


# Descriptive aliases, for callers that prefer the longer names.
kl_divergence_mean = mean_kl
mean_kl_divergence = mean_kl


class PerplexityResult(float):
    """A perplexity value that also carries its nll sum and token count.

    It behaves as a float (comparisons, arithmetic, ``float(result)``), and it
    can also be unpacked as ``ppl, nll_sum, count = result`` or read by
    attribute.  Aggregating windows must sum ``nll_sum`` and ``count`` and call
    ``perplexity_from_sums``; averaging perplexities is wrong.
    """

    def __new__(cls, perplexity, nll_sum, count):
        obj = float.__new__(cls, float(perplexity))
        obj.perplexity = float(perplexity)
        obj.ppl = float(perplexity)
        obj.nll_sum = float(nll_sum)
        obj.count = int(count)
        return obj

    def _parts(self):
        return (self.perplexity, self.nll_sum, self.count)

    def __iter__(self):
        return iter(self._parts())

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return self._parts()[index]


def nll_sum_and_count(logits, targets):
    """Return ``(nll_sum, token_count)`` for logits predicting targets.

    ``logits[t]`` predicts ``targets[t]``; the caller shifts the inputs.
    """
    values = _float64(logits)
    target_values = np.asarray(targets, dtype=np.int64)
    if values.ndim != target_values.ndim + 1:
        raise ValueError(
            "logits must have one more axis than targets, got %s and %s"
            % (values.shape, target_values.shape)
        )
    if values.shape[:-1] != target_values.shape:
        raise ValueError(
            "leading shapes differ: %s versus %s"
            % (values.shape[:-1], target_values.shape)
        )
    if target_values.size == 0:
        raise ValueError("perplexity needs at least one target")
    log_p = log_softmax(values, axis=-1)
    flat_log_p = log_p.reshape(-1, log_p.shape[-1])
    flat_targets = target_values.reshape(-1)
    positions = np.arange(flat_targets.size)
    nll = -flat_log_p[positions, flat_targets]
    return float(np.sum(nll)), int(flat_targets.size)


def nll_sum(logits, targets):
    """Return only the summed negative log likelihood (nats)."""
    return nll_sum_and_count(logits, targets)[0]


def perplexity_from_sums(nll_sum, count):
    """Return PerplexityResult from an already summed nll and token count."""
    count = int(count)
    if count <= 0:
        raise ValueError("perplexity needs a positive token count")
    nll_sum = float(nll_sum)
    return PerplexityResult(np.exp(nll_sum / count), nll_sum, count)


def perplexity(logits, targets):
    """Return perplexity, nll sum and token count.

    ``logits[t]`` predicts ``targets[t]`` (the caller shifts), and the result
    is ``exp(mean negative log likelihood)`` in nats.  The return value is a
    ``PerplexityResult``: a float that also unpacks to
    ``(perplexity, nll_sum, count)``.
    """
    return perplexity_from_sums(*nll_sum_and_count(logits, targets))


def int_logits_to_float(lam, s):
    """Convert int64 logits in units of 2**-s to float64.

    ``lam / 2**s`` is an exact power-of-two scaling, so it is computed with
    ``np.ldexp(lam, -s)``.  No rounding is introduced by this conversion.
    """
    values = _float64(lam)
    return np.ldexp(values, -int(s))


def result_range_failures(result):
    """Return the non-None range failures carried by one result.

    Supports an object with ``range_failures`` or ``range_failure`` and a
    mapping with the same keys.  This keeps ``count_range_failures`` useful
    for both the real reference Result and small test doubles.
    """
    if result is None:
        return []
    if isinstance(result, dict):
        if "range_failures" in result:
            return [item for item in result["range_failures"] if item is not None]
        failure = result.get("range_failure")
        return [] if failure is None else [failure]
    if hasattr(result, "range_failures"):
        return [
            item
            for item in getattr(result, "range_failures")
            if item is not None
        ]
    failure = getattr(result, "range_failure", None)
    return [] if failure is None else [failure]


def count_range_failures(results):
    """Count all non-None range failures among result objects."""
    return sum(len(result_range_failures(result)) for result in results)
