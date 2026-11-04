"""CPU integer reference for the TinyLlama TL-INT-1 arithmetic contract."""

from .primitives import (
    argmax,
    clamp,
    exp_lookup,
    floor_div,
    gate_key,
    isqrt,
    low8,
    norm,
    pad,
    rq,
    softmax_div,
)
from .model import Result, forward
from .trace import TraceWriter, trace_digest

__all__ = [
    "rq",
    "clamp",
    "isqrt",
    "floor_div",
    "low8",
    "softmax_div",
    "exp_lookup",
    "argmax",
    "gate_key",
    "norm",
    "pad",
    "Result",
    "forward",
    "TraceWriter",
    "trace_digest",
]
