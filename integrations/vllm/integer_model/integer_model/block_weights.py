# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Adapted from gguf-py 0.19.0, the Python package gguf of the llama.cpp project: decoded_values
# (from the classes Q4_K, Q5_K, Q6_K, Q8_0 and BF16 of gguf/quants.py, and from its dequantize for
# F16), the numbers that the GGUF container fixes (gguf/constants.py) and the order of the header's
# fields (gguf/gguf_reader.py). Its licence:
#
# MIT License
#
# Copyright (c) 2023 Georgi Gerganov
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Four expressions of this file are those of ggml (llama.cpp), which is under the MIT License.
# That project's notice:
#
# MIT License
#
# Copyright (c) 2023-2024 The ggml authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""An integer model's weights, decoded from quantized blocks; made for full size (400 billion
parameters), where most weights stay as blocks so that they fit in memory.

A real model's GGUF files hold quantized blocks, not integers: a conversion rule defines the
committed integers from them (docs/model.md, "11 From the model's files to the integers"). The GPU
prover's code for the rule (`fold` in names) is prover/weights_blocks/fold_rule.h; the prover
applies it at set-up, and again at each use for the tensors that it keeps as blocks. This module
applies the same rule in PyTorch and is meant to give the same integers, bit for bit;
committed_integers states how far that is tested. The comments' paths NOTICE, OPERATING.md and
tests/ start at integrations/vllm/integer_model/.

- decoded_values and committed_integers are the conversion rule.
- GgufBlocks reads a model's GGUF files; SyntheticBlocks computes the prover's synthetic model.
- BlockWeights serves the integers as a weight source (WeightSource in weights.py).
- main(), run as python -m integer_model.block_weights, compares the SHA-256 of each tensor's
  integers with the checksum file, wsums.txt; for the synthetic model it prints one overall SHA-256.
"""

import argparse
import hashlib
import logging
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch

from .parameters import ParameterError, Parameters, given_digests, read_bytes
from .synthetic_weights import mix64, shift_right64, signed64, stream_entries
from .weights import NotEnoughMemory, WeightsRejected

logger = logging.getLogger(__name__)

if sys.byteorder != "little":
    raise ImportError(
        "integer_model.block_weights reinterprets bytes as numbers and needs a little-endian "
        "machine."
    )


# The conversion rule

# The seven storage types that the rule reads, under their numbers in a GGUF file's tensor table.
# Source of the numbers of the three wider types: gguf/constants.py, GGMLQuantizationType (the top
# of this file names the package).
F32, Q4_K, Q5_K, Q6_K = 0, 12, 13, 14
F16, Q8_0, BF16 = 1, 8, 30
# A block holds 256 consecutive elements of a row. These are its sizes in bytes. The format's own
# block of Q8_0 has 32 elements in 34 bytes, and an element of F16 or BF16 has 2 bytes
# (gguf/constants.py, GGML_QUANT_SIZES). Here a block is eight Q8_0 blocks in the file's order (272
# bytes), or 256 elements of F16 or BF16 (512 bytes), as in fr::block_bytes of the prover's
# fold_rule.h.
BLOCK_ELEMENTS = 256
BLOCK_BYTES = {F32: 1024, Q4_K: 144, Q5_K: 176, Q6_K: 210, Q8_0: 8 * 34, F16: 512, BF16: 512}

MATRIX_LIMIT = 4095  # the largest magnitude of a matrix weight
EMBEDDING_LIMIT = 2**30 - 1  # the largest magnitude of an entry of the embedding


def _binary16(two_bytes: torch.Tensor) -> torch.Tensor:
    """Reads little-endian float16 numbers, shape (.., 2) of bytes, as float32, shape (.., 1).
    Exact."""
    return two_bytes.contiguous().view(torch.float16).to(torch.float32)


def decoded_values(blocks: torch.Tensor, storage_type: int) -> torch.Tensor:
    """The decoded values of docs/model.md, "11.2 Stored blocks and their values": from n blocks,
    shape (n, bytes of a block), to float32, shape (n, 256).

    Sources (licence texts: NOTICE). The GGUF container, the byte layout of the block formats Q4_K,
    Q5_K and Q6_K, and the rule that decodes them are those of ggml, the tensor library of the
    llama.cpp project (github.com/ggml-org/llama.cpp, MIT licence). This function is adapted from
    gguf-py 0.19.0, that project's Python package gguf: the classes Q4_K, Q5_K and Q6_K of
    gguf/quants.py (MIT licence, "Copyright (c) 2023 Georgi Gerganov"), which decode whole arrays of
    blocks with numpy as this function does with PyTorch. The two lines that unpack a Q4_K or Q5_K
    block's 6-bit scales and minima have the two cases and the four expressions of ggml's
    get_scale_min_k4 (ggml/src/ggml-quants.c). The wider storage types Q8_0, F16 and BF16 follow
    the same version of that package: gguf/constants.py gives their numbers (GGMLQuantizationType:
    F16 = 1, Q8_0 = 8, BF16 = 30) and sizes (GGML_QUANT_SIZES), and their three cases below are
    adapted from gguf/quants.py (Q8_0.dequantize_blocks, BF16.dequantize_blocks, and dequantize for
    F16).

    Each RN32 is one float32 operation, rounded to nearest, ties to even:

        Q4_K, Q5_K:   v = RN32(RN32(RN32(h sc) c) - RN32(h_min mn))
        Q6_K:         v = RN32(RN32(h sc) c)
        F32:          v = the stored value

    Here h and h_min are the block's two float16 numbers, sc and mn are the scale and the minimum of
    the element's sub-block, and c is the element's code. h sc is h times sc.

    The wider types follow value8() and value16() of prover/weights_blocks/fold_rule.h:

        Q8_0:         v = RN32(h c)
        F16, BF16:    v = the stored value, widened exactly to float32

    In Q8_0, h is the float16 number at the start of each 34 bytes, and c is one of the 32 signed
    8-bit codes after it. An element of F16 is a float16 number. An element of BF16 is the upper 16
    bits of a float32 number whose lower 16 bits are 0.

    Every device gives the same bits. Each float32 line below is one elementwise IEEE 754
    operation. The products are exact: a float16 number has 11 significant bits, a scale or minimum
    at most 7 and a code at most 5, at most 23 in all, and float32 has 24. Only the subtraction
    rounds, so even a fused multiply-add, which rounds once
    (https://docs.nvidia.com/cuda/floating-point/index.html), gives the same bits. A decoded value
    of Q4_K, Q5_K or Q6_K is never subnormal: it is 0 or a multiple of 2^-24.

    For the wider types nothing rounds. The product of Q8_0 is exact (11 significant bits times 8)
    and is a multiple of 2^-24. The widening of a float16 number is exact. The bits of a BF16
    element are moved as an integer, so a value of BF16 that is subnormal in float32 is kept on
    every device; committed_integers says why it gives the integer 0.
    """
    n = blocks.shape[0]
    if blocks.dtype != torch.uint8 or blocks.shape != (n, BLOCK_BYTES[storage_type]):
        raise ValueError("decoded_values takes a uint8 tensor with one block per row.")
    if storage_type == F32:
        return blocks.view(torch.float32)
    if storage_type == F16:
        return blocks.view(torch.float16).to(torch.float32)
    if storage_type == BF16:
        return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)
    if storage_type == Q8_0:
        # Eight times: bytes 0-1: h. Bytes 2-33: the codes of 32 elements, in order.
        parts = blocks.view(n, 8, 34)
        # integers from -128 to 127: exact
        codes = parts[:, :, 2:34].contiguous().view(torch.int8).to(torch.float32)
        return (_binary16(parts[:, :, 0:2]) * codes).view(n, BLOCK_ELEMENTS)  # RN32(h c)

    if storage_type == Q6_K:
        # Bytes 0-127: the codes' low 4 bits. 128-191: their high 2 bits. 192-207: 16 signed scales.
        # 208-209: h. Element 128 a + 32 t + i has its low bits in byte 64 a + 32 (t mod 2) + i, in
        # the low nibble for t < 2 and in the high nibble for t >= 2, and its high bits in bits 2 t
        # and 2 t + 1 of byte 128 + 32 a + i.
        nibbles = blocks[:, 0:128].reshape(n, 2, 1, 64)
        low = torch.cat([nibbles & 15, nibbles >> 4], dim=2).view(n, 2, 4, 32)
        shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=blocks.device).view(
            1, 1, 4, 1
        )
        high = (blocks[:, 128:192].reshape(n, 2, 1, 32) >> shifts) & 3
        codes = (low | (high << 4)).to(torch.float32) - 32.0  # integers from -32 to 31: exact
        scales = blocks[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
        scaled = _binary16(blocks[:, 208:210]) * scales  # RN32(h sc), one for each 16 elements
        return (scaled.view(n, 16, 1) * codes.view(n, 16, 16)).view(n, BLOCK_ELEMENTS)

    # Q4_K and Q5_K. Bytes 0-1: h. 2-3: h_min. 4-15: eight 6-bit scales and eight 6-bit minima,
    # packed. Q4_K: bytes 16-143 hold the 4-bit codes. Q5_K: bytes 16-47 hold the codes' fifth bits,
    # 48-175 their low 4 bits.
    first, second, third = blocks[:, 4:8], blocks[:, 8:12], blocks[:, 12:16]
    scales = torch.cat([first & 63, (third & 15) | ((first >> 6) << 4)], dim=1)
    minima = torch.cat([second & 63, (third >> 4) | ((second >> 6) << 4)], dim=1)
    # Sub-blocks 2 k and 2 k + 1 (32 elements each) share 32 bytes: the even one has the low
    # nibbles.
    start = 16 if storage_type == Q4_K else 48
    nibbles = blocks[:, start : start + 128].reshape(n, 4, 1, 32)
    codes = torch.cat([nibbles & 15, nibbles >> 4], dim=2).view(n, 8, 32)
    if storage_type == Q5_K:
        # Element i of sub-block j takes bit j of byte 16 + i as its fifth bit.
        sub_blocks = torch.arange(8, dtype=torch.uint8, device=blocks.device).view(1, 8, 1)
        codes = codes | (((blocks[:, 16:48].reshape(n, 1, 32) >> sub_blocks) & 1) << 4)
    scaled = (_binary16(blocks[:, 0:2]) * scales.to(torch.float32)).view(n, 8, 1)  # RN32(h sc)
    lowest = (_binary16(blocks[:, 2:4]) * minima.to(torch.float32)).view(n, 8, 1)  # RN32(h_min mn)
    products = scaled * codes.to(torch.float32)  # RN32(RN32(h sc) c)
    return torch.sub(products, lowest).view(n, BLOCK_ELEMENTS)


def committed_integers(
    values: torch.Tensor, multipliers: torch.Tensor | None, exponent: int, limit: int | None
) -> torch.Tensor:
    """The integers of docs/model.md, "11.4 From a decoded value to an integer", as a float32 tensor
    with the shape of values, (rows, inputs). Under the limits that BlockWeights passes they are
    signed 13-bit integers for a matrix (MATRIX_LIMIT) and of up to 31 bits for the embedding
    (EMBEDDING_LIMIT).

    The element with decoded value v in input column j (the document's input index k) becomes
    w = RNE(2^e RN32(gamma_j v)), where gamma_j is the column's multiplier, e the tensor's exponent,
    and RNE rounds to the nearest integer, ties to even. Without multipliers, w = RNE(2^e v). Raises
    WeightsRejected unless every w is finite and |w| <= limit; limit + 1 must be a power of two (not
    checked). With limit = None nothing is checked.

    The rule grew out of that of VerInf (github.com/JamesPetrie/VerInf, MIT licence).

    Every device gives the same integers. gamma_j v is one rounded float32 multiplication; the
    multiplication by 2^e is exact, and an overflow is rejected. torch.round rounds half to even
    (https://docs.pytorch.org/docs/2.13/generated/torch.round.html), and its result is exact in
    float32. Flushing subnormals to zero changes nothing: with the bounds that BlockWeights checks,
    0 <= e <= 40 and |gamma_j| <= 2^64, a subnormal v or product gives |2^e gamma_j v| < 2^-22,
    which rounds to 0 either way.

    The prover implements this rule in several places. Some apply 2^e to the multiplier before the
    one rounded product (fr::fold in prover/weights_blocks/fold_rule.h), a form that is not proved
    equal to the document's on all inputs. tests/test_block_decoding.py compares this function with
    one of them, fr::weight of fold_rule.h compiled for a CPU, on random and edge values: that
    they agree is tested, not proved. The prover's CUDA decoder (prover/weights_blocks/decode.cuh)
    and the committer's conversion (prover/weights_commitment/fold_weights.cuh) are compared with
    this module only indirectly: in tests/test_against_the_prover.py on synthetic weights, and,
    with real files, through wsums.txt's digests. OPERATING.md, "What has been run", states which
    comparisons have been run.
    """
    products = values if multipliers is None else multipliers * values
    integers = torch.round(products * 2.0**exponent)
    # float32 represents the power of two limit + 1 exactly, but perhaps not limit. A comparison
    # with a NaN is false.
    if limit is not None and not bool((integers.abs() < limit + 1).all()):
        raise WeightsRejected(
            f"A weight is not finite, or its absolute value exceeds {limit:,}. The tensor's "
            "exponent may be too large."
        )
    return integers


def _check_exponent(exponent: int, tensor: str) -> int:
    if not 0 <= exponent <= 40:
        raise WeightsRejected(
            f"{tensor}: the exponent is not from 0 to 40. For a model in files it comes from the "
            "tensor's line in wsums.txt, and the embedding's from fold.txt."
        )
    return exponent


def _check_multipliers(multipliers: torch.Tensor, tensor: str) -> torch.Tensor:
    magnitude = multipliers.abs()
    if not bool(((magnitude == 0) | ((magnitude >= 2.0**-64) & (magnitude <= 2.0**64))).all()):
        raise WeightsRejected(
            f"{tensor}: every column multiplier must be 0 or have an absolute value from 2^-64 to "
            "2^64."
        )
    return multipliers


# Which tensor is which matrix


@dataclass(frozen=True)
class _Kind:
    """One row of the table in docs/model.md, "11.3 Which tensor is which matrix"."""

    stream: int  # the prover's index for this kind of tensor, also used by the synthetic model
    tensor: str  # the tensor's name in a GGUF file, without "blk.<layer>." and ".weight"
    norm: str | None = None  # the norm whose gains are multiplied into the matrix
    # True if the matrix uses c_q (c_0 divided by sqrt(dh), carrying attention's scale); only the
    # query matrix does. A matrix with no norm uses no constant.
    query: bool = False


def _fold_constants(width: int, head_dimension: int) -> tuple[int, int]:
    """The gain-folding constants (c_0, c_q) as float32 bit patterns, for a model of this width and
    head dimension, by FlashProve's own rule (scales/README.md, "The constants of a norm"): with t
    the largest integer with 4^t <= width, c = sqrt(width) / 2^t (so 1 <= c < 2), c_0 = RN32(c) and
    c_q = RN32(c / sqrt(head_dimension)). At width 5,120 head 128 these are 0x3F8F1BBD and
    0x3DCA62C2; at 2,048 and 64 they are 0x3FB504F3 and 0x3E3504F3 (sqrt(2), sqrt(2) / 8;
    docs/tinyllama/ARITHMETIC_CONTRACT.md 4.3). This generalizes the former literals, which were
    right for width 5,120 and head 128 only."""
    t = 0
    while 4 ** (t + 1) <= width:
        t += 1
    c = math.sqrt(width) / (2**t)

    def bits(value: float) -> int:  # the float32 bit pattern of value, rounded to nearest, ties to even
        return struct.unpack("<I", struct.pack("<f", value))[0]

    return bits(c), bits(c / math.sqrt(head_dimension))


# The keys are the names of WeightSource.matrix, and "Emb" for the embedding.
_KINDS = {
    "Wq": _Kind(2, "attn_q", "attn_norm", query=True),
    "Wk": _Kind(3, "attn_k", "attn_norm"),
    "Wv": _Kind(4, "attn_v", "attn_norm"),
    "Wo": _Kind(5, "attn_output"),
    "Wg": _Kind(6, "ffn_gate", "ffn_norm"),
    "Wu": _Kind(7, "ffn_up", "ffn_norm"),
    "Wd": _Kind(8, "ffn_down"),
    "Wr": _Kind(9, "ffn_gate_inp", "ffn_norm"),
    "Wgr": _Kind(10, "ffn_gate_exps", "ffn_norm"),
    "Wur": _Kind(11, "ffn_up_exps", "ffn_norm"),
    "Wdr": _Kind(12, "ffn_down_exps"),
    "Emb": _Kind(13, "token_embd"),
    "Wh": _Kind(14, "output", "output_norm"),
}
_ROUTED = ("Wgr", "Wur", "Wdr")


def tensor_name(p: Parameters, name: str, layer: int | None) -> str:
    """The name that a matrix's tensor has in a GGUF file. For the shared expert ("Wg", "Wu", "Wd"
    in a layer with experts) "_shexp" comes before ".weight"."""
    shared = "_shexp" if name in ("Wg", "Wu", "Wd") and p.has_experts(cast(int, layer)) else ""
    return ("" if layer is None else f"blk.{layer}.") + _KINDS[name].tensor + shared + ".weight"


def tensor_shape(p: Parameters, name: str, layer: int | None) -> tuple[int, int, int]:
    """Returns (experts, outputs, inputs): the number of experts (1 for a tensor that is not
    routed), and the numbers of outputs and inputs of one expert."""
    hidden = p.fe if layer is not None and p.has_experts(layer) else p.f
    experts = p.NE if name in _ROUTED else 1
    outputs, inputs = {
        "Wq": (p.dq, p.d), "Wk": (p.dkv, p.d), "Wv": (p.dkv, p.d), "Wo": (p.d, p.dq),
        "Wg": (hidden, p.d), "Wu": (hidden, p.d), "Wd": (p.d, hidden), "Wr": (p.NE, p.d),
        "Wgr": (p.fe, p.d), "Wur": (p.fe, p.d), "Wdr": (p.d, p.fe),
        "Emb": (p.V, p.d), "Wh": (p.V, p.d),
    }[name]  # fmt: skip
    return experts, outputs, inputs


def all_tensors(p: Parameters) -> list[tuple[str, int | None]]:
    """Every committed tensor of the model, as (name, layer)."""
    tensors: list[tuple[str, int | None]] = []
    for layer in range(p.L):
        names = ["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"] + (
            ["Wr", *_ROUTED] if p.has_experts(layer) else []
        )
        tensors += [(name, layer) for name in names]
    return tensors + [("Emb", None), ("Wh", None)]


class BlockSource(Protocol):
    """Where BlockWeights gets a model's blocks and the numbers that the conversion rule needs with
    them."""

    def storage_type(self, name: str, layer: int | None) -> int: ...

    def exponent(self, name: str, layer: int | None) -> int: ...

    def multipliers(self, name: str, layer: int | None) -> torch.Tensor | None:
        """gamma_j for every input column j, as float32. None for a tensor into which no norm's
        gains are multiplied."""
        ...

    def blocks(self, name: str, layer: int | None, rows: range) -> torch.Tensor:
        """The blocks of the given consecutive rows, shape (blocks, bytes of a block), on any
        device.

        A row is one output. The rows of a routed tensor are numbered consecutively across the
        experts: row r of expert e has the number e * outputs + r.
        """
        ...


# A model's GGUF files


@dataclass(frozen=True)
class _FileTensor:
    """One entry of a GGUF file's tensor table."""

    shard: int  # the number of the file that holds the tensor
    offset: int  # where the tensor's bytes start, counted from the start of that file
    dims: tuple[int, ...]  # dims[0] is the axis whose elements are contiguous in the file
    storage_type: int

    @property
    def elements(self) -> int:
        return math.prod(self.dims)

    @property
    def nbytes(self) -> int:
        return self.elements // BLOCK_ELEMENTS * BLOCK_BYTES[self.storage_type]


# Numbers that the container format fixes, under gguf-py's names.
GGUF_MAGIC = 0x46554747  # the bytes "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
_STRING, _ARRAY = 8, 9  # the two metadata types (GGUFValueType) without a fixed size
# The sizes in bytes of a metadata value of each fixed-size type.
_VALUE_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_UNSIGNED_TYPES = (0, 2, 4, 10)


class _Header:
    """Reads the fields of a GGUF header one after the other. Every number is little-endian."""

    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, "rb")
        self.size = os.fstat(self.file.fileno()).st_size

    def take(self, count: int) -> bytes:
        data = self.file.read(count)
        if len(data) != count:
            raise WeightsRejected(f"{self.path}: the file ends inside its header. Is it truncated?")
        return data

    def skip(self, count: int) -> None:
        if count > self.size - self.file.tell():
            raise WeightsRejected(f"{self.path}: the file ends inside its header. Is it truncated?")
        self.file.seek(count, os.SEEK_CUR)

    def number(self, code: str) -> int:
        return struct.unpack("<" + code, self.take(struct.calcsize(code)))[0]

    def string(self) -> str:
        length = self.number("Q")
        if length > 2**20:
            raise WeightsRejected(f"{self.path}: a string of the header is longer than 2^20 bytes.")
        return self.take(length).decode("utf-8", "replace")

    def value_bytes(self, value_type: int) -> int:
        if value_type not in _VALUE_BYTES:
            raise WeightsRejected(f"{self.path}: a metadata value has an unknown type.")
        return _VALUE_BYTES[value_type]


def _decimal_integer(text: str) -> int:
    """A decimal integer. Raises ValueError for anything else, also for what only Python reads as a
    number, such as "1_000" or " 7"."""
    if not re.fullmatch(r"[+-]?[0-9]+", text):
        raise ValueError(text)
    return int(text)


def _read_shard(path: Path, shard: int) -> tuple[dict[str, int | str], dict[str, _FileTensor]]:
    """Reads one file's header. Returns its metadata (unsigned numbers and strings only) and its
    tensor table.

    A GGUF file of version 3 starts with the magic number, the version, the number of tensors and
    the number of metadata pairs. The pairs follow, then the tensor table. The tensors' bytes start
    at the next multiple of the alignment, and the table's offsets count from there.

    It reads the same fields in the same order as prover/weights_commitment/gguf_file.h, and its
    checks follow that reader's. The numbers that the container format fixes are those of gguf-py's
    gguf/constants.py (GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT, GGUFValueType,
    GGML_QUANT_SIZES), and the order of the header's fields is that of its gguf/gguf_reader.py.
    """
    header = _Header(path)
    with header.file:
        if header.number("I") != GGUF_MAGIC:
            raise WeightsRejected(
                f"{path} is not a GGUF file: it does not start with the bytes GGUF."
            )
        if header.number("I") != GGUF_VERSION:
            raise WeightsRejected(f"{path} is not a GGUF file of version 3.")
        tensor_count, pair_count = header.number("Q"), header.number("Q")
        if max(tensor_count, pair_count) > 2**20:
            raise WeightsRejected(
                f"{path}: the header announces more than 2^20 tensors or metadata pairs."
            )
        metadata: dict[str, int | str] = {}
        for _ in range(pair_count):
            key, value_type = header.string(), header.number("I")
            if value_type == _STRING:
                metadata[key] = header.string()
            elif value_type == _ARRAY:
                element_type, count = header.number("I"), header.number("Q")
                if element_type == _STRING:
                    for _ in range(count):
                        header.string()
                else:
                    header.skip(count * header.value_bytes(element_type))
            else:
                value = header.take(header.value_bytes(value_type))
                if value_type in _UNSIGNED_TYPES:
                    metadata[key] = int.from_bytes(value, "little")
        alignment = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
        if not isinstance(alignment, int) or alignment == 0 or alignment & (alignment - 1):
            raise WeightsRejected(f"{path}: general.alignment is not a power of two.")
        table: dict[str, _FileTensor] = {}
        for entry in range(1, tensor_count + 1):
            # No message prints what was read: after a damaged header it can be bytes of the
            # weights.
            where = f"{path}, entry {entry} of the tensor table"
            name, axes = header.string(), header.number("I")
            if not 1 <= axes <= 4:
                raise WeightsRejected(f"{where}: the tensor does not have 1 to 4 axes.")
            dims = tuple(header.number("Q") for _ in range(axes))
            if any(not 1 <= axis <= 2**30 for axis in dims):
                raise WeightsRejected(f"{where}: an axis of the tensor is 0 or longer than 2^30.")
            storage_type, offset = header.number("I"), header.number("Q")
            if storage_type in (Q4_K, Q5_K, Q6_K) and dims[0] % BLOCK_ELEMENTS:
                raise WeightsRejected(
                    f"{where}: the rows of the quantized tensor are not a multiple of 256 long."
                )
            if storage_type == Q8_0 and dims[0] % 32:
                raise WeightsRejected(
                    f"{where}: the rows of the Q8_0 tensor are not a multiple of 32 long. "
                    "The format stores Q8_0 in blocks of 32 elements, so the file is damaged."
                )
            if offset % alignment:
                raise WeightsRejected(
                    f"{where}: the tensor's offset is not a multiple of the alignment."
                )
            if name in table:
                raise WeightsRejected(f"{where}: an earlier entry has the same name.")
            table[name] = _FileTensor(shard, offset, dims, storage_type)
        end = header.file.tell()
    data = (end + alignment - 1) // alignment * alignment
    return metadata, {
        name: _FileTensor(shard, data + t.offset, t.dims, t.storage_type)
        for name, t in table.items()
    }


class GgufBlocks:
    """The blocks of a model in GGUF files, with the model's set-up directory (`setup_folder`).

    The set-up directory holds two text files:

    - wsums.txt, the checksum file, has one line "<name> <elements> <exponent> <bits> <SHA-256>" per
      committed tensor and a last line "END <number of lines>". The programs in weight_sums/ write
      it (docs/model.md, "11.4 From a decoded value to an integer"). This class does not use
      <bits>, the width of the tensor's integers. cp_commit compares the SHA-256 of every tensor's
      integers with this file and makes no commitment if one differs.
    - fold.txt has two lines: "eres <e>" gives the embedding's exponent, and "qkrows asis" states
      that the rows of Wq and Wk are taken in the file's order.

    The first of `shards` must be the model's first file, whose header holds the metadata.

    sha256, if given, is {"wsums.txt": digest, "fold.txt": digest} (see given_digests in
    parameters.py). A file with another digest raises ParameterError. Each of the two files is read
    once, so the bytes that are hashed are the bytes that are parsed.

    The checks follow those of the prover's readers (prover/weights_commitment/gguf_file.h,
    prover/weights_blocks/gguf_source.cuh) but are not the same: the prover has further limits, for
    example on the routed experts' storage types (BlockWeights::fill in
    prover/weights_blocks/block_weights.cuh), so files that load here may be rejected there and get
    no proof.

    The weights' bytes are not kept in memory: blocks() reads them from the files at every call.
    """

    def __init__(
        self,
        parameters: Parameters,
        setup_folder: str | Path,
        shards: list[str | Path],
        sha256: dict[str, str] | None = None,
    ):
        self._p = parameters
        self._tensors: dict[str, _FileTensor] = {}
        self._files: list[int] = []
        self._paths = [Path(shard) for shard in shards]
        try:
            self._open(Path(setup_folder), given_digests(sha256, ("wsums.txt", "fold.txt")))
        except Exception:
            self.close()
            raise

    def _open(self, setup_folder: Path, sha256: dict[str, str | None]) -> None:
        p = self._p
        for number, shard in enumerate(self._paths):
            metadata, table = _read_shard(shard, number)
            if number == 0:
                architecture = metadata.get("general.architecture")
                if not isinstance(architecture, str) or not architecture:
                    raise WeightsRejected(
                        f"{shard}: the metadata do not name the model's architecture. This must "
                        "be the model's first file, which holds the metadata."
                    )
                prefix = architecture + "."
                width, layers = (
                    metadata.get(prefix + "embedding_length"),
                    metadata.get(prefix + "block_count"),
                )
                head = metadata.get(prefix + "attention.key_length", 128)
                # As in _read_shard, the values are not printed. The conversion rule's constants are
                # a function of the width and head dimension (_fold_constants), so any shape that
                # matches the parameter file is accepted; the parameter file's own checks bound dh.
                if (width, head, layers) != (p.d, p.dh, p.L):
                    raise WeightsRejected(
                        f"{shard}: the model's number of layers, width and head dimension are not "
                        f"those of the parameter file: {p.L}, {p.d} and {p.dh}."
                    )
            for entry, (name, tensor) in enumerate(table.items(), start=1):
                if name in self._tensors:
                    raise WeightsRejected(
                        f"{shard}, entry {entry} of the tensor table: an earlier file has the "
                        "same name."
                    )
                self._tensors[name] = tensor
            self._files.append(os.open(shard, os.O_RDONLY))
        self.sums = self._read_sums(setup_folder / "wsums.txt", sha256["wsums.txt"])
        self._embedding_exponent = self._read_embedding_exponent(
            setup_folder / "fold.txt", sha256["fold.txt"]
        )
        for layer in range(p.L):
            if (f"blk.{layer}.ffn_gate_exps.weight" in self._tensors) != p.has_experts(layer):
                here, there = "the parameter file", "the model's files"
                if not p.has_experts(layer):
                    here, there = there, here
                raise WeightsRejected(
                    f"Layer {layer} has routed experts in {here} but not in {there}."
                )
        for name, layer in all_tensors(p):
            self._check_tensor(name, layer)

    def close(self) -> None:
        for file in self._files:
            os.close(file)
        self._files = []

    def _read_sums(self, path: Path, sha256: str | None) -> dict[str, tuple[int, int, str]]:
        """wsums.txt as a dictionary: tensor name -> (elements, exponent, SHA-256 as written; only
        lower case matches a computed digest)."""
        words = read_bytes(path, sha256).decode(errors="replace").split()
        sums: dict[str, tuple[int, int, str]] = {}
        at = 0
        while at < len(words) and words[at] != "END":
            try:
                name, elements, exponent, _bits, digest = words[at : at + 5]
                line = (_decimal_integer(elements), _decimal_integer(exponent), digest)
            except ValueError:
                raise WeightsRejected(
                    f"{path}: line {len(sums) + 1} must have five fields: "
                    "name, number of elements, exponent, bits, SHA-256."
                ) from None
            # As in _read_shard, no message prints what was read.
            if name in sums:
                raise WeightsRejected(
                    f"{path}: line {len(sums) + 1}: an earlier line lists the same tensor."
                )
            if name not in self._tensors:
                raise WeightsRejected(
                    f"{path}: line {len(sums) + 1}: the model's files have no tensor of this name."
                )
            if self._tensors[name].elements != line[0]:
                raise WeightsRejected(
                    f"{path}: line {len(sums) + 1}: the number of elements is not that of the "
                    "tensor in the model's files."
                )
            sums[name] = line
            at += 5
        if words[at:] != ["END", str(len(sums))]:
            raise WeightsRejected(
                f"{path}: the last line must be END with the number of lines before it."
            )
        return sums

    @staticmethod
    def _read_embedding_exponent(path: Path, sha256: str | None) -> int:
        """Returns the embedding's exponent, from the line "eres <e>".

        The prover's readers of this file (GgufSource::read_fold_words in
        prover/weights_blocks/gguf_source.cuh, FoldSource::open in
        prover/weights_commitment/fold_source.cuh) split it into words of at most 63
        bytes at ASCII's six white-space characters, and end a word at a zero byte. bytes.split()
        splits at the same six characters; a file with a longer word or a zero byte is rejected,
        because those readers would pair its words differently."""
        words = read_bytes(path, sha256).split()
        pairs = list(zip(words[0::2], words[1::2]))
        exponents = [value for word, value in pairs if word == b"eres"]
        orders = [value for word, value in pairs if word == b"qkrows"]
        try:
            if (
                len(exponents) != 1
                or orders != [b"asis"]
                or any(len(word) > 63 or 0 in word for word in words)
            ):
                raise ValueError
            return _decimal_integer(exponents[0].decode("ascii"))
        except ValueError:
            raise WeightsRejected(
                f'{path} must have the lines "eres <integer>" and "qkrows asis", each exactly '
                "once, and no word of more than 63 characters."
            ) from None

    def _tensor(self, name: str, layer: int | None) -> _FileTensor:
        return self._tensors[tensor_name(self._p, name, layer)]

    def _check_tensor(self, name: str, layer: int | None) -> None:
        file_name = tensor_name(self._p, name, layer)
        if file_name not in self._tensors:
            raise WeightsRejected(f"The model's files have no tensor {file_name}.")
        tensor = self._tensors[file_name]
        if tensor.storage_type not in BLOCK_BYTES:
            raise WeightsRejected(
                f"{file_name}: the storage type is not supported. "
                "Supported types: F32 (0), F16 (1), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14), "
                "BF16 (30)."
            )
        experts, outputs, inputs = tensor_shape(self._p, name, layer)
        dims = tensor.dims + (1,) * (4 - len(tensor.dims))
        if dims[0] != inputs or dims[1] * dims[2] != experts * outputs or dims[3] != 1:
            raise WeightsRejected(
                f"{file_name}: the tensor in the model's files does not have the shape that the "
                f"parameter file gives: {experts * outputs:,} rows of {inputs:,} elements."
            )
        if inputs % BLOCK_ELEMENTS:
            raise WeightsRejected(f"{file_name}: the rows are not a multiple of 256 elements long.")
        self._check_extent(file_name, tensor)
        if name != "Emb" and file_name not in self.sums:
            raise WeightsRejected(f"wsums.txt has no line for the tensor {file_name}.")
        norm = self._norm_name(name, layer)
        if norm is not None:
            gains = self._tensors.get(norm)
            if gains is None or gains.storage_type != F32 or gains.elements != self._p.d:
                raise WeightsRejected(
                    f"{norm} must be a tensor of {self._p.d:,} values of type F32."
                )
            self._check_extent(norm, gains)

    def _check_extent(self, file_name: str, tensor: _FileTensor) -> None:
        if tensor.offset + tensor.nbytes > os.fstat(self._files[tensor.shard]).st_size:
            raise WeightsRejected(
                f"{file_name}: the tensor extends past the end of its file. Is the file truncated?"
            )

    @staticmethod
    def _norm_name(name: str, layer: int | None) -> str | None:
        """The name of the tensor with the gains that are multiplied into a matrix, or None if there
        are none."""
        norm = _KINDS[name].norm
        return (
            None if norm is None else ("" if layer is None else f"blk.{layer}.") + norm + ".weight"
        )

    def _read(self, tensor: _FileTensor, first: int, count: int, into: torch.Tensor) -> None:
        """Reads bytes first .. first + count - 1 of a tensor into a one-dimensional uint8 tensor in
        host memory."""
        buffer = memoryview(into.numpy())
        done = 0
        while done < count:
            got = os.preadv(
                self._files[tensor.shard], [buffer[done:count]], tensor.offset + first + done
            )
            if got <= 0:
                raise WeightsRejected(
                    f"{self._paths[tensor.shard]}: the file ends inside a tensor's data."
                )
            done += got

    def storage_type(self, name: str, layer: int | None) -> int:
        return self._tensor(name, layer).storage_type

    def exponent(self, name: str, layer: int | None) -> int:
        return (
            self._embedding_exponent
            if name == "Emb"
            else self.sums[tensor_name(self._p, name, layer)][1]
        )

    def multipliers(self, name: str, layer: int | None) -> torch.Tensor | None:
        """gamma_j = RN32(g_j c_0), for the norm's gains g and the constant c_0."""
        norm = self._norm_name(name, layer)
        if norm is None:
            return None
        stored = torch.empty(4 * self._p.d, dtype=torch.uint8)
        self._read(self._tensors[norm], 0, 4 * self._p.d, stored)
        c0, cq = _fold_constants(self._p.d, self._p.dh)
        constant = torch.tensor([cq if _KINDS[name].query else c0], dtype=torch.int32).view(torch.float32)
        return stored.view(torch.float32) * constant

    def blocks(self, name: str, layer: int | None, rows: range) -> torch.Tensor:
        tensor = self._tensor(name, layer)
        if rows.step != 1 or not 0 <= rows.start <= rows.stop <= tensor.elements // tensor.dims[0]:
            raise IndexError(f"{tensor_name(self._p, name, layer)} has no rows {rows}.")
        block_bytes = BLOCK_BYTES[tensor.storage_type]
        row_bytes = tensor.dims[0] // BLOCK_ELEMENTS * block_bytes
        read = torch.empty(len(rows) * row_bytes, dtype=torch.uint8)
        self._read(tensor, rows.start * row_bytes, len(rows) * row_bytes, read)
        return read.view(-1, block_bytes)


# The synthetic model in blocks


def _mix64_of_number(word: int) -> int:
    return int(mix64(torch.tensor([signed64(word)], dtype=torch.int64))[0])


def _bytes_of(words: torch.Tensor) -> torch.Tensor:
    """The bytes of int64 words, shape (n, k), lowest byte first, shape (n, 8 k)."""
    return words.contiguous().view(torch.uint8)


class SyntheticBlocks:
    """The synthetic model that the committer, cp_commit, generates under `-weights made-up-blocks`,
    computed from the parameter file's seed.

    It needs no files, whatever its size. The class follows
    prover/weights_blocks/made_up_blocks.h (the blocks of type Q4_K and Q5_K and the multipliers)
    and prover/weights_blocks/made_up_source.cuh (the router, the embedding and the head,
    which are F32 tensors). Every tensor has the exponent 12.
    """

    def __init__(self, parameters: Parameters, device: torch.device | str = "cpu"):
        self._p = parameters
        self.device = torch.device(device)

    def storage_type(self, name: str, layer: int | None) -> int:
        # The rule of made_up::type_of in prover/weights_blocks/made_up_blocks.h. At full size 9 of
        # the 24 routed down projections ("Wdr") are Q5_K, the share in the public model's file
        # headers (THIRD_PARTY.md, "ggml"). Which layers they are is arbitrary.
        stream = _KINDS[name].stream
        if name in ("Wr", "Emb", "Wh"):
            return F32
        layer = cast(int, layer)  # every other tensor has a layer
        if name == "Wdr":
            return Q5_K if (layer >> 1) & 7 in (0, 3, 6) else Q4_K
        return Q5_K if stream <= 8 and (layer * 5 + stream) % 4 == 1 else Q4_K

    def exponent(self, name: str, layer: int | None) -> int:
        return 12

    def multipliers(self, name: str, layer: int | None) -> torch.Tensor | None:
        """About seven columns in eight get a value in [0.5, 1), the others one in [1, 1.09375).

        As in the prover's synthetic model, the router and the head get no multipliers.
        """
        if name in ("Wq", "Wk", "Wv"):
            norm = 0
        elif name in ("Wg", "Wu", "Wgr", "Wur"):
            norm = 1
        else:
            return None
        columns = torch.arange(self._p.d, dtype=torch.int64, device=self.device)
        words = mix64(
            columns + signed64(self._p.seed + ((90 + norm) << 40) + (cast(int, layer) << 20))
        )
        high = shift_right64(words, 8)
        bits = torch.where(
            (words & 7) != 0, 0x3F000000 | (high & 0x7FFFFF), 0x3F800000 | (high % 786432)
        )
        return bits.to(torch.int32).view(torch.float32)

    def blocks(self, name: str, layer: int | None, rows: range) -> torch.Tensor:
        numbers = torch.arange(
            rows.start, rows.stop, rows.step, dtype=torch.int64, device=self.device
        )
        if self.storage_type(name, layer) == F32:
            return self._generator_values(name, layer, numbers)
        return self._quantized_blocks(name, cast(int, layer), numbers)

    def _generator_values(self, name: str, layer: int | None, rows: torch.Tensor) -> torch.Tensor:
        """Rows of the router, the embedding or the head: a synthetic integer divided by 4,096, as
        float32.

        The integer is that of synthetic_weights.stream_entries, with bits = 20 for the embedding
        and 13 otherwise. The division and the rule's multiplication by 2^12 are exact, so the rule
        gives the integer back.
        """
        _, outputs, inputs = tensor_shape(self._p, name, layer)
        columns = torch.arange(inputs, dtype=torch.int64, device=self.device)
        if name == "Emb":
            bits, index = 20, rows[:, None] * inputs + columns[None, :]
        else:  # these two are numbered with the output index running fastest
            bits, index = 13, columns[None, :] * outputs + rows[:, None]
            if name == "Wr":
                index = index + cast(int, layer) * inputs * outputs
        integers = stream_entries(self._p.seed, _KINDS[name].stream, index, bits)
        values = integers.to(torch.float32) * (1.0 / 4096.0)
        return values.view(torch.uint8).view(-1, BLOCK_BYTES[F32])

    def _quantized_blocks(self, name: str, layer: int, rows: torch.Tensor) -> torch.Tensor:
        """Rows of a tensor of type Q4_K or Q5_K (made_up::block)."""
        five_bits = self.storage_type(name, layer) == Q5_K
        _, outputs, inputs = tensor_shape(self._p, name, layer)
        in_a_row = inputs // BLOCK_ELEMENTS
        # A block's index combines its layer, its expert, its row within the expert and its position
        # in the row. Word i of the block is mix64(mix64(seed + 81 * stream) + 32 * index + i).
        expert, row = rows // outputs, rows % outputs
        positions = torch.arange(in_a_row, dtype=torch.int64, device=self.device)
        index = ((layer * 1024 + expert) * 65536 + row)[:, None] * 256 + positions[None, :]
        numbers = torch.arange(23 if five_bits else 19, dtype=torch.int64, device=self.device)
        start = _mix64_of_number(self._p.seed + 0x51 * _KINDS[name].stream)
        words = mix64(index.reshape(-1, 1) * 32 + numbers[None, :] + start)

        # Word 0 gives the block's scale h, as the bits of a float16 number: negative once in 16,
        # zero once in 256.
        first = words[:, 0]
        sign = ((shift_right64(first, 20) & 15) == 0).to(torch.int64) << 15
        if five_bits:
            h = sign | ((3 + (first & 1)) << 10) | (shift_right64(first, 1) % 900)
        else:
            h = sign | ((4 + (first & 1)) << 10) | (shift_right64(first, 1) % 960)
        h = torch.where((shift_right64(first, 28) & 255) == 0, 0, h)
        # h_min is h times half the largest code, rounded towards zero, so that the codes' range is
        # centred on zero.
        significand = (1024 + (h & 1023)) * (31 if five_bits else 15)
        exponent = ((h >> 10) & 31) - 1
        for _ in range(5):
            too_wide = significand >= 2048
            significand = torch.where(too_wide, significand >> 1, significand)
            exponent = exponent + too_wide
        h_min = torch.where(h != 0, (h & 0x8000) | (exponent << 10) | (significand - 1024), 0)

        # Words 1 and 2 give the twelve bytes of scales and minima, such that each sub-block's
        # minimum equals its scale. Words 3 to 18 are the 128 bytes of codes, and words 19 to 22 the
        # 32 bytes of fifth bits.
        packed = _bytes_of(words[:, 1:2])[:, 0:4]
        nibble = _bytes_of(words[:, 2:3])[:, 0:4] & 15
        parts = [
            _bytes_of(torch.stack([h | (h_min << 16)], dim=1))[:, 0:4],
            packed,
            packed,
            nibble | (nibble << 4),
        ]
        if five_bits:
            parts.append(_bytes_of(words[:, 19:23]))
        parts.append(_bytes_of(words[:, 3:19]))
        return torch.cat(parts, dim=1)


# The free host memory


def _number_in(file: Path, label: str | None = None) -> int | None:
    """Returns the integer in a kernel text file: the first field, or the field after label. None if
    there is none."""
    try:
        fields = file.read_text().split()
    except (OSError, ValueError):  # ValueError: the file is not text, or the path has a zero byte
        return None
    at = 0 if label is None else fields.index(label) + 1 if label in fields else len(fields)
    return int(fields[at]) if at < len(fields) and re.fullmatch("[0-9]{1,20}", fields[at]) else None


# For cgroup v2 and v1: the memory controller's name in /proc/self/cgroup, which is also its
# directory in /sys/fs/cgroup; the limit files; the usage file; the prefix of active_file and
# inactive_file in memory.stat.
_CGROUP_FILES: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("", ("memory.max", "memory.high"), "memory.current", ""),
    ("memory", ("memory.limit_in_bytes",), "memory.usage_in_bytes", "total_"),
)


def _cgroup_free(
    cgroup: Path, limit_files: tuple[str, ...], usage_file: str, prefix: str
) -> tuple[int, str] | None:
    """Returns one cgroup's estimate for free_host_memory, or None if the cgroup has no limit or
    cannot be read."""
    limits = [
        (limit, file) for file in limit_files if (limit := _number_in(cgroup / file)) is not None
    ]
    usage = _number_in(cgroup / usage_file)
    # "No limit" is the string "max" in v2 and a number near 2^63 in v1.
    if usage is None or not limits or min(limits)[0] >= 2**60:
        return None
    limit, limit_file = min(limits)
    cache = sum(
        _number_in(cgroup / "memory.stat", prefix + kind) or 0
        for kind in ("active_file", "inactive_file")
    )
    free = min(limit, max(0, limit - usage + cache))
    return free, f"{limit_file} - {usage_file} + file cache in {cgroup}"


def free_host_memory(
    proc: Path = Path("/proc"), cgroups: Path = Path("/sys/fs/cgroup")
) -> tuple[int, str] | None:
    """Estimates how many bytes of host memory this process can still use, on Linux.

    Returns (bytes, where the number comes from), or None if nothing can be read. The estimate is
    the smallest of:

    - MemAvailable in /proc/meminfo. It describes the whole machine, even inside a container.
    - For this process's cgroup and each of its parents: limit - usage + file cache. In v2 the limit
      is the lower of memory.max and memory.high. The kernel can reclaim file cache, and
      MemAvailable counts it as free too.

    Not counted: swap, and a limit on a cgroup that a container cannot see.
    """
    estimates = []
    available = _number_in(proc / "meminfo", "MemAvailable:")  # in KiB
    if available is not None:
        estimates.append((1024 * available, f"MemAvailable in {proc / 'meminfo'}"))
    try:
        # A cgroup's name can be any bytes.
        text = (proc / "self" / "cgroup").read_text(errors="surrogateescape")
    except OSError:
        text = ""
    own = [line.split(":", 2) for line in text.split("\n") if line.count(":") >= 2]
    for controller, limit_files, usage_file, prefix in _CGROUP_FILES:
        paths = [path for _, controllers, path in own if controller in controllers.split(",")]
        for path in paths or ["/"]:
            steps = [step for step in path.split("/") if step]
            if ".." in steps:  # the cgroup is outside this process's cgroup namespace
                steps = []
            # A container usually sees its own cgroup as the root, so the longer paths may not
            # exist.
            lineage = [
                (cgroups / controller).joinpath(*steps[:depth]) for depth in range(len(steps) + 1)
            ]
            found = [_cgroup_free(cgroup, limit_files, usage_file, prefix) for cgroup in lineage]
            estimates += [estimate for estimate in found if estimate]
    return min(estimates, default=None)


# The weight source


class _HostSlabs:
    """Host memory for the routed experts' blocks, allocated in slabs of 1 GiB. The blocks of one
    expert's tensor never span two slabs.

    With pinned = True the slabs are page-locked, so that the GPU copies from them directly.
    PyTorch's allocator of such memory rounds every request up to a power of two
    (https://docs.pytorch.org/docs/2.13/generated/torch.cuda.memory.host_memory_stats.html): at full
    size all experts of one routed Q4_K tensor, about 3.0 GB (computed), would take 4 GiB. So a slab
    is requested at exactly a power of two and holds as many whole expert tensors as fit.
    """

    SLAB_BYTES = 2**30

    def __init__(self, pinned: bool):
        self._pinned = pinned
        self._slab = torch.empty(0, dtype=torch.uint8)
        self._used = 0
        self.slabs = 0

    def allocate(self, nbytes: int) -> torch.Tensor:
        if nbytes > self.SLAB_BYTES:
            raise WeightsRejected(
                f"One expert's tensor takes more than {self.SLAB_BYTES:,} bytes, the most that the "
                "loader supports."
            )
        if self._used + nbytes > self._slab.numel():
            self._slab = torch.empty(self.SLAB_BYTES, dtype=torch.uint8, pin_memory=self._pinned)
            self._used = 0
            self.slabs += 1
        self._used += nbytes
        return self._slab[self._used - nbytes : self._used]

    @classmethod
    def bytes_for(cls, sizes: list[int]) -> int:
        """Returns the total size of the slabs that allocate() would create for requests of these
        sizes, in this order.

        Stops at a request that allocate() would reject.
        """
        slabs, used = 0, cls.SLAB_BYTES
        for nbytes in sizes:
            if nbytes > cls.SLAB_BYTES:
                break
            if used + nbytes > cls.SLAB_BYTES:
                slabs, used = slabs + 1, 0
            used += nbytes
        return slabs * cls.SLAB_BYTES


@dataclass
class _Stored:
    """What BlockWeights knows about one tensor: the numbers of the conversion rule, and the
    tensor's shape."""

    storage_type: int
    exponent: int
    multipliers: torch.Tensor | None  # on the device
    experts: int
    outputs: int
    inputs: int


def host_bytes_needed(
    p: Parameters, source: BlockSource, device: torch.device, pinned: bool
) -> int:
    """Returns the host memory in bytes that BlockWeights will hold for a model, from its shapes and
    storage types."""
    experts_blocks: list[int] = []
    others = 0
    for name, layer in all_tensors(p):
        experts, outputs, inputs = tensor_shape(p, name, layer)
        # An unknown storage type counts as 0 bytes here; GgufBlocks rejects one when it opens
        # the files.
        nbytes = (
            outputs
            * (inputs // BLOCK_ELEMENTS)
            * BLOCK_BYTES.get(source.storage_type(name, layer), 0)
        )
        if name in _ROUTED:
            experts_blocks += experts * [nbytes]
        elif name == "Emb":
            others += nbytes
        elif device.type == "cpu":  # the int16 matrices "on the device" are then in host memory too
            others += 2 * outputs * inputs
    # A pinned slab is resident as a whole. An ordinary slab is resident only where it has been
    # written.
    return others + (_HostSlabs.bytes_for(experts_blocks) if pinned else sum(experts_blocks))


class BlockWeights:
    """A weight source (WeightSource in weights.py) over a model that is given as quantized blocks.

        weights = BlockWeights(parameters, "cuda", blocks="synthetic")
        weights = BlockWeights(
            parameters, "cuda", blocks="gguf", setup_folder="setup", shards=["1.gguf", "2.gguf"]
        )

    Where the weights are kept. The sizes are for the full-size shape (400 billion parameters),
    computed from that shape and the block types, not measured (more: OPERATING.md, "Full size").

    - The matrices that every token uses (attention, dense layers, shared experts, routers) and the
      head are decoded once, at loading, and kept on the device as 16-bit integers: 24.2 GB for the
      matrices and 2.1 GB for the head.
    - The routed experts stay as blocks in host memory, about 223 GB. matrix() copies one expert's
      blocks to the device and decodes them there.
    - The embedding stays as blocks in host memory, 0.6 GB if its type is Q4_K. embedding_rows()
      copies the requested rows to the device and decodes them there.

    With wider files only what stays in blocks grows, because the integers have 16 bits whatever the
    file's type. In Q8_0 the routed experts take 411 GB of host memory (44.6 MB for one expert's
    tensor), which _HostSlabs allocates as 412 GB, and the embedding takes 1.1 GB. In F16 or BF16
    the routed experts take 773 GB (83.9 MB for one expert's tensor), allocated as 825 GB, and the
    embedding takes 2.1 GB.

    Keeping the experts in pinned host memory and fetching the chosen ones is a published technique:
    Eliseev and Mazur, "Fast Inference of Mixture-of-Experts Language Models with Offloading"
    (arXiv:2312.17238, sections 3.2 and 3.3), and llama.cpp's option --cpu-moe (its
    tools/cli/README.md).

    Loading decodes every weight once and raises WeightsRejected if one is outside the allowed
    range. With hash_integers = True it also stores, for every tensor, its number of elements and
    the SHA-256 of its integers (fed as _hash describes) in the attribute digests. With
    verify_sums = True it also compares the digests with wsums.txt and rejects the model if one
    differs; hashing is slow at full size (OPERATING.md, "Verify what was loaded").
    files_not_compared is True if the weights come from GGUF files and verify_sums is False.
    sums_verified is True if verify_sums is True: a loaded model then has passed the comparison.

    Arguments:
        parameters, device: the model's public parameters, and the device on which matrix() returns
            its tensors.
        folder: relative paths in setup_folder and shards are relative to this directory.
        blocks: "synthetic" for SyntheticBlocks, "gguf" for GgufBlocks with setup_folder and shards
            (which are allowed only with "gguf"), or a BlockSource. It must be given.
        hash_integers, verify_sums: see above.
        sha256: passed on to GgufBlocks. It is allowed only with blocks = "gguf".
        pin_memory: whether the routed experts' blocks are kept in page-locked host memory. The
            default is to do so if the device is a GPU. The embedding's blocks are in ordinary host
            memory.
        check_free_memory: if True, raise NotEnoughMemory before allocating anything if the weights
            (host_bytes_needed) plus an allowance for the rest of the process need more host memory
            than free_host_memory() finds. The check is skipped where free_host_memory() can read
            nothing. Pass False if the estimate is too low for your system, for example because it
            ignores swap. The check runs once, before loading, which takes minutes: other processes
            can still use up the memory, and the kernel may then kill this process or others.
    """

    # The largest number of weights decoded in one step (but at least one row), which bounds the
    # memory of intermediate tensors. A routed expert's tensor is fetched whole; at full size, 42
    # million weights, it is also decoded in one step.
    WEIGHTS_AT_ONCE = 2**26

    # check_free_memory allows 1/32 of the weights' size for what the process allocates besides the
    # weights. A fixed allowance would reject small models.
    ALLOWANCE_DIVISOR = 32

    def __init__(
        self,
        parameters: Parameters,
        device: torch.device | str,
        folder: str | Path = ".",
        blocks: str | BlockSource | None = None,
        setup_folder: str | Path | None = None,
        shards: list[str] | None = None,
        hash_integers: bool = False,
        verify_sums: bool = False,
        pin_memory: bool | None = None,
        sha256: dict[str, str] | None = None,
        check_free_memory: bool = True,
    ):
        self.device = torch.device(device)
        self._p = parameters
        if (setup_folder is not None or shards is not None) and blocks != "gguf":
            raise ValueError(
                'setup_folder and shards need blocks = "gguf": only then does this class open what '
                "they name."
            )
        if blocks == "synthetic":
            source: BlockSource = SyntheticBlocks(parameters, self.device)
        elif blocks == "gguf":
            if setup_folder is None or not shards:
                raise ValueError('blocks = "gguf" needs setup_folder and shards.')
            source = GgufBlocks(
                parameters, Path(folder) / setup_folder, [Path(folder) / s for s in shards], sha256
            )
        elif blocks is None or isinstance(blocks, str):
            raise ValueError('blocks must be "synthetic", "gguf" or a block source.')
        else:
            source = blocks
        if verify_sums and not isinstance(source, GgufBlocks):
            raise ValueError(
                "verify_sums needs a model in GGUF files: only such a model has a checksum file."
            )
        if sha256 is not None and blocks != "gguf":
            raise ValueError(
                'sha256 needs blocks = "gguf": only then does this class read the set-up directory.'
            )
        self._slabs = _HostSlabs(self.device.type == "cuda" if pin_memory is None else pin_memory)
        self._stored: dict[tuple[str, int | None], _Stored] = {}
        self._on_device: dict[tuple[str, int | None], torch.Tensor] = {}
        self._on_host: dict[tuple[str, int | None, int | None], torch.Tensor] = {}
        # In host memory: one row of blocks per token id.
        self._embedding = torch.empty(0, dtype=torch.uint8)
        self.digests: dict[str, tuple[int, str]] = {}  # tensor name -> (elements, SHA-256)
        self.files_not_compared = isinstance(source, GgufBlocks) and not verify_sums
        self.sums_verified = verify_sums
        self._loading = True
        try:
            if isinstance(source, GgufBlocks) and verify_sums:
                # Only the comparison needs the embedding's line. A missing one is reported now,
                # before any weight is decoded, as GgufBlocks does for the other tensors' lines.
                embedding = tensor_name(parameters, "Emb", None)
                if embedding not in source.sums:
                    raise WeightsRejected(f"wsums.txt has no line for the tensor {embedding}.")
            if check_free_memory:
                self._check_free_memory(source)
            started = time.monotonic()
            logger.info("Loading and decoding the weights of %d layers.", parameters.L)
            for name, layer in all_tensors(parameters):
                if name == "Wq" and layer and layer % 8 == 0:
                    logger.info(
                        "Loaded %d of %d layers (%.0f s).",
                        layer,
                        parameters.L,
                        time.monotonic() - started,
                    )
                self._load(source, name, layer, hash_integers or verify_sums)
            logger.info("Loaded all weights (%.0f s).", time.monotonic() - started)
            if (
                isinstance(source, GgufBlocks)
                and verify_sums
                and (faults := digest_faults(self.digests, source.sums))
            ):
                raise WeightsRejected(
                    "The model's integers are not those of wsums.txt: " + "; ".join(faults) + "."
                )
        finally:
            if isinstance(source, GgufBlocks) and blocks == "gguf":
                source.close()
        self._loading = False
        if self.device.type == "cuda":
            # So that the weights are complete for work on any CUDA stream.
            torch.cuda.synchronize(self.device)

    def _check_free_memory(self, source: BlockSource) -> None:
        weights = host_bytes_needed(self._p, source, self.device, self._slabs._pinned)
        allowance = weights // self.ALLOWANCE_DIVISOR
        free = free_host_memory()
        if free is None or weights + allowance <= free[0]:
            return

        def size(nbytes: int) -> str:
            return f"{nbytes / 1e9:.1f} GB" if nbytes >= 1e9 else f"{nbytes / 1e6:.1f} MB"

        raise NotEnoughMemory(
            f"Not enough free host memory to load the weights: need {size(weights + allowance)} "
            f"({size(weights)} for the weights, {size(allowance)} for the rest of the process), "
            f"found {size(free[0])} ({free[1]}). Nothing was allocated. Use a machine or "
            "container with more free memory; if you know that more is free than was found, skip "
            'this check: add "check_free_memory": false to the entries of "weight_settings" in '
            "integer_model.json or, with python -m integer_model.block_weights, pass "
            "--no-memory-check."
        )

    def _load(self, source: BlockSource, name: str, layer: int | None, hash_integers: bool) -> None:
        file_name = tensor_name(self._p, name, layer)
        experts, outputs, inputs = tensor_shape(self._p, name, layer)
        if inputs % BLOCK_ELEMENTS:
            raise WeightsRejected(
                f"{file_name}: the number of inputs, {inputs:,}, is not a multiple of 256."
            )
        multipliers = source.multipliers(name, layer)
        if multipliers is not None:
            multipliers = _check_multipliers(multipliers.to(self.device), file_name)
        storage_type = source.storage_type(name, layer)
        exponent = _check_exponent(source.exponent(name, layer), file_name)
        # cp_prove keeps a layer's matrix of one of these five types, other than the router, as its
        # blocks (decoded_from_blocks in prover/weights_blocks/blocks.h), and takes at most 24 for
        # its exponent (prover/weights_blocks/block_weights.cuh); cp_commit refuses an exponent
        # above 24 too.
        kept_types = (Q4_K, Q5_K, Q8_0, F16, BF16)
        blocks_kept = layer is not None and name != "Wr" and storage_type in kept_types
        if blocks_kept and exponent > 24:
            raise WeightsRejected(
                f"{file_name}: it is stored as Q4_K, Q5_K, Q8_0, F16 or BF16, so its exponent must "
                "be at most 24, as cp_commit and cp_prove require."
            )
        stored = self._stored[(name, layer)] = _Stored(
            storage_type,
            exponent,
            multipliers,
            experts,
            outputs,
            inputs,
        )
        digest = hashlib.sha256() if hash_integers else None
        step = max(1, self.WEIGHTS_AT_ONCE // inputs)
        pieces = [range(first, min(first + step, outputs)) for first in range(0, outputs, step)]
        # Tensors that stay in blocks are decoded here from the blocks that were kept, with the same
        # method that returns them later (matrix or embedding_rows). So the limit check and the
        # digest cover exactly what the model will read.
        try:
            if name in _ROUTED:
                for expert in range(experts):
                    blocks = source.blocks(
                        name, layer, range(expert * outputs, (expert + 1) * outputs)
                    )
                    kept = self._on_host[(name, layer, expert)] = self._slabs.allocate(
                        blocks.numel()
                    )
                    kept.copy_(blocks.flatten())
                    self._hash(digest, self.matrix(name, layer, expert))
            elif name == "Emb":
                row_bytes = inputs // BLOCK_ELEMENTS * BLOCK_BYTES[stored.storage_type]
                self._embedding = torch.empty((outputs, row_bytes), dtype=torch.uint8)
                for rows in pieces:
                    self._embedding[rows.start : rows.stop] = source.blocks(name, layer, rows).view(
                        -1, row_bytes
                    )
                    token_ids = torch.arange(
                        rows.start, rows.stop, dtype=torch.int64, device=self.device
                    )
                    self._hash(digest, self.embedding_rows(token_ids).to(torch.int32))
            else:
                kept = self._on_device[(name, layer)] = torch.empty(
                    (outputs, inputs), dtype=torch.int16, device=self.device
                )
                for rows in pieces:
                    blocks = source.blocks(name, layer, rows).to(self.device)
                    kept[rows.start : rows.stop] = self._integers(stored, blocks, torch.int16)
                    self._hash(digest, kept[rows.start : rows.stop])
        except WeightsRejected as fault:
            raise WeightsRejected(f"{file_name}: {fault}") from None
        if digest is not None:
            self.digests[file_name] = (experts * outputs * inputs, digest.hexdigest())

    @staticmethod
    def _hash(digest, integers: torch.Tensor) -> None:
        """Feeds the integers to the digest in row-major order, each as a little-endian
        two's-complement number of its dtype's width: 2 bytes in a matrix, 4 in the embedding."""
        if digest is not None:
            digest.update(integers.cpu().numpy())

    def _integers(self, stored: _Stored, blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """The committed integers of whole rows of a tensor, from their blocks on the device. Shape
        (rows, inputs)."""
        in_a_row = stored.inputs // BLOCK_ELEMENTS
        # The limit is checked while the model is loaded. The stored blocks do not change
        # afterwards, so decoding them again gives the same integers, and the check is not repeated.
        limit = None
        if self._loading:
            limit = MATRIX_LIMIT if dtype == torch.int16 else EMBEDDING_LIMIT
        integers = torch.empty(
            (blocks.shape[0] // in_a_row, stored.inputs), dtype=dtype, device=self.device
        )
        step = max(1, self.WEIGHTS_AT_ONCE // stored.inputs)
        for first in range(0, integers.shape[0], step):
            values = decoded_values(
                blocks[first * in_a_row : (first + step) * in_a_row], stored.storage_type
            )
            integers[first : first + step] = committed_integers(
                values.view(-1, stored.inputs), stored.multipliers, stored.exponent, limit
            )
        return integers

    def matrix(
        self, name: str, layer: int | None = None, expert: int | None = None
    ) -> torch.Tensor:
        if name in _ROUTED:
            stored = self._stored[(name, layer)]
            blocks = self._on_host[(name, layer, expert)].to(self.device)
            return self._integers(
                stored, blocks.view(-1, BLOCK_BYTES[stored.storage_type]), torch.int16
            )
        return self._on_device[(name, layer)]

    def embedding_rows(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.numel() and not 0 <= int(token_ids.min()) <= int(token_ids.max()) < self._p.V:
            raise IndexError(f"A token id is outside 0 .. {self._p.V - 1:,}.")
        stored = self._stored[("Emb", None)]
        blocks = (
            self._embedding[token_ids.cpu()]
            .to(self.device)
            .view(-1, BLOCK_BYTES[stored.storage_type])
        )
        return self._integers(stored, blocks, torch.int32).to(torch.int64)

    @property
    def host_bytes(self) -> int:
        """The host memory that is allocated for the routed experts' blocks."""
        return self._slabs.slabs * _HostSlabs.SLAB_BYTES


def digest_faults(
    digests: dict[str, tuple[int, str]], sums: dict[str, tuple[int, int, str]]
) -> list[str]:
    """Compares BlockWeights.digests with GgufBlocks.sums. Returns one phrase per difference, none
    if all agree. The names in the phrases are those that the model's shape gives, not names read
    from wsums.txt."""
    faults = [
        f"{name} differs from wsums.txt"
        for name in sorted(digests.keys() & sums.keys())
        if digests[name][1] != sums[name][2]
    ]
    faults += [f"{name} has no line in wsums.txt" for name in sorted(digests.keys() - sums.keys())]
    return faults + [
        f"line {number} of wsums.txt lists a tensor that is not a committed tensor of the model"
        for number, name in enumerate(sums, start=1)
        if name not in digests
    ]


def main() -> int:
    """The command-line program that the argparse description below explains. Returns its exit
    code."""
    parser = argparse.ArgumentParser(
        prog="python -m integer_model.block_weights",
        description="Decodes every weight of a model stored as quantized blocks and compares the "
        "SHA-256 of each tensor's integers with the checksums in wsums.txt. Without --fold it "
        "decodes the synthetic model instead and prints a single SHA-256 over all tensors (over "
        "the lines that --lines prints). Exit code: 0 for a match, 1 for a mismatch, 2 if the "
        "model cannot be loaded.",
    )
    parser.add_argument(
        "parameter_file",
        help="the parameter file (policy.txt); the quantization scales (KL.bin) are read from the "
        "same directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="the PyTorch device that decodes (default: cuda if available, otherwise cpu)",
    )
    parser.add_argument(
        "--fold", help="the model's set-up directory, which contains wsums.txt and fold.txt"
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        help="a GGUF file of the model; repeat the option for each file, starting with the model's "
        "first file",
    )
    parser.add_argument(
        "--lines",
        action="store_true",
        help="also print each tensor's name, number of elements and SHA-256",
    )
    parser.add_argument(
        "--no-memory-check",
        action="store_true",
        help="skip the check that enough host memory is free (check_free_memory in the docstring "
        "of BlockWeights)",
    )
    arguments = parser.parse_args()
    if bool(arguments.fold) != bool(arguments.shard):
        parser.error(
            "--fold and --shard go together: give the set-up directory and every GGUF file of "
            "the model, or neither for the synthetic model."
        )
    try:
        parameters = Parameters.read(arguments.parameter_file)
        if arguments.fold:
            source: BlockSource = GgufBlocks(parameters, arguments.fold, arguments.shard)
        else:
            source = SyntheticBlocks(parameters, arguments.device)
        try:
            digests = BlockWeights(
                parameters,
                arguments.device,
                blocks=source,
                hash_integers=True,
                check_free_memory=not arguments.no_memory_check,
            ).digests
        finally:
            if isinstance(source, GgufBlocks):
                source.close()
    except (OSError, ParameterError, WeightsRejected) as fault:
        print(f"error: {fault}", file=sys.stderr)
        return 2
    # For example too little memory. Exit code 1 is reserved for a mismatch.
    except Exception as fault:
        print(f"error: {type(fault).__name__}: {fault}", file=sys.stderr)
        return 2
    lines = [f"{name} {elements} {digest}" for name, (elements, digest) in sorted(digests.items())]
    if arguments.lines:
        print("\n".join(lines))
    if not isinstance(source, GgufBlocks):
        digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
        print(f"{len(lines)} tensors. SHA-256 over the lines that --lines prints: {digest}")
        return 0
    faults = digest_faults(digests, source.sums)
    if not faults:
        print(f"MATCH: all {len(digests)} tensors have the SHA-256 listed in wsums.txt.")
        return 0
    print(
        f"MISMATCH with wsums.txt. Tensors of the model: {len(digests)}. Differences: "
        f"{len(faults)}."
    )
    print("\n".join(faults))
    print(
        "Likely causes: the GGUF files are not those from which wsums.txt was made, or the "
        "set-up directory is another model's."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
