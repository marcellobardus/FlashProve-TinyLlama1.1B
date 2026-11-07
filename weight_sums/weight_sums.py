#!/usr/bin/env python3
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

"""Write a model's sums file: one line per committed tensor, with the SHA-256 of its integers.

The committer program, cp_commit, commits to integer matrices. The conversion rule turns each stored
weight of a model in GGUF format into one integer, and the committed tensors are the tensors that it
covers. README.md next to this script summarizes the rule. This script applies the rule to the
shards (the GGUF files) of one model. cp_commit reads the sums file under the name wsums.txt. It
takes each tensor's exponent e (its scale is 2^e) from that file (the embedding's from fold.txt, as
README.md explains) and compares the hashes of its own integers with the file's.

The script takes the reader of the container and the decoder of the blocks from gguf-py, the Python
package of the llama.cpp project, and does its arithmetic with NumPy. bin/weight_sums applies the
same rule in C++, with a reader and a decoder that are written out in its source file. Both decoders
go back to ggml's public code. The two programs must write the same file, so you can run both and
compare.

Pass all shards in a single run, next to each other, with no option between two of them. The script
takes the width, the head dimension and the number of layers from the metadata of the shard that you
list first. It does not know which tensors a model must have, so it may not notice that you left out
a shard.

The sums file is a function of the weights. Where the weights are secret, keep it as secret as the
weight files. For that reason this script prints names, counts and times, and never a weight or a
number computed from one. An error message can still state a fact about the weights, for example
that a tensor has a value that is not finite.
"""

import argparse
import concurrent.futures
import concurrent.futures.process
import errno
import hashlib
import importlib.metadata
import math
import multiprocessing
import os
import re
import signal
import sys
import threading
import time
from typing import NamedTuple

STATUSES = """exit status:
  0   the sums file is complete
  21  bad arguments, an unreadable shard, an output path that is taken or not writable, or a missing package
  22  a malformed GGUF file, missing metadata, a repeated tensor name, a layer number that is too large, a wrong
      dimension, a norm that a committed tensor needs and that is missing, or an error that this script does not foresee
  23  a tensor of a storage type that the conversion rule does not cover
  24  a value that is not finite
  25  a value outside a bound of the conversion rule, an empty tensor, or a model of another width or head dimension
  26  not enough memory, worker processes that cannot be started, or a worker process that something outside this
      script has ended
After each of the statuses 21 to 26 this script has left no file."""

# Exit status by kind of fault. STATUSES above, which --help prints, describes each status.
# In comments and messages, a matrix is any committed tensor but the embedding, a norm is the weight
# vector (the gains) of an RMS normalization, and a matrix reads a norm if the model applies the
# matrix to that normalization's output.
STATUS = {
    "arguments": 21,
    "package": 21,  # numpy or gguf-py is absent, or gguf-py has a version with which this script is not tested
    "file": 21,  # a shard that cannot be opened or read
    "output path": 21,  # something exists at the output path, or the file cannot be written
    "container": 22,  # the header, the metadata, the tensor table, or the dimensions or the position of a tensor
    "metadata": 22,  # a required key is absent, the architecture is not a string, or the layer count is not a number
    "unexpected": 22,  # an error that this script does not foresee
    "repeated name": 22,
    "layer number": 22,
    "matrix width": 22,  # a matrix that reads a norm and has another number of inputs than the norm has gains
    "norm absent": 22,
    "norm length": 22,
    "norm type": 23,
    "tensor type": 23,
    "value not finite": 24,
    "multiplier not finite": 24,
    "product not finite": 24,
    "model shape": 25,  # a width or a head dimension for which the constants of the rule are not valid
    "multiplier too small": 25,
    "all zero": 25,
    "exponent": 25,
    "embedding integer": 25,
    "matrix integer": 25,  # possible only if a shard changes during the run
    "memory": 26,
    "processes": 26,  # the system cannot start the worker processes
}


class Fault(Exception):
    """A reason to stop: a key of STATUS and a message for the user."""

    def __init__(self, kind, message):
        super().__init__(kind, message)  # both go into args, so that a Fault can travel from a worker process
        self.status, self.message = STATUS[kind], message


def say(text):
    """Print a message to standard error. If that is closed or a write fails, go on silently."""
    try:
        if sys.stderr is not None:
            print("weight_sums.py: " + text, file=sys.stderr, flush=True)
    except OSError:
        sys.stderr = None


try:
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFEndian, GGUFReader, GGUFValueType
    from gguf.quants import dequantize
except ImportError as error:
    say(f"{error}. This script needs two packages. Install them with: pip install numpy==2.3.5 gguf==0.19.0")
    sys.exit(STATUS["package"])

np.seterr(all="ignore")  # a result that is not finite is a Fault with a status, not a warning
GGUF_PY_VERSION = "0.19.0"  # the only gguf-py version tested (by weight_sums/tests/run_tests.sh)

# ---------------------------------------------------------------------------------------------------------------------
# The conversion rule's constants.
# The rule is defined in docs/model.md, "11 From the model's files to the integers".
# The bounds EXPONENTS and SMALLEST_MULTIPLIER are stated in
# weight_sums/README.md, "Make the file", in the table of exit statuses (status 25).
#
# The rule grew out of VerInf's rule for converting stored weights to integers: J. Petrie, VerInf,
# source code and README, github.com/JamesPetrie/VerInf, commit 89903c96 (28 August 2026), MIT License. The
# present text of this script has not been compared with VerInf's code; earlier forms were. THIRD_PARTY.md,
# "VerInf", lists the files that apply the rule, and NOTICE has VerInf's licence notice.
#
# After decoding, the rule has two steps, both common in neural-network quantization.
# 1. The gains of an RMS norm are multiplied into the matrices that read the norm's output, so that
#    the norms of the integer model need no gains. See S. Ashkboos et al., "QuaRot: Outlier-Free
#    4-Bit Inference in Rotated LLMs", arXiv:2404.00456, sections 3.4 and 4, and S. Ashkboos et al.,
#    "SliceGPT: Compress Large Language Models by Deleting Rows and Columns", arXiv:2401.15024,
#    sections 2.1 and 3.2, where the factor sqrt(width) is also multiplied in with the gains.
# 2. Each tensor is quantized symmetrically with one power-of-two scale. See M. Nagel et al., "A
#    White Paper on Neural Network Quantization", arXiv:2106.08295, section 2.2. The scale is chosen
#    from the tensor's largest absolute value (the embedding's is set by --eres), and the rounding
#    is to nearest, ties to even. See S. R. Jain et al., "Trained Quantization Thresholds for
#    Accurate and Efficient Fixed-Point Inference of Deep Neural Networks", arXiv:1903.08066,
#    sections 4.2 and 3.2, and for the largest absolute value also H. Wu et al., "Integer
#    Quantization for Deep Learning Inference: Principles and Empirical Evaluation",
#    arXiv:2004.09602, section 3.4.

# The reference shape. The conversion rule's two constants are a function of the width
# (embedding_length in GGUF) and the head dimension; fold_constants() computes them for the model's
# actual shape. HEAD_DIMENSION is the default when a shard's metadata omit attention.key_length.
WIDTH = 5120
HEAD_DIMENSION = 128


def fold_constants(width, head_dimension):
    """The two gain-folding constants (c_0, c_q) as binary32 values, for a model of this width and
    head dimension, by FlashProve's own rule (scales/README.md, "The constants of a norm").

    Let t be the largest integer with 4^t <= width, and c = sqrt(width) / 2^t, so that
    sqrt(width) = c * 2^t and 1 <= c < 2. Then c_0 = RN32(c) folds a norm's gains and sqrt(width)
    into a matrix, and c_q = RN32(c / sqrt(head_dimension)) also carries attention's scale
    1 / sqrt(head_dimension). At width 5,120 and head 128 these are 0x3F8F1BBD and 0x3DCA62C2; at
    width 2,048 and head 64 they are sqrt(2) and sqrt(2) / 8, 0x3FB504F3 and 0x3E3504F3
    (docs/tinyllama/ARITHMETIC_CONTRACT.md, section 4.3). This generalizes the former literals,
    which were right for width 5,120 and head 128 only; at that shape the values are unchanged.
    """
    if width < 1 or head_dimension < 1:
        raise Fault("model shape", "the width and the head dimension must each be at least 1")
    t = 0
    while 4 ** (t + 1) <= width:
        t += 1
    c = math.sqrt(width) / (2**t)  # the exact float64 value; np.float32 rounds it once, ties to even
    return np.float32(c), np.float32(c / math.sqrt(head_dimension))


# A layer's matrices are named blk.<layer>.<kind>.weight, with <layer> in decimal and no leading
# zero. Each kind maps to the norm of the same layer whose output it reads, and to which of the two
# constants is multiplied in with that norm's gains: "query" for c_q (the query matrix carries the
# attention scale), "norm" for c_0. None: the matrix reads no norm.
LAYER_MATRICES = {
    "attn_q": ("attn_norm", "query"),
    "attn_k": ("attn_norm", "norm"),
    "attn_v": ("attn_norm", "norm"),
    "attn_output": None,
    "ffn_gate": ("ffn_norm", "norm"),
    "ffn_up": ("ffn_norm", "norm"),
    "ffn_down": None,
    "ffn_gate_inp": ("ffn_norm", "norm"),
    "ffn_gate_shexp": ("ffn_norm", "norm"),
    "ffn_up_shexp": ("ffn_norm", "norm"),
    "ffn_down_shexp": None,
    "ffn_gate_exps": ("ffn_norm", "norm"),
    "ffn_up_exps": ("ffn_norm", "norm"),
    "ffn_down_exps": None,
}
LAYER_NORMS = ("attn_norm", "ffn_norm")
LAYER_NAME = re.compile(r"blk\.(0|[1-9][0-9]*)\.([a-z_]+)\.weight")
EMBEDDING = "token_embd.weight"
HEAD, HEAD_NORM = "output.weight", "output_norm.weight"

# The storage types that the rule covers. select() rejects a committed tensor of any other type.
# The names are the package's; products() says which of its routines decodes them.
READ_TYPES = (
    GGMLQuantizationType.F32,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.F16,
    GGMLQuantizationType.BF16,
)

LARGEST_MATRIX_INTEGER = 4095  # the integers of a matrix fill 13 signed bits
EMBEDDING_BOUND = 2**30  # every integer w of the embedding has |w| below this
EXPONENTS = range(-100, 101)  # the exponents that the rule allows, also for --eres
SMALLEST_MULTIPLIER = np.float32(2.0**-100)  # a multiplier is 0 or at least this in absolute value

# products() decodes about this many elements at a time. This affects time and memory only.
SLICE_ELEMENTS = 1 << 22

# ---------------------------------------------------------------------------------------------------------------------
# The GGUF container, read with gguf-py's GGUFReader. The comment at the top of ggml/include/gguf.h describes the format
# (llama.cpp as carried in llama-cpp-python 0.3.7, whose changelog names the commit
# 794fe23f29fb40104975c91fe19f23798f7c726e). The format's specification is docs/gguf.md in github.com/ggml-org/ggml.


class Tensor(NamedTuple):
    """One entry of a shard's tensor table."""

    name: str
    entry: int  # the tensor's number in the shard's tensor table, from 1
    storage: GGMLQuantizationType  # the type in which the file stores the elements
    dims: tuple  # as stored: elements that differ by 1 in the first dimension are neighbours in the file
    elements: int
    path: str  # the shard that holds the tensor
    offset: int  # the position of the tensor's first byte in the shard
    length: int  # the number of the tensor's bytes

    def rows(self):
        """Memory-map the tensor's bytes as a 2-D uint8 array, one row per dims[0] elements."""
        count = self.elements // self.dims[0]
        try:
            return np.memmap(self.path, np.uint8, "r", self.offset, (count, self.length // count))
        except (OSError, ValueError) as error:  # ValueError: the file has become shorter
            raise unreadable(self.path, error) from None


def unreadable(path, error):
    """Return the exception to raise for an OSError or numpy.memmap's ValueError on a shard."""
    if isinstance(error, OSError) and error.errno == errno.ENOMEM:
        return MemoryError()
    why = error.strerror if isinstance(error, OSError) else "the file has changed during the run"
    return Fault("file", f"cannot read {path}: {why}")


class Malformed(Exception):
    """A defect that this script found in a shard. read_shard() quotes its message in a Fault."""


class CheckedReader(GGUFReader):
    """A GGUFReader that raises Malformed at the first item that runs past the end of the file.

    GGUFReader silently shortens a read that runs past the end of the file. Given a huge array length, it then loops
    once for each claimed element. The two methods below override internal methods of gguf 0.19.0
    (gguf/gguf_reader.py).
    """

    def _get(self, offset, dtype, count=1, *more, **named):
        found = super()._get(offset, dtype, count, *more, **named)
        if len(found) != count:
            raise Malformed(f"the file ends inside the item that starts at byte {offset}")
        return found

    def _get_field_parts(self, orig_offs, raw_type):
        # An array starts with its element type (4 bytes) and its element count (8 bytes). An
        # element takes at least one byte.
        if raw_type == GGUFValueType.ARRAY and self._get(orig_offs + 4, np.uint64)[0] > len(self.data):
            raise Malformed(f"the array at byte {orig_offs} is longer than the file")
        return super()._get_field_parts(orig_offs, raw_type)


def model_layers(reader):
    """Read the first shard's metadata, and return (number of layers, width, head dimension)."""

    # Messages print <architecture> instead of the architecture's name, as bin/weight_sums does:
    # text from the file can contain terminal control sequences.
    note = "<architecture> is the value of the key general.architecture"

    def value(key, default=None, shown=None):
        field = reader.get_field(key)
        if field is not None:
            return field.contents()
        if default is None:
            named = f"{shown}. {note}" if shown else key
            raise Fault(
                "metadata", f"the first shard has no metadata key {named}. Name the model's first file first"
            )
        return default

    architecture = value("general.architecture")
    if not isinstance(architecture, str):
        raise Fault("metadata", "the metadata key general.architecture of the first shard is not a string")

    def number(name, default=None):
        """Return the value of the key <architecture><name>, of whatever type it is."""
        return value(architecture + name, default, "<architecture>" + name)

    width = number(".embedding_length")
    layers = number(".block_count")
    head_dimension = number(".attention.key_length", HEAD_DIMENSION)
    if not isinstance(layers, (int, float)):
        raise Fault(
            "metadata", f"the metadata key <architecture>.block_count of the first shard is not a number. {note}"
        )
    # The conversion rule's constants (fold_constants) are a function of the width and head
    # dimension, so the rule is defined for any positive whole width and head dimension, not only
    # width 5,120 and head 128. A scale fit (scales/README.md) may still need a narrower range.
    for what, found in (("width", width), ("head dimension", head_dimension)):
        if not isinstance(found, (int, float)) or found < 1 or found != int(found):
            raise Fault(
                "model shape",
                f"the first shard's metadata give a {what} that is not a positive whole number. {note}",
            )
    return layers, int(width), int(head_dimension)


def read_shard(path, is_first):
    """Return the Tensors of a shard and, for the first shard, the model's number of layers (otherwise None)."""
    try:
        reader = CheckedReader(path)
        # GGUFReader also accepts GGUF version 2, and it byte-swaps big-endian files.
        if reader.get_field("GGUF.version").contents() != 3 or reader.endianess != GGUFEndian.LITTLE:
            raise Malformed("it is not a little-endian file of version 3")
        tensors = []
        # In a damaged file, the bytes parsed as a tensor's name can be bytes of a weight. So a
        # message quotes a name only after select() has matched it to a name of the rule, and
        # otherwise gives the entry's number.
        for number, entry in enumerate(reader.tensors, 1):
            dims = tuple(entry.shape.tolist())
            if len(dims) > 4:  # GGML_MAX_DIMS in ggml/include/ggml.h, in the copy of llama.cpp that is named above
                raise Malformed(f"entry {number} of the tensor table has {len(dims)} dimensions")
            if entry.data_offset < reader.data_offset:  # GGUFReader adds the offset to the data's start modulo 2^64
                raise Malformed(f"the offset in entry {number} of the tensor table is too large")
            place = (path, entry.data_offset, entry.n_bytes)
            tensors.append(Tensor(entry.name, number, entry.tensor_type, dims, entry.n_elements, *place))
        return tensors, model_layers(reader) if is_first else None
    except (Fault, MemoryError):
        raise
    except OSError as error:
        raise unreadable(path, error) from None
    except Malformed as error:
        raise Fault("container", f"{path} is not a well-formed GGUF file: {error}") from None
    except Exception as error:
        # GGUFReader raises exceptions of many classes on a malformed file. Their messages can quote
        # bytes of the file.
        raise Fault("container", f"{path} is not a well-formed GGUF file (gguf-py: {type(error).__name__})") from None


# ---------------------------------------------------------------------------------------------------------------------
# Which tensors the rule uses.


def select(tensors, layers, width, norm_constant, query_constant):
    """Return the list of committed tensors and a dict of the norms by name.

    A list entry is (tensor, name of the norm that it reads, constant). The last two are None for a
    tensor that reads no norm. norm_constant and query_constant are c_0 and c_q from
    fold_constants() for this model's width and head dimension; width is its hidden width. Tensors
    with other names are ignored, except that no name may repeat.
    """
    committed, norms = [], {}
    names = set()
    for tensor in tensors:
        if tensor.name in names:
            raise Fault(
                "repeated name",
                f"entry {tensor.entry} of the tensor table of {tensor.path} repeats the name of an earlier tensor",
            )
        names.add(tensor.name)

        in_layer = LAYER_NAME.fullmatch(tensor.name)
        kind = in_layer[2] if in_layer else None
        if tensor.name == HEAD_NORM or kind in LAYER_NORMS:
            norms[tensor.name] = tensor
        elif tensor.name == HEAD:
            committed.append((tensor, HEAD_NORM, norm_constant))
        elif tensor.name == EMBEDDING or (kind in LAYER_MATRICES and LAYER_MATRICES[kind] is None):
            committed.append((tensor, None, None))
        elif kind in LAYER_MATRICES:
            norm_kind, which = LAYER_MATRICES[kind]
            constant = query_constant if which == "query" else norm_constant
            committed.append((tensor, f"blk.{in_layer[1]}.{norm_kind}.weight", constant))
        else:
            continue
        # A number of more than 20 digits exceeds every integer that a GGUF file can store. The
        # length test comes first because int() raises ValueError above its limit on digits.
        if in_layer and (len(in_layer[1]) > 20 or int(in_layer[1]) >= layers):
            # A damaged or crafted name can have as many digits as the file has bytes, so the
            # message gives a long name's place in the table instead of the name.
            place = f"entry {tensor.entry} of the tensor table of {tensor.path}"
            shown = tensor.name if len(in_layer[1]) <= 20 else place
            raise Fault("layer number", f"the layer number of {shown} is not below the first shard's block_count")

    for norm in norms.values():
        if norm.storage != GGMLQuantizationType.F32:
            raise Fault("norm type", f"the norm {norm.name} is not of type F32")
        if norm.elements != width:
            raise Fault("norm length", f"the norm {norm.name} does not have {width} elements")
    for tensor, norm, _ in committed:
        if not tensor.dims:
            raise Fault("container", f"{tensor.name} has no dimensions")
        if tensor.storage not in READ_TYPES:
            raise Fault("tensor type", f"{tensor.name} is not of type F32, F16, BF16, Q8_0, Q4_K, Q5_K or Q6_K")
        if norm is not None and norm not in norms:
            raise Fault("norm absent", f"{tensor.name} needs {norm}, which is in none of the shards")
        if norm is not None and tensor.dims[0] != width:
            raise Fault("matrix width", f"{tensor.name} reads a norm, but its first dimension is not {width}")
        if tensor.elements == 0:
            raise Fault("all zero", f"{tensor.name} has no elements")
    return committed, norms


# ---------------------------------------------------------------------------------------------------------------------
# The arithmetic of the rule.
#
#   RN32(x)  the binary32 number nearest to x, ties to even
#   RNE(x)   the integer nearest to x, ties to even
#   v        the decoded value of an element, defined operation by operation in
#            docs/model.md, "11.2 Stored blocks and their values"
#   g        the gain of the norm that the matrix reads, at the element's index along dims[0]
#   c        the constant of the matrix's kind
#   gamma    the multiplier, RN32(g * c)
#   p        the product, RN32(gamma * v), or v if the tensor reads no norm
#   e        the tensor's exponent
#   w        the integer, RNE(2^e * p)
#
# Every array of v, g, gamma or p is float32, and NumPy applies an operation on two float32 operands
# as one IEEE 754 binary32 operation to each element. Nothing is fused.


def multipliers(norm, constant):
    """Return gamma_j = RN32(g_j * c) for the gains g of a norm."""
    gamma = dequantize(norm.rows(), norm.storage).reshape(-1) * constant
    if not np.isfinite(gamma).all():
        raise Fault("multiplier not finite", f"a gain of {norm.name}, times the rule's constant, is not finite")
    if ((gamma != 0) & (np.abs(gamma) < SMALLEST_MULTIPLIER)).any():
        raise Fault(
            "multiplier too small",
            f"a gain of {norm.name}, times the rule's constant, is not 0 and is below 2^-100 in absolute value",
        )
    return gamma


def products(tensor, gamma):
    """Yield a tensor's products in file order: (p, largest |p|) for each slice of whole rows.

    The decoded values v come from gguf-py. Its dequantize() evaluates d * sc, then (d * sc) * q - dmin * m for Q4_K and
    Q5_K, and d * s, then (d * s) * q for Q6_K, all in float32 (gguf 0.19.0, gguf/quants.py, the methods
    dequantize_blocks of the classes Q4_K, Q5_K and Q6_K). The rule fixes this order. Here
    d and dmin are the two scales of a block, sc and m (in Q6_K: s) the scale and the minimum of a sub-block, and q the
    code of an element, as ggml names them. The block formats and their decoding rule are those of ggml (THIRD_PARTY.md,
    section "ggml"). For Q8_0 it evaluates q * d in float32 (the class Q8_0), which is exact. For F16 it converts the
    stored float16 numbers to float32 (dequantize() itself), and for BF16 it shifts the stored 16 bits left by 16 and
    views them as float32 (the class BF16); both are exact.
    """
    rows = tensor.rows()
    step = max(1, SLICE_ELEMENTS // tensor.dims[0])
    for start in range(0, len(rows), step):
        v = dequantize(rows[start : start + step], tensor.storage)
        p = v if gamma is None else v * gamma
        largest = np.abs(p).max()  # a NaN or an infinity among the p shows up here
        if not np.isfinite(largest):
            if np.isfinite(v).all():
                raise Fault(
                    "product not finite",
                    f"a weight of {tensor.name}, times its norm's gain and the rule's constant, overflows binary32",
                )
            raise Fault("value not finite", f"a decoded value of {tensor.name} is not finite")
        yield p, float(largest)


def exponent_of(largest):
    """Return the maximum e with RNE(2^e * largest) <= 4095, for a binary32 number `largest` > 0.

    RNE rounds the tie 4095.5 to the even number 4096, so the condition is 2^e * largest < 4095.5.
    The products with powers of two are exact in Python's binary64 floats.
    """
    e = 12 - math.frexp(largest)[1]  # 2048 <= 2^e * largest < 4096
    if math.ldexp(largest, e) >= LARGEST_MATRIX_INTEGER + 0.5:
        e -= 1
    return e


def convert(tensor, gamma, eres, decode_twice):
    """Return the line of the sums file for one committed tensor, and the seconds that it took."""
    began = time.monotonic()
    slices = products(tensor, gamma)
    if tensor.name == EMBEDDING:
        e, integer_type = eres, "<i4"
    else:
        if not decode_twice:
            slices = list(slices)
        largest = max(most for _, most in slices)
        if decode_twice:
            slices = products(tensor, gamma)
        if largest == 0:
            raise Fault("all zero", f"every product of {tensor.name} is 0")
        e, integer_type = exponent_of(largest), "<i2"
        if e not in EXPONENTS:
            raise Fault("exponent", f"the exponent of {tensor.name} is outside -100 .. 100")

    # w = RNE(2^e * p). binary64 represents 2^e * p exactly for every binary32 p and every e in
    # EXPONENTS, so numpy.rint does the only rounding. It rounds ties to the even integer (numpy
    # 2.3.5, docstring of numpy.rint, Notes). The SHA-256 is over the integers in little-endian
    # two's complement: 2 bytes each for a matrix and 4 bytes each for the embedding. FIPS 180-4
    # defines SHA-256. An integer that does not fit makes the hash meaningless, and the checks after
    # the loop then raise a Fault.
    scale = math.ldexp(1.0, e)
    lowest = highest = 0
    sha256 = hashlib.sha256()
    for p, _ in slices:
        w = p.astype(np.float64)
        w *= scale
        np.rint(w, out=w)
        lowest, highest = min(lowest, int(w.min())), max(highest, int(w.max()))
        sha256.update(w.astype(integer_type).tobytes())

    # These checks come after the loop, so that a non-finite value anywhere in the tensor is
    # reported instead.
    if tensor.name == EMBEDDING and max(-lowest, highest) >= EMBEDDING_BOUND:
        raise Fault(
            "embedding integer", f"an integer of {tensor.name} is 2^30 or more in absolute value. Use a lower --eres"
        )
    if tensor.name != EMBEDDING and max(-lowest, highest) > LARGEST_MATRIX_INTEGER:
        raise Fault("matrix integer", f"the values of {tensor.name} have changed during the run")

    # The smallest two's-complement width that fits every integer.
    bits = 1
    while not -(2 ** (bits - 1)) <= lowest <= highest < 2 ** (bits - 1):
        bits += 1
    return f"{tensor.name} {tensor.elements} {e} {bits} {sha256.hexdigest()}\n", time.monotonic() - began


def start_worker():
    """Make a worker exit at once on Ctrl-C, and as soon as the main process ends for any reason."""
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    def end_with_the_main_process():
        multiprocessing.parent_process().join()
        os._exit(1)

    threading.Thread(target=end_with_the_main_process, daemon=True).start()


# After a fault, convert_all() keeps the pool of workers here, so that the garbage collector does
# not free it before the script ends.
KEPT = []


def convert_all(committed, norms, eres, workers, decode_twice):
    """Return the sums file's tensor lines, sorted by name as bytes."""
    if not committed:
        return []
    gammas = {}
    for _, norm, constant in committed:
        if norm is not None and (norm, constant) not in gammas:
            gammas[norm, constant] = multipliers(norms[norm], constant)

    lines = {}
    # The start method is set explicitly to fork, with which every test runs. Under forkserver
    # (Python 3.14's default on Linux), the resource tracker warns of leaked semaphores after
    # os._exit(). The fork server also reaps a dead worker at once, so the SIGTERM that the pool
    # sends after a worker has died could reach another process that has the freed process ID.
    # This script has not been run on macOS, where Python's manual calls fork unsafe.
    fork = multiprocessing.get_context("fork")
    pool = concurrent.futures.ProcessPoolExecutor(
        min(workers, len(committed)), mp_context=fork, initializer=start_worker
    )
    try:
        # The largest tensors go first, so that the workers finish at about the same time. Tensors of equal size
        # keep the order of the shards and of their tensor tables.
        waiting = {
            pool.submit(convert, tensor, gammas.get((norm, constant)), eres, decode_twice): tensor
            for tensor, norm, constant in sorted(committed, key=lambda entry: entry[0].elements, reverse=True)
        }
        # The results are taken in the same order. So if several tensors have faults, the first of them in this
        # order decides the exit status and the message, whichever worker finishes first.
        for result, tensor in waiting.items():
            lines[tensor.name], seconds = result.result()
            say(f"{len(lines)} of {len(committed)}: {tensor.name}, {tensor.elements} elements, {seconds:.1f} s")
    except BaseException:
        # This script signals no worker: the pool's management thread may already have reaped a
        # worker that died, and the worker's process ID may then belong to another process. Each
        # worker ends by itself once this process has ended (see start_worker()). Waiting for the
        # workers here could last forever, because a worker waits for a task. Freeing the pool while
        # its management thread is alive can hang the exit: freeing runs a callback that takes one
        # of the pool's locks, in whichever thread the garbage collector is running. So the pool
        # stays in KEPT and is never freed, and the script ends with os._exit(), which waits for no
        # worker (at the end of this file).
        KEPT.append(pool)
        raise
    pool.shutdown()  # waits for the workers, which have finished every task
    return [lines[name] for name in sorted(lines, key=str.encode)]


# ---------------------------------------------------------------------------------------------------------------------
# The output path. This script never overwrites a file. It removes only <out>.tmp, its temporary
# file, and only after checking that the name still leads to the file that this run created.
# <out>.tmp can remain if a signal kills the script during the final write, or if that check or the
# removal fails. The file system at <out> must support hard links.


def check_output_path(out):
    if os.path.basename(out) in ("", ".", ".."):
        raise Fault("output path", "--out does not name a file")
    if os.path.lexists(out):
        # The message must not advise the user to remove this file. If the new file's name is
        # missing after --out, argparse has taken the shard that follows as the output path, so this
        # file is one of the model's files.
        raise Fault(
            "output path",
            f"{out} exists, and this script does not overwrite files. Give --out the name of a new file. "
            f"If {out} is one of the model's files, the new file's name is missing after --out",
        )
    if os.path.lexists(out + ".tmp"):
        raise Fault(
            "output path",
            f"{out}.tmp exists. This script needs that name for its temporary file and does not overwrite files. "
            "Give --out another name",
        )
    folder = os.path.dirname(out) or "."
    if not os.path.isdir(folder) or not os.access(folder, os.W_OK | os.X_OK):
        raise Fault("output path", f"{folder} is not a folder in which a file can be created")


def write_sums_file(out, lines):
    """Write the file under the name <out>.tmp, and give it the name <out> once it is complete."""
    made = False
    try:
        # O_EXCL: the call fails if the name is taken. Mode 0600: anyone who can read the file can
        # test any guess at a tensor against it.
        handle = os.open(out + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        made = os.fstat(handle)
        with open(handle, "w", encoding="ascii", newline="") as file:
            file.write("".join(lines) + f"END {len(lines)}\n")
        os.link(out + ".tmp", out)  # unlike os.rename, this fails if something has appeared at <out>
    except OSError as error:
        raise Fault("output path", f"cannot write {out}: {error.strerror}") from None
    finally:
        # If <out>.tmp has been moved away or another file now has its name, nothing is removed. An
        # instant remains between this check and the removal.
        try:
            if made and os.path.samestat(made, os.lstat(out + ".tmp")):
                os.unlink(out + ".tmp")
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------------------
# The command line. Parser turns argparse's errors into Faults, so that bad arguments give exit
# status 21, not argparse's 2.


class Parser(argparse.ArgumentParser):
    def error(self, message):
        if message.startswith("unrecognized arguments"):  # also if an option separates two shards
            message += ". Name all shards together, before or after the options"
        raise Fault("arguments", f"{message}\n{self.format_usage().rstrip()}")


def read_arguments(words):
    parser = Parser(
        prog="weight_sums.py",
        description=__doc__,
        epilog=STATUSES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eres", type=int, default=11, metavar="<integer>",
        help="e_res, the exponent of token_embd.weight, from -100 to 100 (default: 11). README.md says what it stands "
        "for",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="<integer>",
        help="how many worker processes convert tensors at the same time, at least 1 (default: 8)",
    )
    parser.add_argument(
        "--out", required=True, metavar="<new file>",
        help="where the sums file goes. Nothing may exist there, or at <new file>.tmp",
    )
    parser.add_argument(
        "--decode-twice", action="store_true",
        help="use less memory and more time. Without this flag a worker keeps 4 bytes for each element of the matrix "
        "that it converts. The output is the same",
    )
    parser.add_argument(
        "--any-gguf-version", action="store_true",
        help=f"run with an installed gguf-py other than {GGUF_PY_VERSION}, the only version with which this script is "
        "tested",
    )
    parser.add_argument("shards", nargs="+", metavar="<shard.gguf>", help="the GGUF files of the model")
    given = parser.parse_args(words)
    if given.eres not in EXPONENTS:
        parser.error("argument --eres: the value must be between -100 and 100")
    if given.workers < 1:
        parser.error("argument --workers: the value must be at least 1")
    return given


def check_gguf_py(any_version):
    try:
        installed = importlib.metadata.version("gguf")
    except importlib.metadata.PackageNotFoundError:
        installed = "unknown"
    if installed != GGUF_PY_VERSION and not any_version:
        raise Fault(
            "package",
            f"the installed gguf-py has version {installed}, and this script is tested with {GGUF_PY_VERSION} only. "
            f"Install that version with: pip install gguf=={GGUF_PY_VERSION}. "
            "To run with the installed version anyway, add --any-gguf-version",
        )


def main(words):
    began = time.monotonic()
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # Ctrl-C ends the script at once, as SIGTERM does
    # A caller that ignores SIGCHLD passes that setting on to this script. The kernel would then
    # reap each worker as soon as it ends, and multiprocessing would list the worker as running
    # forever. The system could give the worker's process ID to another process, and the SIGTERM
    # that the pool itself sends to its workers after one of them has died would reach that process.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        given = read_arguments(words)
        check_gguf_py(given.any_gguf_version)
        check_output_path(given.out)

        tensors, (layers, width, head_dimension) = read_shard(given.shards[0], True)
        for path in given.shards[1:]:
            tensors += read_shard(path, False)[0]
        norm_constant, query_constant = fold_constants(width, head_dimension)
        committed, norms = select(tensors, layers, width, norm_constant, query_constant)
        say(f"shards: {len(given.shards)}, tensors: {len(tensors)}, committed tensors: {len(committed)}")

        lines = convert_all(committed, norms, given.eres, given.workers, given.decode_twice)
        write_sums_file(given.out, lines)
    except Fault as fault:
        say(fault.message)
        return fault.status
    except (MemoryError, concurrent.futures.process.BrokenProcessPool):
        # If the system kills a worker for using too much memory, result() raises BrokenProcessPool.
        say(
            "not enough memory, or something outside this script has ended a worker process. "
            "A run takes less memory with fewer --workers and with --decode-twice"
        )
        return STATUS["memory"]
    except OSError as error:  # errors at a shard or at the output path are Faults by now
        say(f"cannot start the worker processes: {error.strerror}. Try fewer --workers")
        return STATUS["processes"]
    except Exception as error:  # the text of an unforeseen error could quote a value, so only its class is printed
        say(
            f"stopped by an unexpected error of the class {type(error).__name__}. "
            "This is a fault of the script, not of your files. Please report it"
        )
        return STATUS["unexpected"]
    say(f"wrote {given.out} (tensor lines: {len(lines)}) in {time.monotonic() - began:.1f} s")
    return 0


if __name__ == "__main__":
    status = main(sys.argv[1:])
    if KEPT:  # workers may still be running after a fault: end without waiting for them (convert_all)
        os._exit(status)  # say() has flushed each line, and nothing goes to standard output on that path
    sys.exit(status)
