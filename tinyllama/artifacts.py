"""Writer, validator and loader for the I2 TL-INT-1 artifact format.

QUESTION: I2.4 gives no code for a caller supplied expected_id that differs from
the manifest hash.  The task asks for A07 on manifest.json; that is what
load_artifact uses.

QUESTION: I2.4 says every rule runs, and also gives A10 its own KL.bin length
rule while A06/A08 cover the same bytes.  When KL.bin has the wrong length this
module reports A10 and suppresses A06/A08 for that one file, so that the
dedicated rule (and the code load_artifact raises) is A10.  All other files keep
the A04/A06/A07 skip rule stated in the task.

QUESTION: The exact JSON key spelling of "every constant of I1" is not fixed by
I2.3.  The canonical spelling below follows the symbols in INTERFACES.md I1
(L, d, H, G, dh, f, V, T and the other lowercase names).  Strict single
spelling is chosen: an alias key such as "D", "DH" or "EXP_S" is itself A15,
with detail "non canonical constant name".

QUESTION: I2.4 has no code for malformed exponents.json.  As instructed, a file
that is not canonical JSON is reported as A08 with detail
"exponents.json not parseable", not as A01 (manifest only) or A15 (params only).
"""

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyllama import contract


@dataclass(frozen=True)
class Dims:
    L: int
    D: int
    H: int
    G: int
    DH: int
    F: int
    V: int
    T: int


FULL = Dims(
    contract.L,
    contract.D,
    contract.H,
    contract.G,
    contract.DH,
    contract.F,
    contract.V,
    contract.T,
)

CODES = {"A%02d" % number: number for number in range(1, 17)}
CODE_NAMES = {
    "A01": "manifest-parse",
    "A02": "format",
    "A03": "path",
    "A04": "missing",
    "A05": "unlisted",
    "A06": "size",
    "A07": "sha256",
    "A08": "shape",
    "A09": "weight-range",
    "A10": "kl-shape",
    "A11": "kl-shift",
    "A12": "kl-norm",
    "A13": "kl-rp",
    "A14": "kl-sh",
    "A15": "params",
    "A16": "table",
}
NAMES = CODE_NAMES

ITEMSIZE = {"u32": 4, "i16": 2, "i32": 4, "json": 0}
NUMPY_DTYPE = {"u32": "<u4", "i16": "<i2", "i32": "<i4"}
TABLE_PATHS = {
    "rope_c": "tables/rope_c.i32",
    "rope_s": "tables/rope_s.i32",
    "softmax_ex": "tables/softmax_ex.i32",
    "silu_ts": "tables/silu_ts.i32",
}
MATRIX_NAMES = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")

# Contract 5.2 / D16: for TL-INT-1 every shift has digit form 2, whose
# ceiling is 32.  A proof profile that fixes forms must supply its own
# per-class ceilings.
SHIFT_CEILINGS = {
    "kq": 32,
    "kk": 32,
    "kv": 32,
    "ko": 32,
    "kg": 32,
    "ku": 32,
    "kd": 32,
}


class ArtifactRejected(ValueError):
    """An artifact violates one of the I2.4 rules."""

    def __init__(self, code, path, detail):
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(
            "{} {} {}: {}".format(code, CODE_NAMES.get(code, code), path, detail)
        )


class _JSONProblem(ValueError):
    pass


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(obj):
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n"
    return text.encode("ascii")


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _JSONProblem("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


def _reject_constant(value):
    raise _JSONProblem("non JSON constant {}".format(value))


def _parse_canonical_json(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _JSONProblem("not UTF-8") from exc
    try:
        obj = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except _JSONProblem:
        raise
    except json.JSONDecodeError as exc:
        raise _JSONProblem("not JSON: {}".format(exc.msg)) from exc
    canonical = _canonical_json_bytes(obj)
    if canonical != raw:
        raise _JSONProblem("not canonical JSON")
    return obj


def _expected_files(dims):
    files = {}
    files["KL.bin"] = ("u32", (dims.L, 13))
    files["emb.i32"] = ("i32", (dims.V, dims.D))
    files["wh.i16"] = ("i16", (dims.V, dims.D))
    for layer in range(dims.L):
        ll = "%02d" % layer
        files["layers/%s.wq.i16" % ll] = ("i16", (dims.H * dims.DH, dims.D))
        files["layers/%s.wk.i16" % ll] = ("i16", (dims.G * dims.DH, dims.D))
        files["layers/%s.wv.i16" % ll] = ("i16", (dims.G * dims.DH, dims.D))
        files["layers/%s.wo.i16" % ll] = ("i16", (dims.D, dims.H * dims.DH))
        files["layers/%s.wg.i16" % ll] = ("i16", (dims.F, dims.D))
        files["layers/%s.wu.i16" % ll] = ("i16", (dims.F, dims.D))
        files["layers/%s.wd.i16" % ll] = ("i16", (dims.D, dims.F))
    files["tables/rope_c.i32"] = ("i32", (dims.T, dims.DH // 2))
    files["tables/rope_s.i32"] = ("i32", (dims.T, dims.DH // 2))
    files["tables/softmax_ex.i32"] = ("i32", (contract.U0,))
    files["tables/silu_ts.i32"] = ("i32", (1 << 16,))
    files["params.json"] = ("json", ())
    files["exponents.json"] = ("json", ())
    return files


def _itemsize(dtype):
    return ITEMSIZE.get(dtype)


def _product(shape):
    try:
        return math.prod(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _dims_constants(dims):
    return [
        ("L", dims.L),
        ("d", dims.D),
        ("H", dims.H),
        ("G", dims.G),
        ("dh", dims.DH),
        ("f", dims.F),
        ("V", dims.V),
        ("T", dims.T),
    ]


def _other_constants():
    return [
        ("NE", contract.NE),
        ("lt", contract.LT),
        ("cs", contract.CS),
        ("theta0", contract.THETA0),
        ("S", contract.S),
        ("rx", contract.RX),
        ("eg", contract.EG),
        ("smax", contract.SMAX),
        ("cw", contract.CW),
        ("weight_bound", contract.WEIGHT_BOUND),
        ("embedding_bound", contract.EMB_BOUND),
        ("u0", contract.U0),
        ("c0_bits", contract.C0_BITS),
        ("cq_bits", contract.CQ_BITS),
        ("pad_id", contract.PAD_ID),
        ("gcl", contract.GCL),
    ]


def _all_constants(dims):
    return _dims_constants(dims) + _other_constants()


NON_CANONICAL_CONSTANT_NAMES = {
    "D": "d",
    "DH": "dh",
    "F": "f",
    "LT": "lt",
    "CS": "cs",
    "THETA0": "theta0",
    "EXP_S": "S",
    "RX": "rx",
    "EG": "eg",
    "SMAX": "smax",
    "CW": "cw",
    "WEIGHT_BOUND": "weight_bound",
    "EMB_BOUND": "embedding_bound",
    "U0": "u0",
    "C0_BITS": "c0_bits",
    "CQ_BITS": "cq_bits",
    "PAD_ID": "pad_id",
    "GCL": "gcl",
}



def _range_conditions():
    return {
        str(key): list(value) for key, value in contract.RANGE_CONDITIONS.items()
    }


def _normalized_range_conditions(value):
    if not isinstance(value, dict):
        return value
    result = {}
    for key, entry in value.items():
        result[str(key)] = list(entry) if isinstance(entry, (list, tuple)) else entry
    return result


def _missing_params_fields(params, dims):
    if not isinstance(params, dict):
        return ["params object"]
    missing = []
    for name, _ in _all_constants(dims):
        if name not in params:
            missing.append(name)
    for name in (
        "contract",
        "e_res",
        "eps1",
        "eps2",
        "epsF",
        "MF",
        "KF",
        "s",
        "digests",
        "tables_sha256",
    ):
        if name not in params:
            missing.append(name)
    digests = params.get("digests")
    if isinstance(digests, dict):
        for name in ("trope", "texp", "tsilu"):
            if name not in digests:
                missing.append("digests.%s" % name)
    elif "digests" in params:
        missing.append("digests object")
    table_sha = params.get("tables_sha256")
    if isinstance(table_sha, dict):
        for name, path in TABLE_PATHS.items():
            if _table_sha(table_sha, name, path) is None:
                missing.append("tables_sha256.%s" % name)
    elif "tables_sha256" in params:
        missing.append("tables_sha256 object")
    return missing


def _expected_eps(e_res):
    return (4096 * (4**e_res) + 100000) // 200000


def _params_problems(params, dims):
    """Return human readable A15 details for params.json."""
    if not isinstance(params, dict):
        return ["params.json is not a JSON object"]

    problems = []
    if params.get("contract") != contract.CONTRACT_ID:
        problems.append(
            "contract is {!r}, expected {!r}".format(
                params.get("contract"), contract.CONTRACT_ID
            )
        )

    for name, expected in _all_constants(dims):
        if name not in params:
            problems.append("missing constant %s" % name)
        else:
            value = params[name]
            if not _is_int(value) or value != expected:
                problems.append(
                    "constant %s is {!r}, expected {!r}".format(
                        name, value, expected
                    )
                )

    for alias in NON_CANONICAL_CONSTANT_NAMES:
        if alias in params:
            problems.append("non canonical constant name")

    if "range_conditions" in params:
        value = _normalized_range_conditions(params["range_conditions"])
        if value != _range_conditions():
            problems.append("range_conditions differs from the contract")

    e_res = params.get("e_res")
    if not _is_int(e_res) or not 3 <= e_res <= 14:
        problems.append("e_res is {!r}, outside 3 .. 14".format(e_res))

    expected_eps = _expected_eps(e_res) if _is_int(e_res) else None
    for field in ("eps1", "eps2"):
        values = params.get(field)
        if not isinstance(values, list) or len(values) != dims.L:
            problems.append("%s is not a list of length %d" % (field, dims.L))
            continue
        for index, value in enumerate(values):
            if not _is_int(value):
                problems.append("%s[%d] is not an integer" % (field, index))
            elif value == 0:
                problems.append("%s[%d] is 0" % (field, index))
            elif expected_eps is not None and value != expected_eps:
                problems.append(
                    "%s[%d] is %d, expected %d"
                    % (field, index, value, expected_eps)
                )

    eps_f = params.get("epsF")
    if not _is_int(eps_f):
        problems.append("epsF is not an integer")
    elif eps_f == 0:
        problems.append("epsF is 0")
    elif expected_eps is not None and eps_f != expected_eps:
        problems.append("epsF is %d, expected %d" % (eps_f, expected_eps))

    s = params.get("s")
    if not _is_int(s) or s < 0 or s > 63:
        problems.append("s is {!r}, outside 0 .. 63".format(s))

    digests = params.get("digests")
    if not isinstance(digests, dict):
        problems.append("digests is not an object")
    else:
        for name in ("trope", "texp", "tsilu"):
            words = digests.get(name)
            if (
                not isinstance(words, list)
                or len(words) != 8
                or not all(_is_int(word) for word in words)
            ):
                problems.append(
                    "digests.%s is not a list of eight integers" % name
                )

    tables_sha = params.get("tables_sha256")
    if not isinstance(tables_sha, dict):
        problems.append("tables_sha256 is not an object")

    return problems


def _norm_problem(M, K):
    if not _is_int(M) or not _is_int(K):
        return "M or K is not an integer"
    if not 24 <= M <= 47:
        return "M is %d, outside 24 .. 47" % M
    if not 17 <= K <= 32:
        return "K is %d, outside 17 .. 32" % K
    if K < M - 14:
        return "K %d is below M - 14 = %d" % (K, M - 14)
    if K > M:
        return "K %d is above M %d" % (K, M)
    return None


def _kl_word_problems(kl, dims, params=None):
    """A11 to A14 for a uint32 [L][13] array."""
    problems = []
    try:
        words = np.asarray(kl).astype(np.int64)
    except (TypeError, ValueError):
        return [("A10", "KL.bin", "KL.bin is not an integer array")]
    if words.shape != (dims.L, 13):
        return [
            (
                "A10",
                "KL.bin",
                "KL.bin shape is {}, expected ({}, 13)".format(
                    words.shape, dims.L
                ),
            )
        ]

    for layer in range(dims.L):
        row = words[layer]
        for index, (name, ceiling) in enumerate(SHIFT_CEILINGS.items()):
            value = int(row[index])
            if value < 1 or value > ceiling:
                problems.append(
                    (
                        "A11",
                        "KL.bin",
                        "layer %d %s is %d, outside 1 .. %d"
                        % (layer, name, value, ceiling),
                    )
                )
        kq, kk = int(row[0]), int(row[1])
        if kq != kk and (kq < 16 or kk < 16):
            problems.append(
                (
                    "A11",
                    "KL.bin",
                    "layer %d has kq %d != kk %d while one is below 16"
                    % (layer, kq, kk),
                )
            )
        for m_index, k_index, label in ((7, 8, "M1/K1"), (9, 10, "M2/K2")):
            detail = _norm_problem(int(row[m_index]), int(row[k_index]))
            if detail is not None:
                problems.append(
                    ("A12", "KL.bin", "layer %d %s: %s" % (layer, label, detail))
                )
        if int(row[12]) != 1:
            problems.append(
                (
                    "A13",
                    "KL.bin",
                    "layer %d rp is %d, expected 1" % (layer, int(row[12])),
                )
            )

    sh_values = [int(words[layer][11]) for layer in range(dims.L)]
    if any(value < 4 or value > 12 for value in sh_values):
        problems.append(
            ("A14", "KL.bin", "sh values %r are outside 4 .. 12" % sh_values)
        )
    if len(set(sh_values)) > 1:
        problems.append(
            ("A14", "KL.bin", "sh differs between layers: %r" % sh_values)
        )

    if isinstance(params, dict):
        mf, kf = params.get("MF"), params.get("KF")
        detail = _norm_problem(mf, kf)
        if detail is not None:
            problems.append(("A12", "params.json", "MF/KF: %s" % detail))
    return problems


_MASK64 = 2**64 - 1
_TABLE_DIGEST_NUMBER = 72
_TABLE_DIGEST_WORDS = 8
_TABLE_DIGEST_BLOCK_WORDS = 32

_IV = (
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
)

_BLAKE2B_SIGMA = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
    (11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
    (7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
    (9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
    (2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
    (12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
    (13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
    (6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
    (10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
)


def _rotr64(word, count):
    return ((word >> count) | (word << (64 - count))) & _MASK64


def _blake2b_init_param(outlen, keylen):
    param = bytes([outlen, keylen, 1, 1]) + bytes(60)
    return [
        _IV[index] ^ int.from_bytes(param[8 * index : 8 * index + 8], "little")
        for index in range(8)
    ]


def _digest_g(round_index, index, a, b, c, d, message):
    a = (
        a
        + b
        + message[_BLAKE2B_SIGMA[round_index][2 * index + 0]]
    ) & _MASK64
    d = _rotr64(d ^ a, 32)
    c = (c + d) & _MASK64
    b = _rotr64(b ^ c, 24)
    a = (
        a
        + b
        + message[_BLAKE2B_SIGMA[round_index][2 * index + 1]]
    ) & _MASK64
    d = _rotr64(d ^ a, 16)
    c = (c + d) & _MASK64
    b = _rotr64(b ^ c, 63)
    return a, b, c, d


def _digest_round(round_index, state, message):
    state[0], state[4], state[8], state[12] = _digest_g(
        round_index, 0, state[0], state[4], state[8], state[12], message
    )
    state[1], state[5], state[9], state[13] = _digest_g(
        round_index, 1, state[1], state[5], state[9], state[13], message
    )
    state[2], state[6], state[10], state[14] = _digest_g(
        round_index, 2, state[2], state[6], state[10], state[14], message
    )
    state[3], state[7], state[11], state[15] = _digest_g(
        round_index, 3, state[3], state[7], state[11], state[15], message
    )
    state[0], state[5], state[10], state[15] = _digest_g(
        round_index, 4, state[0], state[5], state[10], state[15], message
    )
    state[1], state[6], state[11], state[12] = _digest_g(
        round_index, 5, state[1], state[6], state[11], state[12], message
    )
    state[2], state[7], state[8], state[13] = _digest_g(
        round_index, 6, state[2], state[7], state[8], state[13], message
    )
    state[3], state[4], state[9], state[14] = _digest_g(
        round_index, 7, state[3], state[4], state[9], state[14], message
    )


def _digest_compress(state, message, count_low, count_high, last):
    work = [0] * 16
    for index in range(8):
        work[index] = state[index]
    work[8] = _IV[0]
    work[9] = _IV[1]
    work[10] = _IV[2]
    work[11] = _IV[3]
    work[12] = _IV[4] ^ count_low
    work[13] = _IV[5] ^ count_high
    work[14] = _IV[6] ^ (_MASK64 if last else 0)
    work[15] = _IV[7]
    for round_index in range(12):
        _digest_round(round_index, work, message)
    return [
        state[index] ^ work[index] ^ work[index + 8] for index in range(8)
    ]


def _table_digest(words):
    """The FlashProve table digest, eight 32-bit words.

    This is the integer digest algorithm only.  The float table recipes stay
    in the converter and in the tests; the loader never imports them.
    """
    count = len(words)
    if (
        not 0 <= _TABLE_DIGEST_NUMBER < 2**32
        or count >= 2**32
        or not all(0 <= word < 2**32 for word in words)
    ):
        raise ValueError("table digest words are outside 0 .. 2^32 - 1")
    blocks = max(1, -(-count // _TABLE_DIGEST_BLOCK_WORDS))
    count_high = _TABLE_DIGEST_NUMBER + (count << 32)
    state = _blake2b_init_param(64, 0)
    for block_index in range(blocks):
        start = _TABLE_DIGEST_BLOCK_WORDS * block_index
        block = words[start : start + _TABLE_DIGEST_BLOCK_WORDS]
        message = struct.unpack(
            "<16Q",
            struct.pack("<%dI" % len(block), *block).ljust(
                4 * _TABLE_DIGEST_BLOCK_WORDS, b"\0"
            ),
        )
        last = block_index == blocks - 1
        state = _digest_compress(
            state,
            message,
            4 * count
            if last
            else 4 * _TABLE_DIGEST_BLOCK_WORDS * (block_index + 1),
            count_high,
            last,
        )
    digest = struct.pack("<8Q", *state)[: 4 * _TABLE_DIGEST_WORDS]
    return list(struct.unpack("<%dI" % _TABLE_DIGEST_WORDS, digest))


def _table_words(name, values):
    flat = np.asarray(values).reshape(-1).tolist()
    if name == "softmax_ex":
        return flat
    return [value + 4096 for value in flat]


def _table_digests(tables):
    result = {}
    if "rope_c" in tables and "rope_s" in tables:
        result["trope"] = _table_digest(
            _table_words("rope_c", tables["rope_c"])
            + _table_words("rope_s", tables["rope_s"])
        )
    if "softmax_ex" in tables:
        result["texp"] = _table_digest(
            _table_words("softmax_ex", tables["softmax_ex"])
        )
    if "silu_ts" in tables:
        result["tsilu"] = _table_digest(
            _table_words("silu_ts", tables["silu_ts"])
        )
    return result


def _table_sha(mapping, name, path):
    if not isinstance(mapping, dict):
        return None
    return mapping.get(path)


def _array_bytes(array, dtype):
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return contiguous.tobytes(order="C")


def _table_failures(tables, dims, params=None, table_sha=None):
    """A08 shape and A16 digest details for the caller-supplied tables."""
    problems = []
    expected_files = _expected_files(dims)
    for name, path in TABLE_PATHS.items():
        if name not in tables:
            problems.append(("A08", path, "missing table %s" % name))
            continue
        array = np.asarray(tables[name])
        if array.dtype.kind not in "iu":
            problems.append(
                ("A08", path, "table dtype is %s, expected integers" % array.dtype)
            )
            continue
        expected_shape = expected_files[path][1]
        if tuple(array.shape) != tuple(expected_shape):
            problems.append(
                (
                    "A08",
                    path,
                    "table shape is {}, expected {}".format(
                        array.shape, tuple(expected_shape)
                    ),
                )
            )
            continue

    if isinstance(params, dict):
        digests = params.get("digests")
        actual_digests = _table_digests(tables)
        if isinstance(digests, dict):
            for name in ("trope", "texp", "tsilu"):
                if name in actual_digests and digests.get(name) != actual_digests[name]:
                    problems.append(
                        ("A16", "params.json", "digest %s differs" % name)
                    )
        else:
            problems.append(("A16", "params.json", "digests is not an object"))

        sha_map = table_sha
        if sha_map is None:
            sha_map = params.get("tables_sha256")
        if isinstance(sha_map, dict):
            for name, path in TABLE_PATHS.items():
                expected_sha = _table_sha(sha_map, name, path)
                if expected_sha is None:
                    problems.append(
                        ("A16", "params.json", "missing tables_sha256 for %s" % name)
                    )
                    continue
                if name in tables:
                    actual = hashlib.sha256(
                        _array_bytes(tables[name], "<i4")
                    ).hexdigest()
                    if expected_sha != actual:
                        problems.append(
                            (
                                "A16",
                                "params.json",
                                "tables_sha256 for %s differs" % name,
                            )
                        )
        else:
            problems.append(
                ("A16", "params.json", "tables_sha256 is not an object")
            )
    return problems


def _memory_failures(matrices, emb, kl_words, params, tables, dims):
    expected = _expected_files(dims)
    failures = []

    for path, (dtype, shape) in expected.items():
        if dtype == "json" or path == "KL.bin" or path == "emb.i32":
            continue
        if dtype != "i16":
            continue
        if path not in matrices:
            failures.append(("A04", path, "missing matrix"))
            continue
        array = np.asarray(matrices[path])
        if array.dtype != np.int16 or tuple(array.shape) != tuple(shape):
            failures.append(
                (
                    "A08",
                    path,
                    "dtype {} shape {}, expected i16 {}".format(
                        array.dtype, array.shape, tuple(shape)
                    ),
                )
            )
            continue
        if array.size and (
            int(array.min()) < -contract.WEIGHT_BOUND
            or int(array.max()) > contract.WEIGHT_BOUND
        ):
            failures.append(
                (
                    "A09",
                    path,
                    "matrix entry outside -%d .. %d"
                    % (contract.WEIGHT_BOUND, contract.WEIGHT_BOUND),
                )
            )

    for path in matrices:
        if path not in expected or expected[path][0] != "i16":
            failures.append(("A08", path, "not an I2.1 file"))

    emb_array = np.asarray(emb)
    emb_shape = expected["emb.i32"][1]
    if emb_array.dtype != np.int32 or tuple(emb_array.shape) != tuple(emb_shape):
        failures.append(
            (
                "A08",
                "emb.i32",
                "dtype {} shape {}, expected i32 {}".format(
                    emb_array.dtype, emb_array.shape, tuple(emb_shape)
                ),
            )
        )
    elif emb_array.size:
        wide = emb_array.astype(np.int64)
        if int(np.abs(wide).max()) >= contract.EMB_BOUND:
            failures.append(
                (
                    "A09",
                    "emb.i32",
                    "embedding entry has absolute value at least 2^21",
                )
            )

    kl_array = np.asarray(kl_words)
    if kl_array.dtype.kind not in "iu" or tuple(kl_array.shape) != (
        dims.L,
        13,
    ):
        failures.append(
            (
                "A10",
                "KL.bin",
                "dtype {} shape {}, expected u32 ({}, 13)".format(
                    kl_array.dtype, kl_array.shape, dims.L
                ),
            )
        )
    else:
        wide = kl_array.astype(np.int64)
        if wide.size and (int(wide.min()) < 0 or int(wide.max()) > 2**32 - 1):
            failures.append(("A10", "KL.bin", "word outside 0 .. 2^32 - 1"))
        else:
            failures.extend(
                _kl_word_problems(kl_array.astype(np.uint32), dims, params)
            )

    failures.extend(
        ("A15", "params.json", detail)
        for detail in _params_problems(params, dims)
    )
    failures.extend(_table_failures(tables, dims, params))
    return failures


def _sort_failures(failures):
    return sorted(
        failures,
        key=lambda item: (item[0], "" if item[1] is None else str(item[1])),
    )


_CHUNK_BYTES = 1 << 20


def _read(raw_path):
    with open(raw_path, "rb") as handle:
        return handle.read()


def _scan_file(raw_path, dtype=None, keep=False):
    """Read a file once, hashing it and tracking its value range per chunk.

    The dtype is a numpy little-endian dtype.  The returned low and high are
    the minimum and maximum over all chunks, or None for a file with no
    entries.  Raw bytes are kept only when requested or when dtype is None,
    so large data files are never copied into memory.
    """
    digest = hashlib.sha256()
    kept = bytearray() if keep or dtype is None else None
    low = None
    high = None
    with open(raw_path, "rb") as handle:
        while True:
            block = handle.read(_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            if kept is not None:
                kept.extend(block)
            if dtype is not None:
                values = np.frombuffer(block, dtype=dtype)
                if values.size:
                    block_low = int(values.min())
                    block_high = int(values.max())
                    low = block_low if low is None else min(low, block_low)
                    high = block_high if high is None else max(high, block_high)
    return (
        digest.hexdigest(),
        bytes(kept) if kept is not None else None,
        low,
        high,
    )


def _raw_content_failures(path, raw, dims, params):
    """Return A08/A11 to A14 failures that need a file's full bytes."""
    failures = []

    if path == "KL.bin":
        if len(raw) != dims.L * 13 * 4:
            return failures
        array = np.frombuffer(raw, dtype="<u4").reshape(dims.L, 13)
        failures.extend(_kl_word_problems(array, dims, params))
        return failures

    if path == "exponents.json":
        try:
            _parse_canonical_json(raw)
        except _JSONProblem as exc:
            failures.append(
                (
                    "A08",
                    path,
                    "exponents.json not parseable: {}".format(exc),
                )
            )
        return failures

    return failures


def _directory_name_failure(directory, manifest_raw):
    artifact_id = hashlib.sha256(manifest_raw).hexdigest()
    name = os.path.basename(os.path.normpath(os.path.abspath(str(directory))))
    if name != artifact_id:
        return (
            "A07",
            "manifest.json",
            "directory name {} differs from artifact id {}".format(
                name, artifact_id
            ),
        )
    return None


def _validate(directory, dims):
    """Return (failures, directory_failure, manifest_raw, manifest)."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"

    try:
        if not os.path.isfile(manifest_path):
            raise OSError("manifest.json is missing or not a regular file")
        manifest_raw = _read(manifest_path)
    except OSError as exc:
        return (
            [("A01", "manifest.json", "cannot read manifest: {}".format(exc))],
            None,
            None,
            None,
        )
    try:
        manifest = _parse_canonical_json(manifest_raw)
        if not isinstance(manifest, dict):
            raise _JSONProblem("manifest is not a JSON object")
    except _JSONProblem as exc:
        return (
            [("A01", "manifest.json", str(exc))],
            None,
            manifest_raw,
            None,
        )

    expected = _expected_files(dims)
    failures = []

    if (
        manifest.get("format") != "tl-int-artifact"
        or not _is_int(manifest.get("format_version"))
        or manifest.get("format_version") != 1
        or manifest.get("contract") != contract.CONTRACT_ID
    ):
        failures.append(
            (
                "A02",
                "manifest.json",
                "format, format_version or contract differ",
            )
        )

    entries = manifest.get("files")
    if not isinstance(entries, list):
        failures.append(("A01", "manifest.json", "files is not a list"))
        entries = []

    listed = set()
    content_skipped = set()
    params = None
    seen_paths = set()
    order_paths = []
    order_complete = True
    table_raw = {}
    table_sha_actual = {}

    for entry in entries:
        if not isinstance(entry, dict):
            failures.append(("A08", "manifest.json", "file entry is not an object"))
            order_complete = False
            continue
        path = entry.get("path")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        listed_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")

        if not isinstance(path, str):
            failures.append(("A08", "manifest.json", "file path is not a string"))
            order_complete = False
            continue
        if path in seen_paths:
            failures.append(
                ("A01", "manifest.json", "duplicate path in files: %s" % path)
            )
        else:
            seen_paths.add(path)
        order_paths.append(path)
        listed.add(path)

        path_bad = False
        if os.path.isabs(path) or ".." in Path(path).parts or "\\" in path:
            path_bad = True
        candidate = os.path.join(str(directory), path)
        current = str(directory)
        for part in Path(path).parts:
            current = os.path.join(current, part)
            if os.path.islink(current):
                path_bad = True
                break
        try:
            base = os.path.realpath(str(directory))
            resolved = os.path.realpath(candidate)
            if os.path.commonpath([base, resolved]) != base:
                path_bad = True
        except (OSError, ValueError):
            path_bad = True
        if path_bad:
            failures.append(("A03", path, "path leaves the artifact directory"))
            continue

        expected_entry = expected.get(path)
        if expected_entry is None:
            failures.append(("A08", path, "not an I2.1 file"))
            continue
        expected_dtype, expected_shape = expected_entry

        is_kl = path == "KL.bin"
        shape_ok = (
            dtype == expected_dtype
            and isinstance(shape, list)
            and all(_is_int(dim) for dim in shape)
            and shape == list(expected_shape)
        )

        if not os.path.isfile(candidate):
            failures.append(
                ("A04", path, "listed file is absent or not a regular file")
            )
            content_skipped.add(path)
            continue

        try:
            actual_size = os.path.getsize(candidate)
        except OSError as exc:
            failures.append(("A06", path, "cannot stat file: %s" % exc))
            content_skipped.add(path)
            continue

        if is_kl:
            if actual_size != dims.L * 13 * 4:
                failures.append(
                    (
                        "A10",
                        path,
                        "KL.bin is %d bytes, expected %d"
                        % (actual_size, dims.L * 13 * 4),
                    )
                )
                content_skipped.add(path)
                continue
            if not shape_ok:
                failures.append(("A08", path, "dtype or shape differs from I2.1"))
                continue
        else:
            if not shape_ok:
                failures.append(("A08", path, "dtype or shape differs from I2.1"))
                continue

        size_ok = True
        if dtype != "json":
            product = _product(shape) if isinstance(shape, list) else None
            itemsize = _itemsize(dtype)
            if (
                product is None
                or itemsize is None
                or product * itemsize != listed_bytes
            ):
                failures.append(
                    (
                        "A06",
                        path,
                        "listed bytes do not equal product(shape) * itemsize",
                    )
                )
                size_ok = False

        if not _is_int(listed_bytes) or actual_size != listed_bytes:
            failures.append(
                (
                    "A06",
                    path,
                    "file size %d differs from listed %r"
                    % (actual_size, listed_bytes),
                )
            )
            size_ok = False

        if not size_ok:
            content_skipped.add(path)
            continue

        if not isinstance(expected_sha, str):
            failures.append(("A07", path, "sha256 differs"))
            content_skipped.add(path)
            continue

        scan_dtype = NUMPY_DTYPE.get(dtype)
        keep_raw = dtype == "json" or path == "KL.bin" or path in TABLE_PATHS.values()
        try:
            actual_sha, raw, low, high = _scan_file(
                candidate, scan_dtype, keep=keep_raw
            )
        except OSError as exc:
            failures.append(("A06", path, "cannot read file: %s" % exc))
            content_skipped.add(path)
            continue
        if actual_sha != expected_sha:
            failures.append(("A07", path, "sha256 differs"))
            content_skipped.add(path)
            continue

        if path == "emb.i32":
            if low is not None and (
                low < -(contract.EMB_BOUND - 1)
                or high > contract.EMB_BOUND - 1
            ):
                failures.append(
                    (
                        "A09",
                        path,
                        "embedding entry has absolute value at least 2^21",
                    )
                )
            continue

        if dtype == "i16":
            if low is not None and (
                low < -contract.WEIGHT_BOUND
                or high > contract.WEIGHT_BOUND
            ):
                failures.append(
                    (
                        "A09",
                        path,
                        "matrix entry outside -%d .. %d"
                        % (contract.WEIGHT_BOUND, contract.WEIGHT_BOUND),
                    )
                )
            continue

        if path == "params.json":
            try:
                params = _parse_canonical_json(raw)
            except _JSONProblem as exc:
                failures.append(("A15", path, str(exc)))
                continue
            for detail in _params_problems(params, dims):
                failures.append(("A15", path, detail))
            continue

        if path in TABLE_PATHS.values():
            table_raw[path] = raw
            table_sha_actual[path] = actual_sha
            continue

        failures.extend(_raw_content_failures(path, raw, dims, params))

    # A16 uses only the table file digests and the params.json summary.
    if isinstance(params, dict):
        for detail in _params_problems(params, dims):
            if not any(
                code == "A15" and path == "params.json" and detail == existing
                for code, path, existing in failures
            ):
                failures.append(("A15", "params.json", detail))

        mf_kf = _norm_problem(params.get("MF"), params.get("KF"))
        if mf_kf is not None and not any(
            code == "A12" and path == "params.json"
            for code, path, _ in failures
        ):
            failures.append(("A12", "params.json", "MF/KF: %s" % mf_kf))

        file_tables = {
            name: np.frombuffer(table_raw[path], dtype="<i4")
            for name, path in TABLE_PATHS.items()
            if path in table_raw
        }
        actual_digests = _table_digests(file_tables)
        digests = params.get("digests")
        if isinstance(digests, dict):
            for name in ("trope", "texp", "tsilu"):
                if (
                    name in actual_digests
                    and digests.get(name) != actual_digests[name]
                ):
                    failures.append(
                        ("A16", "params.json", "digest %s differs" % name)
                    )

        table_sha = params.get("tables_sha256")
        if isinstance(table_sha, dict):
            for name, path in TABLE_PATHS.items():
                expected_sha = _table_sha(table_sha, name, path)
                if expected_sha is None:
                    failures.append(
                        (
                            "A16",
                            "params.json",
                            "missing tables_sha256 for %s" % name,
                        )
                    )
                elif (
                    path in table_sha_actual
                    and expected_sha != table_sha_actual[path]
                ):
                    failures.append(
                        (
                            "A16",
                            "params.json",
                            "tables_sha256 for %s differs" % name,
                        )
                    )

    # A05 and symlinks anywhere under the directory.
    for root, dirs, files in os.walk(str(directory), followlinks=False):
        root_path = Path(root)
        for directory_name in list(dirs):
            full = root_path / directory_name
            rel = full.relative_to(directory).as_posix()
            if os.path.islink(full):
                failures.append(("A03", rel, "symlink in the artifact directory"))
        for file_name in files:
            full = root_path / file_name
            rel = full.relative_to(directory).as_posix()
            if os.path.islink(full):
                if rel not in listed:
                    failures.append(
                        ("A03", rel, "symlink in the artifact directory")
                    )
                continue
            if rel == "manifest.json":
                continue
            if rel not in listed:
                failures.append(("A05", rel, "file is not listed in the manifest"))

    if order_complete and order_paths != sorted(order_paths):
        failures.append(("A01", "manifest.json", "files is not sorted by path"))

    for path in expected:
        if path not in listed:
            failures.append(
                ("A04", path, "expected file is not listed in the manifest")
            )

    # I2.2/D13 (5): the directory name is the artifact id.  It is kept apart
    # from the other failures so load_artifact can apply D20 without rereading
    # manifest.json.
    directory_failure = _directory_name_failure(directory, manifest_raw)

    return failures, directory_failure, manifest_raw, manifest


def validate_artifact(directory, *, dims=FULL):
    """Return every I2.4 failure as a list of (code, path, detail) tuples."""
    failures, directory_failure, _, _ = _validate(directory, dims)
    if directory_failure is not None:
        failures = list(failures)
        failures.append(directory_failure)
    return _sort_failures(failures)

def write_artifact(
    out_dir,
    matrices,
    emb,
    kl_words,
    params,
    tables,
    provenance,
    *,
    dims=FULL,
    exponents=None,
):
    """Write an artifact below out_dir and return its id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = _missing_params_fields(params, dims)
    if missing:
        raise ValueError(
            "params.json is missing fields: {}".format(", ".join(missing))
        )
    if not isinstance(provenance, dict):
        raise ValueError("provenance must be an object")
    for field in ("source", "converter", "calibration"):
        if field not in provenance:
            raise ValueError("provenance is missing %s" % field)

    if exponents is None:
        exponents = {}
    if not isinstance(exponents, dict):
        raise ValueError("exponents must be an object")

    failures = _memory_failures(
        matrices, emb, kl_words, params, tables, dims
    )
    if failures:
        code, path, detail = _sort_failures(failures)[0]
        raise ArtifactRejected(code, path, detail)

    expected = _expected_files(dims)
    table_names = {path: name for name, path in TABLE_PATHS.items()}

    temporary = Path(
        tempfile.mkdtemp(dir=str(out_dir), prefix=".artifact-building-")
    )
    try:
        files = []
        for path in sorted(expected):
            dtype, shape = expected[path]
            if dtype == "i16":
                raw = _array_bytes(matrices[path], "<i2")
            elif path == "KL.bin":
                raw = _array_bytes(kl_words, "<u4")
            elif path == "emb.i32":
                raw = _array_bytes(emb, "<i4")
            elif path in table_names:
                raw = _array_bytes(tables[table_names[path]], "<i4")
            elif path == "params.json":
                raw = _canonical_json_bytes(params)
            else:
                raw = _canonical_json_bytes(exponents)
            destination = temporary / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            files.append(
                {
                    "path": path,
                    "dtype": dtype,
                    "shape": list(shape),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
            del raw

        manifest = {
            "format": "tl-int-artifact",
            "format_version": 1,
            "contract": contract.CONTRACT_ID,
            "source": provenance["source"],
            "converter": provenance["converter"],
            "calibration": provenance["calibration"],
            "files": files,
        }
        manifest_raw = _canonical_json_bytes(manifest)
        (temporary / "manifest.json").write_bytes(manifest_raw)

        artifact_id = hashlib.sha256(manifest_raw).hexdigest()
        target = out_dir / artifact_id
        if target.exists():
            raise FileExistsError(
                "artifact {} already exists; refusing to overwrite".format(
                    artifact_id
                )
            )
        os.rename(str(temporary), str(target))
        return artifact_id
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


@dataclass
class Artifact:
    id: str
    manifest: dict
    params: dict
    exponents: dict
    matrices: dict
    emb: np.ndarray
    wh: np.ndarray
    kl: np.ndarray
    tables: dict
    dims: Dims = FULL

    def matrix(self, layer, which):
        if which not in MATRIX_NAMES:
            raise ValueError("unknown matrix {!r}".format(which))
        path = "layers/%02d.%s.i16" % (int(layer), which)
        return self.matrices[path]

    def table(self, name):
        return self.tables[name]


def load_artifact(directory, expected_id=None, *, dims=FULL):
    """Validate and load an artifact, with read only memory mapped arrays."""
    directory = Path(directory)
    failures, directory_failure, manifest_raw, manifest = _validate(
        directory, dims
    )

    artifact_id = (
        hashlib.sha256(manifest_raw).hexdigest()
        if manifest_raw is not None
        else None
    )
    if (
        artifact_id is not None
        and expected_id is not None
        and expected_id != artifact_id
    ):
        failures = list(failures)
        failures.append(
            (
                "A07",
                "manifest.json",
                "expected id {} but the manifest hashes to {}".format(
                    expected_id, artifact_id
                ),
            )
        )

    if failures:
        # D20: the directory-name A07 stays out while another failure exists,
        # so the first sorted failure is the same one validate_artifact and
        # the previous loader reported.
        code, path, detail = _sort_failures(failures)[0]
        raise ArtifactRejected(code, path, detail)
    if directory_failure is not None:
        code, path, detail = directory_failure
        raise ArtifactRejected(code, path, detail)

    expected = _expected_files(dims)
    params = _parse_canonical_json(_read(directory / "params.json"))
    exponents = _parse_canonical_json(_read(directory / "exponents.json"))

    matrices = {}
    for path, (dtype, shape) in expected.items():
        if dtype != "i16":
            continue
        matrices[path] = np.memmap(
            directory / path,
            dtype=NUMPY_DTYPE[dtype],
            mode="r",
            shape=tuple(shape),
        )

    emb = np.memmap(
        directory / "emb.i32",
        dtype=NUMPY_DTYPE["i32"],
        mode="r",
        shape=tuple(expected["emb.i32"][1]),
    )
    kl = np.memmap(
        directory / "KL.bin",
        dtype=NUMPY_DTYPE["u32"],
        mode="r",
        shape=tuple(expected["KL.bin"][1]),
    )
    tables = {
        name: np.memmap(
            directory / path,
            dtype=NUMPY_DTYPE["i32"],
            mode="r",
            shape=tuple(expected[path][1]),
        )
        for name, path in TABLE_PATHS.items()
    }

    return Artifact(
        id=artifact_id,
        manifest=manifest,
        params=params,
        exponents=exponents,
        matrices=matrices,
        emb=emb,
        wh=matrices["wh.i16"],
        kl=kl,
        tables=tables,
        dims=dims,
    )
