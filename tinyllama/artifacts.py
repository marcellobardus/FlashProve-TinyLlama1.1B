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
from functools import lru_cache
from pathlib import Path

import numpy as np

from tinyllama import contract
from tinyllama.convert import tables as table_recipes
from tinyllama.manifest import sha256_file


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
        for index, name in enumerate(("kq", "kk", "kv", "ko", "kg", "ku", "kd")):
            value = int(row[index])
            if value < 1 or value > 32:
                problems.append(
                    (
                        "A11",
                        "KL.bin",
                        "layer %d %s is %d, outside 1 .. 32"
                        % (layer, name, value),
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


@lru_cache(maxsize=None)
def _recipe_tuples(dims):
    cosines, sines = table_recipes.rope_tables(
        dims.T, dims.DH, contract.CS, contract.THETA0
    )
    exp = table_recipes.softmax_table(contract.EXP_S, contract.RX)
    silu = table_recipes.silu_table(contract.EG, contract.SMAX)
    return tuple(cosines), tuple(sines), tuple(exp), tuple(silu)


def _recipe_tables(dims):
    cosines, sines, exp, silu = _recipe_tuples(dims)
    return {
        "rope_c": np.asarray(cosines, dtype=np.int32).reshape(
            dims.T, dims.DH // 2
        ),
        "rope_s": np.asarray(sines, dtype=np.int32).reshape(
            dims.T, dims.DH // 2
        ),
        "softmax_ex": np.asarray(exp, dtype=np.int32),
        "silu_ts": np.asarray(silu, dtype=np.int32),
    }


@lru_cache(maxsize=None)
def _recipe_digest_tuples(dims):
    cosines, sines, exp, silu = _recipe_tuples(dims)
    return {
        "trope": tuple(
            table_recipes.table_digest(
                table_recipes.trope_words(cosines, sines)
            )
        ),
        "texp": tuple(
            table_recipes.table_digest(table_recipes.texp_words(exp))
        ),
        "tsilu": tuple(
            table_recipes.table_digest(table_recipes.tsilu_words(silu))
        ),
    }


def _recipe_digests(dims):
    return {
        name: list(words) for name, words in _recipe_digest_tuples(dims).items()
    }


def _table_sha(mapping, name, path):
    if not isinstance(mapping, dict):
        return None
    candidates = (
        path,
        Path(path).name,
        path[: -len(".i32")],
        name,
        name + ".i32",
    )
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _array_bytes(array, dtype):
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    return contiguous.tobytes(order="C")


def _table_failures(tables, dims, params=None, table_sha=None):
    """A16 details for the tables in memory, and optionally params."""
    problems = []
    recipe = _recipe_tables(dims)
    for name, path in TABLE_PATHS.items():
        if name not in tables:
            problems.append(("A08", path, "missing table %s" % name))
            continue
        array = np.asarray(tables[name])
        if array.dtype.kind not in "iu":
            problems.append(("A08", path, "table dtype is %s, expected integers" % array.dtype))
            continue
        expected = recipe[name]
        if array.shape != expected.shape:
            problems.append(
                (
                    "A08",
                    path,
                    "table shape is {}, expected {}".format(
                        array.shape, expected.shape
                    ),
                )
            )
            continue
        array = array.astype(np.int64)
        if not np.array_equal(array, expected):
            mismatch = np.argwhere(array != expected)
            first = tuple(int(x) for x in mismatch[0]) if len(mismatch) else ()
            problems.append(
                ("A16", path, "table entry differs from its recipe at %r" % (first,))
            )

    if isinstance(params, dict):
        digests = params.get("digests")
        expected_digests = _recipe_digests(dims)
        if isinstance(digests, dict):
            for name in ("trope", "texp", "tsilu"):
                if digests.get(name) != expected_digests[name]:
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


def _read(raw_path):
    with open(raw_path, "rb") as handle:
        return handle.read()


def _content_failure(
    path, dtype, shape, raw, dims, expected, params
):
    """Return A09..A15 failures for one file's bytes."""
    failures = []

    if path == "KL.bin":
        if len(raw) != dims.L * 13 * 4:
            return failures
        array = np.frombuffer(raw, dtype="<u4").reshape(dims.L, 13)
        failures.extend(_kl_word_problems(array, dims, params))
        return failures

    if path == "emb.i32":
        array = np.frombuffer(raw, dtype="<i4")
        wide = array.astype(np.int64)
        if wide.size and int(np.abs(wide).max()) >= contract.EMB_BOUND:
            failures.append(
                (
                    "A09",
                    path,
                    "embedding entry has absolute value at least 2^21",
                )
            )
        return failures

    if dtype == "i16" and path in expected:
        array = np.frombuffer(raw, dtype="<i2")
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


def validate_artifact(directory, *, dims=FULL):
    """Return every I2.4 failure as a list of (code, path, detail) tuples."""
    directory = Path(directory)
    manifest_path = directory / "manifest.json"

    try:
        manifest_raw = _read(manifest_path)
    except OSError as exc:
        return [("A01", "manifest.json", "cannot read manifest: {}".format(exc))]
    try:
        manifest = _parse_canonical_json(manifest_raw)
        if not isinstance(manifest, dict):
            raise _JSONProblem("manifest is not a JSON object")
    except _JSONProblem as exc:
        return [("A01", "manifest.json", str(exc))]

    expected = _expected_files(dims)
    failures = []

    if (
        manifest.get("format") != "tl-int-artifact"
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
        failures.append(("A02", "manifest.json", "files is not a list"))
        entries = []

    listed = set()
    content_skipped = set()
    params = None
    params_checked = False

    for entry in entries:
        if not isinstance(entry, dict):
            failures.append(("A08", "manifest.json", "file entry is not an object"))
            continue
        path = entry.get("path")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        listed_bytes = entry.get("bytes")
        expected_sha = entry.get("sha256")

        if not isinstance(path, str):
            failures.append(("A08", "manifest.json", "file path is not a string"))
            continue
        listed.add(path)

        path_bad = False
        if os.path.isabs(path) or ".." in path or "\\" in path:
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
            and shape == list(expected_shape)
        )

        exists = os.path.exists(candidate)
        if not exists:
            failures.append(("A04", path, "listed file is absent"))
            content_skipped.add(path)
            continue

        if is_kl:
            actual_size = os.path.getsize(candidate)
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

        actual_size = os.path.getsize(candidate)
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

        if not isinstance(expected_sha, str) or sha256_file(candidate) != expected_sha:
            failures.append(("A07", path, "sha256 differs"))
            content_skipped.add(path)
            continue

        raw = _read(candidate)
        if path == "params.json":
            try:
                params = _parse_canonical_json(raw)
            except _JSONProblem as exc:
                failures.append(("A15", path, str(exc)))
                params_checked = True
                continue
            for detail in _params_problems(params, dims):
                failures.append(("A15", path, detail))
            params_checked = True

        failures.extend(
            _content_failure(
                path, dtype, shape, raw, dims, expected, params
            )
        )

    # A16 table entries, recomputed from the recipe.  A file that failed
    # A04/A06/A07 is skipped, as for every other content check.
    recipe_arrays = _recipe_tables(dims)
    for name, path in TABLE_PATHS.items():
        if path not in listed or path in content_skipped:
            continue
        table_path = directory / path
        if not table_path.is_file():
            continue
        raw = _read(table_path)
        recipe = recipe_arrays[name]
        actual = np.frombuffer(raw, dtype="<i4")
        if actual.size != recipe.size:
            failures.append(
                (
                    "A16",
                    path,
                    "table has %d entries, recipe has %d"
                    % (actual.size, recipe.size),
                )
            )
        else:
            actual = actual.reshape(recipe.shape)
            if not np.array_equal(actual, recipe):
                mismatch = np.argwhere(actual != recipe)
                first = tuple(int(x) for x in mismatch[0]) if len(mismatch) else ()
                failures.append(
                    (
                        "A16",
                        path,
                        "table entry differs from its recipe at %r" % (first,),
                    )
                )

    # A16 checks that do not need the table files to be readable.
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

        expected_digests = _recipe_digests(dims)
        digests = params.get("digests")
        if isinstance(digests, dict):
            for name in ("trope", "texp", "tsilu"):
                if digests.get(name) != expected_digests[name]:
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
                    path in listed
                    and path not in content_skipped
                    and (directory / path).is_file()
                ):
                    actual = sha256_file(directory / path)
                    if expected_sha != actual:
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

    # If params.json was not listed at all, no A15 run happened above.
    if not params_checked and isinstance(params, dict):
        for detail in _params_problems(params, dims):
            failures.append(("A15", "params.json", detail))

    # I2.2/D13 (5): the directory name is the artifact id.
    directory_failure = _directory_name_failure(directory, manifest_raw)
    if directory_failure is not None:
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
    data = {}
    for path, (dtype, _) in expected.items():
        if dtype == "i16":
            data[path] = _array_bytes(matrices[path], "<i2")
    data["KL.bin"] = _array_bytes(kl_words, "<u4")
    data["emb.i32"] = _array_bytes(emb, "<i4")
    for name, path in TABLE_PATHS.items():
        data[path] = _array_bytes(tables[name], "<i4")
    data["params.json"] = _canonical_json_bytes(params)
    data["exponents.json"] = _canonical_json_bytes(exponents)

    temporary = Path(
        tempfile.mkdtemp(dir=str(out_dir), prefix=".artifact-building-")
    )
    try:
        files = []
        for path in sorted(data):
            raw = data[path]
            destination = temporary / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            dtype, shape = expected[path]
            files.append(
                {
                    "path": path,
                    "dtype": dtype,
                    "shape": list(shape),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )

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
    failures = validate_artifact(directory, dims=dims)
    if failures:
        # The directory-name rule reports A07 on manifest.json, so sorting
        # would make it mask a content fault that used to be raised first.
        # Keep the old code for an artifact that already has another failure.
        directory_failure = None
        try:
            manifest_raw = _read(Path(directory) / "manifest.json")
            directory_failure = _directory_name_failure(directory, manifest_raw)
        except OSError:
            pass
        other_failures = [
            failure for failure in failures if failure != directory_failure
        ]
        code, path, detail = _sort_failures(other_failures or failures)[0]
        raise ArtifactRejected(code, path, detail)

    directory = Path(directory)
    manifest_raw = _read(directory / "manifest.json")
    artifact_id = hashlib.sha256(manifest_raw).hexdigest()

    # QUESTION: I2.4 has no code for a caller supplied expected_id mismatch.
    if expected_id is not None and expected_id != artifact_id:
        raise ArtifactRejected(
            "A07",
            "manifest.json",
            "expected id {} but the manifest hashes to {}".format(
                expected_id, artifact_id
            ),
        )

    manifest = _parse_canonical_json(manifest_raw)
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
