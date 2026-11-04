# Run on the pod (repository at /workspace/repo, checkpoint at /workspace/ckpt):
#   cd /workspace/repo && PYTHONPATH=. python3 -m tinyllama.convert.run \
#       --checkpoint /workspace/ckpt --out /workspace/repo/artifacts/TL-INT-1 \
#       --calib /workspace/calib/fit.npy --calib-sha256 <FIT_SHA256> \
#       --heldout /workspace/calib/heldout.npy --heldout-sha256 <HELDOUT_SHA256> \
#       --report /workspace/repo/artifacts/TL-INT-1/range-report.json
#
# This module has no command line of its own. run.py calls calibrate().

"""Deterministic float32 calibration pass for TL-INT-1 (contract sections 4 and 5).

The pass is a numpy float32 implementation of the TinyLlama Llama architecture:
RMSNorm with the checkpoint gains and eps, rotary applied to the permuted q and k
rows with neighbouring pairs (contract 4.4), grouped query attention with H query
heads and G key/value heads, causal softmax, SwiGLU, the final norm and lm_head.
It reads one layer's tensors at a time from a safetensors file through
tinyllama.convert.load and keeps only the hidden state of all windows, one
layer of weights, and one window's attention work.

Units of the recorded maxima (contract 5.2):

* n1, n2 and nF are the plain RMS normalized vectors, before the norm gain, as
  in FlashProve's float pass. The fit divides their maximum by c before it
  takes the cell exponent, exactly as contract 5.2 says: "Es of a norm output
  n: largest E with (mx / c) 2^E <= 2047". They are not gain folded.
* q is the query after the rotary step, divided by sqrt(dh). This is the
  FlashProve convention and it is what contract 4.3 gives: the q conversion
  constant c_q = c / sqrt(dh) folds the attention scale into Wq, so the integer
  q cell stores q / sqrt(dh).
* k and v are the key and value after rotary, in their own units.
* u is the up projection of the gain folded second norm output.
* The gates, q_proj, k_proj and v_proj are fed the gain folded norm output in
  this float pass, just as Hugging Face does. Folding the gains into the matrix
  is equivalent; see contract 4.2 and 4.3.

BLAS note: the pass uses numpy matrix products, which call a BLAS library. A
BLAS may choose a different summation order on another run or another machine,
so the last bits of a product are not guaranteed to be identical. The operation
order inside this module is fixed, and no threading choice of this module
changes it, but a BLAS is outside this module's control. The fit uses maxima, so
a last bit difference changes a scale only when a maximum sits exactly on a
cell boundary.
"""

import hashlib
import math
import os
from pathlib import Path

import numpy as np

from tinyllama import contract
from tinyllama.convert.load import read_header, read_tensor
from tinyllama.convert.fold import permute_qk


SITES = ("n1", "n2", "q", "k", "v", "u", "nF")


class CalibrationError(ValueError):
    """A calibration input or checkpoint tensor cannot be used."""


def _sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_windows(path, expected_sha256):
    """Load a TL-CAL-1 npy file and verify its digest, shape and dtype.

    The array must be a C order int32 array of shape (n_windows, T) with at
    least one window and at least one position. The returned array is a
    contiguous int32 copy.
    """
    if not isinstance(expected_sha256, str):
        raise CalibrationError("expected_sha256 must be a hex string")
    actual = _sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise CalibrationError(
            "sha256 mismatch for {}: expected {}, got {}".format(
                path, expected_sha256, actual
            )
        )
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise CalibrationError("{} is not a readable npy file: {}".format(path, exc))
    if not isinstance(array, np.ndarray):
        raise CalibrationError("{} does not contain one array".format(path))
    if array.dtype != np.dtype("<i4"):
        raise CalibrationError(
            "{} has dtype {}, expected <i4".format(path, array.dtype)
        )
    if array.ndim != 2:
        raise CalibrationError(
            "{} has shape {}, expected two dimensions".format(path, array.shape)
        )
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise CalibrationError("{} is empty".format(path))
    return np.ascontiguousarray(array, dtype=np.int32)


class CheckpointReader:
    """Read one tensor at a time, with the header and shapes pinned by a manifest."""

    def __init__(self, path, manifest):
        self.path = Path(path)
        self.manifest = manifest
        self._header_bytes, self._header, self._data_start = read_header(self.path)
        self._expected = self._expected_tensors(manifest)
        self._check_header()

    @staticmethod
    def _expected_tensors(manifest):
        tensors = manifest.get("tensors")
        if not isinstance(tensors, dict):
            raise CalibrationError("manifest has no tensor map")
        expected = {}
        for name, spec in tensors.items():
            if isinstance(spec, dict):
                if "dtype" not in spec or "shape" not in spec:
                    raise CalibrationError("manifest entry {!r} is incomplete".format(name))
                dtype, shape = spec["dtype"], spec["shape"]
            elif isinstance(spec, (tuple, list)) and len(spec) == 2:
                dtype, shape = spec
            else:
                raise CalibrationError("manifest entry {!r} is malformed".format(name))
            expected[name] = (dtype, tuple(int(value) for value in shape))
        return expected

    def _check_header(self):
        source = self.manifest.get("model_safetensors", {})
        expected_sha = source.get("header_json_sha256")
        if not isinstance(expected_sha, str):
            raise CalibrationError("manifest has no header sha256")
        actual_sha = hashlib.sha256(self._header_bytes).hexdigest()
        if actual_sha.lower() != expected_sha.lower():
            raise CalibrationError(
                "header sha256 mismatch: expected {}, got {}".format(
                    expected_sha, actual_sha
                )
            )

        actual_names = set(self._header) - {"__metadata__"}
        if actual_names != set(self._expected):
            raise CalibrationError("tensor names do not match the manifest")

        count = source.get("tensor_count", self.manifest.get("tensor_count"))
        if count is not None and count != len(self._expected):
            raise CalibrationError("tensor count does not match the manifest")

        for name, (want_dtype, want_shape) in self._expected.items():
            entry = self._header[name]
            if entry.get("dtype") != want_dtype or tuple(entry.get("shape", ())) != want_shape:
                raise CalibrationError(
                    "tensor {!r} does not match the manifest dtype or shape".format(name)
                )

    def names(self):
        return sorted(self._expected)

    def shape(self, name):
        if name not in self._expected:
            raise CalibrationError("no tensor named {!r}".format(name))
        return self._expected[name][1]

    def tensor(self, name):
        if name not in self._expected:
            raise CalibrationError("no tensor named {!r}".format(name))
        return read_tensor(
            self.path, name, (self._header, self._data_start)
        )


class FloatModel:
    """The float32 TinyLlama model that calibrate() reads one layer at a time."""

    def __init__(self, reader, dims, *, eps=1e-5, theta0=contract.THETA0):
        self.reader = reader
        self.dims = dims
        self.eps = float(eps)
        self.theta0 = float(theta0)

    def tensor(self, name):
        return self.reader.tensor(name)

    def layer_name(self, layer, suffix):
        return "model.layers.{}.{}".format(int(layer), suffix)


def _rms_norm(x, eps):
    """The plain HF RMSNorm output, before the gain, in float32."""
    values = np.asarray(x, dtype=np.float32)
    mean = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    root = np.sqrt(mean + np.float32(eps))
    return values / root


def _silu(x):
    values = np.asarray(x, dtype=np.float32)
    # For very negative values exp(-values) overflows to inf, so the quotient
    # becomes -0.0, which is the correct limit of SiLU. The overflow warning is
    # harmless, so ignore only that warning and keep the expression unchanged.
    with np.errstate(over="ignore"):
        return values / (np.float32(1.0) + np.exp(-values))


def _rotary_neighbour(x, cosines, sines):
    """Neighbouring pair rotary on [positions][heads][dh] float32 values."""
    even = x[..., 0::2]
    odd = x[..., 1::2]
    c = cosines[:, None, :]
    s = sines[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = c * even - s * odd
    out[..., 1::2] = s * even + c * odd
    return out


def _rotary_tables(theta0, positions, dh):
    """Return float32 cosines and sines of shape [positions][dh / 2]."""
    half = dh // 2
    index = np.arange(half, dtype=np.float32)
    frequency = np.float32(1.0) / np.power(
        np.float32(theta0), index / np.float32(half)
    )
    angle = np.asarray(positions, dtype=np.float32)[:, None] * frequency[None, :]
    return np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)


def _attention(q, k, v, heads, groups, dh):
    """Causal grouped query attention for one window, all in float32."""
    positions = q.shape[0]
    group = heads // groups
    scale = np.float32(1.0 / math.sqrt(float(dh)))
    out = np.empty((positions, heads, dh), dtype=np.float32)
    for head in range(heads):
        key_head = head // group
        query = q[:, head, :]
        key = k[:, key_head, :]
        value = v[:, key_head, :]
        scores = np.matmul(query, key.T) * scale
        head_out = np.empty((positions, dh), dtype=np.float32)
        for position in range(positions):
            row = scores[position, : position + 1]
            shifted = row - np.max(row)
            weights = np.exp(shifted)
            total = np.sum(weights, dtype=np.float32)
            head_out[position] = np.matmul(weights, value[: position + 1]) / total
        out[:, head, :] = head_out
    return out.reshape(positions, heads * dh)


def _bump(maxima, site, layer, values, scale=1.0):
    seen = float(np.max(np.abs(np.asarray(values)))) * float(scale)
    of_site = maxima.setdefault(site, {})
    if seen > of_site.get(layer, 0.0):
        of_site[layer] = seen


def float_forward(model, windows, *, record=None, return_logits=False):
    """Run the float32 pass over windows of shape (n_windows, T).

    Return (maxima, logits). maxima is {site: {layer: largest absolute value}},
    with nF keyed by dims.L. logits is None unless return_logits is true, and
    otherwise has shape (n_windows, T, V). The q site is divided by sqrt(dh).
    """
    dims = model.dims
    array = np.asarray(windows)
    if array.ndim != 2:
        raise CalibrationError("windows must have shape (n_windows, T)")
    if array.dtype.kind not in "iu":
        raise CalibrationError("windows must be integers")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise CalibrationError("windows must not be empty")
    if array.shape[1] > dims.T:
        raise CalibrationError(
            "a window has {} positions, the shape admits {}".format(
                array.shape[1], dims.T
            )
        )
    if int(array.min()) < 0 or int(array.max()) >= dims.V:
        raise CalibrationError("a token id is outside 0 .. {}".format(dims.V - 1))
    windows = np.ascontiguousarray(array, dtype=np.int32)

    n_windows, positions = windows.shape
    d = dims.D
    heads, groups, dh = dims.H, dims.G, dims.DH
    if heads % groups:
        raise CalibrationError("H must be divisible by G")
    if dh % 2:
        raise CalibrationError("dh must be even")

    embedding = model.tensor("model.embed_tokens.weight")
    if embedding.shape != (dims.V, d):
        raise CalibrationError("embedding shape does not match dims")
    hidden = np.ascontiguousarray(embedding[windows], dtype=np.float32)
    del embedding

    cosines, sines = _rotary_tables(model.theta0, np.arange(positions), dh)
    maxima = {site: {} for site in SITES}
    q_scale = 1.0 / math.sqrt(float(dh))

    for layer in range(dims.L):
        gain1 = model.tensor(model.layer_name(layer, "input_layernorm.weight"))
        gain2 = model.tensor(
            model.layer_name(layer, "post_attention_layernorm.weight")
        )
        wq = model.tensor(model.layer_name(layer, "self_attn.q_proj.weight"))
        wk = model.tensor(model.layer_name(layer, "self_attn.k_proj.weight"))
        wv = model.tensor(model.layer_name(layer, "self_attn.v_proj.weight"))
        wo = model.tensor(model.layer_name(layer, "self_attn.o_proj.weight"))
        wg = model.tensor(model.layer_name(layer, "mlp.gate_proj.weight"))
        wu = model.tensor(model.layer_name(layer, "mlp.up_proj.weight"))
        wd = model.tensor(model.layer_name(layer, "mlp.down_proj.weight"))

        wq = permute_qk(wq, heads)
        wk = permute_qk(wk, groups)

        for window in range(n_windows):
            x = hidden[window]
            n1 = _rms_norm(x, model.eps)
            _bump(maxima, "n1", layer, n1)
            if record is not None:
                record("n1", layer, n1)

            a = n1 * gain1
            q = np.matmul(a, wq.T).reshape(positions, heads, dh)
            q = _rotary_neighbour(q, cosines, sines)
            _bump(maxima, "q", layer, q, q_scale)
            if record is not None:
                record("q", layer, q)

            k = np.matmul(a, wk.T).reshape(positions, groups, dh)
            k = _rotary_neighbour(k, cosines, sines)
            _bump(maxima, "k", layer, k)
            if record is not None:
                record("k", layer, k)

            v = np.matmul(a, wv.T).reshape(positions, groups, dh)
            _bump(maxima, "v", layer, v)
            if record is not None:
                record("v", layer, v)

            o = _attention(q, k, v, heads, groups, dh)
            x_mid = x + np.matmul(o, wo.T)

            n2 = _rms_norm(x_mid, model.eps)
            _bump(maxima, "n2", layer, n2)
            if record is not None:
                record("n2", layer, n2)

            h = n2 * gain2
            up = np.matmul(h, wu.T)
            _bump(maxima, "u", layer, up)
            if record is not None:
                record("u", layer, up)

            gate = np.matmul(h, wg.T)
            down = np.matmul(_silu(gate) * up, wd.T)
            hidden[window] = x_mid + down

        del gain1, gain2, wq, wk, wv, wo, wg, wu, wd

    for window in range(n_windows):
        nF = _rms_norm(hidden[window], model.eps)
        _bump(maxima, "nF", dims.L, nF)
        if record is not None:
            record("nF", dims.L, nF)

    logits = None
    if return_logits:
        gain_f = model.tensor("model.norm.weight")
        lm_head = model.tensor("lm_head.weight")
        logits = np.empty((n_windows, positions, dims.V), dtype=np.float32)
        for window in range(n_windows):
            nF = _rms_norm(hidden[window], model.eps)
            logits[window] = np.matmul(nF * gain_f, lm_head.T)
        del gain_f, lm_head

    return maxima, logits


def calibrate(model, windows, *, record=None):
    """Return the per site maxima that contract 5.2 needs."""
    maxima, _ = float_forward(model, windows, record=record, return_logits=False)
    return maxima


def _final_norm_range_check(row, m, k, eps, layer, position):
    """Check conditions 1, 2 and 8 of contract 3.2 for one final norm row."""
    from tinyllama.reference.primitives import isqrt, rq

    for index, value in enumerate(row.tolist()):
        if value < -547599908735 or value > 547599908735:
            raise _IntegerRangeFailure(8, "stream x read by a norm", layer, position, index, value)

    vx = sum(int(value) * int(value) for value in row.tolist())
    vx_plus_eps = vx + int(eps)
    if vx_plus_eps >= 2**78:
        raise _IntegerRangeFailure(1, "Vx+eps", layer, position, 0, vx_plus_eps)

    sigma = isqrt(vx_plus_eps)
    rc = ((1 << int(m)) + (sigma // 2)) // sigma
    for index, value in enumerate(row.tolist()):
        w = rq(int(value) * rc, int(k))
        if w < -32768 or w > 32639:
            raise _IntegerRangeFailure(2, "norm quotient w", layer, position, index, w)


class _IntegerRangeFailure(Exception):
    def __init__(self, condition, name, layer, position, index, value):
        self.condition = int(condition)
        self.name = name
        self.layer = int(layer)
        self.position = int(position)
        self.index = int(index)
        self.value = value
        super().__init__(str(self))

    def as_dict(self):
        return {
            "condition": self.condition,
            "name": self.name,
            "layer": self.layer,
            "position": self.position,
            "index": self.index,
            "value": self.value,
        }


def _run_integer_window(artifact, ids, dims):
    """Run the integer reference on one window, returning a failure or None."""
    from tinyllama.reference.layer import RangeFailure, layer as reference_layer
    from tinyllama.reference.primitives import clamp, isqrt, rq

    tables = {
        "C": np.asarray(artifact.table("rope_c"), dtype=np.int64),
        "S": np.asarray(artifact.table("rope_s"), dtype=np.int64),
        "Ts": np.asarray(artifact.table("silu_ts"), dtype=np.int64),
        "EX": np.asarray(artifact.table("softmax_ex"), dtype=np.int64),
    }
    x = np.asarray(artifact.emb[np.asarray(ids, dtype=np.int64)], dtype=np.int64)
    kl = np.asarray(artifact.kl, dtype=np.int64)
    params = artifact.params

    for index in range(dims.L):
        row = kl[index]
        weights = {
            "wq": np.asarray(artifact.matrix(index, "wq"), dtype=np.int64),
            "wk": np.asarray(artifact.matrix(index, "wk"), dtype=np.int64),
            "wv": np.asarray(artifact.matrix(index, "wv"), dtype=np.int64),
            "wo": np.asarray(artifact.matrix(index, "wo"), dtype=np.int64),
            "wg": np.asarray(artifact.matrix(index, "wg"), dtype=np.int64),
            "wu": np.asarray(artifact.matrix(index, "wu"), dtype=np.int64),
            "wd": np.asarray(artifact.matrix(index, "wd"), dtype=np.int64),
        }
        scales = {
            "kq": int(row[0]),
            "kk": int(row[1]),
            "kv": int(row[2]),
            "ko": int(row[3]),
            "kg": int(row[4]),
            "ku": int(row[5]),
            "kd": int(row[6]),
            "M1": int(row[7]),
            "K1": int(row[8]),
            "M2": int(row[9]),
            "K2": int(row[10]),
            "sh": int(row[11]),
            "eps1": int(params["eps1"][index]),
            "eps2": int(params["eps2"][index]),
        }
        try:
            x = reference_layer(
                x, weights, scales, tables, layer_index=index
            )
        except RangeFailure as failure:
            return failure.as_dict()

    for position in range(x.shape[0]):
        try:
            _final_norm_range_check(
                x[position],
                int(params["MF"]),
                int(params["KF"]),
                int(params["epsF"]),
                dims.L,
                position,
            )
        except _IntegerRangeFailure as failure:
            return failure.as_dict()
    return None


def range_report(artifact_or_dir, windows, dims=None, *, float_model=None):
    """Return the P4 range report as a JSON serialisable dict.

    The tuple of per layer pieces in tinyllama.reference is enough to run the
    integer forward: reference.layer does every layer, including its range
    checks, and this module does the final norm's checks. The logits add no
    range condition (contract 3.2), so they are not computed here.

    If float_model is given, the float activation maxima are recorded too and
    compared with the 2047 cell limit. If reference.layer cannot be imported,
    every window is marked "integer reference not available".

    QUESTION FOR LEAD: the contract does not say whether P4 wants one report
    per window or one report over the held-out set. This function reports both:
    a top level range_failure list and a per_window list.
    """
    from tinyllama.artifacts import FULL, load_artifact

    if dims is None:
        dims = FULL
    if isinstance(artifact_or_dir, (str, bytes, os.PathLike)):
        artifact = load_artifact(artifact_or_dir, dims=dims)
    else:
        artifact = artifact_or_dir
    dims = getattr(artifact, "dims", dims)

    array = np.asarray(windows)
    if array.ndim != 2:
        raise CalibrationError("windows must have shape (n_windows, T)")
    if array.dtype.kind not in "iu":
        raise CalibrationError("windows must be integers")
    if array.size and (int(array.min()) < 0 or int(array.max()) >= dims.V):
        raise CalibrationError("a token id is outside 0 .. {}".format(dims.V - 1))
    array = np.ascontiguousarray(array, dtype=np.int32)

    try:
        from tinyllama.reference.layer import RangeFailure  # noqa: F401
        integer_available = True
    except ImportError:
        integer_available = False

    per_window = []
    failures = []
    for number, ids in enumerate(array):
        entry = {"window": number, "integer_reference": "ran", "range_failure": None}
        if integer_available:
            failure = _run_integer_window(artifact, ids, dims)
            entry["range_failure"] = failure
            if failure is not None:
                failures.append(dict(failure, window=number))
        else:
            entry["integer_reference"] = "not available"
        per_window.append(entry)

    float_maxima = None
    float_over = []
    if float_model is not None:
        float_maxima = calibrate(float_model, array)
        for site, of_site in float_maxima.items():
            for layer, value in of_site.items():
                if value > 2047.0:
                    float_over.append(
                        {"site": site, "layer": layer, "max": value, "limit": 2047}
                    )

    return {
        "contract": contract.CONTRACT_ID,
        "artifact_id": getattr(artifact, "id", None),
        "windows": int(array.shape[0]),
        "integer_reference": "ran" if integer_available else "not available",
        "range_failures": failures,
        "range_failure_count": len(failures),
        "per_window": per_window,
        "float_maxima": float_maxima,
        "float_limit": 2047,
        "float_over_limit": float_over,
    }
