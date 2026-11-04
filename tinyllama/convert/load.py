"""A small, strict reader for the safetensors checkpoint format."""

import hashlib
import json
import math
import struct

import numpy as np


class SafetensorsError(ValueError):
    """The safetensors file or its metadata is not acceptable."""


DTYPE_ITEMSIZE = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "I32": 4,
    "I16": 2,
    "U8": 1,
}

_TENSOR_KEYS = frozenset(("dtype", "shape", "data_offsets"))
_METADATA_KEY = "__metadata__"


def _object_without_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise SafetensorsError("duplicate JSON key: {!r}".format(key))
        obj[key] = value
    return obj


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def read_header(path):
    """Return (header_bytes, header_dict, data_start) for a safetensors file."""
    with open(path, "rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise SafetensorsError("file is shorter than the 8-byte header length")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length == 0 or header_length > 100_000_000:
            raise SafetensorsError(
                "header length {} is outside the accepted range".format(header_length)
            )

        handle.seek(0, 2)
        file_size = handle.tell()
        data_start = 8 + header_length
        if data_start > file_size:
            raise SafetensorsError("header extends beyond the end of the file")

        handle.seek(8)
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise SafetensorsError("file ended inside the header")

    data_size = file_size - data_start

    try:
        text = header_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SafetensorsError("header is not valid UTF-8") from exc

    try:
        header = json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise SafetensorsError("header is not valid JSON") from exc

    if not isinstance(header, dict):
        raise SafetensorsError("header is not a JSON object")

    metadata = header.get(_METADATA_KEY)
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise SafetensorsError("__metadata__ is not an object")
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SafetensorsError("__metadata__ must map strings to strings")

    ranges = []
    for name, entry in header.items():
        if name == _METADATA_KEY:
            continue
        if not isinstance(entry, dict):
            raise SafetensorsError("tensor entry {!r} is not an object".format(name))
        if set(entry) != _TENSOR_KEYS:
            raise SafetensorsError(
                "tensor entry {!r} must have exactly {}".format(name, sorted(_TENSOR_KEYS))
            )

        dtype = entry["dtype"]
        shape = entry["shape"]
        offsets = entry["data_offsets"]

        if not isinstance(dtype, str):
            raise SafetensorsError("tensor {!r} has a non-string dtype".format(name))

        if not isinstance(shape, list):
            raise SafetensorsError("tensor {!r} has a non-list shape".format(name))
        for dim in shape:
            if not _is_int(dim) or dim < 0:
                raise SafetensorsError(
                    "tensor {!r} has a shape that is not non-negative integers".format(name)
                )

        if not isinstance(offsets, list) or len(offsets) != 2:
            raise SafetensorsError(
                "tensor {!r} has data_offsets that are not two integers".format(name)
            )
        begin, end = offsets
        if not _is_int(begin) or not _is_int(end):
            raise SafetensorsError(
                "tensor {!r} has non-integer data_offsets".format(name)
            )
        if not 0 <= begin <= end:
            raise SafetensorsError(
                "tensor {!r} has data_offsets outside 0 <= begin <= end".format(name)
            )

        itemsize = DTYPE_ITEMSIZE.get(dtype)
        if itemsize is not None:
            expected = math.prod(shape) * itemsize
            if end - begin != expected:
                raise SafetensorsError(
                    "tensor {!r} has {} bytes but {} are required".format(
                        name, end - begin, expected
                    )
                )

        if end > data_size:
            raise SafetensorsError(
                "tensor {!r} extends beyond the data buffer".format(name)
            )
        ranges.append((begin, end, name))

    expected_start = 0
    for begin, end, name in sorted(ranges, key=lambda item: item[0]):
        if begin == end:
            continue
        if begin != expected_start:
            raise SafetensorsError(
                "tensor {!r} leaves a gap or overlaps another tensor".format(name)
            )
        expected_start = end
    if expected_start != data_size:
        raise SafetensorsError("tensor byte ranges do not cover the data buffer")

    return header_bytes, header, data_start


def _header_and_start(path, header_info, data_start):
    if header_info is None:
        _, header, start = read_header(path)
        return header, start
    if isinstance(header_info, dict):
        if data_start is None:
            _, _, start = read_header(path)
        else:
            start = data_start
        return header_info, start
    if isinstance(header_info, (tuple, list)):
        if len(header_info) == 3:
            _, header, start = header_info
        elif len(header_info) == 2:
            header, start = header_info
        else:
            raise TypeError("header info must have two or three items")
        if data_start is not None:
            start = data_start
        return header, start
    raise TypeError("header info must be a dict or a tuple")


def read_tensor(path, name, header_info=None, data_start=None):
    """Read one tensor as a float32 numpy array without reading the whole file."""
    header, start = _header_and_start(path, header_info, data_start)
    if name == _METADATA_KEY or name not in header:
        raise SafetensorsError("no tensor named {!r}".format(name))

    entry = header[name]
    if not isinstance(entry, dict) or set(entry) != _TENSOR_KEYS:
        raise SafetensorsError("tensor entry {!r} is malformed".format(name))

    dtype = entry["dtype"]
    shape = tuple(entry["shape"])
    begin, end = entry["data_offsets"]

    if dtype not in ("BF16", "F32"):
        raise SafetensorsError("unsupported dtype {!r} for tensor {!r}".format(dtype, name))

    length = end - begin
    with open(path, "rb") as handle:
        handle.seek(start + begin)
        raw = handle.read(length)
    if len(raw) != length:
        raise SafetensorsError("tensor {!r} is truncated".format(name))

    if dtype == "BF16":
        values = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        values = values << np.uint32(16)
        array = values.view(np.float32)
    else:
        array = np.frombuffer(raw, dtype="<f4").copy()

    return array.reshape(shape)


def _expected_from_manifest(manifest):
    tensors = manifest.get("tensors")
    if not isinstance(tensors, dict):
        raise SafetensorsError("manifest has no tensor map")
    expected = {}
    for name, spec in tensors.items():
        if isinstance(spec, dict):
            if "dtype" not in spec or "shape" not in spec:
                raise SafetensorsError("manifest entry {!r} is incomplete".format(name))
            dtype = spec["dtype"]
            shape = spec["shape"]
        elif isinstance(spec, (tuple, list)) and len(spec) == 2:
            dtype, shape = spec
        else:
            raise SafetensorsError("manifest entry {!r} is malformed".format(name))
        expected[name] = (dtype, tuple(shape))
    return expected


def iter_tensors(path, manifest=None, check_header_sha256=True, finite_only=True):
    """Yield (name, float32 array) in sorted name order."""
    header_bytes, header, data_start = read_header(path)
    actual_names = set(header) - {_METADATA_KEY}

    expected = None
    if manifest is not None:
        source = manifest.get("model_safetensors", {})
        if check_header_sha256:
            expected_sha = source.get("header_json_sha256")
            if not isinstance(expected_sha, str):
                raise SafetensorsError("manifest has no header sha256")
            actual_sha = hashlib.sha256(header_bytes).hexdigest()
            if actual_sha.lower() != expected_sha.lower():
                raise SafetensorsError(
                    "header sha256 mismatch: expected {}, got {}".format(
                        expected_sha, actual_sha
                    )
                )

        expected = _expected_from_manifest(manifest)
        if actual_names != set(expected):
            raise SafetensorsError("tensor names do not match the manifest")

        declared_count = source.get("tensor_count", manifest.get("tensor_count"))
        if declared_count is not None:
            if declared_count != len(expected) or declared_count != len(actual_names):
                raise SafetensorsError("tensor count does not match the manifest")

        for name in sorted(actual_names):
            want_dtype, want_shape = expected[name]
            entry = header[name]
            if entry["dtype"] != want_dtype or tuple(entry["shape"]) != tuple(want_shape):
                raise SafetensorsError(
                    "tensor {!r} does not match the manifest dtype or shape".format(name)
                )

    for name in sorted(actual_names):
        array = read_tensor(path, name, (header, data_start))
        if finite_only and not np.all(np.isfinite(array)):
            raise SafetensorsError("tensor {!r} contains a non-finite value".format(name))
        yield name, array
