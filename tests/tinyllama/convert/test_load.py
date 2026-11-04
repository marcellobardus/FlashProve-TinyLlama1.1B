import hashlib
import json
import struct

import numpy as np
import pytest

from tinyllama import contract
from tinyllama import manifest as manifest_module
from tinyllama.convert.load import SafetensorsError, iter_tensors, read_header, read_tensor
from tinyllama.manifest import (
    ManifestMismatch,
    check_file_sha256,
    expected_tensors,
    load_manifest,
    sha256_file,
)


def _header_bytes(header):
    if isinstance(header, (bytes, bytearray)):
        return bytes(header)
    return json.dumps(header, separators=(",", ":")).encode("utf-8")


def _write_raw(path, header, data=b""):
    header_bytes = _header_bytes(header)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)
    return path


def _build_safetensors(path, tensors, metadata=None, header_override=None):
    entries = {}
    if metadata is not None:
        entries["__metadata__"] = metadata

    data = bytearray()
    for name, dtype, shape, raw in tensors:
        begin = len(data)
        data.extend(raw)
        end = len(data)
        entries[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, end],
        }

    header = entries if header_override is None else header_override
    header_bytes = _header_bytes(header)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(data))
    return path


def _stored_header_bytes(path):
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    return raw[8 : 8 + length]


def _bf16_bytes(patterns):
    return struct.pack("<{}H".format(len(patterns)), *patterns)


def _manifest_for(path, tensors):
    return {
        "model_safetensors": {
            "header_json_sha256": hashlib.sha256(_stored_header_bytes(path)).hexdigest(),
            "tensor_count": len(tensors),
        },
        "tensors": {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, dtype, shape, _ in tensors
        },
    }


def _tensor_header(entry):
    return {"t": entry}


def test_bf16_exact_bit_patterns(tmp_path):
    patterns = [0x3F80, 0xBF80, 0x4049, 0x0001, 0x7F7F, 0x8000]
    path = _build_safetensors(
        tmp_path / "bf16.safetensors",
        [("x", "BF16", [len(patterns)], _bf16_bytes(patterns))],
    )
    _, header, data_start = read_header(path)
    array = read_tensor(path, "x", (header, data_start))

    assert array.dtype == np.float32
    assert array[0] == np.float32(1.0)
    assert array[1] == np.float32(-1.0)
    assert array[2] == np.float32(3.140625)
    assert float(array[3]) == 2.0**-133
    assert array[4] == np.float32(3.3895313892515355e38)
    assert array[5] == 0.0
    assert np.signbit(array[5])


def test_bf16_all_patterns_widen_exactly(tmp_path):
    patterns = np.arange(65536, dtype=np.uint16)
    path = _build_safetensors(
        tmp_path / "all-bf16.safetensors",
        [("all", "BF16", [65536], patterns.astype("<u2").tobytes())],
    )
    array = read_tensor(path, "all")
    bits = array.view(np.uint32)

    np.testing.assert_array_equal(bits >> np.uint32(16), patterns.astype(np.uint32))
    np.testing.assert_array_equal(bits & np.uint32(0xFFFF), np.zeros(65536, dtype=np.uint32))


def test_f32_passthrough(tmp_path):
    values = np.array([1.5, -0.0, np.inf, np.nan, 3.25], dtype="<f4")
    path = _build_safetensors(
        tmp_path / "f32.safetensors",
        [("x", "F32", [values.size], values.tobytes())],
    )
    array = read_tensor(path, "x")

    assert array.dtype == np.float32
    np.testing.assert_array_equal(array.view(np.uint32), values.view(np.uint32))


def test_read_header_rejects_short_file(tmp_path):
    path = tmp_path / "short.safetensors"
    path.write_bytes(b"\0" * 7)
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_zero_length(tmp_path):
    path = tmp_path / "zero.safetensors"
    path.write_bytes(struct.pack("<Q", 0))
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_huge_length(tmp_path):
    path = tmp_path / "huge.safetensors"
    path.write_bytes(struct.pack("<Q", 100_000_001) + b"{}")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_header_past_end(tmp_path):
    path = tmp_path / "past-end.safetensors"
    path.write_bytes(struct.pack("<Q", 10) + b"{}")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_invalid_utf8(tmp_path):
    path = _write_raw(tmp_path / "utf8.safetensors", b"\xff\xfe", b"")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_invalid_json(tmp_path):
    path = _write_raw(tmp_path / "json.safetensors", b"{not json", b"")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_duplicate_keys(tmp_path):
    path = _write_raw(
        tmp_path / "duplicate.safetensors",
        b'{"__metadata__":{"a":"1","a":"2"}}',
        b"",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_non_object_top_level(tmp_path):
    path = _write_raw(tmp_path / "array.safetensors", b"[]", b"")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_missing_entry_key(tmp_path):
    path = _write_raw(
        tmp_path / "missing.safetensors",
        {"t": {"dtype": "BF16", "shape": [1]}},
        b"\0\0",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_extra_entry_key(tmp_path):
    path = _write_raw(
        tmp_path / "extra.safetensors",
        {"t": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2], "extra": 1}},
        b"\0\0",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_non_object_entry(tmp_path):
    path = _write_raw(tmp_path / "entry.safetensors", {"t": 1}, b"")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_non_string_dtype(tmp_path):
    path = _write_raw(
        tmp_path / "dtype.safetensors",
        _tensor_header({"dtype": 1, "shape": [1], "data_offsets": [0, 2]}),
        b"\0\0",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


@pytest.mark.parametrize("shape", [1, [-1], [True], [1.5]])
def test_read_header_rejects_bad_shape(tmp_path, shape):
    path = _write_raw(
        tmp_path / "shape.safetensors",
        _tensor_header({"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}),
        b"",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


@pytest.mark.parametrize(
    "offsets",
    [[0], [0, 2, 4], ["0", 2], [False, 2], [-1, 1], [2, 1]],
)
def test_read_header_rejects_bad_data_offsets(tmp_path, offsets):
    path = _write_raw(
        tmp_path / "offsets.safetensors",
        _tensor_header({"dtype": "BF16", "shape": [1], "data_offsets": offsets}),
        b"\0\0",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_offset_size_mismatch(tmp_path):
    path = _write_raw(
        tmp_path / "size.safetensors",
        _tensor_header({"dtype": "BF16", "shape": [2], "data_offsets": [0, 2]}),
        b"\0\0",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_end_beyond_data(tmp_path):
    path = _write_raw(
        tmp_path / "beyond.safetensors",
        _tensor_header({"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}),
        b"",
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_overlap(tmp_path):
    header = {
        "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "b": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]},
    }
    path = _write_raw(tmp_path / "overlap.safetensors", header, b"\0\0\0\0")
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_gap(tmp_path):
    header = {
        "a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
        "b": {"dtype": "BF16", "shape": [1], "data_offsets": [4, 6]},
    }
    path = _write_raw(tmp_path / "gap.safetensors", header, b"\0" * 6)
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_uncovered_data(tmp_path):
    path = _write_raw(
        tmp_path / "uncovered.safetensors",
        _tensor_header({"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}),
        b"\0" * 4,
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_metadata_not_object(tmp_path):
    path = _write_raw(
        tmp_path / "metadata.safetensors", {"__metadata__": []}, b""
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_read_header_rejects_metadata_non_string_value(tmp_path):
    path = _write_raw(
        tmp_path / "metadata-value.safetensors", {"__metadata__": {"x": 1}}, b""
    )
    with pytest.raises(SafetensorsError):
        read_header(path)


def test_iter_tensors_rejects_header_sha_mismatch(tmp_path):
    tensors = [("a", "BF16", [1], _bf16_bytes([0x3F80]))]
    path = _build_safetensors(tmp_path / "sha.safetensors", tensors)
    manifest = _manifest_for(path, tensors)
    manifest["model_safetensors"]["header_json_sha256"] = "00" * 32

    with pytest.raises(SafetensorsError):
        list(iter_tensors(path, manifest))


def test_iter_tensors_rejects_shape_mismatch(tmp_path):
    tensors = [("a", "BF16", [1], _bf16_bytes([0x3F80]))]
    path = _build_safetensors(tmp_path / "shape-manifest.safetensors", tensors)
    manifest = _manifest_for(path, tensors)
    manifest["tensors"]["a"]["shape"] = [2]

    with pytest.raises(SafetensorsError):
        list(iter_tensors(path, manifest))


def test_iter_tensors_rejects_name_mismatch(tmp_path):
    tensors = [("a", "BF16", [1], _bf16_bytes([0x3F80]))]
    path = _build_safetensors(tmp_path / "name-manifest.safetensors", tensors)
    manifest = _manifest_for(path, tensors)
    manifest["tensors"]["b"] = manifest["tensors"].pop("a")

    with pytest.raises(SafetensorsError):
        list(iter_tensors(path, manifest))


def test_iter_tensors_rejects_dtype_mismatch(tmp_path):
    tensors = [("a", "BF16", [1], _bf16_bytes([0x3F80]))]
    path = _build_safetensors(tmp_path / "dtype-manifest.safetensors", tensors)
    manifest = _manifest_for(path, tensors)
    manifest["tensors"]["a"]["dtype"] = "F32"

    with pytest.raises(SafetensorsError):
        list(iter_tensors(path, manifest))


def test_iter_tensors_rejects_tensor_count_mismatch(tmp_path):
    tensors = [("a", "BF16", [1], _bf16_bytes([0x3F80]))]
    path = _build_safetensors(tmp_path / "count-manifest.safetensors", tensors)
    manifest = _manifest_for(path, tensors)
    manifest["model_safetensors"]["tensor_count"] = 2

    with pytest.raises(SafetensorsError):
        list(iter_tensors(path, manifest))


def test_iter_tensors_yields_sorted_names(tmp_path):
    tensors = [
        ("b", "BF16", [1], _bf16_bytes([0x4000])),
        ("a", "BF16", [2], _bf16_bytes([0x3F80, 0xBF80])),
    ]
    path = _build_safetensors(
        tmp_path / "sorted.safetensors", tensors, metadata={"format": "synthetic"}
    )
    manifest = _manifest_for(path, tensors)

    items = list(iter_tensors(path, manifest))

    assert [name for name, _ in items] == ["a", "b"]
    np.testing.assert_array_equal(items[0][1], np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(items[1][1], np.array([2.0], dtype=np.float32))


def test_iter_tensors_rejects_non_finite(tmp_path):
    tensors = [("x", "BF16", [2], _bf16_bytes([0x7F80, 0x7FC0]))]
    path = _build_safetensors(tmp_path / "non-finite.safetensors", tensors)
    manifest = _manifest_for(path, tensors)

    with pytest.raises(SafetensorsError):
        list(iter_tensors(path, manifest))

    items = list(iter_tensors(path, manifest, finite_only=False))
    assert len(items) == 1
    assert np.isinf(items[0][1][0])
    assert np.isnan(items[0][1][1])


def _float32_from_bits(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_contract_constants():
    c0 = _float32_from_bits(contract.C0_BITS)
    cq = _float32_from_bits(contract.CQ_BITS)

    assert c0 == 1.41421353816986083984375
    assert cq == c0 / 8
    assert contract.EXP_S == contract.S == 12
    assert contract.L == 22
    assert contract.D == 2048
    assert contract.DQ == 2048
    assert contract.DKV == 256
    assert contract.CONTRACT_ID == "TL-INT-1"
    assert set(contract.RANGE_CONDITIONS) == {1, 2, 3, 4, 6, 7, 8}
    assert contract.RANGE_CONDITIONS[1] == ("Vx+eps", 0, 2**78 - 1)
    assert contract.RANGE_CONDITIONS[8] == (
        "stream x read by a norm",
        -547599908735,
        547599908735,
    )


def test_real_manifest_loads():
    manifest = load_manifest()

    assert len(manifest["tensors"]) == 201
    assert manifest_module.REVISION == "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    assert manifest_module.REPO_ID == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert manifest_module.MODEL_SHA256.startswith("6e6001da")
    assert len(expected_tensors(manifest)) == 201


def test_sha256_file_and_check_file_sha256(tmp_path):
    path = tmp_path / "digest.bin"
    path.write_bytes(b"abcdef")
    expected = hashlib.sha256(b"abcdef").hexdigest()

    assert sha256_file(path) == expected
    assert sha256_file(path, chunk=2) == expected
    assert check_file_sha256(path, expected) == expected

    with pytest.raises(ManifestMismatch):
        check_file_sha256(path, "0" * 64)
