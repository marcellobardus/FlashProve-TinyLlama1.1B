"""Small synthetic BF16 checkpoints and windows for the converter tests."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from tinyllama.artifacts import Dims


def bf16_roundtrip(values):
    """Round float32 values to BF16 and widen them back to float32."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounded = (bits + np.uint32(0x00007FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
    top = (rounded >> np.uint32(16)).astype(np.uint32)
    return (top << np.uint32(16)).view(np.float32)


def _bf16_bytes(values):
    array = bf16_roundtrip(values)
    bits = array.view(np.uint32)
    return (bits >> np.uint32(16)).astype("<u2").tobytes(order="C")


def _tensor_shapes(dims):
    shapes = {
        "lm_head.weight": (dims.V, dims.D),
        "model.embed_tokens.weight": (dims.V, dims.D),
        "model.norm.weight": (dims.D,),
    }
    for layer in range(dims.L):
        prefix = "model.layers.{}".format(layer)
        shapes[prefix + ".input_layernorm.weight"] = (dims.D,)
        shapes[prefix + ".post_attention_layernorm.weight"] = (dims.D,)
        shapes[prefix + ".self_attn.q_proj.weight"] = (dims.H * dims.DH, dims.D)
        shapes[prefix + ".self_attn.k_proj.weight"] = (dims.G * dims.DH, dims.D)
        shapes[prefix + ".self_attn.v_proj.weight"] = (dims.G * dims.DH, dims.D)
        shapes[prefix + ".self_attn.o_proj.weight"] = (dims.D, dims.H * dims.DH)
        shapes[prefix + ".mlp.gate_proj.weight"] = (dims.F, dims.D)
        shapes[prefix + ".mlp.up_proj.weight"] = (dims.F, dims.D)
        shapes[prefix + ".mlp.down_proj.weight"] = (dims.D, dims.F)
    return shapes


def _tensor_values(name, shape, rng):
    if name.endswith("layernorm.weight") or name == "model.norm.weight":
        return (1.0 + 0.05 * np.abs(rng.standard_normal(shape))).astype(np.float32)
    if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
        scale = 0.1
    else:
        scale = 0.5
    return (scale * rng.standard_normal(shape)).astype(np.float32)


def write_safetensors(path, tensors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = {}
    data = bytearray()
    for name in sorted(tensors):
        values = tensors[name]
        raw = _bf16_bytes(values)
        begin = len(data)
        data.extend(raw)
        entries[name] = {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [begin, begin + len(raw)],
        }
    header = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(data))
    return header


def build_checkpoint(root, dims=None, seed=20261104):
    """Write model.safetensors and manifest.json under root and return their paths."""
    if dims is None:
        dims = Dims(L=2, D=64, H=4, G=2, DH=16, F=128, V=256, T=32)
    root = Path(root)
    checkpoint_dir = root / "ckpt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    shapes = _tensor_shapes(dims)
    tensors = {
        name: bf16_roundtrip(_tensor_values(name, shape, rng))
        for name, shape in shapes.items()
    }
    safetensors = checkpoint_dir / "model.safetensors"
    header = write_safetensors(safetensors, tensors)

    manifest = {
        "repo_id": "synthetic/TinyLlama-test",
        "revision": "synthetic",
        "flashprove_base_commit": "synthetic",
        "model_safetensors": {
            "header_json_sha256": hashlib.sha256(header).hexdigest(),
            "tensor_count": len(tensors),
            "sha256_lfs_etag": hashlib.sha256(safetensors.read_bytes()).hexdigest(),
        },
        "tensors": {
            name: {"dtype": "BF16", "shape": list(values.shape)}
            for name, values in tensors.items()
        },
    }
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="ascii"
    )
    return {
        "dir": checkpoint_dir,
        "safetensors": safetensors,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "dims": dims,
        "tensors": tensors,
    }


def write_windows(path, dims, n_windows=2, seed=7):
    rng = np.random.default_rng(seed)
    array = rng.integers(0, dims.V, size=(n_windows, dims.T), dtype=np.int32)
    array = np.ascontiguousarray(array, dtype="<i4")
    np.save(path, array, allow_pickle=False)
    return array, hashlib.sha256(Path(path).read_bytes()).hexdigest()
