import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

L = 22
D = 2048
H = 32
G = 4
DH = 64
F = 5632
V = 32000
NE = 0
T = 1024
LT = 10
CS = 12
THETA0 = 10000
S = 12
RX = 8
EG = 8
SMAX = 4095
CW = 12
WEIGHT_BOUND = 4095
EMBEDDING_BOUND = 2**21
U0 = 2307
C0_BITS = 0x3FB504F3
CQ_BITS = 0x3E3504F3
PAD_ID = 0
GCL = 1
CONTRACT_ID = "TL-INT-1"

RANGE_CONDITIONS = {
    1: ("Vx+eps", 0, 2**78 - 1),
    2: ("norm quotient w", -32768, 32639),
    3: ("qw, kw", -32768, 32639),
    4: ("vw", -32768, 32639),
    6: ("gh", -16384, 16383),
    7: ("u", -32768, 32639),
    8: ("stream x read by a norm", -547599908735, 547599908735),
}

ITEMSIZE = {"u32": 4, "i16": 2, "i32": 4, "json": 0}

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TEST_TABLES_PATH = REPOSITORY_ROOT / "tests" / "tinyllama" / "test_tables.py"


def canonical_json_bytes(obj):
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    return text.encode("ascii")


@lru_cache(maxsize=None)
def test_tables_module():
    spec = importlib.util.spec_from_file_location("synth_artifact_test_tables", TEST_TABLES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layer_paths(layer):
    ll = "%02d" % layer
    return {
        "wq": "layers/%s.wq.i16" % ll,
        "wk": "layers/%s.wk.i16" % ll,
        "wv": "layers/%s.wv.i16" % ll,
        "wo": "layers/%s.wo.i16" % ll,
        "wg": "layers/%s.wg.i16" % ll,
        "wu": "layers/%s.wu.i16" % ll,
        "wd": "layers/%s.wd.i16" % ll,
    }


def expected_specs():
    specs = {
        "KL.bin": ("u32", [L, 13]),
        "emb.i32": ("i32", [V, D]),
        "wh.i16": ("i16", [V, D]),
        "params.json": ("json", []),
        "exponents.json": ("json", []),
        "tables/rope_c.i32": ("i32", [T, DH // 2]),
        "tables/rope_s.i32": ("i32", [T, DH // 2]),
        "tables/softmax_ex.i32": ("i32", [U0]),
        "tables/silu_ts.i32": ("i32", [1 << 16]),
    }
    for layer in range(L):
        paths = layer_paths(layer)
        specs[paths["wq"]] = ("i16", [H * DH, D])
        specs[paths["wk"]] = ("i16", [G * DH, D])
        specs[paths["wv"]] = ("i16", [G * DH, D])
        specs[paths["wo"]] = ("i16", [D, H * DH])
        specs[paths["wg"]] = ("i16", [F, D])
        specs[paths["wu"]] = ("i16", [F, D])
        specs[paths["wd"]] = ("i16", [D, F])
    return specs


def file_size(dtype, shape):
    return math.prod(shape) * ITEMSIZE[dtype]


def table_arrays():
    tt = test_tables_module()
    cosines, sines = tt._rotary_tables()
    return {
        "rope_c": np.asarray(cosines, dtype=np.int32).reshape(T, DH // 2),
        "rope_s": np.asarray(sines, dtype=np.int32).reshape(T, DH // 2),
        "softmax_ex": np.asarray(tt._exp_table(), dtype=np.int32),
        "silu_ts": np.asarray(tt._silu_table(), dtype=np.int32),
    }


def table_bytes():
    arrays = table_arrays()
    paths = {
        "rope_c": "tables/rope_c.i32",
        "rope_s": "tables/rope_s.i32",
        "softmax_ex": "tables/softmax_ex.i32",
        "silu_ts": "tables/silu_ts.i32",
    }
    data = {}
    shas = {}
    for name, path in paths.items():
        raw = arrays[name].astype("<i4", copy=False).tobytes(order="C")
        data[path] = raw
        shas[path] = hashlib.sha256(raw).hexdigest()
    return data, shas


def kl_words():
    words = [[20, 20, 20, 20, 20, 20, 20, 37, 32, 37, 32, 8, 1] for _ in range(L)]
    return words


def kl_bytes(words):
    flat = [word for row in words for word in row]
    return struct.pack("<%dI" % len(flat), *flat)


def expected_eps(e_res):
    return (4096 * (4**e_res) + 100000) // 200000


def params(table_sha):
    tt = test_tables_module()
    digests = {name: list(words) for name, words in tt.CONTRACT_DIGEST_WORDS.items()}
    eps = expected_eps(11)
    return {
        "contract": CONTRACT_ID,
        "L": L,
        "d": D,
        "H": H,
        "G": G,
        "dh": DH,
        "f": F,
        "V": V,
        "T": T,
        "NE": NE,
        "lt": LT,
        "cs": CS,
        "theta0": THETA0,
        "S": S,
        "rx": RX,
        "eg": EG,
        "smax": SMAX,
        "cw": CW,
        "weight_bound": WEIGHT_BOUND,
        "embedding_bound": EMBEDDING_BOUND,
        "u0": U0,
        "c0_bits": C0_BITS,
        "cq_bits": CQ_BITS,
        "pad_id": PAD_ID,
        "gcl": GCL,
        "range_conditions": {
            str(key): list(value) for key, value in RANGE_CONDITIONS.items()
        },
        "e_res": 11,
        "eps1": [eps] * L,
        "eps2": [eps] * L,
        "epsF": eps,
        "MF": 37,
        "KF": 32,
        "s": 20,
        "digests": digests,
        "tables_sha256": dict(table_sha),
    }


def exponents():
    return {"emb": {"e": 11, "mx": 0}}


def provenance():
    return {
        "source": {
            "repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "model_safetensors_sha256": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
            "flashprove_base_commit": "9e2bdd8ff5bf9c62f18263d7dac189b0400b260c",
        },
        "converter": {
            "git_commit": "synthetic",
            "command": "synthetic artifact from the I2 text",
            "python": "synthetic",
            "numpy": "synthetic",
        },
        "calibration": {
            "set": "TL-CAL-1",
            "fit_sha256": "9b2b8fc6b98904de5c905bcd7eecfd2f7895ddacef28d249ca44e3b594fa2f31",
            "heldout_sha256": "5c9da767e78fdb9b31729a0031aa7c07b9c053ceb5ed976c6542bdf31e83bfb8",
        },
    }


@lru_cache(maxsize=None)
def sha256_zeros(size):
    digest = hashlib.sha256()
    chunk = b"\x00" * (1 << 20)
    remaining = size
    while remaining:
        take = min(remaining, len(chunk))
        digest.update(chunk[:take])
        remaining -= take
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-write")
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)


def write_sparse(path, size, edits=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-write")
    with open(temporary, "wb") as handle:
        handle.truncate(size)
        for offset, data in edits:
            handle.seek(offset)
            handle.write(data)
    os.replace(temporary, path)


def zero_file_sizes():
    sizes = {
        "emb.i32": file_size("i32", [V, D]),
        "wh.i16": file_size("i16", [V, D]),
    }
    for layer in range(L):
        paths = layer_paths(layer)
        sizes[paths["wq"]] = file_size("i16", [H * DH, D])
        sizes[paths["wk"]] = file_size("i16", [G * DH, D])
        sizes[paths["wv"]] = file_size("i16", [G * DH, D])
        sizes[paths["wo"]] = file_size("i16", [D, H * DH])
        sizes[paths["wg"]] = file_size("i16", [F, D])
        sizes[paths["wu"]] = file_size("i16", [F, D])
        sizes[paths["wd"]] = file_size("i16", [D, F])
    return sizes


def build_valid_artifact(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".synth-building-", dir=str(root)))
    try:
        table_data, table_sha = table_bytes()
        small = {
            "KL.bin": kl_bytes(kl_words()),
            "params.json": canonical_json_bytes(params(table_sha)),
            "exponents.json": canonical_json_bytes(exponents()),
        }
        small.update(table_data)

        for path, data in small.items():
            write_bytes(staging / path, data)
        sizes = zero_file_sizes()
        for path, size in sizes.items():
            write_sparse(staging / path, size)

        specs = expected_specs()
        entries = []
        for path in sorted(specs):
            dtype, shape = specs[path]
            if path in sizes:
                size = sizes[path]
                digest = sha256_zeros(size)
            else:
                size = len(small[path])
                digest = hashlib.sha256(small[path]).hexdigest()
            entries.append(
                {
                    "path": path,
                    "dtype": dtype,
                    "shape": list(shape),
                    "bytes": size,
                    "sha256": digest,
                }
            )

        prov = provenance()
        manifest = {
            "format": "tl-int-artifact",
            "format_version": 1,
            "contract": CONTRACT_ID,
            "source": prov["source"],
            "converter": prov["converter"],
            "calibration": prov["calibration"],
            "files": entries,
        }
        manifest_raw = canonical_json_bytes(manifest)
        write_bytes(staging / "manifest.json", manifest_raw)

        artifact_id = hashlib.sha256(manifest_raw).hexdigest()
        target = root / artifact_id
        if target.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return target
        os.rename(staging, target)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="ascii"))


def write_manifest(directory, manifest):
    raw = canonical_json_bytes(manifest)
    write_bytes(Path(directory) / "manifest.json", raw)
    return hashlib.sha256(raw).hexdigest()


def manifest_entry(manifest, path):
    for entry in manifest["files"]:
        if entry.get("path") == path:
            return entry
    raise KeyError(path)


def rewrite_file(
    directory,
    relative_path,
    *,
    data=None,
    size=None,
    edits=(),
    update_manifest=False,
    update_sha=True,
    update_bytes=True,
):
    path = Path(directory) / relative_path
    if data is not None:
        write_bytes(path, data)
    else:
        write_sparse(path, size, edits)
    if update_manifest:
        manifest = read_manifest(directory)
        entry = manifest_entry(manifest, relative_path)
        if update_bytes:
            entry["bytes"] = path.stat().st_size
        if update_sha:
            entry["sha256"] = file_sha256(path)
        write_manifest(directory, manifest)
    return path


def rewrite_json(directory, relative_path, obj, *, update_manifest=True):
    return rewrite_file(
        directory,
        relative_path,
        data=canonical_json_bytes(obj),
        update_manifest=update_manifest,
    )


def copy_artifact(source, destination):
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(str(destination))
    destination.mkdir(parents=True)
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        for name in dirs:
            (destination / relative / name).mkdir(parents=True, exist_ok=True)
        for name in files:
            os.link(root_path / name, destination / relative / name)
    return destination
