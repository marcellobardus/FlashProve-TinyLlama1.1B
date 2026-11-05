import json
import struct

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import synth_artifact as synth


def _normalize(code):
    if isinstance(code, (tuple, list)):
        code = code[0]
    return str(code)[:3]


def _read_kl(directory):
    raw = (directory / "KL.bin").read_bytes()
    return list(struct.unpack("<%dI" % (synth.L * 13), raw))


def _write_kl(directory, words):
    synth.rewrite_file(
        directory,
        "KL.bin",
        data=synth.kl_bytes([words[i : i + 13] for i in range(0, len(words), 13)]),
        update_manifest=True,
    )


def _a01(directory):
    raw = (directory / "manifest.json").read_bytes()
    synth.write_bytes(
        directory / "manifest.json",
        b'{"format":"tl-int-artifact",' + raw[1:],
    )


def _a02(directory):
    manifest = synth.read_manifest(directory)
    manifest["format_version"] = 2
    synth.write_manifest(directory, manifest)


def _a03(directory):
    outside = directory.parent / ("outside-" + directory.name + ".bin")
    synth.write_bytes(outside, b"\x00\x00")
    manifest = synth.read_manifest(directory)
    manifest["files"].append(
        {
            "path": "../" + outside.name,
            "dtype": "i16",
            "shape": [1],
            "bytes": 2,
            "sha256": synth.file_sha256(outside),
        }
    )
    manifest["files"].sort(key=lambda entry: entry["path"])
    synth.write_manifest(directory, manifest)


def _a04(directory):
    (directory / "tables" / "softmax_ex.i32").unlink()


def _a05(directory):
    synth.write_bytes(directory / "unlisted.bin", b"\x00\x00")


def _a06(directory):
    path = "layers/00.wk.i16"
    size = synth.file_size("i16", synth.expected_specs()[path][1])
    synth.rewrite_file(
        directory,
        path,
        size=size - 4,
        update_manifest=True,
        update_sha=True,
        update_bytes=False,
    )


def _a07(directory):
    path = "layers/00.wv.i16"
    size = synth.file_size("i16", synth.expected_specs()[path][1])
    synth.rewrite_file(
        directory,
        path,
        size=size,
        edits=[(0, struct.pack("<h", 1))],
        update_manifest=False,
    )


def _a08(directory):
    manifest = synth.read_manifest(directory)
    entry = synth.manifest_entry(manifest, "layers/00.wk.i16")
    entry["shape"] = [2048, 256]
    synth.write_manifest(directory, manifest)


def _a09(directory):
    path = "layers/00.wk.i16"
    size = synth.file_size("i16", synth.expected_specs()[path][1])
    synth.rewrite_file(
        directory,
        path,
        size=size,
        edits=[(0, struct.pack("<h", 4096))],
        update_manifest=True,
    )


def _a10(directory):
    # A06/A08 cannot also fire: KL.bin's own length rule is checked first.
    synth.rewrite_file(directory, "KL.bin", size=1140, update_manifest=False)


def _a11(directory):
    # kq != kk with one below 16 violates A11 without making sh leave 4 .. 12.
    words = _read_kl(directory)
    words[0] = 15
    words[1] = 14
    _write_kl(directory, words)


def _a12(directory):
    words = _read_kl(directory)
    words[7] = 23
    _write_kl(directory, words)


def _a13(directory):
    words = _read_kl(directory)
    words[12] = 0
    _write_kl(directory, words)


def _a14(directory):
    words = _read_kl(directory)
    words[13 + 11] = 9
    _write_kl(directory, words)


def _a15(directory):
    values = json.loads((directory / "params.json").read_text(encoding="ascii"))
    values["eps1"][0] += 1
    synth.rewrite_json(directory, "params.json", values, update_manifest=True)


def _a16(directory):
    path = "tables/softmax_ex.i32"
    raw = bytearray((directory / path).read_bytes())
    struct.pack_into("<i", raw, 0, 4095)
    synth.rewrite_file(directory, path, data=bytes(raw), update_manifest=True)
    values = json.loads((directory / "params.json").read_text(encoding="ascii"))
    values["tables_sha256"][path] = synth.file_sha256(directory / path)
    synth.rewrite_json(directory, "params.json", values, update_manifest=True)


CASES = {
    "A01": _a01,
    "A02": _a02,
    "A03": _a03,
    "A04": _a04,
    "A05": _a05,
    "A06": _a06,
    "A07": _a07,
    "A08": _a08,
    "A09": _a09,
    "A10": _a10,
    "A11": _a11,
    "A12": _a12,
    "A13": _a13,
    "A14": _a14,
    "A15": _a15,
    "A16": _a16,
}


@pytest.mark.parametrize("expected", sorted(CASES))
def test_malformed_artifact_is_rejected(expected, artifact_copy):
    directory = artifact_copy(expected)
    CASES[expected](directory)

    codes = [_normalize(item) for item in artifacts.validate_artifact(directory)]
    assert expected in codes

    with pytest.raises(artifacts.ArtifactRejected) as caught:
        artifacts.load_artifact(directory)
    assert _normalize(caught.value.code) == expected


def test_valid_artifact_is_accepted(valid_artifact):
    assert artifacts.validate_artifact(valid_artifact) == []
    artifacts.load_artifact(valid_artifact)
