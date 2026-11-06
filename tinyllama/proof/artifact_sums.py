"""Write a wsums.txt-format sums file from a TL-INT-1 artifact (PROVING_PLAN T4.2).

T4.2 compares what cp_commit will commit against our own artifact. On a CPU pod a
BF16 GGUF is exported from the pinned checkpoint by llama.cpp's converter at commit
e853ec9a, and `bin/weight_sums --eres 11` writes wsums.txt from that GGUF: one line
per committed tensor, holding the tensor's exponent and the SHA-256 of its integers
(weight_sums/README.md, "The file's form"). This script writes a file in the same
format from a TL-INT-1 artifact directory, so the two files can be compared line for
line. Equal files mean the integers cp_commit would convert equal our artifact's.

The artifact already stores the committed integers. Each matrix file layers/LL.wX.i16
holds signed 16-bit integers and emb.i32 holds signed 32-bit integers, both little
endian and in row-major (output-unit) order. That is the same order and the same two's
complement byte layout that weight_sums hashes (weight_sums/README.md: four bytes for
token_embd.weight and two for every other tensor). The q and k matrices are already
row-permuted by the converter, as llama.cpp's export is, so no reordering is needed
(docs/tinyllama/ARITHMETIC_CONTRACT.md sections 4.2 and 4.4; fold.txt line qkrows asis).

Each tensor's exponent is read from the artifact's exponents.json, which the converter
writes keyed by the HuggingFace tensor name (tinyllama/convert/run.py). The embedding's
exponent is e_res, which bin/weight_sums takes from --eres and cp_commit from fold.txt;
this script checks it equals the artifact's e_res and, if --eres is given, that the two
agree. No weight value, and nothing computed from one, is printed: the sums file and
this script's messages are kept as the weights are kept (weight_sums/README.md).

Run it against a real artifact directory with

  python -m tinyllama.proof.artifact_sums --artifact <artifact dir> \
      --eres 11 --out wsums.artifact.txt

and compare with cmp against the wsums.txt that bin/weight_sums --eres 11 wrote.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

EMBEDDING = "token_embd.weight"
HEAD = "output.weight"

# A layer's committed tensors: the GGUF kind, the artifact matrix, and the
# HuggingFace tensor name under which exponents.json records the exponent.
LAYER_KINDS = (
    ("attn_q", "wq", "self_attn.q_proj"),
    ("attn_k", "wk", "self_attn.k_proj"),
    ("attn_v", "wv", "self_attn.v_proj"),
    ("attn_output", "wo", "self_attn.o_proj"),
    ("ffn_gate", "wg", "mlp.gate_proj"),
    ("ffn_up", "wu", "mlp.up_proj"),
    ("ffn_down", "wd", "mlp.down_proj"),
)


class ArtifactSumsError(ValueError):
    """The sums file cannot be built from this artifact."""


def _tensor_specs(artifact):
    """Yield (gguf name, integer array, bytes per integer, exponents.json key) for
    every tensor cp_commit consumes. The order does not matter: the lines are sorted
    by name before they are written."""
    yield EMBEDDING, artifact.emb, 4, "model.embed_tokens.weight"
    yield HEAD, artifact.wh, 2, "lm_head.weight"
    for layer in range(int(artifact.dims.L)):
        for kind, which, suffix in LAYER_KINDS:
            name = "blk.%d.%s.weight" % (layer, kind)
            key = "model.layers.%d.%s.weight" % (layer, suffix)
            yield name, artifact.matrix(layer, which), 2, key


def _exponent(exponents, key, name):
    """The integer exponent for a tensor. The converter writes {"e": int, "mx": float}
    under the HuggingFace name; a bare integer is accepted too."""
    entry = exponents.get(key) if isinstance(exponents, dict) else None
    value = entry.get("e") if isinstance(entry, dict) else entry
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactSumsError(
            "exponents.json has no integer exponent for %s (key %s)" % (name, key)
        )
    return int(value)


def _sum_line(name, array, item_bytes, exponent):
    """One wsums.txt line: name elements exponent bits sha256.

    The hash is over the integers in file order as little-endian two's-complement
    numbers, four bytes for the embedding and two for every other tensor, with
    nothing else (weight_sums/README.md). bits is the least b >= 1 with
    -2^(b-1) <= w < 2^(b-1) for every integer w; the prover reads it but does not
    use it.
    """
    dtype = np.dtype("<i4") if item_bytes == 4 else np.dtype("<i2")
    integers = np.ascontiguousarray(array, dtype=dtype)
    digest = hashlib.sha256(integers.tobytes(order="C")).hexdigest()
    elements = int(integers.size)
    lowest = int(integers.min()) if integers.size else 0
    highest = int(integers.max()) if integers.size else 0
    bits = 1
    while not -(2 ** (bits - 1)) <= lowest <= highest < 2 ** (bits - 1):
        bits += 1
    return "%s %d %d %d %s\n" % (name, elements, exponent, bits, digest)


def build_sums(artifact, eres=None):
    """Return the sorted sums-file lines for a loaded artifact.

    If eres is given it must equal the artifact's e_res, so an operator can assert
    the value they will pass to bin/weight_sums --eres.
    """
    params = artifact.params
    art_eres = params.get("e_res") if isinstance(params, dict) else None
    if not isinstance(art_eres, int) or isinstance(art_eres, bool):
        raise ArtifactSumsError("params.json has no integer e_res")
    if eres is not None and eres != art_eres:
        raise ArtifactSumsError(
            "the artifact's e_res is %d, not the requested --eres %d" % (art_eres, eres)
        )

    lines = []
    for name, array, item_bytes, key in _tensor_specs(artifact):
        exponent = _exponent(artifact.exponents, key, name)
        if name == EMBEDDING and exponent != art_eres:
            raise ArtifactSumsError(
                "the embedding exponent %d is not e_res %d" % (exponent, art_eres)
            )
        lines.append(_sum_line(name, array, item_bytes, exponent))
    lines.sort(key=lambda line: line.split(" ", 1)[0].encode("ascii"))
    return lines


def write_sums_file(path, lines):
    """Write the lines and the closing END line to a new file, readable by its owner
    only, as weight_sums does. The file is created with O_EXCL and is never
    overwritten; a last line END <number of lines before it> closes it."""
    body = "".join(lines) + "END %d\n" % len(lines)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    with os.fdopen(handle, "w", encoding="ascii", newline="") as out:
        out.write(body)


def _dims_from(text):
    from tinyllama.artifacts import Dims

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 8 or any(not part for part in parts):
        raise ArtifactSumsError("--dims-test needs eight integers L,D,H,G,DH,F,V,T")
    try:
        return Dims(*[int(part) for part in parts])
    except ValueError:
        raise ArtifactSumsError("--dims-test needs eight integers L,D,H,G,DH,F,V,T") from None


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m tinyllama.proof.artifact_sums",
        description="Write a wsums.txt-format sums file from a TL-INT-1 artifact (PROVING_PLAN T4.2).",
    )
    parser.add_argument("--artifact", required=True, help="the artifact directory to read")
    parser.add_argument("--out", required=True, help="the sums file to write (created new, owner only)")
    parser.add_argument("--artifact-id", help="expected artifact id, checked against the manifest hash")
    parser.add_argument(
        "--eres", type=int, metavar="N",
        help="assert the artifact's e_res is N, the value for bin/weight_sums --eres",
    )
    parser.add_argument("--dims-test", help="test only: load a tiny artifact with dims L,D,H,G,DH,F,V,T")
    args = parser.parse_args(argv)

    from tinyllama.artifacts import FULL, ArtifactRejected, load_artifact

    try:
        dims = _dims_from(args.dims_test) if args.dims_test else FULL
        artifact = load_artifact(args.artifact, args.artifact_id, dims=dims)
        lines = build_sums(artifact, args.eres)
        write_sums_file(args.out, lines)
    except (ArtifactSumsError, ArtifactRejected) as error:
        print("artifact_sums: %s" % error, file=sys.stderr)
        return 2
    except FileExistsError:
        print(
            "artifact_sums: %s exists already; name a file that does not exist yet" % args.out,
            file=sys.stderr,
        )
        return 2
    print("artifact_sums: wrote %s (tensor lines: %d)" % (args.out, len(lines)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
