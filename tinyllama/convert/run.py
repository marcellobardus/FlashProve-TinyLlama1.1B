# Run on the pod (repository at /workspace/repo, checkpoint at /workspace/ckpt,
# calibration npy files at /workspace/calib):
#   cd /workspace/repo && PYTHONPATH=. python3 -m tinyllama.convert.run \
#       --checkpoint /workspace/ckpt --out /workspace/repo/artifacts/TL-INT-1 \
#       --calib /workspace/calib/fit.npy --calib-sha256 <FIT_SHA256> \
#       --heldout /workspace/calib/heldout.npy --heldout-sha256 <HELDOUT_SHA256> \
#       --report /workspace/repo/artifacts/TL-INT-1/range-report.json
#
# Add --test-dims '{"L":2,"D":64,"H":4,"G":2,"DH":16,"F":128,"V":256,"T":32}'
# and --manifest PATH only for the synthetic tests of this repository.

"""Build one TL-INT-1 artifact from a BF16 safetensors checkpoint.

The pipeline reads the checkpoint through the pinned manifest one tensor at a
time, applies the q/k row permutation and gain folding, quantizes, runs the
float32 calibration pass, fits the scales of contract 5.2, writes the artifact,
validates it and writes an optional range report.

Peak RAM at the full TinyLlama size, computed from the design:

* hidden state of all calibration windows: 32 * 1024 * 2048 * 4 = 268,435,456
  bytes (256 MiB);
* the embedding is read whole once and copied into the hidden state, so the
  peak during that step is about 512 MiB;
* one layer of float32 weights: wq and wo are 16 MiB each, wk and wv are 2 MiB
  each, and wg, wu and wd are 44 MiB each, so about 168 MiB;
* int16 artifact matrices: 2,068,840,448 bytes (1.93 GiB), and the int32
  embedding: 262,144,000 bytes (0.24 GiB).

The calibration pass peaks at about 680 MiB. The fold keeps the int16 matrices
as they are made, so near its end it holds 1.93 GiB of matrices, 0.24 GiB of
embedding and 0.17 GiB of one layer of float32 weights, about 2.35 GiB.
write_artifact then keeps both those arrays and their serialised bytes, so the
overall peak is about 1.93 + 0.24 + 1.93 + 0.24 = 4.35 GiB, plus under 1 MiB of
tables and JSON. All float32 weights are never held at once; each layer's
tensors are dropped before the next layer is read. The matrices are dropped
before the range report, which then holds only one integer layer and one
window.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


class RunError(ValueError):
    """The run cannot continue with the arguments or files that it has."""


def _canonical_json_bytes(obj):
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n"
    return text.encode("ascii")


def _table_sha256(array):
    import numpy as np

    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<i4").tobytes(order="C")
    ).hexdigest()


def _parse_test_dims(text):
    from tinyllama.artifacts import Dims

    try:
        obj = json.loads(text)
    except ValueError as exc:
        raise RunError("--test-dims is not JSON: {}".format(exc))
    if not isinstance(obj, dict):
        raise RunError("--test-dims must be a JSON object")
    names = ("L", "D", "H", "G", "DH", "F", "V", "T")
    missing = [name for name in names if name not in obj]
    if missing:
        raise RunError("--test-dims is missing {}".format(", ".join(missing)))
    values = {}
    for name in names:
        value = obj[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RunError("--test-dims {} must be a positive integer".format(name))
        values[name] = value
    if values["H"] * values["DH"] != values["D"]:
        raise RunError("--test-dims must satisfy H * DH == D")
    if values["H"] % values["G"]:
        raise RunError("--test-dims must satisfy H % G == 0")
    if values["DH"] % 2:
        raise RunError("--test-dims must have an even DH")
    if values["T"] > 1024:
        raise RunError("--test-dims T must be at most 1024")
    return Dims(**values)


def _check_output_location(path, *, repo_root=None, what="output"):
    """Refuse a path inside the git tree unless it is below artifacts/."""
    from tinyllama import manifest as manifest_module

    root = Path(repo_root if repo_root is not None else manifest_module.REPO_ROOT)
    root = root.resolve()
    target = Path(path).resolve()
    try:
        common = Path(os.path.commonpath([str(root), str(target)]))
    except ValueError:
        return
    if common != root:
        return
    relative = target.relative_to(root)
    if not relative.parts or relative.parts[0] != "artifacts":
        raise RunError(
            "refusing to write the {} inside the git tree; use a path below "
            "artifacts/ or a path outside the tree".format(what)
        )


def _safetensors_path(checkpoint):
    path = Path(checkpoint)
    if path.is_file():
        return path
    candidate = path / "model.safetensors"
    if candidate.is_file():
        return candidate
    raise RunError(
        "{} is not a safetensors file and has no model.safetensors".format(checkpoint)
    )


def _provenance_command(argv):
    """The command that is recorded, without the output locations.

    The artifact id is the sha256 of manifest.json, so removing --out and
    --report keeps two runs with different output folders at the same id.
    """
    kept = []
    skip = False
    for index, argument in enumerate(argv):
        if skip:
            skip = False
            continue
        if argument in ("--out", "--report"):
            skip = True
            continue
        if argument.startswith("--out=") or argument.startswith("--report="):
            continue
        kept.append(argument)
    return "python -m tinyllama.convert.run " + " ".join(kept)


def _build_tables(dims):
    import numpy as np

    from tinyllama import contract
    from tinyllama.convert import tables

    cosines, sines = tables.rope_tables(
        dims.T, dims.DH, contract.CS, contract.THETA0
    )
    exp = tables.softmax_table(contract.EXP_S, contract.RX)
    silu = tables.silu_table(contract.EG, contract.SMAX)

    arrays = {
        "rope_c": np.asarray(cosines, dtype=np.int32).reshape(dims.T, dims.DH // 2),
        "rope_s": np.asarray(sines, dtype=np.int32).reshape(dims.T, dims.DH // 2),
        "softmax_ex": np.asarray(exp, dtype=np.int32),
        "silu_ts": np.asarray(silu, dtype=np.int32),
    }
    digests = {
        "trope": tables.table_digest(tables.trope_words(cosines, sines)),
        "texp": tables.table_digest(tables.texp_words(exp)),
        "tsilu": tables.table_digest(tables.tsilu_words(silu)),
    }
    paths = {
        "rope_c": "tables/rope_c.i32",
        "rope_s": "tables/rope_s.i32",
        "softmax_ex": "tables/softmax_ex.i32",
        "silu_ts": "tables/silu_ts.i32",
    }
    shas = {paths[name]: _table_sha256(array) for name, array in arrays.items()}
    return arrays, digests, shas


def _fold_checkpoint(reader, dims, e_res, model_norm_gain):
    """Fold, quantize and collect the artifact matrices and exponents."""
    import numpy as np

    from tinyllama import contract
    from tinyllama.convert import fold

    matrices = {}
    exponents = {"q": {}, "k": {}, "v": {}, "o": {}, "gate": {}, "up": {}, "down": {}}
    exponent_info = {}

    def fold_one(name, gains=None, c_bits=None, permute_heads=None):
        values = reader.tensor(name)
        if permute_heads is not None:
            values = fold.permute_qk(values, permute_heads)
        quantized, exponent, mx = fold.fold_and_quantize(values, gains, c_bits)
        exponent_info[name] = {"e": int(exponent), "mx": float(mx)}
        return quantized, int(exponent)

    embedding = reader.tensor("model.embed_tokens.weight")
    emb = fold.quantize_embedding(embedding, e_res)
    exponent_info["model.embed_tokens.weight"] = {
        "e": int(e_res),
        "mx": float(np.max(np.abs(embedding))),
    }
    del embedding

    wh, e_logit = fold_one("lm_head.weight", model_norm_gain, contract.C0_BITS)
    matrices["wh.i16"] = wh
    exponents["lm_head"] = e_logit

    for layer in range(dims.L):
        ll = "%02d" % layer
        gain1 = reader.tensor(
            "model.layers.{}.input_layernorm.weight".format(layer)
        )
        gain2 = reader.tensor(
            "model.layers.{}.post_attention_layernorm.weight".format(layer)
        )

        quantized, exponent = fold_one(
            "model.layers.{}.self_attn.q_proj.weight".format(layer),
            gain1,
            contract.CQ_BITS,
            dims.H,
        )
        matrices["layers/%s.wq.i16" % ll] = quantized
        exponents["q"][layer] = exponent

        quantized, exponent = fold_one(
            "model.layers.{}.self_attn.k_proj.weight".format(layer),
            gain1,
            contract.C0_BITS,
            dims.G,
        )
        matrices["layers/%s.wk.i16" % ll] = quantized
        exponents["k"][layer] = exponent

        quantized, exponent = fold_one(
            "model.layers.{}.self_attn.v_proj.weight".format(layer),
            gain1,
            contract.C0_BITS,
        )
        matrices["layers/%s.wv.i16" % ll] = quantized
        exponents["v"][layer] = exponent

        quantized, exponent = fold_one(
            "model.layers.{}.self_attn.o_proj.weight".format(layer)
        )
        matrices["layers/%s.wo.i16" % ll] = quantized
        exponents["o"][layer] = exponent

        quantized, exponent = fold_one(
            "model.layers.{}.mlp.gate_proj.weight".format(layer),
            gain2,
            contract.C0_BITS,
        )
        matrices["layers/%s.wg.i16" % ll] = quantized
        exponents["gate"][layer] = exponent

        quantized, exponent = fold_one(
            "model.layers.{}.mlp.up_proj.weight".format(layer),
            gain2,
            contract.C0_BITS,
        )
        matrices["layers/%s.wu.i16" % ll] = quantized
        exponents["up"][layer] = exponent

        quantized, exponent = fold_one(
            "model.layers.{}.mlp.down_proj.weight".format(layer)
        )
        matrices["layers/%s.wd.i16" % ll] = quantized
        exponents["down"][layer] = exponent

        del gain1, gain2, quantized

    return emb, matrices, exponents, exponent_info


def main(argv=None):
    """Run the converter. Return 0 on success."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="python -m tinyllama.convert.run",
        description="Build one TL-INT-1 artifact from a BF16 safetensors checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, metavar="DIR")
    parser.add_argument("--out", required=True, metavar="ARTIFACT_PARENT_DIR")
    parser.add_argument("--calib", required=True, metavar="FIT.npy")
    parser.add_argument("--calib-sha256", required=True, metavar="HEX")
    parser.add_argument("--heldout", required=True, metavar="HELDOUT.npy")
    parser.add_argument("--heldout-sha256", required=True, metavar="HEX")
    parser.add_argument("--e-res", type=int, default=11, metavar="N")
    parser.add_argument("--report", default=None, metavar="REPORT.json")
    parser.add_argument("--test-dims", default=None, metavar="JSON")
    parser.add_argument("--manifest", default=None, metavar="PATH")
    args = parser.parse_args(argv)

    try:
        return _run(args, argv)
    except (RunError, ValueError, OSError) as exc:
        print("tinyllama.convert.run: {}".format(exc), file=sys.stderr)
        return 2


def _run(args, argv):
    import numpy as np

    from tinyllama import contract
    from tinyllama import manifest as manifest_module
    from tinyllama.artifacts import FULL, load_artifact, validate_artifact, write_artifact
    from tinyllama.convert.calibrate import (
        CheckpointReader,
        FloatModel,
        calibrate,
        load_windows,
        range_report,
    )
    from tinyllama.convert.fit import build_params, fit_scales

    _check_output_location(args.out, what="artifact")
    if args.report is not None:
        _check_output_location(args.report, what="report")
    if not 3 <= args.e_res <= 14:
        raise RunError("--e-res is {}, outside 3 .. 14".format(args.e_res))

    dims = FULL if args.test_dims is None else _parse_test_dims(args.test_dims)
    if args.manifest is not None:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = manifest_module.load_manifest()

    safetensors = _safetensors_path(args.checkpoint)
    calib = load_windows(args.calib, args.calib_sha256)
    heldout = load_windows(args.heldout, args.heldout_sha256)
    if calib.shape[1] > dims.T or heldout.shape[1] > dims.T:
        raise RunError("a calibration window is longer than the shape's T")

    reader = CheckpointReader(safetensors, manifest)
    model = FloatModel(reader, dims)

    maxima = calibrate(model, calib)
    model_norm_gain = reader.tensor("model.norm.weight")
    emb, matrices, exponents, exponent_info = _fold_checkpoint(
        reader, dims, args.e_res, model_norm_gain
    )
    del model_norm_gain

    table_arrays, digests, table_shas = _build_tables(dims)
    result = fit_scales(maxima, exponents, dims, e_res=args.e_res)
    params = build_params(result, dims, digests=digests, tables_sha256=table_shas)

    source = {
        "repo_id": manifest.get("repo_id", "unknown"),
        "revision": manifest.get("revision", "unknown"),
        "model_safetensors_sha256": manifest.get("model_safetensors", {}).get(
            "sha256_lfs_etag", "unknown"
        ),
        "flashprove_base_commit": manifest.get("flashprove_base_commit", "unknown"),
    }
    provenance = {
        "source": source,
        "converter": {
            "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
            "command": _provenance_command(argv),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "calibration": {
            "set": "TL-CAL-1",
            "fit_sha256": args.calib_sha256,
            "heldout_sha256": args.heldout_sha256,
        },
    }

    artifact_id = write_artifact(
        args.out,
        matrices,
        emb,
        result.kl,
        params,
        table_arrays,
        provenance,
        dims=dims,
        exponents=exponent_info,
    )
    artifact_dir = Path(args.out) / artifact_id
    del matrices, emb, table_arrays
    failures = validate_artifact(artifact_dir, dims=dims)
    if failures:
        raise RunError("the written artifact fails validation: {!r}".format(failures[:3]))

    report = range_report(
        artifact_dir, heldout, dims=dims, float_model=model
    )
    if report["range_failure_count"]:
        print(
            "range report: {} held-out range failures".format(
                report["range_failure_count"]
            ),
            file=sys.stderr,
        )
    if args.report is not None:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="ascii"
        )

    print(artifact_id)
    return 0
