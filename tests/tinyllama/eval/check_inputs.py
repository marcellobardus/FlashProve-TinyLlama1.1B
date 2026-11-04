"""Check the TinyLlama prompt set and the TL-CAL-1 calibration inputs.

Pod command (run from the repository root)::

    python3 tests/tinyllama/eval/check_inputs.py \
        --checkpoint-dir DIR_OR_REPO \
        --parquet PATH/to/wikitext-2-raw-v1/train-00000-of-00001.parquet \
        --out report.json

Optional arguments::

    --fit-npy PATH       write the fit windows (32, 1024) as .npy
    --heldout-npy PATH   write the held-out windows (8, 1024) as .npy
    --skip-calibration   check only the prompt set (TL-PROMPTS-1)
    --skip-prompts       check only the calibration set (TL-CAL-1)

Requirements on the pod: Python 3.13 (or a version whose random.Random and
numpy.lib.format agree with it), numpy, pyarrow, tokenizers and transformers,
plus a local checkpoint snapshot in DIR_OR_REPO.  If DIR_OR_REPO is a Hugging
Face repo id instead of a directory, transformers and huggingface_hub fetch the
pinned files; this program itself never downloads anything.  The imports of
transformers, tokenizers, pyarrow and huggingface_hub are lazy, so the module
can be imported on a machine that has only numpy and pytest.

The prompt check (P5) reads docs/tinyllama/prompts_v1.json and the pinned
revision from docs/tinyllama/checkpoint_manifest.json.  For every prompt it
renders the messages with the pinned chat template, tokenizes the rendered text
with add_special_tokens=True, and compares text, text sha256, ids, ids sha256
(uint32 little endian) and length with the committed values.  It also checks
the JSON's internal hashes without using a tokenizer.

The calibration check follows docs/tinyllama/INTERFACES.md I4.3 (D8): join the
parquet "text" column with two line feeds, encode once with the checkpoint's
tokenizer.json, draw 128 rows of 2048 ids with random.Random(0), keep the first
48 rows with ids 1 and 2 removed, and form 40 windows [1] + K[1023w:1023w+1023].
It compares the parquet, draws, fit and held-out sha256 values with the pinned
ones below.

The report is canonical JSON (sorted keys, indent 1, final line feed) with an
overall "ok" field.  The exit code is 0 if every checked section is ok, else 1.
"""

import argparse
import hashlib
import io
import json
import os
import random
import struct
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS = REPOSITORY_ROOT / "docs" / "tinyllama" / "prompts_v1.json"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "docs" / "tinyllama" / "checkpoint_manifest.json"

# Expected TL-CAL-1 values from docs/tinyllama/INTERFACES.md I4.3 (D8, work/DECISIONS.md).
# They are copied from the document on purpose: this module must not import tinyllama.*.
CALIBRATION_EXPECTED = {
    "n": 2_874_559,
    "draws": 128,
    "draw_length": 2048,
    "draws_used": 48,
    "draws_sha256": "8cd63f617d474124a50ee052eae2fd0be603ea45f191ec8ccb4f443f676fb58c",
    "K": 98_304,
    "window_length": 1024,
    "num_windows": 40,
    "fit_shape": [32, 1024],
    "fit_sha256": "9b2b8fc6b98904de5c905bcd7eecfd2f7895ddacef28d249ca44e3b594fa2f31",
    "heldout_shape": [8, 1024],
    "heldout_sha256": "5c9da767e78fdb9b31729a0031aa7c07b9c053ceb5ed976c6542bdf31e83bfb8",
    "parquet_size": 6_357_543,
    "parquet_sha256": "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
}


def sha256_file(path):
    """Return the sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def npy_bytes(array):
    """Return the bytes of array as a version 1.0 .npy file, '<i4', C order."""
    buffer = io.BytesIO()
    ordered = np.asarray(array, dtype="<i4", order="C")
    np.lib.format.write_array(buffer, ordered, version=(1, 0))
    return buffer.getvalue()


def ids_sha256_u32le(ids):
    """Return sha256 of ids as uint32 little-endian values in order."""
    values = [int(value) for value in ids]
    data = struct.pack("<%dI" % len(values), *values)
    return hashlib.sha256(data).hexdigest()


def text_sha256(text):
    """Return sha256 of text encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def first_difference(actual, expected):
    """Return the first differing index, or None when the lists are equal."""
    limit = min(len(actual), len(expected))
    for index in range(limit):
        if actual[index] != expected[index]:
            return index
    if len(actual) != len(expected):
        return limit
    return None


_MISSING = object()


def _ids_from_mapping(value):
    """Return value["input_ids"] for a Mapping or a keys()/[key] object."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(
                "mapping has no 'input_ids' key: %r" % (type(value).__name__,)
            )
        return value["input_ids"]
    try:
        contains = "input_ids" in value.keys()
    except Exception:
        contains = False
    if not contains:
        return _MISSING
    try:
        return value["input_ids"]
    except Exception as error:
        raise ValueError(
            "cannot read 'input_ids' from %s: %s"
            % (type(value).__name__, error)
        ) from error


def _normalize_sequence(sequence):
    """Flatten one row of a list or tuple, or a single-row batch, to ints."""
    if len(sequence) == 0:
        return []
    first = sequence[0]
    if isinstance(first, (list, tuple)):
        rows = []
        for row in sequence:
            if not isinstance(row, (list, tuple)):
                raise ValueError(
                    "expected a flat list of ids or a single batch row, got a "
                    "mixed sequence"
                )
            rows.append(_normalize_sequence(row))
        if len(rows) != 1:
            raise ValueError(
                "expected a single batch row, got %d rows" % len(rows)
            )
        return rows[0]
    ids = []
    for item in sequence:
        if isinstance(item, (list, tuple, str, bytes)):
            raise ValueError(
                "expected integer ids, got %s" % (type(item).__name__,)
            )
        try:
            ids.append(int(item))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "expected integer ids, got %r" % (item,)
            ) from error
    return ids


def normalize_ids(value):
    """Return a flat list of Python ints from a tokenizer's ids container.

    Accepted shapes are a list or tuple of ints, a single-row nested list such
    as [[...]], a Mapping or BatchEncoding-like object with an "input_ids" key,
    an object with an "input_ids" attribute, an object with an ".ids" attribute
    (tokenizers Encoding), and an object with ".tolist()" (numpy array or tensor,
    including shape (1, n)).  A batch of more than one row and unknown types,
    including str, raise ValueError.
    """
    seen = set()
    while True:
        if isinstance(value, str):
            raise ValueError("cannot normalize ids from str: %r" % (value,))
        if isinstance(value, (list, tuple)):
            return _normalize_sequence(value)
        if id(value) in seen:
            raise ValueError("cannot normalize ids from %s" % type(value).__name__)
        seen.add(id(value))

        if isinstance(value, Mapping) or callable(getattr(value, "keys", None)):
            extracted = _ids_from_mapping(value)
            if extracted is not _MISSING:
                value = extracted
                continue

        attribute = getattr(value, "input_ids", _MISSING)
        if attribute is not _MISSING and not callable(attribute):
            value = attribute
            continue

        attribute = getattr(value, "ids", _MISSING)
        if attribute is not _MISSING and not callable(attribute):
            value = attribute
            continue

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
            continue

        raise ValueError(
            "cannot normalize ids from %s" % (type(value).__name__,)
        )


def compare_prompt(record, text, ids):
    """Compare one rendered prompt and its ids with a prompts_v1.json record."""
    ids = [int(value) for value in ids]
    recorded_ids = [int(value) for value in record["token_ids"]]
    got_text_sha = text_sha256(text)
    got_ids_sha = ids_sha256_u32le(ids)
    difference = first_difference(ids, recorded_ids)
    text_match = text == record["text"]
    text_sha_match = got_text_sha == record["text_sha256"]
    ids_match = ids == recorded_ids
    ids_sha_match = got_ids_sha == record["token_ids_sha256_u32le"]
    n_match = len(ids) == record["n"]
    result = {
        "id": record.get("id"),
        "ok": bool(text_match and text_sha_match and ids_match and ids_sha_match and n_match),
        "text_match": text_match,
        "text_sha256": got_text_sha,
        "text_sha256_recorded": record["text_sha256"],
        "text_sha256_match": text_sha_match,
        "ids_match": ids_match,
        "token_ids_match": ids_match,
        "first_diff": difference,
        "first_differing_index": difference,
        "first_mismatch": difference,
        "ids_first_diff": difference,
        "ids_sha256_u32le": got_ids_sha,
        "ids_sha256_recorded": record["token_ids_sha256_u32le"],
        "ids_sha256_match": ids_sha_match,
        "token_ids_sha256_match": ids_sha_match,
        "n": len(ids),
        "n_recorded": record["n"],
        "n_match": n_match,
        "length_match": n_match,
    }
    if not ids_match:
        result["actual_ids"] = ids
        result["recorded_ids"] = recorded_ids
        result["expected_ids"] = recorded_ids
    return result


def ids_subreport(record, ids=None, error=None):
    """Compare one ids source with a prompt record and return its sub-report."""
    recorded_ids = [int(value) for value in record["token_ids"]]
    if error is not None:
        return {
            "ids": None,
            "n": None,
            "n_recorded": record["n"],
            "sha256_u32le": None,
            "sha256_recorded": record["token_ids_sha256_u32le"],
            "sha256_match": False,
            "ids_match": False,
            "first_difference": None,
            "got_id_at_first_difference": None,
            "recorded_id_at_first_difference": None,
            "ok": False,
            "error": error,
        }
    ids = [int(value) for value in ids]
    got_sha = ids_sha256_u32le(ids)
    difference = first_difference(ids, recorded_ids)
    got_id = None
    recorded_id = None
    if difference is not None:
        if difference < len(ids):
            got_id = ids[difference]
        if difference < len(recorded_ids):
            recorded_id = recorded_ids[difference]
    ids_match = ids == recorded_ids
    sha_match = got_sha == record["token_ids_sha256_u32le"]
    return {
        "ids": ids,
        "n": len(ids),
        "n_recorded": record["n"],
        "sha256_u32le": got_sha,
        "sha256_recorded": record["token_ids_sha256_u32le"],
        "sha256_match": sha_match,
        "ids_match": ids_match,
        "first_difference": difference,
        "got_id_at_first_difference": got_id,
        "recorded_id_at_first_difference": recorded_id,
        "ok": bool(ids_match and sha_match and len(ids) == record["n"]),
    }


def check_prompts_json(document):
    """Check the hashes and lengths inside prompts_v1.json without a tokenizer."""
    results = []
    for record in document["prompts"]:
        got_text_sha = text_sha256(record["text"])
        got_ids_sha = ids_sha256_u32le(record["token_ids"])
        text_sha_match = got_text_sha == record["text_sha256"]
        ids_sha_match = got_ids_sha == record["token_ids_sha256_u32le"]
        n_match = len(record["token_ids"]) == record["n"]
        results.append(
            {
                "id": record.get("id"),
                "ok": bool(text_sha_match and ids_sha_match and n_match),
                "text_sha256": got_text_sha,
                "text_sha256_recorded": record["text_sha256"],
                "text_sha256_match": text_sha_match,
                "ids_sha256_u32le": got_ids_sha,
                "ids_sha256_recorded": record["token_ids_sha256_u32le"],
                "ids_sha256_match": ids_sha_match,
                "n": len(record["token_ids"]),
                "n_recorded": record["n"],
                "n_match": n_match,
            }
        )
    return {
        "prompt_set": document.get("prompt_set"),
        "contract": document.get("contract"),
        "ok": all(result["ok"] for result in results),
        "prompts": results,
    }


def build_windows(
    ids,
    seed=0,
    draws=128,
    draws_used=48,
    draw_length=2048,
    remove_ids=(1, 2),
    window_length=1024,
    num_windows=40,
    prefix_id=1,
    stride=None,
):
    """Apply TL-CAL-1 steps 3 to 5 and return (draws, K, windows).

    The keyword defaults are those of docs/tinyllama/INTERFACES.md I4.3.  A
    smaller test can pass smaller draws, draw_length, window_length and
    num_windows; the sampling rule stays randint(0, len(ids) - draw_length - 1)
    with random.Random(seed).
    """
    ids = [int(value) for value in ids]
    if stride is None:
        stride = window_length - 1
    if len(ids) <= draw_length:
        raise ValueError(
            "need more than %d ids to draw rows of %d" % (draw_length, draw_length)
        )

    generator = random.Random(seed)
    starts = [
        generator.randint(0, len(ids) - draw_length - 1) for _ in range(draws)
    ]
    drawn = [ids[start : start + draw_length] for start in starts]
    removed = set(int(value) for value in remove_ids)
    kept = [
        value
        for row in drawn[:draws_used]
        for value in row
        if value not in removed
    ]

    needed = stride * num_windows
    if len(kept) < needed:
        raise ValueError(
            "need %d kept ids for %d windows, got %d"
            % (needed, num_windows, len(kept))
        )
    windows = [
        [int(prefix_id)] + kept[stride * index : stride * index + stride]
        for index in range(num_windows)
    ]
    return drawn, kept, windows


def load_prompt_set(path=None):
    """Load docs/tinyllama/prompts_v1.json."""
    with open(path or DEFAULT_PROMPTS, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(path=None):
    """Load docs/tinyllama/checkpoint_manifest.json."""
    with open(path or DEFAULT_MANIFEST, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_tokenizer(checkpoint_dir, revision):
    """Load transformers.AutoTokenizer for the checkpoint at the pinned revision."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint_dir, revision=revision)


def resolve_tokenizer_json(checkpoint_dir, revision):
    """Return a local tokenizer.json path, downloading only for a repo id."""
    candidate = Path(checkpoint_dir) / "tokenizer.json"
    if candidate.is_file():
        return candidate
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=checkpoint_dir,
            filename="tokenizer.json",
            revision=revision,
        )
    )


def load_calibration_tokenizer(tokenizer_json_path):
    """Load a tokenizers.Tokenizer from a tokenizer.json file."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_json_path))


def check_prompts(tokenizer, document, lib_tokenizer):
    """Check every prompt with the transformers and tokenizers tokenizers.

    The two ids paths are independent: a failure in one makes only its own
    sub-report not ok.  compare_prompt's top-level fields keep their old names
    and are computed from the transformers path for backward compatibility.
    """
    results = []
    for record in document["prompts"]:
        messages = record["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        lib_ids = None
        try:
            lib_ids = normalize_ids(
                lib_tokenizer.encode(text, add_special_tokens=True)
            )
            lib_report = ids_subreport(record, ids=lib_ids)
        except Exception as error:
            lib_report = ids_subreport(
                record, error="%s: %s" % (type(error).__name__, error)
            )

        transformers_ids = None
        try:
            transformers_ids = normalize_ids(
                tokenizer(text, add_special_tokens=True)
            )
            transformers_report = ids_subreport(record, ids=transformers_ids)
        except Exception as error:
            transformers_report = ids_subreport(
                record, error="%s: %s" % (type(error).__name__, error)
            )

        try:
            ids2 = normalize_ids(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
            ids2_report = {
                "ids": ids2,
                "sha256_u32le": ids_sha256_u32le(ids2),
                "matches_committed_ids": ids2
                == [int(value) for value in record["token_ids"]],
                "informational_only": True,
            }
        except Exception as error:  # informational only, never a failure
            ids2_report = {
                "ids": None,
                "error": "%s: %s" % (type(error).__name__, error),
                "informational_only": True,
            }

        compared = compare_prompt(
            record, text, transformers_ids if transformers_ids is not None else []
        )
        compared["ids_tokenizers_lib"] = lib_report
        compared["ids_transformers"] = transformers_report
        compared["ids2"] = ids2_report
        text_ok = bool(compared["text_match"] and compared["text_sha256_match"])
        compared["ok"] = bool(
            text_ok and lib_report["ok"] and transformers_report["ok"]
        )
        results.append(compared)
    return {
        "ok": all(result["ok"] for result in results),
        "results": results,
    }


def check_calibration(
    parquet_path,
    tokenizer_json_path,
    fit_npy=None,
    heldout_npy=None,
    expected=None,
    tokenizer=None,
):
    """Check TL-CAL-1 inputs and return a report dict (optionally write npy files)."""
    expected = dict(CALIBRATION_EXPECTED, **(expected or {}))
    parquet_size = os.path.getsize(parquet_path)
    parquet_sha = sha256_file(parquet_path)

    import pyarrow.parquet as parquet

    table = parquet.read_table(str(parquet_path), columns=["text"])
    rows = table.column("text").to_pylist()
    text = "\n\n".join(rows)

    if tokenizer is None:
        tokenizer = load_calibration_tokenizer(tokenizer_json_path)
    ids = list(tokenizer.encode(text, add_special_tokens=True).ids)

    drawn, kept, windows = build_windows(
        ids,
        seed=0,
        draws=expected["draws"],
        draws_used=expected["draws_used"],
        draw_length=expected["draw_length"],
        remove_ids=(1, 2),
        window_length=expected["window_length"],
        num_windows=expected["num_windows"],
        prefix_id=1,
    )

    draws_sha = hashlib.sha256(npy_bytes(drawn)).hexdigest()
    fit = np.asarray(windows[: expected["fit_shape"][0]], dtype="<i4", order="C")
    heldout = np.asarray(
        windows[
            expected["fit_shape"][0] : expected["fit_shape"][0]
            + expected["heldout_shape"][0]
        ],
        dtype="<i4",
        order="C",
    )
    fit_data = npy_bytes(fit)
    heldout_data = npy_bytes(heldout)
    fit_sha = hashlib.sha256(fit_data).hexdigest()
    heldout_sha = hashlib.sha256(heldout_data).hexdigest()

    if fit_npy:
        Path(fit_npy).write_bytes(fit_data)
    if heldout_npy:
        Path(heldout_npy).write_bytes(heldout_data)

    parquet_ok = (
        parquet_size == expected["parquet_size"]
        and parquet_sha == expected["parquet_sha256"]
    )
    n_ok = len(ids) == expected["n"]
    draws_ok = draws_sha == expected["draws_sha256"]
    kept_ok = len(kept) == expected["K"]
    fit_ok = list(fit.shape) == expected["fit_shape"] and fit_sha == expected["fit_sha256"]
    heldout_ok = (
        list(heldout.shape) == expected["heldout_shape"]
        and heldout_sha == expected["heldout_sha256"]
    )
    return {
        "ok": bool(
            parquet_ok and n_ok and draws_ok and kept_ok and fit_ok and heldout_ok
        ),
        "parquet": {
            "path": str(parquet_path),
            "size": parquet_size,
            "size_expected": expected["parquet_size"],
            "size_match": parquet_size == expected["parquet_size"],
            "sha256": parquet_sha,
            "sha256_expected": expected["parquet_sha256"],
            "sha256_match": parquet_sha == expected["parquet_sha256"],
            "rows": len(rows),
        },
        "tokenizer_json": str(tokenizer_json_path),
        "n": len(ids),
        "n_expected": expected["n"],
        "n_match": n_ok,
        "draws": {
            "count": len(drawn),
            "length": len(drawn[0]) if drawn else 0,
            "sha256": draws_sha,
            "sha256_expected": expected["draws_sha256"],
            "sha256_match": draws_ok,
        },
        "K": {
            "n": len(kept),
            "n_expected": expected["K"],
            "n_match": kept_ok,
        },
        "fit": {
            "shape": list(fit.shape),
            "shape_expected": expected["fit_shape"],
            "sha256": fit_sha,
            "sha256_expected": expected["fit_sha256"],
            "sha256_match": fit_sha == expected["fit_sha256"],
            "written": str(fit_npy) if fit_npy else None,
        },
        "heldout": {
            "shape": list(heldout.shape),
            "shape_expected": expected["heldout_shape"],
            "sha256": heldout_sha,
            "sha256_expected": expected["heldout_sha256"],
            "sha256_match": heldout_sha == expected["heldout_sha256"],
            "written": str(heldout_npy) if heldout_npy else None,
        },
    }


def write_report(report, path):
    """Write the report as canonical JSON with one final line feed."""
    data = json.dumps(report, sort_keys=True, indent=1) + "\n"
    Path(path).write_text(data, encoding="utf-8")


def main(argv=None):
    """Run the requested checks, write the report, and return an exit code."""
    parser = argparse.ArgumentParser(
        description="Check the TL-PROMPTS-1 prompt set and the TL-CAL-1 calibration inputs."
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="local checkpoint directory or Hugging Face repo id",
    )
    parser.add_argument("--parquet", help="WikiText-2 training parquet file")
    parser.add_argument("--out", required=True, help="path of the JSON report")
    parser.add_argument("--fit-npy", help="optional path for the fit windows")
    parser.add_argument(
        "--heldout-npy", help="optional path for the held-out windows"
    )
    parser.add_argument(
        "--skip-calibration", action="store_true", help="check only the prompts"
    )
    parser.add_argument(
        "--skip-prompts", action="store_true", help="check only the calibration set"
    )
    args = parser.parse_args(argv)

    if not args.skip_calibration and not args.parquet:
        parser.error("--parquet is required unless --skip-calibration is given")

    manifest = load_manifest()
    revision = manifest["revision"]
    report = {
        "contract": "TL-INT-1",
        "prompt_set": "TL-PROMPTS-1",
        "checkpoint": args.checkpoint_dir,
        "revision": revision,
    }
    sections = []

    if not args.skip_prompts:
        document = load_prompt_set()
        internal = check_prompts_json(document)
        tokenizer = load_tokenizer(args.checkpoint_dir, revision)
        tokenizer_json = resolve_tokenizer_json(args.checkpoint_dir, revision)
        lib_tokenizer = load_calibration_tokenizer(tokenizer_json)
        prompts = check_prompts(tokenizer, document, lib_tokenizer)
        prompts_ok = bool(internal["ok"] and prompts["ok"])
        report["prompts"] = {
            "ok": prompts_ok,
            "internal": internal,
            "results": prompts["results"],
        }
        sections.append(prompts_ok)
    else:
        report["prompts"] = {"skipped": True}

    if not args.skip_calibration:
        tokenizer_json = resolve_tokenizer_json(args.checkpoint_dir, revision)
        calibration = check_calibration(
            args.parquet,
            tokenizer_json,
            fit_npy=args.fit_npy,
            heldout_npy=args.heldout_npy,
        )
        report["calibration"] = calibration
        sections.append(calibration["ok"])
    else:
        report["calibration"] = {"skipped": True}

    report["ok"] = all(sections) if sections else True
    write_report(report, args.out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
