"""Quality evaluation: float32 TinyLlama versus the TL-INT-1 integer model.

Pod command (run from the repository root, on the GPU pod)::

    python3 tests/tinyllama/eval/quality_eval.py \
        --checkpoint DIR \
        --artifact artifacts/TL-INT-1/<id> \
        --heldout HELDOUT.npy \
        --out-json quality.json \
        --out-md quality.md \
        [--prompts docs/tinyllama/prompts_v1.json] \
        [--max-windows N] [--max-positions N]

The float baseline is Hugging Face transformers float32 on CPU, at the
revision pinned in docs/tinyllama/checkpoint_manifest.json.  The integer
side is the CPU reference of INTERFACES.md I3.3, loaded lazily.  If that
reference (or its ``forward``) is absent, the program prints a clear message
and exits with code 3 (skip); no artificial failure is reported.

Integer logits ``lam`` are int64 in units of 2**-s, where ``s`` is the
artifact's params.json ``s``.  They are compared with the float logits only
after the exact conversion ``lam / 2**s``.

Held-out perplexity uses ``logits[:-1]`` to predict ``ids[1:]`` in every
window, and aggregates by summing nll and token counts across windows.  Top-1
agreement and mean KL use all positions of a prompt or window.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

import metrics

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROMPTS = REPOSITORY_ROOT / "docs" / "tinyllama" / "prompts_v1.json"
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT / "docs" / "tinyllama" / "checkpoint_manifest.json"
)
CONTRACT_ID = "TL-INT-1"


class ReferenceUnavailable(RuntimeError):
    """tinyllama.reference, or its forward function, is not available."""


@dataclasses.dataclass
class IntegerOutput:
    """One integer forward result: int64 logits plus any range failures."""

    lam: np.ndarray
    next_token: int
    range_failures: list

    @property
    def range_failure(self):
        return self.range_failures[0] if self.range_failures else None


def load_manifest(path=None):
    """Load docs/tinyllama/checkpoint_manifest.json."""
    with open(path or DEFAULT_MANIFEST, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompts(path=None):
    """Return ``[(id, token_ids), ...]`` from prompts_v1.json, in file order."""
    with open(path or DEFAULT_PROMPTS, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    return [
        (str(record["id"]), [int(value) for value in record["token_ids"]])
        for record in document["prompts"]
    ]


def load_heldout(path):
    """Load the held-out windows, normally shape (8, 1024), as int32."""
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2:
        raise ValueError(
            "held-out windows must be a 2-D array, got shape %s"
            % (array.shape,)
        )
    return np.asarray(array, dtype=np.int32)


def _load_reference():
    """Import tinyllama.reference lazily and return it, or raise our skip error."""
    try:
        import tinyllama.reference as module
    except ImportError as error:
        raise ReferenceUnavailable(
            "tinyllama.reference is not importable: %s" % (error,)
        ) from error
    forward = getattr(module, "forward", None)
    if not callable(forward):
        raise ReferenceUnavailable(
            "tinyllama.reference has no callable forward function"
        )
    return module


class FloatRunner:
    """Float32 HF transformers baseline, on CPU with deterministic settings."""

    def __init__(self, checkpoint, revision=None, num_threads=None):
        import torch
        from transformers import AutoModelForCausalLM

        if revision is None:
            revision = load_manifest()["revision"]
        self.checkpoint = str(checkpoint)
        self.revision = revision
        self._torch = torch

        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        if num_threads is not None:
            torch.set_num_threads(int(num_threads))

        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            revision=revision,
            torch_dtype=torch.float32,
        )
        self.model.to("cpu")
        self.model.eval()

    def logits(self, token_ids):
        """Return logits for one prompt as a float32 numpy array [n, V]."""
        token_ids = [int(value) for value in token_ids]
        if not token_ids:
            raise ValueError("one or more token ids are required")
        torch = self._torch
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cpu")
        with torch.no_grad():
            output = self.model(input_ids=input_ids)
        return (
            output.logits[0]
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )

    __call__ = logits
    forward = logits


class IntegerRunner:
    """Adapter around tinyllama.reference.forward from INTERFACES.md I3.3."""

    def __init__(self, artifact_dir):
        self.reference = _load_reference()
        from tinyllama.artifacts import load_artifact

        self.artifact_dir = str(artifact_dir)
        self.artifact = load_artifact(artifact_dir)
        self.artifact_id = self.artifact.id
        self.s = int(self.artifact.params["s"])

    def run(self, token_ids):
        """Return IntegerOutput for one prefix.

        The preferred path asks the reference for the ``lam`` record, which
        gives the logits at every position in one forward pass.  If the
        reference does not emit that record, the fallback runs one prefix per
        position and stacks the returned ``lam_last`` values: that costs n
        forward passes (and n(n+1)/2 positions of work) instead of one.
        """
        token_ids = [int(value) for value in token_ids]
        if not token_ids:
            raise ValueError("one or more token ids are required")

        captured = {}

        def record(name, layer, values):
            if name == "lam":
                captured["lam"] = np.asarray(values, dtype=np.int64).copy()

        result = self.reference.forward(
            self.artifact, token_ids, mode="prefix", record=record
        )
        lam = captured.get("lam")

        if lam is None:
            rows = []
            failures = []
            for end in range(1, len(token_ids) + 1):
                prefix_result = self.reference.forward(
                    self.artifact, token_ids[:end], mode="prefix"
                )
                failures.extend(metrics.result_range_failures(prefix_result))
                row = getattr(prefix_result, "lam_last", None)
                if row is None:
                    raise RuntimeError(
                        "reference result has neither a lam record nor lam_last"
                    )
                rows.append(np.asarray(row, dtype=np.int64))
            lam = np.stack(rows, axis=0)
            next_token = int(np.argmax(lam[-1]))
        else:
            lam = np.asarray(lam, dtype=np.int64)
            if lam.ndim == 1:
                lam = lam[np.newaxis, :]
            failures = metrics.result_range_failures(result)
            next_token = getattr(result, "next_token", None)
            if next_token is None:
                next_token = int(np.argmax(lam[-1]))
            else:
                next_token = int(next_token)

        if lam.shape[0] != len(token_ids):
            raise ValueError(
                "reference returned %d positions for %d tokens"
                % (lam.shape[0], len(token_ids))
            )
        return IntegerOutput(lam=lam, next_token=next_token, range_failures=failures)

    __call__ = run
    forward = run


def _jsonable(value):
    """Return a value built only from JSON-serializable Python types."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _iter_prompts(prompts):
    if isinstance(prompts, dict) and "prompts" in prompts:
        prompts = prompts["prompts"]
    for item in prompts:
        if isinstance(item, dict):
            prompt_id = item.get("id", item.get("prompt_id"))
            yield str(prompt_id), [int(value) for value in item["token_ids"]]
        else:
            prompt_id, token_ids = item
            yield str(prompt_id), [int(value) for value in token_ids]


def _call_float(runner, token_ids):
    if hasattr(runner, "logits"):
        result = runner.logits(token_ids)
    elif hasattr(runner, "run"):
        result = runner.run(token_ids)
    elif callable(runner):
        result = runner(token_ids)
    else:
        raise TypeError("float runner has no logits or run method")
    if isinstance(result, dict):
        result = result.get("logits")
    return np.asarray(result, dtype=np.float64)


def _call_integer(runner, token_ids):
    if hasattr(runner, "run"):
        result = runner.run(token_ids)
    elif hasattr(runner, "logits"):
        result = runner.logits(token_ids)
    elif hasattr(runner, "forward"):
        result = runner.forward(token_ids)
    elif callable(runner):
        result = runner(token_ids)
    else:
        raise TypeError("integer runner has no run, logits or forward method")
    return result


def _integer_lam(result):
    if isinstance(result, dict):
        for key in ("lam", "logits", "int_logits"):
            if key in result:
                return np.asarray(result[key], dtype=np.int64)
        raise ValueError("integer result mapping has no lam")
    if hasattr(result, "lam"):
        return np.asarray(result.lam, dtype=np.int64)
    if isinstance(result, (tuple, list)):
        return np.asarray(result[0], dtype=np.int64)
    return np.asarray(result, dtype=np.int64)


def _integer_scale(runner):
    if hasattr(runner, "s"):
        return int(runner.s)
    artifact = getattr(runner, "artifact", None)
    if artifact is not None and hasattr(artifact, "params"):
        return int(artifact.params["s"])
    raise ValueError("integer runner does not expose its artifact scale s")


def evaluate(float_runner, int_runner, prompts, windows):
    """Compare the float and integer models and return a result dictionary.

    ``prompts`` is a sequence of ``(id, token_ids)`` pairs (or prompt records
    with ``id`` and ``token_ids``).  ``windows`` is a sequence of token id
    arrays, normally the (8, 1024) held-out set.
    """
    scale = _integer_scale(int_runner)
    prompt_rows = []
    all_failures = []
    prompt_positions = 0

    for prompt_id, token_ids in _iter_prompts(prompts):
        float_logits = _call_float(float_runner, token_ids)
        int_result = _call_integer(int_runner, token_ids)
        int_lam = _integer_lam(int_result)
        int_logits = metrics.int_logits_to_float(int_lam, scale)
        if float_logits.shape != int_logits.shape:
            raise ValueError(
                "prompt %s: float logits %s versus integer logits %s"
                % (prompt_id, float_logits.shape, int_logits.shape)
            )

        failures = metrics.result_range_failures(int_result)
        all_failures.extend(failures)
        n_positions = int(float_logits.shape[0])
        prompt_positions += n_positions

        next_float = int(np.argmax(float_logits[-1]))
        next_int = int(np.argmax(int_logits[-1]))
        agreement = metrics.top1_agreement(float_logits, int_logits)
        prompt_rows.append(
            {
                "id": prompt_id,
                "n": n_positions,
                "positions": n_positions,
                "top1_agreement": agreement,
                "top1": agreement,
                "last_position_agreement": bool(
                    np.argmax(float_logits[-1]) == np.argmax(int_logits[-1])
                ),
                "last_top1_agreement": bool(
                    np.argmax(float_logits[-1]) == np.argmax(int_logits[-1])
                ),
                "mean_kl": metrics.mean_kl(float_logits, int_logits),
                "next_token_float": next_float,
                "next_token_int": next_int,
                "range_failure_count": len(failures),
                "range_failures": _jsonable(failures),
            }
        )

    top1_count = 0
    heldout_positions = 0
    kl_sum = 0.0
    nll_float_sum = 0.0
    nll_int_sum = 0.0
    token_count = 0
    window_count = 0

    for ids in windows:
        token_ids = [int(value) for value in ids]
        if not token_ids:
            continue
        window_count += 1
        float_logits = _call_float(float_runner, token_ids)
        int_result = _call_integer(int_runner, token_ids)
        int_lam = _integer_lam(int_result)
        int_logits = metrics.int_logits_to_float(int_lam, scale)
        if float_logits.shape != int_logits.shape:
            raise ValueError(
                "held-out window: float logits %s versus integer logits %s"
                % (float_logits.shape, int_logits.shape)
            )

        failures = metrics.result_range_failures(int_result)
        all_failures.extend(failures)

        positions = int(float_logits.shape[0])
        heldout_positions += positions
        top1_count += int(
            np.sum(
                np.argmax(float_logits, axis=-1)
                == np.argmax(int_logits, axis=-1)
            )
        )
        kl_sum += float(np.sum(metrics.kl_divergence(float_logits, int_logits)))

        if positions >= 2:
            targets = np.asarray(token_ids[1:], dtype=np.int64)
            float_ppl = metrics.perplexity(float_logits[:-1], targets)
            int_ppl = metrics.perplexity(int_logits[:-1], targets)
            nll_float_sum += float_ppl.nll_sum
            nll_int_sum += int_ppl.nll_sum
            token_count += int_ppl.count

    heldout_top1 = (
        float(top1_count) / heldout_positions if heldout_positions else None
    )
    heldout_mean_kl = kl_sum / heldout_positions if heldout_positions else None
    if token_count:
        float_ppl = metrics.perplexity_from_sums(nll_float_sum, token_count)
        int_ppl = metrics.perplexity_from_sums(nll_int_sum, token_count)
        ppl_float = float(float_ppl)
        ppl_int = float(int_ppl)
    else:
        ppl_float = None
        ppl_int = None

    heldout = {
        "windows": window_count,
        "positions": heldout_positions,
        "tokens": token_count,
        "top1_agreement": heldout_top1,
        "top1": heldout_top1,
        "mean_kl": heldout_mean_kl,
        "ppl_float": ppl_float,
        "perplexity_float": ppl_float,
        "ppl_int": ppl_int,
        "perplexity_int": ppl_int,
        "nll_float_sum": nll_float_sum,
        "nll_int_sum": nll_int_sum,
        "range_failure_count": len(all_failures) - sum(
            row["range_failure_count"] for row in prompt_rows
        ),
    }

    failures_json = _jsonable(all_failures)
    artifact_id = getattr(int_runner, "artifact_id", None)
    revision = getattr(float_runner, "revision", None)
    counts = {
        "prompts": len(prompt_rows),
        "prompt_positions": prompt_positions,
        "heldout_windows": window_count,
        "heldout_positions": heldout_positions,
        "heldout_tokens": token_count,
        "range_failures": len(all_failures),
    }

    return {
        "contract": CONTRACT_ID,
        "artifact_id": artifact_id,
        "revision": revision,
        "s": scale,
        "prompts": prompt_rows,
        "heldout": heldout,
        "range_failure_count": len(all_failures),
        "range_failure_total": len(all_failures),
        "range_failures": failures_json,
        "metadata": {
            "contract": CONTRACT_ID,
            "artifact_id": artifact_id,
            "revision": revision,
            "s": scale,
            "counts": counts,
        },
    }


def write_json(result, path):
    """Write the evaluation result as JSON with sorted keys."""
    data = json.dumps(_jsonable(result), sort_keys=True, indent=1) + "\n"
    Path(path).write_text(data, encoding="utf-8")


def _format_number(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return ("%." + str(digits) + "g") % value
    return str(value)


def write_markdown(result, path):
    """Write the metrics table (one row per prompt, then held-out aggregate)."""
    lines = [
        "# TinyLlama TL-INT-1 quality evaluation",
        "",
        "- contract: %s" % result.get("contract"),
        "- artifact id: %s" % result.get("artifact_id"),
        "- checkpoint revision: %s" % result.get("revision"),
        "- logit scale s: %s" % result.get("s"),
        "",
        "| Row | Top-1 agreement | Mean KL | PPL float | PPL int | Range failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in result.get("prompts", []):
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                row.get("id"),
                _format_number(row.get("top1_agreement")),
                _format_number(row.get("mean_kl")),
                "n/a",
                "n/a",
                row.get("range_failure_count", 0),
            )
        )

    heldout = result.get("heldout", {})
    lines.append(
        "| held-out aggregate | %s | %s | %s | %s | %s |"
        % (
            _format_number(heldout.get("top1_agreement")),
            _format_number(heldout.get("mean_kl")),
            _format_number(heldout.get("ppl_float")),
            _format_number(heldout.get("ppl_int")),
            heldout.get("range_failure_count", 0),
        )
    )
    lines.append("")
    lines.append(
        "Range failures in total: %s" % result.get("range_failure_count", 0)
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare float32 TinyLlama with the TL-INT-1 integer model."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="local HF checkpoint directory"
    )
    parser.add_argument(
        "--artifact", required=True, help="TL-INT-1 artifact directory"
    )
    parser.add_argument(
        "--heldout", required=True, help="held-out windows .npy file"
    )
    parser.add_argument("--out-json", required=True, help="output JSON path")
    parser.add_argument("--out-md", required=True, help="output Markdown path")
    parser.add_argument(
        "--prompts",
        default=str(DEFAULT_PROMPTS),
        help="prompts_v1.json path (default: %(default)s)",
    )
    parser.add_argument(
        "--max-windows", type=int, default=None, help="use at most N windows"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="truncate prompts and windows to at most N positions",
    )
    parser.add_argument(
        "--num-threads", type=int, default=None, help="torch CPU thread count"
    )
    return parser


def main(argv=None):
    """Run the evaluation and return an exit code (3 means reference absent)."""
    args = build_parser().parse_args(argv)

    try:
        int_runner = IntegerRunner(args.artifact)
    except ReferenceUnavailable as error:
        print(
            "SKIP: the integer reference is unavailable (%s). "
            "This evaluation needs tinyllama.reference.forward from "
            "INTERFACES.md I3.3; install that workstream and rerun."
            % (error,),
            file=sys.stderr,
        )
        return 3

    prompts = load_prompts(args.prompts)
    windows = load_heldout(args.heldout)

    if args.max_positions is not None:
        limit = int(args.max_positions)
        prompts = [
            (prompt_id, token_ids[:limit])
            for prompt_id, token_ids in prompts
        ]
        windows = [window[:limit] for window in windows]
    if args.max_windows is not None:
        windows = windows[: int(args.max_windows)]

    float_runner = FloatRunner(
        args.checkpoint, num_threads=args.num_threads
    )
    result = evaluate(float_runner, int_runner, prompts, windows)
    write_json(result, args.out_json)
    write_markdown(result, args.out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
