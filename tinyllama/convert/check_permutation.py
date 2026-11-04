# Pod commands for P1, P2 and P3 (repository at /workspace/repo, checkpoint at
# /workspace/ckpt). Run from an ordinary account, not as root. transformers is
# not pinned in this repository's requirements. The modeling code was read at
# upstream commit 0c7f5ab8, and torch 2.9.1 is the pin in requirements.in, so
# install a current transformers release next to that torch:
#
#   cd /workspace/repo
#   pip install torch==2.9.1 transformers
#   PYTHONPATH=. python3 -m tinyllama.convert.check_permutation p1 \
#       --checkpoint /workspace/ckpt --report /workspace/repo/p1.json
#   PYTHONPATH=. python3 -m tinyllama.convert.check_permutation p2 \
#       --checkpoint /workspace/ckpt --report /workspace/repo/p2.json \
#       --e-res 11
#   PYTHONPATH=. python3 -m tinyllama.convert.check_permutation p2 \
#       --checkpoint /workspace/ckpt --report /workspace/repo/p2.json \
#       --params /workspace/repo/artifacts/TL-INT-1/<artifact_id>/params.json
#   PYTHONPATH=. python3 -m tinyllama.convert.check_permutation p3 \
#       --checkpoint /workspace/ckpt --report /workspace/repo/p3.json
#   PYTHONPATH=. python3 -m tinyllama.convert.check_permutation all \
#       --checkpoint /workspace/ckpt --report /workspace/repo/checks.json
#
# Estimated peak RAM: P1 about 5 to 6 GB (the float32 TinyLlama weights in
# torch plus the numpy float32 forward); P2 about 1 GB (one BF16 tensor widened
# and its quantized copy at a time); P3 a few MB (streamed in 1 MiB chunks).
# The "all" run keeps the P1 model only while P1 runs.

"""Pod checks P1, P2 and P3 of the TL-INT-1 arithmetic contract.

P1 compares the HF half split rotary on layer 0 with the converter's row
permutation and neighbour pair rotary, then compares a full float forward.
P2 is a streaming converter dry run per tensor. P3 compares the sha256 of
model.safetensors with the pinned manifest value. The comparison functions take
plain numpy arrays. torch and transformers are imported only inside the thin HF
helpers, so importing this module stays numpy only.

The relative difference used for the score arrays is
max|a - b| / max|a|, where a is the HF score array and b is the neighbour pair
score array.
"""

import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

from tinyllama import contract
from tinyllama import manifest as manifest_module
from tinyllama.artifacts import FULL
from tinyllama.convert import calibrate, fold
from tinyllama.convert.calibrate import CheckpointReader, FloatModel


SCORE_THRESHOLD = 1e-5
LOGIT_THRESHOLD = 1e-3
DEFAULT_E_RES = 11
PROMPTS_PATH = manifest_module.REPO_ROOT / "docs" / "tinyllama" / "prompts_v1.json"


class CheckError(ValueError):
    """A check cannot run with the arguments or files that it has."""


def _safetensors_path(checkpoint):
    path = Path(checkpoint)
    if path.is_file():
        return path
    candidate = path / "model.safetensors"
    if candidate.is_file():
        return candidate
    raise CheckError(
        "{} is not a safetensors file and has no model.safetensors".format(checkpoint)
    )


def _json_number(value):
    value = float(value)
    if math.isfinite(value):
        return value
    return str(value)


def relative_difference(reference, candidate):
    """Return max|reference - candidate| / max|reference|.

    If the reference is all zero, the result is 0 for an equal candidate and
    infinity for any difference. Both arrays are compared as float64 values.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("arrays have different shapes")
    difference = np.max(np.abs(a - b)) if a.size else 0.0
    scale = np.max(np.abs(a)) if a.size else 0.0
    if scale == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return float(difference / scale)


def attention_scores(q, k):
    """Return the H x n x n score array with query head h reading k head h // group."""
    query = np.asarray(q, dtype=np.float64)
    key = np.asarray(k, dtype=np.float64)
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("q and k must have shape (positions, heads, head_dim)")
    positions, heads, depth = query.shape
    if key.shape[0] != positions:
        raise ValueError("q and k have different position counts")
    if key.shape[2] != depth:
        raise ValueError("q and k have different head dimensions")
    groups = key.shape[1]
    if groups < 1 or heads % groups:
        raise ValueError("query heads must be divisible by key heads")
    per_group = heads // groups
    scores = np.empty((heads, positions, positions), dtype=np.float64)
    for head in range(heads):
        key_head = head // per_group
        scores[head] = np.matmul(query[:, head, :], key[:, key_head, :].T)
    return scores


def compare_score_arrays(reference, candidate, threshold=SCORE_THRESHOLD):
    """Compare two score arrays with the relative difference of this module."""
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    measured = relative_difference(reference_array, candidate_array)
    return {
        "relative_difference_definition": "max|a-b| / max|a|",
        "max_relative_difference": _json_number(measured),
        "threshold": threshold,
        "passed": bool(measured < threshold),
        "shape": list(reference_array.shape),
    }


def compare_logits(reference, candidate, threshold=LOGIT_THRESHOLD):
    """Compare full logits, including the greedy token at the last position."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
        raise ValueError("logits must be two arrays of the same shape")
    difference = np.abs(a - b)
    max_all = float(np.max(difference)) if difference.size else 0.0
    max_last = float(np.max(difference[-1])) if difference.size else 0.0
    hf_token = int(np.argmax(a[-1]))
    numpy_token = int(np.argmax(b[-1]))
    same = hf_token == numpy_token
    return {
        "max_abs_difference": _json_number(max_all),
        "max_abs_difference_last": _json_number(max_last),
        "threshold": threshold,
        "hf_next_token": hf_token,
        "numpy_next_token": numpy_token,
        "same_next_token": bool(same),
        "last_position": int(a.shape[0] - 1),
        "passed": bool(same and max_all < threshold),
    }


def resolve_prompt(prompt_id="P0", prompt_ids=None):
    """Return (token ids, source description) for a prompt id."""
    if prompt_ids is not None:
        values = [int(value) for value in prompt_ids]
        if not values:
            raise CheckError("--prompt-ids is empty")
        return values, "command line --prompt-ids"
    if PROMPTS_PATH.is_file():
        with open(PROMPTS_PATH, "r", encoding="utf-8") as handle:
            document = json.load(handle)
        for prompt in document.get("prompts", []):
            if prompt.get("id") == prompt_id:
                return [int(value) for value in prompt["token_ids"]], str(PROMPTS_PATH)
    raise CheckError(
        "{} token ids are not defined in {}; pass --prompt-ids".format(
            prompt_id, PROMPTS_PATH
        )
    )


def numpy_layer0_qk(reader, token_ids, dims=None, eps=1e-5, theta0=contract.THETA0):
    """Layer 0 q and k after permuted rows and neighbour pair rotary."""
    dims = FULL if dims is None else dims
    ids = np.asarray(token_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size < 1:
        raise CheckError("token ids must be a non-empty one dimensional list")
    if int(ids.min()) < 0 or int(ids.max()) >= dims.V:
        raise CheckError("a token id is outside 0 .. {}".format(dims.V - 1))

    embedding = reader.tensor("model.embed_tokens.weight")
    hidden = np.ascontiguousarray(embedding[ids], dtype=np.float32)
    del embedding
    gain = reader.tensor("model.layers.0.input_layernorm.weight")
    normed = calibrate._rms_norm(hidden, eps)
    a = normed * np.asarray(gain, dtype=np.float32)
    del hidden, gain

    wq = fold.permute_qk(
        reader.tensor("model.layers.0.self_attn.q_proj.weight"), dims.H
    )
    wk = fold.permute_qk(
        reader.tensor("model.layers.0.self_attn.k_proj.weight"), dims.G
    )
    positions = ids.shape[0]
    q = np.matmul(a, wq.T).reshape(positions, dims.H, dims.DH)
    k = np.matmul(a, wk.T).reshape(positions, dims.G, dims.DH)
    del a, wq, wk

    cosines, sines = calibrate._rotary_tables(
        theta0, np.arange(positions), dims.DH
    )
    q = calibrate._rotary_neighbour(q, cosines, sines)
    k = calibrate._rotary_neighbour(k, cosines, sines)
    return q, k


def _require_hf():
    import torch
    import transformers
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    return torch, transformers, apply_rotary_pos_emb


def load_hf_model(checkpoint):
    """Load a float32 LlamaForCausalLM from a local checkpoint directory."""
    torch, _transformers, _apply = _require_hf()
    from transformers import LlamaForCausalLM

    path = str(checkpoint)
    try:
        model = LlamaForCausalLM.from_pretrained(
            path, dtype=torch.float32, local_files_only=True
        )
    except TypeError:
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32, local_files_only=True
        )
    model.eval()
    model.to(torch.float32)
    return model


def _hf_cos_sin(model, hidden, position_ids):
    inner = model.model
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is None:
        rotary = inner.layers[0].self_attn.rotary_emb
    try:
        return rotary(hidden, position_ids)
    except TypeError:
        return rotary(hidden, seq_len=position_ids.shape[-1])


def _hf_apply_rotary(apply_rotary_pos_emb, q, k, cos, sin, position_ids):
    try:
        return apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    except (TypeError, IndexError, RuntimeError):
        return apply_rotary_pos_emb(q, k, cos, sin)


def hf_layer0_qk(model, token_ids):
    """Layer 0 q and k from HF, after apply_rotary_pos_emb."""
    import torch

    _torch, _transformers, apply_rotary_pos_emb = _require_hf()
    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    with torch.no_grad():
        hidden = model.model.embed_tokens(ids)
        position_ids = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)
        layer = model.model.layers[0]
        normed = layer.input_layernorm(hidden)
        attention = layer.self_attn
        query = attention.q_proj(normed)
        key = attention.k_proj(normed)

        config = model.config
        heads = int(config.num_attention_heads)
        groups = int(getattr(config, "num_key_value_heads", heads))
        depth = int(getattr(config, "head_dim", config.hidden_size // heads))
        positions = ids.shape[1]
        query = query.view(1, positions, heads, depth).transpose(1, 2)
        key = key.view(1, positions, groups, depth).transpose(1, 2)

        cos, sin = _hf_cos_sin(model, hidden, position_ids)
        query, key = _hf_apply_rotary(
            apply_rotary_pos_emb, query, key, cos, sin, position_ids
        )
        query = query.transpose(1, 2)[0].float().cpu().numpy()
        key = key.transpose(1, 2)[0].float().cpu().numpy()
    return query, key


def hf_logits(model, token_ids):
    """Return HF logits as a float32 numpy array of shape (positions, V)."""
    import torch

    ids = torch.tensor([list(token_ids)], dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=False)
    return output.logits[0].float().cpu().numpy()


def numpy_logits(reader, token_ids, dims=None):
    """Return the calibrate.py float32 forward logits for one prompt."""
    dims = FULL if dims is None else dims
    model = FloatModel(reader, dims)
    windows = np.asarray([list(token_ids)], dtype=np.int32)
    _maxima, logits = calibrate.float_forward(
        model, windows, return_logits=True
    )
    return logits[0]


def run_p1(checkpoint, manifest, prompt_ids=None, prompt_id="P0", dims=None):
    """Run check P1. Parameters are arrays only inside the comparison helpers."""
    dims = FULL if dims is None else dims
    ids, source = resolve_prompt(prompt_id, prompt_ids)
    safetensors = _safetensors_path(checkpoint)
    reader = CheckpointReader(safetensors, manifest)

    numpy_q, numpy_k = numpy_layer0_qk(reader, ids, dims)
    model = load_hf_model(checkpoint)
    hf_q, hf_k = hf_layer0_qk(model, ids)

    hf_scores = attention_scores(hf_q, hf_k)
    neighbour_scores = attention_scores(numpy_q, numpy_k)
    scores = compare_score_arrays(hf_scores, neighbour_scores)

    hf_logit_array = hf_logits(model, ids)
    numpy_logit_array = numpy_logits(reader, ids, dims)
    logits = compare_logits(hf_logit_array, numpy_logit_array)
    del model

    return {
        "passed": bool(scores["passed"] and logits["passed"]),
        "layer": 0,
        "prompt": {
            "id": prompt_id,
            "token_ids": list(ids),
            "source": source,
        },
        "scores": scores,
        "logits": logits,
    }


def _layer_of(name):
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        return int(parts[2])
    raise CheckError("cannot read a layer number from {!r}".format(name))


def _fold_spec(name, dims):
    """Return (kind, gain tensor name, c bits, head count) for one tensor."""
    if name == "model.embed_tokens.weight":
        return "embedding", None, None, None
    if name == "lm_head.weight":
        return "lm_head", "model.norm.weight", contract.C0_BITS, None
    if name.endswith("self_attn.q_proj.weight"):
        layer = _layer_of(name)
        return (
            "q_proj",
            "model.layers.{}.input_layernorm.weight".format(layer),
            contract.CQ_BITS,
            dims.H,
        )
    if name.endswith("self_attn.k_proj.weight"):
        layer = _layer_of(name)
        return (
            "k_proj",
            "model.layers.{}.input_layernorm.weight".format(layer),
            contract.C0_BITS,
            dims.G,
        )
    if name.endswith("self_attn.v_proj.weight"):
        layer = _layer_of(name)
        return (
            "v_proj",
            "model.layers.{}.input_layernorm.weight".format(layer),
            contract.C0_BITS,
            None,
        )
    if name.endswith("self_attn.o_proj.weight"):
        return "o_proj", None, None, None
    if name.endswith("mlp.gate_proj.weight"):
        layer = _layer_of(name)
        return (
            "gate_proj",
            "model.layers.{}.post_attention_layernorm.weight".format(layer),
            contract.C0_BITS,
            None,
        )
    if name.endswith("mlp.up_proj.weight"):
        layer = _layer_of(name)
        return (
            "up_proj",
            "model.layers.{}.post_attention_layernorm.weight".format(layer),
            contract.C0_BITS,
            None,
        )
    if name.endswith("mlp.down_proj.weight"):
        return "down_proj", None, None, None
    if name == "model.norm.weight" or name.endswith("layernorm.weight"):
        return "gain", None, None, None
    return "other", None, None, None


def _failure_reason(message):
    text = message.lower()
    if "non-finite" in text or "non finite" in text:
        return "non_finite"
    if "exponent" in text:
        return "exponent"
    if "gain product" in text:
        return "gamma"
    if "embedding entry" in text or "embedding" in text and "2**21" in text:
        return "embedding_bound"
    return "fold"


def run_p2(checkpoint, manifest, e_res=DEFAULT_E_RES, dims=None):
    """Run the streaming converter dry run of check P2."""
    dims = FULL if dims is None else dims
    e_res = int(e_res)
    safetensors = _safetensors_path(checkpoint)
    reader = CheckpointReader(safetensors, manifest)
    checked = []
    failures = []
    seen = set()

    def fail(name, reason, detail):
        key = (name, reason, detail)
        if key in seen:
            return
        seen.add(key)
        failures.append({"tensor": name, "reason": reason, "detail": detail})

    for name in reader.names():
        values = reader.tensor(name)
        kind, gain_name, c_bits, heads = _fold_spec(name, dims)
        entry = {"name": name, "kind": kind, "shape": list(values.shape)}
        finite = bool(np.all(np.isfinite(values)))
        entry["finite"] = finite
        if not finite:
            fail(name, "non_finite", "tensor contains a non-finite value")

        if kind == "embedding":
            max_abs = float(np.max(np.abs(values))) if values.size else 0.0
            scaled = max_abs * float(2**e_res)
            entry["e_res"] = e_res
            entry["max_abs"] = _json_number(max_abs)
            entry["max_abs_scaled"] = _json_number(scaled)
            if not (scaled < float(2**21)):
                fail(
                    name,
                    "embedding_bound",
                    "max |Emb| * 2^e_res is not below 2^21",
                )
            if finite:
                try:
                    fold.quantize_embedding(values, e_res)
                except fold.FoldError as exc:
                    fail(name, _failure_reason(str(exc)), str(exc))
        elif kind in ("gain", "other"):
            pass
        elif finite:
            gains = None
            if gain_name is not None:
                gains = reader.tensor(gain_name)
                if not bool(np.all(np.isfinite(gains))):
                    fail(
                        gain_name,
                        "non_finite",
                        "gain tensor contains a non-finite value",
                    )
            try:
                folded = values
                if heads is not None:
                    folded = fold.permute_qk(values, heads)
                _quantized, exponent, mx = fold.fold_and_quantize(
                    folded, gains, c_bits
                )
                entry["e"] = int(exponent)
                entry["mx"] = _json_number(mx)
                if not 0 <= int(exponent) <= 24:
                    fail(
                        name,
                        "exponent",
                        "exponent {} is outside 0 .. 24".format(exponent),
                    )
            except fold.FoldError as exc:
                fail(name, _failure_reason(str(exc)), str(exc))
            del gains
        checked.append(entry)
        del values

    def clean(reason):
        return not any(failure["reason"] == reason for failure in failures)

    checks = {
        "no_non_finite": clean("non_finite"),
        "exponents_in_range": clean("exponent"),
        "gammas_in_range": clean("gamma"),
        "embedding_bound": clean("embedding_bound"),
        "no_other_failures": clean("fold"),
    }
    return {
        "passed": not failures,
        "e_res": e_res,
        "tensor_count": len(checked),
        "checks": checks,
        "failures": failures,
        "tensors": checked,
    }


def run_p3(checkpoint, manifest):
    """Compare the streamed sha256 of model.safetensors with the manifest."""
    safetensors = _safetensors_path(checkpoint)
    source = manifest.get("model_safetensors", {})
    expected = source.get("sha256_lfs_etag")
    if not isinstance(expected, str) or not expected:
        raise CheckError("the manifest has no model_safetensors.sha256_lfs_etag")
    actual = manifest_module.sha256_file(safetensors)
    return {
        "passed": bool(actual.lower() == expected.lower()),
        "expected": expected,
        "actual": actual,
        "size_bytes": int(Path(safetensors).stat().st_size),
        "sha256_chunk_bytes": 1 << 20,
    }


def _versions():
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    if "torch" in sys.modules:
        versions["torch"] = str(getattr(sys.modules["torch"], "__version__", "unknown"))
    if "transformers" in sys.modules:
        versions["transformers"] = str(
            getattr(sys.modules["transformers"], "__version__", "unknown")
        )
    return versions


def run_checks(
    mode,
    checkpoint,
    manifest,
    *,
    manifest_source=None,
    e_res=DEFAULT_E_RES,
    e_res_source="default",
    params_path=None,
    prompt_ids=None,
    prompt_id="P0",
    dims=None,
):
    """Run one or all checks and return the report object."""
    selected = ["P1", "P2", "P3"] if mode == "all" else [mode.upper()]
    checks = {}
    for name in selected:
        try:
            if name == "P1":
                checks[name] = run_p1(
                    checkpoint,
                    manifest,
                    prompt_ids=prompt_ids,
                    prompt_id=prompt_id,
                    dims=dims,
                )
            elif name == "P2":
                checks[name] = run_p2(
                    checkpoint, manifest, e_res=e_res, dims=dims
                )
            else:
                checks[name] = run_p3(checkpoint, manifest)
        except Exception as exc:  # a report is still written for a failed check
            checks[name] = {
                "passed": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    inputs = {
        "checkpoint": str(Path(checkpoint)),
        "safetensors": str(_safetensors_path(checkpoint)),
        "manifest": manifest_source,
        "e_res": int(e_res),
        "e_res_source": e_res_source,
        "params": None if params_path is None else str(params_path),
    }
    if "P1" in checks and "prompt" in checks["P1"]:
        inputs["prompt"] = checks["P1"]["prompt"]

    return {
        "contract": contract.CONTRACT_ID,
        "mode": mode,
        "checks_requested": selected,
        "checks": checks,
        "passed": all(check.get("passed", False) for check in checks.values()),
        "inputs": inputs,
        "versions": _versions(),
    }


def write_report(path, report):
    destination = Path(path)
    if destination.parent:
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="ascii"
    )


def _parse_prompt_ids(text):
    values = []
    for piece in text.replace(" ", "").split(","):
        if piece:
            values.append(int(piece))
    if not values:
        raise CheckError("--prompt-ids is empty")
    return values


def _resolve_e_res(args):
    if args.e_res is not None:
        return int(args.e_res), "command line --e-res"
    if args.params is not None:
        with open(args.params, "r", encoding="utf-8") as handle:
            params = json.load(handle)
        if "e_res" not in params:
            raise CheckError("{} has no e_res".format(args.params))
        return int(params["e_res"]), "params.json"
    return DEFAULT_E_RES, "default"


def _resolve_manifest(args):
    if args.manifest is not None:
        with open(args.manifest, "r", encoding="utf-8") as handle:
            return json.load(handle), str(args.manifest)
    return manifest_module.load_manifest(), str(manifest_module.DEFAULT_MANIFEST_PATH)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m tinyllama.convert.check_permutation",
        description="Run TL-INT-1 pod checks P1, P2, P3 or all.",
    )
    parser.add_argument("check", choices=("p1", "p2", "p3", "all"))
    parser.add_argument("--checkpoint", required=True, metavar="DIR")
    parser.add_argument("--report", required=True, metavar="OUT.json")
    parser.add_argument("--e-res", type=int, default=None, metavar="N")
    parser.add_argument("--params", default=None, metavar="params.json")
    parser.add_argument("--manifest", default=None, metavar="PATH")
    parser.add_argument(
        "--prompt-ids",
        default=None,
        metavar="ID,ID,...",
        help="override the P0 ids from docs/tinyllama/prompts_v1.json",
    )
    args = parser.parse_args(argv)

    try:
        e_res, e_res_source = _resolve_e_res(args)
        manifest, manifest_source = _resolve_manifest(args)
        prompt_ids = (
            None
            if args.prompt_ids is None
            else _parse_prompt_ids(args.prompt_ids)
        )
        report = run_checks(
            args.check,
            args.checkpoint,
            manifest,
            manifest_source=manifest_source,
            e_res=e_res,
            e_res_source=e_res_source,
            params_path=args.params,
            prompt_ids=prompt_ids,
        )
    except (CheckError, OSError, ValueError) as exc:
        print(
            "tinyllama.convert.check_permutation: {}".format(exc), file=sys.stderr
        )
        return 2

    write_report(args.report, report)
    state = "PASS" if report["passed"] else "FAIL"
    print(
        "{}: {} (report {})".format(
            state, ", ".join(report["checks_requested"]), args.report
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
