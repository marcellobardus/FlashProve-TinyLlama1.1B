"""Command line for one prompt through the TL-INT-1 CPU reference.

    python -m tinyllama.reference.run --artifact DIR --prompt P0 \
        [--mode prefix|window] [--trace OUT] [--out OUT] [--full]

Token ids come from docs/tinyllama/prompts_v1.json, located from this package
file, and their stored sha256 is checked before the forward pass.  Without
``--trace`` the hash trace goes to
``traces/TL-INT-1/<artifact_id>/<prompt>.<mode>.json`` below the repository
root.  ``--out`` is an alias of ``--trace``.

``--dims-test L,D,H,G,DH,F,V,T`` is a test-only option: it passes ``dims=`` to
``load_artifact`` so tiny synthetic artifacts can be loaded.  It is never used
for a real artifact.
"""

import argparse
import json
from pathlib import Path

from ..artifacts import FULL, Dims, load_artifact
from .model import forward
from .trace import TraceWriter, token_ids_sha256

__all__ = ["main", "load_prompt", "prompts_path"]

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = REPOSITORY_ROOT.joinpath("docs", "tinyllama", "prompts_v1.json")


def prompts_path():
    return PROMPTS_PATH


def load_prompt(prompt_id):
    """Return (prompt object, ids), after checking the stored ids sha256."""
    data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    if data.get("contract") != "TL-INT-1":
        raise ValueError("prompt set contract is not TL-INT-1")
    for prompt in data.get("prompts", []):
        if prompt.get("id") != prompt_id:
            continue
        ids = [int(token) for token in prompt["token_ids"]]
        if token_ids_sha256(ids) != prompt["token_ids_sha256_u32le"]:
            raise ValueError(
                "token ids for %s do not match their stored sha256" % prompt_id
            )
        return prompt, ids
    raise ValueError("unknown prompt %r" % prompt_id)


def _parse_dims_test(text):
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 8 or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--dims-test needs eight comma separated integers L,D,H,G,DH,F,V,T"
        )
    try:
        values = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--dims-test needs eight comma separated integers"
        ) from exc
    return Dims(*values)


def _parser():
    parser = argparse.ArgumentParser(
        description="Run one TL-INT-1 prompt through the integer CPU reference."
    )
    parser.add_argument("--artifact", required=True, help="artifact directory")
    parser.add_argument("--prompt", required=True, help="prompt id in prompts_v1.json")
    parser.add_argument(
        "--mode", choices=("prefix", "window"), default="prefix"
    )
    parser.add_argument(
        "--trace",
        "--out",
        dest="trace",
        default=None,
        help="hash trace output file (--out is an alias)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="also write NAME.LL.npy full traces beside the hash trace",
    )
    parser.add_argument(
        "--dims-test",
        type=_parse_dims_test,
        default=None,
        help=(
            "test only: load a tiny artifact with dims L,D,H,G,DH,F,V,T "
            "(never for a real artifact)"
        ),
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    dims = FULL if args.dims_test is None else args.dims_test
    artifact = load_artifact(args.artifact, dims=dims)
    prompt, ids = load_prompt(args.prompt)

    if args.trace is None:
        trace_path = REPOSITORY_ROOT.joinpath(
            "traces",
            "TL-INT-1",
            artifact.id,
            "%s.%s.json" % (args.prompt, args.mode),
        )
    else:
        trace_path = Path(args.trace)

    writer = TraceWriter(trace_path, full=args.full)
    result = forward(artifact, ids, mode=args.mode, record=writer)
    trace = writer.finish(
        artifact, prompt_id=args.prompt, mode=args.mode, token_ids=ids, result=result
    )

    print("next_token: %s" % result.next_token)
    print("lam_last_sha256: %s" % trace["lam_last_sha256"])
    print("artifact_id: %s" % artifact.id)
    print("trace: %s" % trace_path)
    if result.range_failure is not None:
        failure = result.range_failure
        print(
            "range_failure: condition %d (%s) layer %s position %d index %d"
            % (
                failure.condition,
                failure.name,
                failure.layer,
                failure.position,
                failure.index,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
