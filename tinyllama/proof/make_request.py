"""Write FlashProve request files for the TinyLlama proof (PROVING_PLAN T4.4).

A request file is a window of exactly T 32-bit little-endian tokens followed by
two words r and e, the first and last covered positions (docs/protocol.md;
prover/examples/TARGET-KITS.md, "The examples' requests"). For a prompt of
``len(p)`` ids and a reply of ``len(a)`` ids the file is

    p + a + [0] * (T - len(p) - len(a)) + [len(p) - 1, len(p) + len(a) - 1]

so the proof covers the reply's tokens (positions len(p) - 1 .. len(p) + len(a)
- 1); with len(p) + len(a) < T the token after the window, y, is not covered.

The reply tokens are the integer model's own greedy choices, made by our CPU
reference (tinyllama.reference.model.forward), never by the float model: under
the exact-match policy every covered token must be the integer model's first
choice after the tokens before it, or cp_prove writes ``<name>.failed``.

The first real request is P0's 24 ids, then the next token 1576, then id 0
padding to T: r = 23, e = 24. Build it from the stored prompt set with

  python -m tinyllama.proof.make_request --prompt P0 --reply-ids 1576 \
      --window 64 --out p0.req

or let the reference choose the reply from the committed artifact:

  python -m tinyllama.proof.make_request --prompt P0 --greedy 1 \
      --artifact <artifact dir> --window 64 --out p0.req
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

from tinyllama import contract

U32 = 1 << 32


class RequestError(ValueError):
    """The request cannot be built from the given tokens."""


def _check_ids(ids, what):
    out = []
    for token in ids:
        if not isinstance(token, int) or isinstance(token, bool) or token < 0 or token >= U32:
            raise RequestError("%s has a token that is not a whole number in 0 .. 2^32 - 1: %r" % (what, token))
        out.append(int(token))
    return out


def request_words(prompt_ids, reply_ids, window):
    """The T + 2 words of a request file for this prompt, reply and window T.

    r = len(prompt) - 1 and e = len(prompt) + len(reply) - 1 are the first and
    last covered positions; the window is padded to T with id 0.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise RequestError("the window T must be a positive whole number, not %r" % (window,))
    prompt = _check_ids(prompt_ids, "the prompt")
    reply = _check_ids(reply_ids, "the reply")
    if not prompt:
        raise RequestError("the prompt has no tokens")
    if len(prompt) + len(reply) > window:
        raise RequestError(
            "the prompt and reply are %d tokens, more than the window T = %d" % (len(prompt) + len(reply), window)
        )
    words = prompt + reply + [0] * (window - len(prompt) - len(reply))
    words += [len(prompt) - 1, len(prompt) + len(reply) - 1]
    return words


def write_request(path, prompt_ids, reply_ids, window):
    """Write the request file for this prompt and reply. The file is created new
    and is readable by its owner only (it holds a person's prompt and reply), as
    prover/examples/TARGET-KITS.md writes it. Returns (r, e)."""
    words = request_words(prompt_ids, reply_ids, window)
    blob = struct.pack("<%dI" % len(words), *words)
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    with os.fdopen(handle, "wb") as out:
        out.write(blob)
    return words[-2], words[-1]


def greedy_reply(artifact, prompt_ids, count, *, window=None):
    """``count`` reply tokens, each the integer model's greedy first choice after
    all the tokens before it, from our CPU reference (no float model). ``artifact``
    is a loaded reference artifact; prompt_ids its prompt. A range failure or a
    sequence that would fill the window raises RequestError."""
    from tinyllama.reference.model import forward

    limit = int(artifact.dims.T) if window is None else int(window)
    sequence = _check_ids(prompt_ids, "the prompt")
    if not sequence:
        raise RequestError("the prompt has no tokens")
    reply = []
    for _ in range(count):
        if len(sequence) >= limit:
            raise RequestError("the prompt and reply would fill the window T = %d before %d reply tokens" % (limit, count))
        result = forward(artifact, sequence, mode="prefix")
        if result.range_failure is not None:
            raise RequestError("the reference reported a range failure at reply token %d: %s" % (len(reply), result.range_failure))
        reply.append(int(result.next_token))
        sequence.append(int(result.next_token))
    return reply


def _parse_ids(text, what):
    try:
        return [int(part) for part in text.replace(",", " ").split()]
    except ValueError:
        raise RequestError("%s must be whole numbers separated by commas or spaces" % what) from None


def _dims_from(text):
    from tinyllama.artifacts import Dims

    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 8 or any(not part for part in parts):
        raise RequestError("--dims-test needs eight integers L,D,H,G,DH,F,V,T")
    try:
        values = [int(part) for part in parts]
    except ValueError:
        raise RequestError("--dims-test needs eight integers L,D,H,G,DH,F,V,T") from None
    return Dims(*values)


def _prompt_ids(args):
    if args.prompt_ids is not None:
        return _parse_ids(args.prompt_ids, "--prompt-ids")
    if args.prompt is not None:
        from tinyllama.reference.run import load_prompt

        return load_prompt(args.prompt)[1]
    raise RequestError("give the prompt with --prompt-ids or --prompt")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a FlashProve request file for the TinyLlama proof (PROVING_PLAN T4.4)."
    )
    parser.add_argument("--out", required=True, help="the request file to write (created new)")
    parser.add_argument("--window", type=int, default=contract.T, help="the window T (a power of two, default the contract's)")
    parser.add_argument("--prompt-ids", help="the prompt's token ids, comma or space separated")
    parser.add_argument("--prompt", help="a prompt id from docs/tinyllama/prompts_v1.json, e.g. P0")
    parser.add_argument("--reply-ids", help="the reply's token ids, comma or space separated")
    parser.add_argument("--greedy", type=int, metavar="N", help="append N reply tokens from the CPU reference")
    parser.add_argument("--artifact", help="artifact directory, needed by --greedy")
    parser.add_argument("--artifact-id", help="expected artifact id, checked when --artifact is a directory")
    parser.add_argument("--dims-test", help="test only: load a tiny artifact with dims L,D,H,G,DH,F,V,T")
    args = parser.parse_args(argv)

    try:
        prompt_ids = _prompt_ids(args)
        if (args.reply_ids is None) == (args.greedy is None):
            raise RequestError("give the reply with exactly one of --reply-ids or --greedy")
        if args.reply_ids is not None:
            reply_ids = _parse_ids(args.reply_ids, "--reply-ids")
        else:
            if args.artifact is None:
                raise RequestError("--greedy needs --artifact")
            from tinyllama.artifacts import FULL, load_artifact

            dims = _dims_from(args.dims_test) if args.dims_test else FULL
            artifact = load_artifact(args.artifact, args.artifact_id, dims=dims)
            reply_ids = greedy_reply(artifact, prompt_ids, args.greedy, window=args.window)
        r, e = write_request(args.out, prompt_ids, reply_ids, args.window)
    except RequestError as error:
        print("make_request: %s" % error, file=sys.stderr)
        return 2
    except FileExistsError:
        print("make_request: %s exists already; name a file that does not exist yet" % args.out, file=sys.stderr)
        return 2
    print(
        "make_request: wrote %s (window %d, prompt %d, reply %d, r %d, e %d)"
        % (args.out, args.window, len(prompt_ids), len(reply_ids), r, e)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
