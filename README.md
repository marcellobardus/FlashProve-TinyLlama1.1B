# FlashProve TinyLlama 1.1B

A research prototype: not a production system, and not maintained as a product. Apache 2.0.

This repository builds on FlashProve (Anthropic) to commit to an integer form of [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (TL-INT-1) and to publish CPU-checkable proofs that the committed integer model produced a stated next token on specific prompts.

It is not a claim that FlashProve's Lean soundness theorem covers TinyLlama. The published commitments used `-public-weights`, so the weights are not hidden: the verifier printed `weight hiding: NOT CHECKED BY THIS CALL`. Each published proof is of one next token, not a full reply. The published window is T = 64.

## What is published

Both proofs share this commitment ID:

`d319bc487793a0682821d7e47f1a0cf0cb4e5ee674b34749c65e2d804812c227`

- [P0 T=64 verify package](docs/tinyllama/recorded/p0-tl-int-1-t64/): the user prompt is "What is the capital of France?". The proved next token is 1576, the text `The` (not the rest of a sentence, and not `Paris` alone). On the prove host, `cp_prove` took 8.17 s and `cp_verify` took 4.03 s; the proof is 15608900 bytes.
- [P2 T=64 verify package](docs/tinyllama/recorded/p2-tl-int-1-t64/): the mandate yes/no question from [`docs/tinyllama/prompts_v1.json`](docs/tinyllama/prompts_v1.json). The proved next token is 8241, the text `Yes`. That is what the integer model output, not a claim that this is the compliance-correct answer. On the prove host, `cp_prove` took 11.75 s and `cp_verify` took 7.71 s; the proof is 15596716 bytes.

The archived verifier also printed `YES` against each packaged path on a Linux CPU, with no GPU. The verify times above are from the prove host.

## Verify a published proof (CPU only)

No GPU and no prover are needed. From a clone of `main`:

```sh
git clone https://github.com/marcellobardus/FlashProve-TinyLlama1.1B.git
cd FlashProve-TinyLlama1.1B
./verifier/build.sh

REC=docs/tinyllama/recorded/p0-tl-int-1-t64
ID=$(tr -d ' \n' < "$REC/commitment-id.txt")
bin/cp_verify "$REC/kit" "$REC/ivk" "$REC/in/p0.req" \
  "$REC/out/p0.y" "$REC/out/p0.proof" -id "$ID"
```

Expect exit 0 and last line `YES`. For P2, use `docs/tinyllama/recorded/p2-tl-int-1-t64` with `in/p2.req`, `out/p2.y` and `out/p2.proof`. The package READMEs repeat these `cp_verify` lines and quote the recorded result text.

The `y` files are four zero bytes. The claimed token lives in the request; the verifier printed `y=no` and `next-token=no`.

## Prove a request (one H100 80 GB)

Producing a proof needs one NVIDIA H100 80 GB. The recorded packages are enough to verify without the prover. To prove a new request you need a local `cp_commit` (or a saved `isk` directory from one): `cp_prove` reads that directory, not only the published `isk.txt`.

Build the verifier with `./verifier/build.sh` and the production prover with `prover/build.sh`. Write a T = 64 configuration kit with `python3 -m tinyllama.proof.config_kit` as documented in [docs/tinyllama/PROOF_PREP.md](docs/tinyllama/PROOF_PREP.md) (`--lt 6` selects a 64-token window), or reuse the kit files shipped in the recorded packages.

Commit to the TL-INT-1 integer weights with `-public-weights` and without `-worst-case` or `-worst-case-table`. That is how the published commitments were made. Under `-public-weights` the worst-case check is left out, and `isk.txt` says `seed public` with no `ranges` line.

```sh
# SEED is 32 random bytes, not all zeros. The published packages include their seed.
prover/build/hc/cp_commit -weights WEIGHTS -config KIT -M 1 \
  -ivk IVK -isk ISK -public-weights SEED
```

Build the request from the integer model's own next token, never a guessed token. Use `--window 64` to match the published proofs:

```sh
python3 -m tinyllama.proof.make_request --prompt P0 --reply-ids 1576 \
    --window 64 --out p0.req

python3 -m tinyllama.proof.make_request --prompt P2 --reply-ids 8241 \
    --window 64 --out p2.req

# Or let the CPU integer reference choose N reply tokens:
python3 -m tinyllama.proof.make_request --prompt P0 --greedy 1 \
    --artifact ARTIFACT --window 64 --out p0.req
```

A different prompt that still fits the T = 64 window needs its own request and its own `cp_prove` run. Give the prompt with `--prompt` (an id from `docs/tinyllama/prompts_v1.json`) or `--prompt-ids`, and the reply with `--reply-ids` or `--greedy`. Prompt plus claimed tokens must fit in 64 tokens. The same T = 64 commitment can be reused.

```sh
prover/build/hc/cp_prove -isk ISK -in IN -out OUT -public-weights SEED
```

Then verify with `bin/cp_verify` as above, passing the new request, `y`, proof and commitment ID.

A T = 1024 proof has not been published. That would need a new kit, a new commitment (rotary tables depend on the window size) and another H100 prove. The integer weights themselves would stay the same.

## Upstream FlashProve

The proof system in this repository comes from FlashProve. Upstream pages still in the tree (`guide/`, `docs/`, `lean/`) describe Llama 4 Maverick, weight hiding and a Lean soundness theorem for that original setting. They are FlashProve's own docs, not a description of the TinyLlama proofs published here.

## Licence

Apache 2.0: see [LICENSE](LICENSE). Material from other projects keeps its own licence.
[NOTICE](NOTICE) and [THIRD_PARTY.md](THIRD_PARTY.md) list it.
