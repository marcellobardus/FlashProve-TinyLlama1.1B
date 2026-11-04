# Tasks

Shared interfaces (I1 to I4) must be fixed by the lead before workers start. Workers use separate branches or worktrees.
Resource ceilings are for a non-shared machine; on the shared box only T0 items and light tests are allowed.

## Proposed layout

```
tinyllama/            new Python package, no edits to existing FlashProve sources
  manifest.py         pins and hash checks (reads docs/tinyllama/checkpoint_manifest.json)
  convert/            HF safetensors BF16 to integer artifacts (owned by W1)
  reference/          CPU integer reference (owned by W2)
  artifacts.py        artifact format, loader, validator (I2; W1 writes, W2 reads)
tests/tinyllama/      independent tests, golden vectors, evaluation (owned by W3)
docs/tinyllama/       ARITHMETIC_CONTRACT.md, REPRODUCE.md (lead)
artifacts/            generated, git ignored
```

## Shared interfaces (lead, before delegation)

- I1 ARITHMETIC_CONTRACT.md: shapes, int types, shift positions, rq rounding, clamps, tables, rotary pairing, GQA map, softmax, SiLU, logits and argmax. Default: FlashProve Model with NE=0, rp=1 all layers, d=2048, dh=64 (docs/model.md).
- I2 artifact format: file list, dtypes, little-endian layout, KL.bin style shift words, policy-like parameter file, sha256 manifest.
- I3 trace format: named intermediates per layer and position (reuse model.py record names such as `qw`, `x_mid`).
- I4 fixed prompt set and token ids (tokenizer pinned in manifest), padding rule (D6).

## Tasks

| ID | worker | objective | depends | owned files | acceptance command | expected evidence | ceiling |
|---|---|---|---|---|---|---|---|
| T0.1 | lead | write ARITHMETIC_CONTRACT.md (I1) | audit | docs/tinyllama/ARITHMETIC_CONTRACT.md | review | contract cites docs/model.md sections | docs only |
| T0.2 | lead | fix I2, I3, I4 | T0.1 | tinyllama/artifacts.py spec, docs | review | interface doc committed | docs only |
| T1.1 | W1 converter | stream safetensors by header, BF16 to float32 exact, verify sha256 vs manifest | T0.2 | tinyllama/manifest.py, tinyllama/convert/load.py | `pytest tests/tinyllama/test_manifest.py` | hash match, 201 tensors | RAM <= 6 GB, tensor at a time |
| T1.2 | W1 | q/k row permutation for neighbour pairing (H1), gain folding with d=2048 dh=64 constants, per tensor exponent (docs/model.md:441-449) | T1.1 | tinyllama/convert/fold.py | `pytest tests/tinyllama/test_fold.py` | all |w| <= 4095, Emb below 2^21 | RAM <= 6 GB |
| T1.3 | W1 | calibration and scale fit (shifts, M/K, eps, e_res) reusing scales/ logic where possible | T1.2, T2.1 | tinyllama/convert/calibrate.py | documented command | KL file, range report, no range condition failures on calibration set | GPU or 16 GB RAM machine |
| T1.4 | W1 | artifact writer, loader, validator, malformed artifact refusal | T1.2 | tinyllama/artifacts.py | `pytest tests/tinyllama/test_artifacts.py` | refusal tests pass | small |
| T2.1 | W2 CPU ref | integer forward per contract, numpy int64 or Python int, one layer first | T0.2 | tinyllama/reference/ | `pytest tests/tinyllama/test_reference_block.py` | exact match to W3 golden vectors | RAM <= 8 GB |
| T2.2 | W2 | full model, fixed prompt, logits and greedy token, trace dump | T1.4, T2.1 | tinyllama/reference/ | `python -m tinyllama.reference.run --prompt P0` | reproducible logits sha256 over 2 runs | RAM <= 8 GB |
| T2.3 | W2 | cross check against FlashProve model.py on CPU where shapes allow | T2.2 | tinyllama/reference/crosscheck.py | documented command | equal integers or listed differences | torch CPU, separate machine |
| T3.1 | W3 tests | independent golden vectors from the contract text (rounding ties, negatives, clamps, isqrt, softmax, tables) | T0.1 | tests/tinyllama/ | `pytest tests/tinyllama -k arithmetic` | exact integer agreement | small |
| T3.2 | W3 | table recipes and digests vs FlashProve tables.py | T0.1 | tests/tinyllama/test_tables.py | `pytest tests/tinyllama/test_tables.py` | equal digests | small |
| T3.3 | W3 | quality: BF16 float vs integer, top-1 agreement, logit KL, perplexity on fixed set | T2.2 | tests/tinyllama/eval/ | documented command | metrics table, not bit equality | separate machine |
| T4.1 | lead | REPRODUCE.md for AMD workstream | T2.2, T3.3 | docs/tinyllama/REPRODUCE.md | clean checkout run | log | separate machine |
