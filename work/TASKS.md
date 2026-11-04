# Tasks

Shared interfaces I1 to I4 are defined in docs/tinyllama/INTERFACES.md (v1.0.0-draft) for contract TL-INT-1
(docs/tinyllama/ARITHMETIC_CONTRACT.md). Workers use separate branches or worktrees and touch only their owned files.
Heavy work runs on the RunPod cpu3g-8-32 pod (D7); on the shared box only T0 items and light tests are allowed.

## Layout and file ownership

```
tinyllama/__init__.py        W1 (creates the package first; others do not edit it)
tinyllama/contract.py        W1  constants of I1
tinyllama/manifest.py        W1  checkpoint pins and hash checks
tinyllama/artifacts.py       W1  I2 writer, loader, validator
tinyllama/convert/           W1  safetensors reader, permutation, folding, calibration, fit
tinyllama/reference/         W2  CPU integer reference, trace writer, run command (I3)
tests/tinyllama/convert/     W1  unit tests of W1's own code
tests/tinyllama/reference/   W2  smoke tests of W2's own code
tests/tinyllama/conftest.py  W3
tests/tinyllama/arithmetic/  W3  independent golden vectors from the contract text
tests/tinyllama/artifacts/   W3  malformed artifact tests (one per I2.4 code)
tests/tinyllama/test_tables.py W3
tests/tinyllama/vectors/     W3  committed hash traces and synthetic vectors (I3 size rule)
tests/tinyllama/eval/        W3  prompt loader, float baseline, quality metrics
docs/tinyllama/, work/       lead (contract, interfaces, prompts_v1.json, REPRODUCE.md, records)
artifacts/, traces/, work/scratch/   generated, git ignored
```

No file has two owners. W2 and W3 read I2 artifacts only through tinyllama/artifacts.py (W1). W3 never imports
tinyllama/reference for expected values.

## Tasks

| ID | worker | objective | depends | owned files | acceptance command | expected evidence | ceiling |
|---|---|---|---|---|---|---|---|
| T0.1 | lead | ARITHMETIC_CONTRACT.md (I1) | audit | docs/tinyllama/ARITHMETIC_CONTRACT.md | review | done 2026-11-04, contract TL-INT-1 | docs only |
| T0.2 | lead | INTERFACES.md (I1 to I4), prompts_v1.json | T0.1 | docs/tinyllama/INTERFACES.md, docs/tinyllama/prompts_v1.json | review | done 2026-11-04 | docs only |
| T1.1 | W1 | package skeleton, contract.py, safetensors reader, BF16 to float32 exact, sha256 vs manifest (P3) | T0.2 | tinyllama/__init__.py, contract.py, manifest.py, convert/load.py | `pytest tests/tinyllama/convert/test_load.py` | 201 tensors, header sha256 de4413e1..., file sha256 6e6001da... on the pod | RAM <= 6 GB, tensor at a time |
| T1.2 | W1 | q/k permutation, gain folding with 0x3FB504F3 and 0x3E3504F3, per tensor exponent | T1.1 | convert/fold.py | `pytest tests/tinyllama/convert/test_fold.py` | all \|w\| <= 4095, e in 0 .. 24 (P2) | RAM <= 6 GB |
| T1.5 | W1 | pod check P1 (permutation on real weights) | T1.2 | convert/check_permutation.py | documented command | max relative score difference below 1e-5; same greedy token | pod, about 5 GB |
| T1.3 | W1 | calibration on TL-CAL-1 and scale fit (contract 5.2), range report on held-out (P4) | T1.2, T2.1 | convert/calibrate.py, convert/fit.py | documented command | KL.bin, params.json, zero range failures | pod |
| T1.4 | W1 | artifact writer, loader, validator (I2) | T1.2 | tinyllama/artifacts.py | `pytest tests/tinyllama/convert/test_artifacts.py` and W3's `pytest tests/tinyllama/artifacts` | all A01 to A16 refusals | small |
| T2.1 | W2 | integer layer per contract on synthetic weights | T0.2 | tinyllama/reference/ | `pytest tests/tinyllama/reference -k block` and `pytest tests/tinyllama/arithmetic` | exact match to W3 golden vectors | RAM <= 8 GB |
| T2.2 | W2 | full model, P0 prefix mode, logits, greedy token, hash trace (I3) | T1.4, T2.1 | tinyllama/reference/ | `python -m tinyllama.reference.run --artifact DIR --prompt P0` twice | equal lam_last_sha256 over 2 runs; prefix and window mode equal at real positions | pod, RAM <= 8 GB |
| T2.3 | W2 | cross check with FlashProve model.py arithmetic where possible (O5) | T2.2 | tinyllama/reference/crosscheck.py | documented command | equal integers or listed differences | pod, torch CPU |
| T3.1 | W3 | independent golden vectors (rounding ties, negatives, clamps, isqrt, norm, softmax division, gate key, argmax, padding) | T0.1 | tests/tinyllama/arithmetic/, conftest.py | `pytest tests/tinyllama/arithmetic` | exact integer agreement | small |
| T3.2 | W3 | table recipes and digests vs contract section 6 and FlashProve tables.py | T0.1 | tests/tinyllama/test_tables.py | `pytest tests/tinyllama/test_tables.py` | digests of contract section 6 | small |
| T3.5 | W3 | malformed artifact tests, one per I2.4 code | T0.2 | tests/tinyllama/artifacts/ | `pytest tests/tinyllama/artifacts` | each code raised for its mutation | small |
| T3.4 | W3 | pod check P5: AutoTokenizer ids vs prompts_v1.json; TL-CAL-1 sha256 | T0.2 | tests/tinyllama/eval/check_inputs.py | documented command | equal ids and sha256 | pod, small |
| T3.3 | W3 | quality: float32 vs integer, top-1 agreement, logit KL, perplexity on held-out windows, range failures | T2.2 | tests/tinyllama/eval/ | documented command | metrics table, not bit equality | pod |
| T4.1 | lead | REPRODUCE.md for the AMD workstream | T2.2, T3.3 | docs/tinyllama/REPRODUCE.md | clean checkout run | log | pod |
