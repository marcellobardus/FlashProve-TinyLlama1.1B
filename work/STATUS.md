# Status

Branch `feature/tinyllama-integer-audit` from main `9e2bdd8f`. Not pushed.

## 2026-11-04 source audit (done)

- Checked: clean tree on main, remote origin = marcellobardus/FlashProve-TinyLlama1.1B, HEAD 9e2bdd8f.
- Added ignore rules (/artifacts/, /work/scratch/, *.safetensors, *.gguf, /hf_cache/, /traces/).
- Pinned checkpoint revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 by HF API; header-only read: 201 BF16 tensors, 1,100,048,384 parameters. Files in docs/tinyllama/CHECKPOINT_MANIFEST.md.
- Gap map: docs/tinyllama/GAP_MAP.md.
- Not run: no build, no tests, no model load (box rule: no heavy work on the shared box).

## 2026-11-04 decisions D1 to D8 (done)

- Recorded D1 to D8 as resolved in work/DECISIONS.md (source: Chief of Staff relaying Marcello's defaults; D6 and D8 picked by the lead). New open items O1 to O5.
- Light probes run on the box for D6 and D8 (no model load; files under work/scratch/, git ignored):
  - `python3 -m venv work/scratch/venv && work/scratch/venv/bin/pip install tokenizers==0.22.2 jinja2 pyarrow numpy` (installed tokenizers 0.22.2, Jinja2 3.1.6, pyarrow 26.0.1, numpy 2.5.4; Python 3.13.5).
  - Fetched the WikiText-2 train parquet at revision b08601e0 (6,357,543 bytes); `sha256sum` gives e83889ba...c9f7, equal to FlashProve's pin.
  - `work/scratch/calib_probe.py`: FlashProve's steps 1 to 3 with TinyLlama's tokenizer.json: 36,718 rows, n = 2,874,559 ids, draws sha256 8cd63f617d474124a50ee052eae2fd0be603ea45f191ec8ccb4f443f676fb58c = FlashProve DRAWS_SHA256. TL-CAL-1 fit set (32 x 1024, `<i4` npy) sha256 9b2b8fc6b98904de5c905bcd7eecfd2f7895ddacef28d249ca44e3b594fa2f31 (131,200 bytes); held-out (8 x 1024) 5c9da767e78fdb9b31729a0031aa7c07b9c053ceb5ed976c6542bdf31e83bfb8. Peak RSS 1.18 GB, 12 s. Not written to artifacts.

## 2026-11-04 gap map re-verification (done)

- Re-read the cited code for gain folding, rotary pattern, window, rounding, norm, tables, logits, ranges and parameter rules. Five corrections recorded in docs/tinyllama/GAP_MAP.md section D (constant rule, both-clamps, H1 and H3 verified, theta in the scales converter, model.py NE = 0).
- Commands: `rg`/`grep -n` over weight_sums/, prover/, integrations/, lean/, verifier/, scales/; Python struct for RN32 bit patterns; `work/scratch/perm_probe.py` (numpy, random data): HF rotary vs permuted neighbour pairing, max score difference 1.4e-12; without permutation 3.2e3.
- Fetched for citation only (work/scratch/): transformers modeling_llama.py and utils/chat_template_utils.py at 0c7f5ab8; llama.cpp convert_hf_to_gguf.py and conversion/llama.py at e853ec9a; TinyLlama README.md at fe8a4ea1.

## 2026-11-04 arithmetic contract (done, draft 1.0.0)

- Wrote docs/tinyllama/ARITHMETIC_CONTRACT.md (contract id TL-INT-1): shapes, types, conversion with q/k permutation (verified), gain constants 0x3FB504F3 and 0x3E3504F3, scales and shift positions, rq and clamp examples, bounds at d 2048 and f 5632, norm, tables, rotary, GQA softmax, SwiGLU, logits, argmax, padding, deviation table, pod checks P1 to P5.
- Commands: `python3 work/scratch/tables_probe.py` (FlashProve tables.py functions with torch stubbed): rotary 32,768 entries each in -4096 .. 4096; EX u0 = 2307; Ts 65,536 entries in -18 .. 4095; texp and tsilu digests equal stored FlashProve policy files (prover/tests/configurations/pinned_no_proof_of_work_roomy, integrations/vllm/integer_model/tests/configurations/tiny_without_clamps). Bounds and rounding examples computed with Python integers (outputs in the contract).
- Not run: no model load, no conversion, no tests (none exist yet).

## 2026-11-04 shared interfaces (done, draft 1.0.0)

- Wrote docs/tinyllama/INTERFACES.md (I1 versioning, I2 artifact files and refusal codes A01 to A16, I3 trace records and size rule, I4 prompt set TL-PROMPTS-1 and calibration set TL-CAL-1, worker boundaries) and docs/tinyllama/prompts_v1.json.
- Commands: `work/scratch/venv/bin/python work/scratch/prompt_probe.py work/scratch/prompts_draft.json` and `prompt_write.py`: P0 24 ids (sha256 cf21aeeb...), P1 37, P2 57, P3 25; rendered text ends with `<|assistant|>\n`; chat template sha256 66291cf0... and tokenizer.json sha256 bcd04f0e... equal the manifest.
- Updated work/TASKS.md: file ownership without overlap, new tasks T1.5, T3.4, T3.5, pod checks P1 to P5.

## 2026-11-04 W1 and W3 results, integration (done)

- W1 (branch feature/tinyllama-w1-converter): a9459928 reader, constants, manifest; 1f2eae04 permutation, gain folding. `python3 -m pytest tests/tinyllama/convert -q`: 69 passed. Weak spots for the next W1 round: iter_tensors accepts a None manifest; read_header does not refuse unknown dtypes.
- W3 (branch feature/tinyllama-w3-tests): 361f8850 arithmetic golden vectors, `python3 -m pytest tests/tinyllama/arithmetic -q`: 156 passed, 1 skipped (test_reference_conformance.py, tinyllama.reference not importable); 55f3092d table recipes and digests, `python3 -m pytest tests/tinyllama/test_tables.py -q`: 20 passed. trope, texp and tsilu rebuild exactly from the contract's float64 recipe and match FlashProve scales/policy_tables.py. W3 reported negative checks: the tests fail when vectors are perturbed. Gaps: no full-layer or attention end-to-end vector, no rotary multiply then rq vector; the conformance test skips if primitive names differ (names now fixed in INTERFACES.md I7).
- Integration: merges 31ae63fb (W1) and c06d05c6 (W3) on feature/tinyllama-integer-audit. `python3 -m pytest tests/tinyllama -q`: 245 passed, 1 skipped on the box.
- opencode notes: DeepSeek-V4.1-Flash worked for W1 in about 3 to 3.5 minutes per task; GLM-5.3 stalled twice on long outputs for W1 but worked for W3 (about 22 to 24 minutes per task).
- Docs: contract section 6 column renamed to "sha256 of table words"; INTERFACES.md I7 added.
- Next: W2 T2.1.

## 2026-11-04 W1 round 2, W2, W3 round 2, integration (done)

- W2 T2.1 (branch feature/tinyllama-w2-reference): c4a8a3dc integer primitives per INTERFACES.md I7; b6e1a874
  integer layer. The conformance test passes on all 11 primitives. W2 reports that the reference uses no float.
  Tables are built from the recipe in tinyllama/reference/tables.py; that file is for tests only from now on (D12).
- W3 round 2 (branch feature/tinyllama-w3-artifact-tests): 9b798da5 conformance hardened (fails on a missing
  primitive), full-layer vector and rotary multiply then rq vector; files renamed to tl_ref_independent.py and
  tl_conformance.py. 01898036 malformed artifact tests A01 to A16 and a synthetic artifact helper (sparse, hard
  linked; D15).
- W1 round 2 (branch feature/tinyllama-w1-artifacts): 953c7dc4 iter_tensors requires a manifest, unknown dtype
  refused; 2a211c83 tinyllama/artifacts.py writer, loader, validator and tinyllama/convert/tables.py.
- Integration on feature/tinyllama-integer-audit: merges e510d8de (W1), 8a6dcda7 (W2), 74153128 (W3).
  `python3 -m pytest tests/tinyllama -q` at 74153128 on the box: 377 passed in 122.76s, 0 skipped, so the
  malformed artifact tests run against the real validator.
- opencode notes: DeepSeek-V4.1-Flash for all of W1 round 2 (1 to 12 minutes per task), W2 (about 9 minutes)
  and W3 round 2 (5 to 10 minutes).
- Docs: decisions D11 to D16; INTERFACES.md range check order (I3.1), constant names (I2.3), code notes and
  non-normative `dims=` (I2.4), I7 file names; contract 5.2 clarification of shift ceilings (D16).
- Next: W2 T2.2; W1 follow-ups F1 and F2, then T1.3; T1.5 and the other pod checks.

## 2026-11-04 W2 T2.2 (done on the W2 branch, not yet merged)

- W2 T2.2 (branch feature/tinyllama-w2-model): c8e1f2e8 full model forward and argmax; d7f62c58 run command and
  hash trace; 269e2823 pod-only full-dimension layer timing script (bench_layer). W2 reports 388 passed on the W2
  branch. Not yet merged into feature/tinyllama-integer-audit.
- Docs: D17 (range check order is the compute order, INTERFACES.md I3.1); I3.2 record count fixed to 469.
- Open performance item: numpy int64 matmul has no BLAS, so window mode at 1024 positions at full size may take
  hours, and first_choice is a Python loop. P0 prefix mode (24 positions) is the milestone and is expected to be
  fine. Measure with bench_layer on the pod. Follow-up T2.4 only if the measurements show a need.
- Next: merge W2 T2.2 into the integration branch; bench_layer on the pod; W1 follow-ups F1 and F2, then T1.3.

## 2026-11-04 round 3 integrated (done)

- W2 T2.2, W3 T3.3 and T3.4, W1 F1, F2, T1.3, T1.5 are merged. 520 passed, 1 xpassed on the box (3:42).
- Not run: anything on real weights. Next step is the pod run, details in work/DECISIONS.md (integration round 3).
- Not pushed. No draft PR yet.

## 2026-11-04 artifact loader hardening (done)

- Second review of tinyllama/artifacts.py (work/scratch/artifacts_review2.md): 1 blocker, several should-fix. W1 round 4 fixed them on feature/tinyllama-w1-hardening (03d746f5 T1.6a, 8a1bc871 T1.6b, 005cc44c T1.6c) and merged. D24: A16 checks digests only at load time.
- Verified by me on the worktree: full suite 579 passed, 0 failed, 0 skipped (4:39). The loader no longer imports the float recipe. tests/tinyllama/reference/tiny_artifact.py was changed (it now builds its recipe tables itself because artifacts.py no longer exposes them), in scope as a test helper.
- Unfixed nits 14 to 16 of the review (dead branch, sort tie rule, rename race) are optional.
- Pod: start retries every 30 min fail (host out of vCPU or memory); first failure 16:54 Almaty. Marcello chose to keep retrying the same pod (no replacement).

## Blockers

- Proof: verifier standard form requires NE > 0 (`routed-experts`); dense TinyLlama cannot reach `YES` without an agreed change. D1 resolved as option a: proof compatibility deferred (O1).
- Compute: heavy work on the RunPod cpu3g-8-32 pod (D7). Pod created 2026-11-04 (D7, docs/tinyllama/REPRODUCE.md).

## Remote compute

RunPod cpu3g-8-32 approved by Marcello (D7). Pod l925odnw8dlhn1 created 2026-11-04 14:38 Asia/Almaty (UTC+5); see D7.
