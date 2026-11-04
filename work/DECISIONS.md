# Decisions and open questions

Resolved 2026-11-04 unless marked. Source for D1 to D5 and D7: Chief of Staff relaying Marcello's defaults.
D6 and D8 were picked by the lead under that delegation, with the rationale below. Citations are file:line at
base `9e2bdd8f` unless a path is under `work/scratch/` (git ignored probe files, see work/STATUS.md).

## Resolved

- D1 (resolved, option a). Integer inference only: CPU reference and quality evaluation. Proof compatibility is
  deferred and is an open item (O1). Nothing in this workstream claims proof compatibility, and no verifier
  restriction is relaxed. Rationale: the verifier's standard form requires NE > 0 (`routed-experts`,
  verifier/src/vf_common.h:709, verifier/docs/parameter-file.md:306), so a dense model cannot end with `YES`
  today, and options b and c both need a soundness argument that is outside this assignment.
  docs/tinyllama/ARITHMETIC_CONTRACT.md lists, for every deviation, its implication for proving.
- D2 (resolved). Sequence window T = 1024 (lt = 10). Rationale: FlashProve's `rotary-table` condition allows at
  most 2^10 tokens (verifier/docs/parameter-file.md:316) and parameters.py accepts lt 4 to 10
  (integrations/vllm/integer_model/integer_model/parameters.py:283); lt + S + 1 = 23 <= 24
  (parameters.py:310-313). TinyLlama itself supports 2048 positions (config.json, CHECKPOINT_MANIFEST.md); the
  integer model does not use positions 1024 to 2047.
- D3 (resolved). Rotary encoding on every layer (rp = 1 in all 22 layers) as a documented extension. Verified Lean
  impact: the `model` function in Lean takes the rotary flag per layer as a setting
  (lean/IntegerModel/Model.lean:70, :195, :346-347), so TinyLlama's pattern is expressible there; but the
  library under the soundness theorem fixes the pattern to "no rotary where L mod 4 = 3"
  (lean/Inst/Model/Tables.lean:171, `rotaryOff`) and states it for 48 layers
  (lean/Johnson/Goal/Meaning/RopeFlags.lean:40-42, `flags_in_scope`, `L : Fin 48`). So the TinyLlama model is
  outside the scope of the proved theorem. The verifier and the scales converter only check rp <= 1
  (verifier/src/cp_verify.cpp:1405; scales/words_to_policy.py:327).
- D4 (resolved). New safetensors reader for BF16 in the converter. Rationale: FlashProve's readers take GGUF only
  (docs/model.md:352-366) and block_weights.py has not read a real BF16 model (docs/model.md:406-409); reading
  the pinned safetensors directly keeps the source identity to one sha256 (CHECKPOINT_MANIFEST.md). Hypothesis H2
  (GGUF export) is not pursued.
- D5 (resolved, revisitable). Keep 13-bit weights (-4095 .. 4095 after conversion, docs/model.md:441-449) and
  12-bit activations (clamp to 2047, docs/model.md:46-47). Revisit only if the quality evaluation (T3.3) shows
  unacceptable loss; a change would leave FlashProve's format, where cw = 12 is a constant
  (docs/model.md:46) and the key file shows weights in -4096 .. 4095 (docs/model.md:96-98).
- D6 (resolved, lead). Padding: id 0 (`<unk>`), on the right only, no attention mask. Rationale from verified facts:
  (1) Model has no mask; padding counts as tokens (docs/model.md:283-286). (2) Every value at position t depends
  only on tokens 0 .. t (docs/model.md:288-290), so right padding cannot change any value at a real position.
  (3) FlashProve's integration pads after the reply with 0 (integrations/proof_middleware.py:87-88,
  integrations/ask_and_check.py:124, integrations/spool.py:142). (4) generation_config.json says pad_token_id 0;
  the tokenizer's pad `</s>` is id 2, the eos, which the chat template appends after every message
  (CHECKPOINT_MANIFEST.md, chat template), so padding with 2 would be indistinguishable from end-of-turn
  tokens and from the end-token check of spool.whole_reply (integrations/spool.py:151-158). The reference's
  primary mode takes n <= T real tokens with no padding (as Lean's `model`, docs/model.md:289-290); window mode
  pads to T = 1024 with id 0. Test: logits at every real position are equal, integer for integer, in both modes.
- D7 (resolved). Heavy work (download, conversion, calibration, model-dependent tests, quality evaluation) runs on
  a RunPod cpu3g-8-32 pod approved by Marcello. Nothing heavy on the shared box.
  RunPod pod (created 2026-11-04 14:38 Asia/Almaty, UTC+5): id l925odnw8dlhn1, flavor cpu3g-8-32 (8 vCPU,
  32 GB RAM), secure cloud, 0.32 USD/hr as reported by the API, image
  runpod/pytorch:1.5.1-cu1281-torch280-ubuntu2204, 50 GB container disk, no volume. SSH with a dedicated key
  only; no secrets, host, port or API key are recorded in git. The pod is stopped when idle. Unverified caveats:
  a stop may wipe the container disk (no volume), and the public SSH port may change on restart.
- D8 (resolved, lead). Calibration set `TL-CAL-1`: 32 windows of 1024 ids from the WikiText-2 raw training
  split, made with FlashProve's own sampling rule, steps 1 to 4 (scales/make_calibration_set.py:223-262), with
  TinyLlama's tokenizer. Fully specified in docs/tinyllama/INTERFACES.md, I4.3. Rationale: (1) TinyLlama's
  tokenizer.json has the same sha256 as the Llama 2 tokenizer that FlashProve pins for step 2
  (bcd04f0e..., CHECKPOINT_MANIFEST.md and scales/make_calibration_set.py:95-97), so no gated file and no
  second tokenizer are needed; (2) the input parquet is pinned by revision and sha256 in FlashProve
  (scales/README.md:414, scales/make_calibration_set.py:88) and is 6,357,543 bytes; (3) verified on the box: steps 1 to 3 with TinyLlama's
  tokenizer reproduce FlashProve's DRAWS_SHA256 8cd63f61... (work/STATUS.md). License: WikiText is
  CC BY-SA 3.0 and GFDL (dataset card at the pinned revision); the derived id windows stay out of git, only
  their sha256 is committed. 8 more windows from the same ids are held out for range validation only.
- D10 (resolved, lead, answers to W1 questions). (1) Scaling by a power of two is exact in float32 unless it
  overflows. (2) A zero gamma is allowed: contract P2 says gamma is zero or within 2^-64 .. 2^64.
  (3) iter_tensors must require a manifest; None is not accepted (W1 follow-up in T1.4).
- D11 (resolved, lead, W2 question 1; superseded in part by D17: the check order). Range check order.
  The reference checks layers in forward order and, within a layer, the conditions in the contract's numbering (ARITHMETIC_CONTRACT.md 3.2): 1, 2, 3, 4, 6, 7, 8; for one
  condition positions ascending, then indices ascending; the first failure is reported. Text in INTERFACES.md I3.1.
  Rationale: one reported failure must be the same in every implementation (Python reference, AMD C boundary), so
  the hash traces' `range_failure` field compares exactly.
- D12 (resolved, lead, W2 question 2). Tables in T2.2: the reference loads the four tables from the artifact and
  checks their digests (A16 through load_artifact); the float64 recipe stays in tests and in the offline converter
  only (tinyllama/convert/tables.py). tinyllama/reference/tables.py, which builds tables from the recipe, is for
  tests only from now on. Rationale: the reference must use no float (contract section 12), and the artifact is the
  single source of the tables the AMD workstream also reads.
- D13 (resolved, lead, W1 round 2 questions; documented in INTERFACES.md I2.3 and I2.4). (1) A caller supplied
  expected_id that differs stays A07 on manifest.json. (2) A KL.bin of wrong length reports only A10 (A06 and A08
  suppressed for that file); accepted. (3) Strict spelling of the I1 constant names in params.json is accepted;
  the exact list is in I2.3. (4) A malformed exponents.json reports A08. (5) The directory name must equal the
  artifact id, refused under A07 on manifest.json; needs a small code change, W1 follow-up F1, plus a W3 test.
  (6) A11 must also enforce the per-form shift ceilings of contract 5.2; the contract text did not say which form
  applies, so a clarification was added (D16); W1 follow-up F2 plus a W3 test. Rationale: each answer keeps one
  code per failure and matches what tinyllama/artifacts.py at 2a211c83 already does, except (5) and (6).
- D14 (resolved, lead, W1 deviation). validate_artifact and load_artifact take a keyword `dims=` so tests can use
  small synthetic artifacts. Accepted as test only and documented as non-normative in INTERFACES.md I2.4.
  Rationale: full size artifacts are about 2.33 GB and cannot be built on the shared box.
- D15 (resolved, lead, W3 question). The docs cite tests/tinyllama/arithmetic/tl_ref_independent.py and
  tl_conformance.py (renamed in 9b798da5); INTERFACES.md I7 updated. The synthetic full size artifact used by the
  malformed artifact tests has an apparent size of about 2.3 GB but is sparse and hard linked
  (tests/tinyllama/artifacts/synth_artifact.py, `write_sparse`, `os.link`), about 2.8 MB real use as reported by
  W3; accepted.
- D16 (resolved, lead, contract clarification for D13 (6)). FlashProve's shift ceiling (16, 24 or 32) depends on
  each class's digit form (verifier/docs/parameter-file.md:122, :125; cp_verify.cpp:1389-1391). A TL-INT-1
  artifact states no digit form, so every shift kq, kk, kv, ko, kg, ku, kd has the form 2 ceiling 32 and is at
  least 1. Added to ARITHMETIC_CONTRACT.md 5.2. A proof profile (O1) that fixes forms must check each class's
  ceiling; a lower ceiling is a new contract id.
- D9 (decided by brief). No edits to existing FlashProve sources on this branch; new code under tinyllama/ and
  tests/tinyllama/.
- D17 (resolved, lead, W2 report). Range check order is the compute order. The reported `range_failure` is the
  first failure in the reference's compute order: the steps of ARITHMETIC_CONTRACT.md section 9 for layers 0 to 21
  in forward order, then the final norm of section 10. Per layer: 8 on x, 1 on a_sigma, 2 on a_w, 3 on qw, 3 on kw,
  4 on vw, 8 on x_mid, 1 on m_sigma, 2 on m_w, 6 on gh, 7 on u; final norm: 8 on x_last, 1 on aF_sigma, 2 on aF_w
  (layer 22). Within one check positions ascending, then indices ascending. This withdraws the condition-number
  order of D11 (tinyllama/reference/layer.py `_norm_rows` and `layer` already check in this order; the W2 report
  says test T2.1 locks it). Text in INTERFACES.md I3.1. Rationale: a failing run aborts and carries no valid trace,
  so only the identity (condition, layer, position, index) matters, and a compute-order definition is the one a
  backend can follow without computing past a failure.

- D18 (lead, 2026-11-04). Per-position logits record. W2 must emit the `lam` record for every position
  (INTERFACES.md I3.1 lists it; tinyllama/reference/model.py already does). Answer to W3 round 3.
- D19 (lead, 2026-11-04). Integer to float logit conversion is `lam / 2^s` with s = E_logit (contract line 245).
  quality_eval.py uses exactly this. Answer to W3 round 3.
- D20 (lead, 2026-11-04). F1 behaviour in `load_artifact` is accepted: when another failure exists, the
  directory name A07 is left out so the first sorted failure stays the earlier one; `validate_artifact` still
  lists every failure including A07. Tests that use temporary directories need no renaming. Answer to W1 round 3.
- D21 (lead, 2026-11-04). The P4 range report keeps both views: the held-out set as a whole and each window.
- D22 (lead, 2026-11-04). The transformers version used for P1 is not guessed. The pod run installs the newest
  release compatible with torch 2.9.1, and the exact versions installed (`pip freeze` lines for torch,
  transformers, tokenizers, numpy, pyarrow) are written into REPRODUCE.md after the first pod run and then pinned.
- D23 (lead, 2026-11-04). `e_res` stays fixed at 11 (default) for the first pod run. An automatic search over
  3 to 14 is added only if P4 reports range failures.

- D24 (lead, 2026-11-04). Second review of tinyllama/artifacts.py (finding 8). A16 at load time checks table file
  digests against params.json and the tables_sha256 summary only. The loader does not recompute tables, which keeps
  float out of the loader as D12 requires. The check "table file equals its float64 recipe" stays in tests and in
  the converter (tinyllama/convert/tables.py), and in the pod run. INTERFACES.md I2.4 A16 text amended.
- D25 (lead, 2026-11-04). Review findings 1 to 7, 9 to 13 and 17 of work/scratch/artifacts_review2.md are assigned to
  W1 round 4 (T1.6), with tests for every sub-rule the review listed as untested. Finding 14 to 16: nits, optional.

- D26 (Marcello via Chief of Staff, 2026-11-04 22:03 Almaty). Pod l925odnw8dlhn1 could not start (21 failed start attempts from 16:54 to 21:36, host out of vCPU or memory, 0 USD spent). Marcello approved ONE replacement pod of the same type (cpu3g-8-32, secure cloud, 50 GB container disk, no volume, 0.32 USD/hr, stable image tag) in a different data center or host, and terminating the stuck pod once the new one is confirmed running. No other paid resource, no volume, no GPU. The retry routine was deleted. D7 now refers to the replacement pod; its id and data center are in docs/tinyllama/REPRODUCE.md.

- D27 (lead, 2026-11-05). `python -m tinyllama.convert.run` did nothing on the pod because run.py has no `if __name__ == "__main__"` guard. W1 adds it (T1.7). Pod commands use `python -m` again afterwards.
- D28 (lead, 2026-11-05). P1 failed on real weights at the logits check although layer 0 scores pass (max relative difference 2.5e-07). On P0 the HF float model picks token 1576 and P1's numpy forward picks 29934 (max absolute logit difference 32.4). The integer model, which uses the same converter, picks 1576 and tracks the float model (mean KL 0.0033, top-1 0.9375 on the small runs), so the first suspect is P1's own multi-layer float forward, not the permutation. W1 diagnoses on the box from work/pod_reports/2026-11-04-run2/reports/p1.json and the code, fixes P1, and P1 is rerun on the pod. Not called passed.
- D29 (lead, 2026-11-05). P5 with transformers 5.22.1: tokenization of P0 gives 23 ids (29871 missing after id 2) against the 24 committed ids made with the tokenizers library on tokenizer.json, and the chat template call returns a different shape ('input_ids' error). Decision: pin transformers to a 4.x release on the pod and record the installed version; check_inputs.py must accept both return shapes and must report the tokenizers-library ids and the transformers ids separately, so a difference is visible. The committed ids stay as they are until the 4.x rerun shows which side differs. Open item O6.
- D30 (lead, 2026-11-05). The conversion's range report (P4) ran the integer reference sequentially over 40 windows of 1024 positions: it was still running after 55 minutes at about 217 percent CPU with no output and was killed by the operator without a report. Fix (W1, T1.8): progress line per window, window-level parallelism with a process pool (exact integers, no float), memory-mapped weights shared between processes, a separate `range_report` command so conversion and P4 can run apart, and window selection options. The intra-window speed-up T2.4 is only started if the parallel run is still too slow on the pod.
- D31 (lead, 2026-11-05). `RuntimeWarning: overflow encountered in exp` at calibrate.py:208 (SiLU in float32). W1 checks that it is benign (large negative inputs give a sigmoid of 0) and replaces the expression with an equivalent that raises no warning and gives bit-identical maxima, or reports otherwise.
- D32 (lead, 2026-11-05). The reference keeps no float arithmetic (contract section 12). No float64 BLAS speed-up of the integer matmuls in the reference.

- D28 resolved (lead, 2026-11-05). Cause found by W1 and checked by the lead in the diff: calibrate._attention (the float forward shared by calibration and P1) lacked the 1/sqrt(dh) factor on the scores. The integer path was not affected (the scale is folded into Wq), which is why the integer model agreed with HF. Calibration maxima after the first attention block were wrong, so the artifact f72efbe3... and every number measured with it (P0 hash 50500da8..., quality runs) are SUPERSEDED. Weights are unchanged. Conversion, P4, P1, P0 and quality must be rerun. Fixed in 98342515, merged.
- D33 (lead, 2026-11-05). P1 logits criterion: relative max difference (max abs difference divided by max abs float logit) at most 1e-3 over all 24 positions, AND the same next token at the last position; the absolute difference and both tokens stay in the report. Reason: float32 over 22 layers with a different summation order makes 1e-3 absolute too tight for logits of size 10 to 20. The threshold is not tuned on results: if it fails, report the numbers and decide.
- D34 (lead, 2026-11-05). P4 covers both sets (held-out first, then fit) by default in range_report; the pod run uses `--sets both --workers 8`.

## Open items

- O1 (open, Marcello, deferred by D1). Proof compatibility of a dense TinyLlama. Needs at least: a dense
  standard form or profile (`routed-experts`), the gate clamp without experts (O2), the table digest set without
  a sigmoid table (O3), the rotary pattern and shape in the Lean scope (D3), and conversion tools that accept
  d = 2048, dh = 64 (GAP_MAP.md). None of these is done or claimed.
- O2 (needs Marcello decision; lead default applied). Gate key form. FlashProve allows `gcl 1` only with
  experts (verifier/docs/parameter-file.md:235; parameters.py:304-305) while the standard form requires it
  (parameter-file.md:261, :307). The contract uses the gcl key form (clamped key, docs/model.md:231-237). It gives
  the same values as the unclamped form wherever that is defined (docs/model.md:338;
  parameter-file.md:263, :269) and is defined on a wider domain. Alternative: the unclamped form, which is
  what a dense FlashProve parameter file can state today, with range condition 6 on gw (-32,768 .. 32,639).
- O3 (deferred with O1). Table digests: a FlashProve parameter file states four digests including `tsig`
  (docs/model.md:154-166), and parameters.py refuses digests with NE = 0 (parameters.py:79-82, :526-527). The
  TinyLlama artifact pins three (trope, texp, tsilu) and has no sigmoid table.
- O4 (lead, informational). Gain-folding constant: the contract uses FlashProve's general rule
  c0 = RN32(sqrt(d) / 2^t) with t = 5 at d = 2048 (scales/README.md:595-601), not sqrt(d) / 64. The two choices
  differ by exactly a factor 2 and give the same weight integers with exponent e one apart
  (ARITHMETIC_CONTRACT.md, section 4.3). Marcello may override.
- O5 (open, W2/W3). Cross check against FlashProve model.py (T2.3): model.py needs torch and its parameter file
  reader refuses NE = 0 with digests (parameters.py:526-527). Hypothesis: the cross check can drive model.py's
  arithmetic functions directly with constructed parameters; to be tried on the pod.

## 2026-11-04 integration round 3 (done)

- Merged into feature/tinyllama-integer-audit with --no-ff: c419d358 (W2 T2.2), b2cae17c (W3 round 3),
  3ef23887 (W1 round 3). No conflicts. Full suite on the box: `python3 -m pytest tests/tinyllama -q`, 520 passed,
  1 xpassed in 3:42. The xpass is the F1 test, whose xfail marker is being removed in a separate test commit.
- W1 round 3 (df69f3fe F1, 9f526481 F2, 13fd4a91 T1.3 calibrate/fit/run, 2180bded T1.5 check_permutation).
  W3 round 3 (0f1c11d4 F1 and F2 tests, 0ed9696d T3.4 check_inputs, d902ec92 T3.3 metrics and quality_eval).
- Nothing was run on real weights. P1 to P5, conversion, P4, the P0 run, bench_layer and quality_eval all need the pod.
- Pod prerequisites found in the worker reports: check_inputs.py needs Python 3.13 (or matching random.Random and
  numpy format behaviour), the stock pod image has an older Python, so install 3.13 first (for example with uv).
  quality_eval.py docstring says GPU pod; it runs on the CPU pod (float32 on CPU). Do not copy the Bitdeer key or
  Claude credentials to the pod: coding stays on the box.
- Pod command order (repo at /workspace/repo, checkpoint at /workspace/ckpt):
  1. `python3 tests/tinyllama/eval/check_inputs.py --checkpoint-dir /workspace/ckpt --parquet WIKITEXT_PARQUET --out p5.json --fit-npy /workspace/calib/fit.npy --heldout-npy /workspace/calib/heldout.npy` (P5, and writes the calibration windows).
  2. `PYTHONPATH=. python3 -m tinyllama.convert.check_permutation p3 --checkpoint /workspace/ckpt --report p3.json`
  3. `... check_permutation p1 --checkpoint /workspace/ckpt --report p1.json`
  4. `... check_permutation p2 --checkpoint /workspace/ckpt --report p2.json --e-res 11`
  5. `PYTHONPATH=. python3 -m tinyllama.convert.run --checkpoint /workspace/ckpt --out /workspace/repo/artifacts/TL-INT-1 --calib /workspace/calib/fit.npy --calib-sha256 FIT_SHA256 --heldout /workspace/calib/heldout.npy --heldout-sha256 HELDOUT_SHA256 --report /workspace/repo/artifacts/TL-INT-1/range-report.json`
  6. `python -m tinyllama.reference.run --artifact DIR --prompt P0`, twice, equal `lam_last_sha256`.
  7. bench_layer, then `python3 tests/tinyllama/eval/quality_eval.py --checkpoint /workspace/ckpt --artifact DIR --heldout /workspace/calib/heldout.npy --out-json quality.json --out-md quality.md`.
  Peak RAM estimates from W1 (design, not measured): calibration 680 MiB, folding 2.35 GiB, write_artifact 4.35 GiB,
  P1 5 to 6 GB, P2 about 1 GB. Pod has 32 GB.
