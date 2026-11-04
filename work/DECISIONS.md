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
- D9 (decided by brief). No edits to existing FlashProve sources on this branch; new code under tinyllama/ and
  tests/tinyllama/.

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
