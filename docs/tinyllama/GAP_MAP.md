# FlashProve compatibility gap map for TinyLlama-1.1B-Chat-v1.0

Base: FlashProve `9e2bdd8f`. Checkpoint facts: `CHECKPOINT_MANIFEST.md`.
Method: reading docs and source only. Nothing was built, run, converted or loaded.
Sections A and B are findings with citations (file:line at the base commit). Section C holds hypotheses.
Section D records corrections made on 2026-11-04 after re-verification for the arithmetic contract.

## A. What FlashProve's integer model is

- `docs/model.md:8` "Model is Llama 4 Maverick's text transformer in integer arithmetic". Reference implementations: `integrations/vllm/integer_model/integer_model/model.py` (torch based, `_norm` :368, `_rotary` :415, `_attention` :435, `_feed_forward` :516) and `lean/IntegerModel/Model.lean` (docs/model.md:13-20).
- A dense model is part of the definition: `NE = 0` means all layers dense (docs/model.md:70-71); Lean covers both (lean/IntegerModel/Model.lean:12-15).
- Rounding `rq(a,k) = floor((a + 2^(k-1)) / 2^k)`, ties up; clamp; isqrt; argmax ties to lowest index (docs/model.md:35-38).
- Activations, queries, keys, values clamped to 12 bits signed, clamp(x; 2047) (docs/model.md:46-47).
- Weights: integer, outside Emb in -4096..4095; Emb up to 4 signed bytes, programs accept below 2^30, converter writes at most 21 bits (docs/model.md:96-101, 451-461).
- Shifts per layer in `KL.bin`: kq kk kv ko kg ku kd, M1 K1 M2 K2, sh, rp (docs/model.md:103-111).
- Tables (float64 recipe, digests): rotary C,S; softmax EX; SiLU Ts; sigmoid Sig (docs/model.md:123-166).
- RMSNorm without gains, integer isqrt and reciprocal (docs/model.md:168-182); gains folded into following matrices with constant c_0 including sqrt(d)/64 and 1/sqrt(dh) (docs/model.md:411-437).
- Attention: rotary on neighbouring pairs (2i, 2i+1) (docs/model.md:189-198, 431-433); GQA `g = floor(h / (n_h/n_g))` (docs/model.md:201); integer softmax with max subtraction, table lookup, one rounded division (docs/model.md:204-212); residual not clamped (docs/model.md:213-214).
- FFN: gated SiLU via table on a clamped key, up projection, product, down projection, one shift (docs/model.md:228-242). This is SwiGLU shape.
- Output: final norm, logits lam = aF Wh, greedy argmax lowest id (docs/model.md:266-274). Padding counts as tokens; vLLM integration pads with id 0 (docs/model.md:283-286).
- Range conditions and limb limits (docs/model.md:292-331). Verifier norm limits 24 <= M <= 47, 17 <= K <= 32, K >= M - 14 (docs/model.md:324-325).
- Scales fitting: `scales/README.md` (sections at :472 onward), worst case bounds `scales/worst-case-bounds.md`.

## B. Component gap table

| component | TinyLlama need | status | where / why |
|---|---|---|---|
| Integer rounding, clamp, argmax | same | exists | docs/model.md:35-47 |
| RMSNorm | eps 1e-5, gains per layer | exists, needs scale fit | docs/model.md:168-182; eps becomes integer eps1/eps2/epsF fitted with scales (docs/model.md:116-118, scales/README.md:651) |
| Norm gain folding | fold into q,k,v,gate,up,lm_head | needs extension | constants hardcoded for d=5120, dh=128: weight_sums/weight_sums.py:148-151, weight_sums/weight_sums.cpp:693-697, integrations/vllm/integer_model/integer_model/block_weights.py:306-307 and :600, prover/weights_commitment/fold_weights.cuh:181-186, prover/weights_commitment/fold_source.cuh:136, prover/weights_blocks/gguf_source.cuh:29 and :128; docs/model.md:435-436 "files with other values are rejected". The rule behind the constant is sqrt(d)/2^t with 4^t <= d < 4^(t+1) (scales/README.md:595-601; t = 6 at d = 5120). At d = 2048, t = 5: c_0 = RN32(sqrt(2)) = 0x3FB504F3 and, for Wq, RN32(sqrt(2)/8) = 0x3E3504F3 (correction 1) |
| Input format | HF safetensors BF16 | needs extension | converters read GGUF only (docs/model.md:352-366); BF16 is a read type but "has read no real model's file" in BF16 (docs/model.md:406-409); scales float pass accepts only `llama4` (docs/model.md:360-362, scales/gguf_weights.py:102); the scales converter refuses theta other than 500,000 (`arith-const`, scales/words_to_policy.py:85) (correction 4) |
| Rotary | theta 10000, dh 64, every layer | exists as arithmetic; pattern needs extension | theta 10000 and dh 64 inside the allowed set (docs/model.md:146-150; parameters.py:285,309). Python scales fix NO_ROPE_INTERVAL=4 (scales/gguf_weights.py:135-138); KL.bin has per layer rp (docs/model.md:110-111) but "The model of the Lean soundness theorem fixes this pattern" (docs/model.md:110-111). TinyLlama needs rp=1 in all 22 layers. Lean: `model` takes rp per layer (lean/IntegerModel/Model.lean:195, :346-347); the soundness library fixes `rotaryOff L := L % 4 = 3` (lean/Inst/Model/Tables.lean:171) for 48 layers (lean/Johnson/Goal/Meaning/RopeFlags.lean:40-42). Verifier and converter check only rp <= 1 (verifier/src/cp_verify.cpp:1405, scales/words_to_policy.py:327). Was hypothesis H3 (correction 3) |
| Rotary pair layout | HF q/k rows | needs conversion step | FlashProve pairs neighbouring rows of the file (docs/model.md:431-433). HF pairs entry i with i + dh/2 (rotate_half). The converter applies llama.cpp's q/k row permutation. Verified, was hypothesis H1 (correction 3) |
| GQA | 32 q heads, 4 kv heads | exists | docs/model.md:201; verifier: H*dh=d, H+G<=64, G power of two dividing H (verifier/docs/parameter-file.md:106). 32*64=2048, 36<=64, G=4 divides 32 |
| Softmax | causal, scale 1/sqrt(64) | exists | docs/model.md:204-216; table S 12, rx 8 (parameter-file.md standard form `softmax-table`) |
| SiLU / SwiGLU | silu(gate)*up | exists | docs/model.md:138-139, 228-242 |
| Dense FFN width | f = 5632 | exists | f 1..2^20, prover: at most eight one-bits and below 2^16 (parameter-file.md:107). 5632 = 4096+1024+512, three one-bits |
| MoE / router / sigmoid | none | not needed | NE=0 |
| Embedding, head, untied | separate Emb and Wh | exists | docs/model.md:91, 266-274, 425-426 |
| Vocab | V = 32000 | exists, check prover rule | V 2..2^20; prover: V rounded up to a multiple of 2^(la-lt) must have at most eight one-bits (parameter-file.md:109). 32000 has six one-bits; rounding depends on la, lt (not checked) |
| Layers | L = 22 | exists with ll = 1 | L multiple of 2^ll, at most 28 blocks (parameter-file.md:103-104). 22 = 11 blocks of 2 |
| Hidden width | d = 2048 | exists | d up to 8191 at sh 8 (parameter-file.md:105) |
| Sequence length | max 2048 positions | limited | parameters.py accepts lt 4..10 (parameters.py:283); verifier `rotary-table` needs T <= 1024 (parameter-file.md, standard form table). Window capped at 1024, not 2048 |
| Proof standard form | dense | BLOCKER for proofs | `routed-experts`: standard form requires NE > 0 (verifier/docs/parameter-file.md, standard form table, first row). Under other forms a call ends at best `NOT-A-VERIFICATION` (parameter-file.md:300). Also parameters.py refuses NE=0 with table digests (parameters.py:79-82, 192, 526-527). Also `both-clamps` cannot hold: `rcl` and `gcl` are allowed only with experts (verifier/docs/parameter-file.md:235, parameters.py:304-305) while YES needs both (parameter-file.md:261) (correction 2). So a dense TinyLlama cannot today get `YES` |
| Calibration set | TinyLlama prompts | needs extension | scales/make_calibration_set.py:5-7, :102 is written for Maverick. TinyLlama's tokenizer.json has the same sha256 as the Llama 2 tokenizer that the rule pins for steps 1 to 3 (scales/make_calibration_set.py:93-97), so steps 1 to 3 apply unchanged (D8) |
| CPU reference | integer, CPU | partly exists | model.py runs on a torch device (model.py:178); also tests/block_reference.cpp in integrations/vllm/integer_model/tests. Not yet tried for this shape. model.py's parameter reader refuses NE = 0 (parameters.py:526-527), so it cannot load a TinyLlama file unmodified (correction 5; work/DECISIONS.md O5) |

## C. Hypotheses (not verified)

- H1 (verified 2026-11-04, see D). HF Llama q_proj/k_proj rows use half-split rotary (rotate_half); llama.cpp's HF to GGUF conversion permutes q and k rows so pairs become neighbours. The converter must apply the same permutation before FlashProve's pairing. Verify against transformers `modeling_llama.py` and llama.cpp `convert_hf_to_gguf.py` at pinned versions.
- H2. Exporting TinyLlama to a GGUF with `general.architecture` set to the existing prefix may let weight_sums and block_weights reuse their readers once the d/dh constants are parameterized (docs/model.md:357-361 says they do not test the name).
- H3 (verified 2026-11-04, see D). Lean `model` may already accept rp=1 everywhere as a setting (lean/IntegerModel/Model.lean:70 "Which layers have no rotary step is a setting"), while the soundness theorem pins Maverick's pattern (docs/model.md:110-111). Needs a read of lean/THEOREM.md.
- H4. Fitting 12-bit activations and 13-bit weights on TinyLlama may cost quality; measure, do not assume.
- H5. Removing the `routed-experts` condition, or adding a dense profile, is a verifier and soundness change. It must not be done silently; see work/DECISIONS.md D1.

## D. Corrections after re-verification (2026-11-04)

1. Gain folding constant. Section B said TinyLlama needs RN32(sqrt(2048)/64) and RN32(sqrt(2048)/64/sqrt(64)). That was
   not derived from FlashProve's rule. The scale fit writes M - K = Es + t with sqrt(d) = c 2^t and expects the
   conversion constant to equal c (scales/README.md:595-601, :644-648); at d = 5120 that is sqrt(5120)/64. At d = 2048,
   t = 5 and c = sqrt(2) = sqrt(2048)/32. Bit patterns computed on the box with Python struct: 0x3FB504F3 (c_0) and
   0x3E3504F3 (c_0 / 8). The sqrt(2048)/64 forms are 0x3F3504F3 and 0x3DB504F3, exactly half; they give the same
   weight integers with an exponent e one higher (ARITHMETIC_CONTRACT.md 4.3). Also the prover hardcodes the same
   constants and widths (prover/weights_commitment/fold_weights.cuh:181-186, fold_source.cuh:136,
   prover/weights_blocks/gguf_source.cuh:29, :128); these were missing from the row.
2. Dense standard form. Besides `routed-experts`, a dense model cannot meet `both-clamps`: `rcl` and `gcl` are allowed
   only with experts (verifier/docs/parameter-file.md:235; parameters.py:304-305). The section A line "Lean covers
   both" stands for the function `model`, not for the soundness theorem's scope.
3. H1 and H3 verified. H1: transformers modeling_llama.py at commit 0c7f5ab8c579917b90bfa7502131e6420ec03978
   (sha256 363df58c...669de) defines rotate_half on halves (lines 119-123) with cos and sin repeated over the
   two halves (line 112); llama.cpp at e853ec9a8c72e91e8fb0eaa68e1dfe02f3c396f0 conversion/llama.py permutes q and
   k rows (lines 177-182, applied at 261-265 when undo_permute, which is True for LlamaModel at line 35). A random
   data probe (work/scratch/perm_probe.py) gives equal attention scores (max difference 1.4e-12 at scores up to
   3.3e3) for HF rotary on original rows and FlashProve pairing on permuted rows, and differences up to 3.2e3
   without the permutation. H3: see the Rotary row.
4. The scales converter also fixes theta to 500,000 (scales/words_to_policy.py:85, :330), not only the float pass.
5. model.py cannot load a NE = 0 parameter file with digests (parameters.py:526-527), so T2.3 needs another route.
