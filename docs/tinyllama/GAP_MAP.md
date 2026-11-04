# FlashProve compatibility gap map for TinyLlama-1.1B-Chat-v1.0

Base: FlashProve `9e2bdd8f`. Checkpoint facts: `CHECKPOINT_MANIFEST.md`.
Method: reading docs and source only. Nothing was built, run, converted or loaded.
Sections A and B are findings with citations (file:line at the base commit). Section C holds hypotheses.

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
| Norm gain folding | fold into q,k,v,gate,up,lm_head | needs extension | constants hardcoded for d=5120, dh=128: weight_sums/weight_sums.py:148-151, weight_sums/weight_sums.cpp:693-697, integrations/vllm/integer_model/integer_model/block_weights.py:306-307 and :600-602; docs/model.md:435-436 "files with other values are rejected". TinyLlama needs RN32(sqrt(2048)/64) and RN32(sqrt(2048)/64/sqrt(64)) |
| Input format | HF safetensors BF16 | needs extension | converters read GGUF only (docs/model.md:352-366); BF16 is a read type but "has read no real model's file" in BF16 (docs/model.md:406-409); scales float pass accepts only `llama4` (docs/model.md:360-362, scales/gguf_weights.py:102) |
| Rotary | theta 10000, dh 64, every layer | exists as arithmetic; pattern needs extension | theta 10000 and dh 64 inside the allowed set (docs/model.md:146-150; parameters.py:285,309). Python scales fix NO_ROPE_INTERVAL=4 (scales/gguf_weights.py:135-138); KL.bin has per layer rp (docs/model.md:110-111) but "The model of the Lean soundness theorem fixes this pattern" (docs/model.md:110-111). TinyLlama needs rp=1 in all 22 layers |
| Rotary pair layout | HF q/k rows | needs conversion step | FlashProve pairs neighbouring rows of the file (docs/model.md:431-433). See hypothesis H1 |
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
| Proof standard form | dense | BLOCKER for proofs | `routed-experts`: standard form requires NE > 0 (verifier/docs/parameter-file.md, standard form table, first row). Under other forms a call ends at best `NOT-A-VERIFICATION` (parameter-file.md:300). Also parameters.py refuses NE=0 with table digests (parameters.py:79-82, 192). So a dense TinyLlama cannot today get `YES` |
| Calibration set | TinyLlama prompts | needs extension | scales/make_calibration_set.py:5-7, :102 is written for Maverick |
| CPU reference | integer, CPU | partly exists | model.py runs on a torch device (model.py:178); also tests/block_reference.cpp in integrations/vllm/integer_model/tests. Not yet tried for this shape |

## C. Hypotheses (not verified)

- H1. HF Llama q_proj/k_proj rows use half-split rotary (rotate_half); llama.cpp's HF to GGUF conversion permutes q and k rows so pairs become neighbours. The converter must apply the same permutation before FlashProve's pairing. Verify against transformers `modeling_llama.py` and llama.cpp `convert_hf_to_gguf.py` at pinned versions.
- H2. Exporting TinyLlama to a GGUF with `general.architecture` set to the existing prefix may let weight_sums and block_weights reuse their readers once the d/dh constants are parameterized (docs/model.md:357-361 says they do not test the name).
- H3. Lean `model` may already accept rp=1 everywhere as a setting (lean/IntegerModel/Model.lean:70 "Which layers have no rotary step is a setting"), while the soundness theorem pins Maverick's pattern (docs/model.md:110-111). Needs a read of lean/THEOREM.md.
- H4. Fitting 12-bit activations and 13-bit weights on TinyLlama may cost quality; measure, do not assume.
- H5. Removing the `routed-experts` condition, or adding a dense profile, is a verifier and soundness change. It must not be done silently; see work/DECISIONS.md D1.
