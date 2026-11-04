# Arithmetic contract: TinyLlama-1.1B-Chat-v1.0 integer model

Contract id `TL-INT-1`, document version 1.0.0-draft (2026-11-04). Base: FlashProve `9e2bdd8f`.
Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` at revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
(docs/tinyllama/CHECKPOINT_MANIFEST.md). Versioning and the shared interfaces: docs/tinyllama/INTERFACES.md.

## 0. Status and scope

- This contract defines an integer function from token ids to integer logits and a greedy next token. It is an
  integer inference contract only (D1 option a, work/DECISIONS.md). It is not a proof statement. Nothing here claims
  that FlashProve can prove this function; section 13 lists what each deviation means for proving.
- Floating point is allowed only for offline conversion (section 4), calibration (section 5), table construction
  (section 6) and the float baseline evaluation. The path from token ids to integer logits (sections 7 to 11) uses
  integer arithmetic only: no float in any form, no library argmax, no float matrix product unless it is proved
  exact by the bounds of section 3.
- Base definition: FlashProve's Model (docs/model.md, sections 1 to 9) with NE = 0, the shape of section 1 below,
  rp = 1 in every layer, and the gate key form of `gcl 1` (section 9.1). Where this file says "as FlashProve", the
  cited lines of docs/model.md are normative, and this file only fixes parameters.
- Every claim is either cited (file:line at the base commit, or a pinned external source) or marked
  **hypothesis**. "Probe" means a light script run on the shared box on 2026-11-04 (work/STATUS.md), not a test.

## 1. Shape (fixed)

| symbol | value | source |
|---|---|---|
| L | 22 layers | config.json (CHECKPOINT_MANIFEST.md) |
| d | 2048 | hidden_size |
| n_h, n_g | 32 query heads, 4 key-value heads (group 8) | num_attention_heads, num_key_value_heads |
| dh | 64 = 2048 / 32 | derived |
| dq, dkv | 2048, 256 | n_h dh, n_g dh |
| f | 5632 | intermediate_size |
| V | 32000 | vocab_size |
| NE | 0 (all layers dense) | no expert tensors (CHECKPOINT_MANIFEST.md); docs/model.md:70-71 |
| T | 1024 = 2^10 (lt = 10) | D2 |
| theta0 | 10000 | rope_theta; rope_scaling null |
| eps_model | 1e-5 | rms_norm_eps |
| cs, expS, rx | 12, 12, 8 | as FlashProve standard form (verifier/docs/parameter-file.md:315-316) |
| eg, smax | 8, 4095 | the defaults (docs/model.md:68) |
| cw | 12 bits signed: values -2047 .. 2047 | docs/model.md:46-47 |
| weight bits | 13 bits signed: -4095 .. 4095 after conversion | docs/model.md:441-449; D5 |
| embedding bits | below 2^21 in absolute value | FlashProve converter rule, docs/model.md:459-461 |
| lm_head | untied: Wh is not Emb | tie_word_embeddings false |

## 2. Notation and primitive operations

All quantities are unbounded integers (docs/model.md:32-33). Indices start at 0. floor rounds toward minus
infinity. The primitives are FlashProve's (docs/model.md:35-40), restated with examples that tests must cover.

### 2.1 Signed rounding right shift rq

    rq(a, k) = floor((a + 2^(k-1)) / 2^k),  k >= 1

It divides by 2^k and rounds to nearest; ties go up (toward plus infinity), for negative a too. It is not
round-half-away-from-zero and not round-half-to-even. Worked boundary examples (computed with Python integers on
the box):

| a | k | a / 2^k | rq(a, k) | note |
|---|---|---|---|---|
| 1 | 1 | 0.5 | 1 | tie, up |
| -1 | 1 | -0.5 | 0 | tie, up (toward zero here) |
| 3 | 1 | 1.5 | 2 | tie, up |
| -3 | 1 | -1.5 | -1 | tie, up |
| 7 | 4 | 0.4375 | 0 | below tie |
| 8 | 4 | 0.5 | 1 | tie |
| -8 | 4 | -0.5 | 0 | tie |
| -9 | 4 | -0.5625 | -1 | past tie |
| 24 | 4 | 1.5 | 2 | tie |
| -24 | 4 | -1.5 | -1 | tie |
| -25 | 4 | -1.5625 | -2 | past tie |
| 2^40 - 1 | 20 | 1048575.999999 | 1048576 | large positive |
| -2^40 | 20 | -1048576 | -1048576 | exact |

Implementation rule: compute (a + 2^(k-1)) then an arithmetic (flooring) right shift by k, as FlashProve's
model.py does (integrations/vllm/integer_model/integer_model/arithmetic.py:19-26). In C or C++ the shift of a
negative signed value is arithmetic only from C++20 on; before C++20 and in C it is implementation defined, so a C
implementation must use an explicit floor division or a C++20 compiler. Truncating division (C `/`) is wrong for
negative a. The addition must not overflow the integer type (section 3).

### 2.2 Clamp and saturation

    clamp(a; b) = min(max(a, -b), b)

Symmetric: clamp(2048; 2047) = 2047, clamp(-2048; 2047) = -2047, clamp(40000; 2047) = 2047. The value -2048 never
appears in a 12-bit cell. Clamps are applied only where section 7 to 11 name them; everything else is either exact
or guarded by a range condition (section 3.2), which is a failure, not a saturation.

### 2.3 Other primitives

- isqrt(a): the largest r with r^2 <= a (docs/model.md:37). Exact integer square root (for example Python
  math.isqrt, or a verified Newton iteration on 128-bit integers). Not a float sqrt.
- floor division floor(a / b) with b > 0, also for negative a (arithmetic.py:29-33 uses rounding_mode floor).
- low8(a) = ((a + 128) mod 256) - 128, in -128 .. 127 (docs/model.md:39).
- argmax: the least index at which the maximum is attained (docs/model.md:38; arithmetic.py:46-58 writes the rule
  out instead of using a library argmax). Example: argmax of [5, 9, 9, -1] is 1.

## 3. Integer types, accumulator bounds and range conditions

### 3.1 Bounds at TinyLlama's widths

Inputs to every matrix product are 12-bit activations (|a| <= 2047) or the stream, and weights are 13-bit
(|w| <= 4095; the bounds below use 4096, as FlashProve's scales/words_to_policy.py:168 does). Computed on the box:

| product or sum | bound | log2 | type |
|---|---|---|---|
| a Wq, a Wk, a Wv, a Wg, a Wu over d = 2048 | 2048 x 2047 x 4096 = 17,171,480,576 | 34.00 | int64 |
| for reference: a 5632-term 12 x 13 bit dot product (no TinyLlama product has this form; Wd reads s u, next rows) | 5632 x 2047 x 4096 = 47,221,571,584 | 35.46 | int64 |
| rotary p0, p1 = C A0 -+ S A1, with \|C\|, \|S\| <= 4096 | 2 x 4096 x 17,171,480,576 = 140,668,768,878,592 | 47.00 | int64 |
| score s_ij over dh = 64 | 64 x 2047^2 = 268,173,376 | 28.00 | int64 |
| Z_i (at most 1024 terms of EX <= 4096) | 4,194,304 | 22.00 | int64 |
| acc_i over 1024 positions | 1024 x 4096 x 2047 = 8,585,740,288 | 33.00 | int64 |
| 2 acc + Z (softmax division numerator) | 17,175,674,880 | 34.00 | int64 |
| o Wo over dq = 2048, \|o\| <= 2047 | 17,171,480,576 | 34.00 | int64 |
| sum over j < 5632 of s[j] u[j] Wd[j][k], s <= 4095, \|u\| <= 32767 | 5632 x 4095 x 32767 x 4096 = 3,095,374,363,361,280 | 51.46 | int64 |
| logits aF Wh over d = 2048 | 17,171,480,576 | 34.00 | int64 |
| layer 0 norm: Vx of an Emb row, \|Emb\| < 2^21 | 2048 x 2^42 = 2^53 | 53.00 | int64 |
| Vx + eps at any norm: allowed up to condition 1 | below 2^78 | 78 | 128-bit or Python int |
| x[k] Rc in the norm | at most (sigma + 1)(2^M / sigma + 1/2) <= 2^(M+1) + 2^39, below 2^49 for M <= 47 (since \|x[k]\| <= sqrt(Vx) < sigma + 1 and sigma < 2^39) | 49 | int64 |

Consequences: every product and partial sum except Vx fits in int64 with margin, and every product except Vx is
below 2^53, so FlashProve's float64 exact-product method (arithmetic.py:61-80, exact when the sum of absolute
products is below 2^53) would also be exact at these widths; the CPU reference should still use integer arithmetic.
Vx needs more than 64 bits in the worst case allowed by condition 8 (2048 x 547,599,908,735^2 is about 2^89, which
condition 1 then rejects), so Vx + eps must be computed in 128-bit integers or Python int, as model.py does with its
wide numbers (arithmetic.py:92-160).

### 3.2 Range conditions

The model is partial (docs/model.md:294-296). The reference checks FlashProve's conditions 1 to 8 at every
position, layer and norm, with the column "accepted by the provers and model.py" (docs/model.md:298-307):

| # | value | accepted interval |
|---|---|---|
| 1 | Vx + eps at a norm | below 2^78 |
| 2 | norm quotient w[k] | -32,768 .. 32,639 |
| 3 | qw, kw | -32,768 .. 32,639 |
| 4 | vw | -32,768 .. 32,639 |
| 6 | gh, the high part of a gate key | -16,384 .. 16,383 |
| 7 | up projection u | -32,768 .. 32,639 |
| 8 | stream value x[k] read by a norm | \|x[k]\| <= 547,599,908,735 |

Condition 5 (router) does not apply (NE = 0). A failing condition is reported as a range failure with its number,
layer, position and index (INTERFACES.md I3); the reference then returns no logits for that input. It never
saturates instead. This is the narrower column, chosen so that a later proof profile is not promised a domain that
FlashProve's provers do not accept. The quality evaluation counts range failures as a metric.

## 4. Weight conversion (offline, float)

### 4.1 Source and decoding

The converter reads `model.safetensors` (sha256 6e6001da...4933, CHECKPOINT_MANIFEST.md) with a new BF16
safetensors reader (D4). A BF16 value widens exactly to float32 by appending 16 zero bits (docs/model.md:375,
397-402). Any infinity or NaN in weights, gains or products is a refusal (docs/model.md:454-455).

### 4.2 Which tensor is which matrix

HF stores a linear weight as [out][in] (row = output unit). FlashProve's W[k][j] has the input index first
(docs/model.md:42-44). So W[k][j] = HF_weight[j][k]; the stored layout is the HF layout (INTERFACES.md I2).

| HF tensor | matrix | shape (math, in x out) | reads gains of | constant |
|---|---|---|---|---|
| model.embed_tokens.weight [32000, 2048] | Emb[tok][k] | V x d | none | none |
| layers.l.self_attn.q_proj.weight [2048, 2048] | Wq (rows permuted, 4.4) | d x dq | input_layernorm | c_q |
| layers.l.self_attn.k_proj.weight [256, 2048] | Wk (rows permuted, 4.4) | d x dkv | input_layernorm | c_0 |
| layers.l.self_attn.v_proj.weight [256, 2048] | Wv | d x dkv | input_layernorm | c_0 |
| layers.l.self_attn.o_proj.weight [2048, 2048] | Wo | dq x d | none | none |
| layers.l.mlp.gate_proj.weight [5632, 2048] | Wg | d x f | post_attention_layernorm | c_0 |
| layers.l.mlp.up_proj.weight [5632, 2048] | Wu | d x f | post_attention_layernorm | c_0 |
| layers.l.mlp.down_proj.weight [2048, 5632] | Wd | f x d | none | none |
| lm_head.weight [32000, 2048] | Wh | d x V | model.norm | c_0 |

This is FlashProve's table (docs/model.md:413-426) with HF names. The gain vectors (input_layernorm,
post_attention_layernorm, model.norm, each [2048]) are not stored in the artifact.

### 4.3 Gain folding constants

FlashProve's norm has no gains and divides by the root of the sum of squares, not the mean (docs/model.md:170-182).
The gains and sqrt(d) go into the matrices. FlashProve's scale fit writes sqrt(d) = c 2^t with 4^t <= d < 4^(t+1)
and sets M - K = Es + t, so the conversion constant must be c (scales/README.md:595-601, :644-648). At d = 5120,
t = 6 and c = sqrt(5120)/64 (the constant 0x3F8F1BBD, docs/model.md:435). At d = 2048:

- t = 5, c = sqrt(2048) / 32 = sqrt(2).
- c_0 = RN32(sqrt(2)) = float32 bits 0x3FB504F3 (1.41421353816986083984375).
- c_q = RN32(sqrt(2) / sqrt(64)) = RN32(sqrt(2) / 8) = 0x3E3504F3. It includes the attention scale 1 / sqrt(dh)
  (docs/model.md:216, :415).

The bit patterns were computed on the box with Python struct from the float64 value and checked against a 50-digit
decimal sqrt; sqrt(2) is far from a float32 rounding tie, so the single rounding is unambiguous. The alternative
sqrt(d)/64 (0x3F3504F3, 0x3DB504F3) is exactly half of these and gives the same integers with e one higher
(section 4.5); the contract fixes 0x3FB504F3 and 0x3E3504F3 (work/DECISIONS.md O4).

### 4.4 q and k row permutation (verified)

HF Llama rotates entry i of a head with entry i + dh/2 (rotate_half on halves, cos and sin repeated over both halves;
transformers modeling_llama.py at commit 0c7f5ab8, lines 112 and 119-123). FlashProve rotates neighbouring entries
2i and 2i+1 (docs/model.md:189-198, :431-433). The converter applies llama.cpp's permutation to the output rows of
q_proj (per query head) and k_proj (per key-value head) before anything else (llama.cpp e853ec9a,
conversion/llama.py:177-182, applied at :261-265):

    new row  h dh + 2 i + j  =  old row  h dh + j (dh / 2) + i,   i < dh/2 = 32, j in {0, 1}

for h < 32 in q_proj and h < 4 in k_proj. v_proj and o_proj are not permuted. The scores q . k are unchanged because
the same permutation acts on q and k within every head, and FlashProve's pair i uses the frequency
theta0^(-2i/dh) of HF's index i (docs/model.md:134; modeling_llama.py:105). Signs agree: HF gives
q'[i] = q[i] cos - q[i + 32] sin and q'[i + 32] = q[i + 32] cos + q[i] sin; FlashProve gives p0 = C A0 - S A1 and
p1 = S A0 + C A1 with A0 = q[i], A1 = q[i + 32]. Status: **verified by reading and by a random data probe**
(work/scratch/perm_probe.py: max score difference 1.4e-12 with the permutation, 3.2e3 without). Pod check P1
(section 14) confirms it on the real weights. Gain folding acts on the input index, so it commutes with the
permutation.

### 4.5 From float to integer

As FlashProve (docs/model.md:441-452), per tensor:

1. If the matrix reads gains g: gamma_k = RN32(g_k c) for input index k, and pi = RN32(gamma_k v). Otherwise pi = v.
   Each RN32 is one float32 operation; nothing fused (docs/model.md:349-350).
2. mx = max |pi| over the tensor; mx = 0 is a refusal.
3. e = the largest integer with RNE(2^e mx) <= 4095. w = RNE(2^e pi), so |w| <= 4095. RNE is ties to even.
4. Emb: w = RNE(2^e_res v), no gains; |w| must be below 2^21 (FlashProve's converter limit, docs/model.md:459-461).
5. Refusals: a non-finite value; e outside 0 .. 24 (the bound FlashProve's programs apply to 16-bit source types,
   docs/model.md:455-457); a nonzero gamma_k below 2^-64 or above 2^64 (docs/model.md:458-459).

Hypothesis: every TinyLlama tensor has e in 0 .. 24 and max |Emb| 2^e_res below 2^21 at the fitted e_res. Settled
by pod check P2.

## 5. Scales and rescaling positions

### 5.1 The scale words

Per layer, 13 words in FlashProve's `KL.bin` order (docs/model.md:103-111): kq kk kv ko kg ku kd M1 K1 M2 K2 sh rp.
No mixture-of-experts words (NE = 0). Plus MF, KF, eps1, eps2, epsF, e_res and the logit exponent s (E_logit).

| position in the forward pass | operation | shift or exponents |
|---|---|---|
| 1st norm of layer l | a = N(x; M1, K1, eps1) | M1, K1 |
| after a Wq or a Wk and the rotary step | qw = rq(p, kq), kw = rq(p, kk); kq, kk also remove 2^cs | kq, kk |
| after a Wv | vw = rq(a Wv, kv) | kv |
| attention score to softmax key | u = floor((m - s) / 2^sh) | sh |
| attention output o | rounded division by Z, no shift | none |
| after o Wo, into the stream | ad = rq(o Wo, ko) | ko |
| 2nd norm | m = N(x'; M2, K2, eps2) | M2, K2 |
| after m Wg, the gate key | gw = rq(m Wg, kg), exponent eg = 8 | kg |
| after m Wu | u = rq(m Wu, ku) | ku |
| after (s u) Wd, into the stream | ad = rq(sum s u Wd, kd) | kd |
| final norm | aF = N(x^L; MF, KF, epsF) | MF, KF |
| logits | lam = aF Wh, no shift; lam / 2^s is about the float logit | s |

### 5.2 How the scales are fitted (offline, float)

As FlashProve's fit (scales/README.md:585-666), with t = 5 instead of 6:

- Es of q, k, v, u: the largest E with mx 2^E <= 2047 over the calibration set. Es of a norm output n: the largest
  E with (mx / c) 2^E <= 2047, at most 14 - t = 9.
- K1 = K2 = KF = 32; M = 32 + Es(n) + 5. So 24 <= M <= 47 needs Es(n) in -13 .. 9 (verifier rule
  docs/model.md:324-325; scales/words_to_policy.py:77-79).
- kq = Es(n1) + e(q) - Es(q) + 12, kk likewise, kv = Es(n1) + e(v) - Es(v), ko = Es(v) + e(o) - e_res,
  kg = Es(n2) + e(gate) - 8, ku = Es(n2) + e(up) - Es(u), kd = 6 + Es(u) + e(down) - e_res
  (scales/README.md:624-630; the 6 is the SiLU table's output exponent).
- sh: one value for all layers, the least of Es(q) + Es(k) - 8, in 4 .. 12 (scales/README.md:602-609, :634;
  words_to_policy.py:90). Then Es(q) = 8 + sh - Es(k).
- eps1 = eps2 = epsF = floor((2 d 4^e_res + 100000) / 200000) = floor((4096 4^e_res + 100000) / 200000)
  (scales/README.md:653-662). At e_res = 11 that is 85,899. At e_res = 2 it is 0, which is refused (every eps must be
  at least 1, docs/model.md:120), so e_res >= 3.
- e_res: chosen in 3 .. 14 so that no range condition fails on the calibration set and Emb stays below 2^21; start
  at FlashProve's default 11 (scales/README.md:371-381).
- s = E_logit = Es(nF) + e(lm_head), at most 63 (scales/README.md:670-673).

Shift restrictions kept from FlashProve (none relaxed): every shift at least 1 (docs/model.md:120); kq = kk unless
both are at least 16 (verifier/src/cp_verify.cpp:1400); the per-form shift ceilings 16, 24 or 32
(cp_verify.cpp:1389-1391); the no-overflow laws at the sites up, down, o and qk
(scales/words_to_policy.py:348-382); norm pairs 24 <= M <= 47, 17 <= K <= 32, K >= M - 14 (cp_verify.cpp:1402-1403)
and K <= M (`norm-shifts`, verifier/docs/parameter-file.md:317); rp in {0, 1}, here always 1.

## 6. Public lookup tables

All tables are built offline in IEEE 754 float64 by FlashProve's recipes (docs/model.md:125-152), then frozen as
integer files with two digests each: FlashProve's BLAKE2b table digest (docs/model.md:154-166) and sha256 of the
same 32-bit little-endian words. The reference loads the frozen integers, recomputes both digests and refuses a
mismatch; it never evaluates exp, cos or sin at inference time. Values below were produced on the box by
FlashProve's own functions in tables.py (rotary_tables, softmax_table, silu_table, table_digest) with torch stubbed
out (work/scratch/tables_probe.py).

| table | size | domain | construction (docs/model.md) | range | FlashProve digest words 0..7 | sha256 of digest words |
|---|---|---|---|---|---|---|
| C, S rotary | 1024 x 32 each (32,768 entries each) | t < 1024, i < 32 | 134-135: x = t pow(10000, -2i/64), rha(2^12 cos x), rha(2^12 sin x) | -4096 .. 4096 | trope: 2141436500 1986672181 1784239491 223369813 3762430148 1100108057 3971569448 1221024000 | 973450425e0f39b307e1561d7aaa70904b0fab3d5862e453b70ccfd7c60915c1 |
| EX softmax | u0 = 2307 entries | u = 0 .. 2306 | 136-137: rne(2^12 exp(-u / 2^8)), ends before the first 0 | 1 .. 4096 | texp: 3648852807 2402568834 4186852988 523559909 1873895626 3042864545 1726265496 165292856 | e27fdb2a6e9b273138ee66a6729de3c2c186ac1e81d24e5514ad29c155221b63 |
| Ts SiLU | 65,536 entries | keys -32,768 .. 32,767 | 138-139: c' = clamp of c to -16384 .. 16383, z = c'/256, rha(64 z/(1+exp(-z))), clamp 4095 | -18 .. 4095 | tsilu: 2048398297 192427613 2862900154 2853881485 2797826050 1420723544 3686369553 2593882461 | 0c26f9b15535c1bb9d1878552b5122158670d6eadd8e9f3ba988833e87e7a442 |

Digest words: trope over C(t, i) + 4096 at index t 32 + i, then S likewise; texp over EX; tsilu over Ts + 4096
(docs/model.md:162-166). The sha256 column hashes those same words. Cross-checks: the probe's texp equals the
texp lines of prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt (S 12, rx 8) and its tsilu equals
the tsilu lines of integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/policy.txt, so the
probe's digest chain is FlashProve's. trope for (T 1024, dh 64, theta0 10000) has no FlashProve file to compare;
W3 recomputes it independently (T3.2). Sample entries: C(0, i) = 4096, S(0, i) = 0; C(1, 0..2) = 2213, 2997, 3465;
S(1, 0..2) = 3447, 2792, 2184; C(1023, 31) = 4058, S(1023, 31) = 557; Ts at keys 0, 256, -256 = 0, 47, -17.

The combination theta0 10000, dh 64, T <= 1024, cs 12 is in the set for which FlashProve computed every rotary entry
to be at least 9.32e-8 from a rounding tie (docs/model.md:146-150, computed by scales/tests/table_ties.py, not
proved). The sigmoid table Sig is not used (no router).

## 7. Embedding

x^0_t = Emb[tok_t] (docs/model.md:270), a row of d integers below 2^21 in absolute value, with the stream exponent
e_res. Token ids must be in 0 .. 31999.

## 8. Normalization N(x; M, K, eps)

As docs/model.md:170-177, a root mean square norm without gains:

    Vx    = sum over k < 2048 of x[k]^2              (128-bit or unbounded)
    sigma = isqrt(Vx + eps)
    Rc    = floor((2^M + floor(sigma / 2)) / sigma)  (2^M / sigma rounded half up)
    w[k]  = rq(x[k] Rc, K)
    nr[k] = clamp(w[k]; 2047)

Worked example (illustrative d = 2, not TinyLlama's): x = [3, -4], eps = 1, M = 24, K = 17: Vx = 25, sigma =
isqrt(26) = 5, Rc = floor((16,777,216 + 2) / 5) = 3,355,443, w = [rq(10,066,329, 17), rq(-13,421,772, 17)] =
[77, -102] (real values 76.8 and -102.4). Range conditions 1, 2 and 8 apply (section 3.2). sigma >= 1 because
eps >= 1, so the division is defined.

## 9. One layer

Layer l reads the stream row x_t at every position t < n and uses layer l's scales.

### 9.1 Attention (docs/model.md:186-216)

1. a_t = N(x_t; M1, K1, eps1).
2. A = a_t Wq (2048 entries, 32 heads) and a_t Wk (256 entries, 4 heads). rp = 1 in every layer: in each head, for
   i < 32, with A0, A1 the head's entries 2i and 2i+1, C = C(t, i), S = S(t, i):
   p0 = C A0 - S A1, p1 = S A0 + C A1; qw = rq(p, kq) in a query head, kw = rq(p, kk) in a key head;
   q = clamp(qw; 2047), k = clamp(kw; 2047). Condition 3 on qw, kw.
3. vw = rq(a_t Wv, kv), v = clamp(vw; 2047). Condition 4.
4. GQA and causal softmax. Query head h reads key-value head g = floor(h / 8) (docs/model.md:201; HF repeat_kv
   repeats each key-value head 8 times in order, modeling_llama.py:168-177, so head h reads h // 8). Only j <= i
   (the causal mask is the range of j, not an additive mask):

       s_ij      = sum over c < 64 of q_i[h][c] k_j[g][c]
       m_i       = max over j <= i of s_ij
       u_ij      = floor((m_i - s_ij) / 2^sh)        (>= 0)
       e_ij      = EX[u_ij] if u_ij < 2307, else 0
       Z_i       = sum over j <= i of e_ij           (>= EX[0] = 4096)
       acc_i[c]  = sum over j <= i of e_ij v_j[g][c]
       o_i[h][c] = floor((2 acc_i[c] + Z_i) / (2 Z_i))

   The last line is acc / Z rounded to nearest, ties up: acc = 3, Z = 2 gives 2; acc = -3, Z = 2 gives -1;
   acc = 5, Z = 2 gives 3; acc = -5, Z = 2 gives -2; acc = -7, Z = 4 gives -2. o is not clamped; it is a weighted
   mean of 12-bit values, so |o| <= 2047.
5. ad = rq(o_t Wo, ko); x'_t = x_t + ad. No clamp on the residual (docs/model.md:213-214).

### 9.2 Feed-forward, SwiGLU (docs/model.md:228-242)

m = N(x'; M2, K2, eps2). For each hidden unit j < 5632:

    gw  = rq((m Wg)[j], kg)
    g0  = low8(gw),  gh = (gw - g0) / 256,  gc = clamp(gh; 127)
    key = 256 gc + g0
    s   = Ts[key]
    u   = rq((m Wu)[j], ku)
    ad[k] = rq(sum over j < 5632 of s[j] u[j] Wd[j][k], kd),  x'' = x' + ad

This is the `gcl 1` key form (work/DECISIONS.md O2): where gc differs from gh, Ts is constant, so the clamp only
saturates (docs/model.md:237-238). Examples: gw = 300 gives g0 = 44, gh = 1, key = 300; gw = -300 gives g0 = -44,
gh = -1, key = -300; gw = 128 gives g0 = -128, gh = 1, key = 128; gw = -129 gives g0 = 127, gh = -1, key = -129;
gw = 40000 gives g0 = 64, gh = 156, gc = 127, key = 32576 (Ts constant there, 4095). Conditions 6 on gh and 7 on u.
The SiLU output s is about 64 silu(gw / 256), so kd includes 6 (section 5.2). HF computes down(silu(gate) * up)
(hidden_act silu); the integer form matches that shape.

## 10. Final norm, logits and greedy choice (docs/model.md:266-274)

    aF_t  = N(x^22_t; MF, KF, epsF)
    lam_t = aF_t Wh                     (32000 int64 logits, unit 2^-s)
    y     = argmax of lam_(n-1), ties to the lowest token id

Only the last position's logits are needed for the next token; the reference may compute lam at all positions for
traces. No softmax, temperature or sampling is part of the contract.

## 11. Length, padding and positions

- Prefix mode (primary): n <= 1024 real tokens, no padding; positions 0 .. n-1; y from position n-1. This is Lean's
  `model` on n tokens (docs/model.md:288-290).
- Window mode: the n tokens followed by 1024 - n copies of id 0 (D6). Because every value at position t depends only
  on tokens 0 .. t (docs/model.md:288-290), the values at positions < n equal prefix mode exactly; range conditions
  are then checked at all 1024 positions, padding included (docs/model.md:283-286), which can only add failures.
- More than 1024 tokens is an error, not a truncation.

## 12. What is float and what is integer

| stage | arithmetic |
|---|---|
| reading BF16, gain folding, exponents, RNE to integers (section 4) | float32 as specified, offline |
| calibration statistics and scale fit (section 5) | float64, offline |
| tables (section 6) | float64 recipe, offline, frozen with digests |
| float baseline and quality metrics | float, evaluation only |
| tokens to integer logits and argmax (sections 7 to 10) | integer only |

## 13. Deviations from FlashProve arithmetic

"Agreed" means decided in work/DECISIONS.md. Proving implications are statements about what a future proof would
need; no proof compatibility is claimed (D1).

| item | FlashProve behavior (cited) | TinyLlama behavior | proving implication | status |
|---|---|---|---|---|
| Dense model, NE = 0 | Model allows NE = 0 (docs/model.md:70-71), but the standard form requires NE > 0 (`routed-experts`, verifier/docs/parameter-file.md:306; vf_common.h:709); parameters.py refuses NE = 0 with digests (parameters.py:79-82, :526-527) | all 22 layers dense | no call can end `YES` (at best NOT-A-VERIFICATION, parameter-file.md:300); needs a dense profile with its own soundness argument | agreed, deferred (D1, O1) |
| Shape d 2048, dh 64, L 22, H 32, G 4, f 5632, V 32000 | conversion programs accept only d 5120, dh 128 (weight_sums.py:148-151, block_weights.py:600, prover gguf_source.cuh:29); theorem scope stated for 48 layers (RopeFlags.lean:41) | TinyLlama shape | the verifier admits the shape rules (parameter-file.md:102-109) but a YES at another shape "says nothing about the numerical conditions of any theorem at that shape" (parameter-file.md:332); Lean scope must be restated | agreed (D1) |
| Rotary on every layer | Lean soundness library fixes no rotary at L mod 4 = 3 (lean/Inst/Model/Tables.lean:171; RopeFlags.lean:40-42); Lean `model` takes it per layer (Model.lean:195, :346-347) | rp = 1 in all 22 layers | outside the proved theorem's scope; the verifier itself only checks rp <= 1 (cp_verify.cpp:1405) | agreed (D3) |
| Gain folding constants | 0x3F8F1BBD and 0x3DCA62C2, d 5120, dh 128 only, other files rejected (docs/model.md:435-436) | 0x3FB504F3 and 0x3E3504F3 by the rule of scales/README.md:595-601 | none for a proof statement (no proof checks the conversion, docs/model.md:344-345); cp_commit and cp_prove refuse such files today | agreed by lead (O4) |
| Input format and q/k permutation | GGUF only; neighbouring file rows are pairs (docs/model.md:352-366, :431-433) | BF16 safetensors; converter permutes q, k rows (4.4) | none for the statement; commitment tools cannot read the artifact | agreed (D4) |
| Gate key form without experts | `gcl` only with experts (parameter-file.md:235; parameters.py:304-305); standard form needs gcl 1 (parameter-file.md:261) | gcl 1 key form in a dense model | a dense profile must admit gcl without experts; its lookup pairs sit in the sigmoid table's upper half (parameter-file.md:269), which a dense model lacks | needs Marcello decision (O2) |
| Table digests | four digests including tsig (docs/model.md:154-166); `table-digests` condition (parameter-file.md:310) | three digests (trope, texp, tsilu), no Sig | a dense profile must define the digest set | deferred (O3) |
| theta0 = 10000 | allowed by `rotary-table` (parameter-file.md:316); the scales converter refuses theta other than 500,000 (words_to_policy.py:85) | 10000 | inside the standard form's table condition; only tooling differs | agreed (inherent to checkpoint) |
| Artifact and parameter file | policy.txt with proof lines and digests, KL.bin (docs/model.md:53-56, :103-118) | own manifest and params file, KL.bin layout unchanged (INTERFACES.md I2) | a proof needs policy.txt, layout.txt and KL.bin bound by the configuration digest (parameter-file.md, "Digests of the files"); a generator would be needed | agreed (D1) |
| Window T = 1024 | T <= 1024 (parameter-file.md:316; parameters.py:283) | T = 1024; TinyLlama's 2048 not used | none | agreed (D2) |
| Padding id 0, right | vLLM integration pads with 0 after the reply (proof_middleware.py:87-88) | same, plus prefix mode | none; same as FlashProve | agreed (D6) |
| Range checks | provers' column (docs/model.md:298-307) | same column, failure not saturation | none | agreed (not a deviation) |
| Widths 12 and 13 bits | cw = 12 constant (docs/model.md:46); key file -4096 .. 4095 (docs/model.md:96-98) | unchanged | none now; a change would leave the format | agreed, revisitable (D5) |

## 14. Checks that need the pod (D7)

| id | check | where | settles |
|---|---|---|---|
| P1 | Float32 on real weights: for layer 0 and prompt P0, compute HF post-rotary q and k with modeling_llama's apply_rotary_pos_emb, and with permuted q_proj, k_proj rows and neighbour-pair rotation; compare the 32 x n x n score arrays (expect max relative difference below 1e-5). Then a full float forward with the permuted weights and neighbour rotary vs HF logits on P0 (same greedy token, max abs logit difference below 1e-3). | pod, CPU float32, about 5 GB RAM | permutation on real weights and on the correct axis |
| P2 | Per tensor: no non-finite value; e in 0 .. 24; gamma_k zero or within 2^-64 .. 2^64; max \|Emb\| 2^e_res below 2^21 for the fitted e_res. | pod, converter dry run | section 4.5 hypothesis |
| P3 | sha256 of the downloaded model.safetensors equals 6e6001da...4933 (not yet recomputed locally). | pod | source identity |
| P4 | Range report on TL-CAL-1 and the 8 held-out windows: zero range failures. | pod | scale fit |
| P5 | transformers AutoTokenizer at the pinned revision gives the I4 token ids (INTERFACES.md I4). | pod | tokenizer path |
