# Proving plan: TinyLlama integer inference with FlashProve

Document version 1.0 (2026-11-05). Author: TinyLlama integer workstream (lead's agent). Base: branch
`feature/tinyllama-integer-audit` at 039f75fb. Contract: TL-INT-1, artifact
1d8e5cf46c89d951c1c449dc3c03d963f32176360a016d78e4d2537d39b088a9 (work/DECISIONS.md D35).

How to read numbers in this file. Every number is one of: (a) measured by us, with the report it comes
from; (b) quoted from a file of this repository as `path:line`, or from a named web page; (c) an
**estimate**, marked "Estimate En", with its formula and assumptions. Nothing in this plan was built or
run; it is a reading of the sources at 039f75fb plus arithmetic on quoted numbers.

## Summary (for Marcello)

1. Realistic: yes, with conditions. FlashProve's proof format and programs already define a dense model (NE 0) and every TinyLlama step (norm, rotary on every layer, GQA, softmax, SiLU gate, residual, logits, greedy choice); what blocks a proof today is a few hard gates and constants, not missing proof relations.
2. The gates: the verifier's "standard form" refuses dense models (`routed-experts`, `both-clamps`, `table-digests`), cp_commit refuses real weights under a non-standard form, and the weight readers accept only d 5120 and dh 128. Lifting them means editing FlashProve sources (today forbidden by D9) and a verifier policy change that Lean does not cover.
3. Hardware: the prover runs only on one NVIDIA H100 80 GB with CUDA 13.2 (README.md:63-68, prover/build.sh:617 `-arch=sm_90a`). CPU pods cannot prove; they can do all preparation and can verify.
4. Rough cost (Estimate E6): about 9 H100 hours in three sessions, 26.01 USD at 2.89 USD/hr (H100 PCIe) or 35.91 USD at 3.99 USD/hr (H100 SXM), RunPod secure cloud list prices, plus about 5.5 CPU pod hours at 0.32 USD/hr (1.76 USD).
5. Proving one TinyLlama window should be fast once it runs: Estimate E1 gives about 0.6 s of proving work at T 1024 on an H100, scaled from Maverick's measured 58.796 s per request (guide/performance.md:10). Fixed start-up and build time will dominate; the first GPU session measures the real figure.
6. Main unknown: cp_commit's worst-case range check (mandatory on real weights) has never seen our scales; if it fails we need roomier scales, which means a new contract version and a new quality run.
7. First three steps: M1 build a TinyLlama configuration kit and run the verifier's planning calls on CPU (no pod, no FlashProve edit); M3 measure the gate key margin and run the full-dimension cross check on the approved CPU pod; M2 prove a dense synthetic model at TinyLlama's shape on an H100 (needs GPU approval).
8. Decisions needed from Marcello: (a) allow edits to named FlashProve files on a new proving branch; (b) accept a dense standard form in the verifier (O1); (c) gate key form for the proof, recommended: no clamp (O2 option B); (d) digest set, recommended: three digests (O3); (e) H100 pod approval via Chief of Staff; (f) what to do if the worst-case check fails.
9. What a final "YES" will and will not mean: the C++ verifier accepted a proof that the committed TL-INT-1 integers give the stated greedy token; it is not covered by FlashProve's Lean theorem, whose scope is fixed to Maverick at full size (prover/examples/TARGET-KITS.md:153).
10. Goal of the plan: a `cp_verify` YES on a proof of TinyLlama's greedy next token for prompt P0, under a commitment whose integers equal the TL-INT-1 artifact, first at T 64, then at T 1024.

## 1. What FlashProve proves today

### 1.1 Parties, programs, hardware

- Programs: `cp_commit` commits to weights, `cp_prove` proves each reply, both "run on one GPU"; clients check
  with `cp_verify` on a CPU (AGENTS.md:3-5; prover/README.md:15-19).
- GPU stack, "the only supported platform for the prover": x86-64 Linux 4.14+, CPU with SHA extensions, one
  NVIDIA H100 80 GB used by nothing else, CUDA 13.2 Update 2 (nvcc 13.2.86), gcc/g++ 13 (README.md:63-68;
  prover/docs/build.md:6-10). The kernels are compiled with `-arch=sm_90a` (prover/build.sh:617), so an A100
  or consumer GPU is not an option. Building needs no GPU (prover/build.sh:14) and takes "some 26 to 35
  minutes" on 11 or more cores (prover/docs/build.md:76). NVIDIA driver 580 or later is needed for CUDA 13
  (prover/docs/build.md:9).
- Language: CUDA C++ for the provers (prover/), C++ for the verifier (verifier/src), a shared proof library
  (blocks/), a Lean 4 verifier and soundness proof (lean/), Python tools (scales/, weight_sums/,
  integrations/).
- Field: p = 2^64 - 2^32 + 1, Reed-Solomon codes (prover/examples/TARGET-KITS.md:3).

### 1.2 The statement

A request is a window of exactly T tokens plus a range of positions (docs/protocol.md:29-31). For the
committed integer weights a proof claims (docs/protocol.md:35-41):

1. every value of the integer model's run on the request stays inside its table or range;
2. the checked tokens meet the acceptance policy (default: exact match, each checked token is the top
   choice, docs/protocol.md:43);
3. if position T - 1 is in the range, the next token y is the top choice there.

The integer model is FlashProve's Model (docs/model.md), with NE = 0 meaning "all layers dense"
(docs/model.md:70-71). Padding counts as tokens; every value at position t depends only on tokens 0 .. t
(docs/model.md:283-290).

### 1.3 What the verifier checks

Commitment ID recomputed from the configuration (policy.txt, layout.txt, KL.bin) and the key file; all sizes
and rounds taken from the configuration; every challenge recomputed; every opened column hashed to its root;
then the sub-proofs (docs/protocol.md:211-219). A call can end `YES` only under the "standard parameter
form", 18 named conditions (verifier/docs/parameter-file.md:300-321, code `servedForm` in
verifier/src/vf_common.h:708-730). Otherwise it ends at best `NOT-A-VERIFICATION`, exit 14, which means "No
check failed, but the call was not a valid verification" (verifier/docs/output.md:260).

### 1.4 How a proof run is organised

- Commit once: weights encoded as digits into a weight table under a Merkle tree, a one-time range proof
  (weights in -4096 .. 4095 except Emb), leaf tables linked to the anchor (docs/protocol.md:111-131).
- Prove per batch of 1 to 64 requests (docs/architecture.md:148). Per request: a first chain that commits to
  the trace segment by segment, reduces everything by sumchecks and lookups to linear claims and one claim
  g M e + mu = v about the weight matrix (docs/protocol.md:155-167). Per batch: one shared chain opens the
  leaf table once for all requests (docs/protocol.md:171-185). Per request: a second chain ties the claim to
  the table (docs/protocol.md:187).
- Segments: one layer sub-proof and one attention sub-proof per block of 2^ll layers
  (verifier/docs/parameter-file.md:104), plus the output head (prover/README.md:58). Per request, not per
  token: one proof covers all positions of the window in one prefill pass (guide/performance.md:15, "each
  request's 1,024 prompt tokens count").
- Relations, by lookup call kind (verifier/docs/parameter-file.md:249-255): R16 and R8 range tables, FC clamp
  to -2047 .. 2047, FL the SiLU gate table, FS router sigmoid and gate clamp, FE the output head's
  exponentials, KR, KFD (softmax table) and WH for attention. Every sub-proof ends with a "smallness" test
  showing cells below 2^18 (verifier/docs/parameter-file.md, section 4). Norms are proved by a row of byte
  digits per token (the "ledger"; five limbs with `xl 5`).
- Weight commitments: the key file proves the table well formed; `weight_sums/` writes `wsums.txt`, the
  exponent and SHA-256 of each tensor's integers, and cp_commit "commits only if every hash matches"
  (weight_sums/README.md:3-7).
- Privacy: weights hidden by masks except a declared list (README.md:7-12). TinyLlama is public, so
  `-public-weights` (prover/weights_commitment/arguments.h:197-204) is the natural mode; hiding is then not a
  goal of this plan.

### 1.5 What is proved about the programs

Soundness is proved in Lean for the Lean verifier, for exact match, at one format and one size: the default
setting at full size (Maverick) (AGENTS.md:118-120; prover/examples/TARGET-KITS.md:153). The C++ verifier and
CUDA prover "are written to follow the same protocol and are tested against it, not themselves proved"
(AGENTS.md:118-120). Lean's soundness library fixes Maverick's rotary pattern and 48 layers (D3:
lean/Inst/Model/Tables.lean:171, lean/Johnson/Goal/Meaning/RopeFlags.lean:40-42). So any TinyLlama YES is a
C++ verifier result outside the Lean theorem.

### 1.6 Shapes that have run

Maverick full size (d 5120, L 48, H 40, G 8, dh 128, f 16384, NE 128, V 202048, T 1024;
prover/examples/target_full_f2/policy.txt) and small synthetic kits (d 512, L 8, NE 4, T 64;
prover/examples/target_small_f2_roomy/policy.txt). Every shipped configuration has NE > 0 (prover/tests/
configurations/*/policy.txt, integrations tiny_without_clamps: NE 8). No configuration in the repository has
NE 0, and none has ll 1. The verifier has planning tests for a dense kit (verifier/tests/served_form_test.py:
343-353, 385-389) but no proof of a dense model is recorded anywhere we read.

## 2. Coverage of TinyLlama's primitives

Our contract values come from docs/tinyllama/ARITHMETIC_CONTRACT.md and the artifact's KL.bin and params.json
(work/pod_reports/2026-11-05-run3/collect/x/repo/artifacts/TL-INT-1/1d8e5c.../). KL.bin has 286 words = 22
layers x 13, FlashProve's layout with no expert words (docs/model.md:103-111). Read from it: kq 24 to 27, kk
22 to 25, kv 12 to 13, ko 9 to 14, kg 10 to 14, ku 10 to 14, kd 13 to 17, M1 43 or 44, M2 43, K1 = K2 = 32, sh
7 in every layer, rp 1 in every layer; params.json: MF 43, KF 32, eps 85,899, s 17, e_res 11.

| primitive | FlashProve relation | fits TL-INT-1? | change needed |
|---|---|---|---|
| Embedding lookup | `toksrc 1`, `prompt-source`; Emb digits range-checked in the key file, `wb_emb 21` (docs/model.md:96-101) | yes, artifact embedding_bound 2^21 | none |
| RMSNorm without gains | norm ledger row, M, K from KL.bin, eps lines, `xl 5` | yes: 24 <= M <= 47, 17 <= K <= 32, K >= M - 14 (verifier/src/cp_verify.cpp:1403), K <= M (`norm-shifts`), eps < 2^48 with xl 5 | none |
| q, k, v projections and shifts | matmul sub-proofs, digit forms hq, hv (verifier/docs/parameter-file.md:122, :125) | yes: kq up to 27 needs form 2 (ceiling 32); kq differs from kk but both >= 16, allowed (contract 5.2) | policy lines only |
| Rotary, neighbour pairs, every layer | per-layer flag rp read by the prover's forward (prover/prover/forward.cuh:1523); verifier checks rp <= 1 (cp_verify.cpp:1405); table built by the verifier; `rotary-table` allows theta 10000, dh <= 128, T <= 1024 (vf_common.h:705-707) | yes in the programs; outside Lean scope (D3) | none in programs |
| GQA 32 / 4 | `shape-heads`: H dh = d, H + G <= 64, G power of two dividing H (parameter-file.md:106) | yes: 32 x 64 = 2048, 36 <= 64 | none |
| Causal softmax with table | attention sub-proof, callA KR, KFD, WH; `softmax-table` S 12, rx 8; `one-shift` | yes: S 12, rx 8, sh 7 everywhere | none |
| Attention output, Wo, residual | `ro`, `xs` limbs; ko <= 19 with xl 5 (parameter-file.md, limbs section) | yes: ko <= 14 | none |
| SiLU gate with clamped key (`gcl 1`) | FS table call CG and R8 call G0 (parameter-file.md:269) | no: `gcl` only with experts (cp_verify.cpp:562; prover/common/parameter_file.h:382) | O2: option B (no clamp) needs no code; option A needs prover and verifier work (section 4) |
| SiLU gate without clamp | key `gi` in two limbs, FL table (cp_verify.cpp:562-567) | same values wherever gw fits 16 bits (parameter-file.md:269); domain narrower | measure gw margin (M3) |
| Up projection | `ut 1`, 16-bit range | yes: condition 7 is in the contract | none |
| s u Wd, down shift | `rd`, kd <= 19 with xl 5 | yes: kd <= 17 | none |
| Final norm | MF, KF, epsF lines; MF <= KF + 14 (parameter-file.md:193) | yes: 43 <= 46 | none |
| Logits and greedy argmax | output head sub-proof, callH, FE; exact match | yes; V 32000 has six one-bits and is a multiple of 2^8 (GAP_MAP row Vocab) | none |
| Weight conversion | GGUF reader and fold constants, d 5120 and dh 128 only (prover/weights_commitment/fold_weights.cuh:174-186, fold_source.cuh:136-139, prover/weights_blocks/gguf_source.cuh:29-30, :128-129) | no | T4.1 and T4.2 |
| Sums file | weight_sums constants for d 5120 (weight_sums/weight_sums.py:146-151; weight_sums.cpp:693-697 per GAP_MAP) | no | T4.1 |
| Scales | KL.bin, same 13-word layout | yes | none; MF, KF, eps, s go into policy.txt |
| Table digests | digest lines only with NE > 0 (cp_verify.cpp:1194; parameter_file.h:401; parameters.py:79-82) | no (O3) | T4.3 |
| Standard form | `routed-experts`, `both-clamps`, `table-digests` (vf_common.h:709-714) | no (O1) | T4.3, decision |
| Real-weights commit | production cp_commit refuses a non-standard form (hf_commit.cu:908-914); test options refused on a folder of weights (arguments.h:219) | no | needs T4.3 |
| Worst-case check | mandatory on real weights; status 7 if a site can leave its range (arguments.h:112-117) | unknown | M5 first step; fallback F-WC |
| Lean statements | scope fixed to Maverick full size | no | out of scope; say so in every report |
| Python model.py | refuses NE 0 (parameters.py:304-307, :526-527) | bypassed in our cross check (O5) | only for cross checks, optional |

Honest summary: of 13 arithmetic rows, 11 fit unchanged, 1 (gate clamp) fits in its unclamped form, and 1
(the clamped form) does not. All blockers are in weight ingestion, parameter validation, the standard form and
the worst-case check, not in the proof relations.

## 3. Cost model

### 3.1 Size of the statement (computed from the contract shape)

Computed with Python integers on the box from section 1 of the contract:

| quantity | value |
|---|---|
| committed weights per layer: Wq 2048x2048, Wk, Wv 2048x256, Wo 2048x2048, Wg, Wu 2048x5632, Wd 5632x2048 | 44,040,192 |
| all 22 layers | 968,884,224 |
| Emb and Wh, 32,000 x 2,048 each | 65,536,000 each |
| total committed integers | 1,099,956,224 (equals the manifest's 1,100,048,384 parameters minus 45 norm vectors of 2,048) |
| matrix MACs per token (22 layers + head) | 1,034,420,224 |
| at T = 1024: matrix MACs; attention MACs (QK and AV, causal); softmax entries 22 x 32 x T(T+1)/2; SiLU lookups 22 x T x 5632; logit cells T x V | 1.059e12; 4.73e10; 369,459,200; 126,877,696; 32,768,000 |
| at T = 64: same | 6.62e10; 1.87e8; 1,464,320; 7,929,856; 2,048,000 |

For Maverick from docs/model.md section 2 (d 5120, dq 5120, dkv 1024, f 16384, fe 8192, NE 128, V 202048, 24
dense and 24 MoE layers): total committed integers 400,711,352,320 and active MACs per token 16,149,708,800
(one routed expert per token). Ratios TinyLlama / Maverick: committed 0.00275, active per token 0.0641,
attention at T 1024 (22 x 32 x 64 against 48 x 40 x 128) 0.183.

The repository's own size units (table entries 2^(b + n) from the line `table b n`, proof bytes, rounds and
bits) come from `cp_verify -plan accounting`, which needs no GPU (prover/examples/TARGET-KITS.md:56-58).
Quoted bounds: small kits at most 4,279,884 proof bytes, full size at most 19,254,524
(TARGET-KITS.md:56); b + n is 31 at the small kits and 43 at full size (prover/examples/run_target_example.sh:
84-85). TinyLlama's values are a deliverable of M1, not guessed here.

### 3.2 Measured figures of the closest model (Maverick, one H100)

All from guide/performance.md and docs/figures.tsv, 1,024-token windows, masking on:

- seconds per request at 1, 8, 24 per batch: 58.796, 13.413, 9.206 (guide/performance.md:10-12);
- commit: 38:43.77 on public files (figures.tsv:28), 2,312.68 s on synthetic weights (figures.tsv:98);
- host memory: serving 269.7 GB, committing 286.9 GB by the committer process (guide/performance.md:25;
  figures.tsv:26); GPU memory serving 72,981 MiB, committing 75,097 MiB (figures.tsv:41, :25);
- start-up of a service 68.1 to 79.0 s (guide/performance.md:19);
- verify about 21 s on one thread, up to 3.53 GB (guide/performance.md:23);
- small kit end to end (commit, prove, verify) 23 to 30 s (figures.tsv:79), "10 GB RAM" (README.md:59);
- build about 29 minutes at 24 CPUs (figures.tsv:40).

### 3.3 Estimates for TinyLlama

**Estimate E1 (proving work per request).** Fit t(N) = (F + r N) / N to the batch-1 and batch-24 figures:
F + r = 58.796, F + 24 r = 24 x 9.206 = 220.944, so r = 7.050 s per request and F = 51.746 s per batch. Check:
at N = 8 the fit gives 13.52 s against the measured 13.413 s. Assumptions: r scales with T times matrix MACs
per token; F scales with the number of committed integers; the GPU is equally efficient at TinyLlama's
smaller matrices (optimistic). Then:
- T = 1024: r = 7.050 x 0.0641 = 0.452 s, F = 51.746 x 0.00275 = 0.142 s, so about 0.59 s per single proof.
- T = 64: r = 0.452 x 64 / 1024 = 0.028 s, plus F, so about 0.17 s.
These exclude start-up, commit and build. Because the small kit (much smaller than TinyLlama) already takes 23
to 30 s end to end, E1 is best read as a lower bound; the real figure comes from M2.

**Estimate E2 (commit time).** 2,312.68 s x 0.00275 = 6.4 s of scaled work plus fixed costs. Same caveat.

**Estimate E3 (host memory).** Committing Maverick needs 286.9 GB from 232.13 GB of 4-bit files
(figures.tsv:26, :87), and 855.8 GB for its 16-bit files (figures.tsv:124), so memory follows the stored file
size. TinyLlama in BF16 is 2,200,119,864 bytes (work/STATUS.md, run 2 P3). Scaled: 286.9 x 2.20 / 232.13 =
2.7 GB plus fixed costs. The small kit's stated 10 GB (README.md:59) is a floor for fixed costs. A pod with at
least 64 GB host RAM is therefore an assumption with a wide margin; M2 measures it.

**Estimate E4 (GPU memory).** No scaling rule is documented; the programs "use nearly all" of the GPU's memory
(arguments.h, usage text). An H100 80 GB is required anyway.

**Estimate E5 (proof size and verify time).** Between the small and full kits' bounds, 4.3 MB to 19.3 MB
(TARGET-KITS.md:56); verify at most the order of Maverick's 21 s (assumption: verifier cost grows with the
logarithm of the sizes). M1 gives the exact proof size bound.

**Estimate E6 (pod time and money).** Hours are planning allowances, not measurements; cost = hours x quoted
rate. Rates: CPU pod cpu3g-8-32 0.32 USD/hr (docs/tinyllama/REPRODUCE.md:8); RunPod secure cloud H100 PCIe 2.89
USD/hr and H100 SXM 3.99 USD/hr (runpod.io/pricing, read 2026-11-05).

| session | pod | hours | at 2.89 | at 3.99 | at 0.32 |
|---|---|---|---|---|---|
| M1 kit and planning calls, only if the box lacks RAM | cpu3g-8-32 | 1 | | | 0.32 |
| M3 gate margin, full-dimension cross check | cpu3g-8-32 | 3.5 (run 3's similar chain took 206.85 min, REPRODUCE.md:33) | | | 1.12 |
| T4.2 GGUF export and sums comparison | cpu3g-8-32 | 1 | | | 0.32 |
| M2 synthetic dense proofs | H100 80 GB | 3 (build about 29 min, figures.tsv:40, plus toolkit install and runs) | 8.67 | 11.97 | |
| M5 real weights, T 64 | H100 80 GB | 3 | 8.67 | 11.97 | |
| M6 T 1024, whole reply, negative tests | H100 80 GB | 3 | 8.67 | 11.97 | |
| total | | CPU 5.5, GPU 9 | 26.01 | 35.91 | 1.76 |

The container disk is wiped on stop (REPRODUCE.md:27), so each session repeats setup and builds; that is in the
hours above. A contingency of one more GPU session (3 h, 8.67 or 11.97 USD) is advisable but not included.

## 4. Open items O2 and O3: options and recommendation

### O2, gate key form

- Option A, keep `gcl 1` in a dense model. Needs: the FS table (router sigmoid plus gate clamp pairs) to exist
  without experts in prover and verifier, the G0 and CG segments in a dense layout, removal of the refusals at
  cp_verify.cpp:562, prover/common/parameter_file.h:382, parameters.py:304-307, and the `tsig` digest. The
  function is the TL-INT-1 function on its full domain (gh up to 15 bits, keys up to about 2^22,
  parameter-file.md:269). Cost: CUDA and verifier work in code paths that have never run dense; largest risk.
- Option B, no clamp (recommended for the first proof). A dense file without `gcl` is a form the format
  already defines (docs/format/11-2-feed-forward-segment.md:1133; verifier "dense" test case,
  served_form_test.py:386). The proved function equals TL-INT-1 wherever gw fits 16 bits
  (parameter-file.md:269; contract 9.2); a prompt where some gw leaves -32,768 .. 32,639 has no proof (O2 text
  in work/DECISIONS.md). The artifact's integers do not change. Needs: a contract note "proof profile without
  gate clamp" and a reference mode that checks condition 6 on gw; a measurement of max |gw| (M3).
- Option C, fake experts to unlock `gcl`. Not possible without changing the model: with NE > 0 every odd layer
  becomes a mixture-of-experts layer and f must equal 2 fe with fe a power of two (parameter-file.md:107-110);
  5632 is not twice a power of two.

Recommendation: B now, A only if M3 shows gate keys close to 16 bits on real text.

### O3, table digests

- With option B the dense model reads three tables: SiLU, rotary, softmax. Our artifact pins exactly these three
  (params.json `digests`), with texp and tsilu equal to FlashProve's own files (contract section 6).
- Recommendation: the dense standard form requires the three digest lines tsilu, trope, texp (24 lines) and the
  verifier compares them with its own tables, as it does for four today (parameter-file.md, "Digests of the
  public tables"). With option A, four digests including a `tsig` computed with gcl.
- Alternative: no digest lines for dense. Simpler, but loses the cross-platform table guard that
  `table-digests` exists for; not recommended.

## 5. Window length and proof granularity

- T is a line of the parameter file (`lt`), so it is fixed per commitment and bound into the commitment ID.
  Allowed: lt 4 to 10 under `rotary-table` (vf_common.h:705-707; parameters.py:283). T 1024 is not required.
- Values at positions below T do not depend on T: each position reads only earlier ones (docs/model.md:288-290)
  and the rotary entry for (t, i) is computed from t and i only (contract section 6 recipe). So TL-INT-1 at T 64
  equals TL-INT-1 at T 1024 on the first 64 positions; only the rotary digest differs. This is reasoned; M5 tests
  it by comparing the proved token with our reference.
- Recommendation: first commitment at T 64 (lt 6), covering P0 (24 tokens) plus up to 40 reply tokens; then a
  T 1024 commitment for the stated contract. By E1 the proving work is roughly linear in T.
- Granularity: one request covers a range of positions; a prompt of n tokens and m reply tokens is checked by
  positions n - 1 .. n + m - 2 in one proof (docs/protocol.md:31). So a whole short conversation costs one
  prefill proof, the same as one token. Under exact match every covered token must be the integer model's
  greedy choice, so replies must come from our integer reference (or the prover's own forward), not from the
  float model. The clipped-gap policy for float replies needs fitted limits and is out of scope.
- The first proof: request = P0's 24 ids, then token 1576, then id 0 padding to T; r = 23, e = 24 (format of
  request files: prover/examples/TARGET-KITS.md:110-121). Expected: `covered:` shows that range, last line YES.

## 6. Milestones

Conventions. "Worker" = one coding worker (opencode with Bitdeer models or Claude Code) on its own branch, the
lead merges. New code goes under `tinyllama/proof/` and `tests/tinyllama/proof/` unless a FlashProve file is
named. Every FlashProve edit is listed and needs decision (a). Every pod use follows D7: one pod at a time,
stopped when idle; GPU pods need Chief of Staff approval first.

### M0. Decisions and branch (no compute, lead)

- Goal: decisions (a) to (f) recorded as D36 onward; new branch `feature/tinyllama-proof` from 039f75fb.
- Tasks: record decisions; add this plan's open items O7 (worst case) and O8 (dense standard form) to
  work/DECISIONS.md.
- Acceptance: `git log` shows the decision commit; DECISIONS.md lists the answers.
- Dependencies: Marcello. Pod: none. Risk: decisions delayed; M1 and M3 can still run (they need none).

### M1. TinyLlama configuration kit and planning calls (CPU)

- Goal: policy.txt, KL.bin, layout.txt for TinyLlama dense, no gcl, no digests, at lt 6 and lt 10, accepted
  by `cp_verify` planning calls; exact table size, proof size bound and bits printed.
- Tasks:
  - T1.1 (worker) `tinyllama/proof/make_kit.py`: from the artifact's params.json and KL.bin, write policy.txt
    starting from the 32 frame lines of prover/examples/target_small_f2_roomy/policy.txt (the default
    setting), shape lines (lt, L 22, ll 1, d 2048, H 32, G 4, dh 64, f 5632, fe 0, NE 0, V 32000), eps, theta0
    10000, digit forms chosen from the shift ceilings (hq 2; others the smallest form whose ceiling covers the
    largest shift), wb lines for absent kinds 0 (served_form_test.py:343-353 shows which), MF 43, KF 32, s 17,
    exact-match policy lines as in the kits, xl 5, no gcl, no rcl, no digest lines, call lines without the
    segments XR, LR, LT, DD, LH, SEL, L3, CLR, MR, G0, CG (served_form_test.py:351). la with 2^(la - lt)
    dividing 2048, 5632, 256 and 2304 (parameter-file.md:133), so la - lt <= 8. iota lines by the rule of
    verifier/tests/forest_lines.py. KL.bin copied byte for byte.
  - T1.2 (worker) tests: the kit generator is deterministic; KL.bin equals the artifact's; refusals for a
    wrong artifact id.
  - T1.3 (lead, CPU) build `cp_verify`; run `-plan arrays` and `prover/tools/pack_layout.py` to make
    layout.txt (prover/examples/TARGET-KITS.md:135; prover/docs/layout.md); then `-plan request` and
    `-plan accounting`. Iterate the `table` line and lm lines on the planner's refusals (each refusal names
    its line, parameter-file.md). Note that cp_prove additionally wants lmS, lmh, lmo, lmbA, lmq >= 11
    (parameter-file.md:296).
- Acceptance: `verifier/build.sh` ends `THE VERIFIER IS BUILT:` exit 0; `bin/cp_verify -plan request KIT
  128` prints `plan: not the standard parameter form (...)` naming only routed-experts, both-clamps and
  table-digests (the planning call never names next-token, exact-match or policy-named, parameter-file.md:328);
  `bin/cp_verify -plan accounting KIT 128` prints proof, key and entry sizes and the bits lines. Record b + n.
- Dependencies: none (no FlashProve edit, D9 holds). Pod: the verifier build needs "under 2 GB RAM"
  (README.md:57); the box has about 2 GB free, so run it on the box only if `free` shows more than 2 GB
  available, else on cpu3g-8-32 for about 1 h (0.32 USD). Risk: the frame lines (`group`, `stages`, `table`)
  were chosen for other table sizes (TARGET-KITS.md:92-96); if bits fall below 128 the stage lines need a
  re-derivation whose method is in a non-public report. Fallback: keep the full-size frame lines
  (target_full_f2) and only change `table`; report bits honestly.

### M2. Dense synthetic proofs on an H100 (de-risking (a))

- Goal: show that cp_commit, cp_prove and cp_verify run a dense model (NE 0, ll 1) at TinyLlama's shape, and
  measure time and memory.
- Tasks (lead or a worker on the pod, no FlashProve edit):
  - T2.1 GPU smoke test: `examples/quickstart-gpu.sh`, success line `quickstart-gpu: all steps ended as
    expected` and `release 13.2, V13.2.86` (README.md:113-123). Check `nvidia-smi` driver >= 580 first.
  - T2.2 `prover/build.sh tests` (test build: needed because the production cp_commit refuses a
    non-standard form, hf_commit.cu:908-914, while the test build goes on with `-test-bits`).
  - T2.3 tiny dense kit (one block: L 2 because ll 1 needs L a multiple of 2; d 64, T 64, derived from the
    tiny_without_clamps shape with NE 0): `cp_commit -weights made-up-blocks -test-seed FILE -test-bits N`,
    `cp_prove`, `cp_verify`.
  - T2.4 full dimension single block: the TinyLlama kit with L 2, T 64.
  - T2.5 full TinyLlama shape, L 22, T 64, then T 1024. Record wall times, `cp_prove`'s largest resident set,
    GPU memory by `nvidia-smi` sampling.
- Acceptance for each run: cp_commit exit 0; cp_prove writes `example.proof` and `example.y`, no `.failed`;
  `cp_verify ... -id ID` exits 14 with `NOT-A-VERIFICATION` and the result line `not the standard parameter
  form:` listing only the dense items (and, under test bits, the call's own `no-proof-of-work` if printed).
  No other failure. This is the expected outcome; a NO or any other refusal is a finding.
- Dependencies: M1 kit. Pod: H100 80 GB, about 3 h, 8.67 or 11.97 USD (E6); host RAM at least 64 GB (E3);
  disk 50 GB is enough for the toolkit (about 1.5 GB, build.md:12) and build. Needs Chief of Staff approval.
  Do not use `run_target_example.sh` for kits with b + n above 34: it demands 280 GB of available memory
  (run_target_example.sh:80-90); call the programs directly as its lines 312, 339, 353 do.
- Risk and fallback: the dense or ll 1 code path fails in CUDA. Fallback: localise with the tiny kit and the
  prover's unit tests; this is a FlashProve bug fix (decision (a)); if it is deep, stop and report.

### M3. Margins and cross checks on the CPU pod (de-risking O2, O5)

- Goal: know whether option B's 16-bit gate key holds on real text; finish O5 at full dimensions.
- Tasks:
  - T3.1 (worker) add max |gw| per layer and the count of |gw| outside -32,768 .. 32,639 to the range report
    (tinyllama/reference, a new optional statistic, no change to TL-INT-1 results). Tests on synthetic data.
  - T3.2 (lead, pod) run it on held-out (8 windows) and fit (32 windows); run the O5 cross check at full
    dimensions on real weights (`python -m tinyllama.reference.crosscheck` with full dims).
- Acceptance: report shows per layer max |gw| and zero keys outside 16 bits on all 40 windows (if not zero,
  option A is needed or those prompts are unprovable); crosscheck prints 0 DIFFERENT.
- Dependencies: none on Marcello (CPU pod already approved). Pod: cpu3g-8-32 about 3.5 h, 1.12 USD.
- Risk: keys exceed 16 bits; fallback option A (adds an estimated extra GPU and coding round).

### M4. Code changes for real weights (needs decision (a) and (b))

- T4.1 (worker) fold constants by FlashProve's own rule c = sqrt(d) / 2^t, 4^t <= d < 4^(t+1)
  (scales/README.md:595-601), and c_q = RN32(c / sqrt(dh)), in fold_weights.cuh:174-186, fold_source.cuh:136-
  139, gguf_source.cuh:29-30 and :128-129, weight_sums/weight_sums.py:146-151, weight_sums/weight_sums.cpp:
  693-697, integrations/vllm/integer_model/integer_model/block_weights.py:306-307 and :600. Acceptance: unit
  tests show 0x3F8F1BBD and 0x3DCA62C2 at d 5120, dh 128 (unchanged) and 0x3FB504F3, 0x3E3504F3 at d 2048, dh
  64; the existing suite rows for weight_sums still pass (SUITE.md).
- T4.2 (worker plus lead on CPU pod) GGUF export of the pinned checkpoint with llama.cpp's converter at
  e853ec9a, BF16, which applies the q, k permutation our converter copies (contract 4.4); `bin/weight_sums
  --eres 11` writes wsums.txt; `fold.txt` with e_res 11. A new script `tinyllama/proof/artifact_sums.py`
  computes each tensor's exponent and SHA-256 from the TL-INT-1 artifact in wsums.txt's format. Acceptance:
  the two sums files are equal line for line. This proves that what cp_commit will commit equals our artifact.
  Pod: cpu3g-8-32 about 1 h, 0.32 USD.
- T4.3 (worker) dense standard form: in verifier/src/vf_common.h:709-714 treat NE 0 as meeting
  `routed-experts` and `both-clamps` when neither rcl nor gcl is present; accept 24 digest lines (tsilu,
  trope, texp) for NE 0 in cp_verify.cpp (parse at about :1104-1113, count at :1165, refusal at :1194) and in
  prover/common/parameter_file.h:401, prover/hash_commitment/hf_commit.cu:586-590,
  prover/prover/header.cuh:48; update verifier/docs/parameter-file.md and the dense cases in
  verifier/tests/served_form_test.py:385-389. Acceptance: served_form_test passes with new expected results;
  `cp_verify -plan request KIT 128` on the M1 kit with digest lines prints no "not the standard parameter
  form" line. Risk: this is a policy change outside Lean; it needs decision (b).
- T4.4 (worker) `tinyllama/proof/make_request.py`: request files from token ids (TARGET-KITS.md:110-121), and
  greedy replies from our CPU reference for exact match.

### M5. First real proof at T 64 (H100)

- Goal: a YES for P0's next token under a commitment to TL-INT-1's integers.
- Tasks: build production programs (`../verifier/build.sh`, `./build.sh`); GGUF and sums as T4.2; create
  `.weights-are-public` files (arguments.h:202-203); `cp_commit -weights DIR -config KIT64 -M 1 -ivk ...
  -isk ... -public-weights FILE`; check its line `<n> of <N> tensors' integers have the sums of wsums.txt`
  shows all tensors; make the P0 request; `cp_prove`; `cp_verify`.
- Acceptance: cp_commit exit 0 (status 7 means the worst-case check failed, see F-WC; status 6 means the
  integers differ from wsums.txt); `cp_verify KIT ivk p0.req p0.y p0.proof -id ID` exit 0, last line `YES`,
  `covered:` r = 23, e = 24; report `r=`, `e=`, the `commitment:`, `proof of work:` and `result-v1` lines
  (AGENTS.md:111-114). Negative control: the same request with token 1576 replaced by the second-best token
  gets `.failed` from cp_prove.
- Dependencies: M1, M2, M4 (T4.1 to T4.4), decisions (a), (b), (c), (d), (e). Pod: H100 about 3 h, 8.67 or
  11.97 USD.
- Risk and fallback F-WC: if the worst-case check fails, options are (1) roomier scales with
  scales/worst_case.py, a new contract id (TL-INT-2) and a new quality run on the CPU pod; (2) a FlashProve
  change that skips the check under `-public-weights`, arguable because "Soundness does not rest on it"
  (scales/worst-case-bounds.md:33), but it is a change to a hiding safeguard and needs Marcello. To avoid
  surprises, run cp_commit first thing in the session: it fails before the seed is drawn.

### M6. T 1024 and whole replies (H100)

- Goal: the contract's T 1024 commitment; one proof covering a whole greedy reply; measured rates.
- Tasks: kit at lt 10; commit; prove P0 to P3 with replies of up to 40 greedy tokens each (made by our
  reference); record times and memory; negative tests as in TARGET-KITS.md:80-86 (altered key, leaf, proof,
  request, ID all rejected).
- Acceptance: four YES with `covered:` matching each reply; all altered calls rejected (NO or refusal, never
  YES).
- Dependencies: M5. Pod: H100 about 3 h, 8.67 or 11.97 USD.

### M7. Report and reproducibility (lead)

- docs/tinyllama/PROVING_RESULTS.md with all measured numbers, REPRODUCE.md steps for the GPU pod, the exact
  claim and its limits (not Lean-covered, public weights, C++ verifier only).

## 7. What I (the lead) can do without Marcello, and the safe order

Without Marcello (inside D1, D7, D9 as they stand): M1 entirely (new files under tinyllama/ only, CPU); M3
(CPU pod already approved); the GGUF export and artifact sums script of T4.2 (new code only); T4.4. Order:

1. M1 on the box if memory allows, else a 1 h CPU pod session.
2. M3 and the T4.2 export in one CPU pod session (about 4.5 h, 1.44 USD), stop the pod.
3. Ask Chief of Staff for one H100 session for M2 (3 h); this needs no FlashProve edit (test build only).
4. After decisions (a) and (b): M4 code on the box (light) with workers, then M5, then M6.

Decisions for Marcello: (a) edits to the FlashProve files listed in T4.1 and T4.3 on `feature/tinyllama-proof`
(this lifts D9 for those files); (b) a dense standard form in cp_verify (O1), accepting that a YES is outside
Lean's theorem; (c) O2 option B for the proof profile; (d) O3 three digests; (e) H100 pods: up to 9 h plus
one contingency session, 26.01 to 35.91 USD by E6 (plus 8.67 to 11.97 contingency); (f) the worst-case fallback
(new scales or skip under public weights); also confirm O4 stays as is (it does not affect proving).

## 8. Risks and unknowns, ranked

1. Worst-case check fails on our scales (unknown, cannot be tested without M4 and a GPU). Invalidates: the
   current artifact as a committable configuration; quality numbers if scales must change (new contract id).
2. Dense (NE 0) or ll 1 paths of cp_commit and cp_prove have never run (no such kit in the repository).
   Invalidates: the schedule and E6 hours; possibly needs CUDA fixes.
3. The dense standard form (T4.3) is not approved. Invalidates the goal "a verifier accepts": the best
   outcome is exit 14 "No check failed" (verifier/docs/output.md:260), which is evidence, not acceptance.
4. Security accounting at TinyLlama's table size under 128 bits with the shipped frame lines (M1). Invalidates
   YES at 128 bits until the stage lines are re-derived by the non-public method.
5. Gate keys exceed 16 bits on real text (M3). Invalidates option B for those prompts; needs option A.
6. RunPod H100 host: driver below 580, container blocks madvise(MADV_WIPEONFORK), no SHA extensions, or no
   capacity (build.md:6-9; REPRODUCE.md:27). Invalidates a session; fallback another data center or provider
   (needs approval).
7. Estimates E1 to E6 are scaling arguments from one other model; real costs could be several times higher at
   small sizes. Invalidates the budget, not the plan.
8. The prover's forward differs from our reference somewhere (rounding, rotary indexing at T 64). Invalidates
   M5's exact match; M3's full-dimension cross check and the 52-EQUAL tiny check (O5) reduce this risk.
9. GGUF export or sums mismatch (T4.2). Caught by cp_commit status 6; costs a CPU session.
10. Box memory: about 2 GB free; the verifier build is near that. Fallback: CPU pod.

## 9. What this plan did not verify

- Nothing was built or run: no cp_verify planning call on a TinyLlama kit, no GPU, no pod.
- Whether cp_prove and cp_commit handle NE 0 and ll 1 at run time (code has dense branches, e.g.
  prover/head/stack.cuh:343-369, but no recorded run).
- The exact command options of `cp_prove` and of `cp_commit -test-bits` beyond the usage text read in
  arguments.h and run_target_example.sh.
- RunPod H100 pod host RAM, CPU type, driver version and availability; prices are list prices read on
  2026-11-05.
- Line numbers for weight_sums.cpp:693-697 and block_weights.py:306-307, :600 were re-read and hold; other GAP_MAP citations were taken as recorded.
- The F + r N fit of E1 treats batching as linear although the prover passes over the weights with up to six requests at a time (guide/performance.md:6).
