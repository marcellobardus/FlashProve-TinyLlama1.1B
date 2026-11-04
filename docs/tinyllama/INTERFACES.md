# Shared interfaces I1 to I4

Version 1.0.0-draft (2026-11-04), for contract `TL-INT-1` (docs/tinyllama/ARITHMETIC_CONTRACT.md). Owner: the lead.
Workers W1 (converter and artifacts), W2 (CPU reference) and W3 (tests and evaluation) build against this file and
do not change it; a needed change goes to the lead. Signatures below are pseudo-signatures, not code.

## I1. Arithmetic contract reference and versioning

- The contract id `TL-INT-1` names the integer function of ARITHMETIC_CONTRACT.md. Any change that can change one
  output integer (a constant, a table, a rounding rule, an order of operations, a range interval) is a new id
  (`TL-INT-2`, and so on). Wording changes that change no integer bump only the document version (1.0.0 to 1.0.1).
  A new optional field bumps the minor version.
- Every artifact (I2), trace (I3) and prompt set (I4) states the contract id it was made for. A reader refuses a
  different id; there is no compatibility mode.
- Constants of the contract (section 1 of the contract) are mirrored once in code, in `tinyllama/contract.py`
  (owner W1), as plain named integers: L 22, d 2048, H 32, G 4, dh 64, f 5632, V 32000, NE 0, T 1024, lt 10,
  cs 12, theta0 10000, S 12, rx 8, eg 8, smax 4095, cw 12, weight bound 4095, embedding bound 2^21, u0 2307,
  c0 bits 0x3FB504F3, cq bits 0x3E3504F3, pad id 0, gcl 1, range intervals of contract section 3.2, and
  CONTRACT_ID "TL-INT-1". W3 tests compare contract.py with the tables of the contract document.

## I2. Artifact format

### I2.1 Files

One directory per converted model: `artifacts/TL-INT-1/<artifact_id>/` (git ignored by `/artifacts/`). All
integers are little endian, C order (last index fastest), no header inside a data file, no padding.

| path | dtype | shape | meaning | bytes |
|---|---|---|---|---|
| manifest.json | UTF-8 JSON | | I2.2 | small |
| params.json | UTF-8 JSON | | I2.3 | small |
| KL.bin | u32 | [22][13] | FlashProve KL.bin layout, words kq kk kv ko kg ku kd M1 K1 M2 K2 sh rp per layer (docs/model.md:103-111); no expert words | 1,144 |
| emb.i32 | i32 | [32000][2048] | Emb[tok][k] | 262,144,000 |
| wh.i16 | i16 | [32000][2048] | stored[v][k] = Wh[k][v] (HF lm_head layout, folded with model.norm) | 131,072,000 |
| layers/LL.wq.i16 | i16 | [2048][2048] | stored[j][k] = Wq[k][j], rows permuted (contract 4.4) | 8,388,608 |
| layers/LL.wk.i16 | i16 | [256][2048] | Wk, rows permuted | 1,048,576 |
| layers/LL.wv.i16 | i16 | [256][2048] | Wv | 1,048,576 |
| layers/LL.wo.i16 | i16 | [2048][2048] | stored[j][k] = Wo[k][j] | 8,388,608 |
| layers/LL.wg.i16 | i16 | [5632][2048] | Wg | 23,068,672 |
| layers/LL.wu.i16 | i16 | [5632][2048] | Wu | 23,068,672 |
| layers/LL.wd.i16 | i16 | [2048][5632] | stored[k][j] = Wd[j][k] | 23,068,672 |
| tables/rope_c.i32 | i32 | [1024][32] | C(t, i) | 131,072 |
| tables/rope_s.i32 | i32 | [1024][32] | S(t, i) | 131,072 |
| tables/softmax_ex.i32 | i32 | [2307] | EX[u] | 9,228 |
| tables/silu_ts.i32 | i32 | [65536] | Ts[key] at index key + 32768 | 262,144 |
| exponents.json | UTF-8 JSON | | per tensor: e, mx as a float32 bit pattern, constant bits used; informational, for float interpretation | small |

LL is the layer number in two decimal digits (00 to 21). Every stored matrix has the HF layout [out][in]: one
output unit's weights are contiguous. Total about 2.33 GB (2,330,984,448 bytes of weights).

### I2.2 manifest.json

Canonical JSON: UTF-8, keys sorted, separators "," and ":", no insignificant whitespace, one final line feed,
ASCII only. Fields:

- `format`: "tl-int-artifact"; `format_version`: 1; `contract`: "TL-INT-1".
- `source`: `repo_id`, `revision` (fe8a4ea1...), `model_safetensors_sha256` (6e6001da...4933),
  `flashprove_base_commit` (9e2bdd8f...).
- `converter`: `git_commit` of this repository, `command` (the exact command line), `python`, `numpy` versions.
- `calibration`: `set` ("TL-CAL-1"), `fit_sha256`, `heldout_sha256` (I4.3).
- `files`: a list sorted by `path` of objects `path`, `dtype` (one of "u32", "i16", "i32", "json"), `shape`
  (list, empty for JSON), `bytes`, `sha256`. Every file of I2.1 except manifest.json itself.
- The artifact id is the sha256 of the manifest bytes. The directory name is that id.

### I2.3 params.json

Canonical JSON as above, with: `contract`; every constant of I1; `e_res`; `eps1` and `eps2` (lists of 22);
`epsF`; `MF`, `KF`; `s` (E_logit); `digests` with `trope`, `texp`, `tsilu`, each a list of 8 u32 words
(FlashProve digest, docs/model.md:154-160); `tables_sha256` per table file. No proof-system lines (no policy.txt).

### I2.4 Validation rules

`validate_artifact` applies all rules and reports every failure; `load_artifact` calls it first and loads nothing
if any rule fails. Rejection codes (each needs a malformed-artifact test by W3):

| code | rejected when |
|---|---|
| A01 manifest-parse | manifest.json absent, not UTF-8, not JSON, has a duplicate key, or is not canonical |
| A02 format | format, format_version or contract differ from the reader's |
| A03 path | a path is absolute, contains "..", a backslash or a symlink, or leaves the directory |
| A04 missing | a listed file is absent |
| A05 unlisted | a file in the directory is not listed (manifest.json excepted) |
| A06 size | bytes differ from the product of shape and item size, or from the file's size (truncated or trailing bytes) |
| A07 sha256 | a file's sha256 differs |
| A08 shape | dtype or shape differ from I2.1 |
| A09 weight-range | a matrix entry outside -4095 .. 4095, or an Emb entry with absolute value 2^21 or more |
| A10 kl-shape | KL.bin length is not 1,144 bytes |
| A11 kl-shift | a shift below 1 or above 32; kq differs from kk while one is below 16 |
| A12 kl-norm | M outside 24 .. 47, K outside 17 .. 32, K < M - 14, or K > M (same for MF, KF) |
| A13 kl-rp | rp is not 1 in some layer |
| A14 kl-sh | sh differs between layers or is outside 4 .. 12 |
| A15 params | a constant differs from I1; e_res outside 3 .. 14; an eps differs from floor((4096 4^e_res + 100000) / 200000) or is 0; s above 63 |
| A16 table | a table file differs from its recipe (recomputed), or a digest differs from params.json |

### I2.5 Loader boundary (W1 writes, W2 and W3 read)

- `write_artifact(out_dir, matrices, emb, kl_words, params, tables, provenance)` returns the artifact id. Writes to a
  temporary directory and renames at the end; never overwrites.
- `validate_artifact(dir)` returns a list of (code, path, detail); empty means valid.
- `load_artifact(dir, expected_id=None)` returns an Artifact with typed read only arrays (numpy int16, int32,
  uint32, memory mapped) and parsed params; raises ArtifactRejected(code, path, detail) on the first failing rule
  after running all of them.
- C boundary for the AMD workstream (pseudo prototypes): `int tl_artifact_open(const char *dir, tl_artifact **out)`
  returns 0 or a positive rejection code number (A01 = 1, and so on); `const int16_t *tl_matrix(const tl_artifact *,
  int layer, int which)`; `const int32_t *tl_table(const tl_artifact *, int which)`; `void tl_artifact_close(tl_artifact *)`.

## I3. Trace format

### I3.1 Records

The reference emits, through `record(name, layer, values)`, the intermediates named in FlashProve's model.py
(integrations/vllm/integer_model/integer_model/model.py:252-256), without the expert ones:

| name | layer | shape | contract section |
|---|---|---|---|
| x | 0 .. 21 | [n, 2048] | stream entering the layer |
| a_sigma, a_Rc | 0 .. 21 | [n] | first norm parts |
| a_w, a | 0 .. 21 | [n, 2048] | first norm quotient and output |
| qw, q | 0 .. 21 | [n, 32, 64] | 9.1 step 2 |
| kw, k | 0 .. 21 | [n, 4, 64] | 9.1 step 2 |
| vw, v | 0 .. 21 | [n, 4, 64] | 9.1 step 3 |
| o | 0 .. 21 | [n, 32, 64] | 9.1 step 4 |
| x_mid | 0 .. 21 | [n, 2048] | 9.1 step 5 |
| m_sigma, m_Rc, m_w, m | 0 .. 21 | [n] or [n, 2048] | second norm |
| gw, key, s, u | 0 .. 21 | [n, 5632] | 9.2 |
| x_last | none | [n, 2048] | stream after layer 21 |
| aF_sigma, aF_Rc, aF_w, aF | none | [n] or [n, 2048] | final norm |
| lam | none | [n, 32000] | logits |
| first_choice | none | [n] | argmax per position |

All values are int64 (every recorded value fits, contract 3.1). The order of records is the table's order, layer by
layer.

### I3.2 Files

- Hash trace (committed): `tests/tinyllama/vectors/TL-INT-1/<artifact_id first 16 hex>/<prompt id>.<mode>.json`,
  canonical JSON with `contract`, `artifact_id`, `prompt_id`, `mode` ("prefix" or "window"), `n`,
  `token_ids_sha256_u32le`, `records` (a list of name, layer, shape, sha256 of the values as int64 little endian
  in C order, min, max), `range_failure` (null, or condition number, name, layer, position, index, value),
  `next_token`, `lam_last_sha256`, `top16` (the 16 largest last-position logits as [id, value], by value
  descending, then id ascending). About 470 records (21 per layer and 8 after the layers), about 100 KB per prompt.
- Full trace (not committed): `traces/TL-INT-1/<artifact_id>/<prompt id>.<mode>/NAME.LL.npy` (numpy `<i8`), git
  ignored by `/traces/`. P0 in prefix mode is about 0.2 GB (24 positions).
- Size rule for git: one committed vector file at most 256 KB, all of tests/tinyllama/vectors at most 2 MB.
  Larger evidence is kept as a sha256 in a committed file plus the command that regenerates it.
- Synthetic vectors (W3): small shapes, made from the contract text by W3's own implementation, may hold full
  values within the same size rule.

### I3.3 Reference boundary (W2)

- `forward(artifact, token_ids, mode="prefix", record=None)` returns a Result with `lam_last` (32000 Python or
  int64 integers), `next_token`, `range_failure` (None or a RangeFailure). token_ids: 1 to 1024 ids in 0 .. 31999;
  otherwise ValueError. In window mode the input is padded with id 0 to 1024.
- `TraceWriter(path, full=False)` is a `record` callable; `full=True` also writes the .npy files.
- `trace_digest(path)` returns the hash trace object of I3.2.
- Command line: `python -m tinyllama.reference.run --artifact DIR --prompt P0 [--mode prefix|window] [--trace OUT]`
  prints the next token, lam_last_sha256 and the artifact id.
- C boundary (AMD conformance, pseudo): `int tl_forward(const tl_artifact *, const int32_t *tokens, size_t n,
  int64_t *lam_last, int32_t *next_token, tl_range_failure *fail)`; and per block
  `int tl_layer(const tl_artifact *, int layer, size_t n, const int64_t *x_in, int64_t *x_out, tl_record_fn, void *)`
  so a backend can be compared layer by layer against I3 hashes.

## I4. Prompt set, tokenization and calibration set

### I4.1 Prompt set TL-PROMPTS-1

Machine readable: docs/tinyllama/prompts_v1.json (messages, rendered text, text sha256, token ids, ids sha256).

| id | messages | n | token ids sha256 (u32 LE) |
|---|---|---|---|
| P0 | user: "What is the capital of France?" | 24 | cf21aeeb9804415e7fc352fe231186a7c1a54688238eecb628ed5ec68595e08b |
| P1 | system: "You are a helpful assistant."; user: "Name three primary colors." | 37 | 24ba7d7a91c40538a7115856fecddb1c6b38f322fcbc50ef1589a5959bbbc047 |
| P2 | user: "A mandate allows at most 10% of the portfolio in one stock. The trade would put 12% in one stock. Is the trade allowed? Answer yes or no." | 57 | b20d0d04468b2eec10be1c7499b3bf4fd9509a5ba19c6709c288ad9640de8f06 |
| P3 | user: "Write the number 7 in words." | 25 | 8ab66769e574d332b2d6a0ef5f40e61fef683f5e2636813c5c9f6f5153448e96 |

First milestone prompt: P0, prefix mode. P0's rendered text is `<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n`
and its ids are 1, 529, 29989, 1792, 29989, 29958, 13, 5618, 338, 278, 7483, 310, 3444, 29973, 2, 29871, 13,
29966, 29989, 465, 22137, 29989, 29958, 13.

### I4.2 Rendering and token id policy

- Render with the pinned chat template (sha256 66291cf0..., CHECKPOINT_MANIFEST.md) by Jinja2 with
  trim_blocks and lstrip_blocks, as transformers does (utils/chat_template_utils.py:599 at commit 0c7f5ab8),
  add_generation_prompt true, eos_token `</s>`. The template inserts no BOS.
- Tokenize the rendered text with the pinned tokenizer.json and add_special_tokens true. Its post processor
  prepends `<s>` (id 1); this matches the model card's usage, which renders with tokenize=False and passes the
  text to a pipeline (TinyLlama README.md lines 57-58 at the pinned revision). `</s>` inside the text becomes id 2.
- The ids in prompts_v1.json are normative. The tokenizer is used only to regenerate and check them. They were made
  on the box with tokenizers 0.22.2 and Jinja2 3.1.6. Hypothesis until pod check P5: transformers' AutoTokenizer
  gives the same ids. If not, the JSON stays normative and the difference is recorded.
- Generation stops are not part of the contract; evaluation treats id 2 as end of reply.

### I4.3 Calibration set TL-CAL-1 (D8)

Input: WikiText-2 raw training split, `datasets/Salesforce/wikitext` revision
b08601e04326c79dfdd32d625aee71d232d685c3, file wikitext-2-raw-v1/train-00000-of-00001.parquet, 6,357,543 bytes,
sha256 e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7 (CC BY-SA 3.0 and GFDL). Rule:

1. to 3. FlashProve's steps 1 to 3 (scales/make_calibration_set.py:231-238) with TinyLlama's tokenizer.json:
   join all 36,718 rows with two line feeds, encode once with add_special_tokens true (n = 2,874,559 ids), seed
   random.Random(0), draw 128 starts with randint(0, n - 2049). The 128 rows must give DRAWS_SHA256
   8cd63f617d474124a50ee052eae2fd0be603ea45f191ec8ccb4f443f676fb58c (verified on the box).
4. Concatenate the first 48 drawn rows in draw order and remove every id 1 and 2: K (98,304 ids). No decoding.
5. Window w for w = 0 .. 39 is id 1 followed by K[1023 w] .. K[1023 w + 1022].
6. Fit set: windows 0 .. 31 as npy version 1.0, shape (32, 1024), dtype `<i4`, C order, 131,200 bytes, sha256
   9b2b8fc6b98904de5c905bcd7eecfd2f7895ddacef28d249ca44e3b594fa2f31. Held-out set: windows 32 .. 39, shape
   (8, 1024), sha256 5c9da767e78fdb9b31729a0031aa7c07b9c053ceb5ed976c6542bdf31e83bfb8. Both stay out of git
   (`/artifacts/`).

Checked with Python 3.13.5, numpy 2.5.4, pyarrow 26.0.1, tokenizers 0.22.2. The fit set is used for the scale fit
only; held-out windows and the prompt set are used for range validation (P4), not for fitting.

## I5. Converter boundary (W1)

- `iter_tensors(safetensors_path, manifest)` yields (name, float32 array) one tensor at a time, after checking the
  header sha256 and names, shapes and dtypes against docs/tinyllama/checkpoint_manifest.json.
- `permute_qk(w, n_heads)` returns the row permuted copy (contract 4.4).
- `fold_and_quantize(w, gains, c_bits)` returns (int16 array, e, mx) by contract 4.5.
- `calibrate(float_model, windows)` returns per site maxima; `fit_scales(maxima, exponents, e_res)` returns KL words
  and params by contract 5.2; `range_report(artifact, windows)` returns failures per condition.
- Command lines: `python -m tinyllama.convert --source DIR --calibration FIT.npy --out artifacts/TL-INT-1/`
  prints the artifact id.

## I6. Tests and evaluation boundary (W3)

- W3 reimplements the contract arithmetic independently in tests/tinyllama (not importing tinyllama/reference)
  and compares exactly: rq, clamp, isqrt, norm, softmax division, gate key, argmax, tables and digests.
- Malformed artifacts: one test per code of I2.4, made by mutating a small synthetic artifact.
- Quality (float vs integer, not bit equality): top-1 agreement and logit KL on the prompt set and held-out windows,
  perplexity on the held-out windows, range failure counts. Float baseline: HF transformers float32 on CPU.

## I7. Reference primitives (W2 exports, W3 conformance test reads)

tinyllama.reference must export the primitives below at package level under exactly these names, with these
positional arguments in this order. All integers are Python int (unbounded); lists are Python lists of int.
Source: tests/tinyllama/arithmetic/test_reference_conformance.py (names and calling conventions) and
tests/tinyllama/arithmetic/ref_independent.py (definitions). The test also accepts the alias in brackets, but W2
should use the first name.

| name [alias] | arguments in order | returns | vectors file, case key |
|---|---|---|---|
| rq [rounded_quotient] | a: int, k: int | int, (a + 2^(k-1)) floor-divided by 2^k | rq.json, cases |
| clamp | a: int, bound: int | int, a limited to -bound .. bound | clamp.json, cases |
| isqrt | a: int | int, floor of the square root | isqrt.json, cases |
| floor_div [floor_quotient] | a: int, b: int | int, a // b | floor_div.json, cases |
| low8 [balanced_low_byte] | a: int | int in -128 .. 127, ((a + 128) mod 256) - 128 | low8.json, cases |
| softmax_div [softmax_divide] | acc: int, z: int | int, (2 acc + z) // (2 z) | softmax_div.json, cases |
| exp_lookup [exp_index] | u: int, exp_table: list of int | int, exp_table[u] if u < 2307, else 0 | softmax_exp_index.json, lookup_cases |
| argmax [first_argmax] | values: list of int | int, index of the first maximum | argmax.json, cases |
| gate_key | gw: int | dict with int keys g0, gh, gc, key (a 4-tuple (g0, gh, gc, key) is also accepted) | gate_key.json, cases |
| norm [rms_norm] | x: list of int, m: int, k: int, eps: int | dict with vx, sigma, rc, w, nr (nr: list of int, compared to expected_nr) | norm.json, cases |
| pad [pad_window] | ids: list of int | list of int of length 1024, right padded with id 0; ValueError if more than 1024 ids | padding.json, cases |

Conformance test note: once tinyllama.reference exists, the conformance test must not skip. Today it skips the
whole module when the import fails, skips a primitive when none of its names is found, and skips when no calling
convention fits; a W2 primitive with a different name or argument order would therefore skip silently. The W3
follow-up in work/TASKS.md makes these cases fail.
