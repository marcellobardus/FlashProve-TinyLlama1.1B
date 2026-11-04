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

## Blockers

- Proof: verifier standard form requires NE > 0 (`routed-experts`); dense TinyLlama cannot reach `YES` without an agreed change. D1 resolved as option a: proof compatibility deferred (O1).
- Compute: heavy work on the RunPod cpu3g-8-32 pod (D7). No pod created yet.

## Remote compute

RunPod cpu3g-8-32 approved by Marcello (D7). No resources created by this step.
