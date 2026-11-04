# Status

Branch `feature/tinyllama-integer-audit` from main `9e2bdd8f`. Not pushed.

## 2026-11-04 source audit (done)

- Checked: clean tree on main, remote origin = marcellobardus/FlashProve-TinyLlama1.1B, HEAD 9e2bdd8f.
- Added ignore rules (/artifacts/, /work/scratch/, *.safetensors, *.gguf, /hf_cache/, /traces/).
- Pinned checkpoint revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 by HF API; header-only read: 201 BF16 tensors, 1,100,048,384 parameters. Files in docs/tinyllama/CHECKPOINT_MANIFEST.md.
- Gap map: docs/tinyllama/GAP_MAP.md.
- Not run: no build, no tests, no model load (box rule: no heavy work on the shared box).

## Blockers

- Proof: verifier standard form requires NE > 0 (`routed-experts`); dense TinyLlama cannot reach `YES` without an agreed change. Decision D1.
- Compute: conversion, calibration and model-dependent tests need a machine other than the shared box. Decision D7.

## Remote compute (placeholder)

RunPod setup: to be filled in by the lead. No resources created.
