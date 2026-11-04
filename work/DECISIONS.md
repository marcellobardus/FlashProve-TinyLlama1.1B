# Decisions and open questions

Open items need Marcello or the lead. None decided yet unless marked.

- D1 (open, proof). Verifier standard form requires NE > 0 (`routed-experts`). Options: a) integer model and CPU reference only now, proof compatibility deferred; b) design a documented dense profile with its own soundness argument; c) a trivial expert wrapper (hypothesis, likely changes the function and the cost). Do not relax silently.
- D2 (open). Window length: FlashProve standard form allows T <= 1024; TinyLlama supports 2048. Proposal: fix T = 1024 for the integer model.
- D3 (open). Rotary on every layer: the Lean soundness model pins Maverick's pattern (docs/model.md:110-111). Accept rp=1 everywhere as an extension, with Lean impact recorded.
- D4 (open). Converter input: write a new safetensors reader (proposed) or export a GGUF and reuse existing readers (hypothesis H2).
- D5 (open). Arithmetic widths: keep FlashProve's 13-bit weights and 12-bit activations unchanged (proposed default) unless quality evaluation shows unacceptable loss.
- D6 (open). Padding token: generation_config pad_token_id 0, tokenizer pad `</s>` (2), FlashProve vLLM integration pads with 0 (docs/model.md:285). Pick one and record it.
- D7 (open). Compute for conversion, calibration and tests: not the shared box. RunPod details to be supplied by Marcello.
- D8 (open). Calibration data source and size for TinyLlama chat.
- D9 (decided by brief). No edits to existing FlashProve sources on this branch; new code under tinyllama/ and tests/tinyllama/.
