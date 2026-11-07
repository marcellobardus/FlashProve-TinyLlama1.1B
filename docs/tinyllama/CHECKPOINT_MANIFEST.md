# TinyLlama checkpoint manifest

Machine-readable form: `checkpoint_manifest.json` (full tensor list with shapes and dtypes).
Recorded 2026-11-04 by header-only inspection. The weights were not downloaded or loaded.

## Pins

- FlashProve base commit: `9e2bdd8ff5bf9c62f18263d7dac189b0400b260c` (main, "Initial public release (FlashProve 0.16.0-rc.26)").
- Checkpoint: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, HF revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`
  (field `sha` of `https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- Every file below was fetched from `https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/<revision>/<file>`.

## Weights file

| item | value |
|---|---|
| file | `model.safetensors` (single shard) |
| size | 2,200,119,864 bytes (`x-linked-size`) |
| sha256 | `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933` (LFS `x-linked-etag`; not recomputed locally) |
| header | 23,088 bytes, sha256 of header JSON `de4413e191122aeb9a7fd93b43f92b0a48039f0da7569646bac2ed4ceb2c428c`, metadata `{"format": "pt"}` |
| tensors | 201, all BF16 |
| parameters | 1,100,048,384 (sum of shape products) |

Tensor inventory (N = 0 .. 21):

| tensor | shape | count |
|---|---|---|
| model.embed_tokens.weight | [32000, 2048] | 1 |
| model.layers.N.input_layernorm.weight | [2048] | 22 |
| model.layers.N.self_attn.q_proj.weight | [2048, 2048] | 22 |
| model.layers.N.self_attn.k_proj.weight | [256, 2048] | 22 |
| model.layers.N.self_attn.v_proj.weight | [256, 2048] | 22 |
| model.layers.N.self_attn.o_proj.weight | [2048, 2048] | 22 |
| model.layers.N.post_attention_layernorm.weight | [2048] | 22 |
| model.layers.N.mlp.gate_proj.weight | [5632, 2048] | 22 |
| model.layers.N.mlp.up_proj.weight | [5632, 2048] | 22 |
| model.layers.N.mlp.down_proj.weight | [2048, 5632] | 22 |
| model.norm.weight | [2048] | 1 |
| lm_head.weight | [32000, 2048] | 1 |

No bias tensors, no rotary inv_freq tensors. `lm_head.weight` is stored separately, consistent with `tie_word_embeddings: false`.

## Architecture (config.json)

LlamaForCausalLM, 22 layers, hidden 2048, 32 query heads, 4 KV heads (GQA group 8), head dim 64 (2048/32; derived),
intermediate 5632, vocab 32000, hidden_act silu, rms_norm_eps 1e-05, rope_theta 10000.0, rope_scaling null,
max_position_embeddings 2048, attention_bias false, tie_word_embeddings false, torch_dtype bfloat16, bos 1, eos 2.

## Tokenizer and chat template

| file | sha256 |
|---|---|
| config.json | 486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9 |
| generation_config.json | 18046d04f5bd8b4998095ecabdd17a1bf0053d9acdccead4a05be4a3575f3c5c |
| special_tokens_map.json | 82d96d7a9e6ced037f12394b7ea6a5b02e6ca87e0d11edaa8d60d9be857ce7db |
| tokenizer.json | bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e |
| tokenizer.model | 9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347 |
| tokenizer_config.json | 7b41ba7d0eb91e77914ca3dafde559ea3e19878769b7e68409e89bed5222e77a |

tokenizer_config: LlamaTokenizer, legacy false, bos `<s>`, eos `</s>`, unk `<unk>`, pad `</s>`, padding_side right,
model_max_length 2048, add_generation_prompt supported.
Note: generation_config.json sets `pad_token_id 0` while the tokenizer's pad token is `</s>` (id 2). See work/DECISIONS.md D6.

Chat template (`chat_template` in tokenizer_config.json), sha256 of its UTF-8 bytes
`66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362`:

```
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>\n' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>\n'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}
```

## Reproduce

```
curl -s https://huggingface.co/api/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0   # field sha
R=https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/fe8a4ea1ffedaf415f4da2f062534de366a451e6
curl -sL -r 0-7 $R/model.safetensors        # u64 LE header length (23088)
curl -sL -r 8-23095 $R/model.safetensors    # header JSON
```
