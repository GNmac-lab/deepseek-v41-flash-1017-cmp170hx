# KV cache and context length

DeepSeek-V4.1-Flash keeps a compressed latent per token (shared across layers), an indexer key cache and a small
sliding window, all in the model's own `fp8_ds_mla` format: about 2.2 KB per token across the whole model. The cache
is the same on every card under TP4 (the latent is not split by head).

## Per card (TP4, 64 GiB cards, `GPU_MEMORY_UTILIZATION=0.95`)

| | 262,144 context | 1,048,576 context |
|---|---:|---:|
| Weights and runtime (non-torch) | 54.3–54.5 GiB | 54.3–54.5 GiB |
| Peak activation (2,048-token prefill chunk) + CUDA graphs | 2.5 + 1.2 GiB | ~0.7 GiB more (context-length workspaces) |
| **KV cache** | **2.95 GiB = 1,395,731 tokens** | **2.27 GiB = 1,270,789 tokens** |
| Full-length requests that fit | 5.3 | 1.21 |

1.0.0 had 3.33 GiB = 1,574,996 tokens at 262,144 context and 1,454,546 tokens at 1,048,576; 1.1.0 had 3.23 GiB =
1,526,955 and 2.55 GiB = 1,426,183. 1.2.0 leaves 0.28 GiB less per card for the cache: the CUDA graphs for 3-draft
decode steps (`DSV41_DYNK`) and the prefill slices' buffers.

Two changes in this recipe made room (both bit-identical in output):

- `patch_wo_a_free.py`: the attention output projection `wo_a` is kept only as its FP8 codes (+0.9 GiB per card; the
  bf16 copy the sm_80 path made at load is freed after an exact round-trip check).
- `patch_long_ctx_mem.py`: two workspaces that scale with the context length were sized for large multi-request
  batches. The sparse-attention prefill gathered the whole context of 4 requests at once, and the indexer reserved
  40 × max context of key slots. With 1 request per gather and 4 × max context (running tokens can never exceed the
  KV pool), 1M context fits, and at 262K the pool grew 947K → 1,575K tokens.

## The 1M option

`MAX_MODEL_LEN=1048576` in `.env`. Measured on 1.0.0: needle retrieval passes at 388K (3/3 depths), 776K and
997K tokens (1.1.0: 3/3 at 388K tokens);
1-user decode −2 % and 8-stream throughput −6 % compared with 262K, because decode graphs size the indexer's top-K
for the maximum context; cold prefill slows with context (1,399 tok/s at 170K; a 1M prompt takes ~25 min).

## Not possible here

- Prefill chunks of 1,024 tokens (more KV room) do not start with images enabled: one image needs more than 1,024
  tokens per step. 4,096-token chunks gave ~4 % faster prefill in 1.1.0 but left 848K tokens of KV instead of 1.53M;
  with 1.2.0's prefill slices they gain ~1 % and the cold prefill bench ran out of memory.
- A lower-precision KV cache would change the model's output; `fp8_ds_mla` is the model's native format.
