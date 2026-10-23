# Changelog

## 1.1.0 — 2026-10-23

Faster decode on the same cards, weights and settings. TP4, MiaAI-Lab protocol, structured / code / prose:
1 user 336.0 / 246.8 / 180.0 tok/s (median of 5 passes; 1.0.0: 295.5 / 212.2 / 149.8), 8 users 813.1 / 504.8 /
385.1 (1.0.0: 795.8 / 476.6 / 358.8). PP4, 1 user: 118.3 / 92.2 / 63.8 (1.0.0: 110.5 / 81.2 / 52.0). Cold prefill 1,764 tok/s (1.0.0: 1,759); KV pool 1,526,955 tokens at 262,144 context (1.0.0: 1,574,996).
dqeval NLL 3.48019 vs 3.50409 (−0.68 %; 217 passages lower, 194 higher). HumanEval 156/164, GSM8K 1,273/1,319.

Three changes reorder fp32 sums: the MXFP8 kernels with their config table, the decode `wo_a` kernel, and the
sparse-MLA decode launch config. Every other change is bit-identical. Step times below are 1-user structured decode
steps on the development build, before → after each change (1.0.0: 20.1 ms).

### Routed-expert MoE (coop decode kernel)

- Fewer instructions in the cooperative MoE decode loops (`patch_coop_ptr.py`, `patch_coop_rowclamp.py`,
  `patch_coop_noclamp.py`, `patch_coop_laneclamp.py`; `patch_coop_widths.py` is inert without the plugin hand-off):
  7.33 → 6.39 instructions per weight in the gate/up and down loops; 19.62 → 19.06 ms. Bit-identical.

### Dense layers

- MXFP8 linears (attention, shared experts, indexer, drafter) on ONE weight copy in a fragment-native layout
  (`patch_fp8_gemv_cuda.py`, `dsv41_fp8w.py`, CUDA extension `dsv41_fp8w/`): a GEMV for M <= 8 tokens, a kernel with
  shared-memory-staged activations for M <= 64, and exact dequant + cuBLAS (fp32 reductions) above. Weights over
  32 MB stay on Marlin; weights up to 4 MB keep Marlin for M > 8. Exact e4m3 decode, fp32 sums in a fixed order
  (deterministic). Structured step 19.06 → 18.37 ms; KV pool unchanged.
- Decode `wo_a` (M <= 8) as one CUDA kernel on the exact FP8 codes, with the inverse GPT-J RoPE fused into its
  activation load (`patch_wo_a_cuda.py`, CUDA extension `dsv41_woa/`). It replaces the inverse-RoPE kernel, the
  Triton FP8 `wo_a` and its split-K reduce. Structured step 18.37 → 18.07 ms.
- The sm_80 split-K sparse-MLA decode kernel runs 16 KV tokens per step on 8 warps instead of 32 on 4
  (`patch_attn_cfg.py`). The M <= 64 MXFP8 kernel gets a three-bucket config table for every served shape
  (`patch_mid_table.py`). Structured step 17.95 → 17.64 ms.

### Launch order and per-step overhead

- The fused-MoE runner launches the router GEMM before it puts the shared experts on the aux stream
  (`patch_gate_first.py`): the router no longer waits behind the shared-expert GEMV. 18.07 → 17.95 ms.
- Greedy verification uses the vocab-parallel argmax ids (`patch_greedy_ids.py`): the rejection sampler no longer
  re-reduces the one-hot logits (two full-vocab argmax passes per step).
- The sharded DSpark Markov sampling loop runs as one CUDA graph per batch size (`patch_dspark_sample_graph.py`),
  with the bias GEMV, bf16 add and shard max fused into two Triton kernels (`dsv41_markov.py`,
  `patch_dspark_markov_fused.py`).
- A draft-only FP8 copy of the drafter head shard (`patch_v38.py`, `DSV41_DRAFT_HEAD_FP8=1`, now set by
  `serve.sh`); the target still scores every draft with its bf16 head. With the two rows above: 20.1 → 19.62 ms.
- Runtime variants (`patches/dsv41_variants/`): `dsv41_variants.py` applies small modules in each worker before
  CUDA graph capture; `DSV41_VARIANTS` lists them. Each one self-tests when it loads.
  - `dsv41_topk_meta`: the decode ragged top-k metadata in one launch instead of five;
  - `dsv41_h2d_lean`: no host-to-device copy for index tensors and block-table rows that repeat the previous step;
  - `dsv41_ragged_lean`: `build_ragged_indices_from_dense` in one launch, written straight into the SWA graph
    buffers, and `is_valid_token` in one launch.
  Structured step 17.64 → 17.43 ms.

### Confidence-gated verification (`dsv41_gate`, in the default `DSV41_VARIANTS`)

DSpark's confidence head decides how many of the 5 drafts the target verifies: drafts are kept while the running
product of their confidences is at least `DSV41_GATE_TAU` (default 0.10; 0 = off). A skipped draft gets the router's
expert-id sentinel, so the MoE never loads its experts (~0.8–1.0 ms of a prose step per skipped draft), and the
first skipped draft is force-rejected. Accepted and correction tokens come only from fully computed positions, so
the output does not change. Graph shapes do not change either.

Only steps in which every request verifies 1 + 5 tokens, run as a FULL CUDA graph with greedy sampling, are gated.
Any other step runs with nothing skipped: prefill chunks, draft-less steps, batches with a sampled (temperature > 0)
request, and PP4 (PIECEWISE graphs). 1-user prose 174.7 → 179.9 tok/s (alternating blocks of the same prompts);
MiaAI prose 8 users 366 → 383, code 4 users 355 → 373.

### Settings

- `.env` can set `DSV41_GATE_TAU` (default 0.10; 0 = verify every draft) and `DSV41_VARIANTS` (default: all four;
  empty = none); `serve.sh` passes them to the engine only when they are set.
- `serve.sh` sets `DSV41_DRAFT_HEAD_FP8=1`.

### Validation

- `DSV41_FP8_GEMV_ALL=1 DSV41_WO_A_ALL=1` route every prompt token through the decode-only kernels (8 rows at a
  time), so a prefill-scored NLL evaluation covers them. dqeval NLL −0.47 % vs 1.0.0 (1.3 σ, 217 of 411 passages
  lower).

## 1.0.0 — 2026-10-21

First release: DeepSeek-V4.1-Flash-1017 (hybrid EXL3 2.9-bpw experts + official FP8) on 4× CMP 170HX.

- TP4 (default) and PP4 layouts; 262,144-token context by default, up to 1,048,576.
- Engine image built from pinned upstream sources (`docker/Dockerfile`) plus the patch scripts in `patches/`, each measured on its own.
- TP4, MiaAI-Lab protocol: 1 user 295.5 / 212.2 / 149.8 tok/s (structured / code / prose), 8 users 795.8 / 476.6 /
  358.8; cold prefill 1,759 tok/s; KV pool 1,574,996 tokens. HumanEval 156/164, GSM8K 1,275/1,319.
