# Changelog

## 1.2.0 — 2026-10-26

Faster prefill and multi-user decode on the same cards, weights and settings. TP4, MiaAI-Lab protocol, structured /
code / prose (median of 5 passes on one boot, as for 1.1.0): 1 user 355.7 / 258.4 / 183.6 tok/s (1.1.0: 336.0 /
246.8 / 180.0), 4 users 626.4 / 387.8 / 348.7 (602.4 / 367.5 / 307.8), 8 users 867.6 / 544.0 / 440.5 (813.1 / 504.8 /
385.1). Cold prefill 2,416 tok/s (1.1.0: 1,764, +37 %). KV pool 1,395,731 tokens at 262,144 context (1.1.0:
1,526,955; the extra decode graph sizes for 3-draft steps). PP4 runs the 1.1.0 code paths (one pass: 1 user
117.5 / 90.5 / 63.2, 8 users 340.5 / 213.9 / 163.1, cold prefill 2,311; 1.1.0: 118.3 / 92.2 / 63.8, 340.6 / 215.4 /
160.5, 2,457). dqeval NLL 3.50001 vs 3.48019 (+0.57 %, 1.7 σ; 198 passages lower, 213 higher). HumanEval 154/164, GSM8K
1,275/1,319 (1.1.0: 156, 1,273; the development build: 156, 1,276). 20-minute 8-stream soak: 1,018 requests, 0 errors,
334.8 tok/s (1.1.0: 857, 281.0).

The new paths are NLL-equivalent, not bit-identical: the prefill slices change dense GEMM tiling, the owner sum
rounds once after an fp32 sum, and a narrower verify step is a different batch shape. This model amplifies rounding
strongly (a 1-ulp bf16 change at layer 0 moves short-context prompt logprobs by up to several nats, as much as a
different batch shape), so every change was checked with paired NLL over many passages, not token by token.

### Prefill (TP4)

- Each prefill layer's work after the attention core (`wo_a`, `wo_b`, mHC, MoE) runs in row slices, and the
  tensor-parallel sums run on a side stream beside the next slice's compute (`dsv41_split_reduce`; the split
  all-reduce design is Morrowmake's). The sums are a copy-engine owner sum (`dsv41_ce_ring`, a port of Morrowmake's
  copy-engine all-reduce): each rank copies shard q into rank q's IPC scratch with `cudaMemcpyAsync`, rank q adds the
  four partials in fp32 in rank order in one kernel and copies the result to its peers, and every flag is a stream
  memory operation, so the compute beside it keeps its SMs. NCCL beside the same compute gained nothing.
- Prefill attention runs per slice too (`dsv41_attn_slices`: per-slice attention metadata through vLLM's micro-batch
  builder path), and layers overlap: a layer returns before its last FFN sums land and the next layer waits per
  slice. Two slices, cut at 1,024 rows.
- Morrowmake's second tensor-core prefill MoE schedule, built for the tensor-parallel expert shapes
  (`dsv41_moe_try`, sources in `patches/tc_moe_try/`): bitwise equal to the 1.1.0 kernel; 1,024 rows 6.86 → 5.26 ms.
- Engram rows of the next two prefill chunks are read into the page cache ahead of use (`dsv41_engram_prefetch`;
  the n-gram hashes are computed on the CPU and checked equal to the GPU kernel's).
- The fused mHC kernel uses prefill-sized tiles from 256 rows (`dsv41_mhc_pcfg`).
- Sparse-attention prefill gathers are packed by each request's real width with upstream's chunk planner
  (`dsv41_prefill_pack`): several short prompts share one gather instead of one per request (8-user time to first
  token −10 %); a single request is bit-identical.

### Decode

- Routed-expert decode MoE on Morrowmake's faster lane-GEMM schedule (`dsv41_moe_v2try`, sources in
  `patches/tc_moe_v2try/`), bitwise equal to the 1.1.0 tensor-core decode kernel and faster than the cooperative
  kernel, so it now serves 1–64 rows: development build 1 user 333.5 / 244.2 / 181.4 → 358.3 / 268.1 / 194.1.
- Per-step verify width (`DSV41_DYNK=accept:3,5`, TP4 default; `patches/dsv41_dynk.py`, `patch_dynk.py`): DSpark
  still drafts its 5-token block, and the scheduler picks per step whether the next step verifies 3 or 5 drafts,
  from each request's recent per-position acceptance and measured step costs. Every request in a step verifies the
  same count, so the step stays a uniform decode and replays the FULL CUDA graph family for its query length (graph
  sizes 16/20/28/32 added). Same-boot A-B-A on the development build, prose: 2 users 232 → 248, 4 users 306/316 →
  349, 8 users 420/419 → 445 tok/s; structured, code and 1 user unchanged; HumanEval and GSM8K equal (2/2 and 6/6
  discordant problems). The load-adaptive verify depth idea is Morrowmake's.

### Not changed

PP4 runs the same image with the 1.1.0 variant list and no verify-width change (`serve.sh` sets both). Prefill
still runs eagerly: PIECEWISE CUDA graphs for prefill cut the 1-user time to first token from 121 to 68 ms, but
failed 2 of 30 batched prefills beside a decode load (see docs/tuning-log.md).

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
