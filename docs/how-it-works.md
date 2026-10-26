# How it works

## The stack

| Layer | What | From |
|---|---|---|
| Base | vLLM, official DeepSeek-V4.1 image | vllm-project |
| sm_80 model support | sparse MLA attention on Ampere, PP shadow KV, DSpark drafter (V1), Engram row store read from NVMe | zebgop-ops/dsv41reap-pp, kaka86mm |
| Weights | hybrid checkpoint: LibertAI EXL3 2.9-bpw routed experts (layers 0–39) + the official FP8 release for everything else (attention, shared experts, head, drafter, Engram) | Curtis White's recipe |
| Routed experts | EXL3 kernels: tensor-core kernels for decode (1–64 rows, Morrowmake's lane-GEMM schedule) and prefill (Morrowmake's second schedule); the cooperative kernel for PP4 decode | exllamav3, vllm-exl3, DeepSeekv41-Flash-EXL3-TC-MoE, Morrowmake |
| Dense FP8 layers | this recipe's MXFP8 CUDA kernels on one weight copy for up to 64 tokens; Marlin or exact dequant + cuBLAS above | this recipe, Marlin |
| Parallelism | TP4 with async scheduling and full CUDA graphs for decode; FlashInfer PCIe IPC all-reduce | FlashInfer |
| Speculation | DSpark, 5 drafted tokens per step (the drafter's block size; 6 and 7 do not start); per step 3 or 5 of them are verified (TP4), and its confidence head decides which verified drafts run their experts | DeepSeek, zebgop-ops, Morrowmake (verify width) |

A decode step at one user verifies 6 rows (5 drafts + 1). In 1.1.0 a 1-user prose step with all 6 rows verified
takes ~18.4 ms: 15.9 ms for the target's forward graph, 1.5 ms for the DSpark drafter (vLLM runs drafters in
PIECEWISE graphs), 0.36 ms for the LM head and verification, and ~0.5 ms for input preparation and attention
metadata. The GPUs are never waiting for the CPU. Inside the forward graph, the routed-expert MoE and the two
all-reduces per layer (at the PCIe 2.0 x16 limit since the ring variant) are the largest parts.

The routed experts of one token cost ~4 ms per step (~96 µs per layer: six experts are too few blocks to fill the
card). Each further token in the step costs another 0.6–1.0 ms. This is why skipping drafts that will be rejected
pays (see "Confidence-gated verification" below).

## What this recipe changes

Every row was measured on its own (structured decode step time is the cleanest A/B: restart noise is 0.1–0.2 ms)
and checked for output changes with a paired NLL evaluation (411 short passages); "bit-identical" means every
passage's NLL matched exactly.

### 1.0.0

| Patch | Change | Effect | Output |
|---|---|---|---|
| `patch_attn_splitk.py` | split-K sparse-MLA decode kernels on sm_80 (were a 6-program kernel) | step −6 % short, −14 % at 86K context | same maths |
| `patch_row_store_overlap.py` | Engram rows: page-cache hits read without blocking, all misses in flight at once | prose 92 → 106 tok/s (warm) | byte-identical |
| `patch_engram_async.py` | Engram lookups on a side stream, overlapped with layer 0 | step −2–4 % | identical |
| `patch_coop_branchfree.py`, `patch_coop_down_wide16.py`, `patch_coop_down_epilogue.py` | cooperative MoE kernel: branch-free tile, 512-column down tiles, parallel chunk epilogue | structured 202 → 210 | bit-identical |
| `vllm-exl3/patch_exl3_coop_fastpath.py` | lean coop call (skips ~11 routing kernels per layer) | with the above | identical |
| `patch_dspark_shard.py` | vocab-sharded exact draft argmax | structured 213 | identical drafts |
| `patch_router_mm.py` | Triton small-batch router GEMM | step −0.2 ms | noise-level |
| `patch_greedy_argmax.py` | greedy decode without the full-logits all-gather | step −0.1–0.2 ms | identical tokens |
| `patch_share_t2r.py` | one token→request map for all attention groups | step −0.2–0.3 ms | bit-identical |
| `vllm-exl3/patch_exl3_coop_prep.py` | four casts/fills before each coop call → one kernel | step −0.3–0.4 ms | bit-identical |
| `patch_shared_rebalance.py` | shared-expert blocks onto the lighter routed ranks | step −0.1–0.2 ms | noise-level |
| `patch_mhc_fused.py` | token-blocked Triton kernel for mHC post + pre; sinkhorn forked beside the next all-reduce | structured 227 | residual bit-identical |
| `install_fipcie.py` | FlashInfer PCIe IPC all-reduce (0.7.0.post1) in the image's FlashInfer 0.6.18 | 61 KB all-reduce 57.6 → 42.6 µs; structured 235 | sum order |
| `deepinfer_sm80_attn.patch` | Deepinfer-SM80 sparse-attention decode tile | structured 245 | bit-identical |
| `deepinfer_sm80_rocm.patch` | Deepinfer-SM80 top-K/SWA combiner + decode top-K reuse | prefill +5–7 %, TTFT 0.18 → 0.13 s | bit-identical |
| `patch_coop_mma_swap.py` | coop tile feeds decoded weights as the MMA A operand: one MMA per tile, no padded rows | gate/up −12.7 %, down −12 %; structured 262, prose 138 | bit-identical |
| `patch_v32_wo_a.py`, `vllm-exl3/patch_exl3_ids_cast.py` | attention `wo_a` as an FP8 grouped Triton GEMM (exact FP8 codes recovered from the bf16 weight) | 26.0 → 14.8 µs/layer; prose 145 | noise-level |
| `patch_true_ring.py` | cut-through neighbour-ring all-reduce variant (no barriers, per-pack arrival signal) | 6-token all-reduce 42.6 → 18.2 µs; step −10.6 %; prose 153 | sum order |
| `patch_wo_a_free.py` | `wo_a` kept only as FP8 codes; prefill rebuilds the exact bf16 weight in one scratch buffer | +0.9 GiB per card for the KV cache | bit-identical |
| `patch_long_ctx_mem.py` | context-length workspaces sized for this deployment | KV pool 947K → 1,575K tokens; 1M context fits | bit-identical |

### 1.1.0

Structured step times are for 1 user on the development build, in the order the changes went in (1.0.0: 20.1 ms).

| Patch | Change | Effect | Output |
|---|---|---|---|
| `patch_v38.py` (`DSV41_DRAFT_HEAD_FP8=1`) | draft-only FP8 copy of the drafter's head shard | step −0.1–0.2 ms | identical (the target verifies with its bf16 head) |
| `patch_greedy_ids.py` | greedy verification from the vocab-parallel argmax ids, without two full-vocab argmax passes | with the next two rows | identical |
| `patch_dspark_sample_graph.py` | the sharded DSpark Markov sampling loop (~70 eager kernels) as one CUDA graph per batch size | | identical drafts |
| `dsv41_markov.py`, `patch_dspark_markov_fused.py` | Markov bias GEMV + bf16 add + shard max in two Triton kernels | 16 vs 44 µs per draft position; 20.1 → 19.62 ms for the three rows | bit-identical |
| `patch_coop_ptr.py`, `patch_coop_rowclamp.py`, `patch_coop_noclamp.py`, `patch_coop_laneclamp.py` | coop MoE loops: pointer-walk loads, padding rows load a real row instead of a predicated zero-fill, no prefetch clamp in the steady state, 3-bpw lanes 24–31 reuse lane 23's word | 7.33 → 6.39 instructions per weight; 19.62 → 19.06 ms | bit-identical |
| `patch_fp8_gemv_cuda.py`, `dsv41_fp8w.py`, `dsv41_fp8w/` | MXFP8 linears (attention, shared experts, indexer, drafter) on one weight copy in a fragment-native layout: a GEMV for M ≤ 8, a kernel with shared-memory-staged activations for M ≤ 64, exact dequant + cuBLAS above; weights over 32 MB stay on Marlin | 19.06 → 18.37 ms | sum order (NLL +0.24 %, 0.35 σ) |
| `patch_wo_a_cuda.py`, `dsv41_woa/` | decode `wo_a` (M ≤ 8) as one CUDA kernel on the exact FP8 codes, inverse RoPE fused into its activation load | 18.37 → 18.07 ms | sum order |
| `patch_gate_first.py` | the router GEMM is launched before the shared experts go to the aux stream | 18.07 → 17.95 ms | bit-identical |
| `patch_attn_cfg.py`, `patch_mid_table.py` | sm_80 sparse-MLA decode partial kernel: 16 KV tokens per step on 8 warps (was 32 on 4); a config table for every served shape of the M ≤ 64 MXFP8 kernel | 17.95 → 17.64 ms | sum order |
| `dsv41_variants/` `dsv41_topk_meta`, `dsv41_h2d_lean`, `dsv41_ragged_lean` | decode top-k and ragged-index metadata in one launch each; no host-to-device copy for index tensors and block-table rows that repeat the previous step | 17.64 → 17.43 ms | bit-identical |
| `dsv41_variants/` `dsv41_gate` | confidence-gated verification (below) | prose 1 user +3 %, 8 users +5 % | identical tokens |

"Sum order" changes the order in which partial sums are added: the four ranks' partial sums for the all-reduce,
and fp32 partial sums inside a kernel for the 1.1.0 rows. The paired evaluation stayed within noise for each one.
The validation mode `DSV41_FP8_GEMV_ALL=1 DSV41_WO_A_ALL=1` sends every prompt token through the decode-only kernels
(8 rows at a time), so the prefill-scored NLL evaluation covers them too: −0.47 % vs 1.0.0 (1.3 σ).

### 1.2.0

Prefill numbers are the development build's warm 16K-token prefill and the cold bench of `bench/prefill.py`; decode
numbers are MiaAI passes on one boot.

| Patch | Change | Effect | Output |
|---|---|---|---|
| `dsv41_variants/` `dsv41_split_reduce`, `dsv41_ce_ring` | prefill layer after the attention core in row slices; the TP sums on a side stream as a copy-engine owner sum | cold 1,784 → ~1,960 | fp32 sum, rounded once |
| `dsv41_moe_try` (`patches/tc_moe_try/`) | Morrowmake's second prefill MoE schedule at the TP expert shapes | 1,024 rows 6.86 → 5.26 ms; warm 1,956 → 2,250 (with fused mHC in the slices) | bitwise equal |
| `dsv41_engram_prefetch` | Engram rows of the next two chunks read ahead into the page cache | fresh 12K text 2,053 → 2,187 | bit-identical |
| `dsv41_mhc_pcfg` | prefill-sized tiles for the fused mHC kernel from 256 rows | warm 2,333 → 2,375 | sum order |
| `dsv41_prefill_pack` | sparse-attention prefill gathers packed by real width | 8-user time to first token −10 % | single request bit-identical |
| `dsv41_attn_slices` + the pipelined `dsv41_split_reduce` | prefill attention per slice (per-slice metadata); a layer returns before its last FFN sums land | warm ~2,630, cold 2,167 → 2,446 | NLL +0.10 % (48 × 2,000 tokens) |
| `dsv41_moe_v2try` (`patches/tc_moe_v2try/`) | Morrowmake's lane-GEMM decode MoE at the TP expert shapes, for 1–64 rows | 1 user 333.5 / 244.2 / 181.4 → 358.3 / 268.1 / 194.1 | bitwise equal to the 1.1.0 tensor-core kernel |
| `dsv41_dynk.py`, `patch_dynk.py` (`DSV41_DYNK=accept:3,5`) | per-step verify width, 3 or 5 drafts, from per-request acceptance and step costs | prose 2 / 4 / 8 users 232 → 248, 306 → 349, 420 → 445 | same accepted tokens; batch shape differs |

Prefill layers without an indexer reuse the top-k rows the last indexer layer wrote. Per-slice attention keeps each
slice's rows and restores them before those layers. Without that, a slice read the next slice's selections (future
tokens): on a random 1,000-token sequence repeated once, the first copy's NLL fell 12 % below its causal 13.0.

## Confidence-gated verification

DSpark proposes 5 drafts per step, but in prose most of the later ones are rejected (acceptance by position:
82 / 58 / 40 / 29 / 20 %). The drafter has a trained confidence head. `dsv41_gate` keeps drafts while the running
product of their confidences stays at or above `DSV41_GATE_TAU` (default 0.10). For each draft after that:

- the router gives the draft's token the expert-id sentinel, so the MoE kernels skip it and never load its experts;
- the rejection sampler replaces the draft with an id that can never match, so it is rejected.

Accepted tokens and the correction token always come from positions whose experts did run, and with causal
attention no earlier position depends on a skipped one. The output is therefore the token sequence the ungated
model gives. Graph shapes and async scheduling do not change; skipped drafts behave like rejected ones.

Only steps in which every request verifies exactly 1 + 5 tokens, run as a FULL CUDA graph with greedy sampling, are
gated (a 3-draft step from the verify width is not gated). Prefill chunks, draft-less steps, PP4 (PIECEWISE graphs) and
any batch with a sampled (temperature > 0) request run with nothing skipped. Sampled requests therefore get no gain from the gate.

## What did not help

See [tuning-log.md](tuning-log.md) for the full list with numbers. In short: a host-memory all-reduce (loses above
~10 KB), drafter top-k truncation, an earlier adaptive verification that zeroed the routing weights of unlikely drafts
(step −0.7 ms, no gain in MiaAI), a different EXL3 expert pack (same speed, +1.55 % NLL), larger prefill
chunks (smaller KV pool), coop split-K, a persistent coop kernel, the whole drafter as one CUDA graph, skipping
attention for gated drafts, and vLLM's dual-batch overlap for prefill (it needs data parallelism).
