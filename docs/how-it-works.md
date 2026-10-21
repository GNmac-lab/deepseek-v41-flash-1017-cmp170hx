# How it works

## The stack

| Layer | What | From |
|---|---|---|
| Base | vLLM, official DeepSeek-V4.1 image | vllm-project |
| sm_80 model support | sparse MLA attention on Ampere, PP shadow KV, DSpark drafter (V1), Engram row store read from NVMe | zebgop-ops/dsv41reap-pp, kaka86mm |
| Weights | hybrid checkpoint: LibertAI EXL3 2.9-bpw routed experts (layers 0–39) + the official FP8 release for everything else (attention, shared experts, head, drafter, Engram) | Curtis White's recipe |
| Routed experts | EXL3 kernels: a cooperative decode kernel for small batches, tensor-core kernels for prefill and large batches | exllamav3, vllm-exl3, DeepSeekv41-Flash-EXL3-TC-MoE |
| Parallelism | TP4 with async scheduling and full CUDA graphs for decode; FlashInfer PCIe IPC all-reduce | FlashInfer |
| Speculation | DSpark, 5 drafted tokens per step (the drafter's block size; 6 and 7 do not start) | DeepSeek, zebgop-ops |

A decode step at one user verifies 6 rows (5 drafts + 1). Its time splits roughly into the routed-expert MoE
(~35 %), the two all-reduces per layer (~25 %, at the PCIe 2.0 x16 limit since the ring variant), attention and
the shared experts, and a few hundred small kernels.

## What this recipe changes

Every row was measured on its own (structured decode step time is the cleanest A/B: restart noise is 0.1–0.2 ms)
and checked for output changes with a paired NLL evaluation (411 short passages); "bit-identical" means every
passage's NLL matched exactly.

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

"Sum order" changes the order in which the four ranks' partial sums are added (the same kind of change as
switching all-reduce libraries); the paired evaluation stayed within noise.

## What did not help

See [tuning-log.md](tuning-log.md) for the full list with numbers. In short: a host-memory all-reduce (loses above
~10 KB), an FP8 GEMV for the Marlin layers (ties Marlin once weights come from HBM), drafter top-k truncation and
adaptive verification (acceptance losses cancel the savings), a different EXL3 expert pack (same speed, +1.55 %
NLL), larger prefill chunks (smaller KV pool and slower), folding the inverse RoPE into the `wo_a` kernel (+0.8 ms).
