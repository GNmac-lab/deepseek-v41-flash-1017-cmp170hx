# Tuning log (condensed)

Starting point: the TP4 hybrid recipe on this hardware, 1-user MiaAI prose 85.9 tok/s. Each entry was measured on
its own boot; structured decode step time is the A/B metric (restart noise 0.1–0.2 ms), MiaAI 1-user prose varies
±2 % run to run.

## Kept (in order)

| Build | Change | Structured step | MiaAI 1 user (structured / code / prose) |
|---|---|---:|---|
| start | TP4 hybrid recipe | ~30–35 ms | 183.0 / 139.3 / 85.9 |
| | sm_80 split-K sparse decode, NUMA binding | | 195.9 / 140.2 / 91.9 |
| | Engram row-store overlap, Engram side stream | | prose 103–106 |
| | coop MoE: branch-free tile, wide down tiles, parallel epilogue, lean call | | structured 210 |
| | exact sharded draft argmax, router GEMM, greedy argmax, shared token map, fused coop prep | 28.2 ms | structured 220 |
| | shared-expert rebalance, fused mHC | 26.24 ms | structured 227 |
| | FlashInfer PCIe IPC all-reduce | 25.36 ms | structured 235 |
| | Deepinfer-SM80 attention + top-K combiner, shared h=1 | 23.96 ms | 247.0 / 179.5 / 126.3 |
| | coop MMA operand swap | 22.75 ms | 258.8 / 194.9 / 138.4 |
| | FP8 `wo_a` | 22.36 ms | 265.3 / 192.6 / 144.8 |
| | cut-through ring all-reduce | 20.06 ms | 293.2 / 210.4 / 153.4 |
| | `wo_a` FP8-only, context workspaces (memory only) | 20.25 ms | KV 947K → 1,575K tokens |
| **1.0.0** | | | 295.5 / 212.2 / 149.8 |
| | draft-only FP8 drafter head | ~20.1 ms | |
| | greedy verification from argmax ids, DSpark sampling loop as a CUDA graph, fused Markov step | 19.62 ms | 301.9 / 213.8 / 153.5 |
| | coop loop trims (pointer walk, row clamp, no prefetch clamp, lane clamp) | 19.06 ms | 309.9 / 220.4 / 169.2 |
| | MXFP8 linears on one weight copy (CUDA GEMV / mid kernel) | 18.37 ms | 321 / 240 / 170 |
| | decode `wo_a` CUDA kernel with fused inverse RoPE | 18.07 ms | 327 / 233 / 174 |
| | router GEMM launched before the shared experts | 17.95 ms | 330 / 239 / 166–174 |
| | sparse-MLA decode launch (BLOCK_K 16, 8 warps), mid-kernel config table | 17.64 ms | 335 / 239 / 169 |
| | runtime variants: one-launch decode metadata, repeated-H2D skips | 17.43 ms | 339 / 245 / 169 |
| | confidence-gated verification | (prose step depends on the drafts kept) | 338 / 246 / 182 |

## Tried and dropped

| Idea | Result |
|---|---|
| Host-memory (shared-memory) all-reduce | 21 µs at 1 token but 164 µs at 6 tokens; prose 92 → 64 |
| NCCL tree / no-P2P / vLLM custom all-reduce | slower than the NCCL P2P ring, then FlashInfer, then the ring variant |
| Sinkhorn on a side stream beside full-grid GEMMs | 0 gain: stretched GEMM tails (+0.39 ms) |
| A Triton FP8 GEMV for the Marlin (FP8) layers | ties Marlin once weights stream from HBM (offline benches must rotate copies larger than L2); 1.1.0's CUDA kernels in their own fragment layout do beat Marlin |
| Router GEMM retune | already 8.35 µs offline; the in-step 15–19 µs is contention with the shared-expert stream |
| DSpark top-k truncation, a CUDA-graph draft sampler, a full-graph drafter | acceptance loss, ≤ 0.2 ms, GPUs already 96–97 % busy (1.1.0 captures the sampling loop with fused Markov kernels: kept) |
| DSpark K = 6 or 7 | does not start: K must be a multiple of the drafter's block size (5) |
| Adaptive verification with zeroed routing weights for unlikely drafts | step −0.7 ms but acceptance loss and overhead cancel it; MiaAI unchanged (1.1.0's gate gives skipped drafts no experts at all and forces their rejection: kept) |
| Shared expert after the router | slower (258.5 vs 261.8 structured); in 1.1.0 only the launch order changes, and both still run in parallel |
| Coop k-split 2, fp32-accumulate MMA, simpler activation remat | +0.6 % or slower / not bit-identical |
| Fused top-K ragged prep, router reduce folded into top-k | no measurable gain |
| Inverse RoPE folded into the Triton `wo_a` kernel | +0.8 ms (extra gather loads break the pipelining); 1.1.0's CUDA `wo_a` kernel fuses it and is faster |
| Drafter argmax over the PCIe all-reduce | +0.23 ms (the drafter runs eagerly) |
| Mia-AiLab EAQ-v1 2.9-bpw experts instead of LibertAI | same speed, +1.55 % NLL in our hybrid (2.7 σ) |
| Prefill chunk 4,096 / 1,024 | less KV and slower / does not start with images |
| Coop MoE kernel for up to 512 slots (8-user batches) | same 8-user throughput as the tensor-core path |
| More registers for the coop MoE kernel (1 block per SM instead of 2) | 10–19 % slower; occupancy beats the rematerialized constants |
| CUDA stream / graph-node priority for the MoE critical path | nodes inherit the priority, but a resident small kernel is not faster beside a big one |
| Fusing the Marlin FP8 GEMVs | they already reach 85–93 % of the card's read bandwidth for their size (~2.5 µs fixed cost per kernel) |
| FP8 vocab head for the target (exactness-checked top-k) | the error bound needed for an exact argmax costs most of the bytes saved |
| Clock offsets, raising the 180 W cap | not used (clock caps at 1,470 MHz; power cap kept for thermals) |

## Tried and dropped for 1.1.0

Structured / prose step times against the build of the time.

| Idea | Result |
|---|---|
| Balanced per-expert split of the routed experts across ranks | correct, but speed-neutral (20.17 vs 20.16 ms) |
| Coop kernel geometry: narrow gate/up tiles, split-K per stage, both stages in one persistent kernel | 19.70–20.21 vs 19.62 ms; the offline harness predicted gains for all of them |
| Coop split-K made real (the scratch buffers had silently clamped it to 1) | ×2 20.21, ×4 21.17 vs 20.1 ms |
| Coop byte sums on integer tensor cores (IMMA), fp32-accumulate coop MMA, a 2× unrolled loop | 19.34 vs 19.06, 18.54 vs 18.37, neutral |
| Engram rows through a memory mapping, or a bitmap of pages read before | prose slower (page faults on misses serialize) / neutral |
| A GEMV that reads Marlin's own weight buffer (no second copy) | only −0.12 ms; the kernel on its own layout gives −0.69 |
| Sparse-attention split count 4–24 instead of the heuristic 8 | all slower |
| Shared experts forked after the top-k | 18.03 / 19.35 vs 17.95 / 19.16 ms |
| One-launch router GEMM, coop epilogue writing bf16 | neutral: removing one 2–3 µs launch does not move the step |
| Integer e4m3 decode instead of the lookup table in the Triton attention kernels | 18.83 vs 17.95 ms |
| Retuned mHC and small-GEMV launch configs (37 variants) | the defaults were already best |
| Engram lookups on a kernel-free side stream | +0.17 ms |
| Greedy top-token pair in two kernels | −0.02 ms (neutral) |
| The whole DSpark propose step as one CUDA graph | drafter 1.52 vs 1.41 ms and the target +0.13 ms |
| Skipping attention or the all-reduce share of gated drafts | attention and the ring all-reduce are latency-bound: no gain |
| vLLM dual-batch overlap to hide prefill all-reduces | needs data parallelism (DP > 1); turning it on cut decode 340 → 278 tok/s |
| 4,096-token prefill chunks (again) | prefill +4 %, but the KV pool falls 1.53M → 848K tokens |
| Row-parallel Engram `wkv` + all-reduce | the PCIe all-gather costs more than the GEMV it saves |
| L2 prefetch of FP8 weights in idle windows; CPU C-states off + performance governor | no gain |

## Tried and dropped for 1.2.0

| Idea | Result |
|---|---|
| PIECEWISE CUDA graphs for prefill steps up to 256 tokens | time to first token 121 → 68 ms, code at 2 users +18 %; but 2 of 30 batched 8-prompt prefills with prompt logprobs failed beside an 8-stream decode load (eager: 0 of 30); kept eager |
| 4,096-token prefill chunks with the 1.2.0 prefill pipeline | ~+1 %, and the cold prefill bench ran out of memory |
| A fixed verify width of 4 or 3 drafts | 4 never won; 3 won prose at 2+ users and lost structured and code, hence the acceptance-aware choice |
| `SPEC_TOKENS=3` or `4` (narrower drafter blocks) | boots, but every request fails (the gate assumes 5 drafts) |
| Morrowmake's MoE fold on the decode kernel | bitwise equal, 13–18 % slower (the fold stages spill to a 504-byte stack frame) |
| Morrowmake's `moe_prefill_v2` (512-row chunks) | slower than the 1.1.0 prefill kernel |
| NCCL all-reduce beside prefill compute; the copy-engine ring with Triton flag kernels | no gain (NCCL's kernels hold SMs) / slower (each flag kernel waits for an SM) |
| An IPC all-gather with P2P stores (Engram heads, draft sampling) | slower than NCCL LL inside graphs (19 vs 16 µs tiny, 77 vs 35 µs at 6 × 6 × 256): concurrent P2P writes collapse on these links |
| A faster drafter | 1.38 ms per step, 92 % busy, no kernel above 0.17 ms |
| Engram all-gather on a side stream; gate τ 0.03–0.20; retuned mHC decode tiles | no gain |

