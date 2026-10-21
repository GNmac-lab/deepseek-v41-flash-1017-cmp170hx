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

## Tried and dropped

| Idea | Result |
|---|---|
| Host-memory (shared-memory) all-reduce | 21 µs at 1 token but 164 µs at 6 tokens; prose 92 → 64 |
| NCCL tree / no-P2P / vLLM custom all-reduce | slower than the NCCL P2P ring, then FlashInfer, then the ring variant |
| Sinkhorn on a side stream beside full-grid GEMMs | 0 gain: stretched GEMM tails (+0.39 ms) |
| FP8 GEMV for the Marlin (FP8) layers | ties Marlin once weights stream from HBM (offline benches must rotate copies larger than L2) |
| Router GEMM retune | already 8.35 µs offline; the in-step 15–19 µs is contention with the shared-expert stream |
| DSpark top-k truncation, a CUDA-graph draft sampler, a full-graph drafter | acceptance loss, ≤ 0.2 ms, GPUs already 96–97 % busy |
| DSpark K = 6 or 7 | does not start: K must be a multiple of the drafter's block size (5) |
| Adaptive verification (confidence-gated drafts) | step −0.7 ms but acceptance loss and overhead cancel it; MiaAI unchanged |
| Shared expert after the router | slower (258.5 vs 261.8 structured) |
| Coop k-split 2, fp32-accumulate MMA, simpler activation remat | +0.6 % or slower / not bit-identical |
| Fused top-K ragged prep, router reduce folded into top-k | no measurable gain |
| Inverse RoPE folded into the `wo_a` kernel | +0.8 ms (extra gather loads break the pipelining) |
| Drafter argmax over the PCIe all-reduce | +0.23 ms (the drafter runs eagerly) |
| Mia-AiLab EAQ-v1 2.9-bpw experts instead of LibertAI | same speed, +1.55 % NLL in our hybrid (2.7 σ) |
| Prefill chunk 4,096 / 1,024 | less KV and slower / does not start with images |
| Coop MoE kernel for up to 512 slots (8-user batches) | same 8-user throughput as the tensor-core path |
| More registers for the coop MoE kernel (1 block per SM instead of 2) | 10–19 % slower; occupancy beats the rematerialized constants |
| CUDA stream / graph-node priority for the MoE critical path | nodes inherit the priority, but a resident small kernel is not faster beside a big one |
| Fusing the Marlin FP8 GEMVs | they already reach 85–93 % of the card's read bandwidth for their size (~2.5 µs fixed cost per kernel) |
| FP8 vocab head for the target (exactness-checked top-k) | the error bound needed for an exact argmax costs most of the bytes saved |
| Clock offsets, raising the 180 W cap | not used (clock caps at 1,470 MHz; power cap kept for thermals) |
