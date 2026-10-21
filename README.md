# DeepSeek-V4.1-Flash-1017 on 4× CMP 170HX with Engram on NVMe

Serve [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) on four NVIDIA CMP 170HX
cards, each power-capped at 180 W, with the model's 189 GiB Engram n-gram tables streamed from NVMe instead of held in
memory, and an OpenAI-compatible API: tool calls, reasoning, images, up to 1,048,576 tokens of context.

The engine is vLLM with the community sm_80 stack for this model (sparse attention, EXL3 routed experts from the
LibertAI 2.9-bpw pack, the DSpark drafter, the Engram row store on NVMe) plus this recipe's decode, all-reduce and memory changes. Every change
is a patch script applied to pinned upstream sources, measured on its own, and kept only if it was faster (or freed
memory) without changing the model's output; see [docs/how-it-works.md](docs/how-it-works.md).

## Quick start

```bash
git clone https://github.com/GNmac-lab/deepseek-v41-flash-1017-cmp170hx && cd deepseek-v41-flash-1017-cmp170hx
cp .env.example .env            # set MODELS_DIR (fast NVMe, ~400 GiB free)
tools/download.sh               # model parts + hybrid assembly (resumable)
./start.sh                      # builds the image on first use (~30-45 min), starts, waits until ready
./start.sh smoke                # one chat, one tool call, one image
```

Then talk to `http://127.0.0.1:8000/v1` with model `deepseek-v4.1-flash`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 200}'
```

Reasoning is controlled with a top-level `reasoning_effort`: `none`, `low`, `high`, `xhigh` or `max` (`minimal` and
`medium` are rejected by the model's chat template). `./stop.sh` stops the server cleanly.

## One-time host setup

- 4× CMP 170HX unlocked to 64 GiB each, all 74 SMs. TP4 needs PCIe x16 links and a P2P-capable driver; check with
  `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv` (the cards top out at PCIe 2.0,
  so x16 means 5 GT/s × 16); `tools/p2p_check.sh` shows links and P2P support. With x1/x4 links use `LAYOUT=pp4`.
- Linux, an NVIDIA driver with CUDA 13 support and P2P enabled for these cards (measured with 610.43.02), Docker with
  the NVIDIA Container Toolkit, `curl`, `python3`. Without P2P, set `P2P=0` (slower all-reduces).
- ~400 GiB on fast NVMe for the checkpoints (the 189 GiB Engram tables are read from disk while serving), plus
  ~40 GiB for the image. 64 GB+ of host RAM; spare RAM becomes page cache for the Engram rows.
- Optional: a 180 W power cap per card (`nvidia-smi -pl 180`) was used for every number below.

## The model

A hybrid checkpoint (Curtis White's recipe) assembled by `tools/download.sh` from two MIT-licensed releases, without
copying tensors:

| Part | Format | From |
|---|---|---|
| Routed experts, layers 0–39 (384 per layer) | EXL3 trellis, `mul1` codebook, 3 bpw (2 bpw on layers 18–22) | [LibertAI EXL3 2.9-bpw pack](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) |
| Attention, shared experts, KV compressor, indexer, DSpark drafter (dense) | FP8 e4m3, 32 × 32 power-of-two (UE8M0) block scales | [official release](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| DSpark drafter routed experts | FP4 | official release |
| Token embedding, LM head, vision tower | bf16 | official release |
| Engram n-gram tables (layers 1 and 14, 189 GiB) | FP8 + E8M0 scales, read from NVMe while serving | official release (shards 47–48) |
| KV cache | `fp8_ds_mla` (the model's native format) | — |

Only the routed experts come from the LibertAI pack (its 2.9 bpw average also covers the higher-bitrate dense layers
it quantized, which this recipe takes from the official FP8 release instead). Disk: 197 GB for the LibertAI shards
(the experts are spread across all of them), 18 GB of official non-expert tensors fetched by range requests, 190 GB
of Engram tables.

## Headline numbers

Measured 2026-10-21 on 4× CMP 170HX (74 SMs, PCIe 2.0 x16, P2P on, 180 W per card), with the model's 189 GiB Engram
n-gram tables streamed from NVMe on demand (not loaded into GPU or host memory), 262,144-token context, defaults from
`.env.example`. Decode follows the MiaAI-Lab protocol (see [docs/results.md](docs/results.md)): temperature 0,
thinking off, 400-token cap; 1 user = median of 5 streaming runs, 2/4/8 users = median of 3 aggregate runs.

| | TP4 (default), P2P on | PP4 |
|---|---:|---:|
| 1 user, structured / code / prose (tok/s) | **296 / 212 / 150** | **111 / 81 / 52** |
| 2 users, aggregate | 391 / 216 / 208 | 112 / 75 / 54 |
| 4 users, aggregate | 572 / 336 / 277 | 195 / 133 / 102 |
| 8 users, aggregate | 796 / 477 / 359 | 320 / 188 / 144 |
| Cold prefill (median of 9, ~24-38K-token prompts) | 1,759 tok/s | 2,375 tok/s |
| KV pool at 262,144 context | 1,574,996 tokens (6.0 full-length requests) | 2,539,313 tokens (9.7 full-length requests) |

Decode rates are rounded to whole tok/s; exact values are in [docs/results.md](docs/results.md).

Quality (TP4): HumanEval 156/164, GSM8K 1,275/1,319, scored on the answer after the reasoning. PP4 was measured on
the same x16 host; on x4 links its decode should change little (activations only) while TP4 would slow down sharply.

## Settings

All in `.env` (defaults in `.env.example`):

| Setting | Default | |
|---|---|---|
| `LAYOUT` | `tp4` | `tp4` or `pp4` (see below) |
| `MAX_MODEL_LEN` | `262144` | up to `1048576`, the model's native context (KV pool 1,454,546 tokens; 1-user decode −2 %, 8 users −6 %) |
| `MAX_NUM_SEQS` | `8` | concurrent requests |
| `GPU_MEMORY_UTILIZATION` | `0.95` | |
| `P2P` | `1` | `0` sends NCCL traffic through host memory |
| `BIND`, `PORT` | `127.0.0.1`, `8000` | where the API is published |

## Choosing a layout

| | Tensor-parallel 4 (`tp4`, default) | Pipeline-parallel 4 (`pp4`) |
|---|---|---|
| Each card holds | a quarter of every layer | a quarter of the layers |
| Best for | the fastest answer per request | narrow PCIe links |
| Traffic between cards | two all-reduces per layer per step | activations once per stage |
| Links | PCIe x16 (all-reduces run at the link limit) | x1/x4 is enough |

## Serving other machines

The API has no authentication, so it is published on `127.0.0.1` only. From another machine, forward the port over
SSH instead of widening `BIND`:

```bash
ssh -N -L 127.0.0.1:8000:127.0.0.1:8000 you@gpu-host     # then use http://127.0.0.1:8000/v1 locally
```

## Documentation

- [docs/results.md](docs/results.md) — every measurement, method and quality check
- [docs/how-it-works.md](docs/how-it-works.md) — the stack and each optimization with its measured effect
- [docs/kv-and-context.md](docs/kv-and-context.md) — KV pool sizing and the 1M-token option
- [docs/troubleshooting.md](docs/troubleshooting.md) — boot traps and their causes
- [docs/tuning-log.md](docs/tuning-log.md) — what was tried, including what did not help

## Update

`git pull && docker/build.sh && ./start.sh restart`

## Licences

This recipe is MIT-licensed, except `patches/vllm-exl3/` (AGPL-3.0-only, because it modifies vcruz305/vllm-exl3).
The built image contains AGPL-3.0 code; see [NOTICE.md](NOTICE.md) for every component, its licence and what that
means if you serve others. Model weights: DeepSeek-V4.1-Flash-1017 and the LibertAI EXL3 pack are MIT.

## Credits

- **DeepSeek** for DeepSeek-V4.1-Flash-1017; **LibertAI** for the EXL3 2.9-bpw quantization.
- **Curtis White** ([CurtisAccelerate](https://github.com/CurtisAccelerate)) for the TP4 hybrid-checkpoint recipe and
  the tensor-core EXL3 MoE port.
- **zebgop-ops** (dsv41reap-pp: sm_80 sparse attention, PP shadow KV, DSpark V1, Engram row store), **kaka86mm**
  (dsv41-flash-pp4-170hx), **vcruz305** (vllm-exl3), **turboderp** (exllamav3 and EXL3), **HalfVulpes**
  (Deepinfer-SM80), **FlashInfer** (PCIe IPC all-reduce), **Morrowmake** (the CMP 170HX recipe format, P2P work and
  tensor-core MoE kernels this stack builds on), **MiaAI-Lab** (the decode benchmark), the **vLLM** project.
