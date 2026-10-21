# Notices

This recipe's own scripts, patches and documents are MIT-licensed (see [LICENSE](LICENSE)), except the files in
[`patches/vllm-exl3/`](patches/vllm-exl3/). Those modify vcruz305/vllm-exl3, which is AGPL-3.0-only, so they carry
AGPL-3.0-only too (SPDX headers, licence text in that folder).

The engine image is built from the projects below at the commits shown. None of their source is stored in this
repository: `docker/Dockerfile` fetches each one at build time. The image you build therefore contains
AGPL-3.0-only code (vllm-exl3 with this recipe's changes). If you let other people use a server built from it over
a network, AGPL-3.0 section 13 requires you to offer them the corresponding source: this repository plus the
upstream commits below.

| Component | Commit | Licence | Used for | Changed here |
|---|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm), official image `vllm/vllm-openai:deepseekv41-flash-0909` | `sha256:00d577a6…` | Apache-2.0 | engine base | yes, by patch |
| [zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp) | `138e748` | Apache-2.0 | sm_80 sparse attention, PP shadow KV, DSpark V1, Engram SSD row store | yes, by patch |
| [kaka86mm/dsv41-flash-pp4-170hx](https://github.com/kaka86mm/dsv41-flash-pp4-170hx) | `950f4c0` | Apache-2.0 | four sm_80 / PP fixes, applied unchanged | no |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | `v1.5.3` (`d3739fd`) | MIT | EXL3 kernels | yes: cooperative MoE decode kernel |
| [vcruz305/vllm-exl3](https://github.com/vcruz305/vllm-exl3) | `223e246` | AGPL-3.0-only | EXL3 quantization plugin for vLLM | yes, by patch (`patches/vllm-exl3/`) |
| [CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE) | `88eb6e1` | Apache-2.0 | tensor-core EXL3 MoE (prefill and large batches) | no |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | `v0.7.0.post1` (`946200d`) | Apache-2.0 | PCIe IPC all-reduce, vendored into the image's FlashInfer 0.6.18 | yes: a neighbour-ring variant |
| [HalfVulpes/Deepinfer-SM80](https://github.com/HalfVulpes/Deepinfer-SM80) | `2a828fc` | Apache-2.0 | sm_80 sparse-attention decode and top-K combiner | `patches/deepinfer_sm80_*.patch` are adapted from its patches |

**Not redistributed, downloaded by `tools/download.sh`:**

| Item | Licence | Notes |
|---|---|---|
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | MIT | attention, shared experts, head, draft model, Engram tables |
| [LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) | MIT | routed experts, layers 0–39 (the quantization is LibertAI's) |
| `quantization_config.json` from [CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX](https://github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX) (`b8b9e99`) | no licence stated | the hybrid checkpoint recipe is Curtis White's; the file is fetched from his repository, not copied here |

**Benchmarks:** `bench/mia.py` downloads MiaAI-Lab's `tests/bench_decode.py` from
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks)
(`6278ecb`, AGPL-3.0) and runs it unmodified; it is not included here. The prefill prompts are public-domain
Project Gutenberg texts, downloaded at run time. HumanEval (MIT) and GSM8K (MIT) are read from the Hugging Face
datasets server at run time.
