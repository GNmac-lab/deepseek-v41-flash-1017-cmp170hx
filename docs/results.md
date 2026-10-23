# Results

Release 1.1.0, measured 2026-10-22 on 4× CMP 170HX (64 GiB, 74 SMs, PCIe 2.0 x16, P2P on, 180 W per card) with the
image built by `docker/build.sh`, started by `./start.sh` with the defaults in `.env.example` (262,144-token
context, 8 sequences). Each layout was measured on its own boot, after the image's first-start kernel compilation.
PP4 uses the 10,11,11,8 layer split, PIECEWISE CUDA graphs and synchronous scheduling, and was measured on the same
x16 links (not on x4). The 1.0.0 columns are from the 1.0.0 image on the same host.

## Decode throughput

MiaAI-Lab protocol (`bench/mia.py`, which runs MiaAI-Lab's `bench_decode.py` unmodified): structured, code and
prose prompts; temperature 0; thinking off; 400-token cap. One user = median of five streaming decode rates,
measured after the first token. Two, four and eight users = median of three runs of aggregate throughput (completion
tokens divided by the batch's wall time, including prefill and client overhead). The TP4 1.1.0 column is the median
of 5 such passes on one boot; the other columns are one pass.

| tok/s, structured / code / prose | TP4 1.1.0 | TP4 1.0.0 | PP4 1.1.0 | PP4 1.0.0 |
|---|---:|---:|---:|---:|
| 1 user, streaming | **336.0 / 246.8 / 180.0** | 295.5 / 212.2 / 149.8 | 118.3 / 92.2 / 63.8 | 110.5 / 81.2 / 52.0 |
| 2 users, aggregate | 419.3 / 237.9 / 227.7 | 390.6 / 216.4 / 207.6 | 120.8 / 83.5 / 65.2 | 111.6 / 75.0 / 53.8 |
| 4 users, aggregate | 602.4 / 367.5 / 307.8 | 572.3 / 335.9 / 276.5 | 208.8 / 144.6 / 114.8 | 195.0 / 132.9 / 102.1 |
| 8 users, aggregate | 813.1 / 504.8 / 385.1 | 795.8 / 476.6 / 358.8 | 340.6 / 215.4 / 160.5 | 320.4 / 188.0 / 143.6 |

DSpark drafts 5 tokens per step. Structured output and code are accepted far more often than prose (about 6 vs 3
tokens per step at one user), which is why prose is slowest and varies most between runs (±3–4 % at one user: the
accepted tokens per step change with the text, and greedy text is not bit-reproducible across restarts).

The confidence gate (see [how-it-works.md](how-it-works.md)) works only on all-greedy batches. A request with
temperature > 0 verifies all 5 drafts. Same-prompt A/B on the development build, prose at one user: 174.7 tok/s
ungated vs 179.9 gated (8 × 12 alternating blocks). Prose at 8 users: 366 → 383; code at 4 users: 355 → 373.

## Prefill and memory

| | TP4 1.1.0 | TP4 1.0.0 | PP4 1.1.0 | PP4 1.0.0 |
|---|---:|---:|---:|---:|
| Cold prefill, median of 9 (3 public-domain texts of ~24K, ~34K and ~38K tokens, 3 runs each, no cache hits) | **1,764 tok/s** | 1,759 | 2,457 | 2,375 |
| KV pool at 262,144 context | **1,526,955 tokens** (5.8 full-length requests) | 1,574,996 (6.0) | 2,539,313 (9.7) | 2,539,313 (9.7) |
| KV pool at 1,048,576 context (`MAX_MODEL_LEN=1048576`) | 1,426,183 tokens (1.36) | 1,454,546 (1.39) | — | — |

The first run on each text is ~10 % slower: the Engram rows for new text are read from NVMe until the page cache
holds them. 1.1.0 changes decode, not prefill: prefill time is mostly the 2,048-row all-reduces (~46 % of it, at
the PCIe 2.0 link rate) and the tensor-core MoE (~29 %).

## Quality

`bench/quality.py`: fixed order, up to 8 requests at a time, temperature 0, the server's default reasoning, scored
on the answer after the reasoning (`content`). HumanEval allows 4,096 reply tokens and GSM8K 3,072; length-capped
replies stay in the score.

| | TP4 1.1.0 | TP4 1.0.0 |
|---|---:|---:|
| HumanEval | 156/164 | 156/164 |
| GSM8K | 1,273/1,319 | 1,275/1,319 |

Every change in this recipe was also checked with a paired NLL evaluation (411 passages of up to 42 tokens: the GPL-3
licence text and Python standard-library source, 10,155 scored tokens; passages this short run through the decode
MoE and mid-size kernels). The 1.1.0 image scores 3.48019 nats per token: identical on all 411 passages to the
development build it was checked against, and −0.68 % vs 1.0.0's 3.50409 (217 passages lower, 194 higher: fp32 sum
order in the new kernels, no change in precision). The validation mode that also sends prompt tokens through the
M ≤ 8 decode kernels scored −0.47 % vs 1.0.0.

Needle retrieval (a passphrase in random text at 10 / 50 / 90 % depth): 6/6 at 19K and 72K tokens; with the 1M
option 3/3 at 388K tokens (1.0.0 also passed at 776K and 997K tokens).

## Soak

TP4, 8 concurrent streams of mixed prose prompts (400 tokens, temperature 0) for 20 minutes: 857 requests,
339,337 tokens, 0 errors, 281.0 tok/s mean aggregate, no server errors (1.0.0: 765 requests, 249.8 tok/s). Hottest
card 66 °C core / 70 °C HBM at the 180 W cap (two GPU fans plus a rear 140 mm exhaust fan).
