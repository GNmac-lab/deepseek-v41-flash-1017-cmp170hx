# Results

Measured 2026-10-21 on 4× CMP 170HX (64 GiB, 74 SMs, PCIe 2.0 x16, P2P on, 180 W per card) with the image built
by `docker/build.sh` and the defaults in `.env.example` (262,144-token context, 8 sequences). Each layout was
measured on its own boot, after the image's first-start kernel compilation. PP4 uses the 10,11,11,8 layer split,
PIECEWISE CUDA graphs and synchronous scheduling, and was measured on the same x16 links (not on x4).

## Decode throughput

MiaAI-Lab protocol (`bench/mia.py`, which runs MiaAI-Lab's `bench_decode.py` unmodified): structured, code and
prose prompts; temperature 0; thinking off; 400-token cap. One user = median of five streaming decode rates,
measured after the first token. Two, four and eight users = median of three runs of aggregate throughput (completion
tokens divided by the batch's wall time, including prefill and client overhead).

| tok/s, structured / code / prose | TP4, P2P on | PP4 |
|---|---:|---:|
| 1 user, streaming | **295.5 / 212.2 / 149.8** | 110.5 / 81.2 / 52.0 |
| 2 users, aggregate | 390.6 / 216.4 / 207.6 | 111.6 / 75.0 / 53.8 |
| 4 users, aggregate | 572.3 / 335.9 / 276.5 | 195.0 / 132.9 / 102.1 |
| 8 users, aggregate | 795.8 / 476.6 / 358.8 | 320.4 / 188.0 / 143.6 |

DSpark drafts 5 tokens per step. Structured output and code are accepted far more often than prose (about 6 vs 3
tokens per step at one user), which is why prose is slowest and varies most between runs (±2–4 %).

## Prefill and memory

| | TP4, P2P on | PP4 |
|---|---:|---:|
| Cold prefill, median of 9 (3 public-domain texts of ~24K, ~34K and ~38K tokens, 3 runs each, no cache hits) | **1,759 tok/s** | 2,375 |
| KV pool at 262,144 context | **1,574,996 tokens** (6.0 full-length requests) | 2,539,313 tokens (9.7 full-length requests) |
| KV pool at 1,048,576 context (`MAX_MODEL_LEN=1048576`) | 1,454,546 tokens (1.39) | — |

The first run on each text is ~10 % slower: the Engram rows for new text are read from NVMe until the page cache
holds them.

## Quality

`bench/quality.py`: fixed order, up to 8 requests at a time, temperature 0, the server's default reasoning, scored
on the answer after the reasoning (`content`). HumanEval allows 4,096 reply tokens and GSM8K 3,072; length-capped
replies stay in the score.

| | TP4, P2P on |
|---|---:|
| HumanEval | 156/164 |
| GSM8K | 1,275/1,319 |

Every change in this recipe was also checked with a paired NLL evaluation (411 passages of up to 42 tokens: the GPL-3
licence text and Python standard-library source, 10,155 scored tokens). The final image scores 3.50409 nats per
token; the starting stack scored 3.50517 (−0.03 %, inside the run-to-run spread of sum-order changes).

Needle retrieval (a passphrase in random text at 10 / 50 / 90 % depth): 6/6 at 19K and 72K tokens; with the 1M
option 3/3 at 388K and 1/1 at 776K and 997K.

## Soak

TP4, 8 concurrent streams of mixed prose prompts (400 tokens, temperature 0) for 20 minutes: 765 requests,
302,133 tokens, 0 errors, 249.8 tok/s mean aggregate, no server errors. Hottest card 69 °C core / 73 °C HBM at the
180 W cap (two GPU fans plus a rear 140 mm exhaust fan).
