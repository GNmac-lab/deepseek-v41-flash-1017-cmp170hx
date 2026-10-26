# Results

Release 1.2.0, measured 2026-10-26 on 4× CMP 170HX (64 GiB, 74 SMs, PCIe 2.0 x16, P2P on, 180 W per card) with the
image built by `docker/build.sh`, started by `./start.sh` with the defaults in `.env.example` (262,144-token
context, 8 sequences). Each layout was measured on its own boot, after the image's first-start kernel compilation.
PP4 uses the 10,11,11,8 layer split, PIECEWISE CUDA graphs and synchronous scheduling, and was measured on the same
x16 links (not on x4). The 1.1.0 columns are from the 1.1.0 image on the same host.

## Decode throughput

MiaAI-Lab protocol (`bench/mia.py`, which runs MiaAI-Lab's `bench_decode.py` unmodified): structured, code and
prose prompts; temperature 0; thinking off; 400-token cap. One user = median of five streaming decode rates,
measured after the first token. Two, four and eight users = median of three runs of aggregate throughput (completion
tokens divided by the batch's wall time, including prefill and client overhead). The TP4 columns are the median of
5 such passes on one boot; the PP4 columns are one pass.

| tok/s, structured / code / prose | TP4 1.2.0 | TP4 1.1.0 | PP4 1.2.0 | PP4 1.1.0 |
|---|---:|---:|---:|---:|
| 1 user, streaming | **355.7 / 258.4 / 183.6** | 336.0 / 246.8 / 180.0 | 117.5 / 90.5 / 63.2 | 118.3 / 92.2 / 63.8 |
| 2 users, aggregate | 433.1 / 243.8 / 238.8 | 419.3 / 237.9 / 227.7 | 118.4 / 83.4 / 65.2 | 120.8 / 83.5 / 65.2 |
| 4 users, aggregate | 626.4 / 387.8 / 348.7 | 602.4 / 367.5 / 307.8 | 205.9 / 142.9 / 105.1 | 208.8 / 144.6 / 114.8 |
| 8 users, aggregate | 867.6 / 544.0 / 440.5 | 813.1 / 504.8 / 385.1 | 340.5 / 213.9 / 163.1 | 340.6 / 215.4 / 160.5 |

DSpark drafts 5 tokens per step. Structured output and code are accepted far more often than prose (about 6 vs 3
tokens per step at one user), which is why prose is slowest and varies most between runs (±3–4 % at one user: the
accepted tokens per step change with the text, and greedy text is not bit-reproducible across restarts). The code
prompt's answer is ~95 tokens, so its 2–8-user rates include a large share of time to first token.

Two mechanisms reduce the work for drafts that will be rejected (see [how-it-works.md](how-it-works.md)):

- The verify width (`DSV41_DYNK`, TP4): per step, 3 or 5 drafts are verified, from each request's recent
  per-position acceptance. Same-boot A-B-A on the development build, prose: 2 users 232 → 248, 4 users 306/316 → 349,
  8 users 420/419 → 445 tok/s; structured, code and one user unchanged. A fixed width of 3 wins prose at 2+ users but
  loses structured and code (8 users: 767 vs 872 structured).
- The confidence gate: in 5-draft steps with an all-greedy batch, drafts after the confidence drop skip their routed
  experts. A request with temperature > 0 verifies all its drafts. Same-prompt A/B on the 1.1.0 development build,
  prose at one user: 174.7 tok/s ungated vs 179.9 gated.

## Prefill and memory

| | TP4 1.2.0 | TP4 1.1.0 | PP4 1.2.0 | PP4 1.1.0 |
|---|---:|---:|---:|---:|
| Cold prefill, median of 9 (3 public-domain texts of ~24K, ~34K and ~38K tokens, 3 runs each, no cache hits) | **2,416 tok/s** | 1,764 | 2,311 | 2,457 |
| KV pool at 262,144 context | **1,395,731 tokens** (5.3 full-length requests) | 1,526,955 (5.8) | 2,539,313 (9.7) | 2,539,313 (9.7) |
| KV pool at 1,048,576 context (`MAX_MODEL_LEN=1048576`) | 1,270,789 tokens (1.21) | 1,426,183 (1.36) | — | — |

The first run on each text is ~10 % slower: the Engram rows for new text are read from NVMe until the page cache
holds them. 1.1.0's prefill was bound by its two 2,048-row all-reduces per layer at the PCIe 2.0 link rate; 1.2.0
splits each layer in two row slices so that each slice's sums run beside the other slice's compute, on copy-engine
copies that leave the SMs to the compute, with faster MoE prefill kernels and Engram reads ahead of use. PP4 runs
the 1.1.0 code paths: its prefill difference is within the cold bench's spread between boots (±3–7 %). The TP4 KV
pool is ~9 % smaller because of the CUDA graphs for 3-draft decode steps.

## Quality

`bench/quality.py`: fixed order, up to 8 requests at a time, temperature 0, the server's default reasoning, scored
on the answer after the reasoning (`content`). HumanEval allows 4,096 reply tokens and GSM8K 3,072; length-capped
replies stay in the score.

| | TP4 1.2.0 | TP4 1.1.0 |
|---|---:|---:|
| HumanEval | 154/164 (6 length-capped) | 156/164 |
| GSM8K | 1,275/1,319 | 1,273/1,319 |

The development build of 1.2.0 scored 156/164 and 1,276/1,319 on two runs (verify width on and off). Paired with
those runs, the release image differs on 2 vs 4 and 0 vs 2 HumanEval problems and 6 vs 7 GSM8K problems (McNemar
p ≥ 0.5): run-to-run spread, not a quality change.

Paired NLL evaluation (411 passages of up to 42 tokens: the GPL-3 licence text and Python standard-library source,
10,155 scored tokens; passages this short run through the decode MoE): 3.50001 nats per token vs 1.1.0's 3.48019
(+0.57 %, 1.7 σ; 198 passages lower, 213 higher). The same change appeared when the new decode MoE kernel went in;
a 900-passage check of that step gave −0.22 %. Each prefill change was checked with paired NLL on 48–72 medium-length
passages against the build before it (largest: +0.10 %, prefill attention per slice), and the per-slice attention
with a causality test (a random 1,000-token sequence repeated once: the first copy stays at 13.0 nats per token).

This model is very sensitive to rounding: a 1-ulp bf16 change in one layer-0 output moves short-context prompt
logprobs by up to several nats, as much as running the same prompt in a different batch shape. Same-shape runs are
bit-exact. Compare builds with paired NLL over many passages, not token by token.

Needle retrieval (a passphrase in random text at 10 / 50 / 90 % depth): 6/6 at 19K and 72K tokens; with the 1M
option 3/3 at 388K tokens.

## Soak

TP4, 8 concurrent streams of mixed prose prompts (400 tokens, temperature 0) for 20 minutes: 1,018 requests,
403,570 tokens, 0 errors, 334.8 tok/s mean aggregate, no server errors (1.1.0: 857 requests, 281.0 tok/s). Hottest
card 66 °C core / 71 °C HBM at the 180 W cap (two GPU fans plus a rear 140 mm exhaust fan).
