# Changelog

## 1.0.0 — 2026-10-21

First release: DeepSeek-V4.1-Flash-1017 (hybrid EXL3 2.9-bpw experts + official FP8) on 4× CMP 170HX.

- TP4 (default) and PP4 layouts; 262,144-token context by default, up to 1,048,576.
- Engine image built from pinned upstream sources (`docker/Dockerfile`) plus the patch scripts in `patches/`, each measured on its own.
- TP4, MiaAI-Lab protocol: 1 user 295.5 / 212.2 / 149.8 tok/s (structured / code / prose), 8 users 795.8 / 476.6 /
  358.8; cold prefill 1,759 tok/s; KV pool 1,574,996 tokens. HumanEval 156/164, GSM8K 1,275/1,319.
