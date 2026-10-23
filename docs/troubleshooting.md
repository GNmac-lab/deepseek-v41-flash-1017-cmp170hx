# Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Worker killed by the kernel OOM killer during load (~25 GB host RAM per worker) | `VLLM_EXL3_MODEL_DIR` unset: the EXL3 arena prescan finds no headers and stages every expert in host RAM | `serve.sh` sets it; keep it if you write your own launcher |
| Ranks 1–3 load no experts / shape mismatch at load | the prescan split must match the 128-aligned 640/640/512/512 expert split and its per-rank offsets | applied by `patches/vllm-exl3/patch_prescan_*.py` |
| "no parameter named …" for dense tensors | the LibertAI shards also hold dense EXL3 tensors the hybrid takes from the official release | `DSV41_EXPERTS_ONLY_FILES_RE` (set in `serve.sh`) |
| `No available memory for the cache blocks` at long context | context-length workspaces at their upstream size | the image includes `patch_long_ctx_mem.py`; `serve.sh` sets `DSV41_PREFILL_CHUNK_REQS=1`, `DSV41_INDEXER_BUF_MULT=4` |
| Startup hangs at 100 % GPU and low power | an all-reduce waiting for a rank that never arrives, typically P2P that does not work | check P2P (`P2P=0` to compare); never `kill -9` a multi-GPU vLLM: `./stop.sh` |
| Very slow decode with TP4 | PCIe links narrower than x16 (each step does ~86 all-reduces) | `LAYOUT=pp4` |
| `reasoning_effort must be low, high, xhigh, max…` (HTTP 400) | the chat template rejects `minimal` and `medium` | use `none`, `low`, `high`, `xhigh` or `max` |
| Prose is ~20 % slower right after a restart | Engram rows are read from disk until they are in the page cache | warm up with a few requests |
| A very long prompt shows no log progress for minutes | the engine logs when it emits tokens; a 1M-token prefill takes ~25 min | wait; `nvidia-smi` shows the cards busy |
| Two restarts give slightly different greedy text | TP4 decode is not bit-reproducible across processes (all-reduce timing, routing near-ties) | expected; compare quality with NLL, not text |
| Requests with temperature > 0 decode slower than greedy ones | confidence-gated verification runs only for all-greedy batches; a batch with any sampled request verifies every draft | expected; the output is the same either way |
| A worker stops at startup with `FileNotFoundError` under `dsv41_variants_d/` | a misspelled name in `DSV41_VARIANTS` | use names from `patches/dsv41_variants/variants/`, comma separated; empty = none |
| A worker stops at startup with `dsv41_gate … self-test FAILED` | a variant checks its kernel against the stock one when it loads; a mismatch stops the worker instead of serving wrong output | report it with the log; `DSV41_VARIANTS=` (empty) starts without the variants |
