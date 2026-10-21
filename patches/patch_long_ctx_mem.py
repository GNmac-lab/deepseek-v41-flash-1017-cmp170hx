#!/usr/bin/env python3
"""dsv41 v36: size two context-length workspaces for this deployment instead of for large multi-request batches.

Both are reserved at max_model_len scale during vLLM's memory profile and share one workspace buffer, so the larger
one sets the peak. At 1M context they cost ~3 GiB more than at 262K, which is why max_model_len 1048576 could not start.
  - DeepseekV4 sparse-attention prefill gathers the whole compressed context of PREFILL_CHUNK_SIZE (4) requests at
    once into bf16 [4, max_model_len / compress + window + batched, 512]. DSV41_PREFILL_CHUNK_REQS sets the request
    count (one request at a time: 1/4 of the memory; concurrent prefills just loop).
  - The sparse indexer's prefill K gather is sized 40 x max_model_len entries (132 B each). The running requests'
    tokens can never exceed the KV pool, so a few x max_model_len always suffices; DSV41_INDEXER_BUF_MULT sets it.
Only the grouping of requests into gather chunks changes; every request's attention and top-k are computed the same.
usage: patch_long_ctx_mem.py <site-packages>"""
import sys

sp = sys.argv[1]


def patch(p, pairs, mark):
    s = open(p).read()
    if mark in s:
        print("already patched", p)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


patch(f"{sp}/vllm/models/deepseek_v4_1/attention.py", [
    ("    PREFILL_CHUNK_SIZE: ClassVar[int] = 4\n",
     "    # dsv41 (ds41-patches/patch_long_ctx_mem.py): requests gathered per sparse prefill chunk\n"
     "    PREFILL_CHUNK_SIZE: ClassVar[int] = int(__import__(\"os\").environ.get(\"DSV41_PREFILL_CHUNK_REQS\", \"4\"))\n"),
], "patch_long_ctx_mem.py")

patch(f"{sp}/vllm/v1/attention/backends/mla/indexer.py", [
    ("    return max_model_len * 40\n",
     "    # dsv41 (ds41-patches/patch_long_ctx_mem.py): the multiplier is a batch allowance; running tokens never exceed\n"
     "    # the KV pool, so a few x max_model_len suffices for a pool of a few x max_model_len\n"
     "    return max_model_len * int(__import__(\"os\").environ.get(\"DSV41_INDEXER_BUF_MULT\", \"40\"))\n"),
], "patch_long_ctx_mem.py")
