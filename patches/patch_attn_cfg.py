#!/usr/bin/env python3
"""dsv41: env overrides for the sm80 split-K sparse-MLA decode partial kernel launch: DSV41_ATTN_WARPS (default 8; stock 4),
DSV41_ATTN_STAGES (default 1), DSV41_ATTN_BK (default 16 KV tokens per step; stock 32). Tuned on CMP 170HX at 1 user:
structured step 17.95 -> 17.64 ms. Not bit-identical (fp32 online-softmax order).
usage: patch_attn_cfg.py <vllm/v1/attention/ops/rocm_aiter_mla_sparse.py>"""
import sys

p = sys.argv[1]
s = open(p).read()
if "DSV41_ATTN_WARPS" in s:
    print("already patched")
    sys.exit(0)
old = """            BLOCK_K=block_k,
            NUM_SPLITS=num_splits,
            NUM_STAGES=1,
            num_warps=4,
"""
assert s.count(old) == 2, s.count(old)
s = s.replace(old, """            BLOCK_K=block_k,
            NUM_SPLITS=num_splits,
            NUM_STAGES=int(_ds41_os.environ.get("DSV41_ATTN_STAGES", "1")),   # dsv41 (patch_attn_cfg.py)
            num_warps=int(_ds41_os.environ.get("DSV41_ATTN_WARPS", "8")),
""")
old2 = "    block_k = 32  # KV tokens walked per split-K iteration. Tuned on gfx950.\n"
assert s.count(old2) == 1
s = s.replace(old2, "    block_k = int(_ds41_os.environ.get(\"DSV41_ATTN_BK\", \"16\"))  # dsv41 (patch_attn_cfg.py); stock 32\n")
open(p, "w").write(s)
print("patched", p)
