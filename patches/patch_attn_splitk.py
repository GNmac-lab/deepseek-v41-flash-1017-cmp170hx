#!/usr/bin/env python3
"""Use the generic split-K (flash-decoding) sparse-MLA decode kernels on sm80 when VLLM_DSV41_DECODE_SPLITK=1.

rocm_aiter_mla_sparse._rocm_sparse_attn_decode_ragged_triton sends every non-gfx942/950 device to an untuned
single-pass kernel with a (num_queries, heads/16) grid: one TP4 rank has 16 heads, so a 1-user DSpark verify
step launches 6 programs on 70 SMs. The gfx942 branch splits the KV walk (_sparse_attn_decode_partial_kernel
+ _sparse_attn_decode_reduce_kernel, fp8 via the same LUT); offline on sm80: 1.97x at 6 queries, 7.2x at a
2,048-entry compressed list, equal at 48 queries, max diff 1e-3 (bf16 output rounding).
usage: patch_attn_splitk.py <vllm/v1/attention/ops/rocm_aiter_mla_sparse.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "_DS41_DECODE_SPLITK"
if MARK in s:
    print("already patched"); sys.exit(0)
old = "    if not (_ON_GFX942 or _ON_GFX950):  # Fallback path for un-tuned architectures.\n"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "    if not (_ON_GFX942 or _ON_GFX950 or _DS41_DECODE_SPLITK):  # Fallback path for un-tuned architectures.\n", 1)
anchor = "\ndef _rocm_sparse_attn_decode_ragged_triton(\n"
assert s.count(anchor) == 1
s = s.replace(anchor, "\nimport os as _ds41_os\n_DS41_DECODE_SPLITK = _ds41_os.environ.get(\"VLLM_DSV41_DECODE_SPLITK\", \"0\") == \"1\"\n\n" + anchor, 1)
open(p, "w").write(s)
print("patched", p)
