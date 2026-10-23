#!/usr/bin/env python3
"""dsv41: gemm_mid configs for every served shape in three token buckets (M <= 16, <= 32, > 32), from the
2026-10-22 sweep (bench_w2.py mid, M = 12 / 24 / 48 on CMP 170HX). Replaces the two-bucket table and the
heuristic for the shapes it lacked. Same kernel, same arithmetic per config.
usage: patch_mid_table.py <site-packages>"""
import sys

p = f"{sys.argv[1]}/vllm/model_executor/kernels/linear/mxfp8/dsv41_fp8w.py"
s = open(p).read()
if "_MID3" in s:
    print("already patched")
    sys.exit(0)
old = '''def _mid_cfg(N: int, K: int, M: int) -> tuple:
    cfg = _MID.get((N, K, M <= 24))
'''
new = '''# dsv41 (patch_mid_table.py): (N, K, bucket) with bucket 0: M <= 16, 1: M <= 32, 2: M > 32
_MID3 = {
    (1792, 5120, 0): (1, 10, 2, 2, 4), (1792, 5120, 1): (1, 8, 2, 2, 4), (1792, 5120, 2): (1, 5, 2, 2, 8),
    (1152, 5120, 0): (1, 8, 2, 2, 4), (1152, 5120, 1): (1, 8, 2, 2, 4), (1152, 5120, 2): (1, 8, 4, 2, 8),
    (8192, 1280, 0): (1, 2, 2, 2, 8), (8192, 1280, 1): (1, 2, 4, 2, 8), (8192, 1280, 2): (1, 1, 4, 2, 8),
    (4096, 1280, 0): (1, 4, 2, 2, 4), (4096, 1280, 1): (1, 3, 2, 2, 4), (4096, 1280, 2): (1, 2, 4, 2, 8),
    (5120, 2048, 0): (1, 4, 2, 2, 4), (5120, 2048, 1): (1, 3, 4, 2, 8), (5120, 2048, 2): (1, 3, 4, 2, 8),
    (5120, 1024, 0): (1, 4, 2, 2, 4), (5120, 1024, 1): (1, 2, 2, 2, 4), (5120, 1024, 2): (1, 1, 4, 2, 8),
    (2048, 5120, 0): (1, 8, 2, 2, 4), (2048, 5120, 1): (1, 8, 2, 2, 4), (2048, 5120, 2): (1, 4, 4, 2, 8),
}


def _mid_cfg(N: int, K: int, M: int) -> tuple:
    cfg3 = _MID3.get((N, K, 0 if M <= 16 else (1 if M <= 32 else 2)))
    if cfg3 is not None:
        return cfg3
    cfg = _MID.get((N, K, M <= 24))
'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
