#!/usr/bin/env python3
"""dsv41: fused DSpark Markov sampling step (hybrid/dsv41_markov.py, DSV41_MARKOV_FUSED=1, default) in the sharded draft
sampler: the bias GEMM, the bf16 add, the shard max and the (value, id) pair become two Triton kernels that read each
block of the Markov weight once for every request row. Falls back to the stock ops for any shape/dtype it does not
cover. usage: patch_dspark_markov_fused.py <hybrid/dspark_proposer.py>"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "patch_dspark_markov_fused.py"
if MARK in s:
    print("already patched", p)
    sys.exit(0)
pairs = [
    ('''_DSV41_SAMPLE_GRAPH = _dsv41_os38.environ.get("DSV41_SAMPLE_GRAPH", "1") != "0"
''', '''_DSV41_SAMPLE_GRAPH = _dsv41_os38.environ.get("DSV41_SAMPLE_GRAPH", "1") != "0"
# dsv41 (ds41-patches/patch_dspark_markov_fused.py): fused Markov bias + add + shard max + pair
_DSV41_MARKOV_FUSED = _dsv41_os38.environ.get("DSV41_MARKOV_FUSED", "1") != "0"
'''),
    ('''            li = local[:, i] + torch.nn.functional.linear(e, w2)               # [R, shard] bf16, as the stock add
            v, j = li.max(dim=-1)
            pair = torch.stack([v.float(), (j + start).float()], dim=-1)       # ids < 2^24: exact in fp32
''', '''            if fused:
                pair = _mk.markov_step_pair(local[:, i], e, w2, start)        # dsv41 (patch_dspark_markov_fused.py)
            else:
                li = local[:, i] + torch.nn.functional.linear(e, w2)           # [R, shard] bf16, as the stock add
                v, j = li.max(dim=-1)
                pair = torch.stack([v.float(), (j + start).float()], dim=-1)   # ids < 2^24: exact in fp32
'''),
    ('''        drafts = torch.empty(num_reqs, n, dtype=torch.int64, device=self.device)
        prev = self.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n):
            e = m.markov_embed(prev)                                           # [R, r]
''', '''        drafts = torch.empty(num_reqs, n, dtype=torch.int64, device=self.device)
        prev = self.input_ids[self._anchor_idx[:num_reqs]]
        fused = False
        if _DSV41_MARKOV_FUSED:
            from hybrid import dsv41_markov as _mk

            r_ = w2.shape[1]
            fused = (local.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16 and w2.stride(1) == 1
                     and local.stride(-1) == 1 and r_ & (r_ - 1) == 0 and r_ <= 512)
        for i in range(n):
            e = m.markov_embed(prev)                                           # [R, r]
            if fused and (e.dtype != torch.bfloat16 or e.stride(1) != 1):
                e = e.to(torch.bfloat16).contiguous()
'''),
]
for a, b in pairs:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
