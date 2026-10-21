#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One fused prep kernel for the lean coop decode call (vllm_exl3/exl3.py; apply after patch_exl3_coop_fastpath.py).

The lean coop path still launched four small kernels per MoE layer before the coop kernels: routed ids -> int64,
routing weights -> fp16, the input bf16 -> fp16 (6.9 us for 60 KB) and the zero-filled completion counters
(~13 us per layer in the TP4 decode graph, profiled 2026-10-20). One Triton kernel now writes all four in a single
launch with the same conversions (int widening; fp32/bf16 -> fp16 round-to-nearest-even, as torch's .half()/.to()),
so the coop kernels get bit-identical inputs. DSV41_COOP_PREP=0 = stock casts.
usage: patch_exl3_coop_prep.py <vllm_exl3/exl3.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "_dsv41_coop_prep" in s:
    print("already patched"); sys.exit(0)
assert "patch_exl3_coop_fastpath" in s, "apply patch_exl3_coop_fastpath.py first"


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''            sel_c = ids.reshape(tokens, _topk).to(torch.int64).contiguous()
            rw_c = weights.reshape(tokens, _topk).to(dtype=torch.float16).contiguous()
            xh = x2d.contiguous().half()
            out = torch.empty(tokens, hidden, dtype=torch.float32, device=dev)
''', '''            out = torch.empty(tokens, hidden, dtype=torch.float32, device=dev)
            n_ctr = smax * (inter_dim // 128) + tokens * (hidden // 128) + 2 * smax + 3
            if _DSV41_COOP_PREP and x2d.dtype in (torch.bfloat16, torch.float16):
                sel_c, rw_c, xh, ctr = _dsv41_coop_prep(x2d, ids, weights, tokens, _topk, n_ctr)
            else:
                sel_c = ids.reshape(tokens, _topk).to(torch.int64).contiguous()
                rw_c = weights.reshape(tokens, _topk).to(dtype=torch.float16).contiguous()
                xh = x2d.contiguous().half()
                ctr = torch.zeros(n_ctr, dtype=torch.int32, device=dev)
''')
rep('''            ctr = torch.zeros(
                smax * (inter_dim // 128) + tokens * (hidden // 128) + 2 * smax + 3,
                dtype=torch.int32,
                device=dev,
            )
            exllamav3_ext.exl3_moe_coop(
                xh, sel_c, rw_c, 0, int(n_exp), hidden,''', '''            exllamav3_ext.exl3_moe_coop(
                xh, sel_c, rw_c, 0, int(n_exp), hidden,''')
rep("def apply_exl3_fused_moe(", '''_DSV41_COOP_PREP = os.environ.get("DSV41_COOP_PREP", "1") != "0"
_DSV41_PREP_KERNEL = None


def _dsv41_coop_prep(x2d, ids, weights, tokens, topk, n_ctr):
    """dsv41 (ds41-patches/patch_exl3_coop_prep.py): ids -> int64, weights -> fp16, x -> fp16 and zeroed counters in
    one launch (same conversions as .to(torch.int64) / .to(torch.float16) / .half() / torch.zeros)."""
    global _DSV41_PREP_KERNEL
    import triton
    import triton.language as tl

    if _DSV41_PREP_KERNEL is None:
        @triton.jit
        def _prep(x_ptr, xh_ptr, n_x, ids_ptr, sel_ptr, w_ptr, rw_ptr, n_r, ctr_ptr, n_c, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mx = offs < n_x
            x = tl.load(x_ptr + offs, mask=mx, other=0.0)
            tl.store(xh_ptr + offs, x.to(tl.float32).to(tl.float16), mask=mx)
            mr = offs < n_r
            i = tl.load(ids_ptr + offs, mask=mr, other=0)
            tl.store(sel_ptr + offs, i.to(tl.int64), mask=mr)
            w = tl.load(w_ptr + offs, mask=mr, other=0.0)
            tl.store(rw_ptr + offs, w.to(tl.float32).to(tl.float16), mask=mr)
            mc = offs < n_c
            tl.store(ctr_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=mc)

        _DSV41_PREP_KERNEL = _prep
    dev = x2d.device
    x = x2d.contiguous()
    idc = ids.reshape(-1).contiguous()
    wc = weights.reshape(-1).contiguous()
    xh = torch.empty(tokens, x.shape[1], dtype=torch.float16, device=dev)
    sel = torch.empty(tokens, topk, dtype=torch.int64, device=dev)
    rw = torch.empty(tokens, topk, dtype=torch.float16, device=dev)
    ctr = torch.empty(n_ctr, dtype=torch.int32, device=dev)
    n_x, n_r = x.numel(), tokens * topk
    BLOCK = 1024
    grid = (triton.cdiv(max(n_x, n_r, n_ctr), BLOCK),)
    _DSV41_PREP_KERNEL[grid](x, xh, n_x, idc, sel, wc, rw, n_r, ctr, n_ctr, BLOCK=BLOCK, num_warps=4)
    return sel, rw, xh, ctr


def apply_exl3_fused_moe(''')
open(p, "w").write(s)
print("patched", p)
