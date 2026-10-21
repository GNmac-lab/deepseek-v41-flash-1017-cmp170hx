#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Lean decode path into exllamav3's coop MoE kernel (vllm_exl3/exl3.py, apply_exl3_fused_moe).

Before calling exl3_moe_coop, the plugin built tensors only its stock / fat-expert paths use: the sentinel mapping
of map_topk_to_local (5 kernels), flat_token (arange + repeat_interleave), the expert_count scatter-add, fat,
fat_route and a zero-filled output the kernel overwrites in full. Profiled 2026-10-19 (TP4 decode, CMP 170HX):
~15 tiny kernels per MoE layer on the critical path before the coop launch. When the coop path is certain
(coop on, uniform K, no expert map, tokens * topk <= 256 so no expert can be fat) the call now gets the routed ids
directly — the kernel already drops ids outside [0, n_exp), which is what the sentinel encoded — plus the fp16
routing weights, the fp16 input and the counters. Same kernel inputs: bit-identical outputs.
usage: patch_exl3_coop_fastpath.py <vllm_exl3/exl3.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "patch_exl3_coop_fastpath" in s:
    print("already patched"); sys.exit(0)
anchor = "    local = map_topk_to_local(ids, n_exp, expert_map)\n    topk = int(ids.shape[-1])\n"
assert s.count(anchor) == 1
fast = '''    # dsv41 (ds41-patches/patch_exl3_coop_fastpath.py): decode-shaped coop calls skip the bookkeeping that only the
    # stock and fat-expert paths use; ids go in unmapped (the kernel drops ids outside [0, n_exp) itself)
    _topk = int(ids.shape[-1])
    if (
        _COOP
        and not is_mixedk
        and expert_map is None
        and tokens * _topk <= 256
        and tokens <= FAT_EXPERT_THRESHOLD
        and hasattr(exllamav3_ext, "exl3_moe_coop")
    ):
        flags = getattr(layer, "_exl3_codebook_flags", _MCG_CODEBOOK_FLAGS)
        mcg, mul1 = bool(flags[0]), bool(flags[1])
        inter_dim = int(temps[2].shape[-1])
        if (
            all(bool(flags[i]) == mcg and bool(flags[i + 1]) == mul1 for i in range(0, 6, 2))
            and hidden % 128 == 0
            and inter_dim % 128 == 0
        ):
            k = int(getattr(layer, "_exl3_k", 4))
            slots = tokens * _topk
            smax = 4 * slots
            dev = x2d.device
            sel_c = ids.reshape(tokens, _topk).to(torch.int64).contiguous()
            rw_c = weights.reshape(tokens, _topk).to(dtype=torch.float16).contiguous()
            xh = x2d.contiguous().half()
            out = torch.empty(tokens, hidden, dtype=torch.float32, device=dev)
            had_g = torch.empty((slots, hidden), dtype=torch.float16, device=dev)
            had_u = torch.empty_like(had_g)
            gu_g = torch.empty((smax, 1, inter_dim), dtype=torch.float16, device=dev)
            gu_u = torch.empty_like(gu_g)
            act_out = torch.empty_like(gu_g)
            d_out = torch.empty((smax, 1, hidden), dtype=torch.float32, device=dev)
            ctr = torch.zeros(
                smax * (inter_dim // 128) + tokens * (hidden // 128) + 2 * smax + 3,
                dtype=torch.int32,
                device=dev,
            )
            exllamav3_ext.exl3_moe_coop(
                xh, sel_c, rw_c, 0, int(n_exp), hidden,
                ptrs["gate_trellis"], ptrs["gate_suh"], ptrs["gate_svh"],
                ptrs["up_trellis"], ptrs["up_suh"], ptrs["up_svh"],
                ptrs["down_trellis"], ptrs["down_suh"], ptrs["down_svh"],
                None, None, None,
                k, k, k, mcg, mul1, MOE_ACT_SILU,
                float(limit) if (limit is not None and limit > 0) else 0.0,
                True,
                had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out,
                None, None,
            )
            return out

'''
s = s.replace(anchor, fast + anchor)
open(p, "w").write(s)
print("patched", p)
