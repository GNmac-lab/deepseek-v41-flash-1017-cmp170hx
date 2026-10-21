#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""dsv41 v32: skip the int32 -> int64 top-k id cast on the cooperative MoE prep path (the prep kernel widens ids
itself). (Split from patch_v32_small.py; the Ampere half is patches/patch_v32_wo_a.py.)
usage: patch_exl3_ids_cast.py <site-packages>"""
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


patch(f"{sp}/vllm_exl3/exl3.py", [
    ("""    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
""", """    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
    # dsv41 (ds41-patches/patch_v32_small.py): the coop prep widens ids itself; skip the separate cast there
    if not (_DSV41_COOP_PREP and _COOP and expert_map is None and ids.numel() <= 256
            and get_moe_kernel_backend() != "native"):
        ids = ids.to(torch.long)
"""),
    ("""            return out

    local = map_topk_to_local(ids, n_exp, expert_map)
""", """            return out

    if ids.dtype != torch.long:  # dsv41: the coop path was not taken
        ids = ids.to(torch.long)
    local = map_topk_to_local(ids, n_exp, expert_map)
"""),
], "dsv41 (ds41-patches/patch_v32_small.py)")
