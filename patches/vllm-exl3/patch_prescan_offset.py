#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Arena prescan expert offset: 0 unless expert parallelism is on.

vllm-exl3 sets starting_expert_offset = ep_group.rank * num_experts, but with plain TP (no EP) the EP
group is the TP group, so TP ranks 1-3 look up experts 384..1535, find none, and fall back to host
staging of every expert (~25 GB anon RSS per worker -> OOM on a 94 GB host).
usage: patch_prescan_offset.py <vllm_exl3/exl3.py>
"""
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "[ds41-prescan-offset]"
if MARK in src:
    print("already patched"); sys.exit(0)
old = """        if not hasattr(layer, "starting_expert_offset"):
            try:
                from vllm.distributed.parallel_state import get_ep_group
"""
new = f"""        if not getattr(layer, "use_ep", False):  # {MARK} TP-only: every rank holds all experts
            layer.starting_expert_offset = 0
        if not hasattr(layer, "starting_expert_offset"):
            try:
                from vllm.distributed.parallel_state import get_ep_group
"""
assert src.count(old) == 1, src.count(old)
open(path, "w").write(src.replace(old, new, 1))
print("patched", path)
