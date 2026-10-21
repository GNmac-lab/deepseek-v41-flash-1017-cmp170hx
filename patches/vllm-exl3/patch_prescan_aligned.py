#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Make the vllm-exl3 arena prescan honour VLLM_EXL3_MOE_TP_ALIGN (+ VLLM_EXL3_MOE_TP_ROTATE).

mixed_k_guard.shard_prescanned_trellis_shapes splits the prescanned trellis shapes evenly over TP
(DSV4.1 2304 / 4 = 576 -> 36 tiles), but with the aligned split the loader hands each rank
640/640/512/512 (40/40/32/32 tiles, rotated per layer), so direct fill fails with
"arena slot shape mismatch ... got (320, 40, 48) planned (320, 36, 48)".
usage: patch_prescan_aligned.py <vllm_exl3/mixed_k_guard.py>
"""
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "[ds41-prescan-aligned]"
if MARK in src:
    print("already patched"); sys.exit(0)
old_r = '''        resolver = getattr(exl3_module, "_resolve_tp_geometry", None)
        tp_size = 1
'''
new_r = '''        resolver = getattr(exl3_module, "_resolve_tp_geometry", None)
        tp_size = 1
        _rank = 0
'''
assert src.count(old_r) == 1, src.count(old_r)
src = src.replace(old_r, new_r, 1)
old = '''        return shard_prescanned_trellis_shapes(shapes, int(tp_size))
'''
new = f'''        # {MARK} aligned (uneven, rotated) MoE TP split, as the loader applies it
        _align = exl3_module._moe_tp_align() if hasattr(exl3_module, "_moe_tp_align") else 0
        if _align > 0 and int(tp_size) > 1 and not getattr(layer, "use_ep", False):
            _r = (int(_rank) + int(getattr(layer, "_exl3_tp_rotation", 0) or 0)) % int(tp_size)
            out = {{"gate": {{}}, "up": {{}}, "down": {{}}}}
            for proj, by_eid in shapes.items():
                for eid, shape in by_eid.items():
                    dims = list(shape)
                    d = 1 if proj in ("gate", "up") else 0
                    _off, _len = exl3_module.aligned_tp_split(dims[d] * 16, _r, int(tp_size), _align)
                    dims[d] = _len // 16
                    out[proj][int(eid)] = tuple(dims)
            return out
        return shard_prescanned_trellis_shapes(shapes, int(tp_size))
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)
open(path, "w").write(src)
print("patched", path)
