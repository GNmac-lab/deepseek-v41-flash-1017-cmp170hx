#!/usr/bin/env python3
"""dsv41 coop tile, 3 bpw register path (after patch_coop_ptr.py): lanes 24-31 load lane 23's trellis word instead of
a predicated load that defaults to zero. Only lanes 0-23 are shuffle sources (x_src_a / x_src_b are taken mod 24),
so the extra words are never read: bit-identical, and the per-load predicate and zero-fill leave the hot loop. The
clamped address stays inside the warp's own 24-word span, so no read can leave the trellis.
usage: patch_coop_laneclamp.py <exl3_moe_coop_kernel.cuh>"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "patch_coop_laneclamp.py"
if MARK in s:
    print("already patched", p)
    sys.exit(0)
pairs = [
    ("""    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane;
""", """    // dsv41 (patch_coop_laneclamp.py): at 3 bpw lanes 24-31 read lane 23's word (never a shuffle source)
    const int lane_b = (REG && bits == 3) ? min(lane, 23) : lane;
    const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + (group * WN + wn) * GWORDS + lane_b;
"""),
    ("""            return lane < 24 ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
""", """            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);   // dsv41 (patch_coop_laneclamp.py)
"""),
    ("""                return lane < 24 ? __ldcs(src + l * LSTRIDE) : 0;
""", """                return __ldcs(src + l * LSTRIDE);   // dsv41 (patch_coop_laneclamp.py)
"""),
]
for a, b in pairs:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
