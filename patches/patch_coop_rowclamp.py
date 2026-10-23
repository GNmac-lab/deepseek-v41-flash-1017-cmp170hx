#!/usr/bin/env python3
"""dsv41 coop tile (after patch_coop_ptr.py): lanes whose MMA token row lies past the run's rows load row 0's
activation instead of a zero-filled predicated load. The m16n8k16 product keeps token columns independent, and the
cross-warp reduction only reads rows < nrows, so outputs are bit-identical; the zeroing (CS2R) and the load predicate
leave the hot loop. usage: patch_coop_rowclamp.py <exl3_moe_coop_kernel.cuh>"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "patch_coop_rowclamp.py"
if MARK in s:
    print("already patched", p)
    sys.exit(0)
pairs = [
    ("""    const size_t a_row0 = r0_ok ? (size_t) rows[r0] * a_stride2 : 0;
""", """    const size_t a_row0 = r0_ok ? (size_t) rows[r0] * a_stride2 : 0;
    // dsv41 (patch_coop_rowclamp.py): token rows past nrows read row 0 (their products are never used)
    const size_t a_rowc = (size_t) rows[r0_ok ? r0 : 0] * a_stride2;
"""),
    ("""        const half2* arow = A2 + a_row0 + (size_t) ks0 * 8 + (lane & 3);
""", """        const half2* arow = A2 + a_rowc + (size_t) ks0 * 8 + (lane & 3);
"""),
    ("""            xb[0] = r0_ok ? ap[0] : hzero;
            xb[1] = r0_ok ? ap[4] : hzero;
""", """            xb[0] = ap[0];
            xb[1] = ap[4];
"""),
    ("""        // rows >= nrows hold zeros (their activations were zero) and are never read below
""", """        // rows >= nrows hold row 0's products (patch_coop_rowclamp.py) and are never read below
"""),
]
for a, b in pairs:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
