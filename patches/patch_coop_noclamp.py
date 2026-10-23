#!/usr/bin/env python3
"""dsv41 coop tile (after patch_coop_ptr.py): the main loop runs unclamped while the prefetch PF slices ahead stays in
range (ib + 2 * PF <= myn), and the clamped form only for the remaining PF groups. Same loads, products and fold
order: bit-identical; the per-slice compare + 64-bit select leaves the steady-state loop.
usage: patch_coop_noclamp.py <exl3_moe_coop_kernel.cuh>"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "patch_coop_noclamp.py"
if MARK in s:
    print("already patched", p)
    sys.exit(0)
old = """        for (int ib = 0; ib < full; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];
                const uint32_t* src = (ib + d + PF < myn) ? bq : blast;
"""
new = """        int ib = 0;
        // dsv41 (patch_coop_noclamp.py): every prefetch of these groups is in range, no clamp
        for (; ib + 2 * PF <= myn; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_p(bq, l);
                bq += slice_stride;
                tile_step(ap, bw);
                ap += 8;
                if ((d + 1) % FOLD == 0) fold();
            }
        }
        for (; ib < full; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];
                const uint32_t* src = (ib + d + PF < myn) ? bq : blast;
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
