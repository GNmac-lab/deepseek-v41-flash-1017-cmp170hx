#!/usr/bin/env python3
"""dsv41 coop MoE tile: walk the B prefetch and A fragment loads with incremented pointers instead of recomputing
base + i * stride (64-bit IMAD.WIDE/LEA chains) for every load. Same addresses, same clamp, same order:
bit-identical. usage: patch_coop_ptr.py <exl3_moe_coop_kernel.cuh>"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "patch_coop_ptr.py"
if MARK in s:
    print("already patched", p)
    sys.exit(0)
rep = [
    ("""        auto tile_step = [&] (const int i, const uint32_t* bw)
        {
            FragB xb;
            xb[0] = r0_ok ? arow[(size_t) i * 8] : hzero;
            xb[1] = r0_ok ? arow[(size_t) i * 8 + 4] : hzero;""",
     """        // dsv41 (patch_coop_ptr.py): the caller passes the A fragment pointer (arow + i * 8)
        auto tile_step = [&] (const half2* ap, const uint32_t* bw)
        {
            FragB xb;
            xb[0] = r0_ok ? ap[0] : hzero;
            xb[1] = r0_ok ? ap[4] : hzero;"""),
    ("""        const int full = myn - myn % PF;
        for (int ib = 0; ib < full; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];
                const int nx = min(ib + d + PF, myn - 1);
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(nx, l);
                tile_step(ib + d, bw);
                if ((d + 1) % FOLD == 0) fold();
            }
        }
        for (int i = full; i < myn; ++i)
        {
            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = ld_b(i, l);
            tile_step(i, bw);
        }""",
     """        const int full = myn - myn % PF;
        // dsv41 (patch_coop_ptr.py): bq = source of the prefetch PF slices ahead, clamped to the last slice exactly
        // as min(i + PF, myn - 1) did; ap = this slice's A fragment
        const uint32_t* bq = bp + (size_t) PF * slice_stride;
        const uint32_t* blast = bp + (size_t) max(myn - 1, 0) * slice_stride;
        const half2* ap = arow;
        auto ld_p = [&] (const uint32_t* src, int l) -> uint32_t
        {
            if constexpr (bits == 3)
                return lane < 24 ? __ldcs(src + l * LSTRIDE) : 0;
            else
                return __ldcs(src + l * LSTRIDE);
        };
        for (int ib = 0; ib < full; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];
                const uint32_t* src = (ib + d + PF < myn) ? bq : blast;
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_p(src, l);
                bq += slice_stride;
                tile_step(ap, bw);
                ap += 8;
                if ((d + 1) % FOLD == 0) fold();
            }
        }
        for (int i = full; i < myn; ++i)
        {
            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = ld_b(i, l);
            tile_step(arow + (size_t) i * 8, bw);
        }"""),
]
for a, b in rep:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
