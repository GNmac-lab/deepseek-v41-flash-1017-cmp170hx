#!/usr/bin/env python3
"""Branch-free main loop for exllamav3's coop MoE GEMV tile (exllamav3_ext/quant/exl3_moe_coop_kernel.cuh).

gemv_tile ran every k-slice of its PF-deep prefetch ring behind two guards (`if (i >= myn) break;` and
`if (i + PF < myn)`), which on sm_80 costs divergent-branch bookkeeping in the hottest loop of the decode MoE.
For the register-decode widths (REG: 2/3/4 bpw) the full PF groups now run unguarded (the prefetch index is
clamped to the last slice; re-reads past the end are never consumed), the A-row pointer is hoisted, and the last
myn % PF slices run after with direct loads. Same loads, products and fold order: bit-identical outputs. Other
widths keep the original loop. usage: patch_coop_branchfree.py <exl3_moe_coop_kernel.cuh>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "patch_coop_branchfree" in s:
    print("already patched"); sys.exit(0)
head = "    for (int ib = 0; ib < myn; ib += PF)\n"
assert s.count(head) == 1
dec_start = "                if constexpr (!REG)\n                {\n                    dq_dispatch"
dec_end = "                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);\n"
a = s.index(head)
d0 = s.index(dec_start, a)
d1 = s.index(dec_end, d0)
# the REG decode branches of the original (bits 4 / 2 / 3), verbatim, minus the !REG staged branch
dec = s[d0:d1]
dec = dec[dec.index("                else if constexpr (bits == 4)"):]
dec = dec.replace("                else if constexpr (bits == 4)", "                if constexpr (bits == 4)", 1)
new = """    // dsv41 (ds41-patches/patch_coop_branchfree.py): register-decode widths run the full PF groups without
    // per-slice guards (prefetch index clamped; re-reads past the end are never consumed), then the last
    // myn % PF slices with direct loads. Same loads, products and fold order: bit-identical results.
    if constexpr (REG && !HALF)
    {
        const half2* arow = A2 + a_row0 + (size_t) ks0 * 8 + (lane & 3);
        auto tile_step = [&] (const int i, const uint32_t* bw)
        {
            FragB a01, a23;
            a01[0] = r0_ok ? arow[(size_t) i * 8] : hzero;
            a23[0] = r0_ok ? arow[(size_t) i * 8 + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
""" + dec + """                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
            }
        };
        auto fold = [&] ()
        {
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    acc0[t][f].x += __low2float(ch[t][f][0]);
                    acc0[t][f].y += __high2float(ch[t][f][0]);
                    ch[t][f][0] = hzero;
                }
        };
        static_assert(PF % FOLD == 0, "fold cadence");
        const int full = myn - myn % PF;
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
        }
        if (myn % FOLD) fold();
    }
    else
"""
s = s[:a] + new + s[a:]
open(p, "w").write(s)
print("patched", p)
