#!/usr/bin/env python3
"""dsv41: coop MoE GEMV tile with the decoded weights as the MMA A operand (register-decode widths).

The installed tile feeds the activations (<= 8 rows) as the 16-row A operand of mma.m16n8k16 with rows 8-15 zeroed,
so every 16x16 weight tile needs two MMAs (one per 8-column half) and the compiler re-materializes the zero halves
and the half accumulators around every MMA (~1 move per decoded weight in SASS). The 8 weights a lane decodes per
tile are exactly the A-fragment elements of W^T (16 cols x 16 k): {f0[0], f1[0], f0[1], f1[1]}; the activation
words the tile already loads are the B fragment (16 k x 8 rows). One MMA per tile, no zero padding; the result
fragment holds cols lane/4 and lane/4 + 8 for rows 2(lane%4) and +1. Same products per output.
usage: patch_coop_mma_swap.py <exl3_moe_coop_kernel.cuh>"""
import sys

p = sys.argv[1]
s = open(p).read()
if "mma_wt_h" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:100])
    s = s.replace(old, new)


# 1. MMA helper (weights as A), placed before gemv_tile
rep("template <int bits, int cb, int WIDE, bool HALF = false>\n__device__ __forceinline__ void gemv_tile\n",
    """// dsv41 (ds41-patches/patch_coop_mma_swap.py): D(16 cols x 8 rows) += W^T(16 x 16 k) X(16 k x 8 rows), the
// weight tile as the A operand: a = {f0[0], f1[0], f0[1], f1[1]} (cols lane/4 and +8, k pairs 2(lane%4) and +8),
// b = the activation words (k pairs 2(lane%4) and +8 of row lane/4)
__device__ __forceinline__ void mma_wt_h(const FragB& f0, const FragB& f1, const FragB& x, FragC_h& c)
{
    const uint32_t* w0 = reinterpret_cast<const uint32_t*>(&f0);
    const uint32_t* w1 = reinterpret_cast<const uint32_t*>(&f1);
    const uint32_t* xb = reinterpret_cast<const uint32_t*>(&x);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(w0[0]), "r"(w1[0]), "r"(w0[1]), "r"(w1[1]),
           "r"(xb[0]), "r"(xb[1])
    );
}

template <int bits, int cb, int WIDE, bool HALF = false>
__device__ __forceinline__ void gemv_tile
""")

# 2. tile step: activations as B, one MMA per tile
rep("""        auto tile_step = [&] (const int i, const uint32_t* bw)
        {
            FragB a01, a23;
            a01[0] = r0_ok ? arow[(size_t) i * 8] : hzero;
            a23[0] = r0_ok ? arow[(size_t) i * 8 + 4] : hzero;
            a01[1] = hzero;
            a23[1] = hzero;
""", """        auto tile_step = [&] (const int i, const uint32_t* bw)
        {
            FragB xb;
            xb[0] = r0_ok ? arow[(size_t) i * 8] : hzero;
            xb[1] = r0_ok ? arow[(size_t) i * 8 + 4] : hzero;
""")
old_mma = """                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
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
        };"""
new_mma = """                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }

                mma_wt_h(f0, f1, xb, ch[t][0]);
            }
        };
        auto fold = [&] ()
        {
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                {
                    acc0[t][f].x += __low2float(ch[t][0][f]);
                    acc0[t][f].y += __high2float(ch[t][0][f]);
                    ch[t][0][f] = hzero;
                }
        };"""
rep(old_mma, new_mma)

# 3. cross-warp reduction: the register path's outputs are (row 2(lane%4) / +1, col lane/4 (+8))
old_red = """    if (r0_ok)
    {
        const int c0 = 2 * (lane & 3);
        float* red = sh_red + (wk * ROWS + r0) * TCOLS + wn * COLS;"""
new_red = """    if constexpr (REG && !HALF)
    {
        // rows >= nrows hold zeros (their activations were zero) and are never read below
        const int rr = 2 * (lane & 3);
        float* red0 = sh_red + (wk * ROWS + rr) * TCOLS + wn * COLS;
        float* red1 = red0 + TCOLS;
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
            {
                const int col = t * 16 + f * 8 + (lane >> 2);
                red0[col] = acc0[t][f].x;
                red1[col] = acc0[t][f].y;
            }
    }
    else if (r0_ok)
    {
        const int c0 = 2 * (lane & 3);
        float* red = sh_red + (wk * ROWS + r0) * TCOLS + wn * COLS;"""
rep(old_red, new_red)
open(p, "w").write(s)
print("patched", p)
