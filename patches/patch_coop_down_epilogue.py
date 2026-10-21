#!/usr/bin/env python3
"""Parallel chunk epilogue for exllamav3's coop MoE down stage (apply after patch_coop_branchfree.py and
patch_coop_down_wide16.py).

With 512-column down tiles a block covers 4 output chunks of 128 columns; the stock epilogue finished them one after
another (per chunk: arrival barrier, shared-expert gate reduction, slot reduction, store), which cost more than the
wider GEMV saved (v9). Now: one fence/barrier and one atomic per chunk, the shared-expert gate once per row, and
WK / CPB warps per chunk working on all of the block's last-arrival chunks at once. Per-slot partials go to shared
memory and one warp per chunk sums them in slot order, so the epilogue's fp32 additions match the stock order.
Needs top_k <= 8 (the launcher falls back to the narrow tile otherwise).
usage: patch_coop_down_epilogue.py <exl3_moe_coop_kernel.cuh> <exl3_moe_coop.cu>
"""
import sys

kp, lp = sys.argv[1], sys.argv[2]
k = open(kp).read()
l = open(lp).read()
if "patch_coop_down_epilogue" in k:
    print("already patched"); sys.exit(0)
assert "patch_coop_down_wide16" in k, "apply patch_coop_down_wide16.py first"


def rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


k = rep(k, "constexpr int smem_part_bytes() { return WK * 128 * 4; }",
        "constexpr int SLOTS_EPI = 8;  // dsv41 (patch_coop_down_epilogue.py): per-slot partials, top_k <= 8\n"
        "constexpr int smem_part_bytes() { return (WK * 128 > 4 * SLOTS_EPI * 128 ? WK * 128 : 4 * SLOTS_EPI * 128) * 4; }")
b0 = k.index("void exl3_moe_coop_b_kernel(const MoeCoopParams p)")
kb = k[b0:]
loop = "    for (int r = 0; r < nrows; ++r)\n    for (int j = 0; j < CPB; ++j)\n"
assert kb.count(loop) == 1
new = """    if constexpr (CPB > 1)
    {
        // dsv41 (patch_coop_down_epilogue.py): finish every last-arrival chunk of this block at once, WPC warps per
        // chunk; per-slot partials [CPB][SLOTS_EPI][128] in sh_part, summed in slot order by one warp per chunk
        constexpr int WPC = WK / CPB;
        static_assert(CPB * SLOTS_EPI * 128 * 4 <= smem_part_bytes(), "slot partials");
        __shared__ int sh_last[CPB];
        const int jw = warp / WPC;
        const int wi = warp % WPC;
        for (int r = 0; r < nrows; ++r)
        {
            const int s = sh_rows[r];
            const int row = s / p.topk;
            if (threadIdx.x == 0)
            {
                int n = 0;
                for (int kk = 0; kk < p.topk; ++kk)
                    n += slot_info(p, row * p.topk + kk).active ? 1 : 0;
                sh_n_active = n;
            }
            __threadfence();
            __syncthreads();
            if (threadIdx.x < CPB)
                sh_last[threadIdx.x] = atomicAdd(p.ctr_b + (size_t) row * (p.Ho / 128) + (group * CPB + threadIdx.x) / GPC, 1)
                                       == sh_n_active * GPC * p.ksplit_b - 1;
            __syncthreads();
            bool any = false;
            #pragma unroll
            for (int j = 0; j < CPB; ++j) any |= sh_last[j] != 0;
            if (!any || (p.dbg & 2)) continue;
            __threadfence();

            if (p.sh_gate_w)
            {
                const half* xr = p.x + (size_t) row * p.x_stride;
                float dot = 0.0f;
                for (int i = threadIdx.x * 2; i < p.H; i += THREADS * 2)
                {
                    half2 xv = *((const half2*) (xr + i));
                    half2 wv = *((const half2*) (p.sh_gate_w + i));
                    dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
                }
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    dot += __shfl_xor_sync(0xffffffffu, dot, o);
                if (lane == 0) sh_part[warp] = dot;
                __syncthreads();
                if (threadIdx.x == 0)
                {
                    float tt = 0.0f;
                    for (int w = 0; w < WK; ++w) tt += sh_part[w];
                    sh_gate = 1.0f / (1.0f + __expf(-tt));
                }
                __syncthreads();
            }

            const bool mine = sh_last[jw] != 0;
            const int chunk = (group * CPB + jw) / GPC;
            const int col = chunk * 128 + lane * 4;
            float* slot_part = sh_part + (size_t) jw * SLOTS_EPI * 128 + lane * 4;
            if (mine)
                for (int kk = wi; kk < p.topk; kk += WPC)
                {
                    const int sk = row * p.topk + kk;
                    const SlotInfo sj = slot_info(p, sk);
                    float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
                    if (sj.active)
                    {
                        for (int q = 0; q < p.ksplit_b; ++q)
                        {
                            float t0, t1, t2, t3;
                            load_f4_cg(p.d_out + ((size_t) q * (p.bsz * p.topk) + sk) * p.Ho + col, t0, t1, t2, t3);
                            v0 += t0; v1 += t1; v2 += t2; v3 += t3;
                        }
                        had128(v0, v1, v2, v3, lane);
                        scale_h4(((const half*) p.d_svh[sj.local]) + col, v0, v1, v2, v3);
                        if (p.d_bias) add_h4(((const half*) p.d_bias[sj.local]) + col, v0, v1, v2, v3);
                        v0 = 0.0f + sj.w * v0; v1 = 0.0f + sj.w * v1; v2 = 0.0f + sj.w * v2; v3 = 0.0f + sj.w * v3;
                    }
                    float* sp = slot_part + kk * 128;
                    sp[0] = v0; sp[1] = v1; sp[2] = v2; sp[3] = v3;
                }
            __syncthreads();
            if (mine && wi == 0)
            {
                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                for (int kk = 0; kk < p.topk; ++kk)
                {
                    const float* q = slot_part + kk * 128;
                    o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
                }
                if (p.sh_out)
                {
                    const float gv = p.sh_gate_w ? sh_gate : 1.0f;
                    const float* sh = p.sh_out + (size_t) row * p.H + col;
                    if (col + 0 < p.H) o0 += gv * sh[0];
                    if (col + 1 < p.H) o1 += gv * sh[1];
                    if (col + 2 < p.H) o2 += gv * sh[2];
                    if (col + 3 < p.H) o3 += gv * sh[3];
                }
                float* dst = p.out + (size_t) row * p.out_stride + col;
                if (col + 3 < p.H_out)
                    *((float4*) dst) = make_float4(o0, o1, o2, o3);
                else
                {
                    if (col + 0 < p.H_out) dst[0] = o0;
                    if (col + 1 < p.H_out) dst[1] = o1;
                    if (col + 2 < p.H_out) dst[2] = o2;
                }
            }
            __syncthreads();
        }
    }
    else
"""
kb = kb.replace(loop, new + loop)
k = k[:b0] + kb
l = rep(l, "    const bool wide_b = moe_coop_pick_wide(p.I / 16, slots, device) && p.Ho % 512 == 0;",
        "    const bool wide_b = moe_coop_pick_wide(p.I / 16, slots, device) && p.Ho % 512 == 0 && p.topk <= 8;  // + down epilogue")
open(kp, "w").write(k)
open(lp, "w").write(l)
print("patched", kp, lp)
