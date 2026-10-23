// dsv41 MXFP8 (e4m3 weights, e8m0 scales per row and 32 k) linear kernels for sm_80 on one weight copy in a
// fragment-native layout (ds41-patches/dsv41_fp8w, patch_fp8_gemv_cuda.py).
//
// Layout: per (16-row tile r, 64-k chunk c) 1 KiB = [row half h][lane 32][16 B]; lane (g, t) holds row 16 r + 8 h + g,
// k 64 c + 16 t .. + 15, each 4-byte group stored as (k0, k2, k1, k3). The decode below turns a group into bf16
// pairs (k0, k1) and (k2, k3): the A fragment of mma.m16n8k16 with the weights as the 16-row operand (slots 2t, 2t+1
// <- k0, k1; 2t+8, 2t+9 <- k2, k3), so the token operand (n8) is 4 consecutive activations per step.
// e4m3 -> bf16 is exact: bit move, then x 2^(e - 7) (e = e8m0 scale, a power of two) in one bf16x2 multiply. Scale
// bytes are stored as e + 120 (the bf16 exponent of that factor), one 32-bit word per (tile, chunk, g): rows g / g + 8
// x the chunk's two 32-k blocks. All sums are fp32 in a fixed order: deterministic.
//   gemv_small: M <= 8, K split over the warps of a block, summed in smem.
//   gemm_mid:   8 < M <= 64, 8 warps on 8 x TW tiles share activations staged in smem (cp.async, double buffered);
//               K split over KS blocks, fp32 partials summed by the last block in order.
//   dequant:    the exact bf16 [N, K] weight, for a cuBLAS GEMM at large M.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

template <int HINT>
__device__ __forceinline__ uint4 ld_w(const uint4* p)
{
    uint4 v;
    if constexpr (HINT == 0)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    else
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint4 ld_x(const uint4* p)
{
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t bmul(uint32_t a, uint32_t b)
{
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&a), *reinterpret_cast<const __nv_bfloat162*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

// stored (k0, k2, k1, k3) e4m3 -> bf16 pairs (k0, k1), (k2, k3), times mult
__device__ __forceinline__ void dq4(uint32_t q, uint32_t mult, uint32_t& lo, uint32_t& hi)
{
    const uint32_t h = ((q & 0x7F007F00u) >> 4) | (q & 0x80008000u);
    const uint32_t u = q << 8;
    const uint32_t l = ((u & 0x7F007F00u) >> 4) | (u & 0x80008000u);
    lo = bmul(l, mult);
    hi = bmul(h, mult);
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int i)
{
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t mult_of(uint32_t sw, uint32_t sel)   // bf16x2 of 2^(e - 7)
{
    return __byte_perm(sw, 0, sel) * 0x00800080u;
}

// ------------------------------------------------------------------------------------------------ small M (<= 8)

template <int TW, int DEPTH, int HINT>
__global__ void gemv_small_kernel(const uint4* __restrict__ W, const uint32_t* __restrict__ SC,
                                  const __nv_bfloat16* __restrict__ x, int ldx, int M,
                                  __nv_bfloat16* __restrict__ out, int ldo, int N, int K, int WPT, int TPB)
{
    extern __shared__ float red[];   // [warps][TW][4][32]
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ts = warp / WPT, kp = warp % WPT;
    const int nch = K >> 6, ntile = N >> 4;
    const int r0 = (blockIdx.x * TPB + ts) * TW;
    const int c_lo = kp * nch / WPT, c_hi = (kp + 1) * nch / WPT;
    const int g = lane >> 2, t = lane & 3;
    // token rows >= M read row M - 1: their columns are computed and never stored
    const uint4* xp = reinterpret_cast<const uint4*>(x + (size_t) min(g, M - 1) * ldx) + 2 * t;

    float acc[TW][4];
    #pragma unroll
    for (int i = 0; i < TW; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;

    const uint4* wp[TW];
    const uint32_t* sp[TW];
    #pragma unroll
    for (int i = 0; i < TW; ++i)
    {
        const int r = min(r0 + i, ntile - 1);
        wp[i] = W + (size_t) r * nch * 64 + lane;
        sp[i] = SC + (size_t) r * nch * 8 + g;
    }
    const uint32_t sel_g = 0x4440u | (t >> 1), sel_g8 = 0x4440u | (2 + (t >> 1));

    auto load = [&] (int c, uint4 (&w)[TW][2], uint4 (&xa)[2], uint32_t (&sw)[TW])
    {
        #pragma unroll
        for (int i = 0; i < TW; ++i)
        {
            w[i][0] = ld_w<HINT>(wp[i] + c * 64);
            w[i][1] = ld_w<HINT>(wp[i] + c * 64 + 32);
            sw[i] = __ldg(sp[i] + c * 8);
        }
        xa[0] = ld_x(xp + 8 * c);
        xa[1] = ld_x(xp + 8 * c + 1);
    };
    auto compute = [&] (const uint4 (&wc)[TW][2], const uint4 (&xc)[2], const uint32_t (&sw)[TW])
    {
        uint32_t mg[TW], mg8[TW];
        #pragma unroll
        for (int i = 0; i < TW; ++i) { mg[i] = mult_of(sw[i], sel_g); mg8[i] = mult_of(sw[i], sel_g8); }
        #pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const uint32_t b0 = comp(xc[s >> 1], (s & 1) * 2), b1 = comp(xc[s >> 1], (s & 1) * 2 + 1);
            #pragma unroll
            for (int i = 0; i < TW; ++i)
            {
                uint32_t a0, a1, a2, a3;
                dq4(comp(wc[i][0], s), mg[i], a0, a2);
                dq4(comp(wc[i][1], s), mg8[i], a1, a3);
                mma_bf16(acc[i], a0, a1, a2, a3, b0, b1);
            }
        }
    };

    if (r0 < ntile && c_lo < c_hi)
    {
        uint4 wr[DEPTH][TW][2], xr[DEPTH][2];
        uint32_t sr[DEPTH][TW];
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d)
            if (c_lo + d < c_hi) load(c_lo + d, wr[d], xr[d], sr[d]);
        int c = c_lo;
        for (; c + DEPTH <= c_hi; c += DEPTH)
        {
            #pragma unroll
            for (int d = 0; d < DEPTH; ++d)
            {
                compute(wr[d], xr[d], sr[d]);
                if (c + d + DEPTH < c_hi) load(c + d + DEPTH, wr[d], xr[d], sr[d]);
            }
        }
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d)
            if (c + d < c_hi) compute(wr[d], xr[d], sr[d]);
    }

    #pragma unroll
    for (int i = 0; i < TW; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) red[((warp * TW + i) * 4 + j) * 32 + lane] = acc[i][j];
    __syncthreads();
    if (kp != 0) return;
    #pragma unroll
    for (int i = 0; i < TW; ++i)
    {
        const int r = r0 + i;
        if (r >= ntile) break;
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int w = 0; w < WPT; ++w)
            #pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += red[(((warp + w) * TW + i) * 4 + j) * 32 + lane];
        const int n = r * 16 + g;
        const int m0 = 2 * t, m1 = 2 * t + 1;
        if (m0 < M) { out[(size_t) m0 * ldo + n] = __float2bfloat16_rn(v[0]); out[(size_t) m0 * ldo + n + 8] = __float2bfloat16_rn(v[2]); }
        if (m1 < M) { out[(size_t) m1 * ldo + n] = __float2bfloat16_rn(v[1]); out[(size_t) m1 * ldo + n + 8] = __float2bfloat16_rn(v[3]); }
    }
}

// ------------------------------------------------------------------------------------------------ mid M (9..64)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

template <int NT, int TW, int DEPTH, int WARPS>
__global__ __launch_bounds__(WARPS * 32, 8 / WARPS * 2)   // <= 128 registers: two 8-warp (four 4-warp) blocks per SM
void gemm_mid_kernel(const uint4* __restrict__ W, const uint32_t* __restrict__ SC,
                     const __nv_bfloat16* __restrict__ x, int ldx, int M,
                     __nv_bfloat16* __restrict__ out, int ldo, int N, int K, int KS, int PC,
                     float* __restrict__ part, int* __restrict__ cnt)
{
    constexpr int ROWS = NT * 8;
    extern __shared__ uint4 xs[];   // 2 x [ROWS][8 PC + 1] uint4 (stride = 16 B mod 128: conflict-free LDS.128)
    __shared__ int sh_last;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int nch = K >> 6, ntile = N >> 4;
    const int rb = blockIdx.x / KS, ks = blockIdx.x % KS;
    const int r0 = (rb * WARPS + warp) * TW;
    const int c_lo = ks * nch / KS, c_hi = (ks + 1) * nch / KS;
    const int xstride = 8 * PC + 1;
    const int buf_elems = ROWS * xstride;
    const bool active = r0 < ntile;

    float acc[TW][NT][4];
    #pragma unroll
    for (int i = 0; i < TW; ++i)
        #pragma unroll
        for (int n = 0; n < NT; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0.0f;

    const uint4* wp[TW];
    const uint32_t* sp[TW];
    #pragma unroll
    for (int i = 0; i < TW; ++i)
    {
        const int r = min(r0 + i, ntile - 1);
        wp[i] = W + (size_t) r * nch * 64 + lane;
        sp[i] = SC + (size_t) r * nch * 8 + g;
    }
    const uint32_t sel_g = 0x4440u | (t >> 1), sel_g8 = 0x4440u | (2 + (t >> 1));

    // stage activations of chunks [cs, cs + PC) (clipped to c_hi) for tokens < M into buffer b
    auto stage = [&] (int cs, int b)
    {
        const int nc = min(PC, c_hi - cs);
        uint4* dst = xs + b * buf_elems;
        const int per_row = nc * 8;
        for (int i = threadIdx.x; i < M * per_row; i += blockDim.x)
        {
            const int m = i / per_row, u = i % per_row;
            cp_async16(dst + m * xstride + u, reinterpret_cast<const uint4*>(x + (size_t) m * ldx + cs * 64) + u);
        }
        cp_commit();
    };
    auto load = [&] (int c, uint4 (&w)[TW][2], uint32_t (&sw)[TW])
    {
        #pragma unroll
        for (int i = 0; i < TW; ++i)
        {
            w[i][0] = ld_w<0>(wp[i] + c * 64);
            w[i][1] = ld_w<0>(wp[i] + c * 64 + 32);
            sw[i] = __ldg(sp[i] + c * 8);
        }
    };
    auto compute = [&] (const uint4 (&wc)[TW][2], const uint32_t (&sw)[TW], const uint4* xb, int cc)
    {
        uint32_t mg[TW], mg8[TW];
        #pragma unroll
        for (int i = 0; i < TW; ++i) { mg[i] = mult_of(sw[i], sel_g); mg8[i] = mult_of(sw[i], sel_g8); }
        #pragma unroll
        for (int p = 0; p < 2; ++p)   // step pairs (2p, 2p + 1)
        {
            uint32_t a[TW][2][4];
            #pragma unroll
            for (int i = 0; i < TW; ++i)
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    const int s = 2 * p + e;
                    dq4(comp(wc[i][0], s), mg[i], a[i][e][0], a[i][e][2]);
                    dq4(comp(wc[i][1], s), mg8[i], a[i][e][1], a[i][e][3]);
                }
            #pragma unroll
            for (int n = 0; n < NT; ++n)
            {
                const uint4 bv = xb[(8 * n + g) * xstride + 8 * cc + 2 * t + p];
                #pragma unroll
                for (int i = 0; i < TW; ++i)
                {
                    mma_bf16(acc[i][n], a[i][0][0], a[i][0][1], a[i][0][2], a[i][0][3], bv.x, bv.y);
                    mma_bf16(acc[i][n], a[i][1][0], a[i][1][1], a[i][1][2], a[i][1][3], bv.z, bv.w);
                }
            }
        }
    };

    // phases of PC chunks; weights stream through a DEPTH-deep register ring across phases
    uint4 wr[DEPTH][TW][2];
    uint32_t sr[DEPTH][TW];
    if (active)
    {
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d)
            if (c_lo + d < c_hi) load(c_lo + d, wr[d], sr[d]);
    }
    stage(c_lo, 0);
    int c = c_lo;
    int ph = 0;
    for (int cs = c_lo; cs < c_hi; cs += PC, ++ph)
    {
        if (cs + PC < c_hi) { stage(cs + PC, (ph + 1) & 1); cp_wait<1>(); }
        else cp_wait<0>();
        __syncthreads();
        const uint4* xb = xs + (ph & 1) * buf_elems;
        const int ce = min(cs + PC, c_hi);
        if (active)
        {
            // c runs through the phase in DEPTH-sized groups; the ring slot is (c - c_lo) % DEPTH
            for (; c < ce; ++c)
            {
                const int slot = (c - c_lo) % DEPTH;
                #pragma unroll
                for (int d = 0; d < DEPTH; ++d)
                    if (d == slot)
                    {
                        compute(wr[d], sr[d], xb, c - cs);
                        if (c + DEPTH < c_hi) load(c + DEPTH, wr[d], sr[d]);
                    }
            }
        }
        __syncthreads();   // the buffer is restaged two phases later
    }

    // outputs: D(row, token) = acc[i][n][q]: q0 (g, 2t), q1 (g, 2t + 1), q2 (g + 8, 2t), q3 (g + 8, 2t + 1)
    if (KS > 1)
    {
        float* pp = part + ((size_t) (rb * KS + ks) * WARPS + warp) * (TW * NT * 128) + lane;
        #pragma unroll
        for (int i = 0; i < TW; ++i)
            #pragma unroll
            for (int n = 0; n < NT; ++n)
                #pragma unroll
                for (int q = 0; q < 4; ++q) __stcg(pp + ((i * NT + n) * 4 + q) * 32, acc[i][n][q]);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) sh_last = atomicAdd(cnt + rb, 1) == KS - 1;
        __syncthreads();
        if (!sh_last) return;
        __threadfence();
        #pragma unroll
        for (int i = 0; i < TW; ++i)
            #pragma unroll
            for (int n = 0; n < NT; ++n) acc[i][n][0] = acc[i][n][1] = acc[i][n][2] = acc[i][n][3] = 0.0f;
        const float* p0 = part + ((size_t) rb * KS * WARPS + warp) * (TW * NT * 128) + lane;
        for (int s = 0; s < KS; ++s)   // splits in order
        {
            const float* ps = p0 + (size_t) s * WARPS * (TW * NT * 128);
            #pragma unroll
            for (int i = 0; i < TW; ++i)
                #pragma unroll
                for (int n = 0; n < NT; ++n)
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) acc[i][n][q] += __ldcg(ps + ((i * NT + n) * 4 + q) * 32);
        }
        if (threadIdx.x == 0) cnt[rb] = 0;
    }
    if (!active) return;
    #pragma unroll
    for (int i = 0; i < TW; ++i)
    {
        const int r = r0 + i;
        if (r >= ntile) break;
        const int nrow = r * 16 + g;
        #pragma unroll
        for (int n = 0; n < NT; ++n)
        {
            const int m0 = 8 * n + 2 * t, m1 = m0 + 1;
            if (m0 < M) { out[(size_t) m0 * ldo + nrow] = __float2bfloat16_rn(acc[i][n][0]); out[(size_t) m0 * ldo + nrow + 8] = __float2bfloat16_rn(acc[i][n][2]); }
            if (m1 < M) { out[(size_t) m1 * ldo + nrow] = __float2bfloat16_rn(acc[i][n][1]); out[(size_t) m1 * ldo + nrow + 8] = __float2bfloat16_rn(acc[i][n][3]); }
        }
    }
}

// ------------------------------------------------------------------------------------------------ dequant

__global__ void dequant_kernel(const uint4* __restrict__ W, const uint32_t* __restrict__ SC,
                               __nv_bfloat16* __restrict__ out, int N, int K)
{
    const int nch = K >> 6;
    const size_t total = (size_t) N * K / 16;
    for (size_t u = (size_t) blockIdx.x * blockDim.x + threadIdx.x; u < total; u += (size_t) gridDim.x * blockDim.x)
    {
        const int lane = u & 31, h = (u >> 5) & 1;
        const size_t rc = u >> 6;
        const int r = rc / nch, c = rc % nch;
        const int g = lane >> 2, t = lane & 3;
        const uint4 q = W[u];
        const uint32_t sw = SC[rc * 8 + g];
        const uint32_t m = mult_of(sw, 0x4440u | (2 * h + (t >> 1)));
        uint32_t o[8];
        dq4(q.x, m, o[0], o[1]);
        dq4(q.y, m, o[2], o[3]);
        dq4(q.z, m, o[4], o[5]);
        dq4(q.w, m, o[6], o[7]);
        uint4* dst = reinterpret_cast<uint4*>(out + (size_t) (16 * r + 8 * h + g) * K + 64 * c + 16 * t);
        dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
        dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
    }
}

void check_common(const torch::Tensor& W, const torch::Tensor& SC, int64_t N, int64_t K)
{
    TORCH_CHECK(K % 64 == 0 && N % 16 == 0, "dsv41 fp8w: shape");
    TORCH_CHECK(W.is_cuda() && W.is_contiguous() && W.numel() * W.element_size() == N * K, "dsv41 fp8w: weight");
    TORCH_CHECK(SC.is_contiguous() && SC.numel() * SC.element_size() == N * K / 32, "dsv41 fp8w: scales");
}

void check_x(const torch::Tensor& x, const torch::Tensor& out, int64_t K, int align)
{
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && out.dtype() == torch::kBFloat16 && x.dim() == 2 && out.dim() == 2
                && x.size(1) == K && x.stride(1) == 1 && out.stride(1) == 1 && x.stride(0) % 8 == 0
                && (reinterpret_cast<uintptr_t>(x.data_ptr()) & (align - 1)) == 0, "dsv41 fp8w: x layout");
}

}  // namespace

void gemv_small(torch::Tensor W, torch::Tensor SC, torch::Tensor x, torch::Tensor out, int64_t N, int64_t K,
                int64_t TW, int64_t WPT, int64_t TPB, int64_t DEPTH, int64_t HINT)
{
    check_common(W, SC, N, K);
    check_x(x, out, K, 16);
    const int M = x.size(0);
    TORCH_CHECK(M >= 1 && M <= 8 && WPT >= 1 && TPB >= 1 && WPT * TPB <= 32 && DEPTH >= 1 && DEPTH <= 3
                && (TW == 1 || TW == 2) && (HINT == 0 || HINT == 1), "dsv41 fp8w: small config");
    const at::cuda::CUDAGuard guard(x.device());
    const int ntile = N / 16, grid = (ntile + TW * TPB - 1) / (TW * TPB), threads = 32 * WPT * TPB;
    const int smem = threads / 32 * TW * 4 * 32 * 4;
    auto stream = at::cuda::getCurrentCUDAStream();
#define L(T, D, H) gemv_small_kernel<T, D, H><<<grid, threads, smem, stream>>>((const uint4*) W.data_ptr(), \
        (const uint32_t*) SC.data_ptr(), (const __nv_bfloat16*) x.data_ptr(), x.stride(0), M, \
        (__nv_bfloat16*) out.data_ptr(), out.stride(0), N, K, WPT, TPB)
#define LH(T, D) { if (HINT == 0) L(T, D, 0); else L(T, D, 1); }
#define LD(T) { if (DEPTH == 1) LH(T, 1) else if (DEPTH == 2) LH(T, 2) else LH(T, 3) }
    if (TW == 1) LD(1) else LD(2)
#undef LD
#undef LH
#undef L
}

void gemm_mid(torch::Tensor W, torch::Tensor SC, torch::Tensor x, torch::Tensor out, torch::Tensor part, torch::Tensor cnt,
              int64_t N, int64_t K, int64_t TW, int64_t KS, int64_t PC, int64_t DEPTH, int64_t WARPS)
{
    check_common(W, SC, N, K);
    check_x(x, out, K, 16);
    const int M = x.size(0);
    TORCH_CHECK(WARPS == 4 || WARPS == 8, "dsv41 fp8w: mid warps");
    TORCH_CHECK(M >= 1 && M <= 64 && (TW == 1 || TW == 2) && KS >= 1 && KS <= K / 64 && PC >= 1 && PC <= 8
                && DEPTH >= 1 && DEPTH <= 3, "dsv41 fp8w: mid config");
    const int NT = (M + 7) / 8;
    const int rbs = (N / 16 + WARPS * TW - 1) / (WARPS * TW);
    TORCH_CHECK(KS == 1 || (part.numel() >= (int64_t) rbs * KS * WARPS * TW * NT * 128 && cnt.numel() >= rbs),
                "dsv41 fp8w: mid scratch");
    const at::cuda::CUDAGuard guard(x.device());
    const int grid = rbs * KS, smem = 2 * NT * 8 * (8 * PC + 1) * 16;
    auto stream = at::cuda::getCurrentCUDAStream();
    float* pp = KS > 1 ? part.data_ptr<float>() : nullptr;
    int* cp = KS > 1 ? cnt.data_ptr<int>() : nullptr;
    auto launch = [&] (auto kf)
    {
        if (smem > 48 * 1024) cudaFuncSetAttribute(kf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        kf<<<grid, WARPS * 32, smem, stream>>>((const uint4*) W.data_ptr(), (const uint32_t*) SC.data_ptr(),
                                        (const __nv_bfloat16*) x.data_ptr(), x.stride(0), M,
                                        (__nv_bfloat16*) out.data_ptr(), out.stride(0), N, K, (int) KS, (int) PC, pp, cp);
    };
#define LW(NTV, T, D) { if (WARPS == 4) launch(gemm_mid_kernel<NTV, T, D, 4>); else launch(gemm_mid_kernel<NTV, T, D, 8>); }
#define LT(NTV, T) { if (DEPTH == 1) LW(NTV, T, 1) else if (DEPTH == 2) LW(NTV, T, 2) else LW(NTV, T, 3) }
#define LN(NTV) { if (TW == 1) LT(NTV, 1) else LT(NTV, 2) }
    switch (NT)
    {
    case 1: LN(1) break; case 2: LN(2) break; case 3: LN(3) break; case 4: LN(4) break;
    case 5: LN(5) break; case 6: LN(6) break; case 7: LN(7) break; default: LN(8) break;
    }
#undef LN
#undef LT
#undef LW
}

void dequant(torch::Tensor W, torch::Tensor SC, torch::Tensor out, int64_t N, int64_t K)
{
    check_common(W, SC, N, K);
    TORCH_CHECK(out.dtype() == torch::kBFloat16 && out.is_contiguous() && out.numel() == N * K, "dsv41 fp8w: dequant out");
    const at::cuda::CUDAGuard guard(W.device());
    const size_t total = (size_t) N * K / 16;
    const int grid = (int) ((total + 255) / 256);   // one 16-byte group per thread
    dequant_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*) W.data_ptr(),
        (const uint32_t*) SC.data_ptr(), (__nv_bfloat16*) out.data_ptr(), N, K);
}
