// Cut-through ring all-reduce microbench, 4 GPUs in one process over P2P (bf16 [M, 5120], the TP4 decode message).
// Each 16-byte pack carries its own arrival signal: senders turn +0.0 halves into -0.0, so an all-zero half means
// "not written yet" and receivers spin on the data itself (no flags, no barriers). Reduce-scatter and all-gather run
// as one 2*(W-1)-hop chain per pack around the ring rank -> rank+1, stage-major so every pack is in flight at once.
// Slots are double-buffered by epoch (call parity); a receiver zeroes each slot right after reading it.
// build: nvcc -O3 -std=c++17 -arch=sm_80 bench/ring_ar.cu -o ring_ar; usage: ./ring_ar [iters=200] (server stopped)
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t ck_e = (x);                                                               \
    if (ck_e != cudaSuccess) {                                                          \
      printf("%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(ck_e));         \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

constexpr int W = 4;
constexpr int HID = 5120;
constexpr int SLOTS = 2 * (W - 1);

__device__ __forceinline__ uint4 ld_vol(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ void st_vol(uint4* p, uint4 v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}
__device__ __forceinline__ uint32_t fixz(uint32_t x) {
  uint32_t lo = x & 0xFFFFu, hi = x >> 16;
  lo = lo ? lo : 0x8000u;
  hi = hi ? hi : 0x8000u;
  return lo | (hi << 16);
}
__device__ __forceinline__ uint4 enc(uint4 v) { return {fixz(v.x), fixz(v.y), fixz(v.z), fixz(v.w)}; }
__device__ __forceinline__ bool z2(uint32_t x) { return (x & 0xFFFFu) == 0 || (x >> 16) == 0; }
__device__ __forceinline__ bool hasz(uint4 v) { return z2(v.x) | z2(v.y) | z2(v.z) | z2(v.w); }
__device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b) {
  __nv_bfloat162 r = __hadd2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
  return *reinterpret_cast<uint32_t*>(&r);
}
__device__ __forceinline__ uint4 add4(uint4 a, uint4 b) {
  return {add2(a.x, b.x), add2(a.y, b.y), add2(a.z, b.z), add2(a.w, b.w)};
}
__device__ __forceinline__ uint4 spin(uint4* p) {
  uint4 v;
  do {
    v = ld_vol(p);
  } while (hasz(v));
  st_vol(p, uint4{0u, 0u, 0u, 0u});
  return v;
}

// in/out: [W chunks][chunk packs]; local/remote: [2 epochs][SLOTS][chunk]; remote = the next rank's local buffer
__global__ void __launch_bounds__(1024) ring_ar(const uint4* __restrict__ in, uint4* __restrict__ out, uint4* local,
                                                uint4* remote, int rank, int chunk, int epoch) {
  uint4* L = local + (size_t)epoch * SLOTS * chunk;
  uint4* R = remote + (size_t)epoch * SLOTS * chunk;
  const int t0 = blockIdx.x * blockDim.x + threadIdx.x, st = gridDim.x * blockDim.x;
  // reduce-scatter: at step s this rank forwards the partial of chunk (rank - 1 - s)
  for (int i = t0; i < chunk; i += st) st_vol(R + i, enc(in[((rank + W - 1) % W) * chunk + i]));
#pragma unroll
  for (int s = 1; s < W - 1; ++s) {
    const int c = (rank + 2 * W - 1 - s) % W;
    for (int i = t0; i < chunk; i += st)
      st_vol(R + s * chunk + i, enc(add4(spin(L + (s - 1) * chunk + i), in[c * chunk + i])));
  }
  for (int i = t0; i < chunk; i += st) {
    uint4 v = add4(spin(L + (W - 2) * chunk + i), in[rank * chunk + i]);
    out[rank * chunk + i] = v;
    st_vol(R + (W - 1) * chunk + i, enc(v));
  }
  // all-gather: at step t this rank receives the final chunk (rank - 1 - t) and forwards it unless it is the last hop
#pragma unroll
  for (int t = 0; t < W - 1; ++t) {
    const int c = (rank + 2 * W - 1 - t) % W;
    for (int i = t0; i < chunk; i += st) {
      uint4 v = spin(L + (W - 1 + t) * chunk + i);
      out[c * chunk + i] = v;
      if (t < W - 2) st_vol(R + (W + t) * chunk + i, v);
    }
  }
}

__global__ void nop() {}

static uint16_t f2bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)(u >> 16);  // exact for small integers
}
static float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 200;
  for (int a = 0; a < W; ++a) {
    CK(cudaSetDevice(a));
    for (int b = 0; b < W; ++b)
      if (a != b) {
        int ok;
        CK(cudaDeviceCanAccessPeer(&ok, a, b));
        if (!ok) {
          printf("no P2P %d->%d\n", a, b);
          return 1;
        }
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
      }
  }
  const int MAXM = 48;
  const size_t maxpk = (size_t)MAXM * HID * 2 / 16;
  uint4 *in[W], *out[W], *slot[W];
  cudaStream_t s[W];
  for (int d = 0; d < W; ++d) {
    CK(cudaSetDevice(d));
    CK(cudaMalloc(&in[d], maxpk * 16));
    CK(cudaMalloc(&out[d], maxpk * 16));
    CK(cudaMalloc(&slot[d], 2 * SLOTS * (maxpk / W) * 16));
    CK(cudaMemset(slot[d], 0, 2 * SLOTS * (maxpk / W) * 16));
    CK(cudaStreamCreateWithFlags(&s[d], cudaStreamNonBlocking));
  }
  // graph-node overhead reference: iters empty kernels
  {
    cudaGraphExec_t ge[W];
    for (int d = 0; d < W; ++d) {
      CK(cudaSetDevice(d));
      cudaGraph_t g;
      CK(cudaStreamBeginCapture(s[d], cudaStreamCaptureModeThreadLocal));
      for (int it = 0; it < iters; ++it) nop<<<1, 32, 0, s[d]>>>();
      CK(cudaStreamEndCapture(s[d], &g));
      CK(cudaGraphInstantiate(&ge[d], g, 0));
    }
    cudaEvent_t e0, e1;
    CK(cudaSetDevice(0));
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    float best = 1e9;
    for (int r = 0; r < 5; ++r) {
      CK(cudaEventRecord(e0, s[0]));
      CK(cudaGraphLaunch(ge[0], s[0]));
      CK(cudaEventRecord(e1, s[0]));
      CK(cudaEventSynchronize(e1));
      float ms;
      CK(cudaEventElapsedTime(&ms, e0, e1));
      best = ms < best ? ms : best;
    }
    printf("empty graph node: %.2f us\n", best * 1000 / iters);
  }
  const int Ms[] = {1, 2, 4, 6, 8, 12, 24, 48};
  const int cfg[][2] = {{1, 256}, {1, 512}, {1, 1024}, {2, 256}, {2, 512}, {4, 256}, {4, 512}, {8, 256}, {16, 128}};
  int epoch_ctr = 0;
  for (int M : Ms) {
    const int packs = M * HID * 2 / 16, chunk = packs / W, numel = M * HID;
    std::vector<std::vector<uint16_t>> h(W, std::vector<uint16_t>(numel));
    std::vector<float> ref(numel, 0.f);
    srand(M);
    for (int d = 0; d < W; ++d)
      for (int j = 0; j < numel; ++j) {
        int v = rand() % 17 - 8;
        h[d][j] = f2bf((float)v);
        ref[j] += v;
      }
    for (int d = 0; d < W; ++d) {
      CK(cudaSetDevice(d));
      CK(cudaMemcpy(in[d], h[d].data(), numel * 2, cudaMemcpyHostToDevice));
    }
    for (auto& c : cfg) {
      const int B = c[0], T = c[1];
      // correctness: one call each, alternating epochs to exercise both
      for (int rep = 0; rep < 2; ++rep) {
        for (int d = 0; d < W; ++d) {
          CK(cudaSetDevice(d));
          CK(cudaMemsetAsync(out[d], 0xFF, numel * 2, s[d]));
          ring_ar<<<B, T, 0, s[d]>>>(in[d], out[d], slot[d], slot[(d + 1) % W], d, chunk, epoch_ctr & 1);
        }
        ++epoch_ctr;
        for (int d = 0; d < W; ++d) CK(cudaStreamSynchronize(s[d]));
      }
      int bad = 0;
      for (int d = 0; d < W; ++d) {
        std::vector<uint16_t> o(numel);
        CK(cudaMemcpy(o.data(), out[d], numel * 2, cudaMemcpyDeviceToHost));
        for (int j = 0; j < numel; ++j) bad += bf2f(o[j]) != ref[j];
      }
      // timing: iters calls per device in one graph each, replayed together
      cudaGraphExec_t ge[W];
      for (int d = 0; d < W; ++d) {
        CK(cudaSetDevice(d));
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s[d], cudaStreamCaptureModeThreadLocal));
        for (int it = 0; it < iters; ++it)
          ring_ar<<<B, T, 0, s[d]>>>(in[d], out[d], slot[d], slot[(d + 1) % W], d, chunk, (epoch_ctr + it) & 1);
        CK(cudaStreamEndCapture(s[d], &g));
        CK(cudaGraphInstantiate(&ge[d], g, 0));
        CK(cudaGraphDestroy(g));
      }
      cudaEvent_t e0[W], e1[W];
      for (int d = 0; d < W; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaEventCreate(&e0[d]));
        CK(cudaEventCreate(&e1[d]));
      }
      float best = 1e9;
      for (int r = 0; r < 5; ++r) {
        for (int d = 0; d < W; ++d) {
          CK(cudaSetDevice(d));
          CK(cudaEventRecord(e0[d], s[d]));
          CK(cudaGraphLaunch(ge[d], s[d]));
          CK(cudaEventRecord(e1[d], s[d]));
        }
        epoch_ctr += iters;
        float worst = 0;
        for (int d = 0; d < W; ++d) {
          CK(cudaEventSynchronize(e1[d]));
          float ms;
          CK(cudaEventElapsedTime(&ms, e0[d], e1[d]));
          worst = ms > worst ? ms : worst;
        }
        best = worst < best ? worst : best;
      }
      printf("M=%2d (%6d B) blocks=%2d threads=%4d: %7.2f us/all-reduce  %s\n", M, numel * 2, B, T,
             best * 1000 / iters, bad ? "WRONG" : "exact");
      for (int d = 0; d < W; ++d) {
        CK(cudaSetDevice(d));
        CK(cudaGraphExecDestroy(ge[d]));
        CK(cudaEventDestroy(e0[d]));
        CK(cudaEventDestroy(e1[d]));
      }
    }
  }
  return 0;
}
