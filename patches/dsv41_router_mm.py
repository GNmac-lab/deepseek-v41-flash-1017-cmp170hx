"""Small-batch MoE router GEMM for sm_80 (dsv41, ds41-patches/patch_router_mm.py).

x [M <= 64, K] bf16 @ W[N, K]^T -> fp32, as GateLinear's cuBLAS tier computes it, via a Triton split-K kernel with
a fixed-order reduction (deterministic). At DeepSeek-V4.1 decode sizes (M 6..36, K 5120, N 384) cuBLAS picks a
12-way split-K cutlass kernel plus a reduce (~22 us per layer in serving, next to the shared expert); this is
~6.4 us alone. fp32 accumulation like cuBLAS; results differ only in summation order."""
import torch
import triton
import triton.language as tl

_BN, _BK, _SPLIT, _WARPS = 64, 128, 8, 4


@triton.jit
def _router_partial(x_ptr, w_ptr, p_ptr, M, N, sxm, swn, KCHUNK,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * KCHUNK
    for k0 in range(k_lo, k_lo + KCHUNK, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None] * sxm + offs_k[None, :], mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * swn + offs_k[:, None], mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(p_ptr + (pid_k * BM + offs_m[:, None]) * N + offs_n[None, :], acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _router_reduce(p_ptr, o_ptr, M, N, BM: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs // N
    n = offs % N
    ok = offs < M * N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + (s * BM + m) * N + n, mask=ok, other=0.0)
    tl.store(o_ptr + offs, acc, mask=ok)


def supports(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (x.dim() == 2 and x.shape[0] <= 64 and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16
            and x.shape[1] % (_SPLIT * _BK) == 0 and x.stride(1) == 1 and w.stride(1) == 1)


def router_mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w.shape[0]
    BM = 16 if M <= 16 else (32 if M <= 32 else 64)
    part = torch.empty(_SPLIT * BM, N, dtype=torch.float32, device=x.device)
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    _router_partial[(triton.cdiv(N, _BN), _SPLIT)](x, w, part, M, N, x.stride(0), w.stride(0), K // _SPLIT,
                                                   BM=BM, BN=_BN, BK=_BK, num_warps=_WARPS)
    _router_reduce[(triton.cdiv(M * N, 512),)](part, out, M, N, BM=BM, SPLIT=_SPLIT, BLOCK=512)
    return out
