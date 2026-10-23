# dsv41: small-M FP8 (e4m3fn, 128x128 block scales) W8A16 GEMM for sm80.
#
# vLLM's sm80 path is Marlin, which reaches ~0.55-0.7 TB/s at decode batch sizes and
# rounds the checkpoint's fp32 block scales to bf16 (~3e-3 relative output error).
# This kernel decodes e4m3 to bf16 exactly with Marlin's packed bit trick (inline PTX,
# 4 codes per 32-bit register, one bf16x2 FMA by 2^120), multiplies on bf16 tensor
# cores with fp32 accumulation, and applies the fp32 block scales per 128-wide K
# block. Split-K partials are summed in fixed order, so results are deterministic.
import torch
import triton
import triton.language as tl


@triton.jit
def _e4m3_to_bf16(w_u8):
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b32 hi, lo, t, bias, nz;
        mov.b32 bias, 0x7B807B80;
        mov.b32 nz, 0x80008000;
        and.b32 t, $2, 0x80008000;
        and.b32 hi, $2, 0x7F007F00;
        shr.b32 hi, hi, 4;
        or.b32 hi, hi, t;
        shl.b32 lo, $2, 8;
        and.b32 t, lo, 0x80008000;
        and.b32 lo, lo, 0x7F007F00;
        shr.b32 lo, lo, 4;
        or.b32 lo, lo, t;
        prmt.b32 $0, lo, hi, 0x5410;
        prmt.b32 $1, lo, hi, 0x7632;
        fma.rn.bf16x2 $0, $0, bias, nz;
        fma.rn.bf16x2 $1, $1, bias, nz;
        }""",
        constraints="=r,=r,r",
        args=[w_u8],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=4,
    )


@triton.jit(do_not_specialize=["M"])
def _dsv41_fp8_gemm(x_ptr, w_ptr, s_ptr, o_ptr, p_ptr, M, N, K, stride_xm, stride_wn, stride_sn, stride_om,
                    KSPLIT_LEN, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr,
                    EVEN_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    nmask = offs_n < N
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * KSPLIT_LEN
    for k0 in range(k_lo, k_lo + KSPLIT_LEN, 128):
        part = tl.zeros((BM, BN), dtype=tl.float32)
        for kk in tl.static_range(0, 128, BK):
            k = k0 + kk + offs_k
            xv = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :], mask=mmask[:, None], other=0.0)
            if EVEN_N:
                wu = tl.load(w_ptr + offs_n[None, :] * stride_wn + k[:, None])
            else:
                wu = tl.load(w_ptr + offs_n[None, :] * stride_wn + k[:, None], mask=nmask[None, :], other=0)
            part = tl.dot(xv, _e4m3_to_bf16(wu), part)
        if EVEN_N:
            sc = tl.load(s_ptr + (offs_n // 128) * stride_sn + k0 // 128)
        else:
            sc = tl.load(s_ptr + (offs_n // 128) * stride_sn + k0 // 128, mask=nmask, other=0.0)
        acc += part * sc[None, :]
    omask = mmask[:, None] & nmask[None, :]
    if SPLIT == 1:
        tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(o_ptr.dtype.element_ty), mask=omask)
    else:
        tl.store(p_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=omask)


@triton.jit(do_not_specialize=["MN"])
def _dsv41_fp8_reduce(p_ptr, o_ptr, MN, N, stride_om, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    m = offs // N
    n = offs % N
    tl.store(o_ptr + m * stride_om + n, acc.to(o_ptr.dtype.element_ty), mask=mask)


# (N, K) -> (BN, BK, SPLIT, warps, stages), tuned at M=6 on the CMP 170HX.
_TUNED = {
    (1792, 5120): (128, 128, 5, 8, 3),
    (8192, 1280): (128, 128, 1, 8, 3),
    (5120, 2048): (32, 128, 1, 4, 3),
    (2048, 4096): (128, 128, 4, 8, 3),
    (5120, 896): (32, 128, 1, 4, 3),
}
MAX_M = 16


def _config(N, K):
    cfg = _TUNED.get((N, K))
    if cfg is not None:
        return cfg
    BN = 128 if N >= 4096 else 64
    progs = triton.cdiv(N, BN)
    split = 1
    for s in (1, 2, 4, 5, 8, 10, 16, 20, 40):
        if K % (s * 128) == 0:
            split = s
            if progs * s >= 148:
                break
    return (BN, 128, split, 8 if BN == 128 else 4, 3)


def supports(x: torch.Tensor, w_u8: torch.Tensor, scale: torch.Tensor) -> bool:
    return (x.dim() == 2 and 0 < x.shape[0] <= MAX_M and x.dtype == torch.bfloat16 and x.stride(1) == 1
            and w_u8.shape[1] == x.shape[1] and w_u8.shape[1] % 128 == 0
            and scale.shape == (triton.cdiv(w_u8.shape[0], 128), w_u8.shape[1] // 128))


def fp8_linear(x: torch.Tensor, w_u8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = w_u8.shape[0]
    BN, BK, SPLIT, warps, stages = _config(N, K)
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)
    part = torch.empty(SPLIT, M, N, dtype=torch.float32, device=x.device) if SPLIT > 1 else out
    _dsv41_fp8_gemm[(triton.cdiv(N, BN), SPLIT)](
        x, w_u8, scale, out, part, M, N, K, x.stride(0), w_u8.stride(0), scale.stride(0), out.stride(0),
        K // SPLIT, BM=16, BN=BN, BK=BK, SPLIT=SPLIT, EVEN_N=N % BN == 0, num_warps=warps, num_stages=stages)
    if SPLIT > 1:
        _dsv41_fp8_reduce[(triton.cdiv(M * N, 1024),)](part, out, M * N, N, out.stride(0), SPLIT=SPLIT, BLOCK=1024)
    return out
