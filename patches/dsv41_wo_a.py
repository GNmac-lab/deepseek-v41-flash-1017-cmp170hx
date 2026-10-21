# dsv41: wo_a (grouped attention output projection) as an FP8 W8A16 Triton GEMM for small token counts on sm80.
#
# The Ampere path dequantizes the MXFP8 wo_a to bf16 at load and runs einsum("tgd,grd->tgr"): a batched cuBLAS
# GEMM over 16.8 MB of bf16 per rank, written in [g, t, r] order and then copied to [t, g*r] for wo_b. Here the
# FP8 codes are recovered exactly from the bf16 weight (power-of-two 32x32 block scales: w = q * 2^e, verified per
# layer), decoded to bf16 in registers with Marlin's packed bit trick, multiplied on bf16 tensor cores with fp32
# accumulation, scaled per 32x32 block, and written straight into the contiguous [t, g*r] layout wo_b reads.
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
def _wo_a_kernel(x_ptr, w_ptr, s_ptr, o_ptr, p_ptr, M, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                 KSPLIT_LEN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr,
                 BK: tl.constexpr):
    # x: [M, G, D] bf16; w: [G * R, D] e4m3 bytes; s: [G, R / 32, D / 32] fp32; out: [M, G * R]
    pid_n = tl.program_id(0)          # over G * R / BN
    pid_k = tl.program_id(1)
    nb = R // BN
    g = pid_n // nb
    offs_n = (pid_n % nb) * BN + tl.arange(0, BN)      # within the group
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    mmask = offs_m < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_lo = pid_k * KSPLIT_LEN
    nb32 = (pid_n % nb) * (BN // 32) + tl.arange(0, BN // 32)
    for k0 in range(k_lo, k_lo + KSPLIT_LEN, BK):
        k = k0 + offs_k
        xv = tl.load(x_ptr + offs_m[:, None] * (G * D) + g * D + k[None, :], mask=mmask[:, None], other=0.0)
        wu = tl.load(w_ptr + (g * R + offs_n[None, :]) * D + k[:, None])
        # power-of-two block scales: w * 2^e is exact in bf16, so scale the tile before one BK-deep dot
        kb = k0 // 32 + tl.arange(0, BK // 32)
        sc = tl.load(s_ptr + (g * (R // 32) + nb32[None, :]) * (D // 32) + kb[:, None])        # [BK/32, BN/32]
        sc = tl.reshape(tl.broadcast_to(sc[:, None, :, None], (BK // 32, 32, BN // 32, 32)), (BK, BN))
        acc = tl.dot(xv, _e4m3_to_bf16(wu) * sc.to(tl.bfloat16), acc)
    if SPLIT == 1:
        tl.store(o_ptr + offs_m[:, None] * (G * R) + g * R + offs_n[None, :], acc.to(o_ptr.dtype.element_ty),
                 mask=mmask[:, None])
    else:
        tl.store(p_ptr + pid_k * M * (G * R) + offs_m[:, None] * (G * R) + g * R + offs_n[None, :], acc,
                 mask=mmask[:, None])


@triton.jit(do_not_specialize=["MN"])
def _reduce(p_ptr, o_ptr, MN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, acc.to(o_ptr.dtype.element_ty), mask=mask)


MAX_M = 16
_CFG = {"BN": 64, "SPLIT": 2, "warps": 4, "stages": 3, "BK": 256}


def recover_fp8(w_bf16: torch.Tensor):
    """[G, R, D] bf16 -> (e4m3 bytes [G*R, D], fp32 scales [G, R/32, D/32]) if w is exactly q * 2^e per 32x32
    block, else None."""
    G, R, D = w_bf16.shape
    if R % 32 or D % 32:
        return None
    wb = w_bf16.float().view(G, R // 32, 32, D // 32, 32)
    amax = wb.abs().amax(dim=(2, 4)).clamp_min(2.0 ** -126)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (wb / s[:, :, None, :, None]).to(torch.float8_e4m3fn)
    back = (q.float() * s[:, :, None, :, None]).to(torch.bfloat16).view(G, R, D)
    if not torch.equal(back, w_bf16):
        return None
    return q.view(G * R, D).contiguous().view(torch.uint8), s.contiguous()


def wo_a(x: torch.Tensor, w_u8: torch.Tensor, scale: torch.Tensor, G: int, R: int, D: int) -> torch.Tensor:
    """x [M, G*D] bf16 (contiguous) -> [M, G*R] bf16."""
    M = x.shape[0]
    BN, SPLIT = _CFG["BN"], _CFG["SPLIT"]
    BK = _CFG["BK"]
    while D % (SPLIT * BK):
        SPLIT //= 2
    out = torch.empty(M, G * R, dtype=x.dtype, device=x.device)
    part = torch.empty(SPLIT, M, G * R, dtype=torch.float32, device=x.device) if SPLIT > 1 else out
    _wo_a_kernel[(G * R // BN, SPLIT)](x, w_u8, scale, out, part, M, G, R, D, D // SPLIT, BM=16, BN=BN, SPLIT=SPLIT, BK=BK,
                                       num_warps=_CFG["warps"], num_stages=_CFG["stages"])
    if SPLIT > 1:
        _reduce[(triton.cdiv(M * G * R, 1024),)](part, out, M * G * R, SPLIT=SPLIT, BLOCK=1024)
    return out
