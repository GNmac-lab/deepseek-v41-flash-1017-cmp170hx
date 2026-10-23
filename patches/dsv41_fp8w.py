# dsv41: MXFP8 (e4m3 weights, e8m0 scales per row and 32 k) linears on sm80 with ONE weight copy in a fragment-native
# layout, CUDA (dsv41_fp8w_ext). Marlin reaches ~0.6 TB/s at decode sizes; these kernels:
#   M <= 8       gemv_small: ~9.5 us on a 10 MB weight (Marlin 14-17).
#   9 <= M <= 64 gemm_mid:   activations staged in smem, K split over blocks; about Marlin's speed (8-user decode).
#   M > 64       dequant to the exact bf16 weight (transient) + cuBLAS with fp32 reductions (prefill).
# e4m3 -> bf16 is exact (bit move, x 2^(e - 7)); products are exact and every sum is fp32 in a fixed order:
# deterministic, no extra quantization; not bit-identical to Marlin (summation order).
# Layers over 32 MB keep Marlin only (it already streams them near the HBM rate). Layers up to 4 MB keep Marlin's
# copy too (small N/K shapes; costs ~50 MB per rank) and use it for M > 8. Everything else frees Marlin's weight.
# ds41-patches/patch_fp8_gemv_cuda.py
import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
SMALL_M, MID_M = 8, 64
MAX_BYTES = 32 << 20
DUAL_BYTES = 4 << 20
# validation only: DSV41_FP8_GEMV_ALL=1 runs every M through gemv_small (8 rows at a time)
_ALL = os.environ.get("DSV41_FP8_GEMV_ALL", "0") == "1"

# (N, K) -> gemv_small (tiles/warp, warps/tile, tiles/block, depth, L2 hint), tuned at M=6 on CMP 170HX
_SMALL = {
    (1792, 5120): (1, 16, 1, 2, 1),
    (8192, 1280): (1, 5, 1, 1, 0),
    (5120, 2048): (1, 8, 1, 1, 0),
    (5120, 896): (1, 7, 1, 1, 0),
    (2048, 5120): (1, 10, 1, 3, 1),
    (5120, 1024): (1, 4, 1, 2, 0),
    (256, 5120): (1, 16, 1, 3, 1),
    (5120, 128): (1, 2, 2, 1, 0),
    (1152, 5120): (1, 10, 1, 3, 1),
    (5120, 576): (1, 5, 1, 1, 0),
}
# (N, K, M <= 24) -> gemm_mid (tiles/warp, K split, phase chunks, depth, warps/block)
_MID = {
    (1792, 5120, True): (1, 10, 2, 2, 4), (1792, 5120, False): (1, 5, 4, 2, 8),
    (8192, 1280, True): (1, 2, 2, 2, 8), (8192, 1280, False): (1, 1, 4, 2, 8),
    (5120, 2048, True): (1, 4, 2, 2, 4), (5120, 2048, False): (1, 3, 4, 2, 8),
}
_seen: set = set()


def _small_cfg(N: int, K: int) -> tuple:
    cfg = _SMALL.get((N, K))
    if cfg is not None:
        return cfg
    tiles, nch = N // 16, K // 64
    if tiles <= 160:
        wpt = min(16, nch)
        return (1, wpt, 1, max(1, min(3, nch // wpt)), 0)
    return (1, max(1, min(16, nch // 4 if nch >= 8 else nch)), 1, 1, 0)


def _mid_cfg(N: int, K: int, M: int) -> tuple:
    cfg = _MID.get((N, K, M <= 24))
    if cfg is not None:
        return cfg
    warps = 8 if M > 24 else 4
    rbs = (N // 16 + warps - 1) // warps
    ks = max(1, min(K // 64 // 2, round(130 / rbs)))
    return (1, ks, 2 if M <= 24 else 4, 2, warps)


def prepare(w_u8: torch.Tensor, scale_u8: torch.Tensor):
    """(packed weight, packed scales, mid-kernel counters, keep_marlin) for an [N, K] e4m3 weight with [>= N, >= K/32]
    e8m0 scales, or None when the layer stays on Marlin."""
    N, K = w_u8.shape
    if (N % 16 or K % 64 or N * K > MAX_BYTES or scale_u8.dim() != 2 or scale_u8.shape[0] < N
            or scale_u8.shape[1] < K // 32):
        return None
    e = scale_u8[:N, : K // 32].to(torch.int32)
    if bool((e > 134).any()):   # 2^(e - 7) must stay a normal bf16 (e = 255 is NaN)
        return None
    w = w_u8.reshape(N // 16, 2, 8, K // 64, 4, 4, 4)[..., [0, 2, 1, 3]].permute(0, 3, 1, 2, 4, 5, 6).contiguous()
    b = (e + 120).to(torch.uint8)
    sc = b.reshape(N // 16, 2, 8, K // 64, 2).permute(0, 3, 2, 1, 4).reshape(-1, 4).view(torch.int32).contiguous()
    cnt = torch.zeros((N // 16 + 3) // 4, dtype=torch.int32, device=w_u8.device)
    keep_marlin = N * K <= DUAL_BYTES
    if (N, K) not in _seen:
        _seen.add((N, K))
        logger.info("dsv41 fp8 gemv: N=%d K=%d small %s, %s", N, K, _small_cfg(N, K),
                    "M > 8 on Marlin (kept)" if keep_marlin else f"mid {_mid_cfg(N, K, 16)}/{_mid_cfg(N, K, 48)}, M > 64 cuBLAS; Marlin copy freed")
    return w.view(-1), sc.view(-1), cnt, keep_marlin


def _small(x2, W, SC, N, K):
    import dsv41_fp8w_ext as _ext

    out = torch.empty(x2.shape[0], N, dtype=x2.dtype, device=x2.device)
    cfg = _small_cfg(N, K)
    for r in range(0, x2.shape[0], SMALL_M):
        _ext.gemv_small(W, SC, x2[r:r + SMALL_M], out[r:r + SMALL_M], N, K, *cfg)
    return out


def _fp8_linear(x: torch.Tensor, W: torch.Tensor, SC: torch.Tensor, cnt: torch.Tensor, marlin_w: torch.Tensor | None,
                marlin_s: torch.Tensor | None, workspace: torch.Tensor | None, size_n: int, size_k: int) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    M = x2.shape[0]
    if M == 0:
        return x.new_empty((*x.shape[:-1], size_n))
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:
        x2 = x2.contiguous()
    if M <= SMALL_M or _ALL:
        out = _small(x2, W, SC, size_n, size_k)
    elif marlin_w is not None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_mxfp8_marlin_linear

        return apply_mxfp8_marlin_linear(input=x, weight=marlin_w, weight_scale=marlin_s, workspace=workspace,
                                         size_n=size_n, size_k=size_k, bias=None)
    elif M <= MID_M:
        import dsv41_fp8w_ext as _ext

        tw, ks, pc, d, warps = _mid_cfg(size_n, size_k, M)
        nt = (M + 7) // 8
        rbs = (size_n // 16 + warps * tw - 1) // (warps * tw)
        part = torch.empty(rbs * ks * warps * tw * nt * 128 if ks > 1 else 1, dtype=torch.float32, device=x2.device)
        out = torch.empty(M, size_n, dtype=x2.dtype, device=x2.device)
        _ext.gemm_mid(W, SC, x2, out, part, cnt, size_n, size_k, tw, ks, pc, d, warps)
    else:
        import dsv41_fp8w_ext as _ext

        wb = torch.empty(size_n, size_k, dtype=torch.bfloat16, device=x2.device)
        _ext.dequant(W, SC, wb, size_n, size_k)
        flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False   # fp32 split-K sums only
        try:
            out = torch.matmul(x2, wb.t())
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    return out.view(*x.shape[:-1], size_n)


def _fp8_linear_fake(x, W, SC, cnt, marlin_w, marlin_s, workspace, size_n, size_k):
    return x.new_empty((*x.shape[:-1], size_n))


direct_register_custom_op("dsv41_fp8_linear", _fp8_linear, fake_impl=_fp8_linear_fake)


def linear(x: torch.Tensor, layer) -> torch.Tensor:
    W, SC, cnt = layer._dsv41_w8
    keep = getattr(layer, "_dsv41_keep_marlin", False)
    return torch.ops.vllm.dsv41_fp8_linear(x, W, SC, cnt, layer.weight if keep else None,
                                           layer.weight_scale if keep else None, layer.workspace if keep else None,
                                           layer.output_size_per_partition, layer.input_size_per_partition)
