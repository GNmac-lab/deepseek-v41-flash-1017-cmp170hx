#!/usr/bin/env python3
"""dsv41: decode (M <= 8) wo_a as one CUDA kernel (dsv41_woa_ext) that reads the exact FP8 codes and applies the
inverse GPT-J RoPE in its activation load, replacing the inverse-RoPE Triton kernel + the FP8 Triton wo_a + its
split-K reduce (patch_v32_small.py path). Same codes and scales; fp32 sums in a different fixed order (not
bit-identical). M > 8 keeps the Triton path. DSV41_WO_A_CUDA=0 = previous behaviour.
usage: patch_wo_a_cuda.py <site-packages>"""
import sys

sp = sys.argv[1]
A = f"{sp}/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py"
W = f"{sp}/vllm/models/deepseek_v4_1/ampere/dsv41_wo_a.py"

s = open(W).read()
if "dsv41_wo_a_small" not in s:
    s += '''

# dsv41 (ds41-patches/patch_wo_a_cuda.py): CUDA decode wo_a with the inverse RoPE fused into the activation load
import os as _os

from vllm.utils.torch_utils import direct_register_custom_op as _reg

_WO_A_CUDA = _os.environ.get("DSV41_WO_A_CUDA", "1") != "0"
# validation only: DSV41_WO_A_ALL=1 also runs M > 8 here, 8 rows at a time, so prompt scoring (dqeval) checks it
_WO_A_ALL = _os.environ.get("DSV41_WO_A_ALL", "0") == "1"
_WO_A_CFG = {(2, 1024, 4096): (8, 1, 3)}   # (G, R, D) -> (warps per tile, tiles per block, depth), M=6 on CMP 170HX


def cuda_small_ok(o: torch.Tensor) -> bool:
    if not _WO_A_CUDA or (o.shape[0] > 8 and not _WO_A_ALL) or o.shape[0] < 1:
        return False
    try:
        import dsv41_woa_ext  # noqa: F401
    except ImportError:
        return False
    return True


def _wo_a_small(o: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, w8: torch.Tensor, s: torch.Tensor,
                G: int, R: int, rope_dim: int) -> torch.Tensor:
    import dsv41_woa_ext as _ext

    M, H, HEAD = o.shape
    D = H * HEAD // G
    x = o.reshape(M, -1)
    if not x.is_contiguous() or x.data_ptr() % 16:
        x = x.contiguous()
    pos = positions if positions.dtype == torch.int64 else positions.to(torch.int64)
    out = torch.empty(M, G * R, dtype=torch.bfloat16, device=o.device)
    wpt, tpb, depth = _WO_A_CFG.get((G, R, D), (8, 1, 2))
    for r in range(0, M, 8):
        _ext.wo_a_small(w8, s, x[r:r + 8], pos[r:r + 8], cos_sin, out[r:r + 8], G, R, D, HEAD, HEAD - rope_dim, wpt, tpb, depth)
    return out


def _wo_a_small_fake(o, positions, cos_sin, w8, s, G, R, rope_dim):
    return o.new_empty((o.shape[0], G * R))


_reg("dsv41_wo_a_small", _wo_a_small, fake_impl=_wo_a_small_fake)
'''
    open(W, "w").write(s)
    print("patched", W)

a = open(A).read()
if "dsv41_wo_a_small" not in a:
    # validation mode: let every M reach the FP8 branch (decode keeps the M <= 16 rule)
    o2 = "        if _DSV41_WO_A_FP8 and o.shape[0] <= 16:\n"
    assert a.count(o2) == 1
    a = a.replace(o2, "        if _DSV41_WO_A_FP8 and (o.shape[0] <= 16 or __import__('os').environ.get('DSV41_WO_A_ALL', '0') == '1'):\n")
    old = """            if fp8:
                from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as _w
                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj
"""
    new = """            if fp8:
                from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as _w
                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

                # dsv41 (ds41-patches/patch_wo_a_cuda.py): one CUDA kernel, inverse RoPE fused, for M <= 8
                if _w.cuda_small_ok(o) and o.dim() == 3 and self.rotary_emb.cos_sin_cache.dtype == torch.float32:
                    return self.wo_b(torch.ops.vllm.dsv41_wo_a_small(
                        o, positions, self.rotary_emb.cos_sin_cache, fp8[0], fp8[1],
                        self.n_local_groups, self.o_lora_rank, self.rope_head_dim))
"""
    assert a.count(old) == 1
    a = a.replace(old, new)
    open(A, "w").write(a)
    print("patched", A)
