# dsv41 (runtime variant): the decode ragged top-k metadata in ONE launch instead of five.
#
# compute_global_topk_ragged_indices_and_indptr (models/deepseek_v4_1/amd/rocm.py) runs a count kernel, a
# torch.zeros + cub cumsum (Fill, ScanInit, Scan) and a pack kernel: five dependent launches of ~2 us each on the
# attention critical path of every indexer layer. For small T one program per token recounts the rows before it
# (T <= DSV41_TOPK_META_MAX, default 16, so at most 16 rows), so lens, indptr and the packed slot ids come out of
# one kernel. Integer work only, same masks: identical outputs. Larger T keeps the original path.
# Loading this module self-tests it against the original on the worker's GPU and raises on any difference.
import os

import torch
import triton
import triton.language as tl

from vllm.models.deepseek_v4_1.amd import rocm as _rocm

# the original, kept across reloads so a second load never wraps the variant
_ORIG = getattr(_rocm, "_dsv41_orig_topk_ragged", None) or _rocm.compute_global_topk_ragged_indices_and_indptr
_rocm._dsv41_orig_topk_ragged = _ORIG
MAX_T = int(os.environ.get("DSV41_TOPK_META_MAX", "16"))


@triton.jit
def _row_count(idx_ptr, idx_stride, row, topk, BLOCK: tl.constexpr):
    c = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, BLOCK):
        off = i + tl.arange(0, BLOCK)
        v = tl.load(idx_ptr + row * idx_stride + off, mask=off < topk, other=-1)
        c += tl.sum((v >= 0).to(tl.int32), axis=0)
    return c


@triton.jit
def _dsv41_topk_meta_kernel(lens_ptr, indptr_ptr, ragged_ptr, idx_ptr, idx_stride, valid_ptr, t2r_ptr, bt_ptr,
                            bt_stride, block_size, topk, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    start = tl.zeros((), dtype=tl.int32)
    for j in range(0, t):
        cj = _row_count(idx_ptr, idx_stride, j, topk, BLOCK)
        start += tl.where(tl.load(valid_ptr + j), cj, 0)
    n = tl.where(tl.load(valid_ptr + t), _row_count(idx_ptr, idx_stride, t, topk, BLOCK), 0)
    tl.store(lens_ptr + t, n)
    tl.store(indptr_ptr + t + 1, start + n)
    if t == 0:
        tl.store(indptr_ptr, 0)
    req = tl.load(t2r_ptr + t)
    for i in range(0, topk, BLOCK):
        off = i + tl.arange(0, BLOCK)
        mask = (off < n) & (off < topk)
        local = tl.load(idx_ptr + t * idx_stride + off, mask=mask, other=-1)
        valid = mask & (local >= 0)
        bn = tl.load(bt_ptr + req * bt_stride + local // block_size, mask=valid, other=0)
        slot = tl.where(valid, bn * block_size + local % block_size, -1)
        tl.store(ragged_ptr + start + off, slot, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(topk_indices, token_to_req_indices, block_table, block_size,
                                                  is_valid_token):
    ti = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens, topk = ti.shape
    if num_tokens == 0 or num_tokens > MAX_T or topk == 0:
        return _ORIG(topk_indices, token_to_req_indices, block_table, block_size, is_valid_token)
    dev = ti.device
    lens = torch.empty(num_tokens, dtype=torch.int32, device=dev)
    indptr = torch.empty(num_tokens + 1, dtype=torch.int32, device=dev)
    ragged = torch.empty(num_tokens * topk, dtype=torch.int32, device=dev)
    _dsv41_topk_meta_kernel[(num_tokens,)](lens, indptr, ragged, ti, ti.stride(0), is_valid_token,
                                           token_to_req_indices, block_table, block_table.stride(0), block_size,
                                           topk, BLOCK=1024, num_warps=4)
    return ragged, indptr, lens


def _selftest() -> None:
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device="cpu").manual_seed(0)
    for num_tokens, topk, block_size, reqs in [(1, 512, 64, 1), (6, 512, 64, 1), (6, 2048, 32, 1), (12, 1024, 64, 2),
                                               (16, 512, 256, 3), (5, 100, 16, 1)]:
        for trial in range(3):
            ctx = 4096
            ti = torch.randint(0, ctx, (num_tokens, topk), generator=g, dtype=torch.int32)
            ti[torch.rand(num_tokens, topk, generator=g) < 0.3] = -1          # holes anywhere in a row
            ti[:, topk - topk // (trial + 2):] = -1                             # and a -1 tail
            valid = torch.rand(num_tokens, generator=g) < 0.8
            t2r = torch.randint(0, reqs, (num_tokens,), generator=g, dtype=torch.int32)
            bt = torch.randint(0, 50000, (reqs, ctx // block_size + 8), generator=g, dtype=torch.int32)
            args = (ti.to(dev), t2r.to(dev), bt.to(dev), block_size, valid.to(dev))
            ra, pa, la = _ORIG(*args)
            rb, pb, lb = compute_global_topk_ragged_indices_and_indptr(*args)
            torch.cuda.synchronize()
            nnz = int(pa[-1])
            if not (torch.equal(pa, pb) and torch.equal(la, lb) and torch.equal(ra[:nnz], rb[:nnz])):
                raise RuntimeError(f"dsv41_topk_meta self-test FAILED at T={num_tokens} topk={topk} bs={block_size}")


_selftest()

SETATTR = [{"target": "vllm.models.deepseek_v4_1.amd.rocm.compute_global_topk_ragged_indices_and_indptr",
            "source": "dsv41_topk_meta.compute_global_topk_ragged_indices_and_indptr"}]
