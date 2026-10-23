# dsv41 (runtime variant): build_ragged_indices_from_dense in ONE launch for small row counts.
#
# The stock helper (vllm/v1/attention/ops/rocm_aiter_mla_sparse.py) runs clamp, a zero fill, a cub cumsum (two
# kernels) and a pack kernel: five dependent launches per call, on every KV-cache group's decode metadata and inside
# the sparse-attention ops. For rows <= MAX_ROWS one program per row recounts the clamped lengths before it, writes its
# indptr entry and packs its prefix. Integer work only, same masks: identical outputs. Self-tested on load.
# Also, per KV-cache group in the decode SWA metadata build:
#   - DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build: the fused kernel writes straight into the persistent graph
#     buffers instead of a temporary + two D2D copies (_copy_ragged_to_graph_buffers);
#   - DeepseekSparseSWAMetadataBuilder.build: is_valid_token.copy_(slot_mapping >= 0) (two launches) becomes
#     torch.ge(slot_mapping, 0, out=is_valid_token) (one launch).
import inspect
import os
import re
import textwrap

import torch
import triton
import triton.language as tl

from vllm.models.deepseek_v4_1.amd import rocm as _rocm
from vllm.v1.attention.backends.mla import sparse_swa as _swa
from vllm.v1.attention.ops import rocm_aiter_mla_sparse as _ops

_ORIG = getattr(_ops, "_dsv41_orig_build_ragged", None) or _ops.build_ragged_indices_from_dense
_ops._dsv41_orig_build_ragged = _ORIG
MAX_ROWS = int(os.environ.get("DSV41_RAGGED_MAX_ROWS", "64"))


@triton.jit
def _dsv41_ragged_from_dense_kernel(indices_ptr, lengths_ptr, indptr_ptr, out_ptr, indices_stride0, num_rows_limit,
                                    row_width, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.zeros((), dtype=tl.int32)
    for j in range(0, row):
        lj = tl.load(lengths_ptr + j).to(tl.int32)
        start += tl.minimum(tl.maximum(lj, 0), row_width)
    n = tl.minimum(tl.maximum(tl.load(lengths_ptr + row).to(tl.int32), 0), row_width)
    tl.store(indptr_ptr + row + 1, start + n)
    if row == 0:
        tl.store(indptr_ptr, 0)
    for b in range(0, row_width, BLOCK):
        offs = b + tl.arange(0, BLOCK)
        mask = offs < n
        safe = tl.where(offs < row_width, offs, 0)
        vals = tl.load(indices_ptr + row * indices_stride0 + safe, mask=mask & (offs < row_width), other=-1).to(tl.int32)
        if num_rows_limit >= 0:
            vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)
        tl.store(out_ptr + start + offs, vals, mask=mask)


def build_ragged_indices_from_dense(indices: torch.Tensor, lengths: torch.Tensor, num_rows: int = -1):
    ind = indices.reshape(indices.shape[0], -1)
    rows = ind.shape[0]
    width = ind.shape[1] if ind.ndim == 2 else 0
    if (rows == 0 or rows > MAX_ROWS or width == 0 or ind.numel() == 0 or not ind.is_cuda
            or lengths.device != ind.device or lengths.numel() != rows or ind.stride(-1) != 1):
        return _ORIG(indices, lengths, num_rows)
    lens = lengths.reshape(-1)
    if not lens.is_contiguous():
        return _ORIG(indices, lengths, num_rows)
    indptr = torch.empty(rows + 1, dtype=torch.int32, device=ind.device)
    flat = torch.empty(rows * width, dtype=torch.int32, device=ind.device)
    _dsv41_ragged_from_dense_kernel[(rows,)](ind, lens, indptr, flat, ind.stride(0), int(num_rows), width,
                                             BLOCK=min(1024, triton.next_power_of_2(width)), num_warps=4)
    return flat, indptr


def _ragged_into(indices, lengths, buf_i, buf_p, num_rows, width):
    """build_ragged_indices_from_dense + _copy_ragged_to_graph_buffers, written straight into the graph buffers."""
    ind = indices.reshape(indices.shape[0], -1)
    rows = ind.shape[0]
    if (getattr(_rocm, "_ON_GFX950", False) or rows != num_rows or rows == 0 or rows > MAX_ROWS or width == 0
            or ind.shape[1] != width or not ind.is_cuda or lengths.device != ind.device or lengths.numel() != rows
            or ind.stride(-1) != 1 or not lengths.reshape(-1).is_contiguous()
            or buf_i.numel() < rows * width or buf_p.numel() < rows + 1):
        r_i, r_p = _ORIG(indices, lengths)
        return _rocm._copy_ragged_to_graph_buffers(r_i, r_p, buf_i, buf_p, num_rows, width)
    indptr = buf_p[: rows + 1]
    flat = buf_i[: max(rows * width, 1)]
    _dsv41_ragged_from_dense_kernel[(rows,)](ind, lengths.reshape(-1), indptr, flat, ind.stride(0), -1, width,
                                             BLOCK=min(1024, triton.next_power_of_2(width)), num_warps=4)
    return flat, indptr


def _patch_source(fn, pairs, ns_mod, extra):
    src = textwrap.dedent(inspect.getsource(fn))
    for pat, rep, n_expect in pairs:
        src, n = re.subn(pat, rep, src)
        if n != n_expect:
            raise RuntimeError(f"dsv41_ragged_lean: pattern {pat[:40]!r} matched {n}, expected {n_expect}")
    ns = dict(vars(ns_mod))
    ns.update(extra)
    exec(compile(src, f"<dsv41_ragged_lean {fn.__qualname__}>", "exec"), ns)
    return ns[fn.__name__]


_RB = _rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder
_ORIG_RBUILD = _RB.__dict__.get("_dsv41_orig_build") or _RB.build
_RB._dsv41_orig_build = _ORIG_RBUILD
rocm_swa_build = _patch_source(_ORIG_RBUILD, [
    (r"super\(\)\.build\(", "super(DeepseekV4ROCMAiterSparseSWAMetadataBuilder, self).build(", 1),
    (r"ragged_indices, ragged_indptr = build_ragged_indices_from_dense\(\s*"
     r"(base\.decode_swa_indices\.reshape\(\s*base\.num_decode_tokens, base\.decode_swa_width\s*\)),\s*"
     r"base\.decode_swa_lens,\s*\)\s*"
     r"ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers\(\s*ragged_indices,\s*ragged_indptr,\s*"
     r"(self\.decode_swa_ragged_indices_buffer),\s*(self\.decode_swa_ragged_indptr_buffer),\s*"
     r"(base\.num_decode_tokens),\s*(base\.decode_swa_width),\s*\)",
     r"ragged_indices, ragged_indptr = _dsv41_ragged_into(\1, base.decode_swa_lens, \2, \3, \4, \5)", 1)],
    _rocm, {"_dsv41_ragged_into": _ragged_into})

_SB = _swa.DeepseekSparseSWAMetadataBuilder
_ORIG_SBUILD = _SB.__dict__.get("_dsv41_orig_build") or _SB.build
_SB._dsv41_orig_build = _ORIG_SBUILD
swa_build = _patch_source(_ORIG_SBUILD, [
    (r"is_valid_token\.copy_\(slot_mapping >= 0\)", "torch.ge(slot_mapping, 0, out=is_valid_token)", 1)], _swa, {})


def _selftest() -> None:
    dev = torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device="cpu").manual_seed(1)
    for rows, width, limit, ldt in [(1, 128, -1, torch.int32), (6, 128, -1, torch.int32), (6, 1000, 5000, torch.int64),
                                    (12, 640, -1, torch.int32), (48, 256, 700, torch.int32), (5, 7, -1, torch.int32)]:
        for _ in range(3):
            ind = torch.randint(-1, 1000, (rows, width), generator=g, dtype=torch.int32)
            lens = torch.randint(-3, width + 5, (rows,), generator=g).to(ldt)
            a = (ind.to(dev), lens.to(dev), limit)
            fa, pa = _ORIG(*a)
            fb, pb = build_ragged_indices_from_dense(*a)
            torch.cuda.synchronize()
            nnz = int(pa[-1])
            if not (torch.equal(pa, pb) and torch.equal(fa[:nnz], fb[:nnz]) and fa.shape == fb.shape):
                raise RuntimeError(f"dsv41_ragged_lean self-test FAILED rows={rows} width={width} limit={limit}")
            if limit < 0:   # the into-buffer path (decode metadata) against the stock build + copy
                bi = torch.full((rows * width + 64,), 7, dtype=torch.int32, device=dev)
                bp = torch.full((rows + 9,), 7, dtype=torch.int32, device=dev)
                ci = bi.clone()
                cp = bp.clone()
                ri, rp = _rocm._copy_ragged_to_graph_buffers(fa, pa, ci, cp, rows, width)
                qi, qp = _ragged_into(a[0], a[1], bi, bp, rows, width)
                torch.cuda.synchronize()
                if not (torch.equal(rp, qp) and torch.equal(ri[:nnz], qi[:nnz]) and ri.shape == qi.shape):
                    raise RuntimeError(f"dsv41_ragged_lean into-buffer self-test FAILED rows={rows} width={width}")


_selftest()

SETATTR = [{"target": "vllm.v1.attention.ops.rocm_aiter_mla_sparse.build_ragged_indices_from_dense",
            "source": "dsv41_ragged_lean.build_ragged_indices_from_dense"},
           {"target": "vllm.models.deepseek_v4_1.amd.rocm.build_ragged_indices_from_dense",
            "source": "dsv41_ragged_lean.build_ragged_indices_from_dense"},
           {"target": "vllm.models.deepseek_v4_1.amd.rocm.DeepseekV4ROCMAiterSparseSWAMetadataBuilder.build",
            "source": "dsv41_ragged_lean.rocm_swa_build"},
           {"target": "vllm.v1.attention.backends.mla.sparse_swa.DeepseekSparseSWAMetadataBuilder.build",
            "source": "dsv41_ragged_lean.swa_build"}]
