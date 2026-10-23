"""dsv41: one DSpark Markov sampling step on this rank's vocab shard, fused (Triton).

For draft position i, the stock loop computes li = local[:, i] + F.linear(e, w2) in bf16, then li.max(dim=-1), then
stacks (value, id + start) as an fp32 pair for the all-gather. Here one kernel reads each block of w2 rows once, forms
the same bf16 bias (fp32 dot, rounded to bf16 as cuBLAS's bf16 output is) and the same bf16 sum, and keeps the block's
first maximum; a second kernel reduces the blocks to the fp32 (value, global id) pair. Ties keep the lowest id, as
torch's max does. The dot's summation order differs from cuBLAS, so a bias can differ by one bf16 ulp in rare cases;
only draft tokens can change (the target still verifies every draft).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _markov_partial_kernel(local_ptr, local_rs, e_ptr, e_rs, w2_ptr, w2_rs, pv_ptr, pi_ptr, R, S, NB,
                           R_DIM: tl.constexpr, BLOCK_S: tl.constexpr):
    # one block of w2 rows, read once for every request row
    blk = tl.program_id(0)
    offs = blk * BLOCK_S + tl.arange(0, BLOCK_S)
    m = offs < S
    kr = tl.arange(0, R_DIM)
    w = tl.load(w2_ptr + offs[:, None] * w2_rs + kr[None, :], mask=m[:, None], other=0.0).to(tl.float32)
    for row in range(0, R):
        e = tl.load(e_ptr + row * e_rs + kr).to(tl.float32)
        bias = tl.sum(w * e[None, :], axis=1).to(tl.bfloat16)                              # F.linear's bf16 output
        base = tl.load(local_ptr + row * local_rs + offs, mask=m, other=0.0)
        li = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16).to(tl.float32)  # the bf16 add
        li = tl.where(m, li, float("-inf"))
        v = tl.max(li, axis=0)
        idx = tl.min(tl.where(li == v, offs, S), axis=0)                                   # first maximum
        tl.store(pv_ptr + row * NB + blk, v)
        tl.store(pi_ptr + row * NB + blk, idx)


@triton.jit
def _markov_final_kernel(pv_ptr, pi_ptr, pair_ptr, NB, start, BLOCK_NB: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK_NB)
    m = o < NB
    v = tl.load(pv_ptr + row * NB + o, mask=m, other=float("-inf"))
    i = tl.load(pi_ptr + row * NB + o, mask=m, other=2147483647)
    vm = tl.max(v, axis=0)
    im = tl.min(tl.where(v == vm, i, 2147483647), axis=0)
    tl.store(pair_ptr + row * 2, vm)
    tl.store(pair_ptr + row * 2 + 1, (im + start).to(tl.float32))                         # ids < 2^24: exact


BLOCK_S = 8          # graph-timed sweep at 32320 x 256 (R = 1, 2, 4, 8): BLOCK_S 8, one warp is best


def markov_step_pair(local_i: torch.Tensor, e: torch.Tensor, w2: torch.Tensor, start: int) -> torch.Tensor:
    """fp32 [R, 2] = (max of bf16(local_i + bf16(e @ w2.T)), its global id) over this shard; as the stock pair."""
    R, S = local_i.shape
    r = e.shape[1]
    assert w2.shape == (S, r) and w2.stride(1) == 1 and e.stride(1) == 1 and local_i.stride(1) == 1
    assert local_i.dtype == torch.bfloat16 and e.dtype == torch.bfloat16 and w2.dtype == torch.bfloat16
    assert r & (r - 1) == 0
    NB = triton.cdiv(S, BLOCK_S)
    pv = torch.empty(R, NB, dtype=torch.float32, device=e.device)
    pi = torch.empty(R, NB, dtype=torch.int32, device=e.device)
    _markov_partial_kernel[(NB,)](local_i, local_i.stride(0), e, e.stride(0), w2, w2.stride(0), pv, pi, R, S, NB,
                                  R_DIM=r, BLOCK_S=BLOCK_S, num_warps=1)
    pair = torch.empty(R, 2, dtype=torch.float32, device=e.device)
    _markov_final_kernel[(R,)](pv, pi, pair, NB, start, BLOCK_NB=triton.next_power_of_2(NB), num_warps=4)
    return pair
