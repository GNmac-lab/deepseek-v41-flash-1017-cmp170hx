# dsv41: token-blocked fused mHC post + delayed pre for small batches (sm80).
#
# Upstream on sm80 runs three kernels per sublayer boundary: mhc_post (one
# block per token), hc_prenorm_gemm (re-reads the 2 MB fp32 fn per token) and
# mhc_pre_big_fuse_with_norm (one 96-thread block per token).  With the
# delayed pre-mix, layer_input only needs the previous pre-mix, so:
#   A: post-mapping + bf16 residual + collapse + GEMM/sumsq partials, with fn
#      read once per token tile;
#   B: split reduction + sinkhorn (chunk 0) + RMSNorm over hidden chunks.
# Same bf16 rounding points as upstream; only fp32 summation order differs.
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=["T"])
def _dsv41_mhc_a(x_ptr, res_ptr, post_ptr, comb_ptr, pre_ptr, fn_ptr,
                 res_out_ptr, col_ptr, mix_ptr, sq_ptr, csq_ptr, T,
                 H: tl.constexpr, NOUT: tl.constexpr, HPS: tl.constexpr,
                 BT: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr,
                 DO_POST: tl.constexpr):
    pid_s = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_t = pid_t * BT + tl.arange(0, BT)
    offs_o = pid_o * BO + tl.arange(0, BO)
    offs_k = tl.arange(0, BK)
    tmask = offs_t < T
    omask = offs_o < NOUT
    t4 = offs_t * 4
    if DO_POST:
        p0 = tl.load(post_ptr + t4 + 0, mask=tmask, other=0.0)[:, None]
        p1 = tl.load(post_ptr + t4 + 1, mask=tmask, other=0.0)[:, None]
        p2 = tl.load(post_ptr + t4 + 2, mask=tmask, other=0.0)[:, None]
        p3 = tl.load(post_ptr + t4 + 3, mask=tmask, other=0.0)[:, None]
        t16 = offs_t * 16
        c00 = tl.load(comb_ptr + t16 + 0, mask=tmask, other=0.0)[:, None]
        c01 = tl.load(comb_ptr + t16 + 1, mask=tmask, other=0.0)[:, None]
        c02 = tl.load(comb_ptr + t16 + 2, mask=tmask, other=0.0)[:, None]
        c03 = tl.load(comb_ptr + t16 + 3, mask=tmask, other=0.0)[:, None]
        c10 = tl.load(comb_ptr + t16 + 4, mask=tmask, other=0.0)[:, None]
        c11 = tl.load(comb_ptr + t16 + 5, mask=tmask, other=0.0)[:, None]
        c12 = tl.load(comb_ptr + t16 + 6, mask=tmask, other=0.0)[:, None]
        c13 = tl.load(comb_ptr + t16 + 7, mask=tmask, other=0.0)[:, None]
        c20 = tl.load(comb_ptr + t16 + 8, mask=tmask, other=0.0)[:, None]
        c21 = tl.load(comb_ptr + t16 + 9, mask=tmask, other=0.0)[:, None]
        c22 = tl.load(comb_ptr + t16 + 10, mask=tmask, other=0.0)[:, None]
        c23 = tl.load(comb_ptr + t16 + 11, mask=tmask, other=0.0)[:, None]
        c30 = tl.load(comb_ptr + t16 + 12, mask=tmask, other=0.0)[:, None]
        c31 = tl.load(comb_ptr + t16 + 13, mask=tmask, other=0.0)[:, None]
        c32 = tl.load(comb_ptr + t16 + 14, mask=tmask, other=0.0)[:, None]
        c33 = tl.load(comb_ptr + t16 + 15, mask=tmask, other=0.0)[:, None]
    q0 = tl.load(pre_ptr + t4 + 0, mask=tmask, other=0.0)[:, None]
    q1 = tl.load(pre_ptr + t4 + 1, mask=tmask, other=0.0)[:, None]
    q2 = tl.load(pre_ptr + t4 + 2, mask=tmask, other=0.0)[:, None]
    q3 = tl.load(pre_ptr + t4 + 3, mask=tmask, other=0.0)[:, None]

    acc = tl.zeros((BT, BO, BK), tl.float32)
    sq = tl.zeros((BT, BK), tl.float32)
    csq = tl.zeros((BT, BK), tl.float32)
    rbase = offs_t[:, None] * (4 * H)
    fbase = offs_o[:, None] * (4 * H)
    store_ok = tmask[:, None] & (pid_o == 0)
    for kk in range(0, HPS, BK):
        h = (pid_s * HPS + kk + offs_k)[None, :]
        lm = tmask[:, None]
        r0 = tl.load(res_ptr + rbase + 0 * H + h, mask=lm, other=0.0).to(tl.float32)
        r1 = tl.load(res_ptr + rbase + 1 * H + h, mask=lm, other=0.0).to(tl.float32)
        r2 = tl.load(res_ptr + rbase + 2 * H + h, mask=lm, other=0.0).to(tl.float32)
        r3 = tl.load(res_ptr + rbase + 3 * H + h, mask=lm, other=0.0).to(tl.float32)
        if DO_POST:
            xv = tl.load(x_ptr + offs_t[:, None] * H + h, mask=lm, other=0.0).to(tl.float32)
            n0 = tl.fma(c30, r3, tl.fma(c20, r2, tl.fma(c10, r1, tl.fma(p0, xv, c00 * r0)))).to(tl.bfloat16)
            n1 = tl.fma(c31, r3, tl.fma(c21, r2, tl.fma(c11, r1, tl.fma(p1, xv, c01 * r0)))).to(tl.bfloat16)
            n2 = tl.fma(c32, r3, tl.fma(c22, r2, tl.fma(c12, r1, tl.fma(p2, xv, c02 * r0)))).to(tl.bfloat16)
            n3 = tl.fma(c33, r3, tl.fma(c23, r2, tl.fma(c13, r1, tl.fma(p3, xv, c03 * r0)))).to(tl.bfloat16)
            tl.store(res_out_ptr + rbase + 0 * H + h, n0, mask=store_ok)
            tl.store(res_out_ptr + rbase + 1 * H + h, n1, mask=store_ok)
            tl.store(res_out_ptr + rbase + 2 * H + h, n2, mask=store_ok)
            tl.store(res_out_ptr + rbase + 3 * H + h, n3, mask=store_ok)
            r0 = n0.to(tl.float32)
            r1 = n1.to(tl.float32)
            r2 = n2.to(tl.float32)
            r3 = n3.to(tl.float32)
        col = tl.fma(q3, r3, tl.fma(q2, r2, tl.fma(q1, r1, q0 * r0))).to(tl.bfloat16)
        tl.store(col_ptr + offs_t[:, None] * H + h, col, mask=store_ok)
        colf = col.to(tl.float32)
        csq += colf * colf
        sq += r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
        fm = omask[:, None]
        f0 = tl.load(fn_ptr + fbase + 0 * H + h, mask=fm, other=0.0)
        f1 = tl.load(fn_ptr + fbase + 1 * H + h, mask=fm, other=0.0)
        f2 = tl.load(fn_ptr + fbase + 2 * H + h, mask=fm, other=0.0)
        f3 = tl.load(fn_ptr + fbase + 3 * H + h, mask=fm, other=0.0)
        acc += r0[:, None, :] * f0[None, :, :]
        acc += r1[:, None, :] * f1[None, :, :]
        acc += r2[:, None, :] * f2[None, :, :]
        acc += r3[:, None, :] * f3[None, :, :]
    res = tl.sum(acc, axis=2)
    tl.store(mix_ptr + pid_s * T * NOUT + offs_t[:, None] * NOUT + offs_o[None, :], res,
             mask=tmask[:, None] & omask[None, :])
    smask = tmask & (pid_o == 0)
    tl.store(sq_ptr + pid_s * T + offs_t, tl.sum(sq, axis=1), mask=smask)
    tl.store(csq_ptr + pid_s * T + offs_t, tl.sum(csq, axis=1), mask=smask)


@triton.jit(do_not_specialize=["T", "S"])
def _dsv41_mhc_b(col_ptr, nw_ptr, mix_ptr, sq_ptr, csq_ptr, scale_ptr, base_ptr,
                 li_ptr, post_ptr, comb_ptr, pre_out_ptr, T, S,
                 H: tl.constexpr, NOUT: tl.constexpr, SP: tl.constexpr, BH: tl.constexpr,
                 rms_numel, rms_eps, pre_eps, sk_eps, post_mult, norm_eps,
                 REPEAT: tl.constexpr, PART: tl.constexpr):
    # PART 1: RMSNorm of one hidden chunk per program.  PART 2: mixes +
    # sinkhorn (one program per token).  PART 0: both, the sinkhorn in the
    # extra last program so it overlaps the chunks.
    t = tl.program_id(0)
    c = tl.program_id(1)
    offs_s = tl.arange(0, SP)
    smask = offs_s < S
    do_sk = (PART == 2) or ((PART == 0) and (c == H // BH))
    if not do_sk:
        csq = tl.sum(tl.load(csq_ptr + offs_s * T + t, mask=smask, other=0.0), axis=0)
        rn = libdevice.rsqrt(csq / H + norm_eps)
        h = c * BH + tl.arange(0, BH)
        col = tl.load(col_ptr + t * H + h).to(tl.float32)
        w = tl.load(nw_ptr + h).to(tl.float32)
        tl.store(li_ptr + t * H + h, (col * rn * w).to(tl.bfloat16))
    else:
        sq = tl.sum(tl.load(sq_ptr + offs_s * T + t, mask=smask, other=0.0), axis=0)
        rms = libdevice.rsqrt(sq / rms_numel + rms_eps)
        mbase = offs_s[:, None] * (T * NOUT) + t * NOUT
        j4 = tl.arange(0, 4)
        pre = tl.sum(tl.load(mix_ptr + mbase + j4[None, :], mask=smask[:, None], other=0.0), axis=0) * rms
        pst = tl.sum(tl.load(mix_ptr + mbase + 4 + j4[None, :], mask=smask[:, None], other=0.0), axis=0) * rms
        s0 = tl.load(scale_ptr + 0)
        s1 = tl.load(scale_ptr + 1)
        s2 = tl.load(scale_ptr + 2)
        b_pre = tl.load(base_ptr + j4)
        b_pst = tl.load(base_ptr + 4 + j4)
        tl.store(pre_out_ptr + t * 4 + j4, 1.0 / (1.0 + libdevice.exp(-(pre * s0 + b_pre))) + pre_eps)
        tl.store(post_ptr + t * 4 + j4, 1.0 / (1.0 + libdevice.exp(-(pst * s1 + b_pst))) * post_mult)
        jk = j4[:, None] * 4 + j4[None, :]
        cmix = tl.load(mix_ptr + offs_s[:, None, None] * (T * NOUT) + t * NOUT + 8 + jk[None, :, :],
                       mask=smask[:, None, None], other=0.0)
        cm = tl.sum(cmix, axis=0) * rms * s2 + tl.load(base_ptr + 8 + jk)
        cm = libdevice.exp(cm - tl.max(cm, axis=1)[:, None])
        cm = cm / tl.sum(cm, axis=1)[:, None] + sk_eps
        cm = cm / (tl.sum(cm, axis=0)[None, :] + sk_eps)
        for _ in tl.static_range(REPEAT - 1):
            cm = cm / (tl.sum(cm, axis=1)[:, None] + sk_eps)
            cm = cm / (tl.sum(cm, axis=0)[None, :] + sk_eps)
        tl.store(comb_ptr + t * 16 + jk, cm)


_CFG = {}


def _config(num_tokens):
    # (splits over hidden, token tile, out tile, k block, warps); swept on sm80.
    if num_tokens == 1:
        return _CFG.get("small", (80, 1, 8, 64, 1))
    if num_tokens <= 16:
        return _CFG.get("mid", (80, 4, 8, 32, 1))
    return _CFG.get("large", (20, 8, 8, 32, 2))


_SIDE: dict = {}
_EVENTS: dict = {}
_PENDING: list = []   # (event, keep-alive tensors) forked onto the side stream
_DEFERRED: list = []  # sinkhorn launches waiting for the next all-reduce


def _side_mode() -> str:
    """"ar" = sinkhorn beside the next all-reduce, "1" = beside the next kernels, "0" = inline."""
    mode = os.environ.get("DSV41_MHC_SIDE", "ar")
    if mode == "0":
        return "0"
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    # Breakable (PIECEWISE) capture ends a segment at attention; a fork here
    # would not be joined inside its segment.
    if is_forward_context_available() and (
        get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    ):
        return "0"
    return mode


def _fork_sinkhorn(dev, num_tokens, b_args, bw, keep) -> None:
    main = torch.cuda.current_stream(dev)
    side = _SIDE.get(dev.index)
    if side is None:
        side = _SIDE[dev.index] = torch.cuda.Stream(device=dev)
        _EVENTS[dev.index] = torch.cuda.Event()
    ev = _EVENTS[dev.index]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _dsv41_mhc_b[(num_tokens, 1)](*b_args, 2, num_warps=bw)
        ev.record(side)
    _PENDING.append((ev, keep))


def dsv41_mhc_flush(on_side: bool = True) -> None:
    """Launch deferred sinkhorns: forked beside the caller's next kernel, or inline."""
    while _DEFERRED:
        dev, num_tokens, b_args, bw, keep = _DEFERRED.pop(0)
        if on_side:
            _fork_sinkhorn(dev, num_tokens, b_args, bw, keep)
        else:
            _dsv41_mhc_b[(num_tokens, 1)](*b_args, 2, num_warps=bw)


def dsv41_mhc_join() -> None:
    """Make post/comb/pre_out safe to read on the current stream."""
    dsv41_mhc_flush(on_side=False)
    while _PENDING:
        ev, _keep = _PENDING.pop()
        torch.cuda.current_stream().wait_event(ev)


def _install_all_reduce_hook() -> None:
    # The NCCL LL all-reduce occupies a few SMs and mostly waits on peers, so
    # a sinkhorn forked there does not stretch a full-grid GEMM's tail.
    from vllm.distributed.parallel_state import GroupCoordinator

    orig = GroupCoordinator._all_reduce_out_place
    if getattr(orig, "_dsv41_mhc", False):
        return

    def _all_reduce_out_place(self, input_, *args, **kwargs):
        if _DEFERRED and input_.is_cuda:
            dsv41_mhc_flush(on_side=True)
        return orig(self, input_, *args, **kwargs)

    _all_reduce_out_place._dsv41_mhc = True
    GroupCoordinator._all_reduce_out_place = _all_reduce_out_place


try:
    _install_all_reduce_hook()
except Exception:  # no vLLM (offline tests): "ar" degrades to a launch at the join
    pass


def dsv41_mhc_post_pre(x, residual, post_mix, res_mix, pre_mix, fn, hc_scale, hc_base,
                       rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                       sinkhorn_repeat, norm_weight, norm_eps, do_post=True):
    """Returns (residual_out, post_mix (T,4,1), comb (T,4,4), layer_input, pre_mix_out)."""
    num_tokens, hc_mult, hidden = residual.shape
    assert hc_mult == 4 and residual.is_contiguous() and residual.dtype == torch.bfloat16
    n_out = fn.shape[0]
    assert fn.dtype == torch.float32 and fn.is_contiguous() and fn.shape[1] == 4 * hidden
    assert norm_weight.dtype == torch.bfloat16 and norm_weight.is_contiguous()
    assert pre_mix.dtype == torch.float32 and pre_mix.is_contiguous()
    assert hc_scale.is_contiguous() and hc_base.is_contiguous()
    dev = residual.device
    splits, bt, bo, bk, nw = _config(num_tokens)
    hps = hidden // splits
    assert hps * splits == hidden and hps % bk == 0
    if do_post:
        res_out = torch.empty_like(residual)
        xin = x.reshape(num_tokens, hidden)
        post_in = post_mix.reshape(num_tokens, 4).contiguous()
        comb_in = res_mix.reshape(num_tokens, 16).contiguous()
    else:
        res_out = residual
        xin = post_in = comb_in = residual
    col = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
    mix = torch.empty(splits, num_tokens, n_out, dtype=torch.float32, device=dev)
    sq = torch.empty(splits, num_tokens, dtype=torch.float32, device=dev)
    csq = torch.empty(splits, num_tokens, dtype=torch.float32, device=dev)
    _dsv41_mhc_a[(splits, triton.cdiv(n_out, bo), triton.cdiv(num_tokens, bt))](
        xin, residual, post_in, comb_in, pre_mix, fn, res_out, col, mix, sq, csq,
        num_tokens, hidden, n_out, hps, bt, bo, bk, do_post, num_warps=nw)
    layer_input = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=dev)
    post = torch.empty(num_tokens, 4, dtype=torch.float32, device=dev)
    comb = torch.empty(num_tokens, 16, dtype=torch.float32, device=dev)
    pre_out = torch.empty(num_tokens, 4, dtype=torch.float32, device=dev)
    bh = _CFG.get("b_bh", 1024 if hidden % 1024 == 0 else 512)
    nch = hidden // bh
    b_args = (col, norm_weight, mix, sq, csq, hc_scale, hc_base, layer_input, post, comb, pre_out,
              num_tokens, splits, hidden, n_out, triton.next_power_of_2(splits), bh,
              float(4 * hidden), float(rms_eps), float(hc_pre_eps), float(hc_sinkhorn_eps),
              float(hc_post_mult_value), float(norm_eps), int(sinkhorn_repeat))
    bw = _CFG.get("b_warps", 1)
    mode = _side_mode()
    keep = (mix, sq, post, comb, pre_out)
    if mode == "ar":
        _DEFERRED.append((dev, num_tokens, b_args, bw, keep))
        _dsv41_mhc_b[(num_tokens, nch)](*b_args, 1, num_warps=bw)
    elif mode == "1":
        _fork_sinkhorn(dev, num_tokens, b_args, bw, keep)
        _dsv41_mhc_b[(num_tokens, nch)](*b_args, 1, num_warps=bw)
    else:
        _dsv41_mhc_b[(num_tokens, nch + 1)](*b_args, 0, num_warps=bw)
    return (res_out, post.unsqueeze(-1), comb.view(num_tokens, 4, 4), layer_input, pre_out)
