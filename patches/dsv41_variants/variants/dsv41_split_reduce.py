# dsv41 (runtime variant): prefill all-reduces beside compute, in two row slices.
#
# A decoder layer sums two TP partials per prefill step: wo_b after attention and the MoE output after the FFN, each
# a large NCCL all-reduce (2,048 rows x 5,120 bf16 = 21 MB, ~5 ms at the PCIe Gen2 link rate) that the step waits
# for. Everything after the attention core (wo_a, wo_b, mHC post + pre, MoE) depends only on its own row. For an eager
# step of at least DSV41_SPLIT_MIN_ROWS rows this runs that part in two row slices A = [0, cut) and B = [cut, rows)
# and issues the four all-reduces on a side stream in program order, so each slice's compute runs beside the other
# slice's all-reduce (idea from Morrowmake/vllm-cmp170hx sm80_split_reduce.py, Apache-2.0):
#     main: attention (all rows) | proj(A) | proj(B) | FFN(A) | FFN(B) | join
#     side:                       sum proj(A) | sum proj(B) | sum FFN(A) | sum FFN(B)
# Same kernels, same precision; the NCCL ring may add a row's four partials in another order (message size changes
# its chunking), so outputs are NLL-equivalent, not bit-identical. Decode steps and CUDA graphs are untouched.
# With dsv41_ce_ring loaded the side-stream sums are copy-engine owner sums (no SM data movement), which is what
# makes the overlap real: NCCL's SM kernels starve beside the MoE kernels (measured: no gain with NCCL).
# Measured cold prefill (median of 9): 1,784 -> 1,961-1,975 tok/s (+10 %) with two slices.
# v81: each slice runs the fused mHC post + pre (_dsv41_mhc_post_pre) up to DSV41_MHC_FUSED_MAX rows (here 4096;
# the model default is 128); warm prefill 1,956 -> 2,108 tok/s. Every read of a fused mHC output waits for
# _dsv41_mhc_join() (its sinkhorn runs deferred on a side stream; reading early gave NaN logprobs).
# v81: three slices by default (512 / 512 / 1,024 rows of a 2,048-row step): with dsv41_moe_try's cheaper MoE calls
# the exposed first all-reduce gets shorter; warm prefill 2,142 -> 2,250-2,330 tok/s.
# v83 (superseded by v86): slices 512 / 1,024 / 512 (both exposed all-reduces, the first and the last, are 512 rows): +2.7 % warm
# (A/B x3, 2,219 -> 2,279). Dense GEMM tiling follows the slice rows, so bits move: prompt NLL 0.15154 -> 0.15243.
#   DSV41_SPLIT=0 off (default on), DSV41_SPLIT_CE=0 NCCL sums, DSV41_SPLIT_SLICES (3), DSV41_SPLIT_MIN_ROWS (1024),
#   DSV41_SPLIT_PRIO (side-stream priority, -1), DSV41_SPLIT_TIMING=1 per-phase timing in stats(),
#   DSV41_SPLIT_CUTS boundaries for a 2,048-row step (scaled; DSV41_SPLIT_ALIGN, default 128), default 512,1536;
#   DSV41_SPLIT_CUTS=even: DSV41_SPLIT_SLICES even slices.
# v86: with dsv41_attn_slices the attention runs per slice too (each slice on its own attention metadata) and the
# layers form a cross-layer pipeline (DSV41_PIPE, default on): a layer returns before its FFN slices' sums land and
# the next layer waits for slice i only when it reaches slice i. Two slices (cut at 1,024 rows of a 2,048-row step)
# are best then: warm prefill ~2,480 -> ~2,640 tok/s. Trap fixed here: indexer-less layers reuse the top-k rows
# the last indexer published in a shared buffer, so each slice's rows are stashed and put back (without it the
# first slice saw the second slice's selections: future tokens, NLL 12 % too low). Checked: a random-block-
# then-repeat prompt keeps its first-copy NLL (13.0) and second-copy NLL (0.11); paired NLL +0.10 % (48 prompts).
#   DSV41_PIPE=0 no cross-layer pipeline; DSV41_ATTN_SLICES=0 attention on all rows.
import inspect
import os
import re
import textwrap

import torch

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe.runner import moe_runner as _mr
from vllm.models.deepseek_v4_1.nvidia import model as _m

ALIGN = 512
MHC_MAX = int(os.environ.get("DSV41_MHC_FUSED_MAX", "4096"))   # -> model._DSV41_MHC_FUSED_MAX (SETATTR)
STATS = {"split_steps": 0, "plain_steps": 0, "ffn_prereduced": 0}
_EV: list = []          # timing mode: (kind, [events]) per layer call, read by stats()


def _timing() -> bool:
    return os.environ.get("DSV41_SPLIT_TIMING", "0") == "1"


def _ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e
_DEFER = [False]
_PREREDUCED = [False]
_SIDE: dict = {}
_TK: dict = {"key": None, "stash": {}}

_L = _m.DeepseekV4DecoderLayer
_ORIG_FWD = _L.__dict__.get("_dsv41_split_orig_fwd") or _L.forward
_L._dsv41_split_orig_fwd = _ORIG_FWD
_R = _mr.MoERunner
_ORIG_RED = _R.__dict__.get("_dsv41_split_orig_red") or _R._maybe_reduce_final_output
_R._dsv41_split_orig_red = _ORIG_RED


def _enabled() -> bool:
    return os.environ.get("DSV41_SPLIT", "1") == "1"


def _min_rows() -> int:
    return int(os.environ.get("DSV41_SPLIT_MIN_ROWS", "1024"))


def maybe_reduce_final_output(self, states, trunc_size, output_is_reduced=None):
    """MoE runner's late all-reduce, deferred to the caller while a split slice runs."""
    if _DEFER[0]:
        reduced = self._fused_output_is_reduced if output_is_reduced is None else output_is_reduced
        if reduced or self.moe_config.skip_final_all_reduce or self.moe_config.is_sequence_parallel:
            _PREREDUCED[0] = True
            return _ORIG_RED(self, states, trunc_size, output_is_reduced)
        return states[..., :trunc_size] if trunc_size is not None else states
    return _ORIG_RED(self, states, trunc_size, output_is_reduced)


def _side(dev) -> torch.cuda.Stream:
    s = _SIDE.get(dev)
    if s is None:
        s = _SIDE[dev] = torch.cuda.Stream(device=dev, priority=int(os.environ.get("DSV41_SPLIT_PRIO", "-1")))
    return s


def _ce_ring():
    """dsv41_ce_ring's process ring when DSV41_SPLIT_CE=1 (default) and the module is loaded, else None."""
    if os.environ.get("DSV41_SPLIT_CE", "1") != "1":
        return None
    import sys
    m = sys.modules.get("dsv41_ce_ring")
    return m._RING[0] if m is not None else None


def _ar_async(t: torch.Tensor, side: torch.cuda.Stream):
    """All-reduce t on the side stream after the main stream's work so far; returns (result, done event).
    Copy-engine owner sum (dsv41_ce_ring) when available, else NCCL."""
    main = torch.cuda.current_stream()
    ready = torch.cuda.Event()
    ready.record(main)
    ring = _ce_ring()
    with torch.cuda.stream(side):
        side.wait_event(ready)
        if ring is not None and ring.takes(t):
            r = ring.all_reduce_owner(t)
            STATS["ce_calls"] = STATS.get("ce_calls", 0) + 1
        else:
            r = tensor_model_parallel_all_reduce(t)
    t.record_stream(side)
    r.record_stream(main)
    done = torch.cuda.Event()
    done.record(side)
    return r, done


def _split_ok(layer, x, positions) -> bool:
    if not _enabled() or getattr(layer, "use_sequence_parallel", False):
        if (_timing() and len(_EV) < 4000 and positions.shape[0] >= 1024 and x.dim() == 2
                and not torch.cuda.is_current_stream_capturing()):
            layer._dsv41_t0 = _ev()
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return positions.shape[0] >= max(_min_rows(), 2 * ALIGN) and x.dim() == 2


def _split_end(layer) -> None:
    t0 = layer.__dict__.pop("_dsv41_t0", None)
    if t0 is not None:
        _EV.append(("plain", (t0, _ev())))


def cuts_for(T):
    """Slice boundaries [0, ..., T] of a T-row split step."""
    spec = os.environ.get("DSV41_SPLIT_CUTS", "1024")
    if spec and spec != "even":
        # row boundaries for a 2,048-row step (scaled to T), each a multiple of DSV41_SPLIT_ALIGN (default 128)
        al = int(os.environ.get("DSV41_SPLIT_ALIGN", "128"))
        cuts = [0] + [min(T - al, max(al, (T * int(c) // 2048) // al * al)) for c in spec.split(",")] + [T]
    else:
        k = max(2, int(os.environ.get("DSV41_SPLIT_SLICES", "3")))
        k = min(k, T // ALIGN)
        cuts = [0] + [max(ALIGN, (T * i // k) // ALIGN * ALIGN) for i in range(1, k)] + [T]
    return sorted(set(cuts))


def _attn_metas(T, cuts):
    """dsv41_attn_slices' per-slice attention metadata for this step (built with these cuts), or None."""
    import sys
    m = sys.modules.get("dsv41_attn_slices")
    if m is None or os.environ.get("DSV41_ATTN_SLICES", "1") != "1":
        return None
    return m.metas_for(T, cuts)


def _split_tail(self, positions, x, input_ids, residual, post_mix, res_mix, attn_pre):
    T = positions.shape[0]
    tm = _timing() and len(_EV) < 4000
    if tm:
        evs = [_ev()]
    cuts = cuts_for(T)
    sl = tuple(slice(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a)
    side = _side(x.device)
    main = torch.cuda.current_stream()
    attn = self.attn
    o_proj = attn._o_proj                       # bound method of the class
    metas = _attn_metas(T, cuts)
    wo_b = attn.wo_b
    rr = wo_b.reduce_results
    if metas is not None and len(metas) == len(sl):
        # Attention in row slices too, each on its own attention metadata (the step cut at the same rows):
        # slice i's projection sum runs beside slice i+1's attention. Causal: slice i reads only rows < its end.
        from vllm.forward_context import get_forward_context
        fc = get_forward_context()
        full = fc.attn_metadata
        ys = []
        attn._o_proj = lambda o, pos: o
        wo_b.reduce_results = False
        try:
            tkb = attn.topk_indices_buffer
            has_idx = getattr(attn, "indexer", None) is not None
            if _TK["key"] is not metas:          # a new step: no slice indices yet
                _TK["key"], _TK["stash"] = metas, {}
            for i, s in enumerate(sl):
                n_i = s.stop - s.start
                # Indexer-less layers reuse the top-k rows the last indexer published into the shared buffer. Run
                # per slice, the buffer holds the last slice's rows: put back this slice's rows first.
                if tkb is not None and not has_idx and i in _TK["stash"]:
                    tkb[:n_i].copy_(_TK["stash"][i])
                fc.attn_metadata = metas[i]
                try:
                    o_s = attn(positions[s], x[s], None)
                finally:
                    fc.attn_metadata = full
                if tkb is not None and has_idx:
                    _TK["stash"][i] = tkb[:n_i].clone()
                if i == 0:
                    _m._dsv41_mhc_join()
                ys.append(_ar_async(o_proj(o_s, positions[s]), side))
            STATS["attn_sliced"] = STATS.get("attn_sliced", 0) + 1
        finally:
            del attn._o_proj
            wo_b.reduce_results = rr
        if tm:
            evs.append(_ev())
            evs.append(_ev())
    else:
        attn._o_proj = lambda o, pos: o             # attention core only: return o before wo_a / wo_b
        try:
            o = attn(positions, x, None)
        finally:
            del attn._o_proj
        _m._dsv41_mhc_join()        # the pre-attention mHC outputs (post / comb / pre) are safe to read from here
        if tm:
            evs.append(_ev())
        wo_b.reduce_results = False
        try:
            ys = [_ar_async(o_proj(o[s], positions[s]), side) for s in sl]
        finally:
            wo_b.reduce_results = rr
    if tm:
        evs.append(_ev())
    outs, ress, pms, rms, fps = [], [], [], [], []
    for s, (y, ev) in zip(sl, ys):
        main.wait_event(ev)
        if _m._dsv41_mhc_mode(residual[s], attn_pre[s], False) == 1:
            # fused mHC post + pre (token-blocked; its sinkhorn is deferred to the join below)
            res_s, pm_s, rm_s, x_s, fp_s = _m._dsv41_mhc_post_pre(
                y, residual[s], post_mix[s], res_mix[s], attn_pre[s], self.hc_ffn_fn, self.hc_ffn_scale,
                self.hc_ffn_base, self.rms_norm_eps, self.hc_eps, self.hc_eps, self.hc_post_alpha,
                self.hc_sinkhorn_iters, self.ffn_norm.weight, self.ffn_norm.variance_epsilon)
            STATS["fused_mhc"] = STATS.get("fused_mhc", 0) + 1
        else:
            res_s = _m.mhc_post_tilelang(y, residual[s], post_mix[s], res_mix[s])
            pm_s, rm_s, x_s, fp_s = _m.mhc_pre_delayed_tilelang(
                res_s, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.rms_norm_eps, self.hc_eps,
                self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters, pre_mix=attn_pre[s],
                norm_weight=self.ffn_norm.weight, norm_eps=self.ffn_norm.variance_epsilon)
        _DEFER[0], _PREREDUCED[0] = True, False
        try:
            f_s = self.ffn(x_s, input_ids[s] if input_ids is not None else None)
        finally:
            _DEFER[0] = False
        if _PREREDUCED[0]:
            STATS["ffn_prereduced"] += 1
            outs.append((f_s, None))
        else:
            outs.append(_ar_async(f_s, side))
        ress.append(res_s)
        pms.append(pm_s)
        rms.append(rm_s)
        fps.append(fp_s)
    if tm:
        evs.append(_ev())
    for f, ev in outs:
        if ev is not None:
            main.wait_event(ev)
    if tm:
        evs.append(_ev())
        _EV.append(("split", evs + [e for _, e in ys] + [e for _, e in outs if e is not None]))
    STATS["split_steps"] += 1
    _m._dsv41_mhc_join()        # deferred sinkhorn outputs of every slice written before they are concatenated
    x = torch.cat([f for f, _ in outs], 0)
    return x, torch.cat(ress, 0), torch.cat(pms, 0), torch.cat(rms, 0), torch.cat(fps, 0)


# ----------------------------------------------------------------------------------------------------------------
# Cross-layer pipeline (DSV41_PIPE, default on with dsv41_attn_slices): a split layer runs every stage per row slice
# (pre-attention mHC, attention on the slice's own metadata, projection) and returns before its FFN slices' sums
# finish; the next layer waits for slice i's sum only when it reaches slice i. Slice 1 of layer L+1 thus runs beside
# the last sums of layer L. Layers whose output is read outside a layer (aux hidden states, the last layer) and
# Engram layers (full-row lookup) join every slice first.
_PEND: dict = {}          # data_ptr of a returned FFN output -> [done event per slice | None]
_JOIN: dict = {"ids": set(), "model": None}


def _pipe_join(x):
    p = _PEND.pop(x.data_ptr(), None) if isinstance(x, torch.Tensor) else None
    if p:
        main = torch.cuda.current_stream()
        for ev in p:
            if ev is not None:
                main.wait_event(ev)


def _pipe_ok(layer, x, positions, residual, engram_hashes) -> bool:
    ok = (os.environ.get("DSV41_PIPE", "1") == "1" and residual is not None and x.dim() == 2 and _enabled()
          and not getattr(layer, "use_sequence_parallel", False) and not torch.cuda.is_current_stream_capturing()
          and not (layer.engram is not None and engram_hashes is not None))
    if ok:
        T = positions.shape[0]
        ok = T >= max(_min_rows(), 2 * ALIGN)
        if ok:
            cuts = cuts_for(T)
            metas = _attn_metas(T, cuts)
            ok = metas is not None and len(metas) == len(cuts) - 1
    if not ok:
        _pipe_join(x)
    return ok


def _ar_async_into(t, side, dst):
    """All-reduce t on the side stream into dst (a view of the layer's output); returns the done event."""
    main = torch.cuda.current_stream()
    ready = torch.cuda.Event()
    ready.record(main)
    ring = _ce_ring()
    with torch.cuda.stream(side):
        side.wait_event(ready)
        if ring is not None and ring.takes(t):
            r = ring.all_reduce_owner(t)
            STATS["ce_calls"] = STATS.get("ce_calls", 0) + 1
        else:
            r = tensor_model_parallel_all_reduce(t)
        dst.copy_(r)
    t.record_stream(side)
    done = torch.cuda.Event()
    done.record(side)
    return done


def _pipe_layer(self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, engram_hashes, engram_mask):
    from vllm.forward_context import get_forward_context
    T = positions.shape[0]
    cuts = cuts_for(T)
    sl = tuple(slice(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a)
    metas = _attn_metas(T, cuts)
    pend = _PEND.pop(x.data_ptr(), None)
    side = _side(x.device)
    main = torch.cuda.current_stream()
    attn = self.attn
    o_proj = attn._o_proj
    wo_b = attn.wo_b
    rr = wo_b.reduce_results
    fc = get_forward_context()
    full = fc.attn_metadata
    tkb = attn.topk_indices_buffer
    has_idx = getattr(attn, "indexer", None) is not None
    if _TK["key"] is not metas:
        _TK["key"], _TK["stash"] = metas, {}
    a_res, a_pm, a_rm, a_ap, ys = [], [], [], [], []
    attn._o_proj = lambda o, pos: o
    wo_b.reduce_results = False
    try:
        for i, s in enumerate(sl):
            if pend is not None and pend[i] is not None:
                main.wait_event(pend[i])
            xs, rs, pms, rms, pre_s = x[s], residual[s], post_mix[s], res_mix[s], pre_mix[s]
            mode = _m._dsv41_mhc_mode(rs, pre_s, False)
            if mode != 1:
                rs = _m.mhc_post_tilelang(xs, rs, pms, rms)
            if mode:
                r_, pm_, rm_, xa, ap_ = _m._dsv41_mhc_post_pre(
                    xs if mode == 1 else None, rs, pms, rms, pre_s, self.hc_attn_fn, self.hc_attn_scale,
                    self.hc_attn_base, self.rms_norm_eps, self.hc_eps, self.hc_eps, self.hc_post_alpha,
                    self.hc_sinkhorn_iters, self.attn_norm.weight, self.attn_norm.variance_epsilon, do_post=mode == 1)
            else:
                r_ = rs
                pm_, rm_, xa, ap_ = _m.mhc_pre_delayed_tilelang(
                    rs, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.rms_norm_eps, self.hc_eps,
                    self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters, pre_mix=pre_s,
                    norm_weight=self.attn_norm.weight, norm_eps=self.attn_norm.variance_epsilon)
            n_i = s.stop - s.start
            if tkb is not None and not has_idx and i in _TK["stash"]:
                tkb[:n_i].copy_(_TK["stash"][i])
            fc.attn_metadata = metas[i]
            try:
                o_s = attn(positions[s], xa, None)
            finally:
                fc.attn_metadata = full
            if tkb is not None and has_idx:
                _TK["stash"][i] = tkb[:n_i].clone()
            _m._dsv41_mhc_join()
            ys.append(_ar_async(o_proj(o_s, positions[s]), side))
            a_res.append(r_)
            a_pm.append(pm_)
            a_rm.append(rm_)
            a_ap.append(ap_)
    finally:
        del attn._o_proj
        wo_b.reduce_results = rr
    x_out = torch.empty_like(x)
    x_out.record_stream(side)
    evs, ress, pmsl, rmsl, fps = [], [], [], [], []
    for i, (s, (y, ev)) in enumerate(zip(sl, ys)):
        main.wait_event(ev)
        if _m._dsv41_mhc_mode(a_res[i], a_ap[i], False) == 1:
            res_s, pm_s, rm_s, x_s, fp_s = _m._dsv41_mhc_post_pre(
                y, a_res[i], a_pm[i], a_rm[i], a_ap[i], self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
                self.rms_norm_eps, self.hc_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters,
                self.ffn_norm.weight, self.ffn_norm.variance_epsilon)
        else:
            res_s = _m.mhc_post_tilelang(y, a_res[i], a_pm[i], a_rm[i])
            pm_s, rm_s, x_s, fp_s = _m.mhc_pre_delayed_tilelang(
                res_s, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.rms_norm_eps, self.hc_eps,
                self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters, pre_mix=a_ap[i],
                norm_weight=self.ffn_norm.weight, norm_eps=self.ffn_norm.variance_epsilon)
        _DEFER[0], _PREREDUCED[0] = True, False
        try:
            f_s = self.ffn(x_s, input_ids[s] if input_ids is not None else None)
        finally:
            _DEFER[0] = False
        if _PREREDUCED[0]:
            x_out[s].copy_(f_s)
            evs.append(None)
        else:
            evs.append(_ar_async_into(f_s, side, x_out[s]))
        ress.append(res_s)
        pmsl.append(pm_s)
        rmsl.append(rm_s)
        fps.append(fp_s)
    _m._dsv41_mhc_join()
    out = (x_out, torch.cat(ress, 0), torch.cat(pmsl, 0), torch.cat(rmsl, 0), torch.cat(fps, 0))
    if id(self) in _JOIN["ids"] or os.environ.get("DSV41_PIPE_DEFER", "1") != "1":
        for e in evs:
            if e is not None:
                main.wait_event(e)
    else:
        _PEND[x_out.data_ptr()] = evs
    STATS["pipe_layers"] = STATS.get("pipe_layers", 0) + 1
    return out


_DM = _m.DeepseekV4Model
_ORIG_MODEL_FWD = _DM.__dict__.get("_dsv41_pipe_orig_fwd") or _DM.forward
_DM._dsv41_pipe_orig_fwd = _ORIG_MODEL_FWD


def model_forward(self, *a, **k):
    """Marks the layers whose output is read outside a layer, and joins anything left pending (safety)."""
    if _JOIN["model"] is not self:
        from itertools import islice
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        aux = set(getattr(self, "aux_hidden_state_layers", ()) or ())
        ids = {id(l) for n, l in enumerate(layers, start=self.start_layer) if (n + 1) in aux}
        if layers:
            ids.add(id(layers[-1]))
        _JOIN["ids"], _JOIN["model"] = ids, self
    out = _ORIG_MODEL_FWD(self, *a, **k)
    if _PEND:
        main = torch.cuda.current_stream()
        for evs in _PEND.values():
            for e in evs:
                if e is not None:
                    main.wait_event(e)
        STATS["pipe_leftover"] = STATS.get("pipe_leftover", 0) + 1
        _PEND.clear()
    return out


def _patched_forward():
    src = textwrap.dedent(inspect.getsource(_ORIG_FWD))
    pat = re.compile(r"^( *)x = self\.attn\(positions, x, None\)\n", re.M)
    hits = pat.findall(src)
    if len(hits) != 1:
        raise RuntimeError(f"dsv41_split_reduce: attention call found {len(hits)}x, want 1")
    ind = hits[0]
    hook = (f"{ind}if _dsv41_split_ok(self, x, positions):\n"
            f"{ind}    return _dsv41_split_tail(self, positions, x, input_ids, residual, post_mix, res_mix, attn_pre)\n")
    src = pat.sub(lambda m: hook + m.group(0), src)
    ppat = re.compile(r"^( *)if residual is None:\n", re.M)
    ph = ppat.findall(src)
    if len(ph) != 1:
        raise RuntimeError(f"dsv41_split_reduce: first statement found {len(ph)}x, want 1")
    pind = ph[0]
    phook = (f"{pind}if _dsv41_pipe_ok(self, x, positions, residual, engram_hashes):\n"
             f"{pind}    return _dsv41_pipe_layer(self, x, positions, input_ids, pre_mix, post_mix, res_mix, residual, "
             f"engram_hashes, engram_mask)\n")
    src = ppat.sub(lambda m: phook + m.group(0), src)
    rpat = re.compile(r"^( *)return x, residual, post_mix, res_mix, ffn_pre\n", re.M)
    rh = rpat.findall(src)
    if len(rh) != 1:
        raise RuntimeError(f"dsv41_split_reduce: final return found {len(rh)}x, want 1")
    src = rpat.sub(lambda m: f"{m.group(1)}_dsv41_split_end(self)\n" + m.group(0), src)
    ns = dict(vars(_m))
    ns["_dsv41_split_ok"] = _split_ok
    ns["_dsv41_split_tail"] = _split_tail
    ns["_dsv41_split_end"] = _split_end
    ns["_dsv41_pipe_ok"] = _pipe_ok
    ns["_dsv41_pipe_layer"] = _pipe_layer
    exec(compile(src, "<dsv41_split_reduce forward>", "exec"), ns)
    return ns["forward"]


layer_forward = _patched_forward()
if not re.search(r"attn_pre", inspect.getsource(_ORIG_FWD)):
    raise RuntimeError("dsv41_split_reduce: attn_pre not in the layer forward")


def stats(worker=None) -> dict:
    torch.cuda.synchronize()
    r = dict(STATS)
    sp = [e for k, e in _EV if k == "split"]
    if sp:
        n = len(sp)
        el = lambda i, j: sum(e[i].elapsed_time(e[j]) for e in sp) / n   # noqa: E731
        r["split_layer_ms"] = round(el(0, 4), 3)
        r["attn_core_ms"] = round(el(0, 1), 3)
        r["proj_ms"] = round(el(1, 2), 3)
        r["mhc_ffn_ms"] = round(el(2, 3), 3)
        r["final_wait_ms"] = round(el(3, 4), 3)
        r["n_split"] = n
    pl = [e for k, e in _EV if k == "plain"]
    if pl:
        r["plain_layer_ms"] = round(sum(a.elapsed_time(b) for a, b in pl) / len(pl), 3)
        r["n_plain"] = len(pl)
    _EV.clear()
    for k in STATS:
        STATS[k] = 0
    return r


SETATTR = [{"target": "vllm.models.deepseek_v4_1.nvidia.model.DeepseekV4DecoderLayer.forward",
            "source": "dsv41_split_reduce.layer_forward"},
           {"target": "vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner._maybe_reduce_final_output",
            "source": "dsv41_split_reduce.maybe_reduce_final_output"},
           {"target": "vllm.models.deepseek_v4_1.nvidia.model._DSV41_MHC_FUSED_MAX",
            "source": "dsv41_split_reduce.MHC_MAX"},
           {"target": "vllm.models.deepseek_v4_1.nvidia.model.DeepseekV4Model.forward",
            "source": "dsv41_split_reduce.model_forward"}]
