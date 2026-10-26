# dsv41 (runtime variant): per-slice attention metadata for the split prefill layer (dsv41_split_reduce).
#
# A prefill step of >= DSV41_SPLIT_MIN_ROWS rows is cut at dsv41_split_reduce.cuts_for(T). After the runner builds the
# step's attention metadata, this builds it again for each row slice, through vLLM's own micro-batch path
# (_build_attention_metadata with ubatch_slices: split_attn_metadata + every builder), on separate builder instances
# so the full metadata's persistent buffers are untouched. Slice i is exactly the step cut at its last row: its
# queries see the KV of every earlier row. dsv41_split_reduce then runs attention per slice, so slice i's output
# projection sum overlaps slice i+1's attention.
#   DSV41_ATTN_SLICES=0 off.
import copy
import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.worker import gpu_model_runner as _g
from vllm.v1.worker import ubatch_utils as _uu
from vllm.v1.worker.ubatch_utils import UBatchSlice

logger = init_logger(__name__)
_R = _g.GPUModelRunner
_ORIG = _R.__dict__.get("_dsv41_as_orig") or _R._build_attention_metadata
_R._dsv41_as_orig = _ORIG
STATE = {"metas": None}
_EXTRA: dict = {}
STATS = {"built": 0, "skipped": 0, "errors": 0}


def _cuts(T):
    import sys
    m = sys.modules.get("dsv41_split_reduce")
    if m is None or not hasattr(m, "cuts_for"):
        return None
    if os.environ.get("DSV41_SPLIT", "1") != "1" or T < max(int(os.environ.get("DSV41_SPLIT_MIN_ROWS", "1024")), 1024):
        return None
    return m.cuts_for(T)


_ORIG_MK = _uu.__dict__.get("_dsv41_as_orig_mk") or _uu._make_metadata_with_slice
_uu._dsv41_as_orig_mk = _ORIG_MK


def _make_metadata_with_slice(ubatch_slice, attn_metadata):
    """vLLM's slice plus the fields it drops (positions, is_prefilling) and the computed-token count of a request
    whose first rows belong to the previous slice."""
    if attn_metadata.mm_req_doc_ranges and any(attn_metadata.mm_req_doc_ranges.values()):
        raise RuntimeError("dsv41_attn_slices: image doc ranges not sliced")   # an image prompt stays unsliced
    out = _ORIG_MK(ubatch_slice, attn_metadata)
    rs, ts = ubatch_slice.request_slice, ubatch_slice.token_slice
    if attn_metadata.mm_req_doc_ranges is not None:
        out.mm_req_doc_ranges = {i - rs.start: v for i, v in attn_metadata.mm_req_doc_ranges.items()
                                 if rs.start <= i < rs.stop}
    if attn_metadata.positions is not None:
        out.positions = attn_metadata.positions[ts]
    if attn_metadata.rswa_prefix_lens is not None:
        out.rswa_prefix_lens = attn_metadata.rswa_prefix_lens[rs]
    if attn_metadata.is_prefilling is not None:
        out.is_prefilling = attn_metadata.is_prefilling[rs]
    skipped = int(ts.start - attn_metadata.query_start_loc_cpu[rs.start])
    if skipped > 0 and out._num_computed_tokens_cpu is not None:
        nc = out._num_computed_tokens_cpu.clone()
        nc[0] += skipped
        out._num_computed_tokens_cpu = nc
    return out


def _extra_builders(runner, n):
    kbs_all = getattr(runner, "_kernel_block_sizes", None) or []
    for gid, groups in enumerate(runner.attn_groups):
        kbs = kbs_all[gid] if gid < len(kbs_all) else None
        for g in groups:
            lst = _EXTRA.get(id(g))
            if lst is None or len(lst) < n:
                g2 = copy.copy(g)
                g2.create_metadata_builders(runner.vllm_config, runner.device, kbs, num_metadata_builders=n)
                _EXTRA[id(g)] = g2.metadata_builders


def build(self, *a, **k):
    import time as _t0m
    _t00 = _t0m.perf_counter()
    out = _ORIG(self, *a, **k)
    STATS["full_build_ms"] = STATS.get("full_build_ms", 0.0) + (_t0m.perf_counter() - _t00) * 1000
    STATE["metas"] = None
    if (os.environ.get("DSV41_ATTN_SLICES", "1") != "1" or a or k.get("ubatch_slices") is not None
            or k.get("for_cudagraph_capture") or torch.cuda.is_current_stream_capturing()):
        return out
    T, R, mq = k.get("num_tokens"), k.get("num_reqs"), k.get("max_query_len")
    if T is None or R is None or mq is None or mq <= 6 or k.get("num_tokens_padded") not in (None, T):
        return out
    cuts = _cuts(T)
    if not cuts or len(cuts) < 3:
        STATS["skipped"] += 1
        return out
    try:
        qsl = self.query_start_loc.np[: R + 1]
        slices = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            rs = int(np.searchsorted(qsl, lo, side="right") - 1)
            re_ = int(np.searchsorted(qsl, hi, side="left"))
            slices.append(UBatchSlice(slice(rs, re_), slice(lo, hi)))
        _extra_builders(self, len(slices))
        saved = {}
        for groups in self.attn_groups:
            for g in groups:
                saved[id(g)] = g.metadata_builders
                g.metadata_builders = _EXTRA[id(g)][: len(slices)]
        _uu._make_metadata_with_slice = _make_metadata_with_slice
        import time as _t
        t0 = _t.perf_counter()
        try:
            metas, _ = _ORIG(self, *a, **{**k, "ubatch_slices": slices})
            STATS["build_ms"] = STATS.get("build_ms", 0.0) + (_t.perf_counter() - t0) * 1000
        finally:
            _uu._make_metadata_with_slice = _ORIG_MK
            for groups in self.attn_groups:
                for g in groups:
                    g.metadata_builders = saved[id(g)]
        STATE["metas"] = (T, tuple(cuts), metas)
        STATS["built"] += 1
    except Exception as e:  # noqa: BLE001  the step runs unsliced
        STATS["errors"] += 1
        if STATS["errors"] <= 2:
            import traceback
            STATE["last_error"] = traceback.format_exc()[-3000:]
            logger.warning("dsv41_attn_slices: step left unsliced: %s: %s", type(e).__name__, e)
    return out


def metas_for(T, cuts):
    st = STATE["metas"]
    if st is None or st[0] != T or st[1] != tuple(cuts):
        return None
    return st[2]


def stats(worker=None) -> dict:
    r = dict(STATS)
    r["last_error"] = STATE.get("last_error")
    for k_ in STATS:
        STATS[k_] = 0
    return r


SETATTR = [{"target": "vllm.v1.worker.gpu_model_runner.GPUModelRunner._build_attention_metadata",
            "source": "dsv41_attn_slices.build"}]
