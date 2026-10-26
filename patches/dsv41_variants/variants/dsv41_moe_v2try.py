# dsv41 (runtime variant): decode-sized routed-expert MoE on Morrowmake's faster moe_out_v2 (moe_v2_try.cu).
#
# moe_out_v2_try is csrc/exl3/moe_v2_try.cu of Morrowmake/vllm-cmp170hx@a0c57b5 (Apache-2.0; codec from exllamav3,
# MIT): moe_out_v2 with its lane-GEMM trellis words through a per-warp cp.async shared ring, 8-tile column groups and
# chunked activation loads. Built here for our tensor-parallel expert shapes (hidden 5120, intermediate 640 / 512,
# bits_x2 6 / 4) with the slice layout of the moe_out_v2 we served, so its output is BITWISE equal to moe_out_v2
# (checked at 6..64 rows, K3 and K2). Per call, slowest rank: 48 rows 2.48 -> 1.51 ms (K3), 1.86 -> 1.40 ms (K2).
# It serves every routed-expert call of DSV41_V2TRY_MIN (1) .. 64 rows: it also beats the coop kernel the hook used for
# 1..42 rows (1-user step 17.6 -> 16.6 ms). Coop -> v2 changes rounding for those rows: short-prompt NLL
# +0.57 % (411 passages, t 1.7) and -0.22 % (900 passages, t -1.08), i.e. equivalent.
# MiaAI same boot: 1u 338/176 -> 359/188 (structured/prose), 8u 830/514/379 -> 875/542/422.
#   DSV41_V2TRY=0 off; DSV41_V2TRY_SO / DSV41_V2TRY_NS library path / op namespace.
import os

import torch

from vllm_exl3 import exl3 as _x

SO = os.environ.get("DSV41_V2TRY_SO", "/opt/dsv41-tc-moe/_dsv41_mm_v2try.abi3.so")
NS = os.environ.get("DSV41_V2TRY_NS", "_dsv41_mm_v2try")
STATS = {"v2try_calls": 0}

_HOOK = _x._ds41_tc_hook()
if _HOOK is None:
    raise RuntimeError("dsv41_moe_v2try: the EXL3 TC MoE hook is not loaded")
_PREV = _HOOK.__dict__.get("_dsv41_v2try_prev") or _HOOK.try_apply
_HOOK._dsv41_v2try_prev = _PREV
torch.ops.load_library(SO)
_OPS = getattr(torch.ops, NS)
_NEED = int(_OPS.moe_workspace_size_v2_try(64, 5120, 640, 6, 384))
for _dev, _ws in _HOOK._SHARED_WORKSPACE.items():
    if _ws.numel() < _NEED:
        raise RuntimeError(f"dsv41_moe_v2try: workspace on cuda:{_dev} has {_ws.numel()} elements, needs {_NEED}")


def try_apply(x2d, ids, weights, layer, expert_map, limit):
    """The hook's try_apply with its decode v2 range on moe_out_v2_try."""
    tc = getattr(layer, "_exl3_tc", None)
    if (os.environ.get("DSV41_V2TRY", "1") != "1" or tc is None or x2d.dim() != 2
            or x2d.dtype != torch.bfloat16 or not x2d.is_cuda or int(ids.shape[-1]) != 6):
        return _PREV(x2d, ids, weights, layer, expert_map, limit)
    rows = int(x2d.shape[0])
    if not int(os.environ.get("DSV41_V2TRY_MIN", "1")) <= rows <= 64:
        return _PREV(x2d, ids, weights, layer, expert_map, limit)
    emap = tc["emap"] if expert_map is None else expert_map
    if emap.device != x2d.device:
        return _PREV(x2d, ids, weights, layer, expert_map, limit)
    out = torch.empty_like(x2d)
    w = weights if weights.dtype == torch.float32 else weights.float()
    _OPS.moe_out_v2_try(x2d.contiguous(), ids, w.contiguous(), tc["ptrs"], tc["meta"], emap, tc["intermediate"],
                        tc["bits_x2"], out, tc["workspace"], float(limit or 0.0))
    if not torch.cuda.is_current_stream_capturing():
        STATS["v2try_calls"] += 1
    return out


def stats(worker=None) -> dict:
    r = dict(STATS)
    STATS["v2try_calls"] = 0
    return r


SETATTR = [{"target": "vllm_exl3.exl3._DS41_TC_HOOK.try_apply", "source": "dsv41_moe_v2try.try_apply"}]
