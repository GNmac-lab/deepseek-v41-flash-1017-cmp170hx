# dsv41 (runtime variant): routed-expert prefill MoE on Morrowmake's second tile schedule.
#
# moe_out_prefill_try is csrc/exl3/moe_prefill_try.cu of Morrowmake/vllm-cmp170hx@a0c57b5 (Apache-2.0; codec from
# exllamav3, MIT), instantiated here for our tensor-parallel expert shapes (hidden 5120, intermediate 640 / 512,
# bits_x2 6 / 4; 640 keeps one n16 tile per warp in the gate/up GEMM). Same kernels and arithmetic as the production
# moe_out_prefill, with a faster schedule: experts tiled in decreasing row count, the gate/up operand computed once
# per row by a block, 2-warp down-operand blocks, 2 GEMM blocks per SM, one launch sequence up to 2,048 rows.
# Output is bitwise equal to moe_out_prefill (checked: 192 cases, 65-2,048 rows, all four shapes, skewed routing).
# Measured per call, slowest rank (I 640, K3): 1,024 rows 6.86 -> 5.26 ms, 2,048 rows 8.39 -> 7.63 ms.
# It takes the hook's prefill range (65 .. VLLM_EXL3_TC_PREFILL_ROWS rows); every other call keeps the hook.
#   DSV41_MOE_TRY=0 off; DSV41_MOE_TRY_SO / DSV41_MOE_TRY_NS: library path and op namespace.
import os

import torch

from vllm_exl3 import exl3 as _x

SO = os.environ.get("DSV41_MOE_TRY_SO", "/opt/dsv41-tc-moe/_dsv41_mm_try.abi3.so")
NS = os.environ.get("DSV41_MOE_TRY_NS", "_dsv41_mm_try")
STATS = {"try_calls": 0}

_HOOK = _x._ds41_tc_hook()
if _HOOK is None:
    raise RuntimeError("dsv41_moe_try: the EXL3 TC MoE hook is not loaded")
_ORIG = _HOOK.__dict__.get("_dsv41_try_orig_apply") or _HOOK.try_apply
_HOOK._dsv41_try_orig_apply = _ORIG
torch.ops.load_library(SO)
_OPS = getattr(torch.ops, NS)
_NEED = int(_OPS.moe_workspace_size_prefill_try(_HOOK._TC_PREFILL_ROWS, 5120, 640, 6, 384))
for _dev, _ws in _HOOK._SHARED_WORKSPACE.items():
    if _ws.numel() < _NEED:
        raise RuntimeError(f"dsv41_moe_try: workspace on cuda:{_dev} has {_ws.numel()} elements, needs {_NEED}")


def try_apply(x2d, ids, weights, layer, expert_map, limit):
    """The hook's try_apply with its prefill range on moe_out_prefill_try."""
    tc = getattr(layer, "_exl3_tc", None)
    if (os.environ.get("DSV41_MOE_TRY", "1") != "1" or tc is None or not tc["prefill"] or x2d.dim() != 2
            or x2d.dtype != torch.bfloat16 or not x2d.is_cuda or int(ids.shape[-1]) != 6):
        return _ORIG(x2d, ids, weights, layer, expert_map, limit)
    rows = int(x2d.shape[0])
    if not 65 <= rows <= tc["prefill_rows"]:
        return _ORIG(x2d, ids, weights, layer, expert_map, limit)
    emap = tc["emap"] if expert_map is None else expert_map
    if emap.device != x2d.device:
        return _ORIG(x2d, ids, weights, layer, expert_map, limit)
    out = torch.empty_like(x2d)
    w = weights if weights.dtype == torch.float32 else weights.float()
    _OPS.moe_out_prefill_try(x2d.contiguous(), ids, w.contiguous(), tc["ptrs"], tc["meta"], emap,
                             tc["intermediate"], tc["bits_x2"], out, tc["workspace"], float(limit or 0.0))
    STATS["try_calls"] += 1
    return out


def stats(worker=None) -> dict:
    r = dict(STATS)
    STATS["try_calls"] = 0
    return r


SETATTR = [{"target": "vllm_exl3.exl3._DS41_TC_HOOK.try_apply", "source": "dsv41_moe_try.try_apply"}]
