# dsv41 (runtime variant): drop host -> device copies that repeat the same bytes every decode step.
#
# A decode step issued 27 small H2D copies (~6 us each on PCIe Gen2, mostly latency). Two kinds repeat themselves:
#   - fresh read-only index tensors (spec-decode metadata via async_tensor_h2d; the scatter indices of
#     _prepare_input_ids): equal bytes -> the GPU tensor from an earlier step is returned (content-keyed cache).
#   - BlockTable.commit_block_table: every KV-cache group's rows go up every step; they change only when a block is
#     added. Rows equal to the last commit -> no copy (only clear() writes the device table, and it resets this).
# Same values reach every kernel: identical outputs.
import collections
import inspect
import re
import textwrap

import numpy as np
import torch

from vllm.v1.worker import block_table as _bt
from vllm.v1.worker import gpu_model_runner as _gmr

MAX_ENTRIES = 512
_CACHE: "collections.OrderedDict" = collections.OrderedDict()
STATS = {"hit": 0, "miss": 0, "bt_skip": 0, "bt_copy": 0}

_ORIG_H2D = getattr(_gmr, "_dsv41_orig_async_tensor_h2d", None) or _gmr.async_tensor_h2d
_gmr._dsv41_orig_async_tensor_h2d = _ORIG_H2D
_ORIG_PIDS = _gmr.GPUModelRunner.__dict__.get("_dsv41_orig__prepare_input_ids") or _gmr.GPUModelRunner._prepare_input_ids
_gmr.GPUModelRunner._dsv41_orig__prepare_input_ids = _ORIG_PIDS
_ORIG_COMMIT = _bt.BlockTable.__dict__.get("_dsv41_orig_commit") or _bt.BlockTable.commit_block_table
_bt.BlockTable._dsv41_orig_commit = _ORIG_COMMIT
_ORIG_CLEAR = _bt.BlockTable.__dict__.get("_dsv41_orig_clear") or _bt.BlockTable.clear
_bt.BlockTable._dsv41_orig_clear = _ORIG_CLEAR


def _key(data, dtype, device):
    if isinstance(data, torch.Tensor):
        if data.device.type != "cpu":
            return None
        a = data.contiguous().numpy()
    elif isinstance(data, np.ndarray):
        a = np.ascontiguousarray(data)
    else:
        a = np.asarray(data)
    return (a.tobytes(), a.shape, str(a.dtype), str(dtype), str(device))


def _cached(key, make):
    if key is None:
        return make()
    t = _CACHE.get(key)
    if t is None:
        STATS["miss"] += 1
        t = _CACHE[key] = make()
        if len(_CACHE) > MAX_ENTRIES:
            _CACHE.popitem(last=False)
    else:
        STATS["hit"] += 1
        _CACHE.move_to_end(key)
    return t


def async_tensor_h2d(data, device, dtype=None):
    return _cached(_key(data, dtype, device), lambda: _ORIG_H2D(data, device, dtype))


def _h2d_i64(data, device):
    return _cached(_key(np.asarray(data, dtype=np.int64), torch.int64, device),
                   lambda: torch.tensor(data, dtype=torch.int64, pin_memory=_gmr.PIN_MEMORY).to(device, non_blocking=True))


def _patched_prepare_input_ids():
    src = textwrap.dedent(inspect.getsource(_ORIG_PIDS))
    pat = re.compile(r"torch\.tensor\(\s*(\w+),\s*dtype=torch\.int64,\s*pin_memory=PIN_MEMORY\s*\)"
                     r"\.to\(self\.device,\s*non_blocking=True\)")
    new, n = pat.subn(r"_dsv41_h2d_i64(\1, self.device)", src)
    if n != 4:
        raise RuntimeError(f"dsv41_h2d_lean: expected 4 index uploads in _prepare_input_ids, found {n}")
    ns = dict(vars(_gmr))
    ns["_dsv41_h2d_i64"] = _h2d_i64
    exec(compile(new, "<dsv41_h2d_lean _prepare_input_ids>", "exec"), ns)
    return ns["_prepare_input_ids"]


prepare_input_ids = _patched_prepare_input_ids()


def commit_block_table(self, num_reqs: int) -> None:
    rows = self.block_table.np[:num_reqs]
    sh = self.__dict__.get("_dsv41_shadow")
    if sh is not None and sh.shape == rows.shape and np.array_equal(sh, rows):
        STATS["bt_skip"] += 1
        return
    _ORIG_COMMIT(self, num_reqs)
    self._dsv41_shadow = rows.copy()
    STATS["bt_copy"] += 1


def clear(self) -> None:
    self.__dict__.pop("_dsv41_shadow", None)
    _ORIG_CLEAR(self)


def stats(worker=None) -> dict:
    r = dict(STATS, entries=len(_CACHE))
    for k in STATS:
        STATS[k] = 0
    return r


_P = "vllm.v1.worker."
SETATTR = [{"target": _P + "gpu_model_runner.async_tensor_h2d", "source": "dsv41_h2d_lean.async_tensor_h2d"},
           {"target": _P + "gpu_model_runner.GPUModelRunner._prepare_input_ids", "source": "dsv41_h2d_lean.prepare_input_ids"},
           {"target": _P + "block_table.BlockTable.commit_block_table", "source": "dsv41_h2d_lean.commit_block_table"},
           {"target": _P + "block_table.BlockTable.clear", "source": "dsv41_h2d_lean.clear"}]
