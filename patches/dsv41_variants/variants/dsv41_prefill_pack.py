# dsv41 (runtime variant): sparse-attention prefill gathers packed by the real context of each request.
#
# The sm80 (DeepInfer rocm.py) prefill path gathers each request's compressed + sliding-window K into a workspace of
# DSV41_PREFILL_CHUNK_REQS (1) request slots of the worst-case width (max_model_len / ratio + window + batched rows)
# and loops one request at a time, so a step that prefills 7 short prompts runs the gather / combine / attention
# kernels 7x per layer (~225 ms of CPU-bound launches at 8 users). Here the loop follows upstream's
# get_prefill_chunk_plan (the nvidia FlashMLA path's planner): consecutive requests share one gather while their
# summed real widths fit the same workspace, each chunk laid out with its own width (chunk_N, chunk_M). No workspace
# growth; every request's attention reads the same K rows with the same indices relative to its own block.
#   DSV41_PREFILL_PACK=0 off.
import inspect
import os
import re
import textwrap

from vllm.models.deepseek_v4_1.amd import rocm as _r

_C = _r.DeepseekV41ROCMAiterMLAAttention
_ORIG = _C.__dict__.get("_dsv41_pack_orig") or _C._forward_prefill
_C._dsv41_pack_orig = _ORIG
STATS = {"packed_calls": 0, "chunks": 0, "requests": 0}


def _plan(swa_metadata, ratio, swa_only, pcs):
    cache = swa_metadata.__dict__.setdefault("_dsv41_pack_plans", {})
    key = (ratio, swa_only, pcs)
    p = cache.get(key)
    if p is None:
        p = cache[key] = swa_metadata.get_prefill_chunk_plan(compress_ratio=ratio if not swa_only else 1,
                                                           prefill_chunk_size=pcs, has_compressed=not swa_only)
    return p


def _patched():
    src = inspect.getsource(_ORIG)          # class-indented; dedented after the anchors are replaced
    old = re.compile(
        r"        num_chunks = \(num_prefills \+ self\.PREFILL_CHUNK_SIZE - 1\) // \(\n"
        r"            self\.PREFILL_CHUNK_SIZE\n"
        r"        \)\n\n"
        r"        workspace_manager = current_workspace_manager\(\)\n"
        r"        kv = workspace_manager\.get_simultaneous\(\n"
        r"            \(\(self\.PREFILL_CHUNK_SIZE, M, q\.shape\[-1\]\), torch\.bfloat16\),\n"
        r"        \)\[0\]\n"
        r"        for chunk_idx in range\(num_chunks\):\n"
        r"            chunk_start = chunk_idx \* self\.PREFILL_CHUNK_SIZE\n"
        r"            chunk_end = min\(chunk_start \+ self\.PREFILL_CHUNK_SIZE, num_prefills\)\n"
        r"            chunk_size = chunk_end - chunk_start\n")
    if len(old.findall(src)) != 1:
        raise RuntimeError("dsv41_prefill_pack: prefill loop anchor not found")
    new = (
        "        workspace_manager = current_workspace_manager()\n"
        "        kv_full = workspace_manager.get_simultaneous(\n"
        "            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),\n"
        "        )[0]\n"
        "        _cap = kv_full.shape[0] * kv_full.shape[1]\n"
        "        _pl = _dsv41_pack_plan(swa_metadata, self.compress_ratio, swa_only, self.PREFILL_CHUNK_SIZE)\n"
        "        _dsv41_pack_stats(len(_pl), num_prefills)\n"
        "        for chunk_start, chunk_end, N, M in _pl:\n"
        "            chunk_size = chunk_end - chunk_start\n"
        "            assert chunk_size * M <= _cap, (chunk_size, M, _cap)\n"
        "            kv = kv_full.view(-1, q.shape[-1])[: chunk_size * M].view(chunk_size, M, q.shape[-1])\n")
    src = old.sub(lambda m: new, src)
    a = "                    num_rows=self.PREFILL_CHUNK_SIZE * M,\n"
    if src.count(a) != 1:
        raise RuntimeError("dsv41_prefill_pack: num_rows anchor not found")
    src = src.replace(a, "                    num_rows=chunk_size * M,\n")
    src = textwrap.dedent(src)
    ns = dict(vars(_r))
    ns["_dsv41_pack_plan"] = _plan
    ns["_dsv41_pack_stats"] = _stats
    exec(compile(src, "<dsv41_prefill_pack _forward_prefill>", "exec"), ns)
    return ns["_forward_prefill"]


def _stats(chunks, reqs):
    STATS["packed_calls"] += 1
    STATS["chunks"] += chunks
    STATS["requests"] += reqs


_PACKED = _patched()


def forward_prefill(self, *a, **k):
    if os.environ.get("DSV41_PREFILL_PACK", "1") != "1":
        return _ORIG(self, *a, **k)
    return _PACKED(self, *a, **k)


def stats(worker=None) -> dict:
    r = dict(STATS)
    for k in STATS:
        STATS[k] = 0
    return r


SETATTR = [{"target": "vllm.models.deepseek_v4_1.amd.rocm.DeepseekV41ROCMAiterMLAAttention._forward_prefill",
            "source": "dsv41_prefill_pack.forward_prefill"}]
