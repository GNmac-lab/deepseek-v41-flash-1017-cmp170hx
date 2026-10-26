# dsv41 (runtime variant): read the Engram rows of a prompt's next prefill chunks into the page cache ahead of time.
#
# With DSV41_ENGRAM_STORAGE=ssd the Engram layers read their FP8 rows from the checkpoint shard on the NVMe (host
# callback, engram_ssd/row_store.cpp); rows of a new text miss the page cache and the layer waits for the reads
# (cold prompt ~10 % slower than the same prompt warm). The prompt is known when its first chunk is scheduled, so after
# each step this computes, on the CPU, the n-gram hash ids of the next DSV41_ENGRAM_PREFETCH_CHUNKS chunks (the
# _hash_ids_kernel formula of vllm/models/deepseek_v4_1/common/engram.py, without dead tokens) for this rank's hash
# heads and reads those rows through a second row store (own file handle and threads) on a background thread.
# Only the page cache changes: the model reads every row exactly as before, so outputs are bit-identical.
#   DSV41_ENGRAM_PREFETCH=0 off; DSV41_ENGRAM_PREFETCH_CHUNKS (2); DSV41_ENGRAM_PREFETCH_THREADS (8).
import ctypes as C
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.worker import gpu_model_runner as _gmr

logger = init_logger(__name__)
_R = _gmr.GPUModelRunner
_ORIG_EXEC = _R.__dict__.get("_dsv41_prefetch_orig_exec") or _R.execute_model
_R._dsv41_prefetch_orig_exec = _ORIG_EXEC
STATS = {"jobs": 0, "rows": 0, "errors": 0, "busy_s": 0.0}
_STATE: dict = {"init": None}
_REQS: dict = {}          # req_id -> [prompt np.int64, prefetched_until]
_LOCK = threading.Lock()


def _on() -> bool:
    return os.environ.get("DSV41_ENGRAM_PREFETCH", "1") == "1"


class _Plan:
    """Everything one rank needs to hash and read its Engram rows (built once from the loaded model)."""

    def __init__(self, runner):
        from engram_ssd import engram_ssd as es
        from vllm.models.deepseek_v4_1.common import engram as eg

        mdl = runner.get_model()
        mdl = mdl.unwrap() if hasattr(mdl, "unwrap") else mdl
        hs = [m for m in mdl.modules() if isinstance(m, eg.NgramHashState)]
        ens = [m for m in mdl.modules() if isinstance(m, eg.Engram)
               and getattr(m.embed_tokens, "ssd_table", None) is not None]
        if len(hs) != 1 or not ens:
            raise RuntimeError(f"dsv41_engram_prefetch: {len(hs)} hash states, {len(ens)} SSD engram layers")
        h = hs[0]
        self.token_map = h.token_map.cpu().numpy().astype(np.int64)
        self.mult = h.multipliers.cpu().numpy().astype(np.int64)          # [layers, ngram]
        self.primes = h.primes.cpu().numpy().astype(np.int64)             # [layers, ngram - 1, heads]
        self.offsets = h.offsets.cpu().numpy().astype(np.int64)           # [layers, hash cols]
        self.pad = int(h.pad_id)
        self.ngram = self.mult.shape[1]
        self.heads = self.primes.shape[2]
        model_dir = os.environ.get("DSV41_ENGRAM_DIR") or os.environ.get("DSV41_MODEL_DIR")
        lib = es._lib()
        nthreads = int(os.environ.get("DSV41_ENGRAM_PREFETCH_THREADS", "8"))
        self.layers = []
        for e in ens:
            emb = e.embed_tokens
            layer_id = h.layout.layer_ids[e.layer_hash_index]
            path, rows, woff, soff = es.find_engram_shard(model_dir, layer_id)
            store = lib.row_store_open(path.encode(), rows, woff, soff, nthreads)
            if not store:
                raise RuntimeError(f"dsv41_engram_prefetch: cannot open {path}")
            lib.row_store_range(store, emb.vocab_start_idx, emb.vocab_end_idx)
            self.layers.append((e.layer_hash_index, emb.head_start, emb.part_n_hash_cols, store))
        self.lib = lib
        self.chunk = runner.scheduler_config.max_num_batched_tokens
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="dsv41-engram-prefetch")
        self.buf_w = np.empty(0, dtype=np.uint8)
        self.buf_s = np.empty(0, dtype=np.uint8)

    def hashes(self, prompt: np.ndarray, start: int, end: int) -> np.ndarray:
        """[end - start, layers, hash cols] int64 hash ids of positions start .. end - 1."""
        pos = np.arange(start, end, dtype=np.int64)
        comp = self.token_map[prompt]
        n_layers = self.mult.shape[0]
        out = np.empty((end - start, n_layers, (self.ngram - 1) * self.heads), dtype=np.int64)
        for layer in range(n_layers):
            rolling = np.zeros(end - start, dtype=np.int64)
            blocked = np.zeros(end - start, dtype=bool)
            for shift in range(self.ngram):
                back = pos - shift
                blocked |= back < 0
                value = np.where(blocked, self.pad, comp[np.maximum(back, 0)])
                rolling ^= value * self.mult[layer, shift]
                if shift:
                    cols = slice((shift - 1) * self.heads, shift * self.heads)
                    out[:, layer, cols] = (rolling[:, None] % self.primes[layer, shift - 1][None, :]
                                           + self.offsets[layer, cols][None, :])
        return out

    def read(self, prompt: np.ndarray, start: int, end: int) -> None:
        import time
        t0 = time.perf_counter()
        ids_all = self.hashes(prompt, start, end)
        for layer, h0, nh, store in self.layers:
            ids = np.ascontiguousarray(ids_all[:, layer, h0:h0 + nh].reshape(-1))
            n = ids.shape[0]
            if self.buf_w.shape[0] < n * 256:
                self.buf_w = np.empty(n * 256, dtype=np.uint8)
                self.buf_s = np.empty(n * 8, dtype=np.uint8)
            self.lib.row_store_lookup_sync(store, ids.ctypes.data, self.buf_w.ctypes.data,
                                           self.buf_s.ctypes.data, n)
            STATS["rows"] += n
        STATS["jobs"] += 1
        STATS["busy_s"] += time.perf_counter() - t0


def _job(plan, prompt, start, end):
    try:
        plan.read(prompt, start, end)
    except Exception as e:  # noqa: BLE001  a hint only: never fail the step
        STATS["errors"] += 1
        if STATS["errors"] <= 3:
            logger.warning("dsv41_engram_prefetch: %s: %s", type(e).__name__, e)


def _plan(runner):
    p = _STATE["init"]
    if p is None:
        try:
            p = _Plan(runner)
            logger.info("dsv41 Engram prefetch on: %d SSD layers, %d chunk(s) ahead", len(p.layers),
                        int(os.environ.get("DSV41_ENGRAM_PREFETCH_CHUNKS", "2")))
        except Exception as e:  # noqa: BLE001
            logger.warning("dsv41_engram_prefetch off: %s: %s", type(e).__name__, e)
            p = False
        _STATE["init"] = p
    return p or None


def _kick(runner, so) -> None:
    plan = _plan(runner)
    if plan is None:
        return
    for rid in getattr(so, "finished_req_ids", ()) or ():
        _REQS.pop(rid, None)
    computed = {}
    for nr in so.scheduled_new_reqs:
        if nr.prompt_token_ids:
            _REQS[nr.req_id] = [np.asarray(nr.prompt_token_ids, dtype=np.int64), 0]
        computed[nr.req_id] = nr.num_computed_tokens
    cr = so.scheduled_cached_reqs
    for rid, c in zip(cr.req_ids, cr.num_computed_tokens):
        computed[rid] = c
    ahead = int(os.environ.get("DSV41_ENGRAM_PREFETCH_CHUNKS", "2")) * plan.chunk
    for rid, n in so.num_scheduled_tokens.items():
        st = _REQS.get(rid)
        if st is None or rid not in computed:
            continue
        prompt, done = st
        nxt = computed[rid] + n
        end = min(nxt + ahead, prompt.shape[0])
        start = max(nxt, done)
        if start < end:
            st[1] = end
            plan.pool.submit(_job, plan, prompt, start, end)
        elif nxt >= prompt.shape[0]:
            _REQS.pop(rid, None)          # prompt done: decode needs no prefetch


def execute_model(self, scheduler_output, *a, **k):
    out = _ORIG_EXEC(self, scheduler_output, *a, **k)
    if _on():
        try:
            _kick(self, scheduler_output)
        except Exception as e:  # noqa: BLE001
            STATS["errors"] += 1
            if STATS["errors"] <= 3:
                logger.warning("dsv41_engram_prefetch: %s: %s", type(e).__name__, e)
    return out


def stats(worker=None) -> dict:
    r = dict(STATS)
    r["tracked_reqs"] = len(_REQS)
    r["busy_s"] = round(r["busy_s"], 3)
    for k_ in ("jobs", "rows", "errors"):
        STATS[k_] = 0
    STATS["busy_s"] = 0.0
    return r


def check(worker=None) -> dict:
    """Diagnostic: CPU hashes vs the GPU _hash_ids_kernel on one 3,000-token sequence (positions 0..T-1)."""
    import triton
    from vllm.models.deepseek_v4_1.common import engram as eg
    plan = _plan(worker.model_runner)
    if plan is None:
        return {"error": "no plan"}
    mdl = worker.model_runner.get_model()
    mdl = mdl.unwrap() if hasattr(mdl, "unwrap") else mdl
    h = next(m for m in mdl.modules() if isinstance(m, eg.NgramHashState))
    g = torch.Generator().manual_seed(7)
    T = 3000
    toks = torch.randint(0, plan.token_map.shape[0], (T,), generator=g)
    toks[100:140] = toks[0:40]                       # repeated n-grams
    dev = h.token_map.device
    ids = toks.to(dev)
    out = torch.empty((T, plan.mult.shape[0], (plan.ngram - 1) * plan.heads), dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, T], dtype=torch.int32, device=dev)
    pos = torch.arange(T, dtype=torch.int64, device=dev)
    dead = torch.zeros(T, dtype=torch.bool, device=dev)
    lb = torch.full((1, plan.ngram - 1), -1, dtype=torch.int32, device=dev)
    lbd = torch.zeros((1, plan.ngram - 1), dtype=torch.bool, device=dev)
    eg._hash_ids_kernel[(triton.cdiv(T, 32), plan.mult.shape[0])](
        ids, h.token_map, dead, pos, None, qsl, h.multipliers, h.primes, h.offsets, None, lb, lbd, out, T, 0,
        h.pad_id, input_stride=1, mask_stride=1, position_stride=1, table_stride=0, table_col_stride=0,
        query_stride=1, num_query_rows=1, num_table_rows=0, max_blocks=0, cache_block_size=h.block_size,
        MAX_NGRAM=plan.ngram, num_heads=plan.heads, BLOCK_T=32, BLOCK_H=triton.next_power_of_2(plan.heads),
        dead_id=eg.DEAD_ID, lookback_depth=plan.ngram - 1, lookback_row_stride=plan.ngram - 1,
        lookback_col_stride=1, lookback_mask_row_stride=plan.ngram - 1, lookback_mask_col_stride=1, num_warps=4)
    gpu = out.cpu().numpy().astype(np.int64)
    cpu_full = plan.hashes(toks.numpy().astype(np.int64), 0, T)
    cpu_tail = plan.hashes(toks.numpy().astype(np.int64), 1000, T)     # a later chunk, with lookback
    return {"equal_full": bool((gpu == cpu_full).all()), "equal_tail": bool((gpu[1000:] == cpu_tail).all()),
            "mismatch": int((gpu != cpu_full).sum()), "shape": list(gpu.shape), "layers": len(plan.layers)}


SETATTR = [{"target": "vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model",
            "source": "dsv41_engram_prefetch.execute_model"}]
