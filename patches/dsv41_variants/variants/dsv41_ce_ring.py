# dsv41 (runtime variant): copy-engine all-reduce for prefill-sized tensor-parallel messages.
#
# Port of Morrowmake/vllm-cmp170hx vllm/sm80/distributed/ce_ring_all_reduce.py (Apache-2.0), which follows the PCIe
# DMA all-reduce of b12x (Apache-2.0, local-inference-lab). A reduce-scatter + all-gather ring whose data moves by
# copy-engine copies (cudaMemcpyAsync into the next rank's IPC scratch) while the SMs only signal, wait and add, so
# the compute beside it keeps its SMs (an NCCL ring moves the same bytes with SM kernels that stay resident).
# Changes here: the ring path only; IPC buffers through CudaRTLibrary (this image has no vllm._C); own error word.
# Each element is summed in the ring's order in the message dtype (BF16 rounding per add, as NCCL's ring).
# Production path: all_reduce_owner (owner sum, every flag a CUDA stream memory operation, one FP32 sum kernel
# per call), used by dsv41_split_reduce for prefill all-reduces beside compute. Built collectively at worker
# start and checked against NCCL there (DSV41_PREFILL_CE=0 skips it).
#   call: dsv41_ce_ring.selftest -> max |ring - NCCL|; dsv41_ce_ring.bench -> alone / beside-compute timings
import ctypes
import pickle
from dataclasses import dataclass

import torch
import torch.distributed as dist

FLAG_STRIDE_BYTES = 128
SCRATCH_ALIGN = 256
DTYPES = (torch.float32, torch.bfloat16)
WAIT_TIMEOUT_NS = 60_000_000_000


@dataclass(frozen=True)
class Step:
    k: int
    reduce: bool
    send: int
    recv: int


def ring_steps(world: int, rank: int) -> list[Step]:
    """Shard indices this rank sends to rank + 1 and receives from rank - 1, per step."""
    steps = []
    for k in range(2 * (world - 1)):
        if k < world - 1:
            steps.append(Step(k, True, (rank - k) % world, (rank - k - 1) % world))
        else:
            j = k - (world - 1)
            steps.append(Step(k, False, (rank + 1 - j) % world, (rank - j) % world))
    return steps


def _ready_slot(world: int) -> int:
    return 2 * (world - 1)


def flag_bytes(world: int) -> int:
    # ring: steps + ready; owner sum: ready per sender + 2 x (world - 1) step flags
    return (_ready_slot(world) + 1 + world + 2 * (world - 1)) * FLAG_STRIDE_BYTES


class _Ptr:
    __slots__ = ("_ptr", "dtype")

    def __init__(self, ptr: int, dtype: torch.dtype):
        self._ptr, self.dtype = ptr, dtype

    def data_ptr(self) -> int:
        return self._ptr


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _store_release_sys(ptr, value):
        tl.inline_asm_elementwise("st.release.sys.global.u32 [$1], $2; mov.u32 $0, 0;", "=r,l,r",
                                  [ptr, value], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _load_acquire_sys(ptr):
        return tl.inline_asm_elementwise("ld.acquire.sys.global.u32 $0, [$1];", "=r,l",
                                         [ptr], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _now_ns(ptr):
        return tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,l",
                                         [ptr], dtype=tl.int64, is_pure=False, pack=1)

    @triton.jit
    def set_flag(flag_ptr, value):
        _store_release_sys(flag_ptr, value)

    @triton.jit
    def wait_flag(flag_ptr, value, err_ptr, timeout_ns):
        start = _now_ns(flag_ptr)
        cur = _load_acquire_sys(flag_ptr)
        late = _now_ns(flag_ptr) - start > timeout_ns
        while (cur < value) & (late == 0):
            cur = _load_acquire_sys(flag_ptr)
            late = _now_ns(flag_ptr) - start > timeout_ns
        if cur < value:
            tl.atomic_or(err_ptr, 1, sem="relaxed")

    @triton.jit
    def add(dst_ptr, a_ptr, b_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(a_ptr + offs, mask=mask)
        b = tl.load(b_ptr + offs, mask=mask)
        tl.store(dst_ptr + offs, (a.to(tl.float32) + b.to(tl.float32)).to(a.dtype), mask=mask)

    @triton.jit
    def sum4(dst_ptr, p0, p1, p2, p3, n, BLOCK: tl.constexpr):
        # FP32 sum of the four ranks' partials in rank order, rounded once to dst's dtype.
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        acc = tl.load(p0 + offs, mask=mask).to(tl.float32)
        acc = acc + tl.load(p1 + offs, mask=mask).to(tl.float32)
        acc = acc + tl.load(p2 + offs, mask=mask).to(tl.float32)
        acc = acc + tl.load(p3 + offs, mask=mask).to(tl.float32)
        tl.store(dst_ptr + offs, acc.to(dst_ptr.dtype.element_ty), mask=mask)

    return set_flag, wait_flag, add, sum4


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class CeRingAllReduce:
    """The TP group's copy-engine ring (collective: every rank constructs it at the same time)."""

    ADD_BLOCK = 2048

    def __init__(self, group, capacity_bytes: int):
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary, cudaIpcMemHandle_t

        self.world, self.rank = group.world_size, group.rank_in_group
        shard = _cdiv(capacity_bytes, self.world)
        self.slot_bytes = _cdiv(shard, SCRATCH_ALIGN) * SCRATCH_ALIGN
        self.capacity_bytes = self.slot_bytes * self.world
        steps = 2 * (self.world - 1)
        self._rt = rt = CudaRTLibrary()
        cpu = group.cpu_group

        def agree(err: str | None, what: str) -> None:
            """Every rank reports; any rank's error is raised on all ranks (no rank left waiting)."""
            errs = [None] * self.world
            dist.all_gather_object(errs, (self.rank, err), group=cpu)
            bad = [(r, e) for r, e in errs if e]
            if bad:
                raise RuntimeError(f"dsv41_ce_ring {what}: " + "; ".join(f"rank {r}: {e}" for r, e in bad))

        def shared(nbytes: int) -> list[int]:
            err, ptr, handle = None, None, None
            try:
                ptr = rt.cudaMalloc(nbytes)
                handle = bytes(rt.cudaIpcGetMemHandle(ptr))
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            handles = [None] * self.world
            dist.all_gather_object(handles, (self.rank, handle, err), group=cpu)
            bad = [(r, e) for r, _, e in handles if e]
            if bad:
                raise RuntimeError("dsv41_ce_ring alloc: " + "; ".join(f"rank {r}: {e}" for r, e in bad))
            out = [0] * self.world
            err = None
            for r, h, _ in handles:
                try:
                    out[r] = (ptr.value if r == self.rank else
                              rt.cudaIpcOpenMemHandle(cudaIpcMemHandle_t.from_buffer_copy(h)).value)
                except Exception as e:  # noqa: BLE001
                    err = (err or "") + f" open {r}: {type(e).__name__}: {e};"
            agree(err, "ipc open")
            return out

        self.scratch = shared(self.slot_bytes * steps)
        self.flags = shared(flag_bytes(self.world))
        rt.cudaMemset(ctypes.c_void_p(self.flags[self.rank]), 0, flag_bytes(self.world))
        torch.cuda.synchronize()
        dist.barrier(group=cpu)
        self.epoch = 0
        lib = rt.lib
        self._memcpy = lib.cudaMemcpyAsync
        self._memcpy.restype = ctypes.c_int
        self._memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        self._set, self._wait, self._add, self._sum4 = _kernels()
        cu = ctypes.CDLL("libcuda.so.1")
        self._wv = getattr(cu, "cuStreamWriteValue32_v2", None) or cu.cuStreamWriteValue32
        self._wt = getattr(cu, "cuStreamWaitValue32_v2", None) or cu.cuStreamWaitValue32
        for f in (self._wv, self._wt):
            f.restype = ctypes.c_int
            f.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
        # owner-sum layout: per rank 2 x (world - 1) receive slots + flags, after the ring's
        self.owner_flags0 = (_ready_slot(self.world) + 1) * FLAG_STRIDE_BYTES
        self.error = torch.zeros(1, dtype=torch.int32, device="cuda")

    def takes(self, t: torch.Tensor) -> bool:
        if not t.is_cuda or t.dtype not in DTYPES or not t.is_contiguous():
            return False
        if torch.cuda.is_current_stream_capturing():
            return False
        nbytes = t.numel() * t.element_size()
        return 0 < nbytes <= self.capacity_bytes and nbytes % (16 * self.world) == 0

    def _copy(self, dst: int, src: int, nbytes: int, stream: int) -> None:
        rc = self._memcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, 4, ctypes.c_void_p(stream))
        if rc != 0:
            raise RuntimeError(f"cudaMemcpyAsync failed: cudaError {rc}")

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Sum of every rank's t around the ring, queued on the current stream."""
        out = torch.empty_like(t)
        self.epoch += 1
        e = self.epoch
        world, rank = self.world, self.rank
        n = t.numel() // world
        sb = n * t.element_size()
        stream = torch.cuda.current_stream().cuda_stream
        nxt, prv = (rank + 1) % world, (rank - 1) % world
        flags, scratch, inp, o = self.flags, self.scratch, t.data_ptr(), out.data_ptr()
        ready = _ready_slot(world) * FLAG_STRIDE_BYTES
        self._set[(1,)](_Ptr(flags[prv] + ready, torch.int32), e)
        self._wait[(1,)](_Ptr(flags[rank] + ready, torch.int32), e, self.error, WAIT_TIMEOUT_NS)
        for s in ring_steps(world, rank):
            src = (inp if s.k == 0 else o) + s.send * sb
            self._copy(scratch[nxt] + s.k * self.slot_bytes, src, sb, stream)
            self._set[(1,)](_Ptr(flags[nxt] + s.k * FLAG_STRIDE_BYTES, torch.int32), e)
            self._wait[(1,)](_Ptr(flags[rank] + s.k * FLAG_STRIDE_BYTES, torch.int32), e, self.error,
                             WAIT_TIMEOUT_NS)
            slot = scratch[rank] + s.k * self.slot_bytes
            dst = o + s.recv * sb
            if s.reduce:
                self._add[(_cdiv(n, self.ADD_BLOCK),)](_Ptr(dst, t.dtype), _Ptr(inp + s.recv * sb, t.dtype),
                                                     _Ptr(slot, t.dtype), n, BLOCK=self.ADD_BLOCK)
            else:
                self._copy(dst, slot, sb, stream)
        return out


def _slot_index(world: int, sender: int, receiver: int) -> int:
    """Receiver-side slot index (0..world-2) of sender's shard."""
    return (sender - receiver) % world - 1


def _owner_all_reduce(self, t: torch.Tensor) -> torch.Tensor:
    """Owner sum with stream memory operations for every flag: each rank copies shard q to rank q (copy engine),
    rank q sums the four partials in FP32 in rank order and rounds once (one kernel), then copies the result shard
    to every peer (copy engine). The only SM work is the one sum kernel per call."""
    if self.world != 4:
        raise ValueError("owner sum kernel is written for four ranks")
    out = torch.empty_like(t)
    self.epoch += 1
    e = self.epoch
    world, rank = self.world, self.rank
    n = t.numel() // world
    sb = n * t.element_size()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    flags, scratch, inp, o = self.flags, self.scratch, t.data_ptr(), out.data_ptr()
    base = self.owner_flags0
    ready = lambda r, sender: base + _slot_index(world, sender, r) * FLAG_STRIDE_BYTES      # noqa: E731
    stepf = lambda r, k: base + (world + k) * FLAG_STRIDE_BYTES                             # noqa: E731
    peers = [(rank + d) % world for d in range(1, world)]

    def wv(addr, v):
        rc = self._wv(stream, addr, v, 0)
        if rc != 0:
            raise RuntimeError(f"cuStreamWriteValue32 failed: {rc}")

    def wt(addr, v):
        rc = self._wt(stream, addr, v, 0)                                              # CU_STREAM_WAIT_VALUE_GEQ
        if rc != 0:
            raise RuntimeError(f"cuStreamWaitValue32 failed: {rc}")

    for q in peers:                       # my receive slots in q's buffer are free: tell q
        wv(flags[q] + ready(q, rank), e)
    for q in peers:                       # q's slots for me are free
        wt(flags[rank] + ready(rank, q), e)
    for q in peers:                       # reduce-scatter: shard q to rank q
        k = _slot_index(world, rank, q)
        self._copy(scratch[q] + k * self.slot_bytes, inp + q * sb, sb, stream.value)
        wv(flags[q] + stepf(q, k), e)
    parts = []
    for s_ in range(world):
        if s_ == rank:
            parts.append(inp + rank * sb)
            continue
        k = _slot_index(world, s_, rank)
        wt(flags[rank] + stepf(rank, k), e)
        parts.append(scratch[rank] + k * self.slot_bytes)
    mine = o + rank * sb
    self._sum4[(_cdiv(n, self.ADD_BLOCK),)](_Ptr(mine, t.dtype), *[_Ptr(p_, t.dtype) for p_ in parts], n,
                                           BLOCK=self.ADD_BLOCK)
    g = world - 1
    for q in peers:                       # all-gather: my result shard to every peer
        k = g + _slot_index(world, rank, q)
        self._copy(scratch[q] + k * self.slot_bytes, mine, sb, stream.value)
        wv(flags[q] + stepf(q, k), e)
    for s_ in peers:
        k = g + _slot_index(world, s_, rank)
        wt(flags[rank] + stepf(rank, k), e)
        self._copy(o + s_ * sb, scratch[rank] + k * self.slot_bytes, sb, stream.value)
    return out


CeRingAllReduce.all_reduce_owner = _owner_all_reduce
_RING: list = [None]


def ring(capacity_bytes: int = 2048 * 5120 * 2):
    if _RING[0] is None:
        from vllm.distributed.parallel_state import get_tp_group
        _RING[0] = CeRingAllReduce(get_tp_group(), capacity_bytes)
    return _RING[0]


def selftest(worker=None) -> dict:
    from vllm.distributed import tensor_model_parallel_all_reduce
    try:
        r = ring()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out = {}
    for rows in (1024, 2048):
        g = torch.Generator(device="cuda").manual_seed(1234 + r.rank)
        x = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda", generator=g)
        a = r.all_reduce(x)
        c = r.all_reduce_owner(x)
        b = tensor_model_parallel_all_reduce(x.clone())
        torch.cuda.synchronize()
        out[rows] = {"ring_vs_nccl": float((a.float() - b.float()).abs().max()),
                     "owner_vs_nccl": float((c.float() - b.float()).abs().max()),
                     "max_abs": float(b.float().abs().max())}
    out["error_word"] = int(r.error.item())
    return out


def bench(worker=None) -> dict:
    from vllm.distributed import tensor_model_parallel_all_reduce
    r = ring()
    dev = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(2048, 5120, dtype=torch.bfloat16, device=dev)
    w = torch.randn(5120, 5120, dtype=torch.bfloat16, device=dev)
    x = torch.randn(1024, 5120, dtype=torch.bfloat16, device=dev)
    side = torch.cuda.Stream(device=dev, priority=-1)

    def timed(fn, n=4):
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(n):
            fn()
        t1.record()
        torch.cuda.synchronize()
        return round(t0.elapsed_time(t1) / n, 3)

    mm = lambda: [a @ w for _ in range(8)]  # noqa: E731
    res = {"compute_ms": timed(mm)}
    for name, ar in (("nccl", tensor_model_parallel_all_reduce), ("ce", r.all_reduce),
                     ("owner", r.all_reduce_owner)):
        def alone():
            with torch.cuda.stream(side):
                ar(x)
                ar(x)
            torch.cuda.current_stream().wait_stream(side)

        def both():
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                ar(x)
                ar(x)
            mm()
            main.wait_stream(side)
        alone()
        res[f"{name}_2x1024_alone_ms"] = timed(alone)
        res[f"{name}_beside_compute_ms"] = timed(both)
    res["error_word"] = int(r.error.item())
    return res


def _startup_check() -> None:
    """Owner sum vs NCCL on a test message; a mismatch beyond rounding stops the worker."""
    from vllm.distributed import tensor_model_parallel_all_reduce
    r = ring()
    g = torch.Generator(device="cuda").manual_seed(4242 + r.rank)
    x = torch.randn(512, 5120, dtype=torch.bfloat16, device="cuda", generator=g)
    a = r.all_reduce_owner(x)
    b = tensor_model_parallel_all_reduce(x.clone())
    torch.cuda.synchronize()
    diff = float((a.float() - b.float()).abs().max())
    tol = 0.02 * float(b.float().abs().max())
    if not diff <= tol or int(r.error.item()) != 0:
        raise RuntimeError(f"dsv41_ce_ring startup check FAILED: max |owner - NCCL| = {diff} (tol {tol})")


SETATTR: list = []
import os as _os  # noqa: E402
if _os.environ.get("DSV41_PREFILL_CE", "1") == "1":
    from vllm.distributed.parallel_state import get_tp_group as _tp
    if _tp().world_size == 4:
        _startup_check()
