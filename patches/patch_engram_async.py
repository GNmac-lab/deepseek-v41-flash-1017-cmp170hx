#!/usr/bin/env python3
"""Overlap the SSD Engram lookups with layer 0 (vllm/models/deepseek_v4_1/common/engram.py).

`prepare_embeddings` runs both Engram layers' lookups (host callback + NVMe reads for missing rows) on the main
stream before the decoder layers, so every rank's GPU idles for them and the slowest rank delays the first
all-reduce (profiled 2026-10-19 on v6: rank 2 ~1.2 ms/step). Now the lookup runs on a side stream forked from the
main stream, and `embed` (layer 1 / 14) joins it before reading the staged rows. Only for FULL CUDA graphs and
eager runs: breakable (PIECEWISE) capture ends a graph segment at every attention op, and a side stream forked
here could not be joined inside the segment, so those keep the old main-stream lookup.
Same rows, same math; DSV41_ENGRAM_ASYNC=0 = old behaviour. usage: patch_engram_async.py <common/engram.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "DSV41_ENGRAM_ASYNC" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new, 1)


rep('''        self._extra_staged_rows = [
            torch.empty_like(self.staged_rows)
            for _ in range(vllm_config.parallel_config.num_ubatches - 1)
        ]
''', '''        self._extra_staged_rows = [
            torch.empty_like(self.staged_rows)
            for _ in range(vllm_config.parallel_config.num_ubatches - 1)
        ]
        # dsv41: SSD lookups on a side stream, joined in `embed` (patch_engram_async.py).
        self._async_ssd = (
            self.embed_tokens._ssd
            and self.embed_tokens.dp_size == 1
            and vllm_config.parallel_config.num_ubatches <= 1  # 0 when DBO is off
            and _os.environ.get("DSV41_ENGRAM_ASYNC", "1") != "0"
        )
        self._ready: torch.cuda.Event | None = None
        self._pending = False

    _side_streams: dict = {}

    def _async_lookup_ok(self) -> bool:
        if not self._async_ssd:
            return False
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context, is_forward_context_available

        # Breakable (PIECEWISE) capture breaks at attention ops; the fork would not be joined in its segment.
        return not (
            is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
        )
''')
rep('''        rows = self._staged_rows_for_ubatch()[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        self.embed_tokens.lookup(hash_ids, rows)
''', '''        rows = self._staged_rows_for_ubatch()[: hash_ids.shape[0]]
        assert rows.shape[0] == hash_ids.shape[0], "engram staging buffer too small"
        if self._async_lookup_ok():
            dev = hash_ids.device
            side = Engram._side_streams.get(dev.index)
            if side is None:
                side = Engram._side_streams[dev.index] = torch.cuda.Stream(device=dev)
            if self._ready is None:
                self._ready = torch.cuda.Event()
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):
                self.embed_tokens.lookup(hash_ids, rows)
                self._ready.record(side)
            self._pending = True
            return
        self.embed_tokens.lookup(hash_ids, rows)
''')
rep('''        """Gather heads, returning only local tokens when SP is enabled."""
''', '''        """Gather heads, returning only local tokens when SP is enabled."""
        if self._pending:  # join the side-stream lookup started in prepare_embeddings
            torch.cuda.current_stream().wait_event(self._ready)
            self._pending = False
''')
open(p, "w").write(s)
print("patched", p)
