#!/usr/bin/env python3
"""dsv41: replay the DSpark draft-token sampling loop as one CUDA graph (DSV41_SAMPLE_GRAPH=1, default).

The vocab-sharded sequential Markov sampling (_sample_sequential_sharded: draft head, then per draft position the
Markov bias GEMM, add, shard max, (value, id) pair, all-gather, argmax, gather) runs eagerly after the drafter forward:
about 70 small kernels per step, each waiting on its CPU launch (the profile shows a ~5 us idle gap before nearly every
one). This captures the same code once per batch size (lazily, on all TP ranks together; the all-gather is pynccl, which
is graph-safe) and replays it from a static copy of the sampled hidden states. Same kernels, same order, same inputs:
identical draft tokens. Graphs are skipped while another capture is running (vLLM's own warmup/capture passes).
usage: patch_dspark_sample_graph.py <hybrid/dspark_proposer.py>"""
import sys

p = sys.argv[1]
s = open(p).read()
MARK = "patch_dspark_sample_graph.py"
if MARK in s:
    print("already patched", p)
    sys.exit(0)
pairs = [
    ('''_DSV41_DRAFT_HEAD_FP8 = _dsv41_os38.environ.get("DSV41_DRAFT_HEAD_FP8", "0") == "1"
''', '''_DSV41_DRAFT_HEAD_FP8 = _dsv41_os38.environ.get("DSV41_DRAFT_HEAD_FP8", "0") == "1"
# dsv41 (ds41-patches/patch_dspark_sample_graph.py): the sharded draft sampling loop as one CUDA graph per batch size
_DSV41_SAMPLE_GRAPH = _dsv41_os38.environ.get("DSV41_SAMPLE_GRAPH", "1") != "0"
'''),
    ('''            or self._conf_threshold() > 0
        ):
            return None
        n = self.num_speculative_tokens
        si = head.shard_indices
''', '''            or self._conf_threshold() > 0
        ):
            return None
        if _DSV41_SAMPLE_GRAPH:
            return self._dsv41_sample_graphed(num_reqs, sample_hidden)
        return self._dsv41_sample_body(num_reqs, sample_hidden)

    def _dsv41_sample_graphed(self, num_reqs: int, sample_hidden: torch.Tensor) -> torch.Tensor:
        """dsv41 (ds41-patches/patch_dspark_sample_graph.py): replay the sampling body from a per-shape CUDA graph."""
        if not sample_hidden.is_cuda or torch.cuda.is_current_stream_capturing():
            return self._dsv41_sample_body(num_reqs, sample_hidden)
        graphs = self.__dict__.setdefault("_dsv41_sample_graphs", {})
        key = (num_reqs, tuple(sample_hidden.shape), sample_hidden.dtype)
        ent = graphs.get(key)
        if ent is None:
            static_in = sample_hidden.detach().clone()
            self._dsv41_sample_body(num_reqs, static_in)        # warm-up: lazy FP8 head copy, Triton JIT, pynccl
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                static_out = self._dsv41_sample_body(num_reqs, static_in)
            ent = graphs[key] = (g, static_in, static_out)
            logger.info("dsv41 DSpark sampling graph captured: %d request(s), %d draft rows", num_reqs,
                        sample_hidden.shape[0])
        g, static_in, static_out = ent
        static_in.copy_(sample_hidden)
        g.replay()
        return static_out.clone()

    def _dsv41_sample_body(self, num_reqs: int, sample_hidden: torch.Tensor) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_gather

        m = self.model
        lp, head = m.logits_processor, m.lm_head
        n = self.num_speculative_tokens
        si = head.shard_indices
'''),
]
for a, b in pairs:
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
open(p, "w").write(s)
print("patched", p)
