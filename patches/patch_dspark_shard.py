#!/usr/bin/env python3
"""Exact vocab-sharded DSpark draft argmax (hybrid/dspark_proposer.py).

Per step the DeepSeek-V4.1 DSpark proposer all-gathered its 5 positions' draft logits over the 129,280-token
vocab (~275 us), then per position added the Markov transition bias with a replicated [V, 256] GEMM (66 MB read,
~48 us) and took a single-block argmax (~38 us) — ~0.75 ms per step (profiled 2026-10-19). Now each TP rank keeps
its vocab shard of the base logits, adds the bias for its shard only (a quarter of the GEMM), takes the local max,
and only (value, index) pairs are all-gathered per position; the first maximum wins, as in the full-vocab argmax.
Same drafts (up to bf16 GEMM rounding), so acceptance is unchanged; the target still verifies every token.
DSV41_DSPARK_SHARD=0 = stock path. usage: patch_dspark_shard.py <hybrid/dspark_proposer.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "DSV41_DSPARK_SHARD" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep('''        self._dbg = os.environ.get("DSV41_DEBUG_DSPARK") == "1"
''', '''        self._dbg = os.environ.get("DSV41_DEBUG_DSPARK") == "1"
        # dsv41 (ds41-patches/patch_dspark_shard.py): vocab-sharded exact draft argmax
        self._shard_argmax = os.environ.get("DSV41_DSPARK_SHARD", "1") != "0"
''')
rep('''    def _sample_sequential(self, num_reqs: int, sample_hidden: torch.Tensor) -> torch.Tensor:
        n = self.num_speculative_tokens
''', '''    def _sample_sequential_sharded(self, num_reqs: int, sample_hidden: torch.Tensor):
        """Draft argmax over vocab shards; None when the stock path must run."""
        from vllm.distributed import get_tensor_model_parallel_world_size, tensor_model_parallel_all_gather

        m = self.model
        lp, head = m.logits_processor, m.lm_head
        if (
            get_tensor_model_parallel_world_size() < 2
            or getattr(head, "tp_size", 1) < 2
            or lp.scale != 1.0
            or lp.soft_cap is not None
            or self._draft_topk is not None
            or self._conf_threshold() > 0
        ):
            return None
        n = self.num_speculative_tokens
        si = head.shard_indices
        start, end = si.org_vocab_start_index, si.org_vocab_end_index
        local = lp._get_logits(m.model.norm(sample_hidden), head, None, skip_gather=True)
        local = local[:, : end - start].view(num_reqs, n, end - start)       # drop this shard's padding columns
        w2 = m.model.markov_head.markov_w2.weight[start:end]                  # bias rows of this shard's tokens
        drafts = torch.empty(num_reqs, n, dtype=torch.int64, device=self.device)
        prev = self.input_ids[self._anchor_idx[:num_reqs]]
        for i in range(n):
            e = m.markov_embed(prev)                                           # [R, r]
            li = local[:, i] + torch.nn.functional.linear(e, w2)               # [R, shard] bf16, as the stock add
            v, j = li.max(dim=-1)
            pair = torch.stack([v.float(), (j + start).float()], dim=-1)       # ids < 2^24: exact in fp32
            g = tensor_model_parallel_all_gather(pair, dim=-1)                 # [R, 2 * tp], rank (= vocab) order
            best = g[:, 0::2].argmax(dim=-1, keepdim=True)                     # first max = lowest id on ties
            tok = m.map_draft_to_target(g[:, 1::2].gather(1, best).squeeze(1).to(torch.int64))
            drafts[:, i] = tok
            prev = tok
        return drafts

    def _sample_sequential(self, num_reqs: int, sample_hidden: torch.Tensor) -> torch.Tensor:
        n = self.num_speculative_tokens
        if self._shard_argmax:
            _d = self._sample_sequential_sharded(num_reqs, sample_hidden)
            if _d is not None:
                return _d
''')
open(p, "w").write(s)
print("patched", p)
