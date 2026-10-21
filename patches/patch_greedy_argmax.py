#!/usr/bin/env python3
"""Greedy decode without the full-logits all-gather (vllm/v1/worker/gpu_model_runner.py).

Every step the runner gathered the target model's vocab-parallel logits across TP ranks ([tokens, 129,280], ~290 us
on this rig) only for the sampler to take argmax when every request is greedy. When every request is greedy and
nothing downstream reads real logits (no logprobs, penalties, allowed-token masks, bad words, active logit-bias /
min-tokens processors, structured output), the runner now asks LogitsProcessor.get_top_tokens for the vocab-parallel
argmax (local argmax + (value, id) gather) and hands the sampler one-hot logits at those ids: its argmax, the
rejection sampler's target argmax and the bonus token are the same tokens. Anything else takes the stock path.
DSV41_GREEDY_ARGMAX=0 = off. usage: patch_greedy_argmax.py <gpu_model_runner.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
if "_dsv41_greedy_logits" in s:
    print("already patched"); sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)


rep("class GPUModelRunner(", '''_DSV41_GREEDY_ARGMAX = os.environ.get("DSV41_GREEDY_ARGMAX", "1") != "0"


def _dsv41_greedy_logits(runner, h, scheduler_output):
    """dsv41 (ds41-patches/patch_greedy_argmax.py): one-hot logits at the vocab-parallel argmax when every request is
    greedy and nothing downstream needs real logits; None = compute the full logits."""
    if not _DSV41_GREEDY_ARGMAX or getattr(scheduler_output, "has_structured_output_requests", True):
        return None
    md = runner.input_batch.sampling_metadata
    if not (md.all_greedy and md.max_num_logprobs is None and md.no_penalties
            and md.allowed_token_ids_mask is None and not md.bad_words_token_ids):
        return None
    for proc in md.logitsprocs.non_argmax_invariant:
        state = getattr(proc, "biases", None)
        if state is None:
            state = getattr(proc, "min_toks", None)
        if state is None or state:          # unknown processor, or active for some request
            return None
    lm = getattr(runner.model, "language_model", runner.model)
    lp, head = getattr(lm, "logits_processor", None), getattr(lm, "lm_head", None)
    if lp is None or head is None or getattr(head, "tp_size", 1) < 2 or not hasattr(lp, "get_top_tokens"):
        return None
    ids = lp.get_top_tokens(head, h).reshape(-1, 1).to(torch.int64)
    out = torch.zeros(h.shape[0], lp.org_vocab_size, dtype=h.dtype, device=h.device)
    return out.scatter_(1, ids, 1.0)


class GPUModelRunner(''')
rep('''                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.''', '''                sample_hidden_states = hidden_states[logits_indices]
                logits = _dsv41_greedy_logits(self, sample_hidden_states, scheduler_output)
                if logits is None:
                    logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.''')
if "\nimport os\n" not in s:
    s = "import os\n" + s
open(p, "w").write(s)
print("patched", p)
