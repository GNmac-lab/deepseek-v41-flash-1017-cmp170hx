#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""dsv41: greedy spec-decode verification straight from the vocab-parallel argmax ids (DSV41_GREEDY_IDS=1, default).

patch_greedy_argmax.py already computes the target's argmax per logits row with get_top_tokens (local max + an
all-gather of (value, index) pairs) and hands the sampler one-hot logits at those ids. The rejection sampler then
re-derives the same ids from the one-hot rows: it gathers the bonus and target rows, widens them to fp32, clones
them, and runs two full-vocab argmax reductions (about 38 us each at 6 x 129,280). This patch tags the one-hot
tensor with its ids, and when the rejection sampler gets a tagged tensor in the case that produced it (every request
greedy, no draft probabilities, no logprobs, no synthetic mode), it feeds those ids to the same greedy rejection
kernel. The one-hot argmax is exactly those ids, so the sampled tokens are identical.
usage: patch_greedy_ids.py <vllm package dir>"""
import sys

d = sys.argv[1]
MARK = "dsv41 greedy ids"


def patch(p, pairs):
    s = open(p).read()
    if MARK in s:
        print("already patched", p)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


patch(f"{d}/v1/worker/gpu_model_runner.py", [
    ("""    out = torch.zeros(h.shape[0], lp.org_vocab_size, dtype=h.dtype, device=h.device)
    return out.scatter_(1, ids, 1.0)
""", """    out = torch.zeros(h.shape[0], lp.org_vocab_size, dtype=h.dtype, device=h.device)
    out.scatter_(1, ids, 1.0)
    out._dsv41_ids = ids.view(-1)  # dsv41 greedy ids (patch_greedy_ids.py): the rejection sampler reads these
    return out
"""),
])

patch(f"{d}/v1/sample/rejection_sampler.py", [
    ("""        assert metadata.max_spec_len <= MAX_SPEC_LEN

        bonus_logits_indices = metadata.bonus_logits_indices
""", """        assert metadata.max_spec_len <= MAX_SPEC_LEN

        # dsv41 greedy ids (ds41-patches/patch_greedy_ids.py): one-hot logits from the vocab-parallel argmax carry
        # their ids; in the case that built them, verify from the ids instead of re-reducing the one-hot rows
        _ids = getattr(logits, "_dsv41_ids", None)
        if (
            _ids is not None
            and _DSV41_GREEDY_IDS
            and draft_probs is None
            and sampling_metadata.all_greedy
            and sampling_metadata.max_num_logprobs is None
            and not sampling_metadata.logprob_token_ids
            and not self.synthetic_mode
        ):
            return _dsv41_greedy_verify(metadata, _ids)

        bonus_logits_indices = metadata.bonus_logits_indices
"""),
    ("""PLACEHOLDER_TOKEN_ID: tl.constexpr = -1
""", """PLACEHOLDER_TOKEN_ID: tl.constexpr = -1

import os as _dsv41_os

# dsv41 greedy ids (ds41-patches/patch_greedy_ids.py)
_DSV41_GREEDY_IDS = _dsv41_os.environ.get("DSV41_GREEDY_IDS", "1") != "0"


def _dsv41_greedy_verify(metadata, ids):
    \"\"\"Greedy rejection sampling from per-logits-row argmax ids (the one-hot logits' argmax).\"\"\"
    batch_size = len(metadata.num_draft_tokens)
    bonus = ids[metadata.bonus_logits_indices].to(torch.int32).unsqueeze(-1).contiguous()
    target_argmax = ids[metadata.target_logits_indices]
    out = torch.full(
        (batch_size, metadata.max_spec_len + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=ids.device,
    )
    rejection_greedy_sample_kernel[(batch_size,)](
        out,
        metadata.cu_num_draft_tokens,
        metadata.draft_token_ids,
        target_argmax,
        bonus,
        None,
        metadata.max_spec_len,
        None,
        None,
        SYNTHETIC_MODE=False,
    )
    return SamplerOutput(sampled_token_ids=out, logprobs_tensors=None)
"""),
])
