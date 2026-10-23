# dsv41 (runtime variant): confidence-gated verification. Skip the routed-expert MoE for draft tokens that will most
# likely be rejected, and force-reject from the first skipped draft, with every shape unchanged (T = 6 decode graphs,
# async scheduling untouched).
#
# Each request r verifies tokens [x0, d1..d5] at batch positions 6r..6r+5 (uniform decode, R <= 8 requests). MASK[t] = 1
# keeps token t; per request the mask is a prefix: tokens 0..m_r.
#   - router (dsv4_topk, FULL decode graphs with 6R tokens): a masked token gets the expert id sentinel NUM_EXPERTS and
#     weight 0, which the coop MoE kernel (R <= 7) and the tensor-core MoE (R = 8) both drop, so its experts are never
#     loaded;
#   - rejection sampler (R requests, 5 drafts each, all greedy): draft j > m_r becomes -1 and can never match, so at
#     most d1..dm_r are accepted and the correction token comes from the logits at token m_r, which is fully computed.
#   - only a step in which every scheduled request verifies exactly 1 + 5 tokens, run as a FULL graph with all-greedy
#     sampling, is gated. Any other step runs with MASK reset to ones before its forward (nothing skipped, nothing
#     forced): vLLM also sends a 6-token prefill chunk and draft-less 6k-token steps to the FULL 6R graphs, a PIECEWISE
#     or eager forward (PP) never masks the router, and random rejection sampling would index probabilities with -1.
# Within a gated step any mask is exact as long as the router and the rejection sampler read the same one, which they
# do; a mask written for an older batch layout only costs speed.
# Tokens 0..m never depend on later tokens (causal attention, per-token MoE / mHC / Engram), so every accepted and
# correction token is the one the ungated model gives: outputs unchanged. Masked tokens behave like rejected drafts.
#   DSV41_GATE_M=m      fixed m (0..5), for measuring step cost by verified drafts (development only)
#   DSV41_GATE_TAU=tau  dynamic: DSpark's confidence head, keep drafts while cumprod(conf) >= tau (in the sampling graph);
#                       default 0.10 (fastbench prose: +3.5% simulated, ~0 overhead); 0 = off
# Measured (v75, 1 user): each skipped token saves ~0.8-1.0 ms of a prose step (prose step by verified drafts m = 5..0:
# 18.66 17.71 16.85 15.89 15.13 14.67 ms). Structured / code outputs are identical under fixed m = 1 and m = 3.
import dataclasses
import os

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.fused_moe.router import dsv4_topk as _tk
from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router as _fr
from vllm.v1.sample import rejection_sampler as _rs

N_SPEC = 5
MAX_REQS = 8
_dev = torch.device("cuda", torch.cuda.current_device())
MASK = getattr(_tk, "_dsv41_gate_mask", None)
if MASK is None or MASK.device != _dev or MASK.numel() < 6 * MAX_REQS + 2:
    MASK = torch.ones(6 * MAX_REQS + 2, dtype=torch.int32, device=_dev)
    _tk._dsv41_gate_mask = MASK
_DRAFT_IDX: dict = {}


def _draft_idx(reqs: int) -> torch.Tensor:
    """Batch positions of the drafts d1..d5 of requests 0..reqs-1 (int64, cached)."""
    t = _DRAFT_IDX.get(reqs)
    if t is None:
        t = _DRAFT_IDX[reqs] = torch.tensor([6 * r + 1 + j for r in range(reqs) for j in range(N_SPEC)],
                                            dtype=torch.int64, device=_dev)
    return t
STATS = {"masked_topk_calls": 0, "forced": 0, "gated_steps": 0, "ungated_steps": 0}
TAU = getattr(_tk, "_dsv41_gate_tau", None)
if TAU is None or TAU.device != _dev:
    TAU = torch.zeros(1, dtype=torch.float32, device=_dev)
    _tk._dsv41_gate_tau = TAU

_ORIG_TOPK = getattr(_tk, "_dsv41_gate_orig_topk", None) or _tk.dsv4_topk
_tk._dsv41_gate_orig_topk = _ORIG_TOPK
_ORIG_RS = _rs.RejectionSampler.__dict__.get("_dsv41_gate_orig_forward") or _rs.RejectionSampler.forward
_rs.RejectionSampler._dsv41_gate_orig_forward = _ORIG_RS


DEFAULT_TAU = "0.10"


def _tau() -> float:
    return float(os.environ.get("DSV41_GATE_TAU", DEFAULT_TAU) or 0)


def _enabled() -> bool:
    return os.environ.get("DSV41_GATE_M", "") != "" or _tau() > 0


def _dynamic() -> bool:
    return os.environ.get("DSV41_GATE_M", "") == "" and _tau() > 0


_FIXED = None    # DSV41_GATE_M's mask (device), restored before each gated step


def set_m(worker=None) -> list:
    global _FIXED
    TAU.fill_(_tau())
    m = os.environ.get("DSV41_GATE_M", "")
    if m != "":
        k = max(0, min(N_SPEC, int(m)))
        _FIXED = torch.tensor(([1] * (k + 1) + [0] * (N_SPEC - k)) * MAX_REQS + [1, 1], dtype=torch.int32,
                              device=_dev)
        MASK.copy_(_FIXED)
    else:
        _FIXED = None
        MASK.fill_(1)
    return MASK.tolist()


@triton.jit
def _dsv41_topk_masked_kernel(gating_output_ptr, correction_bias_ptr, topk_weights_ptr, topk_ids_ptr,
                              routed_scaling_factor, input_ids_ptr, bias_vl_ptr, image_sentinel_lo, mask_ptr,
                              NUM_EXPERTS: tl.constexpr, BLOCK_N: tl.constexpr, HAS_VL: tl.constexpr):
    # _dsv4_topk_kernel (sm80: no PDL) plus the token mask
    row = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_N)
    expert_mask = expert_offsets < NUM_EXPERTS
    bias = tl.load(correction_bias_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
    if HAS_VL:
        token_id = tl.load(input_ids_ptr + row).to(tl.int64)
        bias_vl = tl.load(bias_vl_ptr + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
        is_image = (token_id >= image_sentinel_lo) & (token_id < image_sentinel_lo + 5)
        bias = tl.where(is_image, bias_vl, bias)
    logits = tl.load(gating_output_ptr + row * NUM_EXPERTS + expert_offsets, mask=expert_mask, other=0.0).to(tl.float32)
    weights = tl.sqrt(tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits))))
    current = tl.where(expert_mask, weights + bias, -float("inf"))
    current = tl.where(current == current, current, -1e30)
    topk_offsets = tl.arange(0, 8)
    selected_weights = tl.zeros([8], dtype=tl.float32)
    selected_ids = tl.zeros([8], dtype=tl.int32)
    for slot in tl.static_range(6):
        max_value = tl.max(current, axis=0)
        candidate = tl.where(current == max_value, expert_offsets, NUM_EXPERTS)
        expert_id = tl.min(candidate, axis=0).to(tl.int32)
        selected_weight = tl.sum(tl.where(expert_offsets == expert_id, weights, 0.0), axis=0)
        is_slot = topk_offsets == slot
        selected_weights = tl.where(is_slot, selected_weight, selected_weights)
        selected_ids = tl.where(is_slot, expert_id, selected_ids)
        current = tl.where(expert_offsets == expert_id, -float("inf"), current)
    weight_sum = tl.sum(selected_weights, axis=0)
    selected_weights *= routed_scaling_factor / tl.where(weight_sum > 0.0, weight_sum, 1.0)
    keep = tl.load(mask_ptr + row) != 0
    selected_weights = tl.where(keep, selected_weights, 0.0)
    selected_ids = tl.where(keep, selected_ids, NUM_EXPERTS)
    output_mask = topk_offsets < 6
    output_offsets = row * 6 + topk_offsets
    tl.store(topk_weights_ptr + output_offsets, selected_weights, mask=output_mask)
    tl.store(topk_ids_ptr + output_offsets, selected_ids, mask=output_mask)


def _use_mask(gating_output, bias_vl, image_sentinel_lo) -> bool:
    t = gating_output.shape[0]
    if not _enabled() or t % (N_SPEC + 1) or not N_SPEC + 1 <= t <= (N_SPEC + 1) * MAX_REQS:
        return False
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return is_forward_context_available() and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL


def dsv4_topk(gating_output, correction_bias, indices_dtype, routed_scaling_factor, input_ids=None, bias_vl=None,
              image_sentinel_lo=0):
    if not _use_mask(gating_output, bias_vl, image_sentinel_lo):
        return _ORIG_TOPK(gating_output, correction_bias, indices_dtype, routed_scaling_factor, input_ids=input_ids,
                          bias_vl=bias_vl, image_sentinel_lo=image_sentinel_lo)
    num_tokens, num_experts = gating_output.shape
    has_vl = bias_vl is not None and image_sentinel_lo > 0
    if bias_vl is not None:
        assert input_ids is not None and bias_vl.dtype == torch.float32 and bias_vl.is_contiguous()
    topk_weights = gating_output.new_empty((num_tokens, 6), dtype=torch.float32)
    topk_ids = gating_output.new_empty((num_tokens, 6), dtype=indices_dtype)
    _dsv41_topk_masked_kernel[(num_tokens,)](gating_output, correction_bias, topk_weights, topk_ids,
                                             routed_scaling_factor, input_ids, bias_vl, image_sentinel_lo, MASK,
                                             NUM_EXPERTS=num_experts, BLOCK_N=triton.next_power_of_2(num_experts),
                                             HAS_VL=has_vl, num_warps=1)
    STATS["masked_topk_calls"] += 1
    return topk_weights, topk_ids


def rejection_forward(self, metadata, draft_probs, logits, sampling_metadata):
    reqs = len(metadata.num_draft_tokens)
    if (_enabled() and 1 <= reqs <= MAX_REQS and metadata.num_draft_tokens == [N_SPEC] * reqs
            and metadata.draft_token_ids.numel() == N_SPEC * reqs and getattr(sampling_metadata, "all_greedy", False)):
        d = metadata.draft_token_ids
        d = torch.where(MASK[_draft_idx(reqs)] != 0, d, torch.full_like(d, -1))
        metadata = dataclasses.replace(metadata, draft_token_ids=d)
        STATS["forced"] += 1
    return _ORIG_RS(self, metadata, draft_probs, logits, sampling_metadata)


from hybrid.dspark_proposer import DSparkProposer as _D  # noqa: E402

_ORIG_BODY = _D.__dict__.get("_dsv41_gate_orig_body") or _D._dsv41_sample_body
_D._dsv41_gate_orig_body = _ORIG_BODY


def _keep(conf: torch.Tensor) -> torch.Tensor:
    """[R, 5] confidences -> flat int32 keep flags (per request: drafts while cumprod(conf) >= TAU)."""
    return (torch.cumprod(conf, 1) >= TAU).to(torch.int32).cumprod(1).to(MASK.dtype).reshape(-1)


def sample_body(self, num_reqs, sample_hidden):
    """DSpark draft sampling (captured into its lazy graph) + the gate: keep drafts while cumprod(conf) >= TAU."""
    drafts = _ORIG_BODY(self, num_reqs, sample_hidden)
    if 1 <= num_reqs <= MAX_REQS and _dynamic() and self.model.model.confidence_head is not None:
        anchor = self.input_ids[self._anchor_idx[:num_reqs]].view(num_reqs, 1).to(drafts.dtype)
        prev = torch.cat([anchor, drafts[:, :-1]], dim=1)                   # [R, 5] token before each draft
        emb = self.model.markov_embed(prev.reshape(-1))
        conf = self.model.compute_confidence(sample_hidden, emb).reshape(num_reqs, N_SPEC).float()
        MASK.index_copy_(0, _draft_idx(num_reqs), _keep(conf))
    return drafts


from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _R  # noqa: E402

_ORIG_FWD = _R.__dict__.get("_dsv41_gate_orig_fwd") or _R._model_forward
_R._dsv41_gate_orig_fwd = _ORIG_FWD


_ORIG_EXEC = _R.__dict__.get("_dsv41_gate_orig_exec") or _R.execute_model
_R._dsv41_gate_orig_exec = _ORIG_EXEC


def _decode_layout(so) -> bool:
    """Every scheduled request verifies exactly 1 + N_SPEC tokens (a decode with N_SPEC drafts, not a prefill chunk)."""
    n, spec = so.num_scheduled_tokens, so.scheduled_spec_decode_tokens
    return 1 <= len(n) <= MAX_REQS and all(v == N_SPEC + 1 and len(spec.get(r, ())) == N_SPEC for r, v in n.items())


def execute_model(self, scheduler_output, *a, **k):
    self._dsv41_gate_layout = _decode_layout(scheduler_output)
    return _ORIG_EXEC(self, scheduler_output, *a, **k)


def _full_graph() -> bool:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available

    return is_forward_context_available() and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL


def model_forward(self, *a, **k):
    """Only an all-greedy FULL-graph step of 5-draft decodes is gated; any other step runs with MASK = ones."""
    if _enabled() and not torch.cuda.is_current_stream_capturing():
        md = getattr(getattr(self, "input_batch", None), "sampling_metadata", None)
        if getattr(self, "_dsv41_gate_layout", False) and md is not None and md.all_greedy and _full_graph():
            if _FIXED is not None:
                MASK.copy_(_FIXED)
            STATS["gated_steps"] += 1
        else:
            MASK.fill_(1)
            STATS["ungated_steps"] += 1
    return _ORIG_FWD(self, *a, **k)


def stats(worker=None) -> dict:
    r = dict(STATS, mask=MASK.tolist(), tau=TAU.item())
    for k in list(STATS):
        STATS[k] = 0
    return r


def _selftest() -> None:
    g = torch.Generator(device="cpu").manual_seed(3)
    saved = MASK.clone()
    try:
        for ne in (384, 256):
            go = (torch.randn(6, ne, generator=g) * 3).to(_dev)
            bias = (torch.randn(ne, generator=g) * 0.1).to(_dev)
            bvl = (torch.randn(ne, generator=g) * 0.1).to(_dev)
            ids_in = torch.tensor([5, 1000, 1001, 1004, 1005, 7], device=_dev)    # tokens 1..3 are image sentinels
            w0, i0 = _ORIG_TOPK(go, bias, torch.int32, 2.5, input_ids=ids_in, bias_vl=bvl, image_sentinel_lo=1000)
            for mask in ([1] * 8, [1, 1, 1, 0, 0, 0, 1, 1]):
                MASK[:8].copy_(torch.tensor(mask, dtype=torch.int32))
                w1 = torch.empty_like(w0)
                i1 = torch.empty_like(i0)
                _dsv41_topk_masked_kernel[(6,)](go, bias, w1, i1, 2.5, ids_in, bvl, 1000, MASK, NUM_EXPERTS=ne,
                                                BLOCK_N=triton.next_power_of_2(ne), HAS_VL=True, num_warps=1)
                torch.cuda.synchronize()
                keep = torch.tensor(mask[:6], device=_dev).bool()
                ok = (torch.equal(w1[keep], w0[keep]) and torch.equal(i1[keep], i0[keep])
                      and bool((i1[~keep] == ne).all()) and bool((w1[~keep] == 0).all()))
                if not ok:
                    raise RuntimeError(f"dsv41_gate topk self-test FAILED experts={ne} mask={mask}")
    finally:
        MASK.copy_(saved)


_selftest()


def _selftest_mask() -> None:
    saved, tau = MASK.clone(), TAU.clone()
    try:
        TAU.fill_(0.10)
        conf = torch.tensor([[0.9, 0.9, 0.1, 0.9, 0.9], [0.05, 0.9, 0.9, 0.9, 0.9], [0.99] * 5], device=_dev)
        MASK.fill_(1)
        MASK.index_copy_(0, _draft_idx(3), _keep(conf))
        want = [1, 1, 1, 0, 0, 0] + [1, 0, 0, 0, 0, 0] + [1] * 6
        if MASK[:18].tolist() != want:
            raise RuntimeError(f"dsv41_gate mask self-test FAILED: {MASK[:18].tolist()}")
    finally:
        MASK.copy_(saved)
        TAU.copy_(tau)


_selftest_mask()


def _selftest_layout() -> None:
    so = lambda n, spec: dataclasses.make_dataclass("S", ["num_scheduled_tokens", "scheduled_spec_decode_tokens"])(n, spec)
    d5 = [-1] * N_SPEC
    cases = [(so({"a": 6, "b": 6}, {"a": d5, "b": d5}), True),
             (so({"a": 6}, {}), False),                                   # a 6-token prefill chunk
             (so({"a": 6, "b": 6}, {"a": d5}), False),                     # decode + 6-token prefill
             (so({str(i): 1 for i in range(6)}, {}), False),               # draft-less 6-request step
             (so({"a": 3, "b": 3}, {"a": [-1, -1], "b": [-1, -1]}), False),
             (so({str(i): 6 for i in range(MAX_REQS + 1)}, {str(i): d5 for i in range(MAX_REQS + 1)}), False)]
    for s_, want in cases:
        if _decode_layout(s_) != want:
            raise RuntimeError(f"dsv41_gate layout self-test FAILED: {s_} -> {not want}")


_selftest_layout()
for _r in range(1, MAX_REQS + 1):    # created here: no host-to-device copy may run inside a graph capture
    _draft_idx(_r)
set_m()
SETATTR = [{"target": "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router.dsv4_topk",
            "source": "dsv41_gate.dsv4_topk"},
           {"target": "vllm.v1.sample.rejection_sampler.RejectionSampler.forward",
            "source": "dsv41_gate.rejection_forward"},
           {"target": "hybrid.dspark_proposer.DSparkProposer._dsv41_sample_body", "source": "dsv41_gate.sample_body"},
           {"target": "vllm.v1.worker.gpu_model_runner.GPUModelRunner._model_forward",
            "source": "dsv41_gate.model_forward"},
           {"target": "vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model",
            "source": "dsv41_gate.execute_model"}]
