#!/usr/bin/env python3
"""dsv41: in the fused-MoE runner, launch the router GEMM (critical path: gate -> top-k -> routed experts) before the
shared experts are enqueued on the aux stream. Stock order enqueues the shared experts first; their GEMV fills the
SMs and the router starts ~9 us late on every MoE layer (v61 profile). The shared experts still wait on an event
recorded before the gate, so they overlap the router / top-k chain exactly as before; only the launch order changes.
Same kernels and inputs: bit-identical. DSV41_GATE_FIRST=0 = stock.
usage: patch_gate_first.py <site-packages>"""
import sys

p = f"{sys.argv[1]}/vllm/model_executor/layers/fused_moe/runner/moe_runner.py"
s = open(p).read()
if "DSV41_GATE_FIRST" in s:
    print("already patched")
    sys.exit(0)
old = """        shared_experts_overlapping = False
        if self._shared_experts is not None:
            shared_experts_overlapping = self._shared_experts.maybe_forward_async(
                shared_experts_input
            )

        # If the Runner holds the gate, apply it after the stream sync,
        # so it can run overlapped with the
        # NOTE: in future PR, MoE runner will always hold the gate.
        if self.gate is not None:
            if self._fse_fuse_gate:
                self._maybe_fuse_gate_weights()
                router_logits = F.linear(hidden_states, self._combined_gate_weight)
            else:
                router_logits, _ = self.gate(hidden_states)
"""
new = """        shared_experts_overlapping = False
        # dsv41 (ds41-patches/patch_gate_first.py): record the shared experts' input point, launch the gate (router
        # GEMM) on the main stream, then enqueue the shared experts on the aux stream behind that event
        se = self._shared_experts
        if (__import__("os").environ.get("DSV41_GATE_FIRST", "1") != "0" and se is not None and self.gate is not None
                and not self._fse_fuse_gate
                and se._determine_shared_experts_order(shared_experts_input)
                == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED):
            idx = se._output_idx
            assert se._output[idx] is None
            se._input_ready_event[idx].record(torch.cuda.current_stream())
            router_logits, _ = self.gate(hidden_states)
            with torch.cuda.stream(se._stream):
                se._input_ready_event[idx].wait(se._stream)
                se._output[idx] = se._layer(shared_experts_input)
                se._output_ready_event[idx].record(se._stream)
            shared_experts_overlapping = True
        else:
            if self._shared_experts is not None:
                shared_experts_overlapping = self._shared_experts.maybe_forward_async(
                    shared_experts_input
                )

            # If the Runner holds the gate, apply it after the stream sync,
            # so it can run overlapped with the
            # NOTE: in future PR, MoE runner will always hold the gate.
            if self.gate is not None:
                if self._fse_fuse_gate:
                    self._maybe_fuse_gate_weights()
                    router_logits = F.linear(hidden_states, self._combined_gate_weight)
                else:
                    router_logits, _ = self.gate(hidden_states)
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
if "SharedExpertsOrder" not in s.split("class ")[0]:
    s = s.replace("from vllm.model_executor.layers.fused_moe.runner.shared_experts import (",
                  "from vllm.model_executor.layers.fused_moe.runner.shared_experts import (\n    SharedExpertsOrder,", 1)
open(p, "w").write(s)
print("patched", p)
