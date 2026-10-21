#!/usr/bin/env python3
"""Per-layer uneven TP split of the FP8 shared expert, complementing the EXL3 routed-expert split.
Files: vllm/models/deepseek_v4/nvidia/model.py (DeepseekV4MLP), vllm/models/deepseek_v4_1/nvidia/model.py (loader).

The routed experts' intermediate (2304 = 18 Hadamard blocks of 128) splits 5/5/4/4 over TP=4, rotated per layer
(vllm_exl3 moe_tp_rotation: rank r takes chunk (r + L) % 4; chunks 0-1 are 640 wide). The two 640-wide ranks are the
critical path of every MoE layer (the MoE all-reduce on the others waits ~50 us, profiled 2026-10-20), yet the shared
expert (TP-even, 640 padded columns per rank, Marlin FP8 on a side stream) also runs on them. Now each target layer
splits the shared expert unevenly: H blocks (DSV41_SHARED_HEAVY_BLOCKS, default 2) on that layer's two heavy ranks,
the rest on the two light ranks. The layer is built as full-width non-TP linears, loaded by the stock loaders (which
convert the checkpoint's FP8 scale layout), and narrowed to this rank's columns at the end of load_weights, before
the Marlin repack. Partial sums still meet in the MoE all-reduce: same result up to summation order. Draft (DSpark)
layers keep the stock split. DSV41_SHARED_REBALANCE=0 = stock. usage: patch_shared_rebalance.py <SP>
"""
import sys

SP = sys.argv[1]
V4 = SP + "/vllm/models/deepseek_v4/nvidia/model.py"
V41 = SP + "/vllm/models/deepseek_v4_1/nvidia/model.py"


def rep(s, old, new):
    assert s.count(old) == 1, (old[:70], s.count(old))
    return s.replace(old, new)


s = open(V4).read()
assert "_dsv41_shared_plan" not in s
s = rep(s, "class DeepseekV4MLP(nn.Module):", '''_DSV41_SHARED_SLICES: dict = {}   # target layer index -> (start column, width) of this rank's shared-expert slice


def _dsv41_shared_plan(prefix: str, intermediate_size: int, quant_config, is_sequence_parallel: bool):
    """dsv41 (ds41-patches/patch_shared_rebalance.py): (start, width) of this rank's shared-expert columns on this
    layer, or None for the stock even split."""
    import os

    if os.environ.get("DSV41_SHARED_REBALANCE", "1") == "0" or is_sequence_parallel:
        return None
    if getattr(quant_config, "weight_block_size", None) is None or intermediate_size % 128:
        return None
    if os.environ.get("VLLM_EXL3_MOE_TP_ALIGN", "0") != "128":
        return None
    m = re.search(r"layers\\.(\\d+)\\.", prefix + ".")
    if m is None:
        return None
    layer = int(m.group(1))
    from vllm.config import get_current_vllm_config
    from vllm.distributed import get_tensor_model_parallel_world_size

    hf = get_current_vllm_config().model_config.hf_text_config
    if layer >= hf.num_hidden_layers:
        return None                                          # DSpark draft layers: stock split
    tp = get_tensor_model_parallel_world_size()
    if tp < 2:
        return None
    rot = layer % tp if os.environ.get("VLLM_EXL3_MOE_TP_ROTATE", "0") == "1" else 0
    base, rem = divmod(hf.moe_intermediate_size // 128, tp)
    heavy = [c < rem for c in range(tp)]                      # aligned_tp_split: chunks < rem get the extra block
    nh = sum(heavy)
    if nh == 0:
        return None
    nblk = intermediate_size // 128
    h = int(os.environ.get("DSV41_SHARED_HEAVY_BLOCKS", "2"))
    if h < 1 or (nblk - nh * h) % (tp - nh):
        return None
    l = (nblk - nh * h) // (tp - nh)
    sizes = [h if heavy[c] else l for c in range(tp)]
    chunk = (get_tensor_model_parallel_rank() + rot) % tp
    plan = (sum(sizes[:chunk]) * 128, sizes[chunk] * 128)
    _DSV41_SHARED_SLICES[layer] = plan
    return plan


class DeepseekV4MLP(nn.Module):''')
s = rep(s, '''        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded''', '''        super().__init__()

        plan = _dsv41_shared_plan(prefix, intermediate_size, quant_config, is_sequence_parallel)
        if plan is not None and not reduce_results:
            # full-width, non-TP; narrowed to this rank's columns after loading (_dsv41_slice_shared_params)
            self._dsv41_plan, self._dsv41_full = plan, intermediate_size
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size, [intermediate_size] * 2, bias=False, quant_config=quant_config,
                disable_tp=True, prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size, hidden_size, bias=False, quant_config=quant_config,
                reduce_results=False, disable_tp=True, prefix=f"{prefix}.down_proj",
            )
            if hidden_act != "silu":
                raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
            self.act_fn = SiluAndMulWithClamp(swiglu_limit) if swiglu_limit is not None else SiluAndMul()
            return

        # If is_sequence_parallel, the input and output tensors are sharded''')
open(V4, "w").write(s)

t = open(V41).read()
t = rep(t, '''            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(''', '''            if pad_shared_expert and ".shared_experts." in name and not _dsv41_planned(name):
                loaded_weight = self._pad_shared_expert_weight(''')
t = rep(t, '''        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(''', '''        _dsv41_slice_shared_params(self)
        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(''')
t = rep(t, "class DeepseekV4MoE(DeepseekV4MoEBase):", '''def _dsv41_planned(name: str) -> bool:
    import re as _re

    from vllm.models.deepseek_v4.nvidia.model import _DSV41_SHARED_SLICES

    m = _re.search(r"layers\\.(\\d+)\\.", name)
    return bool(m) and int(m.group(1)) in _DSV41_SHARED_SLICES


def _dsv41_slice_shared_params(model) -> None:
    """dsv41 (ds41-patches/patch_shared_rebalance.py): narrow every rebalanced shared expert (loaded full width) to
    this rank's columns, in parameter format, before process_weights_after_loading repacks it."""
    n_mod = 0
    for mod in model.modules():
        plan = getattr(mod, "_dsv41_plan", None)
        if plan is None:
            continue
        start, width = plan
        full = mod._dsv41_full
        gu, dn = mod.gate_up_proj, mod.down_proj
        for _, p in gu.named_parameters(recurse=False):
            if p.dim() < 1 or (2 * full) % p.shape[0]:
                continue
            g = (2 * full) // p.shape[0]                  # intermediate columns per parameter row
            if g > width or start % g or width % g:
                continue                                  # e.g. per-shard scalar scales: unchanged
            a, n = start // g, width // g
            o = full // g
            p.data = torch.cat([p.data[a:a + n], p.data[o + a:o + a + n]], dim=0).contiguous()
        for _, p in dn.named_parameters(recurse=False):
            if p.dim() < 2 or full % p.shape[1]:
                continue
            g = full // p.shape[1]
            if g > width or start % g or width % g:
                continue
            p.data = p.data[:, start // g:(start + width) // g].contiguous()
        gu.output_size = gu.output_size_per_partition = 2 * width
        for attr in ("output_sizes", "output_partition_sizes", "logical_widths"):
            if hasattr(gu, attr):
                setattr(gu, attr, [width, width])
        dn.input_size = dn.input_size_per_partition = width
        n_mod += 1
    if n_mod:
        logger.info("dsv41: %d shared experts narrowed to this rank's slice", n_mod)


class DeepseekV4MoE(DeepseekV4MoEBase):''')
open(V41, "w").write(t)
print("patched", V4, V41)
