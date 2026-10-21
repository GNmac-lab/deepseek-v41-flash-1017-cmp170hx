#!/usr/bin/env python3
"""dsv41 v35: keep the attention wo_a only as exact FP8 codes; frees ~0.72 GiB per card for the KV cache.

Since v32 decode (<= 16 tokens) runs wo_a from FP8 codes recovered exactly from the bf16 weight (power-of-two 32x32
block scales), but the 16.8 MB/layer bf16 weight stayed allocated for the prefill einsum. Now the recovery runs right
after weight loading (before vLLM sizes the KV cache), each layer's Triton dequant is checked bit-for-bit against the
bf16 weight, and only then is the bf16 weight freed. Prefill rebuilds the identical bf16 weight into one shared
scratch buffer per device and runs the unchanged einsum, so outputs are bit-identical to v34.
DSV41_WO_A_FREE=0 keeps the bf16 weight (v34 behaviour).
usage: patch_wo_a_free.py <site-packages>"""
import sys

sp = sys.argv[1]


def patch(p, pairs, mark):
    s = open(p).read()
    if mark in s:
        print("already patched", p)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w").write(s)
    print("patched", p)


DEQUANT = '''

@triton.jit
def _dequant_kernel(w_ptr, s_ptr, o_ptr, R: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, BD: tl.constexpr):
    # o[g * R + r, d] = e4m3(w[g * R + r, d]) * s[g, r / 32, d / 32]; power-of-two scales, so exact in bf16
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.program_id(1) * BD + tl.arange(0, BD)
    w = tl.load(w_ptr + rows[:, None] * D + cols[None, :])
    srow = (rows // R) * (R // 32) + (rows % R) // 32
    s = tl.load(s_ptr + srow[:, None] * (D // 32) + (cols // 32)[None, :])
    tl.store(o_ptr + rows[:, None] * D + cols[None, :], _e4m3_to_bf16(w) * s.to(tl.bfloat16))


def dequant(w_u8: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, G: int, R: int, D: int) -> torch.Tensor:
    """dsv41 (ds41-patches/patch_wo_a_free.py): rebuild the exact bf16 wo_a [G, R, D] from its FP8 codes into
    `out` (>= G*R*D bf16 elements)."""
    o = out[: G * R * D].view(G * R, D)
    _dequant_kernel[(G * R // 32, D // 128)](w_u8, scale, o, R, D, BR=32, BD=128, num_warps=4)
    return o.view(G, R, D)
'''

W = f"{sp}/vllm/models/deepseek_v4_1/ampere/dsv41_wo_a.py"
s = open(W).read()
if "def dequant(" not in s:
    open(W, "w").write(s.rstrip("\n") + "\n" + DEQUANT)
    print("patched", W)

A = f"{sp}/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py"
patch(A, [
    ("""    def _o_proj(self, o, positions):
        # wo_a went through the MXFP8 *emulation* kernel""", """    def _dsv41_post_load(self):
        # dsv41 (ds41-patches/patch_wo_a_free.py): recover wo_a's FP8 codes before the KV cache is sized, verify
        # the dequant bit-for-bit, then free the bf16 weight. Prefill rebuilds it into a shared scratch buffer.
        if not (_DSV41_WO_A_FP8 and _DSV41_WO_A_FREE) or getattr(self, "_dsv41_wo_a_freed", False):
            return
        w = self.wo_a.weight.data
        if w.element_size() < 2 or w.numel() == 0:
            return
        from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as _w

        G, R = self.n_local_groups, self.o_lora_rank
        D = w.numel() // (G * R)
        w3 = w.view(G, R, D).to(torch.bfloat16)
        fp8 = _w.recover_fp8(w3)
        if not fp8:
            print(f"dsv41 wo_a free: NOT exact, keeping bf16 ({self.prefix})", flush=True)
            return
        scratch = _DSV41_WO_A_SCRATCH.get(w.device)
        if scratch is None or scratch.numel() < G * R * D:
            scratch = torch.empty(G * R * D, dtype=torch.bfloat16, device=w.device)
            _DSV41_WO_A_SCRATCH[w.device] = scratch
        if not torch.equal(_w.dequant(fp8[0], fp8[1], scratch, G, R, D), w3):
            print(f"dsv41 wo_a free: dequant mismatch, keeping bf16 ({self.prefix})", flush=True)
            return
        self._dsv41_wo_a_fp8 = fp8
        self._dsv41_wo_a_dims = (G, R, D)
        self._dsv41_wo_a_checked = True
        self.wo_a.weight = torch.nn.Parameter(torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False)
        if hasattr(self.wo_a, "_dsv4_wo_a_bf16"):
            del self.wo_a._dsv4_wo_a_bf16
        self._dsv41_wo_a_freed = True
        del w, w3

    def _o_proj(self, o, positions):
        # wo_a went through the MXFP8 *emulation* kernel"""),
    ("""                return self.wo_b(_w.wo_a(o_ref, fp8[0], fp8[1], G, R, o_ref.shape[1] // G))
        return super()._o_proj(o, positions)
""", """                return self.wo_b(_w.wo_a(o_ref, fp8[0], fp8[1], G, R, o_ref.shape[1] // G))
        if getattr(self, "_dsv41_wo_a_freed", False):
            # dsv41 (ds41-patches/patch_wo_a_free.py): exact bf16 wo_a rebuilt for the stock einsum (prefill)
            from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as _w

            G, R, D = self._dsv41_wo_a_dims
            fp8 = self._dsv41_wo_a_fp8
            wo_a._dsv4_wo_a_bf16 = _w.dequant(fp8[0], fp8[1], _DSV41_WO_A_SCRATCH[fp8[0].device], G, R, D)
        return super()._o_proj(o, positions)
"""),
    ("""_DSV41_WO_A_FP8 = _dsv41_wos.environ.get("DSV41_WO_A_FP8", "1") != "0"
""", """_DSV41_WO_A_FP8 = _dsv41_wos.environ.get("DSV41_WO_A_FP8", "1") != "0"
_DSV41_WO_A_FREE = _dsv41_wos.environ.get("DSV41_WO_A_FREE", "1") != "0"
_DSV41_WO_A_SCRATCH = {}
"""),
], "_dsv41_post_load")

U = f"{sp}/vllm/model_executor/model_loader/utils.py"
patch(U, [
    ("""    # Process HPC modules (HpcRopeNorm, etc.) that rely on
""", """    # dsv41 (ds41-patches/patch_wo_a_free.py): model post-load hooks, before vLLM measures memory for the KV cache
    _dsv41_hooks = [m for _, m in model.named_modules() if callable(getattr(m, "_dsv41_post_load", None))]
    for _dsv41_m in _dsv41_hooks:
        _dsv41_m._dsv41_post_load()
    if _dsv41_hooks:
        _dsv41_freed = sum(bool(getattr(m, "_dsv41_wo_a_freed", False)) for m in _dsv41_hooks)
        print(f"dsv41 wo_a free: {_dsv41_freed}/{len(_dsv41_hooks)} layers kept as FP8 only", flush=True)
        torch.cuda.empty_cache()

    # Process HPC modules (HpcRopeNorm, etc.) that rely on
"""),
], "patch_wo_a_free.py")
