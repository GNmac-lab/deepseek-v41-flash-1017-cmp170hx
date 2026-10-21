#!/usr/bin/env python3
"""dsv41 v32: wo_a as an FP8 grouped GEMM (dsv41_wo_a.py) in the Ampere attention's _o_proj for <= 16 tokens. The FP8
codes are recovered exactly from the bf16 weight (power-of-two 32x32 scales) and the output is written in wo_b's
[t, g*r] layout (no flatten copy). DSV41_WO_A_FP8=0 = stock einsum. (Split from patch_v32_small.py; the plugin half
is patches/vllm-exl3/patch_exl3_ids_cast.py.)
usage: patch_v32_wo_a.py <site-packages> <dsv41_wo_a.py>"""
import shutil
import sys

sp, mod = sys.argv[1], sys.argv[2]
shutil.copy(mod, f"{sp}/vllm/models/deepseek_v4_1/ampere/dsv41_wo_a.py")


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


patch(f"{sp}/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py", [
    ("""            self._dsv41_wo_a_checked = True
        return super()._o_proj(o, positions)
""", """            self._dsv41_wo_a_checked = True
        # dsv41 (ds41-patches/patch_v32_small.py): FP8 grouped wo_a straight into wo_b's input layout
        if _DSV41_WO_A_FP8 and o.shape[0] <= 16:
            fp8 = getattr(self, "_dsv41_wo_a_fp8", None)
            if fp8 is None:  # first small eager call (warmup, before graph capture)
                from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as _w

                fp8 = _w.recover_fp8(wo_a._dsv4_wo_a_bf16) or False
                self._dsv41_wo_a_fp8 = fp8
                print(f"dsv41 wo_a fp8: {'ok' if fp8 else 'NOT exact, bf16 path'} ({self.prefix})", flush=True)
            if fp8:
                from vllm.models.deepseek_v4_1.ampere import dsv41_wo_a as _w
                from vllm.v1.attention.ops.rocm_aiter_mla_sparse import _fused_inverse_rope_gptj

                o_ref = _fused_inverse_rope_gptj(o, positions, self.rotary_emb.cos_sin_cache, self.rope_head_dim)
                o_ref = o_ref.reshape(o.shape[0], -1)
                G, R = self.n_local_groups, self.o_lora_rank
                if not o_ref.is_contiguous():
                    o_ref = o_ref.contiguous()
                return self.wo_b(_w.wo_a(o_ref, fp8[0], fp8[1], G, R, o_ref.shape[1] // G))
        return super()._o_proj(o, positions)
"""),
    ("\nimport torch\n", "\nimport os as _dsv41_wos\n_DSV41_WO_A_FP8 = _dsv41_wos.environ.get(\"DSV41_WO_A_FP8\", \"1\") != \"0\"\nimport torch\n"),
], "_DSV41_WO_A_FP8")
