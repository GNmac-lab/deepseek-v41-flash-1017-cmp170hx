#!/usr/bin/env python3
"""dsv41: fused mHC post + delayed pre for small batches on sm80 (ds41-patches/dsv41_mhc_fused.py).

Per sublayer boundary upstream runs mhc_post + hc_prenorm_gemm + mhc_pre_big_fuse_with_norm (~24 us at 6 tokens,
the GEMM re-reads the 2 MB fp32 fn once per token).  Now: one token-blocked Triton kernel (post-mapping, bf16 residual,
collapse, GEMM partials) + an RMSNorm kernel, with the sinkhorn on a side stream that is joined before the next
post-mapping and before the layer returns (FULL graphs / eager only; PIECEWISE runs it inline).  Residual is
bit-identical; mixes differ only in fp32 summation order.
DSV41_MHC_FUSED=0 = stock, DSV41_MHC_FUSED_MAX = largest token count (default 128), DSV41_MHC_SIDE=0 = no side stream.
usage: patch_mhc_fused.py <site-packages>"""
import os
import shutil
import sys

sp = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "dsv41_mhc_fused.py"), f"{sp}/vllm/model_executor/kernels/mhc/dsv41_mhc_fused.py")
p = f"{sp}/vllm/models/deepseek_v4_1/nvidia/model.py"
s = open(p).read()
if "_dsv41_mhc_mode" in s:
    print("already patched"); sys.exit(0)


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:120])
    s = s.replace(old, new)


def call_block(anchor, indent):
    """Return the full call text starting at anchor up to its closing paren line."""
    i = s.index(anchor)
    j = s.index("\n" + " " * indent + ")\n", i) + indent + 3
    return s[i:j]


rep("from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton\n",
    "from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton\n"
    "from vllm.model_executor.kernels.mhc.dsv41_mhc_fused import (\n"
    "    dsv41_mhc_join as _dsv41_mhc_join,\n"
    "    dsv41_mhc_post_pre as _dsv41_mhc_post_pre,\n"
    ")\n"
    "import os as _dsv41_os\n"
    "_DSV41_MHC_FUSED = _dsv41_os.environ.get(\"DSV41_MHC_FUSED\", \"1\") != \"0\"\n"
    "_DSV41_MHC_FUSED_MAX = int(_dsv41_os.environ.get(\"DSV41_MHC_FUSED_MAX\", \"128\"))\n\n\n"
    "def _dsv41_mhc_mode(residual, pre_mix, engram_between):\n"
    "    \"\"\"0 = upstream, 1 = fused post + pre, 2 = fused pre only (post already applied).\"\"\"\n"
    "    if (not _DSV41_MHC_FUSED or pre_mix is None or residual.dim() != 3\n"
    "            or not 0 < residual.shape[0] <= _DSV41_MHC_FUSED_MAX or not residual.is_contiguous()\n"
    "            or residual.shape[1] != 4 or not residual.is_cuda):\n"
    "        return 0\n"
    "    return 2 if engram_between else 1\n")

# Boundary 1: previous FFN output -> attention input (optional Engram in between).
rep("""        else:
            residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
            if self.engram is not None and engram_hashes is not None:
""", """        else:
            _dsv41_mode = _dsv41_mhc_mode(
                residual, pre_mix, self.engram is not None and engram_hashes is not None
            )
            if _dsv41_mode != 1:
                residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
            if self.engram is not None and engram_hashes is not None:
""")
anchor = """            post_mix, res_mix, x, attn_pre = mhc_pre_delayed_tilelang(
                residual,
                self.hc_attn_fn,"""
old = call_block(anchor, 12)
assert "pre_mix=pre_mix" in old and "attn_norm" in old
new = """            if _dsv41_mode:
                residual, post_mix, res_mix, x, attn_pre = _dsv41_mhc_post_pre(
                    x if _dsv41_mode == 1 else None, residual, post_mix, res_mix, pre_mix,
                    self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base, self.rms_norm_eps,
                    self.hc_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters,
                    self.attn_norm.weight, self.attn_norm.variance_epsilon, do_post=_dsv41_mode == 1,
                )
            else:
""" + "\n".join(("    " + l) if l else l for l in old.split("\n"))
rep(old, new)

# Boundary 2: attention output -> FFN input.
anchor2 = """        residual = mhc_post_tilelang(x, residual, post_mix, res_mix)
        post_mix, res_mix, x, ffn_pre = mhc_pre_delayed_tilelang("""
i = s.index(anchor2)
j = s.index("\n        )\n", i) + len("\n        )\n")
old2 = s[i:j]
assert "pre_mix=attn_pre" in old2 and "ffn_norm" in old2
new2 = """        _dsv41_mhc_join()
        if _dsv41_mhc_mode(residual, attn_pre, False) == 1:
            residual, post_mix, res_mix, x, ffn_pre = _dsv41_mhc_post_pre(
                x, residual, post_mix, res_mix, attn_pre,
                self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base, self.rms_norm_eps,
                self.hc_eps, self.hc_eps, self.hc_post_alpha, self.hc_sinkhorn_iters,
                self.ffn_norm.weight, self.ffn_norm.variance_epsilon,
            )
        else:
""" + "\n".join(("    " + l) if l else l for l in old2.split("\n")) 
rep(old2, new2.rstrip(" ") )

# Join the FFN-boundary sinkhorn before handing post/comb/pre to the next layer.
rep("""        x = self.ffn(x, input_ids)
        if _dsv41_gdump_dir() and 0 <= _gi < 3:
            _dsv41_gdump_tensor(f"F{_gi:02d}out", x)
        return x, residual, post_mix, res_mix, ffn_pre
""", """        x = self.ffn(x, input_ids)
        if _dsv41_gdump_dir() and 0 <= _gi < 3:
            _dsv41_gdump_tensor(f"F{_gi:02d}out", x)
        _dsv41_mhc_join()
        return x, residual, post_mix, res_mix, ffn_pre
""")
open(p, "w").write(s)
print("patched", p)
