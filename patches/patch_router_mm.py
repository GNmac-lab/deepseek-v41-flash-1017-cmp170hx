#!/usr/bin/env python3
"""Route GateLinear's sm_80 cuBLAS tier through dsv41_router_mm for M <= 64 (vllm/.../router/gate_linear.py).
DSV41_ROUTER_MM=0 = stock. usage: patch_router_mm.py <gate_linear.py>"""
import sys

p = sys.argv[1]
s = open(p).read()
if "dsv41_router_mm" in s:
    print("already patched"); sys.exit(0)
old = '''        # Tier 4: cuBLAS bf16→fp32
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
'''
assert s.count(old) == 1
new = '''        # dsv41 (ds41-patches/patch_router_mm.py): sm_80 small-batch router GEMM (Triton, fixed-order split-K)
        if self.allow_cublas_router_gemm and _DSV41_ROUTER_MM and x.is_cuda and x.shape[0] <= 64:
            from vllm.model_executor.layers.fused_moe.router import dsv41_router_mm as _drm

            if _drm.supports(x, self.weight):
                return _drm.router_mm(x, self.weight), None
''' + old
s = s.replace(old, new)
s = s.replace("logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n"
              "import os as _os\n_DSV41_ROUTER_MM = _os.environ.get(\"DSV41_ROUTER_MM\", \"1\") != \"0\" and "
              "current_platform.is_device_capability((8, 0))\n", 1)
open(p, "w").write(s)
print("patched", p)
