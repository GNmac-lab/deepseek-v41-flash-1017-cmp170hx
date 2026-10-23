#!/usr/bin/env python3
"""dsv41: MXFP8 linears on sm80 (the checkpoint's 32x32 power-of-two FP8 blocks, served as e4m3 + e8m0 per row and
32 k) through one custom op on a single fragment-native weight copy (dsv41_fp8w.py + the dsv41_fp8w_ext extension):
gemv_small for M <= 8, gemm_mid for M <= 64, exact dequant + cuBLAS above. Marlin's repacked copy is not built
(weights over 32 MB keep Marlin only; weights up to 4 MB keep both and use Marlin for M > 8).
DSV41_FP8_GEMV=0 = stock. usage: patch_fp8_gemv_cuda.py <site-packages>"""
import os
import shutil
import sys

sp = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "dsv41_fp8w.py"), f"{sp}/vllm/model_executor/kernels/linear/mxfp8/dsv41_fp8w.py")
p = f"{sp}/vllm/model_executor/kernels/linear/mxfp8/marlin.py"
s = open(p).read()
if "_dsv41_w8" in s:
    print("already patched")
    sys.exit(0)


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep("import torch\n", "import os as _dsv41_os\n\nimport torch\n")
rep("""        prepare_mxfp8_layer_for_marlin(layer)
""", """        # dsv41 (patch_fp8_gemv_cuda.py): one fragment-native copy for the CUDA kernels; Marlin's only where kept
        N, K = layer.output_size_per_partition, layer.input_size_per_partition
        if (_dsv41_os.environ.get("DSV41_FP8_GEMV", "1") != "0" and layer.weight.dtype == torch.float8_e4m3fn
                and tuple(layer.weight.shape) == (N, K) and getattr(layer, "bias", None) is None):
            from vllm.platforms import current_platform

            if current_platform.is_device_capability((8, 0)):
                from . import dsv41_fp8w as _g

                g8 = _g.prepare(layer.weight.data.view(torch.uint8), layer.weight_scale.data.view(torch.uint8))
                if g8 is not None:
                    W8, SC8, cnt8, keep = g8
                    layer._dsv41_w8 = (W8, SC8, cnt8)
                    layer._dsv41_keep_marlin = keep
                    if not keep:
                        # free the e4m3 weight and scales (Marlin's repack is never built for this layer)
                        layer.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8, device=W8.device),
                                                          requires_grad=False)
                        layer.weight_scale = torch.nn.Parameter(torch.empty(0, dtype=torch.uint8, device=W8.device),
                                                                requires_grad=False)
                        layer.workspace = None
                        return
        prepare_mxfp8_layer_for_marlin(layer)
""")
rep("""        return apply_mxfp8_marlin_linear(
""", """        if bias is None and getattr(layer, "_dsv41_w8", None) is not None:
            from . import dsv41_fp8w as _g

            return _g.linear(x, layer)
        return apply_mxfp8_marlin_linear(
""")
open(p, "w").write(s)
print("patched", p)
