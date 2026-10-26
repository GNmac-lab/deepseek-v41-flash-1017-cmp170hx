# dsv41 (runtime variant): prefill-sized tile config for the fused mHC kernel A.
#
# dsv41_mhc_fused picks its kernel-A tiles by token count; every call above 16 tokens uses the decode-tuned "large"
# tiles (20 hidden splits, 8 tokens, 8 outputs, k 32, 2 warps), which read the residual once per 8-output tile (3x
# for the 24 mixes) and the 2 MB fp32 fn once per 8 tokens. Since v81 the fused path also runs prefill slices (up to
# 4,096 rows), so calls of DSV41_MHC_PCFG_MIN (256) rows or more take DSV41_MHC_PCFG (20,16,32,16,4): one 32-wide
# output tile and 16-token tiles. Kernel A + B at 1,024 rows 0.465 -> 0.365 ms; warm prefill +1.0 % (A/B x3).
# fp32 summation order changes (max |diff| 0.002 at bf16 outputs); decode (<= 48 tokens) is untouched.
import os

from vllm.model_executor.kernels.mhc import dsv41_mhc_fused as F

_ORIG = F.__dict__.get("_dsv41_pcfg_orig") or F._config
F._dsv41_pcfg_orig = _ORIG
CFG = tuple(int(v) for v in os.environ.get("DSV41_MHC_PCFG", "20,16,32,16,4").split(","))
MIN = int(os.environ.get("DSV41_MHC_PCFG_MIN", "256"))


def config(num_tokens):
    return CFG if num_tokens >= MIN else _ORIG(num_tokens)


SETATTR = [{"target": "vllm.model_executor.kernels.mhc.dsv41_mhc_fused._config", "source": "dsv41_mhc_pcfg.config"}]
