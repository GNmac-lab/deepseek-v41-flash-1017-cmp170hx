#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Wire CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE exl3_tc_hook into the vcruz305 vllm-exl3 plugin.

try_build(layer, inners) after each build_exl3_fused_state call; try_apply(...) at the top of
apply_exl3_fused_moe (it returns None for rows that stay on coop/fused, or when the flags are off).
The hook module is loaded from VLLM_EXL3_TC_HOOK (default /opt/dsv41-tc-moe/exl3_tc_hook.py) only
when VLLM_EXL3_TC_MOE_V2=1 or VLLM_EXL3_TC_MOE_PREFILL=1.
usage: patch_tc_hook.py <vllm_exl3/exl3.py>
"""
import re
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "[ds41-tc-moe]"
if MARK in src:
    print("already patched"); sys.exit(0)

loader = '''

# [ds41-tc-moe] tensor-core EXL3 MoE hook (CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE)
_DS41_TC_HOOK = None


def _ds41_tc_hook():
    global _DS41_TC_HOOK
    if _DS41_TC_HOOK is None:
        import importlib.util as _ilu
        hook = os.environ.get("VLLM_EXL3_TC_HOOK", "/opt/dsv41-tc-moe/exl3_tc_hook.py")
        on = os.environ.get("VLLM_EXL3_TC_MOE_V2") == "1" or os.environ.get("VLLM_EXL3_TC_MOE_PREFILL") == "1"
        if on and os.path.exists(hook):
            spec = _ilu.spec_from_file_location("exl3_tc_hook", hook)
            mod = _ilu.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _DS41_TC_HOOK = mod
        else:
            _DS41_TC_HOOK = False
    return _DS41_TC_HOOK or None
'''
src = src.rstrip("\n") + "\n" + loader

# try_build after every build_exl3_fused_state(layer, inners) call site
n_build = 0


def _build(m):
    global n_build
    n_build += 1
    ind = m.group(1)
    return (f"{ind}build_exl3_fused_state(layer, inners)\n"
            f"{ind}if _ds41_tc_hook() is not None:  # {MARK}\n"
            f"{ind}    _ds41_tc_hook().try_build(layer, inners)\n")


src = re.sub(r"^([ \t]+)build_exl3_fused_state\(layer, inners\)\n", _build, src, flags=re.M)
assert n_build >= 1, "no build_exl3_fused_state call site"

old = '''    """One exl3_moe launch per layer, with fat-expert fallback. Supports mixed-K via exl3_moe_mixedk."""
'''
new = old + f'''    _tc = _ds41_tc_hook()  # {MARK}
    if _tc is not None:
        _tc_out = _tc.try_apply(x2d, ids, weights, layer, expert_map, limit)
        if _tc_out is not None:
            return _tc_out
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)
open(path, "w").write(src)
print(f"patched {path}: try_build at {n_build} call site(s), try_apply in apply_exl3_fused_moe")
