#!/usr/bin/env python3
"""dsv41: install vllm/dsv41_variants.py + the variant modules, and apply them in Worker.compile_or_warm_up_model
before graph capture. usage: patch_variants.py <site-packages/vllm> <dir with dsv41_variants.py and variant .py files>"""
import os
import shutil
import sys

vllm, src = sys.argv[1], sys.argv[2]
shutil.copy(os.path.join(src, "dsv41_variants.py"), os.path.join(vllm, "dsv41_variants.py"))
d = os.path.join(vllm, "dsv41_variants_d")
os.makedirs(d, exist_ok=True)
for f in sorted(f for f in os.listdir(os.path.join(src, "variants")) if f.endswith(".py")):
    shutil.copy(os.path.join(src, "variants", f), os.path.join(d, f))
    print("variant", f)
p = os.path.join(vllm, "v1/worker/gpu_worker.py")
s = open(p).read()
M = "patch_variants.py"
if M in s:
    print("already patched", p)
    sys.exit(0)
s += '''

# dsv41 (patches/dsv41_variants/patch_variants.py): runtime variants go in before any graph capture
def _dsv41_variants_first(fn):
    def w(self, *a, **k):
        import vllm.dsv41_variants
        vllm.dsv41_variants.apply()
        return fn(self, *a, **k)
    return w


Worker.compile_or_warm_up_model = _dsv41_variants_first(Worker.compile_or_warm_up_model)
'''
open(p, "w").write(s)
print("patched", p)
