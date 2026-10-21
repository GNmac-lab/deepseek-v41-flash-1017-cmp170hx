#!/usr/bin/env python3
"""Per-file weight filter for the hybrid checkpoint (Curtis 1017 weight split).

DSV41_EXPERTS_ONLY_FILES_RE (regex on the shard basename): in matching files, load only the main
routed experts (^layers.N.ffn.experts.); everything else in those files is skipped, so the LibertAI
EXL3 shards contribute experts only and attention/head/draft/embed come from the official files.
Extends the dsv41reap-pp overlay's _dsv41_skip(name) in weight_utils.py with the shard path.
usage: patch_skip_file_experts_only.py <vllm/model_executor/model_loader/weight_utils.py>
"""
import sys

path = sys.argv[1]
src = open(path).read()
MARK = "DSV41_EXPERTS_ONLY_FILES_RE"
if MARK in src:
    print("already patched"); sys.exit(0)
old = '''def _dsv41_skip(name: str) -> bool:
    if _DSV41_ONLY_RE is not None and _DSV41_ONLY_RE.search(name) is None:
        return True
'''
new = '''_DSV41_EXPERTS_ONLY_FILES = (
    _dsv41_re.compile(_dsv41_os.environ["DSV41_EXPERTS_ONLY_FILES_RE"])
    if _dsv41_os.environ.get("DSV41_EXPERTS_ONLY_FILES_RE")
    else None
)
_DSV41_MAIN_EXPERT = _dsv41_re.compile(r"^(model\\.)?layers\\.\\d+\\.ffn\\.experts\\.")


def _dsv41_skip(name: str, st_file: str | None = None) -> bool:
    if (_DSV41_EXPERTS_ONLY_FILES is not None and st_file is not None
            and _DSV41_EXPERTS_ONLY_FILES.search(_dsv41_os.path.basename(st_file))
            and not _DSV41_MAIN_EXPERT.search(name)):
        return True
    if _DSV41_ONLY_RE is not None and _DSV41_ONLY_RE.search(name) is None:
        return True
'''
assert src.count(old) == 1, src.count(old)
src = src.replace(old, new, 1)
old_call = "if should_skip_weight(name, local_expert_ids) or _dsv41_skip(name):"
n = src.count(old_call)
assert n >= 1, n
src = src.replace(old_call, "if should_skip_weight(name, local_expert_ids) or _dsv41_skip(name, st_file):")
open(path, "w").write(src)
print(f"patched {path}: {n} iterator call site(s)")
