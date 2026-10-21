"""Assemble the hybrid checkpoint (Curtis White's 1017 recipe) as a directory of relative symlinks, no tensor copies:
LibertAI EXL3 2.9 bpw routed experts (layers 0-39) + official 1017 FP8 everything else (the fetched slim files).
config.json = the official 1017 config with quantization_config replaced by Curtis's quantization_config.json.
serve.sh sets DSV41_EXPERTS_ONLY_FILES_RE so the non-expert tensors inside the LibertAI shards are skipped.
Usage: make_hybrid.py MODELS_DIR CURTIS_QCFG [EXPERT_PACK_DIR] [OUT_NAME]
(an absolute EXPERT_PACK_DIR is linked by absolute path; mount it at the same path in the container)"""
import json
import os
import re
import sys

M, QCFG = sys.argv[1], sys.argv[2]
LIB = sys.argv[3] if len(sys.argv) > 3 else "DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw"
OFF = "DeepSeek-V4.1-Flash-1017-official-nonexpert"
OUT = os.path.join(M, sys.argv[4] if len(sys.argv) > 4 else "DeepSeek-V4.1-Flash-1017-hybrid")
os.makedirs(OUT, exist_ok=True)
expert = re.compile(r"^layers\.\d+\.ffn\.experts\.")

lib_map = json.load(open(os.path.join(M, LIB, "model.safetensors.index.json")))["weight_map"]
off_map = json.load(open(os.path.join(M, OFF, "model.safetensors.index.json")))["weight_map"]
wm = {k: v for k, v in lib_map.items() if expert.match(k)}
dup = set(wm) & set(off_map)
assert not dup, f"official files also carry routed experts: {sorted(dup)[:3]}"
wm.update(off_map)


def link(src_dir, name):
    dst = os.path.join(OUT, name)
    if os.path.lexists(dst):
        os.unlink(dst)
    os.symlink(os.path.join(src_dir, name) if os.path.isabs(src_dir) else os.path.join("..", src_dir, name), dst)


for f in sorted(set(v for k, v in lib_map.items() if expert.match(k))):
    link(LIB, f)
for f in sorted(set(off_map.values())):
    link(OFF, f)
# tokenizer and prompt-format files from the LibertAI pack (copied from the base repo there)
skip = {"config.json", "model.safetensors.index.json", "quantization_config.json", "exl3_k_map.json", "README.md"}
for f in os.listdir(os.path.join(M, LIB)):
    if f.endswith(".safetensors") or f in skip or f.startswith("."):
        continue
    link(LIB, f)
cfg = json.load(open(os.path.join(M, OFF, "official-config.json")))
cfg["quantization_config"] = json.load(open(QCFG))
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
json.dump(json.load(open(QCFG)), open(os.path.join(OUT, "quantization_config.json"), "w"), indent=2)
json.dump({"metadata": {"dsv41_hybrid": "curtis-1017: LibertAI routed experts 0-39 + official 1017"},
           "weight_map": wm}, open(os.path.join(OUT, "model.safetensors.index.json"), "w"), indent=1)
print(f"{OUT}: {len(wm)} tensors ({sum(1 for k in wm if expert.match(k))} routed-expert), "
      f"{len(set(wm.values()))} shard links")
