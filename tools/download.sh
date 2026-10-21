#!/usr/bin/env bash
# Download everything the hybrid checkpoint needs into $MODELS_DIR (about 400 GiB in total), then assemble it:
#   1. LibertAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw (MIT): the routed experts, layers 0-39 (~196 GiB, every shard:
#      experts are spread across all of them)
#   2. deepseek-ai/DeepSeek-V4.1-Flash-1017 (MIT): only the non-expert tensors, by HTTP range requests (~17.5 GiB)
#   3. the same official repo's Engram tables: shards 47 and 48 with the index and config (~189 GiB); they are read
#      from disk at run time, so keep them on fast NVMe
#   4. Curtis White's quantization_config.json (the hybrid's EXL3/FP8 split), from his repository at a fixed commit
# Resumable: rerun after an interruption. PARALLEL (default 4) sets concurrent downloads; HF_TOKEN is optional.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
M=${MODELS_DIR:?set MODELS_DIR in .env}
P=${PARALLEL:-4}
LIB_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
OFF_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
QCFG_URL=https://raw.githubusercontent.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX/b8b9e993e015f35c22ab7202b4ef3c05f8b663f0/quantization_config.json
LIB=$M/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
ENG=$M/DeepSeek-V4.1-Flash-1017-engram
OFF=$M/DeepSeek-V4.1-Flash-1017-official-nonexpert
mkdir -p "$LIB" "$ENG" "$OFF"
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
export AUTH_HDR="${HF_TOKEN:+Authorization: Bearer $HF_TOKEN}"

fetch() {  # fetch REPO FILE DEST_DIR: resumable, retried
  local repo=$1 f=$2 dst=$3
  mkdir -p "$dst/$(dirname "$f")"
  for _ in 1 2 3 4 5; do
    if curl -sfL -C - --retry 3 ${AUTH_HDR:+-H "$AUTH_HDR"} -o "$dst/$f" "https://huggingface.co/$repo/resolve/main/$f"; then
      echo "ok   $repo/$f"; return 0
    fi
    sleep 5
  done
  echo "FAIL $repo/$f"; return 1
}
export -f fetch

echo "== 1/4 LibertAI EXL3 pack -> $LIB"
curl -sfL "${AUTH[@]}" "https://huggingface.co/api/models/$LIB_REPO/tree/main?recursive=true" |
  python3 -c "import json, sys; [print(e['path']) for e in json.load(sys.stdin) if e['type'] == 'file']" > "$LIB/.files"
# big shards first so the tail is short
{ grep safetensors "$LIB/.files" || true; grep -v safetensors "$LIB/.files" || true; } |
  xargs -P "$P" -I{} bash -c 'fetch "$0" "$1" "$2"' "$LIB_REPO" {} "$LIB"

echo "== 2/4 official non-expert tensors -> $OFF"
python3 tools/fetch_official_nonexpert.py "$OFF"

echo "== 3/4 official Engram tables -> $ENG"
printf '%s\n' model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json |
  xargs -P 2 -I{} bash -c 'fetch "$0" "$1" "$2"' "$OFF_REPO" {} "$ENG"

echo "== 4/4 hybrid view"
curl -sfL -o "$M/quantization_config.json" "$QCFG_URL"
python3 tools/make_hybrid.py "$M" "$M/quantization_config.json"
echo "done: $M/DeepSeek-V4.1-Flash-1017-hybrid"
