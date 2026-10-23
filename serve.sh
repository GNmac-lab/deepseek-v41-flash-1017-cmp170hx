#!/usr/bin/env bash
# Start the engine container with the settings in .env (see .env.example). Prefer ./start.sh, which also waits for
# the server and can run the smoke test.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] && set -a && . ./.env && set +a

LAYOUT=${LAYOUT:-tp4}
IMAGE=${IMAGE:-ds41-cmp170hx:latest}
NAME=${NAME:-ds41-flash}
MODELS_DIR=${MODELS_DIR:?set MODELS_DIR in .env}
BIND=${BIND:-127.0.0.1}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
P2P=${P2P:-1}
CACHE_DIR=${CACHE_DIR:-$PWD/cache}
SPEC_TOKENS=${SPEC_TOKENS:-5}

MODEL=/models/DeepSeek-V4.1-Flash-1017-hybrid
ENGRAM=/models/DeepSeek-V4.1-Flash-1017-engram
for d in DeepSeek-V4.1-Flash-1017-hybrid DeepSeek-V4.1-Flash-1017-engram; do
  [ -d "$MODELS_DIR/$d" ] || { echo "missing $MODELS_DIR/$d (run tools/download.sh, then tools/make_hybrid.py)"; exit 1; }
done
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then docker stop -t 120 "$NAME" > /dev/null; fi
docker rm "$NAME" > /dev/null 2>&1 || true
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "$used" -gt 2000 ] && { echo "GPUs are busy ($used MiB in use): stop the other GPU process first"; exit 1; }

if [ "$P2P" = 1 ]; then P2P_ENV=(-e NCCL_P2P_LEVEL=SYS); else P2P_ENV=(-e NCCL_P2P_DISABLE=1 -e NCCL_IGNORE_DISABLED_P2P=1); fi
mkdir -p "$CACHE_DIR/root-cache" "$CACHE_DIR/triton" "$CACHE_DIR/tilelang"
# Each worker on its GPU's NUMA node when the host has several (NUMA_BIND=0 turns it off)
NUMA=(); [ "${NUMA_BIND:-1}" = 1 ] && [ "$(ls -d /sys/devices/system/node/node[0-9]* | wc -l)" -gt 1 ] && NUMA=(--numa-bind)

COMMON_ENV=(
  -e VLLM_PLUGINS=vllm_exl3 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXL3_MODEL_DIR="$MODEL"
  -e VLLM_USE_V2_MODEL_RUNNER=0 -e DSV41_DSPARK_V1=1 -e VLLM_EXL3_COOP=1
  -e VLLM_EXL3_TC_MOE_V2=1 -e VLLM_EXL3_TC_MOE_PREFILL=1 -e VLLM_EXL3_TC_PREFILL_ROWS=2048
  -e VLLM_EXL3_TC_MOE_SO=/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so -e VLLM_EXL3_TC_HOOK=/opt/dsv41-tc-moe/exl3_tc_hook.py
  -e VLLM_EXL3_MOE_TP_ALIGN=128 -e VLLM_EXL3_MOE_TP_ROTATE=1
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 "${P2P_ENV[@]}"
  -e DSV41_ENGRAM_STORAGE=ssd -e DSV41_ENGRAM_DIR="$ENGRAM"
  -e DSV41_EXPERTS_ONLY_FILES_RE='^model-[0-9]+-of-000(43|39)\.safetensors$'
  -e TORCH_CUDA_ARCH_LIST=8.0 -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/inductor
  -e VLLM_DSV41_DECODE_SPLITK=1 -e DSV41_SHARED_HEAVY_BLOCKS=1 -e DSV41_DRAFT_HEAD_FP8=1
  -e DSV41_PREFILL_CHUNK_REQS=1 -e DSV41_INDEXER_BUF_MULT=4
)
# Optional engine knobs, passed only when set in .env: DSV41_VARIANTS (runtime variant list; empty = none),
# DSV41_GATE_TAU (confidence-gated verification threshold; 0 = off)
OPT_ENV=()
for v in DSV41_VARIANTS DSV41_GATE_TAU; do
  if [ -n "${!v+x}" ]; then OPT_ENV+=(-e "$v=${!v}"); fi
done
case "$LAYOUT" in
  tp4)
    PAR=(--tensor-parallel-size 4 --pipeline-parallel-size 1)
    LAYOUT_ENV=(-e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1)
    GRAPHS='{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1,2,3,4,5,6,8,12,18,24,30,36,42,48]}'
    SCHED=(--async-scheduling)
    ;;
  pp4)
    PAR=(--tensor-parallel-size 1 --pipeline-parallel-size 4)
    LAYOUT_ENV=(-e VLLM_PP_LAYER_PARTITION="${PP_PARTITION:-10,11,11,8}")
    GRAPHS='{"cudagraph_mode": "PIECEWISE", "cudagraph_capture_sizes": [1,2,3,4,5,6,8,12,18,24,30,36,42,48]}'
    SCHED=()
    ;;
  *) echo "LAYOUT must be tp4 or pp4"; exit 1 ;;
esac

# The API has no authentication. It is published on $BIND only (127.0.0.1 by default): reach it from another
# machine with an SSH local forward (ssh -N -L 127.0.0.1:$PORT:127.0.0.1:$PORT user@this-host), not a wider bind.
docker run -d --name "$NAME" --gpus all --ipc=host --cap-add SYS_NICE \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODELS_DIR:/models:ro" \
  -v "$CACHE_DIR/root-cache:/root/.cache" -v "$CACHE_DIR/triton:/root/.triton" -v "$CACHE_DIR/tilelang:/root/.tilelang" \
  -p "$BIND:$PORT:$PORT" \
  "${COMMON_ENV[@]}" "${LAYOUT_ENV[@]}" "${OPT_ENV[@]}" \
  --entrypoint vllm "$IMAGE" serve "$MODEL" \
  --quantization exl3 --load-format safetensors --safetensors-load-strategy lazy \
  "${PAR[@]}" \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 2048 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype fp8_ds_mla --enable-prefix-caching \
  --compilation-config "$GRAPHS" "${SCHED[@]}" \
  --speculative-config "{\"method\": \"dspark\", \"num_speculative_tokens\": $SPEC_TOKENS}" \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 \
  "${NUMA[@]}" \
  ${EXTRA_ARGS:-} > /dev/null
echo "started $NAME ($LAYOUT) on $BIND:$PORT; logs: docker logs -f $NAME"
