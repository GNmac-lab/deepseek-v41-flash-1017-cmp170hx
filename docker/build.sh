#!/usr/bin/env bash
# Build the engine image from the official vLLM DeepSeek-V4.1 image and the pinned upstream sources
# (~30-45 min, mostly compiling CUDA extensions for sm_80). Usage: docker/build.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-${IMAGE:-ds41-cmp170hx:latest}}
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t "$TAG" .
echo "built $TAG"
