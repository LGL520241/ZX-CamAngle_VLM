#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/xiaozheng/qwen3/local_qwen3_vl_model/cpatonn-mirror/Qwen3-VL-4B-Instruct-AWQ-4bit}"

docker compose -f "${SCRIPT_DIR}/qwen-vllm.compose.yml" up -d
docker compose -f "${SCRIPT_DIR}/qwen-vllm.compose.yml" ps
