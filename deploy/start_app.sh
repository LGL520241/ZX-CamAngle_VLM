#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xiaozheng/miniconda3/envs/CamAngle_VLM/bin/python}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-/mnt/nas/logs/CamAngle_VLM}"
export VLM_ENABLED="${VLM_ENABLED:-true}"
export VLM_BASE_URL="${VLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export VLM_MODEL="${VLM_MODEL:-Qwen3-VL-4B-Instruct-AWQ-4bit}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m uvicorn cam_angle.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-20350}"
