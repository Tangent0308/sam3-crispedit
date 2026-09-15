#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${REFEDIT_QUALITY_PYTHON_BIN:-${REPO_ROOT}/.venv-refedit-vllm/bin/python}"
INPUT_DIR="${REFEDIT_QUALITY_INPUT_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit}"
OUTPUT_DIR="${REFEDIT_QUALITY_OUTPUT_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-quality-prefilter-qwen38}"
MODEL_PATH="${REFEDIT_QUALITY_MODEL_PATH:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}"

mkdir -p "${OUTPUT_DIR}/logs"

exec "${PYTHON_BIN}" -u "${REPO_ROOT}/refedit_quality_prefilter.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 1 \
  --batch-size 4 \
  --vllm-max-num-seqs 4 \
  --vllm-max-model-len 8192 \
  --max-new-tokens 1024 \
  --parse-retries 1 \
  --vllm-gpu-memory-utilization 0.85 \
  --progress-mininterval 5
