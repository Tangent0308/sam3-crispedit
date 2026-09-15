#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REFEDIT_PYTHON_BIN:-${REPO_ROOT}/.venv-refedit-vllm/bin/python}"
SOURCE_ROOT="${REFEDIT_SOURCE_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit}"
QUALITY_ROOT="${REFEDIT_QUALITY_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-quality-prefilter-qwen38}"
OUTPUT_ROOT="${REFEDIT_OUTPUT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38}"

REFEDIT_QUALITY_PYTHON_BIN="${PYTHON_BIN}" \
REFEDIT_QUALITY_INPUT_DIR="${SOURCE_ROOT}" \
REFEDIT_QUALITY_OUTPUT_DIR="${QUALITY_ROOT}" \
  bash "${SCRIPT_DIR}/run_refedit_quality_prefilter_full.sh"

REFEDIT_PYTHON_BIN="${PYTHON_BIN}" \
REFEDIT_SOURCE_ROOT="${SOURCE_ROOT}" \
REFEDIT_QUALITY_ROOT="${QUALITY_ROOT}" \
REFEDIT_OUTPUT_ROOT="${OUTPUT_ROOT}" \
  bash "${SCRIPT_DIR}/run_refedit_filtered_mask_full.sh"
