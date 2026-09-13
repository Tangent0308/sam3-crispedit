#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RUNTIME_VENV=${CRISPEDIT_RUNTIME_VENV:-"${REPO_ROOT}/.venv-crispedit-runtime"}
VLLM_VENV=${CRISPEDIT_VLLM_VENV:-"${REPO_ROOT}/.venv-crispedit-vllm"}
RUNTIME_PYTHON=${CRISPEDIT_RUNTIME_PYTHON:-python3.11}
VLLM_PYTHON_VERSION=${CRISPEDIT_VLLM_PYTHON_VERSION:-3.11}
QWEN_VL_PATH=${CRISPEDIT_QWEN_MODEL_PATH:-/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct}
SAM3_PATH=${CRISPEDIT_SAM3_CHECKPOINT_PATH:-}

usage() {
  cat <<EOF
Usage: bash scripts/setup_crispedit_envs.sh

Creates two fresh environments without changing models or datasets:
  runtime: ${RUNTIME_VENV}  (Qwen3-VL prefilter and SAM3 mask)
  vLLM:    ${VLLM_VENV}  (Qwen3.5 grounding)

Environment overrides:
  CRISPEDIT_RUNTIME_VENV
  CRISPEDIT_VLLM_VENV
  CRISPEDIT_RUNTIME_PYTHON
  CRISPEDIT_VLLM_PYTHON_VERSION
  CRISPEDIT_QWEN_MODEL_PATH
  CRISPEDIT_SAM3_CHECKPOINT_PATH
  TORCH_INDEX_URL, TORCH_VERSION, TORCHVISION_VERSION
  TORCH_BACKEND, VLLM_WHEEL_URL
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unexpected argument: $1" >&2
  usage >&2
  exit 1
fi

for target in "${RUNTIME_VENV}" "${VLLM_VENV}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to reuse an existing environment: ${target}" >&2
    echo "Set CRISPEDIT_RUNTIME_VENV/CRISPEDIT_VLLM_VENV to fresh paths." >&2
    exit 1
  fi
done

runtime_args=(
  --python-bin "${RUNTIME_PYTHON}"
  --venv-dir "${RUNTIME_VENV}"
  --qwen-model-path "${QWEN_VL_PATH}"
)
if [[ -n "${SAM3_PATH}" ]]; then
  runtime_args+=(--sam3-checkpoint-path "${SAM3_PATH}")
fi

echo "[crispedit] Installing the Qwen3-VL prefilter/SAM3 runtime"
bash "${SCRIPT_DIR}/setup_env.sh" "${runtime_args[@]}"

echo "[crispedit] Installing the isolated Qwen3.5/vLLM grounding runtime"
PYTHON_VERSION="${VLLM_PYTHON_VERSION}" \
  bash "${SCRIPT_DIR}/setup_vllm_env.sh" "${VLLM_VENV}"

cat <<EOF

[crispedit] Both environments are ready.

Prefilter and SAM3:
  ${RUNTIME_VENV}/bin/python

Qwen3.5/vLLM grounding:
  ${VLLM_VENV}/bin/python

Detailed commands:
  ${REPO_ROOT}/docs/CRISPEDIT_MASK.md
EOF
