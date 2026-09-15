#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REFEDIT_PYTHON_BIN:-${REPO_ROOT}/.venv-refedit-vllm/bin/python}"
SOURCE_ROOT="${REFEDIT_SOURCE_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit}"
QUALITY_ROOT="${REFEDIT_QUALITY_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-quality-prefilter-qwen38}"
OUTPUT_ROOT="${REFEDIT_OUTPUT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/RefEdit-mask-prefiltered-qwen38}"
GROUNDING_ROOT="${OUTPUT_ROOT}/grounding"
MASK_ROOT="${OUTPUT_ROOT}/masks"
FINAL_ROOT="${OUTPUT_ROOT}/final"
AUDIT_ROOT="${OUTPUT_ROOT}/audit"
LOG_ROOT="${OUTPUT_ROOT}/logs"
MAIN_LOG="${LOG_ROOT}/full_labeling.log"
GROUNDING_LOG="${LOG_ROOT}/grounding.log"
MASK_LOG="${LOG_ROOT}/mask.log"

mkdir -p "${GROUNDING_ROOT}" "${MASK_ROOT}" "${FINAL_ROOT}" "${AUDIT_ROOT}" "${LOG_ROOT}"
cd "${REPO_ROOT}"

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] RefEdit filtered mask labeling started" | tee -a "${MAIN_LOG}"
echo "source=${SOURCE_ROOT}" | tee -a "${MAIN_LOG}"
echo "prefilter=${QUALITY_ROOT}" | tee -a "${MAIN_LOG}"
echo "output=${OUTPUT_ROOT}" | tee -a "${MAIN_LOG}"

"${PYTHON_BIN}" scripts/validate_refedit_quality_prefilter.py \
  --input-dir "${SOURCE_ROOT}" \
  --quality-dir "${QUALITY_ROOT}" \
  --report-json "${QUALITY_ROOT}/audit/validation_report.json" \
  2>&1 | tee -a "${MAIN_LOG}"
prefilter_validation_status=${PIPESTATUS[0]}
if [[ ${prefilter_validation_status} -ne 0 ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] prefilter validation failed status=${prefilter_validation_status}" | tee -a "${MAIN_LOG}"
  exit "${prefilter_validation_status}"
fi

"${PYTHON_BIN}" -u refedit_mllm_grounding.py \
  --input-dir "${SOURCE_ROOT}" \
  --prefilter-manifest-dir "${QUALITY_ROOT}/manifest" \
  --output-dir "${GROUNDING_ROOT}" \
  --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 2 \
  --batch-size 8 \
  --request-batch-size 4 \
  --progress-mininterval 2 \
  2>&1 | tee -a "${GROUNDING_LOG}" "${MAIN_LOG}"
grounding_status=${PIPESTATUS[0]}
if [[ ${grounding_status} -ne 0 ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] grounding failed status=${grounding_status}" | tee -a "${MAIN_LOG}"
  exit "${grounding_status}"
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] grounding complete; starting SAM3" | tee -a "${MAIN_LOG}"
"${PYTHON_BIN}" -u refedit_grounded_mask_runner.py \
  --input-dir "${SOURCE_ROOT}" \
  --grounding-dir "${GROUNDING_ROOT}" \
  --output-dir "${MASK_ROOT}" \
  --devices 0,1,2,3,4,5,6,7 \
  --progress-mininterval 2 \
  2>&1 | tee -a "${MASK_LOG}" "${MAIN_LOG}"
mask_status=${PIPESTATUS[0]}
if [[ ${mask_status} -ne 0 ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] mask stage failed status=${mask_status}" | tee -a "${MAIN_LOG}"
  exit "${mask_status}"
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] mask complete; validating selected run" | tee -a "${MAIN_LOG}"
"${PYTHON_BIN}" scripts/validate_refedit_masks.py \
  --input-dir "${SOURCE_ROOT}" \
  --selection-manifest-dir "${QUALITY_ROOT}/manifest" \
  --grounding-dir "${GROUNDING_ROOT}" \
  --mask-dir "${MASK_ROOT}" \
  --report-json "${AUDIT_ROOT}/validation_report.json" \
  2>&1 | tee -a "${MAIN_LOG}"
validation_status=${PIPESTATUS[0]}
if [[ ${validation_status} -ne 0 ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] validation failed status=${validation_status}" | tee -a "${MAIN_LOG}"
  exit "${validation_status}"
fi

"${PYTHON_BIN}" scripts/build_refedit_final_mask_dataset.py \
  --input-dir "${SOURCE_ROOT}" \
  --prefilter-manifest-dir "${QUALITY_ROOT}/manifest" \
  --grounding-dir "${GROUNDING_ROOT}" \
  --mask-dir "${MASK_ROOT}" \
  --output-dir "${FINAL_ROOT}" \
  2>&1 | tee -a "${MAIN_LOG}"
final_status=${PIPESTATUS[0]}
if [[ ${final_status} -ne 0 ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] finalization failed status=${final_status}" | tee -a "${MAIN_LOG}"
  exit "${final_status}"
fi

"${PYTHON_BIN}" scripts/validate_refedit_final_mask_dataset.py \
  --input-dir "${SOURCE_ROOT}" \
  --final-dir "${FINAL_ROOT}" \
  --report-json "${FINAL_ROOT}/audit/validation_report.json" \
  2>&1 | tee -a "${MAIN_LOG}"
final_validation_status=${PIPESTATUS[0]}
if [[ ${final_validation_status} -ne 0 ]]; then
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] final validation failed status=${final_validation_status}" | tee -a "${MAIN_LOG}"
  exit "${final_validation_status}"
fi

echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] RefEdit filtered mask labeling complete" | tee -a "${MAIN_LOG}"
