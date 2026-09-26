#!/usr/bin/env bash
# Run from tmux: bash scripts/run_crispedit_mask_pipeline.sh RUN_DIR [SELECTION_JSON]
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_dir=${1:?Usage: run_crispedit_mask_pipeline.sh RUN_DIR [SELECTION_JSON]}
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
selection_args=()
if [[ -n ${2:-} ]]; then selection_args=(--selection-file "$(realpath "$2")"); fi
dataset_dir=${CRISPEDIT_INPUT_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/source/CrispEdit-2M}
quality_dir=${CRISPEDIT_QUALITY_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/prefilter/quality/manifest}
difficulty_dir=${CRISPEDIT_DIFFICULTY_DIR:-/mnt/bn/strategy-mllm-train/user/tanyue/CrispEdit-labeling/prefilter/scene/manifest}
ground_python=${CRISPEDIT_VLLM_PYTHON:-$repo_dir/.venv-crispedit/bin/python}
mask_python=${CRISPEDIT_SAM_PYTHON:-$repo_dir/.venv-crispedit/bin/python}
devices=${CRISPEDIT_DEVICES:-0,1,2,3,4,5,6,7}
model=${CRISPEDIT_GROUNDING_MODEL_PATH:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}
checkpoint=${CRISPEDIT_SAM3_CHECKPOINT_PATH:-/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt}
cd "$repo_dir"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
"$ground_python" -u crispedit_mllm_grounding.py \
  --input-dir "$dataset_dir" --keep-manifest-dir "$quality_dir" \
  --difficulty-manifest-dir "$difficulty_dir" --output-dir "$run_dir/grounding" \
  --model-path "$model" --devices "$devices" --tensor-parallel-size 2 \
  --inference-backend vllm --grounding-mode two-pass \
  --batch-size "${CRISPEDIT_BATCH_SIZE:-16}" --request-batch-size "${CRISPEDIT_BATCH_SIZE:-16}" \
  --max-images-per-generate 0 --max-new-tokens "${CRISPEDIT_GROUNDING_TOKENS:-1536}" --observation-max-new-tokens "${CRISPEDIT_OBSERVATION_TOKENS:-3072}" \
  --vllm-gpu-memory-utilization 0.85 --vllm-max-model-len 32768 \
  --vllm-max-images-per-prompt 16 --vllm-mm-encoder-tp-mode data \
  --progress-mininterval 5 "${selection_args[@]}" 2>&1 | tee -a "$run_dir/grounding.log"
"$mask_python" -u crispedit_grounded_mask_runner.py \
  --input-dir "$dataset_dir" --grounding-dir "$run_dir/grounding" \
  --output-dir "$run_dir/mask" --checkpoint-path "$checkpoint" \
  --devices "$devices" --preview-dir "$run_dir/previews" \
  --preview-rows-per-shard 4 --progress-mininterval 5 \
  "${selection_args[@]}" 2>&1 | tee -a "$run_dir/mask.log"
"$mask_python" -u scripts/validate_crispedit_mask_pipeline.py \
  --input-dir "$dataset_dir" --quality-dir "$quality_dir" \
  --difficulty-dir "$difficulty_dir" --run-dir "$run_dir" \
  "${selection_args[@]}" 2>&1 | tee -a "$run_dir/validation.log"
printf 'PIPELINE_SUCCESS_UTC=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
