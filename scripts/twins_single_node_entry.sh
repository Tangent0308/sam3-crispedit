#!/usr/bin/env bash
# Resident Qwen3.8-27B planner/auditor + six independent 40-step editors.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${TWINS_RUN_ROOT:?Set a new twins output directory}"
: "${TWINS_SOURCE_ROOT:=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016}"
: "${GROUP_RUNTIME_ROOT:=/opt/tiger/sa1b-group-production-runtime}"
: "${TWINS_TARGET:=15000}"
: "${GROUP_ARCHIVE_PATH_MAP:=/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/path_map.json}"
[[ "${ARNOLD_WORKER_NUM:-1}" == 1 && "${ARNOLD_WORKER_GPU:-8}" == 8 ]] || { echo 'Requires one worker with eight GPUs' >&2; exit 2; }
[[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo 'Unset CUDA_VISIBLE_DEVICES; entry assigns GPUs 0..7' >&2; exit 2; }
[[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) == 8 ]] || { echo 'Expected eight visible GPUs' >&2; exit 2; }
export GROUP_RUNTIME_ROOT PYTHONUNBUFFERED=1 WANDB_MODE=disabled WANDB_DISABLED=true
export OMP_NUM_THREADS=4 CUDA_MODULE_LOADING=LAZY SA1B_MM_ENCODER_ATTN_BACKEND=TORCH_SDPA DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TWINS_RUN_ROOT/logs"
exec > >(tee -a "$TWINS_RUN_ROOT/logs/single_node.log") 2>&1
cd "$repo"
if [[ "${GROUP_SKIP_SETUP:-0}" != 1 ]]; then bash scripts/group_production/setup_runtime.sh; fi
mllm="${TWINS_MLLM_PYTHON:-$GROUP_RUNTIME_ROOT/mllm/bin/python}"
editor="${TWINS_EDITOR_PYTHON:-$GROUP_RUNTIME_ROOT/editor/bin/python}"
for runtime in "$mllm" "$editor"; do
  [[ -x "$runtime" ]] || { echo "Missing runtime: $runtime" >&2; exit 2; }
  "$runtime" -m sa1b_pipeline.fonts --render
done
exec "$mllm" scripts/run_twins_single_node.py \
  --root "$TWINS_RUN_ROOT" --source-root "$TWINS_SOURCE_ROOT" --target "$TWINS_TARGET" \
  --archive-path-map "$GROUP_ARCHIVE_PATH_MAP" \
  --mllm-python "$mllm" --editor-python "$editor" "$@"
