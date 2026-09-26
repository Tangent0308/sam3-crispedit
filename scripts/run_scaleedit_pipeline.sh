#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_dir=${1:?Usage: bash scripts/run_scaleedit_pipeline.sh OUTPUT_DIR [SELECTION_JSON]}
python_bin=${SCALEEDIT_PYTHON:-.venv-scaleedit-current/bin/python}
source_dir=${SCALEEDIT_SOURCE:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-source}
devices=${SCALEEDIT_DEVICES:-0,1,2,3,4,5,6,7}
filter_run=${SCALEEDIT_FILTER_RUN:-$run_dir}
mkdir -p "$run_dir/logs"
selection=()
if [[ -n ${2:-} ]]; then selection=(--selection-file "$2"); fi
common=(--input-dir "$source_dir" --devices "$devices" --batch-size 4 --vllm-max-num-seqs 4 --vllm-enforce-eager "${selection[@]}")
for stage in quality scene grounding mask; do
  if [[ -n ${SCALEEDIT_FILTER_RUN:-} && ( $stage == quality || $stage == scene ) ]]; then continue; fi
  upstream=()
  if [[ $stage != quality ]]; then upstream+=(--quality-dir "$filter_run/quality"); fi
  if [[ $stage == grounding || $stage == mask ]]; then upstream+=(--scene-dir "$filter_run/scene"); fi
  if [[ $stage == mask ]]; then upstream+=(--grounding-dir "$run_dir/grounding"); fi
  tp=1
  if [[ $stage == grounding ]]; then tp=${SCALEEDIT_GROUND_TP:-2}; fi
  "$python_bin" -u scripts/run_scaleedit_pipeline.py --stage "$stage" \
    --output-dir "$run_dir/$stage" --tensor-parallel-size "$tp" \
    "${common[@]}" "${upstream[@]}" 2>&1 | tee -a "$run_dir/logs/$stage.log"
done
