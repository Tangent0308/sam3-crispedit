#!/usr/bin/env bash
set -euo pipefail

# Pilot/full run entry point. Keep the SA-1B handoff root and work root
# explicit so a missing mount fails before a multi-GPU model load.
SA1B_ROOT=${SA1B_ROOT:-/mnt/hdfs/byte_ttlive_strategy_llm/user/haobo.yuan/datasets}
WORK_ROOT=${WORK_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work}
MODEL=${MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}
IMAGE_ROOT="$SA1B_ROOT/sa1b_tfd_1k"
DENSE_ROOT="$SA1B_ROOT/sa1b_dense_label/annotation"

[[ -d "$IMAGE_ROOT" && -d "$DENSE_ROOT" ]] || {
  echo "SA-1B handoff roots are unavailable: $IMAGE_ROOT / $DENSE_ROOT" >&2
  exit 2
}
[[ -f ids.txt ]] || { echo "create ids.txt with a pilot/sample of SA IDs" >&2; exit 2; }

mkdir -p "$WORK_ROOT"
python -m sa1b_pipeline.normalize --image-records-dir "$IMAGE_ROOT" --dense-records-dir "$DENSE_ROOT" --id-file ids.txt --output-dir "$WORK_ROOT/candidates"
python -m sa1b_pipeline.select --manifest "$WORK_ROOT/candidates/candidates.jsonl" --output "$WORK_ROOT/selection/selected_cases.jsonl" --model-id "$MODEL" --tensor-parallel-size "${TP_SIZE:-8}"
python -m sa1b_pipeline.editor --manifest "$WORK_ROOT/selection/selected_cases.jsonl" --output-dir "$WORK_ROOT/generated" --model-id "${QWEN_IMAGE_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1}" --steps "${STEPS:-40}"
python -m sa1b_pipeline.validate --manifest "$WORK_ROOT/generated/generated.jsonl" --output "$WORK_ROOT/validation/validation.jsonl" --model-id "$MODEL"
