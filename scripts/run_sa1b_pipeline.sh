#!/usr/bin/env bash
set -euo pipefail

# Pilot/full run entry point. Keep the SA-1B handoff root and work root
# explicit so a missing mount fails before a multi-GPU model load.
SA1B_ROOT=${SA1B_ROOT:-/mnt/hdfs/byte_ttlive_strategy_llm/user/haobo.yuan/datasets}
WORK_ROOT=${WORK_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_edit_work}
PREP_PYTHON=${PREP_PYTHON:-python}
MLLM_PYTHON=${MLLM_PYTHON:-python}
EDITOR_PYTHON=${EDITOR_PYTHON:-python}
EDITOR_RUNTIME_ROOT=${EDITOR_RUNTIME_ROOT:-}
export DIFFUSION_ATTENTION_BACKEND=${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}
if [[ -n "$EDITOR_RUNTIME_ROOT" ]]; then
  export LD_LIBRARY_PATH="$EDITOR_RUNTIME_ROOT/lib/python3.12/site-packages/nvidia/cu13/lib:$EDITOR_RUNTIME_ROOT/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
fi
MODEL=${MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}
IMAGE_ROOT="$SA1B_ROOT/sa1b_tfd_1k"
DENSE_ROOT="$SA1B_ROOT/sa1b_dense_label/annotation"

# Set HDFS_IMAGE_URI/HDFS_DENSE_URI for a bounded remote pilot when the SG2
# FUSE mount is not present. This mode is deliberately shard-scoped and
# bounded by SCAN_RECORDS/MINIMUM_MATCHES.
HDFS_IMAGE_URI=${HDFS_IMAGE_URI:-}
HDFS_DENSE_URI=${HDFS_DENSE_URI:-}
SCAN_RECORDS=${SCAN_RECORDS:-5000}
MINIMUM_MATCHES=${MINIMUM_MATCHES:-100}

if [[ -n "$HDFS_IMAGE_URI" || -n "$HDFS_DENSE_URI" ]]; then
  [[ -n "$HDFS_IMAGE_URI" && -n "$HDFS_DENSE_URI" ]] || {
    echo "set both HDFS_IMAGE_URI and HDFS_DENSE_URI for remote mode" >&2
    exit 2
  }
  NORMALIZE_ARGS=(--hdfs-image-uri "$HDFS_IMAGE_URI" --hdfs-dense-uri "$HDFS_DENSE_URI"
    --scan-records "$SCAN_RECORDS" --minimum-matches "$MINIMUM_MATCHES")
else
  [[ -d "$IMAGE_ROOT" && -d "$DENSE_ROOT" ]] || {
    echo "SA-1B handoff roots are unavailable: $IMAGE_ROOT / $DENSE_ROOT" >&2
    exit 2
  }
  [[ -f ids.txt ]] || { echo "create ids.txt with a pilot/sample of SA IDs" >&2; exit 2; }
  NORMALIZE_ARGS=(--image-records-dir "$IMAGE_ROOT" --dense-records-dir "$DENSE_ROOT" --id-file ids.txt)
fi

mkdir -p "$WORK_ROOT"
"$PREP_PYTHON" -m sa1b_pipeline.normalize "${NORMALIZE_ARGS[@]}" --output-dir "$WORK_ROOT/candidates"
"$MLLM_PYTHON" -m sa1b_pipeline.select --manifest "$WORK_ROOT/candidates/candidates.jsonl" --output "$WORK_ROOT/selection/selected_cases.jsonl" --model-id "$MODEL" --tensor-parallel-size "${TP_SIZE:-8}"
"$EDITOR_PYTHON" -m sa1b_pipeline.editor --manifest "$WORK_ROOT/selection/selected_cases.jsonl" --output-dir "$WORK_ROOT/generated" --model-id "${QWEN_IMAGE_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1}" --steps "${STEPS:-40}" --tensor-parallel-size "${EDITOR_TP_SIZE:-8}"
"$MLLM_PYTHON" -m sa1b_pipeline.validate --manifest "$WORK_ROOT/generated/generated.jsonl" --output "$WORK_ROOT/validation/validation.jsonl" --model-id "$MODEL"
