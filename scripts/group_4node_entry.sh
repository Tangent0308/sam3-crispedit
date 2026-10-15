#!/usr/bin/env bash
set -euo pipefail

# Arnold launches the same script on four workers. Each worker owns one
# deterministic selection shard and writes to its own resumable root.
: "${GROUP_ROOT:?set GROUP_ROOT to the independent grouped-data root}"
: "${GROUP_KIND:?set GROUP_KIND to twins or multi}"
: "${GROUP_SPLIT:?set GROUP_SPLIT to train or eval}"
: "${GROUP_SELECTION:?set GROUP_SELECTION to the preselected JSONL}"
: "${GROUP_MANIFEST:?set GROUP_MANIFEST to train_candidates.jsonl or eval_candidates.jsonl}"
: "${GROUP_RESERVATION:?set GROUP_RESERVATION to eval_reservation.json}"
: "${GROUP_PLANNER_MODEL:?set GROUP_PLANNER_MODEL}"
: "${GROUP_EDITOR_MODEL:?set GROUP_EDITOR_MODEL}"
: "${GROUP_AUDIT_MODEL:?set GROUP_AUDIT_MODEL}"
: "${ARNOLD_WORKER_NUM:?Arnold did not provide ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?Arnold did not provide ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?Arnold did not provide ARNOLD_ID}"

if [[ "$ARNOLD_WORKER_NUM" != 4 || "$ARNOLD_WORKER_GPU" != 8 || ! "$ARNOLD_ID" =~ ^[0-3]$ ]]; then
  echo "grouped pipeline requires four workers with eight GPUs each and rank 0..3" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export WANDB_DISABLED=true
# The production editor runtime is CUDA 12.9; avoid incompatible automatic
# FlashAttention selection in vLLM-Omni.
export DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-TORCH_SDPA}"

SHARD_ROOT="${GROUP_SHARD_ROOT:-$GROUP_ROOT/shards/${GROUP_KIND}_${GROUP_SPLIT}}"
SHARD="$SHARD_ROOT/selected_rank${ARNOLD_ID}.jsonl"
RUN_ROOT="$GROUP_ROOT/runs/${GROUP_KIND}_${GROUP_SPLIT}/rank${ARNOLD_ID}"
SELECTED="$RUN_ROOT/inputs/selected.jsonl"
[[ -r "$GROUP_SELECTION" ]] || { echo "missing source selection: $GROUP_SELECTION" >&2; exit 2; }
[[ -r "$SHARD" ]] || { echo "missing shard: $SHARD" >&2; exit 2; }
mkdir -p "$RUN_ROOT/inputs" "$RUN_ROOT/logs"

if [[ -e "$SELECTED" ]]; then
  cmp -s "$SHARD" "$SELECTED" || { echo "existing selected.jsonl differs from shard; use the same selection" >&2; exit 2; }
else
  tmp="$SELECTED.tmp.$$"
  cp "$SHARD" "$tmp"
  mv "$tmp" "$SELECTED"
fi

TARGET="$(wc -l < "$SHARD")"
LOG="$RUN_ROOT/logs/rank${ARNOLD_ID}.log"
ARGS=(
  --kind "$GROUP_KIND"
  --manifest "$GROUP_MANIFEST"
  --output-root "$RUN_ROOT"
  --target "$TARGET"
  --split "$GROUP_SPLIT"
  --reservation "$GROUP_RESERVATION"
  --planner-model "$GROUP_PLANNER_MODEL"
  --planner-backend "${GROUP_PLANNER_BACKEND:-vllm}"
  --planner-tp "${GROUP_PLANNER_TP:-4}"
  --planner-batch-size "${GROUP_PLANNER_BATCH_SIZE:-8}"
  --max-new-tokens "${GROUP_MAX_NEW_TOKENS:-1200}"
  --outline-planning-crops
  --locate-planning-scene
  --editor-model "$GROUP_EDITOR_MODEL"
  --audit-model "$GROUP_AUDIT_MODEL"
  --audit-tp "${GROUP_AUDIT_TP:-1}"
  --audit-batch-size "${GROUP_AUDIT_BATCH_SIZE:-4}"
)
if [[ "${GROUP_PLAN_ONLY:-0}" == 1 ]]; then
  ARGS+=(--plan-only)
fi
if [[ "${GROUP_PUBLISH:-0}" == 1 ]]; then
  : "${GROUP_HUMAN_REVIEW:?set GROUP_HUMAN_REVIEW to the completed human_review.jsonl ledger}"
  ARGS+=(--human-review "$GROUP_HUMAN_REVIEW")
  ARGS+=(--publish)
fi

echo "rank=$ARNOLD_ID target=$TARGET root=$RUN_ROOT" | tee -a "$LOG"
python "$ROOT_DIR/scripts/run_group_pipeline.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG"
