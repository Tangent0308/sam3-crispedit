#!/usr/bin/env bash
# Copy this complete file to a shared pre-clone path or paste it into Arnold entry.
# Run the SAME entry on all four workers. Each worker clones to node-local storage.
set -euo pipefail
: "${SAMTOK_RUN_ID:?Set one new unique run ID, identical on all four workers}"
: "${ARNOLD_WORKER_NUM:?Arnold topology missing}"
: "${ARNOLD_WORKER_GPU:?Arnold topology missing}"
: "${ARNOLD_ID:?Arnold topology missing}"
[[ "$SAMTOK_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'Requires 4 nodes x 8 GPUs' >&2; exit 2; }
export SAMTOK_RUN_ROOT="${SAMTOK_RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID}"
export SAMTOK_DATA_ROOT="${SAMTOK_DATA_ROOT:-$SAMTOK_RUN_ROOT/data/source}"
export SAMTOK_PARQUET="${SAMTOK_PARQUET:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet}"
export SAMTOK_REPO_DIR="${SAMTOK_REPO_DIR:-/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID/node$ARNOLD_ID/repo}"
export SAMTOK_REPO_URL="${SAMTOK_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}"
export SAMTOK_BRANCH="${SAMTOK_BRANCH:-samtok-derived-edit-labeling}"
mkdir -p "$SAMTOK_RUN_ROOT/logs" "$SAMTOK_RUN_ROOT/control"
# Atomic claim: old run IDs and accidental duplicate entries must never reuse markers.
(set -o noclobber; printf '%s\n' "$(hostname) $$" > "$SAMTOK_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.claim") || exit 1
exec > >(tee -a "$SAMTOK_RUN_ROOT/logs/bootstrap.node$ARNOLD_ID.log") 2>&1
on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf '{"error":"bootstrap node %s exited %s; see bootstrap log"}\n' "$ARNOLD_ID" "$rc" \
      > "$SAMTOK_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json.tmp"
    mv "$SAMTOK_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json.tmp" "$SAMTOK_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json"
  fi
}
trap on_exit EXIT
export PYTHONUNBUFFERED=1
# No credentials are printed; retain standard git credential helpers.
if [[ "${SAMTOK_KEEP_PROXY:-0}" != 1 ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
fi
[[ ! -e "$SAMTOK_REPO_DIR" ]] || { echo "Clone path already exists: $SAMTOK_REPO_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$SAMTOK_REPO_DIR")"
git clone --branch "$SAMTOK_BRANCH" --single-branch "$SAMTOK_REPO_URL" "$SAMTOK_REPO_DIR"
cd "$SAMTOK_REPO_DIR"
mkdir -p "$SAMTOK_RUN_ROOT/reports"
git rev-parse HEAD > "$SAMTOK_RUN_ROOT/reports/checkout.node$ARNOLD_ID.txt"
if [[ -n "${SAMTOK_EXPECTED_COMMIT:-}" ]]; then
  [[ "$(git rev-parse HEAD)" == "$SAMTOK_EXPECTED_COMMIT" ]] || { echo 'Unexpected branch revision' >&2; exit 1; }
fi
python3 -m pip install --user --index-url "${SAMTOK_PACKAGE_INDEX:-https://bytedpypi.byted.org/simple/}" 'uv==0.11.32'
export UV_BIN="$(python3 -c 'import site; print(site.getuserbase())')/bin/uv"
export SAMTOK_RUNTIME_ROOT="$SAMTOK_REPO_DIR/.runtime"
export SAMTOK_MLLM_PYTHON="$SAMTOK_RUNTIME_ROOT/mllm/bin/python"
export SAMTOK_EDITOR_PYTHON="$SAMTOK_RUNTIME_ROOT/editor/bin/python"
export SAMTOK_SAM_PYTHON="$SAMTOK_RUNTIME_ROOT/sam/bin/python"
export SAMTOK_SAM3_SOURCE="$SAMTOK_RUNTIME_ROOT/sam3-source"
export SAMTOK_QWEN38_MODEL="${SAMTOK_QWEN38_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}"
export SAMTOK_QWEN21_MODEL="${SAMTOK_QWEN21_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1}"
export SAMTOK_SAM3_CHECKPOINT="${SAMTOK_SAM3_CHECKPOINT:-/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt}"
for file in "$SAMTOK_QWEN38_MODEL/config.json" "$SAMTOK_QWEN21_MODEL/model_index.json" "$SAMTOK_SAM3_CHECKPOINT"; do
  [[ -r "$file" ]] || { echo "Missing shared input: $file" >&2; exit 1; }
done
bash scripts/labeling/setup_env.sh
export SAMTOK_ENV_REPORT="$SAMTOK_RUNTIME_ROOT/environment.json"
export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "${SAMTOK_STAGE_EDITOR_MODEL:-1}" == 1 ]]; then
  editor_cache="${SAMTOK_MODEL_CACHE_ROOT:-/opt/tiger/tanyue/labeling_model_cache/$SAMTOK_RUN_ID}/qwen21"
  "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.stage_labeling_model \
    --source "$SAMTOK_QWEN21_MODEL" --destination "$editor_cache"
  export SAMTOK_QWEN21_MODEL="$editor_cache"
fi
# Node 0 materializes once if no prepared manifest was supplied. All original
# regions are kept; 0 means all positives, not an invented 100k duplication.
if [[ "$ARNOLD_ID" == 0 ]]; then
  if [[ ! -f "$SAMTOK_DATA_ROOT/annotations.jsonl" ]]; then
    "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.prepare_removal_inputs \
      --parquet "$SAMTOK_PARQUET" --out-root "$SAMTOK_DATA_ROOT" \
      --limit-sources "${SAMTOK_LIMIT_SOURCES:-0}"
  fi
  printf '{"ready":true}\n' > "$SAMTOK_RUN_ROOT/control/data.ok.json.tmp"
  mv "$SAMTOK_RUN_ROOT/control/data.ok.json.tmp" "$SAMTOK_RUN_ROOT/control/data.ok.json"
fi
start_wait=$SECONDS
until [[ -f "$SAMTOK_RUN_ROOT/control/data.ok.json" ]]; do
  if compgen -G "$SAMTOK_RUN_ROOT/control/*.failed.json" > /dev/null; then
    echo 'Peer bootstrap failed; see experiments logs' >&2; exit 1
  fi
  (( SECONDS - start_wait < 10800 )) || { echo 'Data preparation timeout' >&2; exit 1; }
  sleep 2
done
# No MASTER_PORT / ARNOLD_WORKER_HOSTS rendezvous: independent data shards.
exec "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.run_multinode_labeling \
  --data-root "$SAMTOK_DATA_ROOT" --run-root "$SAMTOK_RUN_ROOT" --run-id "$SAMTOK_RUN_ID" \
  --rank "$ARNOLD_ID" --gpus 0,1,2,3,4,5,6,7
