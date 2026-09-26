#!/usr/bin/env bash
# Copy this complete file to a shared pre-clone path or paste it into Arnold entry.
# Run the SAME entry on all four workers. Each worker clones to node-local storage.
set -euo pipefail
: "${SAMTOK_RUN_ID:?Set one run ID, identical on all four workers}"
: "${ARNOLD_WORKER_NUM:?Arnold topology missing}"
: "${ARNOLD_WORKER_GPU:?Arnold topology missing}"
: "${ARNOLD_ID:?Arnold topology missing}"
[[ "$SAMTOK_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid run ID' >&2; exit 2; }
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'Requires 4 nodes x 8 GPUs' >&2; exit 2; }
export SAMTOK_RUN_ROOT="${SAMTOK_RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID}"
export SAMTOK_PIPELINE_MODE="${SAMTOK_PIPELINE_MODE:-remove}"
[[ "$SAMTOK_PIPELINE_MODE" == remove || "$SAMTOK_PIPELINE_MODE" == multitype ]] || { echo 'SAMTOK_PIPELINE_MODE must be remove or multitype' >&2; exit 2; }
export SAMTOK_RESUME="${SAMTOK_RESUME:-0}"
[[ "$SAMTOK_RESUME" == 0 || "$SAMTOK_RESUME" == 1 ]] || { echo 'SAMTOK_RESUME must be 0 or 1' >&2; exit 2; }
resume_args=()
attempt_suffix=""
export SAMTOK_CONTROL_ROOT="$SAMTOK_RUN_ROOT"
if [[ "$SAMTOK_RESUME" == 1 ]]; then
  : "${SAMTOK_ATTEMPT_ID:?Resume needs a NEW attempt ID, identical on all four workers}"
  [[ "$SAMTOK_ATTEMPT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid attempt ID' >&2; exit 2; }
  [[ -f "$SAMTOK_RUN_ROOT/reports/partition.json" ]] || { echo 'No prepared run to resume' >&2; exit 2; }
  export SAMTOK_CONTROL_ROOT="$SAMTOK_RUN_ROOT/attempts/$SAMTOK_ATTEMPT_ID"
  attempt_suffix="/attempts/$SAMTOK_ATTEMPT_ID"
  resume_args=(--resume)
elif [[ -n "${SAMTOK_ATTEMPT_ID:-}" ]]; then
  echo 'SAMTOK_ATTEMPT_ID is only valid with SAMTOK_RESUME=1' >&2; exit 2
fi
if [[ "$SAMTOK_PIPELINE_MODE" == multitype ]]; then
  export SAMTOK_DATA_ROOT="${SAMTOK_DATA_ROOT:-$SAMTOK_RUN_ROOT/data/add_replace_attribute}"
else
  export SAMTOK_DATA_ROOT="${SAMTOK_DATA_ROOT:-$SAMTOK_RUN_ROOT/data/source}"
fi
export SAMTOK_PARQUET="${SAMTOK_PARQUET:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/SAMTok_Training_Data/mix_gres8k_ver4k_train.parquet}"
export SAMTOK_REPO_DIR="${SAMTOK_REPO_DIR:-/opt/tiger/tanyue/labeling_runs/$SAMTOK_RUN_ID$attempt_suffix/node$ARNOLD_ID/repo}"
export SAMTOK_REPO_URL="${SAMTOK_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}"
export SAMTOK_BRANCH="${SAMTOK_BRANCH:-samtok-derived-edit-labeling}"
mkdir -p "$SAMTOK_CONTROL_ROOT/logs" "$SAMTOK_CONTROL_ROOT/control" "$SAMTOK_RUN_ROOT/control"
# Held by the shell and inherited by exec; released automatically after job exit.
exec 9>>"$SAMTOK_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.lock"
flock -n 9 || { echo 'Another attempt is active on this rank' >&2; exit 1; }
# Each restart gets new barriers and claims; old markers remain as evidence.
(set -o noclobber; printf '%s\n' "$(hostname) $$" > "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.claim") || exit 1
exec > >(tee -a "$SAMTOK_CONTROL_ROOT/logs/bootstrap.node$ARNOLD_ID.log") 2>&1
on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    printf '{"error":"bootstrap node %s exited %s; see bootstrap log"}\n' "$ARNOLD_ID" "$rc" \
      > "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json.tmp"
    mv "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json.tmp" "$SAMTOK_CONTROL_ROOT/control/bootstrap.node$ARNOLD_ID.failed.json"
  fi
}
trap on_exit EXIT
check_peers() {
  if compgen -G "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/*.failed.json" > /dev/null; then
    echo 'Peer bootstrap failed; see experiments logs' >&2; exit 1
  fi
}
export PYTHONUNBUFFERED=1
# No credentials are printed; retain standard git credential helpers.
if [[ "${SAMTOK_KEEP_PROXY:-0}" != 1 ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
fi
[[ ! -e "$SAMTOK_REPO_DIR" ]] || { echo "Clone path already exists: $SAMTOK_REPO_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$SAMTOK_REPO_DIR")"
git clone --branch "$SAMTOK_BRANCH" --single-branch "$SAMTOK_REPO_URL" "$SAMTOK_REPO_DIR"
cd "$SAMTOK_REPO_DIR"
mkdir -p "$SAMTOK_CONTROL_ROOT/reports"
git rev-parse HEAD > "$SAMTOK_CONTROL_ROOT/reports/checkout.node$ARNOLD_ID.txt"
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
# Do not stage tens of GB or prepare data while another worker failed its imports.
check_peers
printf '{"ready":true}\n' > "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$ARNOLD_ID.ok.json.tmp"
mv "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$ARNOLD_ID.ok.json.tmp" "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$ARNOLD_ID.ok.json"
start_wait=$SECONDS
while true; do
  check_peers
  all_ready=1
  for peer_rank in 0 1 2 3; do
    [[ -f "${SAMTOK_CONTROL_ROOT:-$SAMTOK_RUN_ROOT}/control/environment.node$peer_rank.ok.json" ]] || all_ready=0
  done
  (( all_ready == 1 )) && break
  (( SECONDS - start_wait < 10800 )) || { echo 'Peer environment timeout' >&2; exit 1; }
  sleep 2
done
export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "${SAMTOK_STAGE_EDITOR_MODEL:-1}" == 1 ]]; then
  editor_cache="${SAMTOK_MODEL_CACHE_ROOT:-/opt/tiger/tanyue/labeling_model_cache/$SAMTOK_RUN_ID}/qwen21"
  "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.stage_labeling_model \
    --source "$SAMTOK_QWEN21_MODEL" --destination "$editor_cache" --run-root "$SAMTOK_CONTROL_ROOT" "${resume_args[@]}"
  export SAMTOK_QWEN21_MODEL="$editor_cache"
fi
# Node 0 materializes once if no prepared manifest was supplied. All original
# regions are kept; 0 means all positives, not an invented 100k duplication.
check_peers
if [[ "$ARNOLD_ID" == 0 ]]; then
  if [[ ! -f "$SAMTOK_DATA_ROOT/annotations.jsonl" ]]; then
    if [[ "$SAMTOK_PIPELINE_MODE" == multitype ]]; then
      "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.prepare_multitype_inputs \
        --parquet "$SAMTOK_PARQUET" --out-root "$SAMTOK_DATA_ROOT" \
        --limit-sources "${SAMTOK_LIMIT_SOURCES:-0}" --force-index
    else
      "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.prepare_removal_inputs \
        --parquet "$SAMTOK_PARQUET" --out-root "$SAMTOK_DATA_ROOT" \
        --limit-sources "${SAMTOK_LIMIT_SOURCES:-0}" --run-root "$SAMTOK_CONTROL_ROOT" "${resume_args[@]}"
    fi
  fi
  check_peers
  printf '{"ready":true}\n' > "$SAMTOK_CONTROL_ROOT/control/data.ok.json.tmp"
  mv "$SAMTOK_CONTROL_ROOT/control/data.ok.json.tmp" "$SAMTOK_CONTROL_ROOT/control/data.ok.json"
fi
start_wait=$SECONDS
until [[ -f "$SAMTOK_CONTROL_ROOT/control/data.ok.json" ]]; do
  check_peers
  (( SECONDS - start_wait < 10800 )) || { echo 'Data preparation timeout' >&2; exit 1; }
  sleep 2
done
check_peers
# No MASTER_PORT / ARNOLD_WORKER_HOSTS rendezvous: independent data shards.
if [[ "$SAMTOK_RESUME" == 1 ]]; then resume_args+=(--attempt-id "$SAMTOK_ATTEMPT_ID"); fi
if [[ "$SAMTOK_PIPELINE_MODE" == multitype ]]; then
  exec "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.run_multinode_multitype_labeling \
    --data-root "$SAMTOK_DATA_ROOT" --run-root "$SAMTOK_RUN_ROOT" --run-id "$SAMTOK_RUN_ID" \
    --rank "$ARNOLD_ID" --gpus 0,1,2,3,4,5,6,7 "${resume_args[@]}"
else
  exec "$SAMTOK_SAM_PYTHON" -m synthesis_pipeline.run_multinode_labeling \
    --data-root "$SAMTOK_DATA_ROOT" --run-root "$SAMTOK_RUN_ROOT" --run-id "$SAMTOK_RUN_ID" \
    --rank "$ARNOLD_ID" --gpus 0,1,2,3,4,5,6,7 "${resume_args[@]}"
fi
