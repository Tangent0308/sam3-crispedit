#!/usr/bin/env bash
# Run on every Arnold worker AFTER cloning this repository to node-local disk.
set -Eeuo pipefail
: "${ARNOLD_WORKER_NUM:?Expected four Arnold workers}"
: "${ARNOLD_WORKER_GPU:?Expected eight GPUs per worker}"
: "${ARNOLD_ID:?Expected node rank}"
: "${CRISPEDIT_RUN_ID:?Set the same unique run ID on all workers}"
[[ $ARNOLD_WORKER_NUM == 4 && $ARNOLD_WORKER_GPU == 8 && $ARNOLD_ID =~ ^[0-3]$ ]] || exit 2
[[ $CRISPEDIT_RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $repo_dir != /mnt/* ]] || { echo 'Clone code on node-local disk (/opt or /tmp).' >&2; exit 2; }
for obsolete in CRISPEDIT_RUNTIME_ARCHIVE CRISPEDIT_BASE_PYTHON CRISPEDIT_LOCAL_RUNTIME_DIR CRISPEDIT_PYTHON CRISPEDIT_SAM_PYTHON; do
  [[ -z ${!obsolete:-} ]] || { echo "Unset obsolete environment override: $obsolete" >&2; exit 2; }
done
shared_base=/mnt/bn/strategy-mllm-train/user/tanyue
run_dir=${CRISPEDIT_RUN_DIR:-$shared_base/experiments/CrispEdit/labeling_4node_${CRISPEDIT_RUN_ID}}
attempt=initial
resume_args=()
if [[ ${CRISPEDIT_RESUME:-0} == 1 ]]; then
  : "${CRISPEDIT_RESUME_TOKEN:?Set a new shared resume token}"
  attempt=$CRISPEDIT_RESUME_TOKEN
  [[ $attempt != initial && $attempt =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
  resume_args=(--resume --resume-token "$attempt")
  if [[ ${CRISPEDIT_ALLOW_CODE_CHANGE_ON_RESUME:-0} == 1 ]]; then
    resume_args+=(--allow-code-change-on-resume)
  fi
fi
control=$run_dir/bootstrap_control/$attempt
mkdir -p "$control" "$run_dir/logs"
exec > >(tee -a "$run_dir/logs/bootstrap.node${ARNOLD_ID}.log") 2>&1
exec 8>"$run_dir/bootstrap.node${ARNOLD_ID}.lock"
flock -n 8 || { echo 'This rank already has a running bootstrap' >&2; exit 2; }
[[ ! -f $control/node${ARNOLD_ID}.started ]] || { echo 'Attempt already used; choose a new run or resume token' >&2; exit 2; }
touch "$control/node${ARNOLD_ID}.started"
trap 'code=$?; if (( code != 0 )); then printf "exit=%s node=%s; see logs/bootstrap.node%s.log\n" "$code" "$ARNOLD_ID" "$ARNOLD_ID" > "$control/node${ARNOLD_ID}.failed"; fi' EXIT
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export CRISPEDIT_EXPECTED_GPUS=8
cd "$repo_dir"
echo "$(date -u +%FT%TZ) node${ARNOLD_ID}: local clone=$repo_dir commit=$(git rev-parse HEAD)"
if ! command -v uv >/dev/null; then
  python3 -m pip install --user uv==0.11.32
  export PATH="$(python3 -c 'import site; print(site.getuserbase())')/bin:$PATH"
fi
bash scripts/setup_crispedit_env.sh
python_bin=$repo_dir/.venv-crispedit/bin/python
model=${CRISPEDIT_MODEL_DIR:-$shared_base/models/pretrained_models/Qwen3.8-27B}
export PYTHONPATH="$repo_dir"
"$python_bin" -u scripts/preflight_crispedit_env.py --gpus 8 --model-path "$model" \
  --output-json "$control/node${ARNOLD_ID}.ready.json" \
  2>&1 | tee -a "$run_dir/logs/preflight.node${ARNOLD_ID}.log"
echo "$(date -u +%FT%TZ) node${ARNOLD_ID}: preflight complete; waiting for all four nodes"
started=$SECONDS
for rank in 0 1 2 3; do
  until [[ -f $control/node${rank}.ready.json ]]; do
    if compgen -G "$control/*.failed" >/dev/null; then echo 'A worker failed; see bootstrap/preflight logs' >&2; exit 1; fi
    (( SECONDS - started < 7200 )) || { echo 'Worker installation/preflight wait timed out' >&2; exit 1; }
    sleep 2
  done
done
"$python_bin" scripts/preflight_crispedit_env.py --cluster-dir "$control"
"$python_bin" -u scripts/run_crispedit_pipeline.py \
  --run-dir "$run_dir" --source-dir "${CRISPEDIT_INPUT_DIR:-$shared_base/datasets/CrispEdit-2M}" \
  --quality-dir "${CRISPEDIT_QUALITY_DIR:-$shared_base/datasets/CrispEdit-2M-qwen38-pair-prefilter}" \
  --scene-dir "${CRISPEDIT_SCENE_DIR:-$shared_base/datasets/CrispEdit-2M-difficult-local-edit}" \
  --label-dir "${CRISPEDIT_LABEL_DIR:-$run_dir/labels}" \
  --model-path "$model" \
  --checkpoint-path "${CRISPEDIT_SAM3_CHECKPOINT_PATH:-/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt}" \
  --python "$python_bin" --sam-python "$python_bin" \
  --nodes 4 --rank "$ARNOLD_ID" --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 1 --batch-size "${CRISPEDIT_FILTER_BATCH_SIZE:-4}" \
  --grounding-tp 2 --grounding-batch-size "${CRISPEDIT_GROUNDING_BATCH_SIZE:-16}" \
  "${resume_args[@]}"
