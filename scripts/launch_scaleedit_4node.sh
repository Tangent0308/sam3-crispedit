#!/usr/bin/env bash
# Standalone Arnold entry: copy this file into the job entry on all four nodes.
set -Eeuo pipefail
: "${SCALEEDIT_RUN_ID:?Set one unique run ID shared by all four nodes}"
: "${ARNOLD_WORKER_NUM:?Expected four workers}"
: "${ARNOLD_WORKER_GPU:?Expected eight GPUs per worker}"
: "${ARNOLD_ID:?Expected rank 0..3}"
[[ $ARNOLD_WORKER_NUM == 4 && $ARNOLD_WORKER_GPU == 8 && $ARNOLD_ID =~ ^[0-3]$ ]] || exit 2
[[ $SCALEEDIT_RUN_ID =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*$ ]] || exit 2
shared_base=/mnt/bn/strategy-mllm-train/user/tanyue
export SCALEEDIT_RUN_DIR=${SCALEEDIT_RUN_DIR:-$shared_base/experiments/ScaleEdit/labeling_4node_${SCALEEDIT_RUN_ID}}
branch=${SCALEEDIT_BRANCH:-scaleedit-labeling}
url=${SCALEEDIT_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}
repo_dir=${SCALEEDIT_REPO_DIR:-/opt/tiger/tanyue/workspaces/sam3-crispedit-scaleedit-labeling-${SCALEEDIT_RUN_ID}-node${ARNOLD_ID}}
[[ $(realpath -m "$repo_dir") != /mnt/* ]] || { echo 'Repository must be node-local' >&2; exit 2; }
attempt=${SCALEEDIT_ATTEMPT:-initial}
[[ $attempt =~ ^[A-Za-z0-9_-][A-Za-z0-9._-]*$ ]] || exit 2
if [[ ${SCALEEDIT_RESUME:-0} == 1 ]]; then
  [[ $attempt != initial ]] || { echo 'Resume requires a new SCALEEDIT_ATTEMPT' >&2; exit 2; }
else
  [[ $attempt == initial ]] || exit 2
fi
control=$SCALEEDIT_RUN_DIR/bootstrap_control/$attempt
mkdir -p "$control" "$SCALEEDIT_RUN_DIR/logs" "$(dirname "$repo_dir")"
exec > >(tee -a "$SCALEEDIT_RUN_DIR/logs/entry.node${ARNOLD_ID}.log") 2>&1
exec 8>"$SCALEEDIT_RUN_DIR/bootstrap.node${ARNOLD_ID}.lock"
flock -n 8 || { echo 'This node already has a running bootstrap' >&2; exit 2; }
trap 'code=$?; if (( code != 0 )); then printf "exit=%s node=%s; see entry.node%s.log\n" "$code" "$ARNOLD_ID" "$ARNOLD_ID" > "$control/node${ARNOLD_ID}.failed"; fi' EXIT
if [[ ! -e $repo_dir ]]; then
  git clone --single-branch --branch "$branch" "$url" "$repo_dir"
fi
[[ -d $repo_dir/.git ]] || { echo "Not a completed clone: $repo_dir" >&2; exit 2; }
cd "$repo_dir"
[[ $(git remote get-url origin) == "$url" ]] || { echo 'Existing clone has a different origin' >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Existing clone has tracked modifications' >&2; exit 2; }
git fetch --no-tags origin "$branch"
target_commit=${SCALEEDIT_COMMIT:-$(git rev-parse FETCH_HEAD)}
[[ $target_commit =~ ^[0-9a-f]{40}$ ]] || exit 2
git merge-base --is-ancestor "$target_commit" FETCH_HEAD
git checkout --detach "$target_commit"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
if ! command -v uv >/dev/null; then
  python3 -m pip install --user uv==0.11.32
  export PATH="$(python3 -c 'import site; print(site.getuserbase())')/bin:$PATH"
fi
bash scripts/setup_scaleedit_env.sh
python_bin=$repo_dir/.venv-scaleedit-current/bin/python
model=${SCALEEDIT_MODEL_DIR:-$shared_base/models/pretrained_models/Qwen3.8-27B}
"$python_bin" -u scripts/preflight_scaleedit_env.py --gpus 8 --model-path "$model" \
  --output-json "$control/node${ARNOLD_ID}.ready.json" \
  2>&1 | tee -a "$SCALEEDIT_RUN_DIR/logs/preflight.node${ARNOLD_ID}.log"
started=$SECONDS
for rank in 0 1 2 3; do
  until [[ -f $control/node${rank}.ready.json ]]; do
    if compgen -G "$control/*.failed" >/dev/null; then echo 'Peer bootstrap failed' >&2; exit 1; fi
    (( SECONDS - started < 7200 )) || { echo 'Peer bootstrap timed out' >&2; exit 1; }
    sleep 2
  done
done
"$python_bin" scripts/preflight_scaleedit_env.py --cluster-dir "$control"
resume=()
if [[ ${SCALEEDIT_RESUME:-0} == 1 ]]; then resume=(--resume --attempt "$attempt"); fi
"$python_bin" -u scripts/run_scaleedit_4node.py \
  --run-dir "$SCALEEDIT_RUN_DIR" \
  --source-dir "${SCALEEDIT_INPUT_DIR:-$shared_base/datasets/ScaleEdit-filtered-balanced-final-task-100k}" \
  --model-path "$model" \
  --checkpoint-path "${SCALEEDIT_SAM3_CHECKPOINT_PATH:-/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt}" \
  --nodes 4 --rank "$ARNOLD_ID" --devices 0,1,2,3,4,5,6,7 \
  --batch-size "${SCALEEDIT_FILTER_BATCH_SIZE:-4}" \
  --grounding-tp 2 --grounding-batch-size "${SCALEEDIT_GROUNDING_BATCH_SIZE:-4}" \
  "${resume[@]}"
