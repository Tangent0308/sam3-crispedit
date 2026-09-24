#!/usr/bin/env bash
# Copy this entrypoint to shared storage before submitting four Arnold workers.
set -Eeuo pipefail
: "${ARNOLD_WORKER_NUM:?Expected four Arnold workers}"
: "${ARNOLD_WORKER_GPU:?Expected eight GPUs per worker}"
: "${ARNOLD_ID:?Expected node rank}"
: "${CRISPEDIT_RUN_ID:?Set the same unique run ID on all workers}"
[[ $ARNOLD_WORKER_NUM == 4 && $ARNOLD_WORKER_GPU == 8 && $ARNOLD_ID =~ ^[0-3]$ ]] || exit 2
[[ $CRISPEDIT_RUN_ID =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
shared_base=/mnt/bn/strategy-mllm-train/user/tanyue
repo_dir=${CRISPEDIT_REPO_DIR:-$shared_base/workspaces/crispedit_${CRISPEDIT_RUN_ID}}
run_dir=${CRISPEDIT_RUN_DIR:-$shared_base/experiments/CrispEdit/labeling_4node_${CRISPEDIT_RUN_ID}}
attempt=initial
resume_args=()
if [[ ${CRISPEDIT_RESUME:-0} == 1 ]]; then
  : "${CRISPEDIT_RESUME_TOKEN:?Set a new shared resume token}"
  attempt=$CRISPEDIT_RESUME_TOKEN
  [[ $attempt != initial && $attempt =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
  resume_args=(--resume --resume-token "$attempt")
fi
control=$run_dir/bootstrap_control/$attempt
mkdir -p "$control" "$run_dir/logs"
exec > >(tee -a "$run_dir/logs/bootstrap.node${ARNOLD_ID}.log") 2>&1
trap 'code=$?; if (( code != 0 )); then printf "exit=%s node=%s\n" "$code" "$ARNOLD_ID" > "$control/node${ARNOLD_ID}.failed"; fi' EXIT
python_bin=${CRISPEDIT_PYTHON:-$repo_dir/.venv-crispedit/bin/python}
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
if [[ $ARNOLD_ID == 0 ]]; then
  if [[ ! -d $repo_dir ]]; then
    mkdir -p "$(dirname "$repo_dir")"
    git clone --single-branch --branch "${CRISPEDIT_BRANCH:-crispedit打标}" \
      "${CRISPEDIT_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}" "$repo_dir"
  elif [[ ${CRISPEDIT_USE_EXISTING_REPO:-0} != 1 && ${CRISPEDIT_RESUME:-0} != 1 ]]; then
    echo 'Repository exists; use a new run or explicitly set CRISPEDIT_USE_EXISTING_REPO=1' >&2
    exit 2
  fi
  cd "$repo_dir"
  if [[ -z ${CRISPEDIT_RUNTIME_ARCHIVE:-} && ! -x $python_bin ]]; then
    if ! command -v uv >/dev/null; then
      python3 -m pip install --user uv==0.11.32
      export PATH="$(python3 -c 'import site; print(site.getuserbase())')/bin:$PATH"
    fi
    VENV_DIR="$(dirname "$(dirname "$python_bin")")" UV_LINK_MODE=copy \
      bash scripts/setup_crispedit_env.sh
  fi
  if [[ -z ${CRISPEDIT_RUNTIME_ARCHIVE:-} ]]; then
    "$python_bin" -c 'import torch,vllm,pyarrow,cv2; from sam3.model_builder import build_sam3_image_model; assert torch.cuda.device_count()==8'
  fi
  printf '%s\n' "$repo_dir" > "$control/environment.tmp"
  mv "$control/environment.tmp" "$control/environment.ok"
fi
started=$SECONDS
until [[ -f $control/environment.ok ]]; do
  if compgen -G "$control/*.failed" >/dev/null; then echo 'Environment setup failed; see bootstrap logs' >&2; exit 1; fi
  (( SECONDS - started < 7200 )) || { echo 'Environment wait timed out' >&2; exit 1; }
  sleep 2
done
cd "$repo_dir"
if [[ -n ${CRISPEDIT_RUNTIME_ARCHIVE:-} ]]; then
  : "${CRISPEDIT_BASE_PYTHON:?Set a shared Python 3.12 base interpreter}"
  python_bin=$(bash scripts/cache_crispedit_runtime.sh "$CRISPEDIT_RUNTIME_ARCHIVE" \
    "$CRISPEDIT_BASE_PYTHON" "${CRISPEDIT_LOCAL_RUNTIME_DIR:-/tmp/crispedit-runtime-${USER:-worker}-${CRISPEDIT_RUN_ID}}")
  export CRISPEDIT_SAM_PYTHON="$python_bin"
fi
"$python_bin" -c 'import torch,vllm,pyarrow; from sam3.model_builder import build_sam3_image_model; assert torch.cuda.device_count()==8'
sam_python=${CRISPEDIT_SAM_PYTHON:-$python_bin}
"$sam_python" -c 'import torch; from sam3.model.sam3_image_processor import Sam3Processor; assert torch.cuda.device_count()==8'
touch "$control/node${ARNOLD_ID}.ready"
started=$SECONDS
for rank in 0 1 2 3; do
  until [[ -f $control/node${rank}.ready ]]; do
    if compgen -G "$control/*.failed" >/dev/null; then echo 'A worker preflight failed; see bootstrap logs' >&2; exit 1; fi
    (( SECONDS - started < 7200 )) || { echo 'Worker preflight timed out' >&2; exit 1; }
    sleep 2
  done
done
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" -u scripts/run_crispedit_pipeline.py \
  --run-dir "$run_dir" --source-dir "${CRISPEDIT_INPUT_DIR:-$shared_base/datasets/CrispEdit-2M}" \
  --quality-dir "${CRISPEDIT_QUALITY_DIR:-$shared_base/datasets/CrispEdit-2M-qwen38-pair-prefilter}" \
  --scene-dir "${CRISPEDIT_SCENE_DIR:-$shared_base/datasets/CrispEdit-2M-difficult-local-edit}" \
  --label-dir "${CRISPEDIT_LABEL_DIR:-$run_dir/labels}" \
  --model-path "${CRISPEDIT_MODEL_DIR:-$shared_base/models/pretrained_models/Qwen3.8-27B}" \
  --checkpoint-path "${CRISPEDIT_SAM3_CHECKPOINT_PATH:-/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt}" \
  --python "$python_bin" --sam-python "${CRISPEDIT_SAM_PYTHON:-$python_bin}" \
  --nodes 4 --rank "$ARNOLD_ID" --devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 1 --batch-size "${CRISPEDIT_FILTER_BATCH_SIZE:-4}" \
  --grounding-tp 2 --grounding-batch-size "${CRISPEDIT_GROUNDING_BATCH_SIZE:-16}" \
  "${resume_args[@]}"
