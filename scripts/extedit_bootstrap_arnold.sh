#!/usr/bin/env bash
# extedit production run on ARNOLD.  Paste this whole file as the start command of every worker (4 workers x 8 GPUs)
# after editing the run settings; docs/EXTEDIT_PROTOCOL2.md section 4.4 has the ready-to-paste settings.
#
# Workers are independent: worker k takes every n-th shard of the shuffled shard lists (the same order on every
# worker), labels its rows with the streaming pipeline (Qwen3.8-27B data parallel on GPUs 0-6, SAM3 + SAMTok on
# GPU 7) and exports <run>/node<k>/final.  A restarted worker resumes where it stopped: every stage skips the rows it
# already wrote.  Merging the nodes and publishing to datasets/ is a separate step after review (publish).
exec 2>&1  # ARNOLD's log page shows stdout; route every shell error there as well.

# ===== run settings (edit these) =====
# Identical on all workers; a new ID starts a new run, the same ID resumes it.
export EXTEDIT_RUN_ID="${EXTEDIT_RUN_ID:-extedit_protocol2_100k_20261021}"
# Shared root of production runs: the run goes to $EXTEDIT_ROOT/$EXTEDIT_RUN_ID.
export EXTEDIT_ROOT="${EXTEDIT_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs}"
# Raw rows per type ON EACH WORKER (0 = every row of the worker's shards).
export EXTEDIT_CRISPEDIT="${EXTEDIT_CRISPEDIT:-replace=3750,color=2000,remove=1250,add=625}"
export EXTEDIT_SCALEEDIT="${EXTEDIT_SCALEEDIT:-object_surface_text_editing=3750,building_surface_text_editing=3750,object_replacement=3750,color_change=2000,material_change=0,object_removal=1250,object_addition=625}"
# Code at exact commits: the labeling branch, and SAMTokEdit v3 for the SAMTok codec and the CUDA check.
export EXTEDIT_REPO_URL="${EXTEDIT_REPO_URL:-https://github.com/Tangent0308/sam3-crispedit.git}"
export EXTEDIT_BRANCH="${EXTEDIT_BRANCH:-extedit-protocol2-labeling}"
export EXTEDIT_COMMIT="${EXTEDIT_COMMIT:?set EXTEDIT_COMMIT to the labeling commit to run}"
export EXTEDIT_V3_URL="${EXTEDIT_V3_URL:-https://github.com/Tangent0308/samtok_edit.git}"
export EXTEDIT_V3_BRANCH="${EXTEDIT_V3_BRANCH:-qwen-image-2.1-v3}"
export EXTEDIT_V3_COMMIT="${EXTEDIT_V3_COMMIT:-31fb966}"
export EXTEDIT_MODEL="${EXTEDIT_MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}"
# Number of workers; 1 runs the same pipeline on a single 8-GPU machine.
export EXTEDIT_NNODES="${EXTEDIT_NNODES:-4}"

# ===== ARNOLD checks =====
: "${ARNOLD_WORKER_NUM:?ARNOLD must inject ARNOLD_WORKER_NUM}"
: "${ARNOLD_WORKER_GPU:?ARNOLD must inject ARNOLD_WORKER_GPU}"
: "${ARNOLD_ID:?ARNOLD must inject ARNOLD_ID for this worker}"
[[ "$ARNOLD_WORKER_NUM" == "$EXTEDIT_NNODES" ]] || { echo "Expected $EXTEDIT_NNODES workers, ARNOLD reports $ARNOLD_WORKER_NUM"; exit 2; }
[[ "$ARNOLD_WORKER_GPU" == 8 ]] || { echo "Each worker needs 8 GPUs (7 for the 27B server, 1 for SAM3), got $ARNOLD_WORKER_GPU"; exit 2; }
[[ "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$EXTEDIT_NNODES" ]] || { echo "ARNOLD_ID must be 0..$((EXTEDIT_NNODES - 1))"; exit 2; }
# torch launcher variables would make the vLLM engines join a foreign process group.
unset PORT MASTER_ADDR MASTER_PORT NODE_RANK NNODES GPUS_PER_NODE RANK LOCAL_RANK WORLD_SIZE
NODE="$ARNOLD_ID"
RUN="$EXTEDIT_ROOT/$EXTEDIT_RUN_ID"
WORK="/tmp/extedit-${EXTEDIT_RUN_ID}-node${NODE}"
NODE_RUN="$RUN/node$NODE"
mkdir -p "$RUN/bootstrap" "$NODE_RUN" "$WORK" || { echo "Cannot create $RUN"; exit 2; }
# A full NAS directory quota rejects every new file while df still reports free space.
PROBE="$RUN/bootstrap/.write_probe_node${NODE}"
( head -c 1048576 /dev/zero > "$PROBE" && rm -f "$PROBE" ) || { echo "Cannot write to $RUN (quota?)"; exit 2; }
if [[ -f "$NODE_RUN/SUCCESS.json" ]]; then
  echo "Worker $NODE of run $EXTEDIT_RUN_ID already finished ($NODE_RUN/SUCCESS.json); nothing to do."
  exit 0
fi
exec > >(tee -a "$RUN/bootstrap/node${NODE}.log") 2>&1
ATTEMPT=$(( $(cat "$NODE_RUN/attempts" 2>/dev/null || echo 0) + 1 ))
echo "$ATTEMPT" > "$NODE_RUN/attempts"
[[ -f "$NODE_RUN/failure.json" ]] && mv "$NODE_RUN/failure.json" "$NODE_RUN/failure.attempt$((ATTEMPT - 1)).json"
phase() { echo "$1" > "$NODE_RUN/phase"; echo "[$(date -u +%FT%TZ)] node $NODE attempt $ATTEMPT: $1"; }
SERVER="" PIPELINE=""
stop_children() {
  [[ -n "$PIPELINE" ]] && kill -TERM -- "-$PIPELINE" 2>/dev/null || true
  [[ -n "$SERVER" ]] && kill -TERM -- "-$SERVER" 2>/dev/null || true
}
failed() {
  local result=${1:-$?}
  trap - ERR TERM INT
  stop_children
  printf '{"error":"failed during %s; see bootstrap/node%s.log","exit_code":%d,"attempt":%d}\n' \
    "$(cat "$NODE_RUN/phase" 2>/dev/null || echo unknown)" "$NODE" "$result" "$ATTEMPT" > "$NODE_RUN/failure.json"
  exit "$result"
}
set -Eeuo pipefail
trap failed ERR
trap 'failed 143' TERM
trap 'failed 130' INT
echo "[$(date -u +%FT%TZ)] node $NODE of $EXTEDIT_NNODES: start, run $EXTEDIT_RUN_ID, attempt $ATTEMPT, host $(hostname)"

# ===== code: exact commits in node-local /tmp =====
phase code
clone() {  # url branch commit dir
  if [[ ! -d "$4/.git" ]] || [[ "$(git -C "$4" rev-parse HEAD)" != "$(git -C "$4" rev-parse "$3^{commit}" 2>/dev/null)" ]]; then
    rm -rf "$4"
    git clone --quiet --branch "$2" --single-branch "$1" "$4"
    git -C "$4" checkout --quiet "$3"
  fi
  git -C "$4" rev-parse HEAD
}
LABELING_HEAD=$(clone "$EXTEDIT_REPO_URL" "$EXTEDIT_BRANCH" "$EXTEDIT_COMMIT" "$WORK/labeling")
V3_HEAD=$(clone "$EXTEDIT_V3_URL" "$EXTEDIT_V3_BRANCH" "$EXTEDIT_V3_COMMIT" "$WORK/v3")
printf '{"labeling":"%s","v3":"%s"}\n' "$LABELING_HEAD" "$V3_HEAD" > "$NODE_RUN/commits.attempt$ATTEMPT.json"
REPO="$WORK/labeling"

# ===== environments, CUDA, model =====
phase environment
export EXTEDIT_V3_REPO="$WORK/v3" UV_CACHE_DIR="${UV_CACHE_DIR:-$WORK/uv-cache}"
bash "$REPO/scripts/extedit_setup_env.sh" vllm "$WORK/env-vllm"
bash "$REPO/scripts/extedit_setup_env.sh" worker "$WORK/env-worker"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$WORK/v3/src" "$WORK/env-worker/bin/python" -B \
  -m samtok_edit21.distributed.cuda_readiness --output "$RUN/bootstrap/node${NODE}-cuda" --expected 8 \
  --timeout 600 --interval 15
phase model
MODEL="${EXTEDIT_LOCAL_MODELS:-$WORK/models}/$(basename "$EXTEDIT_MODEL")"
"$WORK/env-worker/bin/python" - "$EXTEDIT_MODEL" "$MODEL" <<'PY'
import shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in src.rglob("*"):  # copy what is missing or of another size (a copy cut by a restart)
    if f.is_file():
        out = dst / f.relative_to(src)
        if not out.exists() or out.stat().st_size != f.stat().st_size:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, out.with_name(out.name + ".part"))
            out.with_name(out.name + ".part").replace(out)
print(f"model at {dst}")
PY

# ===== 27B server =====
phase server
PORT=8011
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT="$WORK/vllm-cache" \
  PATH="$WORK/env-vllm/bin:$PATH" \
  "$WORK/env-vllm/bin/vllm" serve "$MODEL" --served-model-name qwen38-27b --data-parallel-size 7 \
  --max-model-len 20000 --gpu-memory-utilization 0.93 --max-num-seqs 64 --enable-prefix-caching \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 6}' --host 127.0.0.1 --port "$PORT" \
  > "$NODE_RUN/vllm.attempt$ATTEMPT.log" 2>&1 &
SERVER=$!
for _ in $(seq 1 180); do
  curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q qwen38 && break
  kill -0 "$SERVER" 2>/dev/null || { echo "vLLM server exited; see $NODE_RUN/vllm.attempt$ATTEMPT.log"; false; }
  sleep 10
done
curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q qwen38 || { echo "vLLM server not ready after 30 minutes"; false; }

# ===== pipeline: two passes, the second retries rows whose calls failed in the first =====
for pass in 1 2; do
  phase "pipeline pass $pass"
  setsid env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$WORK/env-worker/bin/python" -B \
    "$REPO/scripts/extedit_pipeline.py" run --run "$NODE_RUN" --whole-shards --node "$NODE/$EXTEDIT_NNODES" \
    --crispedit "$EXTEDIT_CRISPEDIT" --scaleedit "$EXTEDIT_SCALEEDIT" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" --plan-concurrency 168 --audit-concurrency 112 \
    --segment-devices 7 --workers 48 --v3-repo "$WORK/v3" >> "$NODE_RUN/pipeline.log" 2>&1 &
  PIPELINE=$!
  while kill -0 "$PIPELINE" 2>/dev/null; do  # the pipeline needs the server: stop at once if it dies
    if ! kill -0 "$SERVER" 2>/dev/null; then
      echo "vLLM server exited during the pipeline; see $NODE_RUN/vllm.attempt$ATTEMPT.log"
      false
    fi
    sleep 30
  done
  wait "$PIPELINE"
  PIPELINE=""
done

phase finished
"$WORK/env-worker/bin/python" -B "$REPO/scripts/extedit_pipeline.py" status --runs "$NODE_RUN" | tee "$NODE_RUN/status.txt"
cp "$NODE_RUN/final/summary.json" "$NODE_RUN/SUCCESS.json"
stop_children
echo "[$(date -u +%FT%TZ)] node $NODE: done"
