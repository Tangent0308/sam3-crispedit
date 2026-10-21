#!/bin/bash
# One node of an extedit production run (docs/EXTEDIT_PROTOCOL2.md section 4.2).
#   27B server: data parallel on GPUs 0-6 (prefix caching on); SAM3 + SAMTok worker on GPU 7;
#   pipeline:   scripts/extedit_pipeline.py run --whole-shards --node K/N, every stage streaming.
# usage: extedit_node.sh <run root> <node index K> <node count N> [extra arguments of "run", e.g. quotas]
# environment:
#   WORKER_PY  python of the worker environment (requirements/extedit-worker.txt)
#   VLLM       vllm executable of the server environment (requirements/extedit-vllm.txt)
#   MODEL      Qwen3.8-27B checkpoint (copied to local disk first when LOCAL_MODEL is set)
#   V3_REPO    SAMTokEdit v3 checkout (SAMTok codec)
set -euo pipefail
ROOT=${1:?run root}; K=${2:?node index}; N=${3:?node count}; shift 3
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORKER_PY=${WORKER_PY:-python3}
VLLM=${VLLM:-vllm}
MODEL=${MODEL:-/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B}
V3_REPO=${V3_REPO:-/opt/tiger/tanyue/samtok_edit_qwen-image-2.1-v3}
PORT=${PORT:-8011}
SERVER_GPUS=${SERVER_GPUS:-0,1,2,3,4,5,6}
SEGMENT_GPUS=${SEGMENT_GPUS:-7}
RUN="$ROOT/node$K"
mkdir -p "$RUN/logs"

if [ -n "${LOCAL_MODEL:-}" ]; then  # reading 52 GB once from local disk beats every engine reading the NAS
  mkdir -p "$LOCAL_MODEL"
  cp -rn "$MODEL/." "$LOCAL_MODEL/"
  MODEL=$LOCAL_MODEL
fi
DP=$(echo "$SERVER_GPUS" | tr ',' '\n' | wc -l)
# torch launcher variables would make the vLLM engines join a foreign process group
env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT \
  CUDA_VISIBLE_DEVICES="$SERVER_GPUS" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_CACHE_ROOT="$RUN/vllm-cache" \
  "$VLLM" serve "$MODEL" --served-model-name qwen38-27b --data-parallel-size "$DP" --max-model-len 20000 \
  --gpu-memory-utilization 0.93 --max-num-seqs 64 --enable-prefix-caching --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 6}' --host 127.0.0.1 --port "$PORT" > "$RUN/logs/vllm.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -q qwen38; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM server exited; see $RUN/logs/vllm.log"; exit 1; }
  sleep 10
done
echo "node $K/$N: server ready $(date -u +%FT%TZ)"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$WORKER_PY" -B "$REPO/scripts/extedit_pipeline.py" run --run "$RUN" \
  --whole-shards --node "$K/$N" --url "http://127.0.0.1:$PORT/v1/chat/completions" \
  --plan-concurrency $((24 * DP)) --audit-concurrency $((16 * DP)) --segment-devices "$SEGMENT_GPUS" \
  --v3-repo "$V3_REPO" "$@"
echo "node $K/$N: done $(date -u +%FT%TZ)"
