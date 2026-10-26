#!/usr/bin/env bash
# Single-target production (single-target v2.1) on Arnold: the same command on every worker
# (deployment/arnold_entry.sh sets SINGLE_RUN_ROOT and runs this file). Same environments and setup as the
# twins/multi production (scripts/evidence_production/launch_4node.sh).
set -euo pipefail
: "${SINGLE_RUN_ROOT:?Set SINGLE_RUN_ROOT to the prepared and bundled experiment directory}"
: "${SINGLE_RUNTIME_ROOT:=/opt/tiger/sa1b-evidence-production-runtime}"
: "${ARNOLD_WORKER_NUM:?Missing Arnold worker count}"
: "${ARNOLD_WORKER_GPU:?Missing Arnold GPU count}"
: "${ARNOLD_ID:?Missing Arnold rank}"
world=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["world_size"])' "$SINGLE_RUN_ROOT/config.json")
[[ "$ARNOLD_WORKER_NUM" == "$world" && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$world" ]] \
  || { echo "Requires $world Arnold workers x 8 GPUs (got ${ARNOLD_WORKER_NUM} x ${ARNOLD_WORKER_GPU}, rank ${ARNOLD_ID})" >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=LAZY SA1B_FORCE_DIRECT_VLLM=1
mkdir -p "$SINGLE_RUN_ROOT/logs"
exec > >(tee -a "$SINGLE_RUN_ROOT/logs/node${ARNOLD_ID}.log") 2>&1
echo "$(date '+%F %T') node ${ARNOLD_ID}/${world} on $(hostname)"
bundle="$SINGLE_RUN_ROOT/deployment/code.tar.gz"
expected=$(cat "$SINGLE_RUN_ROOT/deployment/code.sha256")
[[ "$(sha256sum "$bundle" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Code bundle checksum mismatch' >&2; exit 2; }
[[ "$(sha256sum "$SINGLE_RUN_ROOT/config.json" | cut -d' ' -f1)" == "$(cat "$SINGLE_RUN_ROOT/deployment/config.sha256")" ]] \
  || { echo 'Frozen config changed' >&2; exit 2; }
sg2=/mnt/hdfs/sg_byte_ttlive_strategy_llm/user/haobo.yuan/datasets
# Shards prepared in advance (warm-pool) need no HDFS; feeding more on this worker does.
[[ -r "$sg2/sa1b_tfd_1k/sa_000000_unified_tot1000.index" && -r "$sg2/sa1b_dense_label/annotation/all1000.index" ]] \
  || echo "WARNING: SA-1B HDFS mount not readable ($sg2): this rank can only use shards prepared in advance" >&2
repo="/opt/tiger/sa1b-single-production-code/$expected"
if [[ ! -d "$repo" ]]; then
  temp="$repo.tmp.$$"; mkdir -p "$temp"; tar -xzf "$bundle" -C "$temp"; mv "$temp" "$repo"
fi
cd "$repo"
export PYTHONPATH="$repo"
if [[ "${SINGLE_SKIP_SETUP:-0}" != 1 ]]; then
  GROUP_RUNTIME_ROOT="$SINGLE_RUNTIME_ROOT" bash scripts/group_production/setup_runtime.sh
  bash scripts/evidence_production/setup_extras.sh "$SINGLE_RUNTIME_ROOT"
fi
export SA1B_OCR_LIB="$SINGLE_RUNTIME_ROOT/ocr-lib" SA1B_FEEDER_LIB="$SINGLE_RUNTIME_ROOT/feeder-lib"
SA1B_FEEDER_LIB="$SA1B_FEEDER_LIB" "$SINGLE_RUNTIME_ROOT/mllm/bin/python" -c \
  'import os,sys; sys.path.append(os.environ["SA1B_FEEDER_LIB"]); import crc32c; from tfd_utils.pb2 import Example; print("feeder libs OK")'
SA1B_OCR_LIB="$SA1B_OCR_LIB" "$SINGLE_RUNTIME_ROOT/editor/bin/python" -c \
  'import os,sys; sys.path.append(os.environ["SA1B_OCR_LIB"]); from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print("RapidOCR OK")'
for role in mllm editor; do
  "$SINGLE_RUNTIME_ROOT/$role/bin/python" -m sa1b_pipeline.fonts --render
done
exec "$SINGLE_RUNTIME_ROOT/mllm/bin/python" scripts/run_single_production_4node.py --root "$SINGLE_RUN_ROOT" node \
  --rank "$ARNOLD_ID" --mllm-python "$SINGLE_RUNTIME_ROOT/mllm/bin/python" \
  --editor-python "$SINGLE_RUNTIME_ROOT/editor/bin/python" ${SINGLE_NODE_EXTRA_ARGS:-}
