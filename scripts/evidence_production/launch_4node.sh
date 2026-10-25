#!/usr/bin/env bash
# Evidence production (twins + multi) on Arnold: the same command on every worker
# (deployment/arnold_entry.sh sets EVIDENCE_RUN_ROOT and runs this file).
set -euo pipefail
: "${EVIDENCE_RUN_ROOT:?Set EVIDENCE_RUN_ROOT to the prepared and bundled experiment directory}"
: "${EVIDENCE_RUNTIME_ROOT:=/opt/tiger/sa1b-evidence-production-runtime}"
: "${ARNOLD_WORKER_NUM:?Missing Arnold worker count}"
: "${ARNOLD_WORKER_GPU:?Missing Arnold GPU count}"
: "${ARNOLD_ID:?Missing Arnold rank}"
world=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["world_size"])' "$EVIDENCE_RUN_ROOT/config.json")
[[ "$ARNOLD_WORKER_NUM" == "$world" && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-9]+$ && "$ARNOLD_ID" -lt "$world" ]] \
  || { echo "Requires $world Arnold workers x 8 GPUs (got ${ARNOLD_WORKER_NUM} x ${ARNOLD_WORKER_GPU}, rank ${ARNOLD_ID})" >&2; exit 2; }
unset CUDA_VISIBLE_DEVICES
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=LAZY SA1B_FORCE_DIRECT_VLLM=1
mkdir -p "$EVIDENCE_RUN_ROOT/logs"
exec > >(tee -a "$EVIDENCE_RUN_ROOT/logs/node${ARNOLD_ID}.log") 2>&1
echo "$(date '+%F %T') node ${ARNOLD_ID}/${world} on $(hostname)"
bundle="$EVIDENCE_RUN_ROOT/deployment/code.tar.gz"
expected=$(cat "$EVIDENCE_RUN_ROOT/deployment/code.sha256")
[[ "$(sha256sum "$bundle" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Code bundle checksum mismatch' >&2; exit 2; }
[[ "$(sha256sum "$EVIDENCE_RUN_ROOT/config.json" | cut -d' ' -f1)" == "$(cat "$EVIDENCE_RUN_ROOT/deployment/config.sha256")" ]] \
  || { echo 'Frozen config changed' >&2; exit 2; }
sg2=/mnt/hdfs/sg_byte_ttlive_strategy_llm/user/haobo.yuan/datasets
# Shards fed in advance (prefeed) need no HDFS; feeding more on this worker does.
[[ -r "$sg2/sa1b_tfd_1k/sa_000000_unified_tot1000.index" && -r "$sg2/sa1b_dense_label/annotation/all1000.index" ]] \
  || echo "WARNING: SA-1B HDFS mount not readable ($sg2): this rank can only use shards fed in advance" >&2
repo="/opt/tiger/sa1b-evidence-production-code/$expected"
if [[ ! -d "$repo" ]]; then
  temp="$repo.tmp.$$"; mkdir -p "$temp"; tar -xzf "$bundle" -C "$temp"; mv "$temp" "$repo"
fi
cd "$repo"
export PYTHONPATH="$repo"
if [[ "${EVIDENCE_SKIP_SETUP:-0}" != 1 ]]; then
  GROUP_RUNTIME_ROOT="$EVIDENCE_RUNTIME_ROOT" bash scripts/group_production/setup_runtime.sh
  bash scripts/evidence_production/setup_extras.sh "$EVIDENCE_RUNTIME_ROOT"
fi
export SA1B_OCR_LIB="$EVIDENCE_RUNTIME_ROOT/ocr-lib" SA1B_FEEDER_LIB="$EVIDENCE_RUNTIME_ROOT/feeder-lib"
SA1B_FEEDER_LIB="$SA1B_FEEDER_LIB" "$EVIDENCE_RUNTIME_ROOT/mllm/bin/python" -c \
  'import os,sys; sys.path.append(os.environ["SA1B_FEEDER_LIB"]); import crc32c; from tfd_utils.pb2 import Example; print("feeder libs OK")'
SA1B_OCR_LIB="$SA1B_OCR_LIB" "$EVIDENCE_RUNTIME_ROOT/editor/bin/python" -c \
  'import os,sys; sys.path.append(os.environ["SA1B_OCR_LIB"]); from rapidocr_onnxruntime import RapidOCR; RapidOCR(); print("RapidOCR OK")'
for role in mllm editor; do
  "$EVIDENCE_RUNTIME_ROOT/$role/bin/python" -m sa1b_pipeline.fonts --render
done
exec "$EVIDENCE_RUNTIME_ROOT/mllm/bin/python" scripts/run_evidence_production_4node.py --root "$EVIDENCE_RUN_ROOT" node \
  --rank "$ARNOLD_ID" --mllm-python "$EVIDENCE_RUNTIME_ROOT/mllm/bin/python" \
  --editor-python "$EVIDENCE_RUNTIME_ROOT/editor/bin/python" ${EVIDENCE_NODE_EXTRA_ARGS:-}
