#!/usr/bin/env bash
# Use the same command on all four 8-GPU Arnold workers. No DDP or wandb.
set -euo pipefail
: "${ARNOLD_WORKER_NUM:?Arnold topology missing}"
: "${ARNOLD_WORKER_GPU:?Arnold topology missing}"
: "${ARNOLD_ID:?Arnold rank missing}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'Requires 4 workers x 8 GPUs' >&2; exit 2; }
export SA1B_RUN_ROOT="${SA1B_RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014}"
export SA1B_RUNTIME_ROOT="${SA1B_RUNTIME_ROOT:-/opt/tiger/sa1b-production-runtime}"
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=LAZY
mkdir -p "$SA1B_RUN_ROOT/logs" "$SA1B_RUN_ROOT/control"
# Shared lock prevents two hosts/attempts from executing the same rank.
exec 9>>"$SA1B_RUN_ROOT/control/bootstrap.node$ARNOLD_ID.lock"
flock -n 9 || { echo 'This rank already has an active owner' >&2; exit 2; }
exec > >(tee -a "$SA1B_RUN_ROOT/logs/node$ARNOLD_ID.log") 2>&1
bundle="$SA1B_RUN_ROOT/deployment/code.tar.gz"
expected=$(cat "$SA1B_RUN_ROOT/deployment/code.sha256")
[[ "$(sha256sum "$bundle" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Code bundle hash mismatch' >&2; exit 2; }
repo="/opt/tiger/sa1b-production-code/$expected"
if [[ ! -d "$repo" ]]; then
  temporary="$repo.tmp.$$"
  mkdir -p "$temporary"
  tar -xzf "$bundle" -C "$temporary"
  mv "$temporary" "$repo"
fi
cd "$repo"
export PYTHONPATH="$repo"
# Check shared volumes before installing three large GPU runtimes.
python3 - "$SA1B_RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
c = json.loads((root/'config.json').read_text())
if c.get('source_backend') == 'shared_record_cache_v1':
    from sa1b_pipeline.production_source_cache import validate_cache
    ready = validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl', c.get('source_cache_ready_sha256'))
    print(f"Shared source cache verified: {ready['sources']} images, {ready['records']} records; HDFS mount not required", flush=True)
    paths = []
else:
    paths = [Path(c['source_root'])/'sa1b_tfd_1k/sa_000000_unified_tot1000.index',
             Path(c['source_root'])/'sa1b_dense_label/annotation/all1000.index']
paths += [Path(c['planner_model'])/'config.json', Path(c['editor_model'])/'model_index.json',
         Path(c['audit_model'])/'config.json']
for path in paths:
    with path.open('rb') as f:
        assert f.read(1), f'Empty required input: {path}'
print('Shared inputs and model configurations are accessible', flush=True)
PY
if [[ "${SA1B_SKIP_SETUP:-0}" != 1 ]]; then bash scripts/production/setup_runtime.sh; fi
exec "$SA1B_RUNTIME_ROOT/control/bin/python" -m scripts.run_production_4node --root "$SA1B_RUN_ROOT" run \
  --rank "$ARNOLD_ID" \
  --planner-python "$SA1B_RUNTIME_ROOT/planner/bin/python" \
  --editor-python "$SA1B_RUNTIME_ROOT/editor/bin/python" \
  --audit-python "$SA1B_RUNTIME_ROOT/audit/bin/python"
