#!/usr/bin/env bash
set -euo pipefail
: "${GROUP_RUN_ROOT:?Set GROUP_RUN_ROOT to the prepared grouped-data directory}"
: "${GROUP_RUNTIME_ROOT:=/opt/tiger/sa1b-group-production-runtime}"
: "${ARNOLD_WORKER_NUM:?Missing Arnold worker count}"
: "${ARNOLD_WORKER_GPU:?Missing Arnold GPU count}"
: "${ARNOLD_ID:?Missing Arnold rank}"
[[ "$ARNOLD_WORKER_NUM" == 4 && "$ARNOLD_WORKER_GPU" == 8 && "$ARNOLD_ID" =~ ^[0-3]$ ]] || { echo 'Requires actual 4 workers x 8 GPUs, ranks 0..3' >&2; exit 2; }
export GROUP_RUN_ROOT GROUP_RUNTIME_ROOT
export WANDB_MODE=disabled WANDB_DISABLED=true PYTHONUNBUFFERED=1 CUDA_MODULE_LOADING=LAZY
mkdir -p "$GROUP_RUN_ROOT/logs"
exec > >(tee -a "$GROUP_RUN_ROOT/logs/node${ARNOLD_ID}.log") 2>&1
bundle="$GROUP_RUN_ROOT/deployment/code.tar.gz"
expected=$(cat "$GROUP_RUN_ROOT/deployment/code.sha256")
[[ "$(sha256sum "$bundle" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Code bundle checksum mismatch' >&2; exit 2; }
[[ "$(sha256sum "$GROUP_RUN_ROOT/config.json" | cut -d' ' -f1)" == "$(cat "$GROUP_RUN_ROOT/deployment/config.sha256")" ]] || { echo 'Frozen config changed' >&2; exit 2; }
repo="/opt/tiger/sa1b-group-production-code/$expected"
if [[ ! -d "$repo" ]]; then
  temp="$repo.tmp.$$"; mkdir -p "$temp";tar -xzf "$bundle" -C "$temp";mv "$temp" "$repo"
fi
cd "$repo"
export PYTHONPATH="$repo"
python3 - "$GROUP_RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);c=json.loads((root/'config.json').read_text())
assert (root/'inputs/ready.json').is_file(), 'source preparation incomplete'
import hashlib
frozen=json.loads((root/'deployment/manifest.json').read_text())
for name,key in [('inputs/ready.json','input_ready_sha256'),('deployment/models.json','model_inventory_sha256')]:
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==frozen[key], f'frozen input changed: {name}'
for name,inventory in json.loads((root/'deployment/models.json').read_text()).items():
 for filename,meta in inventory.items():
  stat=(Path(name)/filename).stat()
  assert stat.st_size==meta['size'] and stat.st_mtime_ns==meta['mtime_ns'], f'frozen model changed: {name}/{filename}'
for key in ('planner_model','blind_model','audit_model','editor_model','site_model'):
 p=Path(c[key]);assert p.is_dir(), f'missing shared model: {p}'
from sa1b_pipeline.fonts import verify_fonts
verify_fonts()
print('Bundled fonts OK (SHA-256 verified)')
print('Shared inputs/models accessible; protocol 2.0, CUDA 12.9, 40 steps')
PY
if [[ "${GROUP_SKIP_SETUP:-0}" != 1 ]]; then bash scripts/group_production/setup_runtime.sh; fi
for role in mllm editor; do
  "$GROUP_RUNTIME_ROOT/$role/bin/python" -m sa1b_pipeline.fonts --render
done
exec "$GROUP_RUNTIME_ROOT/mllm/bin/python" scripts/run_group_production_4node.py --root "$GROUP_RUN_ROOT" node \
  --rank "$ARNOLD_ID" --mllm-python "$GROUP_RUNTIME_ROOT/mllm/bin/python" \
  --editor-python "$GROUP_RUNTIME_ROOT/editor/bin/python"
