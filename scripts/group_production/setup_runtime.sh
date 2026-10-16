#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GROUP_RUNTIME_ROOT:=/opt/tiger/sa1b-group-production-runtime}"
: "${UV_BIN:=/home/tiger/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
  python3 -m pip install --user --index-url https://bytedpypi.byted.org/simple/ uv==0.11.32
  UV_BIN="$(python3 -c 'import site; print(site.getuserbase())')/bin/uv"
fi
mkdir -p "$GROUP_RUNTIME_ROOT"
exec 8>"$GROUP_RUNTIME_ROOT/setup.lock"
flock -n 8 || { echo 'Runtime setup already active on this host' >&2; exit 2; }
export UV_HTTP_TIMEOUT=600
for role in mllm editor; do
  envdir="$GROUP_RUNTIME_ROOT/$role"
  lock="$repo/requirements/group_production/$role.lock.txt"
  expected=$(sha256sum "$lock" | cut -d' ' -f1)
  if [[ ! -x "$envdir/bin/python" ]]; then "$UV_BIN" venv --python 3.12.13 "$envdir"; fi
  if [[ ! -f "$envdir/ready.sha256" || "$(cat "$envdir/ready.sha256")" != "$expected" ]]; then
    "$UV_BIN" pip sync --python "$envdir/bin/python" --link-mode=copy \
      --index-url https://bytedpypi.byted.org/simple/ --extra-index-url https://download.pytorch.org/whl/cu129 \
      --index-strategy unsafe-best-match "$lock"
    "$UV_BIN" pip install --python "$envdir/bin/python" --no-deps --reinstall \
      --index-url https://bytedpypi.byted.org/simple/ "$(sed -n '/^opencv-python-headless==/p' "$lock")"
    if [[ "$role" == editor ]]; then
      omni="$GROUP_RUNTIME_ROOT/vllm-omni"
      [[ -d "$omni/.git" ]] || git clone --filter=blob:none https://github.com/vllm-project/vllm-omni.git "$omni"
      git -C "$omni" checkout --detach 44ea27c8094095bbffd88fa3befdfaa55ba4bc50
      "$UV_BIN" pip install --python "$envdir/bin/python" --no-deps -e "$omni"
    fi
  fi
  check_role=planner
  [[ "$role" != editor ]] || check_role=editor
  PYTHONPATH="$repo" "$envdir/bin/python" "$repo/scripts/production/check_runtime.py" --role "$check_role"
  "$envdir/bin/python" -c 'from transformers import Sam3Model, Sam3VideoModel, Sam3Processor; import scipy, pycocotools; print("SAM3 and geometry imports OK")'
  printf '%s\n' "$expected" > "$envdir/ready.sha256"
done
