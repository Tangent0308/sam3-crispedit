#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${SA1B_RUNTIME_ROOT:=/opt/tiger/sa1b-production-runtime}"
: "${UV_BIN:=/home/tiger/.local/bin/uv}"
if [[ ! -x "$UV_BIN" ]]; then
  python3 -m pip install --user --index-url https://bytedpypi.byted.org/simple/ 'uv==0.11.32'
  UV_BIN="$(python3 -c 'import site; print(site.getuserbase())')/bin/uv"
fi
mkdir -p "$SA1B_RUNTIME_ROOT"
exec 8>"$SA1B_RUNTIME_ROOT/setup.lock"
flock -n 8 || { echo 'Another runtime setup is active on this host' >&2; exit 2; }
export UV_HTTP_TIMEOUT=600
for role in planner editor audit; do
  runtime="$SA1B_RUNTIME_ROOT/$role"
  lockfile="$repo/requirements/production/$role.lock.txt"
  expected=$(sha256sum "$lockfile" | cut -d' ' -f1)
  if [[ -f "$runtime/ready.sha256" ]]; then
    [[ "$(cat "$runtime/ready.sha256")" == "$expected" ]] || { echo "Runtime $role has different pins; use a new SA1B_RUNTIME_ROOT" >&2; exit 2; }
    "$runtime/bin/python" "$repo/scripts/production/check_runtime.py" --role "$role"
    continue
  fi
  if [[ ! -x "$runtime/bin/python" ]]; then "$UV_BIN" venv --python 3.12.13 "$runtime"; fi
  # Sync exactly the lock: remove stale CUDA 13 packages from failed installs.
  "$UV_BIN" pip sync --python "$runtime/bin/python" --link-mode=copy \
    --index-url https://bytedpypi.byted.org/simple/ --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match "$lockfile"
  if [[ "$role" == editor ]]; then
    omni="$SA1B_RUNTIME_ROOT/vllm-omni"
    [[ -d "$omni/.git" ]] || git clone --filter=blob:none https://github.com/vllm-project/vllm-omni.git "$omni"
    git -C "$omni" checkout --detach 44ea27c8094095bbffd88fa3befdfaa55ba4bc50
    "$UV_BIN" pip install --python "$runtime/bin/python" --no-deps -e "$omni"
  fi
  "$runtime/bin/python" "$repo/scripts/production/check_runtime.py" --role "$role"
  printf '%s\n' "$expected" > "$runtime/ready.sha256"
done
control="$SA1B_RUNTIME_ROOT/control"
[[ -x "$control/bin/python" ]] || "$UV_BIN" venv --python 3.12.13 "$control"
"$UV_BIN" pip install --python "$control/bin/python" 'tfd-utils==1.3.0' 'pycocotools==2.0.11' 'numpy==2.2.6' 'pillow==11.3.0'
