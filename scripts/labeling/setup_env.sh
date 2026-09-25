#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
: "${SAMTOK_RUNTIME_ROOT:=$REPO_DIR/.runtime}"
: "${SAMTOK_PACKAGE_INDEX:=https://bytedpypi.byted.org/simple/}"
: "${UV_BIN:=uv}"
export UV_LINK_MODE=copy
mkdir -p "$SAMTOK_RUNTIME_ROOT"
SAMTOK_RUNTIME_ROOT="$(cd "$SAMTOK_RUNTIME_ROOT" && pwd)"
export SAMTOK_RUNTIME_ROOT
SAM_REV=fff5ca124cf2551dd73c0de2af9c64bdadeea0b3
OMNI_REV=44ea27c8094095bbffd88fa3befdfaa55ba4bc50

clone_exact() {
  local url="$1" revision="$2" destination="$3"
  if [[ ! -d "$destination/.git" ]]; then
    [[ ! -e "$destination" ]] || { echo "Unexpected dependency directory: $destination" >&2; return 1; }
    git clone --no-checkout "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || { echo "Dependency revision mismatch" >&2; return 1; }
  [[ -z "$(git -C "$destination" status --porcelain)" ]] || { echo "Dirty dependency: $destination" >&2; return 1; }
}
clone_exact https://github.com/Tangent0308/sam3-crispedit.git "$SAM_REV" "$SAMTOK_RUNTIME_ROOT/sam3-source"
clone_exact https://github.com/vllm-project/vllm-omni.git "$OMNI_REV" "$SAMTOK_RUNTIME_ROOT/omni-source"

for role in sam mllm editor; do
  envdir="$SAMTOK_RUNTIME_ROOT/$role"
  [[ ! -e "$envdir" ]] || { echo "Environment exists; use a fresh runtime root: $envdir" >&2; exit 1; }
  "$UV_BIN" venv --python 3.12.13 "$envdir"
  "$UV_BIN" pip install --python "$envdir/bin/python" --no-deps \
    --index-url "$SAMTOK_PACKAGE_INDEX" --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    -r "requirements/labeling-$role.lock.txt"
  "$envdir/bin/python" -c 'from synthesis_pipeline.check_labeling_environment import check_opencv; print(check_opencv())'
done
# Preserve the tested SAM source import without resolving its old numpy<2 metadata.
# Omni's GUI OpenCV dependency shares cv2 with headless; --no-deps is intentional.
# Its image APIs are provided by the pinned headless wheel; no GUI/video client is used.
# Preserve the exact Omni official implementation; regional behavior lives in this repo.
"$UV_BIN" pip install --python "$SAMTOK_RUNTIME_ROOT/editor/bin/python" \
  --no-deps --index-url "$SAMTOK_PACKAGE_INDEX" -e "$SAMTOK_RUNTIME_ROOT/omni-source"
export SAMTOK_MLLM_PYTHON="$SAMTOK_RUNTIME_ROOT/mllm/bin/python"
export SAMTOK_EDITOR_PYTHON="$SAMTOK_RUNTIME_ROOT/editor/bin/python"
export SAMTOK_SAM_PYTHON="$SAMTOK_RUNTIME_ROOT/sam/bin/python"
export SAMTOK_SAM3_SOURCE="$SAMTOK_RUNTIME_ROOT/sam3-source"
python3 -m synthesis_pipeline.check_labeling_environment --out "$SAMTOK_RUNTIME_ROOT/environment.json"
echo "Environment installed and import/CUDA checks passed: $SAMTOK_RUNTIME_ROOT"
