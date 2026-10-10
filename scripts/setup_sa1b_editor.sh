#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR=${SA1B_EDITOR_RUNTIME:-$REPO_DIR/.runtime/sa1b-editor}
UV_BIN=${UV_BIN:-uv}
OMNI_DIR=${VLLM_OMNI_SOURCE:-$RUNTIME_DIR/vllm-omni}
OMNI_REV=${VLLM_OMNI_REV:-44ea27c8094095bbffd88fa3befdfaa55ba4bc50}
[[ ! -e "$RUNTIME_DIR" ]] || { echo "Use a fresh SA1B_EDITOR_RUNTIME: $RUNTIME_DIR" >&2; exit 2; }
"$UV_BIN" venv --python 3.12.13 "$RUNTIME_DIR"
"$UV_BIN" pip install --python "$RUNTIME_DIR/bin/python" --no-deps --link-mode=copy \
  --index-url https://bytedpypi.byted.org/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
  -r "$REPO_DIR/requirements/labeling-editor.lock.txt"
git clone --filter=blob:none https://github.com/vllm-project/vllm-omni.git "$OMNI_DIR"
git -C "$OMNI_DIR" checkout --detach "$OMNI_REV"
"$UV_BIN" pip install --python "$RUNTIME_DIR/bin/python" --no-deps --link-mode=copy -e "$OMNI_DIR" -e "$REPO_DIR"
cat <<EOF
Editor runtime ready: $RUNTIME_DIR
export EDITOR_PYTHON=$RUNTIME_DIR/bin/python
export EDITOR_RUNTIME_ROOT=$RUNTIME_DIR
export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA
EOF
