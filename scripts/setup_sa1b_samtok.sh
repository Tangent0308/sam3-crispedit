#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMTOK_ROOT=${SAMTOK_REPO_ROOT:-$REPO_DIR/.runtime/Sa2VA}
RUNTIME_DIR=${SA1B_SAMTOK_RUNTIME:-$REPO_DIR/.runtime/sa1b-samtok}
UV_BIN=${UV_BIN:-uv}
SAMTOK_COMMIT=${SAMTOK_COMMIT:-b6fed1a7928ad4adfa124f1fe79031c1f58ec7a1}

if [[ ! -d "$SAMTOK_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$SAMTOK_ROOT")"
  git clone https://github.com/bytedance/Sa2VA.git "$SAMTOK_ROOT"
fi
git -C "$SAMTOK_ROOT" fetch --depth 1 origin "$SAMTOK_COMMIT"
git -C "$SAMTOK_ROOT" checkout --detach "$SAMTOK_COMMIT"

if [[ ! -e "$RUNTIME_DIR" ]]; then
  "$UV_BIN" venv --python 3.12.13 "$RUNTIME_DIR"
  "$UV_BIN" pip install --python "$RUNTIME_DIR/bin/python" --link-mode=copy \
    --index-url https://bytedpypi.byted.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
    -r "$REPO_DIR/requirements/labeling-samtok.txt"
fi

echo "SAMTOK_REPO_ROOT=$SAMTOK_ROOT"
echo "SA1B_SAMTOK_RUNTIME=$RUNTIME_DIR"
echo "SAMTOK_COMMIT=$(git -C "$SAMTOK_ROOT" rev-parse HEAD)"
"$RUNTIME_DIR/bin/python" -c 'import torch, transformers, mmengine, xtuner; print(torch.__version__, transformers.__version__, mmengine.__version__, xtuner.__version__)'
