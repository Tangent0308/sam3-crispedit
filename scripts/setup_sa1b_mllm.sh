#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR=${SA1B_MLLM_RUNTIME:-$REPO_DIR/.runtime/sa1b-mllm}
UV_BIN=${UV_BIN:-uv}
[[ ! -e "$RUNTIME_DIR" ]] || { echo "Use a fresh SA1B_MLLM_RUNTIME: $RUNTIME_DIR" >&2; exit 2; }
"$UV_BIN" venv --python 3.12.13 "$RUNTIME_DIR"
"$UV_BIN" pip install --python "$RUNTIME_DIR/bin/python" --no-deps --link-mode=copy \
  --index-url https://bytedpypi.byted.org/simple/ \
  --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
  -r "$REPO_DIR/requirements/labeling-mllm.lock.txt"
"$RUNTIME_DIR/bin/python" -c 'import torch, vllm, transformers; print(torch.__version__, torch.version.cuda, vllm.__version__, transformers.__version__)'
