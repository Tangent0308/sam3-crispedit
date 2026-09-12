#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${1:-"$REPO_DIR/.venv-vllm"}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
TORCH_BACKEND=${TORCH_BACKEND:-cu129}
VLLM_WHEEL_URL=${VLLM_WHEEL_URL:-https://wheels.vllm.ai/2a02f6efe319c885e3ccbcecde402e0028f9ec1e/vllm-0.28.1rc1.dev628%2Bg2a02f6efe.cu129-cp38-abi3-manylinux_2_28_x86_64.whl}

command -v uv >/dev/null || {
  echo "uv is required; install it before creating the vLLM environment." >&2
  exit 1
}

uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
uv pip install \
  --python "$VENV_DIR/bin/python" \
  --torch-backend "$TORCH_BACKEND" \
  "$VLLM_WHEEL_URL" \
  pyarrow \
  Pillow

"$VENV_DIR/bin/python" -c \
  'import torch, vllm; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
