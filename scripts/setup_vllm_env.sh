#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
VENV_DIR=${1:-"${REPO_ROOT}/.venv-crispedit-vllm"}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}
TORCH_BACKEND=${TORCH_BACKEND:-cu129}
VLLM_WHEEL_URL=${VLLM_WHEEL_URL:-https://wheels.vllm.ai/2a02f6efe319c885e3ccbcecde402e0028f9ec1e/vllm-0.28.1rc1.dev628%2Bg2a02f6efe.cu129-cp38-abi3-manylinux_2_28_x86_64.whl}

command -v uv >/dev/null 2>&1 || {
  echo "uv is required; install it before creating the CrispEdit vLLM environment." >&2
  exit 1
}

if [[ -e "${VENV_DIR}" ]]; then
  echo "Refusing to reuse an existing environment: ${VENV_DIR}" >&2
  echo "Choose a new path or remove the old environment explicitly." >&2
  exit 1
fi

echo "[crispedit-vllm] Creating ${VENV_DIR}"
uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"

echo "[crispedit-vllm] Installing the production vLLM wheel and grounding dependencies"
uv pip install \
  --python "${VENV_DIR}/bin/python" \
  --torch-backend "${TORCH_BACKEND}" \
  "${VLLM_WHEEL_URL}" \
  pyarrow \
  Pillow \
  tqdm \
  ninja
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e "${REPO_ROOT}"

echo "[crispedit-vllm] Verifying imports and Qwen3.5 support"
"${VENV_DIR}/bin/python" - <<'PY'
import importlib
import torch
from vllm.model_executor.models import ModelRegistry

for name in (
    "vllm",
    "transformers",
    "pyarrow",
    "PIL",
    "tqdm",
    "crispedit.mask.grounding_runner",
):
    module = importlib.import_module(name)
    print("IMPORTED", name, getattr(module, "__version__", ""))

architecture = "Qwen3_5MoeForConditionalGeneration"
if architecture not in ModelRegistry.get_supported_archs():
    raise RuntimeError(f"vLLM does not support {architecture}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the CrispEdit vLLM environment")
print("TORCH", torch.__version__, "CUDA", torch.version.cuda)
print("CUDA_DEVICES", torch.cuda.device_count())
print("SUPPORTED", architecture)
PY

cat <<EOF

[crispedit-vllm] Environment ready.

Run grounding with:
  "${VENV_DIR}/bin/python" -u crispedit_mllm_grounding.py --inference-backend vllm ...
EOF
