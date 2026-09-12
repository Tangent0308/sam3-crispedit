#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-3.12}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-scaleedit-vllm}"
VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.15.1}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCHCODEC_VERSION="${TORCHCODEC_VERSION:-0.16.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
VLLM_WHEEL_URL="${VLLM_WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it before creating the ScaleEdit vLLM environment." >&2
  exit 1
fi

if [[ -e "${VENV_DIR}" ]]; then
  echo "Refusing to reuse an existing environment: ${VENV_DIR}" >&2
  echo "Choose a new VENV_DIR or remove the old environment explicitly." >&2
  exit 1
fi

echo "[scaleedit-vllm] Creating a fresh environment at ${VENV_DIR}"
uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"

echo "[scaleedit-vllm] Installing the CUDA 12.9 PyTorch runtime"
uv pip install \
  --python "${VENV_DIR}/bin/python" \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  "torchcodec==${TORCHCODEC_VERSION}"

echo "[scaleedit-vllm] Installing the CUDA 12.9 vLLM wheel and ScaleEdit runtime"
uv pip install \
  --python "${VENV_DIR}/bin/python" \
  "${VLLM_WHEEL_URL}" \
  "transformers==${TRANSFORMERS_VERSION}" \
  "numpy>=1.26,<2" \
  "timm>=1.0.17" \
  "ftfy==6.1.1" \
  regex \
  "iopath>=0.1.10" \
  typing_extensions \
  huggingface_hub \
  accelerate \
  einops \
  pycocotools \
  opencv-python-headless \
  pyarrow \
  Pillow \
  tqdm \
  pytest
uv pip install --python "${VENV_DIR}/bin/python" --no-deps -e .

echo "[scaleedit-vllm] Verifying imports and Qwen3.5 support"
"${VENV_DIR}/bin/python" - <<'PY'
import importlib
import torch
from vllm.model_executor.models import ModelRegistry

for name in (
    "vllm",
    "transformers",
    "cv2",
    "numpy",
    "pyarrow",
    "PIL",
    "timm",
    "ftfy",
    "iopath",
    "einops",
    "pycocotools",
    "sam3",
    "scaleedit",
):
    module = importlib.import_module(name)
    print("IMPORTED", name, getattr(module, "__version__", ""))

architecture = "Qwen3_5MoeForConditionalGeneration"
if architecture not in ModelRegistry.get_supported_archs():
    raise RuntimeError(f"vLLM does not support {architecture}")
if torch.version.cuda != "12.9":
    raise RuntimeError(f"expected a CUDA 12.9 PyTorch build, got {torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the ScaleEdit vLLM environment")
print("TORCH_CUDA", torch.version.cuda)
print("CUDA_DEVICES", torch.cuda.device_count())
print("SUPPORTED", architecture)
PY

cat <<EOF

[scaleedit-vllm] Environment ready.

Run grounding with:
  "${VENV_DIR}/bin/python" -u scaleedit_mllm_grounding.py --inference-backend vllm ...
EOF
