#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-sam3-crispedit}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
QWEN_MODEL_PATH="${CRISPEDIT_QWEN_MODEL_PATH:-/mnt/bn/strategy-mllm-train/common/models/Qwen3-VL-8B-Instruct}"
SAM3_CHECKPOINT_PATH="${CRISPEDIT_SAM3_CHECKPOINT_PATH:-}"
PREFETCH_SAM3=0
RECREATE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_env.sh [options]

Options:
  --python-bin PATH            Python executable to use for venv creation
  --venv-dir PATH              Virtualenv directory to create/use
  --torch-index-url URL        PyTorch wheel index URL (default: cu128 wheels)
  --torch-version VERSION      Torch version (default: 2.8.0)
  --torchvision-version VER    Torchvision version (default: 0.23.0)
  --qwen-model-path PATH       Local Qwen3-VL model path to validate
  --sam3-checkpoint-path PATH  Optional local SAM3 checkpoint path to validate
  --prefetch-sam3              Try a lightweight HF checkpoint prefetch check
  --recreate                   Remove and recreate the virtualenv
  -h, --help                   Show this help

Environment overrides:
  PYTHON_BIN, VENV_DIR, TORCH_INDEX_URL, TORCH_VERSION, TORCHVISION_VERSION,
  CRISPEDIT_QWEN_MODEL_PATH, CRISPEDIT_SAM3_CHECKPOINT_PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-version)
      TORCH_VERSION="$2"
      shift 2
      ;;
    --torchvision-version)
      TORCHVISION_VERSION="$2"
      shift 2
      ;;
    --qwen-model-path)
      QWEN_MODEL_PATH="$2"
      shift 2
      ;;
    --sam3-checkpoint-path)
      SAM3_CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --prefetch-sam3)
      PREFETCH_SAM3=1
      shift
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

log() {
  printf '[setup_env] %s\n' "$*"
}

warn() {
  printf '[setup_env][warn] %s\n' "$*" >&2
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_cmd "$PYTHON_BIN"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  UV_BIN=""
fi

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "pyproject.toml not found under ${REPO_ROOT}; run this from the sam3 repo." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  log "Detected nvidia-smi: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd '; ' -)"
else
  warn "nvidia-smi not found; GPU runtime may be unavailable."
fi

if [[ "$RECREATE" == "1" && -d "$VENV_DIR" ]]; then
  log "Removing existing virtualenv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtualenv at $VENV_DIR"
  if "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    :
  elif [[ -n "$UV_BIN" ]]; then
    warn "python -m venv failed; falling back to uv venv"
    rm -rf "$VENV_DIR"
    "$UV_BIN" venv --seed --python "$PYTHON_BIN" "$VENV_DIR"
  else
    echo "Failed to create virtualenv with $PYTHON_BIN -m venv, and uv is not available for fallback." >&2
    exit 1
  fi
else
  log "Reusing existing virtualenv at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_PYTHON -m pip"

log "Upgrading pip/setuptools/wheel (pin setuptools<81 for pkg_resources compatibility)"
$PIP install --upgrade pip 'setuptools<81' wheel

log "Installing CUDA-enabled torch/torchvision"
$PIP install \
  --index-url "$TORCH_INDEX_URL" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}"

log "Installing editable sam3 package with CrispEdit runtime extra"
$PIP install -e '.[crispedit]'

log "Running import verification"
"$VENV_PYTHON" - <<'PY'
import importlib
import os
from pathlib import Path

mods = [
    'torch',
    'torchvision',
    'transformers',
    'pyarrow',
    'cv2',
    'PIL',
    'einops',
    'pycocotools',
    'crispedit.prefilter.runner',
    'crispedit.mask.grounding_runner',
    'crispedit.mask.runner',
    'crispedit.legacy.pipeline',
    'sam3',
]
for name in mods:
    importlib.import_module(name)
    print('IMPORTED', name)

import torch
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
print('TORCH_VERSION', torch.__version__)
print('TORCHVISION_VERSION', importlib.import_module('torchvision').__version__)
print('CUDA_VERSION', torch.version.cuda)
print('CUDA_AVAILABLE', torch.cuda.is_available())
print('CUDA_DEVICE_COUNT', torch.cuda.device_count())
print('IMPORTED build_sam3_image_model', build_sam3_image_model.__name__)
print('IMPORTED Sam3Processor', Sam3Processor.__name__)
PY

if [[ -n "$QWEN_MODEL_PATH" ]]; then
  if [[ -d "$QWEN_MODEL_PATH" ]]; then
    log "Found Qwen model path: $QWEN_MODEL_PATH"
  else
    warn "Qwen model path does not exist: $QWEN_MODEL_PATH"
  fi
fi

if [[ -n "$SAM3_CHECKPOINT_PATH" ]]; then
  if [[ -f "$SAM3_CHECKPOINT_PATH" ]]; then
    log "Found local SAM3 checkpoint: $SAM3_CHECKPOINT_PATH"
  else
    warn "Local SAM3 checkpoint path does not exist: $SAM3_CHECKPOINT_PATH"
  fi
else
  if command -v huggingface-cli >/dev/null 2>&1; then
    if huggingface-cli whoami >/dev/null 2>&1; then
      log "Hugging Face auth detected; SAM3 can use HF download if checkpoint-path is omitted."
    else
      warn "Hugging Face auth not detected. Run 'huggingface-cli login' if you want HF checkpoint download."
    fi
  else
    warn "huggingface-cli not found. Install/login if you need HF checkpoint download."
  fi
fi

if [[ "$PREFETCH_SAM3" == "1" ]]; then
  log "Running lightweight SAM3 HF prefetch check"
  "$VENV_PYTHON" - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt')
print('SAM3_PREFETCHED', path)
PY
fi

cat <<EOF

[setup_env] Environment bootstrap complete.

Activate:
  source "$VENV_DIR/bin/activate"

Direct interpreter:
  "$VENV_PYTHON"

Recommended environment variables:
  export CRISPEDIT_QWEN_MODEL_PATH="$QWEN_MODEL_PATH"
EOF

if [[ -n "$SAM3_CHECKPOINT_PATH" ]]; then
  cat <<EOF
  export CRISPEDIT_SAM3_CHECKPOINT_PATH="$SAM3_CHECKPOINT_PATH"
EOF
else
  cat <<'EOF'
  # Optional: export CRISPEDIT_SAM3_CHECKPOINT_PATH=/path/to/sam3_checkpoint.pt
EOF
fi

cat <<'EOF'

Next steps:
  1) Verify HF auth if needed:
       huggingface-cli whoami
  2) Run the smoke prefilter:
       python crispedit_mllm_prefilter.py --help
  3) Inspect the grounding and mask stages:
       python crispedit_mllm_grounding.py --help
       python crispedit_grounded_mask_runner.py --help

Production docs:
  - README.md
  - docs/CRISPEDIT_PREFILTER.md
  - docs/CRISPEDIT_MASK.md
EOF
