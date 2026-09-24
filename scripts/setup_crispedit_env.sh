#!/usr/bin/env bash
# Install the tested Qwen3.8/vLLM + SAM3 stack on shared storage.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv_dir=${VENV_DIR:-$repo_dir/.venv-crispedit}
command -v uv >/dev/null || { echo "uv is required" >&2; exit 2; }
export UV_PYTHON_INSTALL_DIR="$repo_dir/.uv-python"
if [[ -z ${PYTHON_BIN:-} ]]; then
  uv python install 3.12
  PYTHON_BIN=$(uv python find --managed-python 3.12)
fi
shared_python=$PYTHON_BIN
[[ "$shared_python" == "$repo_dir/.uv-python/"* ]] || { echo "Python must be inside shared repository" >&2; exit 2; }

if [[ ! -e "$venv_dir" ]]; then
  uv venv --python "$shared_python" "$venv_dir"
elif [[ "${CRISPEDIT_ALLOW_EXISTING_VENV:-0}" != 1 ]]; then
  echo "Refusing to reuse an existing venv: $venv_dir" >&2
  exit 2
fi

if ! "$venv_dir/bin/python" -c 'import torch; assert torch.version.cuda == "12.9"' >/dev/null 2>&1; then
  uv pip install --python "$venv_dir/bin/python" \
    --index-url https://download.pytorch.org/whl/cu129 \
    'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0' 'torchcodec==0.16.0'
fi

# Current vLLM wheel metadata requests NumPy 2 while the tested runtime uses
# NumPy 1.26. Install the complete working environment as exact pins without
# asking the resolver to change the already validated package combination.
uv pip install --python "$venv_dir/bin/python" --no-deps \
  -r "$repo_dir/scripts/crispedit_packages.txt"
# Both OpenCV distributions are present in the tested environment. Their cv2
# files overlap, so make the known-working 4.11 build the final installed one.
uv pip install --python "$venv_dir/bin/python" --no-deps --reinstall \
  'opencv-python==4.11.0.86'
uv pip install --python "$venv_dir/bin/python" --no-deps -e "$repo_dir"

"$venv_dir/bin/python" - <<'PY'
import cv2
import numpy
import pyarrow
import torch
import transformers
import vllm
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
assert torch.version.cuda == "12.9"
assert torch.cuda.device_count() == 8
print("READY", torch.__version__, transformers.__version__, vllm.__version__, numpy.__version__, cv2.__version__)
PY
