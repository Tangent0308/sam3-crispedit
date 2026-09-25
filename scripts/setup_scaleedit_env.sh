#!/usr/bin/env bash
# Node-local Qwen3.8/vLLM + SAM3.
set -euo pipefail
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $repo_dir != /mnt/* ]] || { echo 'Use a node-local git clone' >&2; exit 2; }
export UV_PYTHON_INSTALL_DIR="$repo_dir/.uv-python"
export UV_CACHE_DIR=${SCALEEDIT_UV_CACHE_DIR:-$repo_dir/.uv-cache}
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
[[ $(realpath -m "$UV_CACHE_DIR") != /mnt/* ]] || { echo 'Cache must be node-local' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv==0.11.32 first' >&2; exit 2; }
exec 9>"$repo_dir/.scaleedit-env.lock"
flock 9
venv_dir=$repo_dir/.venv-scaleedit-current
uv python install 3.12.13
python_bin=$(uv python find --managed-python 3.12.13)
[[ $(realpath "$python_bin") == "$repo_dir/.uv-python/"* ]] || { echo 'Python must be local to clone' >&2; exit 2; }
if [[ ! -e $venv_dir ]]; then
  uv venv --python "$python_bin" "$venv_dir"
  touch "$venv_dir/.scaleedit-managed"
elif [[ ! -f $venv_dir/.scaleedit-managed ]]; then
  echo "Unmanaged environment exists: $venv_dir" >&2
  exit 2
fi
uv pip uninstall --python "$venv_dir/bin/python" opencv-python opencv-contrib-python \
  opencv-python-headless opencv-contrib-python-headless
uv pip install --python "$venv_dir/bin/python" --no-deps \
  --index-url https://download.pytorch.org/whl/cu129 \
  'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0' 'torchcodec==0.16.0'
uv pip install --python "$venv_dir/bin/python" --no-deps -r "$repo_dir/scripts/scaleedit_packages.txt"
uv pip install --python "$venv_dir/bin/python" --no-deps -e "$repo_dir"
"$venv_dir/bin/python" - <<'PY'
import cv2
import torch
import transformers
import vllm
import sam3
from importlib.metadata import distributions
packages = {d.metadata['Name'].lower() for d in distributions()}
opencv = {p for p in packages if p.startswith('opencv-')}
assert opencv == {'opencv-python-headless'}, opencv
assert 'GUI:                           NONE' in cv2.getBuildInformation()
print('Environment ready:', torch.__version__, transformers.__version__, vllm.__version__)
print('Visible GPUs:', torch.cuda.device_count())
PY
