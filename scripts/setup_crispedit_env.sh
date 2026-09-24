#!/usr/bin/env bash
# Install the pinned Qwen3.8/vLLM + SAM3 environment inside a node-local clone.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
[[ $repo_dir != /mnt/* ]] || { echo 'Clone the repository on node-local disk (/opt or /tmp).' >&2; exit 2; }
venv_dir=$repo_dir/.venv-crispedit
export UV_PYTHON_INSTALL_DIR="$repo_dir/.uv-python"
export UV_CACHE_DIR=${CRISPEDIT_UV_CACHE_DIR:-/opt/tiger/tanyue/.cache/crispedit-uv}
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=5
[[ $(realpath -m "$UV_CACHE_DIR") != /mnt/* ]] || { echo 'uv cache must be node-local' >&2; exit 2; }
command -v uv >/dev/null || { echo 'Install uv first: python3 -m pip install --user uv==0.11.32' >&2; exit 2; }
exec 9>"$repo_dir/.crispedit-env.lock"
flock 9
fingerprint=$(sha256sum "$repo_dir/scripts/crispedit_packages.txt" "$repo_dir/scripts/setup_crispedit_env.sh" \
  "$repo_dir/scripts/preflight_crispedit_env.py" "$repo_dir/pyproject.toml" | sha256sum | cut -d' ' -f1)
ready=$venv_dir/.crispedit-ready
if [[ -f $ready && $(<"$ready") == "$fingerprint" ]]; then
  "$venv_dir/bin/python" "$repo_dir/scripts/preflight_crispedit_env.py" --gpus "${CRISPEDIT_EXPECTED_GPUS:-8}"
  echo "Environment already verified: $venv_dir"
  exit 0
fi
echo 'Installing node-local Python 3.12.13 and pinned dependencies'
uv python install 3.12.13
python_bin=$(uv python find --managed-python 3.12.13)
[[ $(realpath "$python_bin") == "$repo_dir/.uv-python/"* ]] || { echo 'Base Python must live inside this clone' >&2; exit 2; }
if [[ ! -e $venv_dir ]]; then
  uv venv --python "$python_bin" "$venv_dir"
elif [[ ! -f $venv_dir/.crispedit-installing && ! -f $ready ]]; then
  echo "Unmanaged environment exists: $venv_dir; use a fresh local clone" >&2
  exit 2
fi
touch "$venv_dir/.crispedit-installing"
# The cv2 distributions share files; remove all variants before installing one.
uv pip uninstall --python "$venv_dir/bin/python" opencv-python opencv-python-headless \
  opencv-contrib-python opencv-contrib-python-headless
uv pip install --python "$venv_dir/bin/python" --no-deps \
  --index-url https://download.pytorch.org/whl/cu129 \
  'torch==2.13.0' 'torchvision==0.28.0' 'torchaudio==2.11.0' 'torchcodec==0.16.0'
# Preserve the validated NumPy 1.26 stack; exact pins prevent transitive
# resolution from reinstalling GUI OpenCV or upgrading NumPy.
uv pip install --python "$venv_dir/bin/python" --no-deps -r "$repo_dir/scripts/crispedit_packages.txt"
uv pip install --python "$venv_dir/bin/python" --no-deps -e "$repo_dir"
"$venv_dir/bin/python" "$repo_dir/scripts/preflight_crispedit_env.py" --gpus "${CRISPEDIT_EXPECTED_GPUS:-8}"
printf '%s\n' "$fingerprint" > "$ready"
echo "Environment ready: $venv_dir"
