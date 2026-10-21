#!/usr/bin/env bash
# Node-local environment of the extedit pipeline, built from its lock file (the versions of the 2026-10-20/21
# validation):
#   vllm    Qwen3.8-27B server (requirements/extedit-vllm-lock.txt)
#   worker  every other stage and the 27B clients (requirements/extedit-worker-lock.txt)
# usage: extedit_setup_env.sh <vllm|worker> <environment directory>
# environment: EXTEDIT_PYTHON (default /usr/bin/python3.11), EXTEDIT_INDEX (bytedpypi), EXTEDIT_TORCH_INDEX (PyTorch
# cu129), EXTEDIT_V3_REPO (SAMTokEdit v3 checkout, needed to validate the worker's SAMTok codec).
set -Eeuo pipefail
KIND="${1:?vllm or worker}"
ENV="${2:?environment directory}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${EXTEDIT_PYTHON:-/usr/bin/python3.11}"
INDEX="${EXTEDIT_INDEX:-https://bytedpypi.byted.org/simple/}"
TORCH_INDEX="${EXTEDIT_TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
LOCK="$REPO/requirements/extedit-$KIND-lock.txt"
[[ -f "$LOCK" ]] || { echo "unknown environment '$KIND' (vllm or worker)" >&2; exit 2; }
export UV_LINK_MODE=copy
UV="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "$UV" ]]; then
  "$PYTHON_BIN" -m pip install --index-url "$INDEX" --target "${ENV}-uv" uv==0.12.15
  UV="${ENV}-uv/bin/uv"
fi
[[ -f "$ENV/pyvenv.cfg" ]] || "$UV" venv --python "$PYTHON_BIN" --no-python-downloads "$ENV"
"$UV" pip sync --python "$ENV/bin/python" --index-url "$INDEX" --extra-index-url "$TORCH_INDEX" \
  --index-strategy unsafe-best-match "$LOCK"
"$UV" pip check --python "$ENV/bin/python"
if [[ "$KIND" == vllm ]]; then
  "$ENV/bin/python" - <<'PY'
import sys, torch, transformers, vllm
assert sys.version_info[:2] == (3, 11)
assert torch.__version__ == "2.13.0+cu129", torch.__version__
assert transformers.__version__ == "5.17.0", transformers.__version__
assert vllm.__version__.startswith("0.31.0"), vllm.__version__
print(f"extedit vllm environment: Python 3.11 / torch {torch.__version__} / vLLM {vllm.__version__}")
PY
else
  : "${EXTEDIT_V3_REPO:?set EXTEDIT_V3_REPO to the SAMTokEdit v3 checkout}"
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO:$EXTEDIT_V3_REPO/src:$EXTEDIT_V3_REPO/third_party" "$ENV/bin/python" -B - <<'PY'
import sys, torch, cv2, pycocotools, aiohttp
assert sys.version_info[:2] == (3, 11)
assert torch.__version__ == "2.13.0+cu129", torch.__version__
from sam3.model_builder import build_sam3_image_model  # noqa: F401  (SAM3 from this repository)
from crispedit.mask.pipeline import segment_grounded_box  # noqa: F401
from samtok_edit21.models.codec import load_codec  # noqa: F401  (released SAMTok codec)
from extedit import gate, plan, segment, sources  # noqa: F401
print(f"extedit worker environment: Python 3.11 / torch {torch.__version__} / OpenCV {cv2.__version__}")
PY
fi
