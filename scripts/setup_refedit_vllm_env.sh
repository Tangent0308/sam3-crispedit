#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REFEDIT_ENV_DIR="${REFEDIT_ENV_DIR:-${REPO_ROOT}/.venv-refedit-vllm}"

VENV_DIR="${REFEDIT_ENV_DIR}" bash "${SCRIPT_DIR}/setup_scaleedit_vllm_env.sh"
"${REFEDIT_ENV_DIR}/bin/python" - <<'PY'
import refedit

print("IMPORTED refedit", refedit.GROUND_PROMPT_VERSION)
PY

echo "[refedit-vllm] Environment ready at ${REFEDIT_ENV_DIR}"
