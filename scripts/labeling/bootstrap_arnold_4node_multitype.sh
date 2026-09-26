#!/usr/bin/env bash
# Arnold entry for the independent add/replace/attribute run.
# It reuses the audited clone/install/bootstrap contract while selecting the
# non-remove runner and a separate data/output root. Run identically on all
# four Arnold workers.
set -euo pipefail
export SAMTOK_PIPELINE_MODE=multitype
export SAMTOK_RUN_ID="${SAMTOK_RUN_ID:?Set a unique multitype run ID}"
export SAMTOK_RUN_ROOT="${SAMTOK_RUN_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments/SAMTok_Derived_Edit_Labeling/four_node/$SAMTOK_RUN_ID}"
export SAMTOK_DATA_ROOT="${SAMTOK_DATA_ROOT:-$SAMTOK_RUN_ROOT/data/add_replace_attribute}"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bootstrap_arnold_4node.sh" "$@"
