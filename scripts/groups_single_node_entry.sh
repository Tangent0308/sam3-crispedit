#!/usr/bin/env bash
# Shared 27B candidate entry; choose one dataset per eight-GPU node.
set -euo pipefail
: "${GROUP_KIND:?Set GROUP_KIND to twins or multi}"
: "${GROUP_RUN_ROOT:?Set a separate output directory for this dataset}"
case "$GROUP_KIND" in
  twins) default_target=15000 ;;
  multi) default_target=5000 ;;
  *) echo 'GROUP_KIND must be twins or multi' >&2; exit 2 ;;
esac
export TWINS_RUN_ROOT="$GROUP_RUN_ROOT"
export TWINS_TARGET="${GROUP_TARGET:-$default_target}"
export TWINS_SOURCE_ROOT="${GROUP_SOURCE_ROOT:-/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline/02_production_runs/sa1b_samtok_group_train_20k_20261016}"
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo/scripts/twins_single_node_entry.sh" --kind "$GROUP_KIND" "$@"
