#!/usr/bin/env bash
# Cache a trusted environment bundle locally; model/data/output stay shared.
# Bundle layout: lib/python3.12/site-packages/ and bin/ninja (no interpreter).
set -euo pipefail
archive=${1:?Usage: cache_crispedit_runtime.sh ARCHIVE BASE_PYTHON CACHE_DIR}
base_python=${2:?Shared Python 3.12 interpreter required}
cache_dir=${3:?Absolute node-local cache directory required}
[[ $cache_dir == /* && $cache_dir != / && -f $archive && -x $base_python ]] || exit 2
[[ $cache_dir != /mnt/bn/* ]] || { echo 'Runtime cache must be node-local' >&2; exit 2; }
"$base_python" -c 'import sys; assert sys.version_info[:2] == (3,12)'
mkdir -p "$(dirname "$cache_dir")"
exec 9>"${cache_dir}.lock"
flock 9
# Adjacent checksum is generated once by the bundle producer; verify before
# marking this node's extracted environment ready.
read -r expected_sha _ < "${archive}.sha256"
[[ $expected_sha =~ ^[0-9a-f]{64}$ ]] || exit 2
if [[ -f $cache_dir/bundle.sha256 ]]; then
  [[ $(<"$cache_dir/bundle.sha256") == "$expected_sha" ]] || { echo 'Cache belongs to another bundle; choose a new cache path' >&2; exit 2; }
  [[ -x $cache_dir/bin/python ]] || exit 2
else
  [[ ! -e $cache_dir ]] || { echo 'Incomplete cache exists; choose a new cache path' >&2; exit 2; }
  staging=$(mktemp -d "${cache_dir}.tmp.XXXXXX")
  echo "Extracting trusted runtime bundle into $staging" >&2
  tar -xzf "$archive" -C "$staging"
  actual_sha=$(sha256sum "$archive")
  [[ ${actual_sha%% *} == "$expected_sha" ]] || { echo "Bundle checksum failed; diagnostic staging retained: $staging" >&2; exit 2; }
  # Python venv can relocate these installed libraries; console entrypoints
  # from the original machine are intentionally absent (except native ninja).
  mv "$staging" "$cache_dir"
  "$base_python" -m venv --without-pip "$cache_dir"
  printf '%s\n' "$expected_sha" > "$cache_dir/bundle.sha256"
fi
printf '%s\n' "$cache_dir/bin/python"
