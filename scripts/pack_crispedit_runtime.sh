#!/usr/bin/env bash
# Run once after setup_crispedit_env.sh; distribute installed libraries, not code.
set -euo pipefail
venv_dir=${1:?Usage: pack_crispedit_runtime.sh VENV_DIR ARCHIVE.tar.gz}
archive=${2:?Output archive required}
[[ -d $venv_dir/lib/python3.12/site-packages && -f $venv_dir/bin/ninja ]] || exit 2
[[ ! -e $archive && ! -e ${archive}.sha256 ]] || { echo 'Refusing to replace an existing runtime bundle' >&2; exit 2; }
mkdir -p "$(dirname "$archive")"
temporary=$(mktemp "${archive}.tmp.XXXXXX")
tar --exclude='__pycache__' --exclude='__editable__*' --exclude='sam3-*.dist-info' \
  -cf - -C "$venv_dir" lib bin/ninja | gzip -1 > "$temporary"
mv "$temporary" "$archive"
sha256sum "$archive" > "${archive}.sha256"
echo "Runtime bundle ready: $archive"
