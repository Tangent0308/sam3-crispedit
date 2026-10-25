#!/usr/bin/env python3
"""Freeze a self-contained single-node entry on the shared output volume."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.fonts import verify_fonts
from sa1b_pipeline.production_state import atomic_text

LAUNCH = '''#!/usr/bin/env bash
set -euo pipefail
deployment=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TWINS_RUN_ROOT=$(dirname "$deployment")
export GROUP_RUN_ROOT="$TWINS_RUN_ROOT"
expected=$(cat "$deployment/code.sha256")
[[ "$(sha256sum "$deployment/code.tar.gz" | cut -d' ' -f1)" == "$expected" ]] || { echo 'Code checksum mismatch' >&2; exit 2; }
code="/opt/tiger/sa1b-twins-single-code/$expected"
if [[ ! -d "$code" ]]; then
  temporary="$code.tmp.$$"
  mkdir -p "$temporary"
  tar -xzf "$deployment/code.tar.gz" -C "$temporary"
  mv "$temporary" "$code"
fi
if [[ -n "${GROUP_KIND:-}" ]]; then
  exec bash "$code/scripts/groups_single_node_entry.sh" "$@"
fi
exec bash "$code/scripts/twins_single_node_entry.sh" "$@"
'''


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(argv)
    repo=Path(__file__).resolve().parents[1]
    verify_fonts()
    out=args.root.resolve()/'deployment'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'code.tar.gz').exists():
        raise FileExistsError('Deployment already frozen; choose another root rather than overwrite')
    paths=[]
    for folder in ('sa1b_pipeline','scripts','utils','requirements','docs'):
        paths.extend(p for p in (repo/folder).rglob('*') if p.is_file()
                     and not any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(repo).parts)
                     and p.suffix not in ('.pyc','.pyo'))
    paths.extend(repo/name for name in ('README.md','pyproject.toml','requirements.txt'))
    with tarfile.open(out/'code.tar.gz','w:gz') as bundle:
        for path in sorted(paths):bundle.add(path,arcname=path.relative_to(repo))
    checksum=hashlib.sha256((out/'code.tar.gz').read_bytes()).hexdigest()
    atomic_text(out/'code.sha256',checksum+'\n')
    atomic_text(out/'launch_single_node.sh',LAUNCH)
    print(f'exec bash {out}/launch_single_node.sh')


if __name__=='__main__':main()
