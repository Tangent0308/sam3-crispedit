"""Freeze the current tested working tree for identical four-node deployment."""
import argparse
import shutil
import tarfile
from pathlib import Path

from sa1b_pipeline.production_state import atomic_text, file_hash
from scripts.run_production_4node import REPO, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    load_config(a.root)
    out = a.root/'deployment'
    out.mkdir(exist_ok=True)
    if (out/'code.tar.gz').exists():
        raise FileExistsError('deployment bundle already frozen; do not replace an active run')
    temp = out/'code.tar.gz.tmp'
    with tarfile.open(temp, 'w:gz') as archive:
        for name in ['sa1b_pipeline', 'scripts', 'utils', 'requirements', 'pyproject.toml']:
            source = REPO/name
            paths = [source] if source.is_file() else sorted(source.rglob('*'))
            for path in paths:
                if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                    archive.add(path, arcname=str(path.relative_to(REPO)))
    temp.replace(out/'code.tar.gz')
    atomic_text(out/'code.sha256', file_hash(out/'code.tar.gz')+'\n')
    shutil.copy2(REPO/'scripts/production/bootstrap_arnold_4node.sh', out/'launch_4node.sh')
    atomic_text(out/'arnold_entry.sh', '#!/usr/bin/env bash\nset -euo pipefail\n'
                f'export SA1B_RUN_ROOT={str(a.root)!r}\n'
                'exec bash "$SA1B_RUN_ROOT/deployment/launch_4node.sh"\n')
    print(out/'arnold_entry.sh')


if __name__ == '__main__':
    main()
