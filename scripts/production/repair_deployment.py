"""Publish the startup-only repair over a stopped deployment, preserving journals."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import tarfile
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from sa1b_pipeline.production_state import atomic_json, atomic_text, file_hash

REPO = Path(__file__).resolve().parents[2]
# Deliberately exclude planner code, prompts, image editing logic and audit rules.
PATCH_FILES = (
    'requirements/production/audit.lock.txt',
    'scripts/production/setup_runtime.sh',
    'scripts/production/check_runtime.py',
    'sa1b_pipeline/model_cache.py',
    'sa1b_pipeline/production_generation.py',
    'sa1b_pipeline/audit_backend.py',
    'utils/qwen21_omni.py',
    'scripts/run_production_4node.py',
)


def repair(root):
    root = Path(root)
    deployment = root/'deployment'
    with ExitStack() as stack:
        # Hold both entry-point and rank locks throughout publication. Refuse
        # to alter any deployment whose jobs are running on another worker.
        for rank in range(4):
            for path in (root/f'control/bootstrap.node{rank}.lock', root/f'nodes/node{rank}/owner.lock'):
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(path.open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = json.loads((root/'config.json').read_text())
        old_hash = (deployment/'code.sha256').read_text().strip()
        if file_hash(deployment/'code.tar.gz') != old_hash:
            raise ValueError('deployed bundle checksum mismatch')
        with tempfile.TemporaryDirectory(prefix='sa1b-repair-') as temporary:
            tree = Path(temporary)/'code'
            tree.mkdir()
            with tarfile.open(deployment/'code.tar.gz') as bundle:
                for member in bundle:
                    destination = tree/member.name
                    if not destination.resolve().is_relative_to(tree.resolve()):
                        raise ValueError('unsafe archive member path')
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as source, destination.open('wb') as out:
                            shutil.copyfileobj(source, out)
                        destination.chmod(member.mode)
                    else:
                        raise ValueError('deployment must contain regular files only')
            for name, expected in old['code_hashes'].items():
                if file_hash(tree/name) != expected:
                    raise ValueError(f'old frozen file changed: {name}')
            for name in PATCH_FILES:
                shutil.copy2(REPO/name, tree/name)
            # Previously submitted planner calls keep exactly the same code.
            snapshots = 0
            for directory in (root/'nodes').glob('node*/waves/*/planning_*/code_snapshot'):
                for snapshot in directory.glob('*.py'):
                    matches = list(tree.rglob(snapshot.name))
                    if not any(file_hash(p) == file_hash(snapshot) for p in matches):
                        raise ValueError(f'planner resume snapshot incompatible: {snapshot}')
                    snapshots += 1
            config = json.loads(json.dumps(old))
            config['code_hashes'].update({name: file_hash(tree/name) for name in PATCH_FILES})
            bundle_path = Path(temporary)/'code.tar.gz'
            with tarfile.open(bundle_path, 'w:gz') as bundle:
                for path in sorted(tree.rglob('*')):
                    if path.is_file():
                        bundle.add(path, arcname=str(path.relative_to(tree)))
            archive = deployment/'history'/time.strftime('startup_repair_%Y%m%d_%H%M%S', time.gmtime())
            archive.mkdir(parents=True)
            for name in ('code.tar.gz', 'code.sha256'):
                shutil.copy2(deployment/name, archive/name)
            shutil.copy2(root/'config.json', archive/'config.json')
            try:
                staging = deployment/'code.tar.gz.repair'
                shutil.copy2(bundle_path, staging)
                staging.replace(deployment/'code.tar.gz')
                atomic_json(root/'config.json', config)
                atomic_text(deployment/'code.sha256', file_hash(deployment/'code.tar.gz')+'\n')
            except BaseException:
                for name in ('code.tar.gz', 'code.sha256'):
                    shutil.copy2(archive/name, deployment/name)
                shutil.copy2(archive/'config.json', root/'config.json')
                raise
            report = {'previous_bundle': old_hash, 'bundle_sha256': file_hash(deployment/'code.tar.gz'),
                      'archive': str(archive), 'patched_files': list(PATCH_FILES),
                      'planner_snapshot_files_verified': snapshots,
                      'config_changed_fields': ['code_hashes'], 'dataset_root': str(root),
                      'checkpoints_modified': False, 'published_unix': time.time()}
            atomic_json(root/'verification/startup_repair/deployment.json', report)
            return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    print(json.dumps(repair(parser.parse_args().root), indent=2))
