import fcntl
import json
import tarfile

import pytest

from sa1b_pipeline.production_state import atomic_json, atomic_text, file_hash
from scripts.production import repair_deployment as repair


def prepared(tmp_path, monkeypatch):
    repo, root = tmp_path/'repo', tmp_path/'run'
    repo.mkdir()
    (repo/'runtime.py').write_text('new_runtime = True\n')
    old = tmp_path/'old'
    old.mkdir()
    (old/'runtime.py').write_text('old_runtime = True\n')
    (old/'planner.py').write_text('frozen_prompt = True\n')
    (root/'deployment').mkdir(parents=True)
    with tarfile.open(root/'deployment/code.tar.gz', 'w:gz') as bundle:
        for file in old.iterdir():
            bundle.add(file, arcname=file.name)
    atomic_text(root/'deployment/code.sha256', file_hash(root/'deployment/code.tar.gz')+'\n')
    atomic_json(root/'config.json', {'target_total': 50000, 'planner_model': 'canonical-model',
                'code_hashes': {f.name: file_hash(f) for f in old.iterdir()}})
    snapshot = root/'nodes/node0/waves/00000/planning_0/code_snapshot/planner.py'
    atomic_text(snapshot, (old/'planner.py').read_text())
    checkpoint = snapshot.parent.parent/'journal/finished.json'
    atomic_json(checkpoint, {'status': 'complete', 'case': 'retain-me'})
    monkeypatch.setattr(repair, 'REPO', repo)
    monkeypatch.setattr(repair, 'PATCH_FILES', ('runtime.py',))
    return root, checkpoint


def test_stopped_deployment_repair_retains_snapshots_and_checkpoints(tmp_path, monkeypatch):
    root, checkpoint = prepared(tmp_path, monkeypatch)
    before = checkpoint.read_bytes()
    old = json.loads((root/'config.json').read_text())
    report = repair.repair(root)
    updated = json.loads((root/'config.json').read_text())
    assert report['planner_snapshot_files_verified'] == 1
    assert checkpoint.read_bytes() == before
    assert {k: v for k, v in old.items() if k != 'code_hashes'} == {
        k: v for k, v in updated.items() if k != 'code_hashes'}
    assert old['code_hashes']['planner.py'] == updated['code_hashes']['planner.py']
    assert file_hash(root/'deployment/code.tar.gz') == report['bundle_sha256']


def test_running_rank_blocks_repair(tmp_path, monkeypatch):
    root, _ = prepared(tmp_path, monkeypatch)
    (root/'control').mkdir()
    before = file_hash(root/'deployment/code.tar.gz')
    with (root/'control/bootstrap.node0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            repair.repair(root)
    assert file_hash(root/'deployment/code.tar.gz') == before
