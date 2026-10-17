import errno
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sa1b_pipeline import production_state as state
from scripts.group_production import stage
from scripts import run_group_production_4node as coordinator


@pytest.mark.parametrize('error', [errno.ENOENT, errno.ESTALE])
def test_read_retries_visibility_without_writing(tmp_path, monkeypatch, error):
    path = tmp_path / 'contract.json'
    state.atomic_json(path, {'model': 'm'})
    original = Path.read_text
    calls, sleeps = [], []
    def delayed(p, *args, **kwargs):
        calls.append(p)
        if len(calls) < 3:
            raise OSError(error, 'shared file not visible', str(p))
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', delayed)
    monkeypatch.setattr(state.time, 'sleep', sleeps.append)
    before = path.stat().st_mtime_ns
    state.require_json(path, {'model': 'm'})
    assert sleeps == [.1, .2]
    assert path.stat().st_mtime_ns == before


def test_missing_contract_has_bounded_wait_and_is_never_created(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(state.time, 'sleep', sleeps.append)
    path = tmp_path / 'missing.json'
    with pytest.raises(FileNotFoundError):
        state.require_json(path, {})
    assert len(sleeps) == 7 and sum(sleeps) < 6
    assert not path.exists()


def test_changed_or_corrupt_contract_is_not_retried_or_repaired(tmp_path, monkeypatch):
    def no_sleep(_):
        pytest.fail('permanent errors must fail immediately')
    monkeypatch.setattr(state.time, 'sleep', no_sleep)
    path = tmp_path / 'contract.json'
    state.atomic_json(path, {'model': 'old'})
    with pytest.raises(ValueError, match='config changed'):
        state.require_json(path, {'model': 'new'})
    assert json.loads(path.read_text()) == {'model': 'old'}
    path.write_text('{')
    with pytest.raises(json.JSONDecodeError):
        state.require_json(path, {})
    assert path.read_text() == '{'


def test_permission_error_is_not_retried(monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, 'denied')
    monkeypatch.setattr(Path, 'read_text', denied)
    monkeypatch.setattr(state.time, 'sleep', lambda _: pytest.fail('unexpected retry'))
    with pytest.raises(PermissionError):
        state.read_json_visible('contract.json')


def test_worker_uses_existing_contract_and_skips_cached_audit(tmp_path, monkeypatch):
    from sa1b_pipeline.group_audit import group_audit_contract
    from sa1b_pipeline import audit_backend
    group = {'twin_id': 'g1'}
    state.write_rows(tmp_path / 'generated.jsonl', [group])
    cache = tmp_path / 'audit_checkpoints'
    state.atomic_json(cache / 'contract.json', group_audit_contract('twins', 'model'))
    state.atomic_json(cache / 'g1.json', {'input_sha256': state.digest(group), 'group': group})
    before = (cache / 'contract.json').stat().st_mtime_ns
    monkeypatch.setattr(audit_backend, 'VLLMAuditor', lambda *a, **kw: pytest.fail('cached audit must not load a model'))
    monkeypatch.setattr(sys, 'argv', ['stage', '--stage', 'audit', '--wave', str(tmp_path), '--kind', 'twins', '--model', 'model'])
    stage.main()
    assert state.read_rows(tmp_path / 'audited_worker0.jsonl') == [group]
    assert (cache / 'contract.json').stat().st_mtime_ns == before
    (cache / 'contract.json').unlink()
    monkeypatch.setattr(state.time, 'sleep', lambda _: None)
    with pytest.raises(FileNotFoundError):
        stage.main()
    assert not (cache / 'contract.json').exists()


def test_parent_publishes_before_all_four_audit_readers(tmp_path, monkeypatch):
    from sa1b_pipeline.group_audit import group_audit_contract
    wave = tmp_path / 'wave'
    (wave / 'inputs').mkdir(parents=True)
    state.write_rows(wave / 'planned.jsonl', [{'twin_id': 'p'}])
    groups = [{'twin_id': f'g{i}'} for i in range(4)]
    state.write_rows(wave / 'generated.jsonl', groups)
    args = SimpleNamespace(root=tmp_path, mllm_python='mllm', editor_python='editor', editor_devices='0,1,2,3,4,5,6,7')
    c = {'planner_model': 'p', 'planner_tp': 4, 'blind_model': 'b', 'editor_model': 'e', 'audit_model': 'a', 'site_model': 's'}
    called = []
    def fake(wave, name, cmd, devices):
        if name.startswith('audit'):
            state.require_json(wave / 'audit_checkpoints/contract.json', group_audit_contract('twins', 'a'))
            called.append(name)
            state.write_rows(wave / f'audited_worker{devices}.jsonl', [groups[int(devices)]])
    monkeypatch.setattr(coordinator, 'command', fake)
    coordinator.run_wave(args, c, 'twins', wave, [{'candidate': {'source_id': 's'}}])
    assert sorted(called) == ['audit0', 'audit1', 'audit2', 'audit3']
    assert state.read_rows(wave / 'audited.jsonl') == groups
