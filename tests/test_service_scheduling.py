import json
import os

from sa1b_pipeline.inference_services import RemoteAuditor, RemoteEditor, ServicePool
from scripts.group_production.inference_worker import next_batch


def _pool(tmp_path, dispatch='round_robin'):
    pool = ServicePool(python='python', role='editor', devices=['0', '1', '2'], model='m', directory=tmp_path,
                       dispatch=dispatch)
    pool.workers = [(tmp_path / str(i), None, None) for i in range(3)]
    pool.outstanding = [0, 0, 0]
    return pool


def test_least_loaded_dispatch_avoids_busy_workers_and_round_robin_is_unchanged(tmp_path):
    pool = _pool(tmp_path)
    assert [pool._pick() for _ in range(4)] == [0, 1, 2, 0]
    pool = _pool(tmp_path, 'least_loaded')
    pool.outstanding = [2, 0, 1]
    assert pool._pick() == 1
    pool.outstanding = [1, 1, 1]
    picks = [pool._pick() for _ in range(3)]
    assert sorted(picks) == [0, 1, 2]                   # ties rotate
    import pytest
    with pytest.raises(ValueError):
        ServicePool(python='p', role='editor', devices=['0'], model='m', directory=tmp_path, dispatch='random')


def test_priority_requests_jump_the_queue_fifo_within_a_priority(tmp_path):
    def write(name, value, t):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        os.utime(path, ns=(t, t))
        return path
    twin_a = write('a.json', {'group': {}}, 1)
    twin_b = write('b.json', {'group': {}}, 2)
    multi_1 = write('c.json', {'unit': {}, 'priority': 1}, 3)
    multi_2 = write('d.json', {'unit': {}, 'priority': 1}, 4)
    assert next_batch([twin_b, multi_2, twin_a, multi_1], 'editor', 1) == [multi_1]
    assert next_batch([twin_b, twin_a], 'editor', 1) == [twin_a]          # no priorities: plain FIFO
    blind = write('e.json', {'request_kind': 'blind'}, 5)
    audit_multi = write('f.json', {'job_id': 'x', 'priority': 1}, 6)
    audit_twin = write('g.json', {'job_id': 'y'}, 0)
    assert next_batch([audit_twin, audit_multi, blind], 'audit', 8) == [blind]   # blind probes still first
    assert next_batch([audit_twin, audit_multi], 'audit', 8) == [audit_multi, audit_twin]


class _Recorder:
    model = 'm'
    batch_size = 8
    dtype = 'bf16'

    def __init__(self, priority_scheduling):
        self.priority_scheduling, self.payloads = priority_scheduling, []

    def request(self, payload):
        self.payloads.append(payload)
        return {'job_id': payload.get('job_id'), 'image': None, 'context': {}}


def test_multi_adapters_tag_requests_only_when_enabled():
    on, off = _Recorder(True), _Recorder(False)
    for pool, expected in ((on, 1), (off, None)):
        RemoteAuditor(pool, kind='multi').call_batch([{'job_id': 'j1'}])
        RemoteAuditor(pool, kind='twins').call_batch([{'job_id': 'j2'}])
        assert pool.payloads[0].get('priority') == expected
        assert 'priority' not in pool.payloads[1]
        editor = RemoteEditor(pool)
        assert editor.priority == (expected or 0)
