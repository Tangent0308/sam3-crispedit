from concurrent.futures import ThreadPoolExecutor

import pytest

from sa1b_pipeline.shared_operation_quota import OPERATIONS, SharedOperationQuota


def test_atomic_inflight_reservations_across_replicas(tmp_path):
    path = tmp_path/'quota.db'
    first = SharedOperationQuota(path, {'version': 1})

    def reserve(worker):
        ledger = SharedOperationQuota(path, {'version': 1})
        try:
            return [ledger.reserve(f'{worker}:{i}', f'fp-{i}', str(worker)) for i in range(27)]
        finally:
            ledger.close()

    with ThreadPoolExecutor(max_workers=4) as pool:
        priorities = [p for group in pool.map(reserve, range(4)) for p in group]
    assert len(priorities) == 108
    assert first.counts() == {'accepted': {}, 'inflight': dict.fromkeys(OPERATIONS, 27)}
    first.close()


def test_actual_fallback_rejection_idempotency_and_binding(tmp_path):
    ledger = SharedOperationQuota(tmp_path/'q.db', {'model': 'v1'})
    assert ledger.reserve('a', 'fp', 'one')[0] == 'add'
    assert ledger.reserve('a', 'fp', 'one')[0] == 'add'
    with pytest.raises(ValueError, match='owned'):
        ledger.reserve('a', 'fp', 'two')
    with pytest.raises(ValueError, match='content'):
        ledger.reserve('a', 'changed', 'one')
    ledger.finish('a', 'one', 'attribute')
    ledger.finish('a', 'one', 'attribute')
    assert ledger.reserve('b', 'fp', 'two')[0] == 'add'
    ledger.finish('b', 'two', None)
    assert ledger.counts() == {'accepted': {'attribute': 1}, 'inflight': {}}
    with pytest.raises(ValueError, match='cannot change'):
        ledger.finish('a', 'one', 'add')
    with pytest.raises(ValueError, match='completed'):
        ledger.reserve('a', 'fp', 'one')
    with pytest.raises(ValueError, match='configuration'):
        SharedOperationQuota(tmp_path/'q.db', {'model': 'v2'})
    ledger.reserve('c', 'fp', 'crashed')
    ledger.close()
    recovered = SharedOperationQuota(tmp_path/'q.db', {'model': 'v1'})
    assert recovered.counts()['inflight'] == {'add': 1}
    assert recovered.cancel_pending('other') == 0
    assert recovered.cancel_pending('crashed') == 1
    assert recovered.counts()['inflight'] == {}
    recovered.close()
