import concurrent.futures
import time

import pytest

from sa1b_pipeline.work_queue import WorkQueue


def test_parallel_claims_deliver_every_job_once(tmp_path):
    path = tmp_path/'queue.db'
    queue = WorkQueue(path)
    for i in range(47):
        queue.enqueue({'job_id': str(i), 'row': {'value': i}})
    queue.seal()

    def run():
        worker = WorkQueue(path)
        processed = []
        try:
            while batch := worker.claim(4):
                for job, lease in batch:
                    worker.finish(job['job_id'], lease, {'job_id': job['job_id']})
                    processed.append(job['job_id'])
        finally:
            worker.close()
        return processed

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        done = [job for jobs in pool.map(lambda _: run(), range(4)) for job in jobs]
    assert len(done) == len(set(done)) == 47
    assert queue.counts() == {'done': 47}
    assert len(queue.results()) == 47
    queue.close()


def test_restart_fences_expired_worker_and_preserves_completed(tmp_path):
    path = tmp_path/'queue.db'
    queue = WorkQueue(path)
    queue.enqueue({'job_id': 'a'})
    _job, old_lease = queue.claim(1)[0]
    queue.db.execute('UPDATE jobs SET expires=?', (time.time()-1,))
    queue.close()
    resumed = WorkQueue(path)
    _job, lease = resumed.claim(1)[0]
    with pytest.raises(RuntimeError, match='stale'):
        resumed.finish('a', old_lease, {'job_id': 'a'})
    resumed.finish('a', lease, {'job_id': 'a', 'decision': 'fail'})
    assert resumed.claim(1) == []
    assert resumed.results()[0]['decision'] == 'fail'
    resumed.close()


def test_duplicate_content_config_and_result_binding(tmp_path):
    queue = WorkQueue(tmp_path/'queue.db')
    assert queue.enqueue({'job_id': 'a', 'fingerprint': 'v1'})
    assert not queue.enqueue({'job_id': 'a', 'fingerprint': 'v1'})
    with pytest.raises(ValueError, match='different content'):
        queue.enqueue({'job_id': 'a', 'fingerprint': 'v2'})
    queue.bind_config({'model': 'v1'})
    with pytest.raises(ValueError, match='different model'):
        queue.bind_config({'model': 'v2'})
    _job, lease = queue.claim(1)[0]
    with pytest.raises(ValueError, match='another job'):
        queue.finish('a', lease, {'job_id': 'b'})
    queue.release('a', lease)
    assert queue.counts() == {'pending': 1}
    queue.seal()
    assert not queue.enqueue({'job_id': 'a', 'fingerprint': 'v1'})
    with pytest.raises(ValueError, match='sealed'):
        queue.enqueue({'job_id': 'b'})
    queue.close()


def test_timed_batch_wait_does_not_hoard_and_seal_flushes(tmp_path):
    path = tmp_path/'queue.db'
    first, second = WorkQueue(path), WorkQueue(path)
    first.enqueue({'job_id': 'a'})
    assert first.claim(4, min_count=4, max_wait_seconds=60) == []
    assert second.claim(4, min_count=4, max_wait_seconds=60) == []
    assert first.counts() == {'pending': 1}
    for name in ['b', 'c', 'd']:
        first.enqueue({'job_id': name})
    batch = second.claim(4, min_count=4, max_wait_seconds=60)
    assert len(batch) == 4
    for job, lease in batch:
        second.finish(job['job_id'], lease, {'job_id': job['job_id']})
    first.enqueue({'job_id': 'tail'})
    first.seal()
    assert len(first.claim(4, min_count=4, max_wait_seconds=60)) == 1
    first.close()
    second.close()


def test_timed_batch_timeout_and_legacy_migration(tmp_path):
    import sqlite3
    path = tmp_path/'legacy.db'
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, '
               'payload TEXT NOT NULL, state TEXT NOT NULL DEFAULT \'pending\', '
               'lease TEXT, expires REAL, attempts INTEGER NOT NULL DEFAULT 0, result TEXT)')
    db.execute('INSERT INTO jobs(id,fingerprint,payload) VALUES (?,?,?)', ('old', 'fp', '{"job_id":"old"}'))
    db.commit()
    db.close()
    queue = WorkQueue(path)
    assert queue.claim(8, min_count=8, max_wait_seconds=60) == []
    queue.db.execute('UPDATE jobs SET created_at=?', (time.time()-61,))
    assert len(queue.claim(8, min_count=8, max_wait_seconds=60)) == 1
    queue.close()


def test_export_never_replaces_manifest_with_incomplete_results(tmp_path):
    import json
    queue = WorkQueue(tmp_path/'q.db')
    destination = tmp_path/'validation.jsonl'
    destination.write_text('previous complete result\n')
    queue.enqueue({'job_id': 'a'})
    queue.seal()
    with pytest.raises(ValueError, match='incomplete'):
        queue.export_results(destination)
    assert destination.read_text() == 'previous complete result\n'
    job, lease = queue.claim(1)[0]
    queue.finish(job['job_id'], lease, {'job_id': 'a', 'decision': 'pass'})
    queue.export_results(destination)
    assert json.loads(destination.read_text()) == {'job_id': 'a', 'decision': 'pass'}
    queue.close()


def test_indexed_claim_preserves_global_order_between_expired_and_pending(tmp_path):
    queue = WorkQueue(tmp_path/'q.db')
    for name in ['expired-first', 'pending-second', 'pending-third']:
        queue.enqueue({'job_id': name})
    queue.claim(1)
    queue.db.execute("UPDATE jobs SET expires=0 WHERE state='running'")
    assert [job['job_id'] for job, _ in queue.claim(2)] == ['expired-first', 'pending-second']
    queue.close()
