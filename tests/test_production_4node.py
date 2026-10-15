import hashlib
import json

import pytest
from PIL import Image

from sa1b_pipeline.operation_balance import OperationBalance
from sa1b_pipeline.production_generation import JournalQueue, run_generation
from sa1b_pipeline.production_sources import materialize
from sa1b_pipeline.production_state import (
    OPERATIONS,
    PlanningJournal,
    atomic_json,
    portable,
    read_rows,
    write_rows,
)
from sa1b_pipeline.work_queue import WorkQueue
from scripts.run_production_4node import collect, finalize


def case(op, i=0):
    return {'id': f'{op}-{i}', 'source': {'source_id': f'source-{i}'},
            'units': [{'unit_id': 'U1', 'operation': op, 'object_id': op}]}


def result(job, passed=True):
    c = job['case']
    op = c['units'][0]['operation']
    return {'job_id': job['job_id'], 'source_id': c['source']['source_id'], 'object_id': op,
            'operation': op, 'decision': 'pass' if passed else 'fail', 'operation_audit_required': True,
            'semantic_audit': {'decision': 'pass' if passed else 'fail', 'operation_correct': True,
                               'target_done': True, 'non_target_preserved': True, 'natural': True}}


def job(c):
    return {'job_id': c['id']+'/U1', 'case': c}


def test_unequal_remaining_quotas_and_recovery(tmp_path):
    cases = [case(op, i) for i in range(4) for op in OPERATIONS]
    targets = {'add': 1, 'replace': 2, 'remove': 3, 'attribute': 0}
    q = WorkQueue(tmp_path/'queue.db')
    original = OperationBalance(cases, targets, 24)
    journal = JournalQueue(q, tmp_path/'dispatch')
    original.dispatch(journal)
    dispatched = [json.loads(p.read_text()) for p in (tmp_path/'dispatch').glob('*.json')]
    assert len(dispatched) == 6
    first = dispatched[0]
    outcomes = [result(first, False)]
    recovered = OperationBalance(cases, targets, 24)
    recovered.restore(dispatched, outcomes)
    assert len(recovered.pending) == 5
    recovered.dispatch(journal)
    assert len(recovered.pending) == 6
    all_jobs = {j['job_id']: j for j in [json.loads(p.read_text()) for p in (tmp_path/'dispatch').glob('*.json')]}
    recovered.observe([result(all_jobs[k]) for k in recovered.pending.copy()])
    assert recovered.complete
    assert recovered.report()['automatically_passed'] == targets
    assert recovered.report()['requested_total'] == 6
    q.close()


def test_dispatch_survives_local_database_loss(tmp_path):
    class BrokenQueue:
        def enqueue(self, _job):
            raise OSError('local disk died after shared intent')
    dispatched = job(case('remove'))
    with pytest.raises(OSError):
        JournalQueue(BrokenQueue(), tmp_path/'dispatch').enqueue(dispatched)
    saved = [json.loads(p.read_text()) for p in (tmp_path/'dispatch').glob('*.json')]
    scheduler = OperationBalance([dispatched['case']], dict.fromkeys(OPERATIONS, 1), 24)
    scheduler.restore(saved, [])
    assert scheduler.pending == {dispatched['job_id']: 'remove'}
    with pytest.raises(ValueError):
        scheduler.restore([dispatched, dispatched], [])


def test_completed_generation_resume_needs_no_gpu_or_old_db(tmp_path):
    cases = [case(op) for op in OPERATIONS]
    root, local = tmp_path/'shared', tmp_path/'local'
    write_rows(tmp_path/'plans.jsonl', cases)
    for c in cases:
        j = job(c)
        key = hashlib.sha256(j['job_id'].encode()).hexdigest()
        atomic_json(root/'dispatch'/f'{key}.json', j)
        atomic_json(root/'results'/f'{key}.json', {'result': result(j)})
    rows, unused = run_generation(tmp_path/'plans.jsonl', root, local, dict.fromkeys(OPERATIONS, 1),
                                  {'editor_model': 'frozen-editor', 'audit_model': 'frozen-auditor'}, lambda _: None)
    assert len(rows) == 4 and not unused
    assert (root/'complete.json').exists()
    assert not list(root.glob('*.db'))
    second, _ = run_generation(tmp_path/'plans.jsonl', root, local, dict.fromkeys(OPERATIONS, 1),
                               {'editor_model': 'frozen-editor', 'audit_model': 'frozen-auditor'}, lambda _: None)
    assert second == rows


def test_planner_crash_never_reissues_ambiguous_mask(tmp_path):
    contract = {'model': 'frozen', 'manifest': 'hash'}
    journal = PlanningJournal(tmp_path/'plan', contract)
    journal.submit([['sa_1', '0'], ['sa_1', '1']])
    journal.finish(['sa_1', '0'], {'stage': 'plan'}, {'status': 'accepted'}, case('add'))
    resumed = PlanningJournal(tmp_path/'plan', contract, resume=True)
    assert {r['status'] for r in resumed.entries()} == {'submitted', 'complete'}
    with pytest.raises(ValueError, match='already submitted'):
        resumed.submit([['sa_1', '1']])
    with pytest.raises(ValueError, match='changed'):
        PlanningJournal(tmp_path/'plan', {'model': 'different'}, resume=True)


def test_nested_media_are_copied_not_linked_and_move_with_root(tmp_path):
    outside = tmp_path/'old-experiment'
    outside.mkdir()
    Image.new('RGB', (9, 9), 'red').save(outside/'source.png')
    Image.new('L', (9, 9), 255).save(outside/'mask.png')
    root = tmp_path/'dataset'
    row = {'source_image': str(outside/'source.png'), 'mask': str(outside/'mask.png'),
           'nested': {'images': [str(outside/'source.png')], 'execution_mask_path': str(outside/'mask.png')},
           'change_mask': None, 'instruction': 'Make the bag red.'}
    saved = portable(row, root)
    assert saved['nested']['images'][0] == saved['source_image']
    assert not (root/saved['source_image']).is_symlink()
    (outside/'source.png').unlink()
    (outside/'mask.png').unlink()
    moved = tmp_path/'moved'
    root.rename(moved)
    assert portable(saved, moved) == saved
    with pytest.raises(FileNotFoundError):
        portable({'edited_image': '/nonexistent/target.png'}, moved)


def prepared_final(root):
    from sa1b_pipeline.production_state import file_hash
    from sa1b_pipeline.production_generation import REPO
    hashes = {f'sa1b_pipeline/{name}.py': file_hash(REPO/f'sa1b_pipeline/{name}.py')
              for name in ('production_state', 'operation_balance', 'shared_operation_quota')}
    atomic_json(root/'config.json', {'target_total': 16, 'nodes': 4, 'code_hashes': hashes})
    atomic_json(root/'inputs/ready.json', {})
    for rank in range(4):
        directory = root/f'nodes/node{rank}'
        rows = [result(job(case(op, rank))) for op in OPERATIONS]
        path = directory/'target.png'
        path.parent.mkdir(parents=True)
        Image.new('RGB', (3, 3), 'blue').save(path)
        for row in rows:
            row['edited_image'] = path.relative_to(root).as_posix()
        write_rows(directory/'auto_pass.jsonl', rows)
        done = {'rank': rank, 'status': 'complete',
                'automatically_passed': dict.fromkeys(OPERATIONS, 1)}
        atomic_json(directory/'done.json', done)
        atomic_json(directory/'progress.json', done)


def test_four_node_finalize_exact_balanced_unique_and_portable(tmp_path):
    prepared_final(tmp_path)
    finalize(tmp_path)
    rows = read_rows(tmp_path/'final/auto_pass.jsonl')
    assert len(rows) == 16
    assert json.loads((tmp_path/'final/summary.json').read_text())['operations'] == dict.fromkeys(OPERATIONS, 4)


def test_finalize_recovers_mount_lock_failure_preserving_history(tmp_path, monkeypatch):
    import fcntl
    prepared_final(tmp_path)
    def unsupported(*args):
        raise OSError(38, 'Function not implemented')
    monkeypatch.setattr(fcntl, 'flock', unsupported)
    node = tmp_path/'nodes/node2'
    failure = {'error': '[Errno 38] Function not implemented', 'time': 123}
    atomic_json(node/'failure.json', failure)
    atomic_json(node/'progress.json', {'status': 'failed', **failure})
    finalize(tmp_path)
    assert json.loads((node/'progress.json').read_text())['status'] == 'complete'
    assert json.loads((node/'failure.json').read_text()) == failure
    assert not (tmp_path/'final/finalize.lock.d').exists()


@pytest.mark.parametrize('damage', ['missing_rank', 'duplicate_mask', 'wrong_type', 'failed_audit', 'missing_image'])
def test_finalization_refuses_incomplete_or_corrupt_data(tmp_path, damage):
    prepared_final(tmp_path)
    file = tmp_path/'nodes/node3/auto_pass.jsonl'
    rows = read_rows(file)
    if damage == 'missing_rank':
        (tmp_path/'nodes/node3/done.json').unlink()
    elif damage == 'duplicate_mask':
        rows[0]['source_id'] = 'source-0'
    elif damage == 'wrong_type':
        rows[0]['operation'] = 'attribute'
    elif damage == 'failed_audit':
        rows[0]['semantic_audit']['operation_correct'] = False
    else:
        rows[0]['edited_image'] = 'missing.png'
    write_rows(file, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        finalize(tmp_path)
    assert not (tmp_path/'final/auto_pass.jsonl').exists()


def test_collect_does_not_count_uncommitted_wave_twice(tmp_path):
    first = tmp_path/'waves/00000'
    write_rows(first/'generation/validation.jsonl', [result(job(case('add')))])
    write_rows(first/'generation/unused_plans.jsonl', [case('remove')])
    atomic_json(first/'complete.json', {})
    write_rows(tmp_path/'waves/00001/generation/validation.jsonl', [result(job(case('add', 1)))])
    rows, unused = collect(tmp_path)
    assert len(rows) == 1 and unused == [case('remove')]


def test_source_keeps_all_valid_masks_without_quality_selection(tmp_path, monkeypatch):
    import io

    import numpy as np
    from pycocotools import mask as coco

    from sa1b_pipeline import production_sources as sources
    buf = io.BytesIO()
    Image.new('RGB', (10, 10), 'red').save(buf, format='JPEG')
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:2, 1:2] = 1  # A tiny valid mask must not be filtered by size.
    rle = coco.encode(np.asfortranarray(mask))
    rle['counts'] = rle['counts'].decode()
    objects = [{'object_id': 0, 'segmentation': rle}, {'object_id': 1, 'segmentation': rle},
               {'object_id': 2, 'segmentation': None}]
    def record(_base, _entry, subdir):
        if subdir == 'sa1b_tfd_1k':
            return {'key': b'sa_1', 'jpg': buf.getvalue()}
        return {'key': b'sa_1', 'objects_json': json.dumps(objects).encode()}
    monkeypatch.setattr(sources, 'read_record', record)
    row = materialize({'source_id': 'sa_1', 'image': {}, 'annotation': {}}, tmp_path, '/unused',
                      {'source_ids': [], 'image_hashes': [], 'benchmark_sha256': 'hash'})
    assert [o['object_id'] for o in row['candidate']['objects']] == ['0', '1']
    assert len(row['invalid_masks']) == 1


def test_frozen_input_partition_cannot_change(tmp_path):
    from sa1b_pipeline.production_state import file_hash
    from scripts.run_production_4node import load_config
    path = tmp_path/'inputs/node0.jsonl'
    write_rows(path, [{'source_id': 'sa_1'}])
    atomic_json(tmp_path/'inputs/ready.json', {})
    atomic_json(tmp_path/'config.json', {'code_hashes': {}, 'input_hashes': {'inputs/node0.jsonl': file_hash(path)}})
    load_config(tmp_path)
    write_rows(path, [{'source_id': 'sa_2'}])
    with pytest.raises(ValueError, match='partition/exclusion changed'):
        load_config(tmp_path)


def test_four_rank_two_wave_orchestration_refills_only_failed_type(tmp_path, monkeypatch):
    """Exercise the actual rank loop and finalizer with deterministic model doubles."""
    import sys
    from argparse import Namespace

    from scripts import run_production_4node as runner
    root = tmp_path/'dataset'
    model = tmp_path/'fake-model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model_index.json').write_text('{}')
    config = {'target_total': 16, 'target_per_node_per_type': 1, 'batch_sources': 1,
              'planner_model': str(model), 'audit_model': str(model), 'editor_model': str(model),
              'source_root': '/unused', 'code_hashes': {}}
    from sa1b_pipeline.production_state import file_hash
    from sa1b_pipeline.production_generation import REPO
    config['nodes'] = 4
    config['code_hashes'] = {f'sa1b_pipeline/{name}.py': file_hash(REPO/f'sa1b_pipeline/{name}.py')
                             for name in ('production_state', 'operation_balance', 'shared_operation_quota')}
    atomic_json(root/'config.json', config)
    atomic_json(root/'inputs/exclusion.json', {})
    monkeypatch.setattr(runner, 'load_config', lambda _: dict(config))
    monkeypatch.setattr(runner.signal, 'signal', lambda *_args: None)
    monkeypatch.setattr(runner, 'emit', lambda _: None)
    def source(entry, *_args):
        return {'source_id': entry['source_id'], 'status': 'ready',
                'candidate': {'source_id': entry['source_id'], 'objects': [{'object_id': op} for op in OPERATIONS]}}
    monkeypatch.setattr(runner, 'materialize', source)
    planned_types = []
    def plan(_wave, candidates, weights, _config, _progress):
        sid = candidates[0]['source_id']
        types = [op for op in OPERATIONS if weights[op]]
        planned_types.append((sid, types))
        return [{'id': sid+'-'+op, 'source': {'source_id': sid},
                 'units': [{'unit_id': 'U1', 'object_id': op, 'operation': op}]} for op in types]
    monkeypatch.setattr(runner, 'planner', plan)
    def generate(manifest, output, _local, targets, _config, progress):
        scheduler = OperationBalance(read_rows(manifest), targets, 24)
        q = WorkQueue(tmp_path/(output.parent.parent.parent.name+output.parent.name+'.db'))
        scheduler.dispatch(q)
        outcomes = []
        for j, lease in q.claim(24):
            passed = not (j['case']['units'][0]['operation'] == 'remove' and output.parent.name == '00000')
            r = result(j, passed)
            outcomes.append(r)
            q.finish(j['job_id'], lease, r)
        scheduler.observe(outcomes)
        progress(scheduler.report())
        write_rows(output/'validation.jsonl', outcomes)
        write_rows(output/'unused_plans.jsonl', scheduler.unused_cases())
        q.close()
        return outcomes, scheduler.unused_cases()
    monkeypatch.setattr(runner, 'run_generation', generate)
    for rank in range(4):
        write_rows(root/f'inputs/node{rank}.jsonl', [{'source_id': f'sa_{rank}_wave{i}'} for i in range(2)])
        runner.run_node(Namespace(root=root, rank=rank, devices='0,1,2,3,4,5,6,7',
                        planner_python=sys.executable, editor_python=sys.executable,
                        audit_python=sys.executable, local_root=tmp_path/'local'))
    assert len(read_rows(root/'final/auto_pass.jsonl')) == 16
    assert all(ops == ['remove'] for sid, ops in planned_types if sid.endswith('wave1'))
    for rank in range(4):
        progress = json.loads((root/f'nodes/node{rank}/progress.json').read_text())
        assert progress['status'] == 'complete'
        assert progress['automatically_passed'] == dict.fromkeys(OPERATIONS, 1)
