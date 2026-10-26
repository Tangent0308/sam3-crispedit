import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_single_waves as rw  # noqa: E402
import run_single_production_4node as prod  # noqa: E402


def test_plan_scales_to_the_wave_and_favours_scarce_operations():
    plan = rw.plan_wave(6250, Counter(), Counter(), wave_targets=1700)
    assert abs(sum(plan['units'].values()) - 1700) < 1 and plan['scale'] < 1
    # equal need: the operation with the lowest yield gets the most units
    assert max(plan['units'], key=plan['units'].get) == 'attribute'
    later = rw.plan_wave(6250, Counter({'add': 9000, 'remove': 6000, 'replace': 7000, 'attribute': 8000}),
                         Counter({'add': 6250, 'remove': 3500, 'replace': 4300, 'attribute': 4400}), wave_targets=1700)
    assert later['need']['add'] == 0 and later['units']['add'] == 0
    assert max(later['units'], key=later['units'].get) == 'remove'   # the largest remaining need
    small = rw.plan_wave(100, Counter({op: 150 for op in rw.OPERATIONS}),
                         Counter({'add': 100, 'remove': 96, 'replace': 100, 'attribute': 100}), wave_targets=1700)
    assert small['need'] == {'add': 0, 'remove': 4, 'replace': 0, 'attribute': 0} and small['scale'] == 1
    assert small['units']['remove'] > 4 / small['yield']['remove'] * 1.9   # small need: wide margin
    stopped = rw.plan_wave(100, Counter({op: 150 for op in rw.OPERATIONS}),
                           Counter({'add': 100, 'remove': 98, 'replace': 100, 'attribute': 100}), wave_targets=1700, gap=2)
    assert stopped.get('stopped_by_gap') and not any(stopped['units'].values())


def _row(sid, ops):
    return {'candidate': {'source_id': sid}, 'objects': [{'object_id': i} for i in range(len(ops))], 'operations': ops}


def test_filter_selection_drops_complete_operations_and_resumes(tmp_path):
    inputs = tmp_path / 'wave01/inputs/multi/inputs'
    inputs.mkdir(parents=True)
    rows = [_row('a', ['remove', 'add']), _row('b', ['add', 'attribute']), _row('c', ['remove', 'replace', 'add'])]
    (inputs / 'selected.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (inputs / 'manifest.jsonl').write_text(''.join(json.dumps(r['candidate']) + '\n' for r in rows))
    need = {'add': 0, 'remove': 10, 'replace': 3, 'attribute': 0}
    info = rw.filter_selection(tmp_path / 'wave01', need)
    assert info['groups_selected'] == 3 and info['groups_kept'] == 2
    assert info['targets_dropped'] == {'add': 3, 'attribute': 1}
    kept = rw.rows(inputs / 'selected.jsonl')
    assert [r['operations'] for r in kept] == [['remove'], ['remove', 'replace']]
    assert [c['source_id'] for c in rw.rows(inputs / 'manifest.jsonl')] == ['a', 'c']
    assert len(rw.rows(inputs / 'selected_full.jsonl')) == 3
    assert rw.filter_selection(tmp_path / 'wave01', need) == info   # resume: same result, no second filtering


def _single(job, op):
    return {'job_id': job, 'operation': op, 'provenance': {'kind': 'multi'}}


def _finished_wave(root, name, singles, planned=()):
    w = root / name
    (w / 'run/single').mkdir(parents=True)
    (w / 'inputs/multi/inputs').mkdir(parents=True)
    (w / 'run/session.json').write_text(json.dumps({'seconds': 100.0}))
    (w / 'run/single/singles.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in singles))
    (w / 'inputs/multi/inputs/selected.jsonl').write_text(''.join(json.dumps(_row(f'{name}{i}', ops)) + '\n'
                                                                  for i, ops in enumerate(planned)))
    return w


def test_export_keeps_the_first_quota_per_operation_in_wave_order(tmp_path):
    _finished_wave(tmp_path, 'wave01', [_single('g1/U1', 'add'), _single('g1/U2', 'remove'), _single('g2/U1', 'add')],
                   planned=[['add', 'remove'], ['add']])
    _finished_wave(tmp_path, 'wave02', [_single('g3/U1', 'add'), _single('g3/U2', 'remove'), _single('g1/U1', 'add')])
    summary = rw.export(tmp_path, 2)
    balanced = rw.rows(tmp_path / 'final/singles_balanced.jsonl')
    assert [(s['job_id'], s['wave']) for s in balanced] == [('g1/U1', 'wave01'), ('g1/U2', 'wave01'),
                                                             ('g2/U1', 'wave01'), ('g3/U2', 'wave02')]
    assert [s['job_id'] for s in rw.rows(tmp_path / 'final/surplus.jsonl')] == ['g3/U1']
    assert summary['balanced'] == {'add': 2, 'remove': 2, 'replace': 0, 'attribute': 0}
    assert summary['shortfall']['replace'] == 2 and summary['planned_targets'] == {'add': 2, 'remove': 1}
    planned, exported = rw.wave_counts(tmp_path / 'wave01')
    assert (tmp_path / 'wave01/counts.json').exists() and exported == Counter({'add': 2, 'remove': 1})


def test_stale_run_is_set_aside_only_when_unfinished_and_code_differs(tmp_path, monkeypatch):
    wave = tmp_path / 'wave01'
    (wave / 'run').mkdir(parents=True)
    (wave / 'run/single_contract.json').write_text(json.dumps({'code': {'a.py': 'old'}}))
    monkeypatch.setattr(rw, 'code_hashes', lambda: {'a.py': 'old'})
    assert rw.supersede_stale_run(wave) is None
    monkeypatch.setattr(rw, 'code_hashes', lambda: {'a.py': 'new'})
    (wave / 'run/session.json').write_text('{}')
    assert rw.supersede_stale_run(wave) is None          # finished waves are never touched
    (wave / 'run/session.json').unlink()
    kept = rw.supersede_stale_run(wave)
    assert kept and kept.exists() and not (wave / 'run').exists()


def _config(world=2, quota=2):
    return {'world_size': world, 'quota': {'unit': 'singles', 'total': quota * 4 * world,
                                           'per_rank_per_operation': quota},
            'waves': {'wave_targets': 100, 'wave_cap_groups': 50, 'stop_gap': 2, 'max_waves': 3,
                      'max_planning_chunks': 4, 'groups_in_flight': 24},
            'pool': {'selection_workers': 8}, 'selection_flags': prod.SELECTION_FLAGS, 'seed': 7,
            'models': {'planner': '/m/planner', 'editor': '/m/editor', 'site': '/m/site'}}


def test_merge_tops_up_a_short_operation_from_other_ranks_surplus(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps(_config()))
    for r, singles in ((0, [_single('a/U1', 'add'), _single('a/U2', 'remove')]),
                       (1, [_single('b/U1', 'add'), _single('b/U2', 'add'), _single('b/U3', 'add'),
                            _single('c/U1', 'remove'), _single('c/U2', 'remove'), _single('c/U3', 'remove')])):
        node = tmp_path / f'nodes/node{r}'
        _finished_wave(node / 'waves', 'wave01', singles)
        rw.export(node / 'waves', 2)
        (node / 'done.json').write_text('{}')
    prod.merge(SimpleNamespace(root=tmp_path, partial=False))
    out = rw.rows(tmp_path / 'final/singles_balanced.jsonl')
    summary = json.loads((tmp_path / 'final/summary.json').read_text())
    # rank 0 has one add and one remove (quota 2); rank 1's third add and third remove fill the totals (4 each)
    assert summary['balanced'] == {'add': 4, 'remove': 4, 'replace': 0, 'attribute': 0}
    assert summary['topped_up_from_surplus'] == {'add': 1, 'remove': 1} and summary['surplus_left'] == {}
    assert sum(1 for s in out if s.get('topped_up')) == 2 and {s['rank'] for s in out} == {0, 1}


def rw_args(cmd):
    """The coordinator command parses with run_single_waves' own argument parser."""
    captured = {}

    def fake_main(args):
        captured['args'] = args
        raise SystemExit(0)
    import argparse
    real = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = lambda self, argv=None: fake_main(real(self, argv))
        with pytest.raises(SystemExit):
            rw.main(cmd[2:])
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured['args']


def test_coordinator_command_and_shard_avoidance(tmp_path):
    cmd = prod.coordinator(tmp_path, _config(), 1, '/py/mllm', '/py/editor')
    assert cmd[1].endswith('scripts/run_single_waves.py') and '--quota-per-op' in cmd
    assert cmd[cmd.index('--seed') + 1] == '8' and cmd[cmd.index('--service-root') + 1].endswith('node1')
    assert json.loads(cmd[cmd.index('--selection-flags-json') + 1]) == prod.SELECTION_FLAGS
    parsed = rw_args(cmd)
    assert parsed.selection_flags_json == prod.SELECTION_FLAGS and parsed.quota_per_op == 2
    hook = json.loads(cmd[cmd.index('--before-wave-json') + 1])
    assert hook[-3:] == ['prepare-pool', '--rank', '1'] and cmd[cmd.index('--') + 1] == '--planner-model'
    order = tmp_path / 'order.json'
    order.write_text(json.dumps({'0': [5, 6, 7], '1': [8, 9, 10]}))
    assert prod.avoided_shards(order, 2) == {5, 6, 8, 9}
    from sa1b_pipeline.production_pool import shard_order
    mine = shard_order(3, 4, exclude={5, 6, 8, 9, 667})
    assert not ({s for v in mine.values() for s in v} & {5, 6, 8, 9, 667})


def test_rank_status_progress_and_eta(tmp_path):
    c = _config(world=1, quota=10)
    node = tmp_path / 'nodes/node0'
    (node / 'waves').mkdir(parents=True)
    (node / 'waves/progress.json').write_text(json.dumps({
        'updated': 1.0, 'waves_finished': 2, 'exported': {'add': 12, 'remove': 6, 'replace': 8, 'attribute': 4},
        'balanced': {'add': 10, 'remove': 6, 'replace': 8, 'attribute': 4}}))
    (node / 'waves/ledger.json').write_text(json.dumps({'waves': [{'seconds': 540}, {'seconds': 540}]}))
    s = prod.rank_status(tmp_path, c, 0)
    assert s['progress'] == 0.7 and s['state'] == 'not_started'
    assert s['eta_hours'] == pytest.approx(round((0.3 / 0.35) * 600 / 3600, 1))


def test_warm_pool_feeds_the_first_shards_of_every_rank_once(tmp_path, monkeypatch):
    from sa1b_pipeline import production_pool
    fed = []

    class FakePool:
        def __init__(self, root, rank, config):
            self.rank, self.order = rank, [[10, 11, 12], [20, 21, 22]][rank]

        def done(self, kind):
            return [10] if kind == 'shards' and self.rank == 0 else []

        def feed(self, shards):
            fed.append((self.rank, shards))
    monkeypatch.setattr(production_pool, 'Pool', FakePool)
    monkeypatch.setattr(production_pool, 'write_tables', lambda root, shards: None)
    (tmp_path / 'config.json').write_text(json.dumps(dict(_config(world=2), gpus_per_node=8)))
    prod.warm_pool(SimpleNamespace(root=tmp_path, ranks=None, shards_per_rank=2, parallel=1, feed_only=True))
    assert sorted(fed) == [(0, [11]), (1, [20]), (1, [21])]
