import json
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_evidence_waves as rw  # noqa: E402


def _group(gid, *ops):
    return {'multi_id': gid, 'units': [{'unit_id': f'U{i}', 'operation': op} for i, op in enumerate(ops, 1)]}


def test_multi_export_takes_groups_with_surplus_to_fill_a_short_operation():
    # Colour edits only occur next to additions that are already at quota (full run 4 pattern).
    groups = [_group('a', 'add', 'add', 'remove'), _group('b', 'add', 'remove', 'replace'),
              _group('c', 'replace', 'remove'), _group('d', 'attribute', 'add'),
              _group('e', 'attribute', 'replace')]
    greedy, gcount = rw._greedy_multi(groups, 2)
    assert gcount['attribute'] < 2                       # the old export left colour short
    chosen, count = rw.multi_export(groups, 2)
    assert all(count[op] >= 2 for op in rw.OPERATIONS)
    # Among subsets reaching every quota, the least overshoot wins.
    assert sum(max(0, count[op] - 2) for op in rw.OPERATIONS) == min(
        sum(max(0, c[op] - 2) for op in rw.OPERATIONS)
        for c in [Counter(u['operation'] for g in sub for u in g['units'])
                  for sub in ([groups[i] for i in range(5) if m >> i & 1] for m in range(32))]
        if all(c[op] >= 2 for op in rw.OPERATIONS))


def test_plan_counts_only_exportable_multi_targets():
    targets = {'twins': 1, 'multi': 2}
    zero = {kd: Counter() for kd in ('twins', 'multi')}
    accepted = {'twins': Counter({op: 1 for op in rw.OPERATIONS}), 'multi': Counter({op: 2 for op in rw.OPERATIONS})}
    caps = {'twins': 24, 'multi': 20}
    plan = rw.plan_wave(targets, zero, accepted, rw.PRIOR, caps)
    assert plan['multi']['groups'] == 0
    plan = rw.plan_wave(targets, zero, accepted, rw.PRIOR, caps,
                        exportable={'multi': Counter({'add': 2, 'remove': 2, 'replace': 2, 'attribute': 1})})
    assert plan['multi']['need'] == {'add': 0, 'remove': 0, 'replace': 0, 'attribute': 1} and plan['multi']['groups'] >= 1
    assert plan['twins']['groups'] == 0                    # twins still use accepted counts


def test_stop_gap_skips_a_wave_for_a_small_remaining_need_only():
    targets = {'twins': 4, 'multi': 6}
    zero = {kd: Counter() for kd in ('twins', 'multi')}
    accepted = {'twins': Counter({'add': 3, 'remove': 4, 'replace': 4, 'attribute': 4}),
                'multi': Counter({'add': 6, 'remove': 3, 'replace': 6, 'attribute': 6})}
    caps = {'twins': 24, 'multi': 20}
    plan = rw.plan_wave(targets, zero, accepted, rw.PRIOR, caps)
    assert plan['twins']['groups'] > 0 and plan['multi']['groups'] > 0
    plan = rw.plan_wave(targets, zero, accepted, rw.PRIOR, caps, gaps={'twins': 2, 'multi': 2})
    assert plan['twins']['groups'] == 0 and plan['twins']['stopped_by_gap']       # 1 twin group short: stop
    assert plan['multi']['groups'] > 0 and 'stopped_by_gap' not in plan['multi']  # 3 multi targets short: go on
    done = {'twins': Counter({op: 4 for op in rw.OPERATIONS}), 'multi': Counter({op: 6 for op in rw.OPERATIONS})}
    plan = rw.plan_wave(targets, zero, done, rw.PRIOR, caps, gaps={'twins': 2, 'multi': 2})
    assert not any(plan[kd].get('stopped_by_gap') for kd in plan)                 # nothing left: not a stop


def _objective(counts, target, n):
    return (min(min(counts[op], target) for op in rw.OPERATIONS), sum(min(counts[op], target) for op in rw.OPERATIONS),
            -sum(max(0, counts[op] - target) for op in rw.OPERATIONS), -n)


def test_typed_export_reaches_the_exact_objective_and_scales():
    rng = random.Random(0)
    for _ in range(200):
        groups = [_group(str(i), *rng.sample(rw.OPERATIONS, rng.randint(2, 4))) for i in range(rng.randint(1, 12))]
        if rng.random() < .3:
            groups.append(_group('dup', 'add', 'add', 'remove'))
        target = rng.randint(1, 6)
        exact, ce = rw.multi_export(groups, target)              # <= EXACT_MULTI_GROUPS: exact search
        typed, ct = rw._typed_multi_export(groups, target)
        assert _objective(ce, target, len(exact)) == _objective(ct, target, len(typed))
    many = [_group(str(i), *rng.sample(rw.OPERATIONS, rng.choice([2, 2, 3, 4]))) for i in range(20000)]
    chosen, counts = rw.multi_export(many, 3000)
    assert all(counts[op] == 3000 for op in rw.OPERATIONS) and len({g['multi_id'] for g in chosen}) == len(chosen)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_wave_counts_are_cached_once_a_wave_is_finished(tmp_path):
    wave = tmp_path / 'wave07'
    _write(wave / 'inputs/twins/inputs/selected.jsonl', [{'operation': 'add'}, {'operation': 'remove'}])
    _write(wave / 'inputs/multi/inputs/selected.jsonl', [{'operations': ['add', 'remove']}])
    _write(wave / 'run/twins/generation/accepted.jsonl', [{'twin_id': 't', 'operation': 'add'}])
    _write(wave / 'run/multi/generation/accepted.jsonl', [_group('m', 'add', 'remove')])
    sel, acc, groups = rw.wave_counts(wave)
    assert not (wave / 'counts.json').exists()               # not finished: nothing cached
    (wave / 'run/session.json').write_text('{"seconds": 5}')
    assert rw.wave_counts(wave)[:2] == (sel, acc)
    (wave / 'run/multi/generation/accepted.jsonl').unlink()  # later reads come from counts.json
    s2, a2, g2 = rw.wave_counts(wave)
    assert (s2, a2) == (sel, acc) and [u['operation'] for u in g2['multi'][0]['units']] == ['add', 'remove']
    assert rw.wave_counts(wave, cache=False)[2]['multi'] == []
    for name in ('wave100', 'wave09', 'wave10'):
        (tmp_path / name).mkdir()
    assert [w.name for w in rw.waves(tmp_path)] == ['wave07', 'wave09', 'wave10', 'wave100']


def test_multi_group_target_follows_targets_per_accepted_group():
    args = SimpleNamespace(multi_groups=1000, multi_targets_per_op=None)
    assert rw.multi_target(args, 0, 0) == 588                 # prior 2.35 targets per group
    assert rw.multi_target(args, 3000 - 8 * 2.35, 1000 - 8) == 750
    assert rw.multi_target(SimpleNamespace(multi_groups=None, multi_targets_per_op=7), 10, 3) == 7
    # The export prefers fuller groups; the target follows its targets per group so that it ships ~the groups asked.
    groups = [_group(str(i), 'add', 'remove', 'replace', 'attribute') for i in range(300)]
    groups += [_group(f'p{i}', *pair) for i, pair in enumerate([('add', 'remove'), ('replace', 'attribute')] * 300)]
    args = SimpleNamespace(multi_groups=200, multi_targets_per_op=None)
    units = sum(len(g['units']) for g in groups)
    plain = rw.multi_target(args, units, len(groups))
    target = rw.multi_target(args, units, len(groups), groups)
    assert target > plain
    chosen, _ = rw.multi_export(groups, target)
    assert len(chosen) >= 200 > len(rw.multi_export(groups, plain)[0])


def test_wave_budget_splits_one_wave_by_remaining_need():
    zero = {kd: Counter() for kd in ('twins', 'multi')}
    caps = {'twins': 10_000, 'multi': 10_000}
    plan = rw.plan_wave({'twins': 3125, 'multi': 14700}, zero, zero, rw.PRIOR, caps, margin=1.0)
    plan = rw.fit_budget(plan, 30_000, caps)
    cost = sum(rw.GROUP_COST[kd] * plan[kd]['groups'] for kd in plan)
    assert 29_000 < cost <= 30_000
    ratio = plan['twins']['groups'] / plan['multi']['groups']
    wanted = plan['twins']['budget']['wanted_groups'] / plan['multi']['budget']['wanted_groups']
    assert abs(ratio - wanted) < .02
    small = rw.plan_wave({'twins': 3, 'multi': 2}, zero, zero, rw.PRIOR, caps, margin=1.0)
    assert rw.fit_budget(small, 30_000, caps)['twins']['budget']['scale'] == 1.0   # small needs: not scaled


def test_unfinished_wave_from_other_code_is_set_aside_and_finished_waves_kept(tmp_path):
    wave = tmp_path / 'wave03'
    (wave / 'run').mkdir(parents=True)
    (wave / 'run/dual_contract.json').write_text(json.dumps({'code': {'sa1b_pipeline/x.py': 'old'}}))
    kept = rw.supersede_stale_run(wave)
    assert kept is not None and kept.exists() and not (wave / 'run').exists()
    (wave / 'run').mkdir()
    (wave / 'run/dual_contract.json').write_text(json.dumps({'code': rw.code_hashes()}))
    assert rw.supersede_stale_run(wave) is None and (wave / 'run').exists()        # same code: resume
    (wave / 'run/dual_contract.json').write_text(json.dumps({'code': {'old': 'x'}}))
    (wave / 'run/session.json').write_text('{}')
    assert rw.supersede_stale_run(wave) is None and (wave / 'run').exists()        # finished: never touched
