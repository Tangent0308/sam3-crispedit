#!/usr/bin/env python3
"""Single-target wave coordinator (one rank): balance the FINAL exported singles per operation.

Every wave:
* remaining need per operation = quota - singles exported by the finished waves;
* selection units = need / yield x margin. The yield is exported / planned targets per operation so far,
  smoothed toward the single-target v2.1 validation yields; the margin 1.1 + 2 / sqrt(need) gives small needs
  more headroom, so the last waves rarely fall short. The units are scaled to about --wave-targets planned
  targets per wave;
* the production multi selection (select_evidence_groups.py over the per-image menus, these deficits) picks
  images with 2-4 targets, most-needed operations first. Targets of operations whose need is already zero
  (the selector adds them only to reach its two-target minimum) are left out of the wave
  (single_planning.drop_complete_operations); planning keeps one-target groups
  (run_single_validation.py --single-target-groups);
* run_single_validation.py plans and generates the wave (single-target v2.1). A wave whose run was started by
  other code (a hotfix restart) runs again from planning on the same selection.

No further wave runs when the remaining need is at most --stop-gap (never before the first wave). The export
keeps the first ``quota`` singles per operation in wave order (final/singles_balanced.jsonl); the others go to
final/surplus.jsonl (the 4-node merge tops up a short operation of one rank from the others' surplus).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sa1b_pipeline.production_state import atomic_json, file_hash  # noqa: E402

OPERATIONS = ('add', 'remove', 'replace', 'attribute')
# Exported singles / planned targets per operation, single-target v2.1 validation waves 02-04.
PRIOR = {'add': .617, 'remove': .587, 'replace': .604, 'attribute': .547}
PSEUDO = 20.0
TARGETS_PER_GROUP = 3.4      # planned targets per selected group with all four operations needed (wave04)


def rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def waves(root):
    """Wave directories in wave order (wave100 after wave99)."""
    return sorted((w for w in Path(root).glob('wave*') if w.name[4:].isdigit()), key=lambda w: int(w.name[4:]))


def wave_counts(wave: Path, cache=True) -> tuple[Counter, Counter]:
    """Planned targets and exported singles per operation of one wave (finished waves: cached in counts.json)."""
    path = wave / 'counts.json'
    if cache and path.exists():
        c = json.loads(path.read_text())
        return Counter(c['planned']), Counter(c['exported'])
    planned = Counter(op for r in rows(wave / 'inputs/multi/inputs/selected.jsonl') for op in r['operations'])
    exported = Counter(s['operation'] for s in rows(wave / 'run/single/singles.jsonl'))
    if cache and (wave / 'run/session.json').exists():
        atomic_json(path, {'planned': dict(planned), 'exported': dict(exported)})
    return planned, exported


def code_hashes() -> dict:
    """The code fingerprint run_single_validation.py freezes in <wave>/run/single_contract.json."""
    return {str(f.relative_to(REPO)): file_hash(f) for folder in ('sa1b_pipeline', 'scripts', 'utils')
            for f in (REPO / folder).rglob('*.py')}


def supersede_stale_run(wave: Path) -> Path | None:
    """An unfinished wave started by other code cannot resume (its run and every group checkpoint freeze the
    code). Set its run directory aside (kept for inspection) so that the wave runs again from planning on the
    same selection; finished waves are never touched."""
    run = wave / 'run'
    contract = run / 'single_contract.json'
    if (run / 'session.json').exists() or not contract.exists():
        return None
    if json.loads(contract.read_text()).get('code') == code_hashes():
        return None
    kept = wave / f'run.superseded.{int(time.time())}'
    run.rename(kept)
    print(json.dumps({'wave': wave.name, 'superseded_run': str(kept), 'reason': 'code changed since the wave started'}),
          flush=True)
    return kept


def margin(need: float) -> float:
    return 1.1 + 2 / math.sqrt(max(1.0, need))


def plan_wave(quota: int, planned: Counter, exported: Counter, *, wave_targets: float, gap: int = 0,
              prior: dict = PRIOR) -> dict:
    """Need, yield and selection units per operation for the next wave (empty units: stop)."""
    need = {op: max(0, quota - exported[op]) for op in OPERATIONS}
    stopped = 0 < sum(need.values()) <= gap
    yields = {op: (exported[op] + PSEUDO * prior[op]) / (planned[op] + PSEUDO) for op in OPERATIONS}
    units = {op: (need[op] / max(.1, yields[op]) * margin(need[op]) if need[op] and not stopped else 0.0)
             for op in OPERATIONS}
    total = sum(units.values())
    scale = min(1.0, wave_targets / total) if total else 1.0
    units = {op: u * scale for op, u in units.items()}
    active = sum(1 for u in units.values() if u > 0)
    return {'need': need, 'yield': {op: round(y, 3) for op, y in yields.items()},
            'units': {op: round(u, 2) for op, u in units.items()}, 'scale': round(scale, 4),
            'groups_estimate': math.ceil(sum(units.values()) / max(1.0, min(active, TARGETS_PER_GROUP))) if active else 0,
            **({'stopped_by_gap': True} if stopped else {})}


def filter_selection(wave: Path, need: dict) -> dict:
    """Leave targets of complete operations out of the wave's selection (kept beside as selected_full.jsonl)."""
    from sa1b_pipeline.single_planning import drop_complete_operations
    inputs = wave / 'inputs/multi/inputs'
    full = inputs / 'selected_full.jsonl'
    if not full.exists():
        (inputs / 'selected.jsonl').rename(full)
    if (inputs / 'selected.jsonl').exists():
        return json.loads((inputs / 'selection_filter.json').read_text())
    kept, dropped = drop_complete_operations(rows(full), need)
    sids = {r['candidate']['source_id'] for r in kept}
    manifest = [c for c in rows(inputs / 'manifest.jsonl') if c['source_id'] in sids]
    info = {'groups_selected': len(rows(full)), 'groups_kept': len(kept), 'targets_dropped': dict(dropped),
            'targets_kept': dict(Counter(op for r in kept for op in r['operations'])),
            'targets_per_group': {str(n): c for n, c in sorted(Counter(len(r['operations']) for r in kept).items())}}
    atomic_json(inputs / 'selection_filter.json', info)
    for name, data in (('manifest.jsonl', manifest), ('selected.jsonl', kept)):   # selected.jsonl last: the marker
        temp = inputs / f'{name}.partial'
        temp.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data))
        temp.replace(inputs / name)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True, help='this rank\'s waves directory')
    p.add_argument('--sources-root', type=Path, required=True)
    p.add_argument('--reservation', type=Path, required=True)
    p.add_argument('--exclude', type=Path, nargs='*', default=[])
    p.add_argument('--affordance-cache', type=Path, required=True)
    p.add_argument('--quota-per-op', type=int, required=True, help='exported singles wanted per operation')
    p.add_argument('--wave-targets', type=float, default=1700, help='planned targets per wave (about 1,000 singles)')
    p.add_argument('--wave-cap-groups', type=int, default=3000, help='at most this many selected images per wave')
    p.add_argument('--max-waves', type=int, default=1000)
    p.add_argument('--stop-gap', type=int, default=0,
                   help='no further wave when the total remaining need is at most this (never before the first wave)')
    p.add_argument('--seed', type=int, default=20261027)
    p.add_argument('--menu-dir', type=Path, required=True)
    p.add_argument('--selection-workers', type=int, default=32)
    p.add_argument('--selection-flags-json', type=json.loads, default=[],
                   help='JSON list of flags forwarded to select_evidence_groups.py, e.g. ["--worn-check"]')
    p.add_argument('--service-root', type=Path, default=Path('/tmp/sa1b-single-services'))
    p.add_argument('--max-planning-chunks', type=int, default=4, help='planning parts per wave: one per 100 groups')
    p.add_argument('--groups-in-flight', type=int, default=24)
    p.add_argument('--before-wave-json', type=json.loads,
                   help='JSON argv run before each wave\'s selection with "--plan <wave>/plan.json" appended '
                        '(production: prepare enough labelled sources and menus)')
    p.add_argument('validation_args', nargs=argparse.REMAINDER,
                   help='after "--": arguments forwarded to run_single_validation.py (models, pythons)')
    args = p.parse_args(argv)
    validation_args = [a for a in args.validation_args if a != '--']
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'waves': []}
    planned, exported = Counter(), Counter()
    seconds = 0.0
    for k in range(args.max_waves):
        wave = root / f'wave{k + 1:02d}'
        if k:  # the previous wave is finished: the loop moves on only after a complete wave
            s, e = wave_counts(root / f'wave{k:02d}')
            planned.update(s)
            exported.update(e)
            seconds += json.loads((root / f'wave{k:02d}/run/session.json').read_text()).get('seconds', 0)
        plan = plan_wave(args.quota_per_op, planned, exported, wave_targets=args.wave_targets,
                         gap=args.stop_gap if k else 0)
        print(json.dumps({'wave': k + 1, 'quota_per_op': args.quota_per_op, 'exported_so_far': dict(exported),
                          'plan': plan}), flush=True)
        atomic_json(root / 'progress.json', {
            'updated': time.time(), 'waves_finished': k, 'quota_per_op': args.quota_per_op,
            'session_seconds': round(seconds, 1), 'planned': dict(planned), 'exported': dict(exported),
            'balanced': {op: min(args.quota_per_op, exported[op]) for op in OPERATIONS},
            'next_wave': {x: plan[x] for x in ('need', 'units', 'groups_estimate')}})
        if not any(plan['units'].values()):
            break
        if len(ledger['waves']) <= k:
            ledger['waves'].append({'wave': k + 1, 'plan': plan, 'started': time.time()})
            atomic_json(ledger_path, ledger)
        plan = ledger['waves'][k]['plan']   # a resumed wave keeps the plan it was selected with
        if not (wave / 'inputs/evidence_selection_summary.json').exists():
            if args.before_wave_json:
                zero = dict.fromkeys(OPERATIONS, 0)
                atomic_json(wave / 'plan.json', {'wave': k + 1, 'plan': {
                    'twins': {'groups': 0, 'deficits': zero},
                    'multi': {'groups': plan['groups_estimate'], 'deficits': plan['units']}}})
                subprocess.run([*args.before_wave_json, '--plan', str(wave / 'plan.json')], check=True)
            previous = [root / f'wave{j + 1:02d}' / 'inputs/evidence_selection_summary.json' for j in range(k)]
            cap = min(args.wave_cap_groups, math.ceil(sum(plan['units'].values())) + 1)
            subprocess.run([sys.executable, str(REPO / 'scripts/select_evidence_groups.py'),
                            '--sources-root', str(args.sources_root), '--inputs-root', str(wave / 'inputs'),
                            '--reservation', str(args.reservation), '--affordance-cache', str(args.affordance_cache),
                            '--cache-only', '--seed', str(args.seed), '--twins', '0', '--multi', str(cap),
                            '--initial-deficits', json.dumps({'twins': dict.fromkeys(OPERATIONS, 0),
                                                              'multi': plan['units']}),
                            '--menu-dir', str(args.menu_dir), '--workers', str(args.selection_workers),
                            *args.selection_flags_json, '--exclude', *[str(x) for x in list(args.exclude) + previous]],
                           check=True)
        info = filter_selection(wave, plan['need'])
        if not info['groups_kept']:
            print('selection found no eligible groups; stopping', flush=True)
            break
        if not (wave / 'run/session.json').exists():
            supersede_stale_run(wave)
            env = {key: value for key, value in os.environ.items() if key != 'CUDA_VISIBLE_DEVICES'}
            chunks = max(1, min(args.max_planning_chunks, math.ceil(info['groups_kept'] / 100)))
            subprocess.run([sys.executable, str(REPO / 'scripts/run_single_validation.py'),
                            '--inputs-root', str(wave / 'inputs'), '--output-root', str(wave / 'run'),
                            '--reservation', str(args.reservation), '--single-target-groups',
                            '--service-root', str(args.service_root / wave.name), '--planning-chunks', str(chunks),
                            '--groups-in-flight', str(args.groups_in_flight), *validation_args],
                           check=True, env=env)
        s, e = wave_counts(wave)
        session = json.loads((wave / 'run/session.json').read_text())
        ledger['waves'][k].update(selection=info, planned=dict(s), exported=dict(e), seconds=session.get('seconds'),
                                  finished=time.time())
        atomic_json(ledger_path, ledger)
    export(root, args.quota_per_op)


def export(root, quota: int, out=None) -> dict:
    """Balanced export of all finished waves: the first ``quota`` singles per operation in wave order."""
    root = Path(root)
    finished = [w for w in waves(root) if (w / 'run/session.json').exists()]
    balanced, surplus, seen = [], [], set()
    count, planned, seconds = Counter(), Counter(), 0.0
    for wave in finished:
        seconds += json.loads((wave / 'run/session.json').read_text()).get('seconds', 0)
        planned.update(wave_counts(wave, cache=False)[0])
        for s in rows(wave / 'run/single/singles.jsonl'):
            if s['job_id'] in seen:
                continue
            seen.add(s['job_id'])
            s = dict(s, wave=wave.name)
            if count[s['operation']] < quota:
                balanced.append(s)
                count[s['operation']] += 1
            else:
                surplus.append(s)
    final = Path(out) if out else root / 'final'
    final.mkdir(parents=True, exist_ok=True)
    for name, data in (('singles_balanced.jsonl', balanced), ('surplus.jsonl', surplus)):
        temp = final / f'{name}.partial'
        with temp.open('w') as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        temp.replace(final / name)
    summary = {'quota_per_op': quota, 'waves': len(finished), 'session_seconds': round(seconds, 1),
               'planned_targets': dict(planned), 'balanced': {op: count[op] for op in OPERATIONS},
               'balanced_total': len(balanced), 'surplus': dict(Counter(s['operation'] for s in surplus)),
               'shortfall': {op: max(0, quota - count[op]) for op in OPERATIONS}}
    atomic_json(final / 'summary.json', summary)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == '__main__':
    main()
