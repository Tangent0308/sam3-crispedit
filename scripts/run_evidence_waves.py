#!/usr/bin/env python3
"""Wave coordinator: balance FINAL accepted edit types for twins and multi.

Each wave selects new source images (scripts/select_evidence_groups.py) with
per-operation selection units = remaining accepted need / estimated yield,
runs the full pipeline (scripts/run_dual_validation.py) on that wave, and
counts what was actually accepted. Yields start from priors and are updated
from every finished wave (Laplace-smoothed), so operations that fail more are
selected more often instead of ending under-represented. The final export
keeps a balanced subset (twins: per-operation cap; multi: greedy group subset).

A wave is an ordinary dual-validation root, so a crash resumes inside the wave.

Production scale (scripts/run_evidence_production_4node.py): a finished wave keeps its counts in
counts.json, the multi export of many groups is solved over composition types, ``--multi-groups``
sets the per-operation multi target from a group count, and ``--before-wave-json`` runs a command
(the source-pool preparation) before each wave's selection.
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
from sa1b_pipeline.production_state import atomic_json  # noqa: E402

OPERATIONS = ('add', 'remove', 'replace', 'attribute')
PRIOR = {'twins': {'add': .3, 'remove': .5, 'replace': .35, 'attribute': .7},
         'multi': {'add': .5, 'remove': .55, 'replace': .5, 'attribute': .6}}
PSEUDO = 4.0
# --multi-groups: targets per accepted multi group before data (full runs 3-5: 2.36, 2.40, 2.21).
MULTI_TARGETS_PER_GROUP, MULTI_PSEUDO_GROUPS = 2.35, 8.0


def rows(path):
    p = Path(path)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def wave_counts(wave, cache=True):
    """Selected and accepted units per kind/operation for one finished wave, and its accepted groups.

    With ``cache`` a finished wave (run/session.json) keeps the counts in counts.json, groups reduced to
    their operations (enough for planning and multi_export); the final export reads full groups."""
    path = wave / 'counts.json'
    if cache and path.exists():
        c = json.loads(path.read_text())
        return ({kd: Counter(v) for kd, v in c['selected'].items()}, {kd: Counter(v) for kd, v in c['accepted'].items()},
                {'twins': [{'operation': op} for op in c['twin_operations']],
                 'multi': [{'units': [{'operation': op} for op in ops]} for ops in c['multi_operations']]})
    sel = {'twins': Counter(), 'multi': Counter()}
    acc = {'twins': Counter(), 'multi': Counter()}
    groups = {'twins': [], 'multi': []}
    for r in rows(wave / 'inputs/twins/inputs/selected.jsonl'):
        sel['twins'][r['operation']] += 1
    for r in rows(wave / 'inputs/multi/inputs/selected.jsonl'):
        sel['multi'].update(r['operations'])
    for g in rows(wave / 'run/twins/generation/accepted.jsonl'):
        acc['twins'][g['operation']] += 1
        groups['twins'].append(g)
    for g in rows(wave / 'run/multi/generation/accepted.jsonl'):
        acc['multi'].update(u['operation'] for u in g['units'])
        groups['multi'].append(g)
    if cache and (wave / 'run/session.json').exists():
        atomic_json(path, {'selected': {kd: dict(v) for kd, v in sel.items()},
                           'accepted': {kd: dict(v) for kd, v in acc.items()},
                           'twin_operations': [g['operation'] for g in groups['twins']],
                           'multi_operations': [[u['operation'] for u in g['units']] for g in groups['multi']]})
    return sel, acc, groups


def multi_target(args, accepted_units, accepted_groups, groups=None):
    """Accepted multi targets wanted per operation: fixed, or from --multi-groups and the mean number of
    targets per accepted group so far (prior MULTI_TARGETS_PER_GROUP).

    With ``groups`` (the accepted groups), the export's own targets per group count too: multi_export
    prefers fuller groups, so a target from the plain mean ships fewer groups than asked (smoke run:
    98 of 100). The larger of the two targets is used."""
    if args.multi_groups is None:
        return args.multi_targets_per_op
    mean = ((accepted_units + MULTI_TARGETS_PER_GROUP * MULTI_PSEUDO_GROUPS)
            / (accepted_groups + MULTI_PSEUDO_GROUPS))
    target = math.ceil(args.multi_groups * mean / len(OPERATIONS))
    if groups:
        chosen, counts = multi_export(groups, target)
        if chosen:
            target = max(target, math.ceil(args.multi_groups * sum(counts.values()) / len(chosen) / len(OPERATIONS)))
    return target


def plan_wave(targets, selected, accepted, prior, caps, margin=1.0, exportable=None, gaps=None):
    """Selection units per operation and group counts for the next wave.

    ``margin`` > 1 over-selects (need / yield x margin): a wave that falls one or two units short
    costs a whole extra wave of model loading, while the surplus is still valid accepted data.
    ``exportable`` (per kind, per operation) replaces accepted counts for the remaining need: a
    multi target accepted in a group that the balanced export cannot use does not count.
    ``gaps`` (per kind, units: twin groups / multi targets): a kind whose total remaining need is
    at most its gap is not selected again; the export then stays that much short (time over balance).
    """
    out = {}
    for kind in ('twins', 'multi'):
        have = (exportable or {}).get(kind) or accepted[kind]
        need = {op: max(0, targets[kind] - have[op]) for op in OPERATIONS}
        stopped = 0 < sum(need.values()) <= (gaps or {}).get(kind, 0)
        if stopped:
            need = dict.fromkeys(OPERATIONS, 0)
        yields = {op: (accepted[kind][op] + PSEUDO * prior[kind][op]) / (selected[kind][op] + PSEUDO)
                  for op in OPERATIONS}
        units = {op: (margin * need[op] / max(.1, yields[op]) if need[op] else 0.0) for op in OPERATIONS}
        if kind == 'twins':
            groups = sum(math.ceil(u) for u in units.values())
            deficits = {op: 2 * u for op, u in units.items()}  # selector counts twin targets (2 per group)
        else:
            active = sum(1 for u in units.values() if u > 0)
            groups = math.ceil(sum(units.values()) / max(2, min(4, active))) if active else 0
            deficits = units
        out[kind] = {'need': need, 'yield': {op: round(y, 3) for op, y in yields.items()},
                     'selection_units': {op: round(u, 2) for op, u in units.items()},
                     'groups': min(caps[kind], groups), 'deficits': deficits,
                     **({'stopped_by_gap': True} if stopped else {})}
    return out


# --wave-budget: busy GPU-seconds per SELECTED group (planning + editing + audit) and selected targets
# per selected multi group, full runs 3-5.
GROUP_COST = {'twins': 63.0, 'multi': 105.0}
MULTI_SELECTED_TARGETS = 2.78


def fit_budget(plan, budget, caps, cost=GROUP_COST, targets_per_group=MULTI_SELECTED_TARGETS):
    """Groups per kind for one wave of about ``budget`` busy GPU-seconds, split in proportion to both kinds'
    remaining selection need, so that twins and multi finish together and every wave carries about the
    same work (production; per-kind caps still bound each kind)."""
    want = {'twins': sum(math.ceil(u) for u in plan['twins']['selection_units'].values()),
            'multi': math.ceil(sum(plan['multi']['selection_units'].values()) / targets_per_group)}
    total = sum(cost[kd] * want[kd] for kd in want)
    scale = min(1.0, budget / total) if total else 1.0
    for kd, n in want.items():
        plan[kd]['groups'] = min(caps[kd], max(1, math.floor(n * scale))) if n else 0
        plan[kd]['budget'] = {'wanted_groups': n, 'scale': round(scale, 4)}
    return plan


EXACT_MULTI_STATES = 200_000
EXACT_MULTI_GROUPS = 64


def multi_export(groups, target):
    """Whole multi groups (one image each, its targets ship together) with the most balanced counts.

    Objective, in order: the smallest per-operation count capped at the target, the total capped
    count, the least overshoot, then the fewest groups. Exact search over reachable count vectors
    for up to EXACT_MULTI_GROUPS groups and EXACT_MULTI_STATES vectors; beyond that the same
    objective is solved over composition types (``_typed_multi_export``), and only without scipy
    the greedy below is used. The greedy refuses any overshoot and left operations short when they
    only occur in groups with surplus operations (full runs 3/4: replace 4 of 6 and colour 2 of 6
    exported although 6 were accepted).
    """
    if len(groups) > EXACT_MULTI_GROUPS:
        return _typed_multi_export(groups, target)
    states = {(0,) * len(OPERATIONS): ()}          # count vector -> fewest group indices reaching it
    for i, g in enumerate(groups):
        comp = Counter(u['operation'] for u in g['units'])
        vec = tuple(comp[op] for op in OPERATIONS)
        for counts, chosen in list(states.items()):
            new = tuple(a + b for a, b in zip(counts, vec))
            if new not in states or len(states[new]) > len(chosen) + 1:
                states[new] = chosen + (i,)
        if len(states) > EXACT_MULTI_STATES:
            return _typed_multi_export(groups, target)

    def key(item):
        counts, chosen = item
        return (min(min(c, target) for c in counts), sum(min(c, target) for c in counts),
                -sum(max(0, c - target) for c in counts), -len(chosen))
    _, best = max(states.items(), key=key)
    chosen = [groups[i] for i in sorted(best)]
    return chosen, Counter(u['operation'] for g in chosen for u in g['units'])


def _typed_multi_export(groups, target):
    """multi_export's objective, exactly, for many groups: groups with the same per-operation counts are
    interchangeable, so an integer program over the composition types (at most a few dozen) is solved
    once per objective level (scipy HiGHS); each type contributes its earliest groups."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        return _greedy_multi(groups, target)
    types = {}
    for i, g in enumerate(groups):
        comp = Counter(u['operation'] for u in g['units'])
        types.setdefault(tuple(comp[op] for op in OPERATIONS), []).append(i)
    keys = list(types)
    k, p = len(keys), len(OPERATIONS)
    if not k:
        return [], Counter()
    comp = np.array(keys, dtype=float).T                  # p x k: units of each operation per group
    n = k + 2 * p + 1                                     # x (k, integer) | y (p) | m | o (p)
    y0, m0, o0 = k, k + p, k + p + 1
    lines, lo, hi = [], [], []

    def line(lower, upper, **coef):
        v = np.zeros(n)
        for name, (index, value) in coef.items():
            v[index] = value
        lines.append(v); lo.append(lower); hi.append(upper)
        return v
    for j in range(p):
        v = line(-np.inf, 0, y=(y0 + j, 1)); v[:k] = -comp[j]                    # y_j <= count_j
        line(-np.inf, 0, m=(m0, 1), y=(y0 + j, -1))                             # m <= y_j
        v = line(-np.inf, target, o=(o0 + j, -1)); v[:k] = comp[j]              # count_j - o_j <= target
    bounds = Bounds(np.r_[np.zeros(k + p + 1), np.zeros(p)],
                    np.r_[[len(types[t]) for t in keys], np.full(p, float(target)), [float(target)], np.full(p, np.inf)])
    integrality = np.r_[np.ones(k), np.zeros(n - k)]
    objectives = [('max', np.eye(n)[m0]), ('max', np.r_[np.zeros(k), np.ones(p), 0, np.zeros(p)]),
                  ('min', np.r_[np.zeros(k + p + 1), np.ones(p)]), ('min', np.r_[np.ones(k), np.zeros(n - k)])]
    result = None
    for sense, c in objectives:
        result = milp(-c if sense == 'max' else c, constraints=LinearConstraint(np.array(lines), lo, hi),
                      integrality=integrality, bounds=bounds)
        if not result.success:
            return _greedy_multi(groups, target)
        best = float(c @ result.x)
        # Keep this level (values are integral at the optimum) while optimizing the next one.
        lines.append(c); lo.append(best - .5 if sense == 'max' else -np.inf); hi.append(np.inf if sense == 'max' else best + .5)
    take = np.rint(result.x[:k]).astype(int)
    chosen = sorted(i for t, x in zip(keys, take) for i in types[t][:x])
    return [groups[i] for i in chosen], Counter(u['operation'] for i in chosen for u in groups[i]['units'])


def _greedy_multi(groups, target):
    multi, mcount = [], Counter()
    targets = {'multi': target}
    pending = list(groups)
    # Greedy: repeatedly add the group that most reduces remaining need without overshoot.
    while pending:
        def gain(g):
            ops = Counter(u['operation'] for u in g['units'])
            useful = sum(min(n, max(0, targets['multi'] - mcount[op])) for op, n in ops.items())
            over = sum(max(0, mcount[op] + n - targets['multi']) for op, n in ops.items())
            return useful - 2 * over, -over
        best = max(pending, key=gain)
        if gain(best)[0] <= 0:
            break
        pending.remove(best)
        multi.append(best)
        mcount.update(u['operation'] for u in best['units'])
    return multi, mcount


def balanced_export(all_groups, targets):
    twins = []
    count = Counter()
    for g in all_groups['twins']:
        if count[g['operation']] < targets['twins']:
            twins.append(g)
            count[g['operation']] += 1
    multi, mcount = multi_export(all_groups['multi'], targets['multi'])
    return twins, multi, count, mcount


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sources-root', type=Path, required=True)
    p.add_argument('--reservation', type=Path, required=True)
    p.add_argument('--exclude', type=Path, nargs='*', default=[])
    p.add_argument('--affordance-cache', type=Path, required=True)
    p.add_argument('--twin-groups-per-op', type=int, required=True, help='accepted twin groups wanted per operation')
    p.add_argument('--multi-targets-per-op', type=int, help='accepted multi targets wanted per operation')
    p.add_argument('--multi-groups', type=int,
                   help='instead of --multi-targets-per-op: accepted multi GROUPS wanted; the per-operation target '
                        'is this x mean targets per accepted group / 4, re-estimated after every wave')
    p.add_argument('--max-waves', type=int, default=3)
    p.add_argument('--wave-cap-twins', type=int, default=24)
    p.add_argument('--wave-cap-multi', type=int, default=20)
    p.add_argument('--wave-budget', type=float,
                   help='Production: size each wave to about this many busy GPU-seconds, split between twins and '
                        'multi by remaining need (fit_budget); the caps above still bound each kind')
    p.add_argument('--prior', type=json.loads, default=None, help='JSON yields {"twins": {op: y}, "multi": {...}}')
    p.add_argument('--seed', type=int, default=20261023)
    p.add_argument('--selection-margin', type=float, default=1.0,
                   help='Over-select each wave by this factor (need / estimated yield x margin); 1 keeps the old plan')
    p.add_argument('--stop-gap', type=int, default=0,
                   help='Do not run another wave for a kind whose total remaining need (twin groups / multi targets) '
                        'is at most this; the export stays that much short. Never applies to the first wave')
    p.add_argument('--stop-gap-fraction', type=float, default=0.0,
                   help='Same, as a fraction of the kind\'s total target (4 x per-operation target); the larger '
                        'of the two gaps applies')
    p.add_argument('--fill-margin', type=float,
                   help='Margin from the second wave on (default: --selection-margin). Later waves fill small '
                        'remaining needs, where one more wave of model loading costs more than extra groups')
    p.add_argument('--selection-python', default=sys.executable)
    p.add_argument('--twin-members-add', type=int, choices=[2, 3, 4],
                   help='forwarded to select_evidence_groups.py (over-select add twin candidates)')
    p.add_argument('--worn-check', action='store_true',
                   help='forwarded to select_evidence_groups.py (no removal of worn shoes/helmets/garments)')
    p.add_argument('--covering-check', action='store_true',
                   help='forwarded to select_evidence_groups.py (no removal of snow/moss/tarps lying on objects)')
    p.add_argument('--no-surface-recolor', action='store_true',
                   help='forwarded to select_evidence_groups.py (no colour edits of sky/water/ground)')
    p.add_argument('--skip-grayscale', action='store_true',
                   help='forwarded to select_evidence_groups.py (no black-and-white source photos)')
    p.add_argument('--menu-dir', type=Path, help='forwarded to select_evidence_groups.py (production menus)')
    p.add_argument('--selection-workers', type=int, default=16,
                   help='forwarded to select_evidence_groups.py --workers (with --menu-dir)')
    p.add_argument('--before-wave-json', type=json.loads,
                   help='JSON argv run before each wave\'s selection, with "--plan <wave>/plan.json" appended '
                        '(production: prepare enough labelled sources and menus)')
    p.add_argument('dual_args', nargs=argparse.REMAINDER,
                   help='after "--": arguments forwarded to run_dual_validation.py (models, pythons, flags)')
    args = p.parse_args(argv)
    if (args.multi_targets_per_op is None) == (args.multi_groups is None):
        p.error('give exactly one of --multi-targets-per-op and --multi-groups')
    dual_args = [a for a in args.dual_args if a != '--']
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    prior = args.prior or PRIOR
    ledger_path = root / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'waves': []}
    selected = {'twins': Counter(), 'multi': Counter()}
    accepted = {'twins': Counter(), 'multi': Counter()}
    multi_groups = []
    seconds = 0.0
    for k in range(args.max_waves):
        wave = root / f'wave{k + 1:02d}'
        if k:  # counts of the previous wave (finished: the loop only moves on after a complete wave)
            s, a, g = wave_counts(root / f'wave{k:02d}')
            multi_groups += g['multi']
            seconds += json.loads((root / f'wave{k:02d}/run/session.json').read_text()).get('seconds', 0)
            for kind in selected:
                selected[kind].update(s[kind])
                accepted[kind].update(a[kind])
        targets = {'twins': args.twin_groups_per_op,
                   'multi': multi_target(args, sum(accepted['multi'].values()), len(multi_groups), multi_groups)}
        exportable = {'multi': multi_export(multi_groups, targets['multi'])[1]}
        margin = args.selection_margin if k == 0 or args.fill_margin is None else args.fill_margin
        gaps = None
        if k > 0:
            gaps = {kd: max(args.stop_gap, math.ceil(args.stop_gap_fraction * 4 * targets[kd])) for kd in targets}
        caps = {'twins': args.wave_cap_twins, 'multi': args.wave_cap_multi}
        plan = plan_wave(targets, selected, accepted, prior, caps, margin=margin, exportable=exportable, gaps=gaps)
        if args.wave_budget:
            plan = fit_budget(plan, args.wave_budget, caps)
        print(json.dumps({'wave': k + 1, 'targets': targets, 'accepted_so_far': {kd: dict(v) for kd, v in accepted.items()},
                          'multi_exportable': dict(exportable['multi']),
                          'plan': {kd: {x: v[x] for x in ('need', 'yield', 'groups', 'stopped_by_gap') if x in v}
                                   for kd, v in plan.items()}}), flush=True)
        atomic_json(root / 'progress.json', {
            'updated': time.time(), 'waves_finished': k, 'targets': targets, 'session_seconds': round(seconds, 1),
            'accepted': {kd: dict(v) for kd, v in accepted.items()}, 'multi_groups_accepted': len(multi_groups),
            'exportable': {'twins_groups': {op: min(targets['twins'], accepted['twins'][op]) for op in OPERATIONS},
                           'multi_targets': {op: min(targets['multi'], exportable['multi'][op]) for op in OPERATIONS}},
            'next_wave': {kd: {x: plan[kd][x] for x in ('need', 'groups') if x in plan[kd]} for kd in plan}})
        if not plan['twins']['groups'] and not plan['multi']['groups']:
            break
        if len(ledger['waves']) <= k:
            ledger['waves'].append({'wave': k + 1, 'targets': targets, 'plan': plan, 'margin': margin,
                                    'started': time.time()})
            atomic_json(ledger_path, ledger)
        if not (wave / 'inputs/evidence_selection_summary.json').exists():
            if args.before_wave_json:
                atomic_json(wave / 'plan.json', {'wave': k + 1, 'targets': targets, 'plan': plan})
                subprocess.run([*args.before_wave_json, '--plan', str(wave / 'plan.json')], check=True)
            previous = [root / f'wave{j + 1:02d}' / 'inputs/evidence_selection_summary.json' for j in range(k)]
            cmd = [args.selection_python, str(REPO / 'scripts/select_evidence_groups.py'),
                   '--sources-root', str(args.sources_root), '--inputs-root', str(wave / 'inputs'),
                   '--reservation', str(args.reservation), '--affordance-cache', str(args.affordance_cache),
                   '--cache-only', '--seed', str(args.seed),
                   '--twins', str(plan['twins']['groups']), '--multi', str(plan['multi']['groups']),
                   '--initial-deficits', json.dumps({kd: plan[kd]['deficits'] for kd in plan}),
                   '--exclude', *[str(x) for x in list(args.exclude) + previous]]
            if args.twin_members_add:
                cmd += ['--twin-members-add', str(args.twin_members_add)]
            if args.worn_check:
                cmd += ['--worn-check']
            if args.covering_check:
                cmd += ['--covering-check']
            if args.no_surface_recolor:
                cmd += ['--no-surface-recolor']
            if args.skip_grayscale:
                cmd += ['--skip-grayscale']
            if args.menu_dir:
                cmd += ['--menu-dir', str(args.menu_dir), '--workers', str(args.selection_workers)]
            subprocess.run(cmd, check=True)
        kinds = [kd for kd in ('twins', 'multi') if rows(wave / f'inputs/{kd}/inputs/selected.jsonl')]
        if not kinds:
            print('selection found no eligible groups; stopping', flush=True)
            break
        if not (wave / 'run/session.json').exists():
            env = {key: value for key, value in os.environ.items() if key != 'CUDA_VISIBLE_DEVICES'}
            cmd = [sys.executable, str(REPO / 'scripts/run_dual_validation.py'), '--inputs-root', str(wave / 'inputs'),
                   '--output-root', str(wave / 'run'), '--reservation', str(args.reservation),
                   '--kinds', ','.join(kinds), *dual_args]
            subprocess.run(cmd, check=True, env=env)
        s, a, _ = wave_counts(wave)
        session = json.loads((wave / 'run/session.json').read_text())
        ledger['waves'][k].update(selected={kd: dict(v) for kd, v in s.items()}, accepted={kd: dict(v) for kd, v in a.items()},
                                  seconds=session.get('seconds'), finished=time.time())
        atomic_json(ledger_path, ledger)
    export(root, args, len(ledger['waves']))


def export(root, args, n_waves, out=None):
    """Balanced export of all finished waves into ``out`` (default <root>/final); returns the summary."""
    root = Path(root)
    all_groups = {'twins': [], 'multi': []}
    totals = {'selected': {'twins': Counter(), 'multi': Counter()}, 'accepted': {'twins': Counter(), 'multi': Counter()}}
    seconds = 0.0
    finished = [w for w in waves(root) if (w / 'run/session.json').exists()]
    for wave in finished:
        s, a, g = wave_counts(wave, cache=False)
        seconds += json.loads((wave / 'run/session.json').read_text()).get('seconds', 0)
        for kd in all_groups:
            all_groups[kd].extend(g[kd])
            totals['selected'][kd].update(s[kd])
            totals['accepted'][kd].update(a[kd])
    targets = {'twins': args.twin_groups_per_op,
               'multi': multi_target(args, sum(totals['accepted']['multi'].values()), len(all_groups['multi']),
                                     all_groups['multi'])}
    twins, multi, tcount, mcount = balanced_export(all_groups, targets)
    # Single-target samples salvaged from groups that could not ship (opt-in --keep-singles);
    # a separate product, not counted toward the twin/multi balance.
    singles = [s for wave in finished for kd in ('twins', 'multi') for s in rows(wave / f'run/{kd}/generation/singles.jsonl')]
    final = Path(out) if out else root / 'final'
    final.mkdir(parents=True, exist_ok=True)
    for name, data in (('twins_balanced.jsonl', twins), ('multi_balanced.jsonl', multi), ('singles.jsonl', singles)):
        with (final / name).open('w') as f:
            for g in data:
                f.write(json.dumps(g, ensure_ascii=False) + '\n')
    summary = {'targets': targets, 'waves': n_waves, 'session_seconds': round(seconds, 1),
               **({'multi_groups_target': args.multi_groups} if args.multi_groups is not None else {}),
               'selected': {kd: dict(v) for kd, v in totals['selected'].items()},
               'accepted': {kd: dict(v) for kd, v in totals['accepted'].items()},
               'exported': {'twins_groups': dict(tcount), 'multi_targets': dict(mcount),
                            'twins_group_count': len(twins), 'multi_group_count': len(multi)},
               'shortfall': {'twins_groups': {op: max(0, targets['twins'] - tcount[op]) for op in OPERATIONS},
                             'multi_targets': {op: max(0, targets['multi'] - mcount[op]) for op in OPERATIONS},
                             **({'multi_groups': max(0, args.multi_groups - len(multi))}
                                if args.multi_groups is not None else {})},
               'stop_gap': {'units': args.stop_gap, 'fraction': args.stop_gap_fraction},
               'singles': {'count': len(singles), 'by_operation': dict(Counter(x['operation'] for x in singles)),
                           'by_kind': dict(Counter(x['provenance']['kind'] for x in singles))}}
    atomic_json(final / 'summary.json', summary)
    print(json.dumps(summary, indent=1))
    return summary


def waves(root):
    """Wave directories in wave order (wave100 after wave99)."""
    return sorted((w for w in Path(root).glob('wave*') if w.name[4:].isdigit()), key=lambda w: int(w.name[4:]))


if __name__ == '__main__':
    main()
