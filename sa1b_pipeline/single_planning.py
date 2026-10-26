"""Single-target groups in planning (single-target production, opt-in, this process only).

The multi planning entry (scripts/run_group_pipeline.py) requires two to five targets per group in three
places: the selection-row check, ``group_planning.build_multi_group`` (``fewer than two feasible targets``) and
``multi_execution_contract.retain_feasible_plan`` (the multi type contract). Single-target production plans
1-4 independent targets of one image and exports every target on its own, so a group with one needed
target (the other operations are complete) or one feasible target (the others failed their evidence) is
still useful: wave03 rejected 50 of 300 groups at planning, 36 of them with exactly one feasible target.

``enable`` replaces the three rules in the current process (run_single_validation.py --single-target-groups);
the shared modules on disk and the twins/multi runs are not changed:
* selection rows may hold 1-4 targets with distinct operations;
* ``build_multi_group`` keeps a group with one feasible target. The replacement is the shared function itself
  with only that minimum changed (derived from its source, so every other rule stays identical; if the shared
  line changes, ``enable`` refuses instead of guessing);
* retention keeps every target that passes the member schema (the multi type contract does not apply:
  targets are edited, audited and exported independently).
"""
from __future__ import annotations

import inspect
import textwrap
from collections import Counter

VERSION = 'single-target-groups-1'
MULTI_MINIMUM = "if len(units)<2:raise ValueError('fewer than two feasible targets: '+str(dropped))"
SINGLE_MINIMUM = "if len(units)<1:raise ValueError('no feasible target: '+str(dropped))"
_STATE = {}


def single_target_builder():
    """``build_multi_group`` with a minimum of one feasible target (compiled once from the shared source)."""
    if 'builder' not in _STATE:
        from . import group_planning
        source = textwrap.dedent(inspect.getsource(group_planning.build_multi_group))
        if source.count(MULTI_MINIMUM) != 1:
            raise RuntimeError('group_planning.build_multi_group changed; review single-target planning')
        source = source.replace(MULTI_MINIMUM, SINGLE_MINIMUM).replace(
            'def build_multi_group(', 'def build_single_target_group(', 1)
        namespace = dict(vars(group_planning))   # same globals: helpers, relative imports (__package__)
        exec(compile(source, group_planning.__file__ + ' [single-target minimum]', 'exec'), namespace)
        _STATE['builder'] = namespace['build_single_target_group']
    return _STATE['builder']


def validate_selection_rows(rows, *, aggregate_balance=False) -> None:
    """Selection rows of single-target planning: 1-4 targets, one operation per target, no repeats."""
    for index, row in enumerate(rows):
        objects = row.get('objects') or []
        operations = row.get('operations') or []
        if len(operations) != len(objects):
            raise RuntimeError(f'selection row {index} has {len(operations)} operations for {len(objects)} objects')
        if not 1 <= len(objects) <= 4 or len(set(operations)) != len(operations):
            raise RuntimeError(f'selection row {index} needs 1-4 targets with distinct operations: {operations}')


def retain_valid_targets(group, adjacent_pairs=None):
    """Keep every target that passes the member schema; protect the masks of the others (as the multi
    retention does for removed targets). Raises when none remains."""
    from .group_schema import _validate_member
    kept, removed = [], []
    for unit in group['units']:
        try:
            _validate_member(dict(unit, support_mask=unit['mask']), require_resolved_regions=False)
        except (ValueError, KeyError, TypeError):
            removed.append(unit)
            continue
        kept.append(unit)
    if not kept:
        raise ValueError('no target satisfies the member schema')
    ids = {u['unit_id'] for u in kept}
    group.update(units=kept, n_targets=len(kept), operations=[u['operation'] for u in kept],
                 generation_order=[uid for uid in group['generation_order'] if uid in ids], adjacent_pairs=[])
    group['edit_dependencies'] = [p for p in group.get('edit_dependencies', []) if set(p) <= ids]
    if removed:
        group.setdefault('protected_source_masks', []).extend(u['mask'] for u in removed)
        group.setdefault('planning_retention', {}).setdefault('dropped', []).extend(
            {'unit_id': u['unit_id'], 'object_id': u['object_id'], 'mask': u['mask'],
             'reason': 'member schema', 'stage': 'single_target_retention'} for u in removed)
    group['single_target_planning'] = VERSION
    return group


def enable(run_group_pipeline) -> dict:
    """Apply the single-target rules to the given run_group_pipeline module and multi_execution_contract."""
    from . import multi_execution_contract
    if _STATE.get('enabled') is run_group_pipeline:
        return {'version': VERSION}
    run_group_pipeline._validate_multi_selection_operations = validate_selection_rows
    run_group_pipeline.build_multi_group = single_target_builder()
    multi_execution_contract.retain_feasible_plan = retain_valid_targets
    _STATE['enabled'] = run_group_pipeline
    return {'version': VERSION}


def drop_complete_operations(rows, need: dict) -> tuple[list, Counter]:
    """Selection rows without targets of operations whose remaining need is zero. The multi selection adds such
    targets only to reach its two-target minimum; planning and editing them would make surplus samples."""
    out, dropped = [], Counter()
    for row in rows:
        keep = [i for i, op in enumerate(row['operations']) if need.get(op, 0) > 0]
        for i, op in enumerate(row['operations']):
            if i not in keep:
                dropped[op] += 1
        if not keep:
            continue
        new = dict(row, objects=[row['objects'][i] for i in keep], operations=[row['operations'][i] for i in keep])
        evidence = row.get('evidence_selection')
        members = evidence.get('members') if isinstance(evidence, dict) else None
        ids = [str(row['objects'][i].get('object_id')) for i in keep]
        if isinstance(members, dict) and set(map(str, members)) == {str(o.get('object_id')) for o in row['objects']}:
            new['evidence_selection'] = dict(evidence, members={k: v for k, v in members.items() if str(k) in ids})
        if len(keep) < len(row['operations']):
            new['single_target_dropped'] = [op for i, op in enumerate(row['operations']) if i not in keep]
        out.append(new)
    return out, dropped
