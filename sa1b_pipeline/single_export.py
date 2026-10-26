"""Single-target samples from evidence-gated multi groups (single-target v2, opt-in, new module).

In parallel-components generation every independent component is edited from the ORIGINAL image and
audited on its own (component audit + verifier). A component holding exactly one target is therefore an
ordinary single-target sample: source -> that component's output, which differs from the source only
inside the target's support (the fusion proof checks this pixel by pixel). This module exports every such
target that passed its own checks, from accepted and non-accepted groups alike: a group can fail as a
group (for example fewer than two retained targets) while its targets pass on their own.

Not exported: targets of dependent chains (a component with several targets carries several edits), and
targets dropped from the fusion (overlap or record contract), whose decisions were made for the group.
Records use the single-target release fields of ``single_salvage.record``.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

VERSION = 'component-singles-1'


def component_singles(group: dict) -> list[dict]:
    from .multi_transactional import member_passes
    from .single_salvage import record
    # 'retained_contract_rejected': the group kept fewer than two targets, so it cannot ship as a multi
    # group; the target it kept passed its own audit and is still an ordinary single-target sample.
    if (group.get('joint_audit') or {}).get('status') not in ('complete', 'retained_contract_rejected'):
        return []
    parallel = group.get('parallel') or {}
    alone = {comp[0] for comp in parallel.get('components', []) if len(comp) == 1}
    fused = {c['kept'][0]: c for c in (parallel.get('fusion') or {}).get('components', [])
             if len(c.get('kept') or []) == 1}
    accepted = {d['unit_id'] for d in (group.get('retention') or {}).get('decisions', []) if d.get('accepted')}
    units = group.get('units') or []
    out = []
    for unit in units:
        uid = unit['unit_id']
        audit = unit.get('semantic_audit') or {}
        if uid not in alone or uid not in fused or uid not in accepted or not member_passes(audit):
            continue
        rec = record('multi', group['multi_id'], group, unit, fused[uid]['image'], audit,
                     reason='independent_component', others=[u['unit_id'] for u in units if u['unit_id'] != uid])
        rec['provenance'].update(export=VERSION, image_sha256=fused[uid]['image_sha256'],
                                 group_accepted=bool((group.get('joint_audit') or {}).get('pass')))
        out.append(rec)
    return out


def export_run(generation_dir: Path) -> tuple[list[dict], dict]:
    """Singles of one multi generation directory (its audited.jsonl) and a small summary."""
    path = Path(generation_dir) / 'audited.jsonl'
    groups = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    singles = [s for g in groups for s in component_singles(g)]
    planned = Counter(op for g in groups for op in (g.get('retention') or {}).get('planned_operations', []))
    summary = {'version': VERSION, 'groups': len(groups), 'singles': len(singles),
               'by_operation': dict(Counter(s['operation'] for s in singles)), 'planned_targets': dict(planned),
               'from_accepted_groups': sum(s['provenance']['group_accepted'] for s in singles)}
    return singles, summary
