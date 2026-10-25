"""Reconcile the audit's Ref-uniqueness and scope doubts with independent reader evidence (opt-in).

The main audit judges Ref uniqueness inside a long thinking call, and in the paired A/B it
rejected 7 targets on Ref grounds alone (6 good, 1 minor, 0 bad). In one of them (two small
clouds) the doubt was right and the audit even proposed a working repair, but the code
dropped the target because the same response marked the Ref non-unique. The planning reader
(clean photo + phrase -> one box per fitting instance) is an independent test of the same
property, so this module lets reader evidence settle the disagreement:

* the audit proposes a different Ref: the reader checks it (one short call, no reasoning);
  exactly one box on the target -> the repair is adopted;
* otherwise, if the planning reader already passed this exact draft Ref -> the draft is kept
  and the audit's doubt is recorded as advisory;
* otherwise the audit's verdict stands.

Scope: the audit sometimes reports scope_status="wrong_entity" for an edit it also judged
image_ok with the requested operation observed, typically an addition on a surface patch of a
large host ("a sailboat on the open sea"). In the paired A/B and full run 2, all 9 such targets
were good (assistant blind review); whenever the audit also failed the image (14 targets) the
image checks rejected them anyway. When the planning reader placed this exact Ref on the mask,
the scope doubt is recorded and scope_status set to "matched". A failed image check, a different
observed operation or no reader evidence leaves the audit's verdict in place.

A proposed Ref that the instruction revision would reject (for instance "the wheel near the
No. 9 panel": no periods allowed) is not adopted: the member would fail as an invalid revision
although its reader-verified draft is usable (full run 4 lost a twin group this way).

Only Ref-uniqueness and scope fields of the raw audit row change; image checks are untouched.
"""
from __future__ import annotations

import re
from copy import deepcopy

VERSION = 'audit-ref-reconcile-3-valid-repair'


def _norm(text) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'^the\s+', '', str(text or '').strip().lower()))


def planning_reader(group: dict, member: dict) -> str | None:
    """Reader status ('pass', 'ambiguous', ...) recorded at planning for this exact Ref, else None."""
    raw = (group.get('planning') or {}).get('raw') or {}
    ref = _norm((member.get('instruction_spec') or {}).get('ref'))
    for entry in raw.get('members') or raw.get('targets') or []:
        if str(entry.get('object_id')) != str(member.get('object_id')) or _norm(entry.get('ref_phrase')) != ref:
            continue
        for verdict in (entry.get('evidence_gates') or {}).get('verdicts') or []:
            if verdict.get('gate') == 'reader':
                return verdict.get('status')
    return None


def adoptable(member: dict, row: dict) -> bool:
    """Would the audit's proposed Ref pass the instruction revision applied after normalization?"""
    from .audit_instruction_revision import revise
    from .multi_component_audit import expand_compact_result
    trial = expand_compact_result({'units': [{**deepcopy(row), 'reference_unique': True}]}, [member])['units'][0]
    if (trial.get('instruction_revision') or {}).get('mode') != 'ref':
        return True
    return revise(deepcopy(member), trial, image_ok=True) == ''


def reader_job(job_id: str, source_path: str, ref: str) -> dict:
    from .evidence_gates import READER, READER_SCHEMA, TOKEN_LIMITS
    return {'job_id': job_id, 'images': [source_path], 'prompt': READER.format(ref=ref), 'output_schema': READER_SCHEMA,
            'max_tokens': TOKEN_LIMITS['reader'], 'enable_thinking': False, 'image_max_edge': 1536}


def reconcile(raw: dict, members: list[dict], group: dict, auditor, *, prefix: str) -> tuple[dict, list[dict], int]:
    """(raw with reconciled Ref fields, records, extra calls). ``raw`` is the compact aligned audit response."""
    from .evidence_gates import mask_box_1000, reader_verdict
    from .group_audit import _call_batched
    rows = raw.get('units') if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        return raw, [], 0
    raw = deepcopy(raw)
    rows = raw['units']
    by_id = {m['unit_id']: m for m in members}
    records, jobs, pending = [], [], []
    for row in rows:
        member = by_id.get(row.get('unit_id')) if isinstance(row, dict) else None
        if member is None:
            continue
        if (row.get('scope_status') not in (None, 'matched') and row.get('image_ok') is True
                and row.get('observed_operation') == member.get('operation')
                and planning_reader(group, member) == 'pass'):
            records.append({'unit_id': member['unit_id'], 'kind': 'scope', 'audit_scope_status': row['scope_status'],
                            'planning_reader': 'pass', 'outcome': 'scope_doubt_overruled_by_reader'})
            row['scope_status'] = 'matched'
        if row.get('reference_unique') is not False:
            continue
        draft = (member.get('instruction_spec') or {}).get('ref', '')
        proposed = row.get('final_ref') if isinstance(row.get('final_ref'), str) else ''
        record = {'unit_id': member['unit_id'], 'kind': 'reference', 'draft_ref': draft, 'audit_ref_issue': row.get('ref_issue'),
                  'audit_final_ref': proposed, 'planning_reader': planning_reader(group, member)}
        records.append(record)
        if proposed.strip() and _norm(proposed) != _norm(draft) and adoptable(member, row):
            jobs.append(reader_job(f"{prefix}::reader::{member['unit_id']}", group['source_image'], proposed))
            pending.append((row, member, record))
        else:
            if proposed.strip() and _norm(proposed) != _norm(draft):
                record['repair_invalid'] = True
            _keep_verified_draft(row, record)
    for (row, member, record), output in zip(pending, _call_batched(auditor, jobs) if jobs else []):
        boxes = (output.get('semantic_audit') or {}).get('boxes')
        status, best = reader_verdict(boxes, mask_box_1000(member.get('mask_v1') or member['mask']))
        record.update(repair_reader=status, repair_iou=round(best, 3))
        if status == 'pass':
            row['reference_unique'] = True
            record['outcome'] = 'repair_verified'
        else:
            _keep_verified_draft(row, record)
    return raw, records, len(jobs)


def _keep_verified_draft(row: dict, record: dict) -> None:
    if record['planning_reader'] == 'pass':
        # No verified defect remains: the canonical path keeps the original wording exactly.
        row.update(reference_unique=True, ref_issue='', final_ref='')
        record['outcome'] = 'kept_reader_verified_draft'
    else:
        record['outcome'] = 'audit_verdict_kept'
