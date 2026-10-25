"""Single-target samples salvaged from twin and multi groups that cannot ship as groups (opt-in).

A twin group ships only when every member passes; a multi group needs at least two retained
targets. With ``keep_singles`` on, a member/target that passed every check of its own is kept
as an ordinary single-target sample instead of being discarded with its group:

* twins: any member of a non-accepted group whose own deterministic check, region resolution,
  cross-target preservation (its edit leaves the other member's selection unchanged) and
  semantic audit (main audit + optional verifier) all pass;
* multi: the one target left when a group ends with exactly one retained target.

Records follow the single-target release fields (absolute paths) plus ``provenance``.
"""
from __future__ import annotations

from copy import deepcopy

VERSION = 'single-salvage-1'
CROSS_LIMIT = .02   # same limit as the twin cross-target preservation check


def twin_member_eligible(group: dict, member: dict) -> bool:
    """Checks a member must pass on its own, before (and independent of) the semantic audit."""
    if not member.get('edited_image') or member.get('blind_eligible') is not True:
        return False
    if (member.get('deterministic') or {}).get('decision') == 'fail':
        return False
    pairs = ((group.get('group_checks') or {}).get('cross_target_preservation') or {}).get('pairs') or []
    return all(p.get('changed_fraction_in_other_source_mask', 1.) <= CROSS_LIMIT
               for p in pairs if p.get('member') == member['unit_id'])


def twin_singles(group: dict, audits: dict) -> list[dict]:
    """Single records for a twin group that did not ship; ``audits`` maps unit_id to its normalized audit."""
    checks = group.get('group_checks') or {}
    if checks.get('group_pass') or checks.get('generation_skipped'):
        return []
    out = []
    for member in group['members']:
        audit = audits.get(member['unit_id']) or {}
        if twin_member_eligible(group, member) and audit.get('decision') == 'pass':
            others = [m['unit_id'] for m in group['members'] if m['unit_id'] != member['unit_id']]
            out.append(record('twins', group['twin_id'], group, member, member['edited_image'], audit,
                              reason='twin_group_not_accepted', others=others))
    return out


def multi_single(group: dict) -> dict | None:
    """The single record for a multi group that ended with exactly one retained target, else None."""
    if (group.get('joint_audit') or {}).get('pass') or len(group.get('units') or []) != 1:
        return None
    from .multi_transactional import member_passes
    unit = group['units'][0]
    audit = unit.get('semantic_audit') or {}
    if not member_passes(audit):
        return None
    image = unit.get('component_audit_image') or group.get('joint_target')
    planned = [d['unit_id'] for d in (group.get('retention') or {}).get('decisions', [])]
    return record('multi', group['multi_id'], group, unit, image, audit, reason='multi_single_target_left',
                  others=[u for u in planned if u != unit['unit_id']])


def record(kind: str, group_id: str, group: dict, member: dict, image: str, audit: dict, *, reason: str,
           others: list[str]) -> dict:
    from PIL import Image
    with Image.open(group['source_image']) as source:
        size = list(source.size)
    spec = deepcopy(member.get('instruction_spec') or {})
    verdict = audit.get('verifier')
    return {
        'job_id': f"{group_id}/{member['unit_id']}", 'case_id': group_id, 'unit_id': member['unit_id'],
        'source_id': group.get('source_id'), 'object_id': member.get('object_id'), 'operation': member['operation'],
        'instruction': member.get('instruction'), 'instruction_noref': member.get('instruction_noref'),
        'instruction_protocol_version': member.get('instruction_protocol_version') or spec.get('version'),
        'instruction_spec': spec, 'ref_phrase': spec.get('ref') or member.get('ref'), 'scope': spec.get('scope'),
        'mask_role': member.get('mask_role'), 'bbox': member.get('bbox'), 'point': member.get('point'),
        'payload': spec.get('payload', ''), 'source_image': group['source_image'], 'edited_image': image,
        'mask': member.get('mask'), 'support_mask': member.get('support_mask'), 'image_size': size,
        'audit': {'decision': 'pass', 'source_state': audit.get('source_state'), 'result_state': audit.get('result_state'),
                  'verifier': verdict},
        'provenance': {'version': VERSION, 'kind': kind, 'group_id': group_id, 'reason': reason,
                       'other_units': others},
        'manual_review': 'not_performed',
    }
