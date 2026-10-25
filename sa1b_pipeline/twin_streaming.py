"""Bounded twin pipeline with persistent services and per-group checkpoints."""
from copy import deepcopy
from pathlib import Path
import time

from .blind_identifiability import run_blind_probes
from .group_audit import audit_twin_groups_joint
from .inference_services import RemoteAuditor, RemoteBlind
from .production_state import atomic_json, digest, file_hash

RETRY_VERSION = 'twin-generation-retry-1'
RETRY_SEED = 1


def media_hashes(group):
    paths = {group['source_image']}
    for member in group['members']:
        paths.update(member[k] for k in ('mask', 'mask_v1', 'edited_image', 'support_mask') if member.get(k))
        unit = member.get('unit', {})
        paths.update(unit[k] for k in ('mask_path', 'execution_mask_path', 'protected_mask_path') if unit.get(k))
    return {p: file_hash(p) for p in sorted(paths)}


# Failures that a new generation cannot fix: the Ref or instruction text, or the group contract.
TEXT_FAILURES = {'reference_not_unique', 'ref_matches_mask_failed', 'invalid_instruction_revision',
                 'invalid_optimized_instruction', 'missing_revision_spec', 'instruction_revision_requires_valid_image'}


def retry_plan(group):
    """Members to re-render (and to keep) for one generation retry of a rejected twin group, or None.

    Retried: members whose own audit or deterministic check failed, or whose edit spilled into the
    other member's region. Not retried: Ref/text failures, a failed blind/size contract or support gap
    (properties of the plan), and groups where nothing member-specific failed.

    Not recommended (2026-10-25): on the 7 twin groups of full runs 2-3 rejected for image reasons,
    the retry was accepted once in 5 attempts, and that group (additions on a printed picture, now
    dropped by the medium gate) kept a member the blind review had judged bad. The other four failed
    again for the same plan-level reason (shorts drawn for "long trousers", the old sign kept).
    """
    checks = group.get('group_checks') or {}
    if checks.get('group_pass') or checks.get('generation_skipped'):
        return None
    for key in ('blind_protocol', 'support_gap'):
        if (checks.get(key) or {}).get('pass') is False:
            return None
    redo = set()
    for member in group['members']:
        audit = member.get('semantic_audit') or {}
        if TEXT_FAILURES & set(audit.get('failure_reasons') or []):
            return None
        if audit.get('decision') == 'fail' or (member.get('deterministic') or {}).get('decision') == 'fail':
            redo.add(member['unit_id'])
    for pair in (checks.get('cross_target_preservation') or {}).get('pairs') or []:
        if pair.get('changed_fraction_in_other_source_mask', 0) > 0.02:
            redo.add(pair['member'])
    if not redo:
        return None
    return {'regenerate': sorted(redo), 'keep': sorted(m['unit_id'] for m in group['members'] if m['unit_id'] not in redo)}


def process_twin(plan, root, editors, audits, *, local_reference=False):
    """One blind request, generation, one joint audit; with editors.generation_retry, one retry
    (new seed for the failed members only) when the audit rejected the group for image reasons."""
    import json
    root = Path(root)
    gid = plan['twin_id']
    if Path(gid).name != gid or gid in ('.', '..'):
        raise ValueError('invalid twin ID')
    checkpoint = root/'results'/f'{gid}.json'
    fingerprint = digest({'plan': plan, 'inputs': media_hashes(plan),
                          'editor': editors.model, 'audit': audits.model,
                          'policy': 'twins-stream-v9-compact-audit',
                          'audit_prompt_version': __import__('sa1b_pipeline.twin_compact_audit', fromlist=['VERSION']).VERSION, 'local_reference':local_reference, 'steps': 40,
                          **({'verifier': __import__('sa1b_pipeline.edit_verifier', fromlist=['contract']).contract(audits.verifier)}
                             if getattr(audits, 'verifier', None) is not None else {}),
                          **({'keep_singles': __import__('sa1b_pipeline.single_salvage', fromlist=['VERSION']).VERSION}
                             if getattr(audits, 'keep_singles', False) else {}),
                          **({'ref_reconcile': __import__('sa1b_pipeline.audit_reconcile', fromlist=['VERSION']).VERSION}
                             if getattr(audits, 'ref_reconcile', False) else {}),
                          **({'generation_retry': RETRY_VERSION} if getattr(editors, 'generation_retry', False) else {})})
    if checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError('streamed twin input changed: '+gid)
        if saved['media_sha256'] != media_hashes(saved['group']):
            raise ValueError('streamed twin output changed: '+gid)
        return saved
    start = time.monotonic()
    group = deepcopy(plan)
    if local_reference:
        from .group_generation import _unit_dict
        for member in group['members']:
            member['unit'] = {**_unit_dict(member), 'generation_reference_frame':'crop-entity-v1'}
    run_blind_probes([group], RemoteBlind(audits), root/'blind'/gid)
    blind_done = time.monotonic()
    status = 'blind_rejected'
    retry = None
    generation_done = audit_done = blind_done
    from .twin_contract import blind_probe_satisfied
    if blind_probe_satisfied(group):
        generated = root/'generated'/gid/'group.json'
        generation_contract = generated.with_name('input.json')
        from .production_state import immutable_json
        immutable_json(generation_contract, {'fingerprint': fingerprint})
        if generated.exists():
            saved = json.loads(generated.read_text())
            hashes = generated.with_name('media_sha256.json')
            # A crash between these atomic writes leaves a verifiable render.json
            # for every member; re-enter the worker to finish the checkpoint.
            if hashes.exists():
                if json.loads(hashes.read_text()) != media_hashes(saved):
                    raise ValueError('streamed generation output changed: '+gid)
                group = saved
            else:
                group = editors.request({'group': group, 'directory': str(generated.parent), 'seed': 0})
        else:
            group = editors.request({'group': group, 'directory': str(generated.parent), 'seed': 0})
        atomic_json(generated.with_name('media_sha256.json'), media_hashes(group))
        generation_done = time.monotonic()
        if group['group_checks'].get('generation_skipped'):
            status = 'generation_rejected'
        else:
            generated_group = deepcopy(group)
            group = audit_twin_groups_joint([group], RemoteAuditor(audits, kind='twins'), root/'audit_panels', exact_selector=True)[0]
            status = 'accepted' if group['group_checks'].get('group_pass') else 'audit_rejected'
            plan_retry = retry_plan(group) if getattr(editors, 'generation_retry', False) else None
            if plan_retry:
                retry_group = deepcopy(generated_group)
                for member in retry_group['members']:
                    if member['unit_id'] in plan_retry['regenerate']:
                        member['retry_seed'] = RETRY_SEED
                        member.pop('edited_image', None)
                    else:
                        member['retry_keep'] = True
                retry_dir = root/'generated'/f'{gid}__retry1'
                retried = editors.request({'group': retry_group, 'directory': str(retry_dir), 'seed': 0})
                atomic_json(retry_dir/'media_sha256.json', media_hashes(retried))
                retry = {'version': RETRY_VERSION, **plan_retry, 'seed': RETRY_SEED, 'first_status': status,
                         'first_failure_reasons': {m['unit_id']: (m.get('semantic_audit') or {}).get('failure_reasons')
                                                   for m in group['members']}}
                if retried['group_checks'].get('generation_skipped'):
                    retry['status'] = 'generation_rejected'
                else:
                    first = group
                    group = audit_twin_groups_joint([retried], RemoteAuditor(audits, kind='twins'), root/'audit_panels_retry1',
                                                    exact_selector=True)[0]
                    status = 'accepted' if group['group_checks'].get('group_pass') else 'audit_rejected'
                    retry['status'] = status
                    if status != 'accepted' and first.get('singles') and not group.get('singles'):
                        group['singles'] = first['singles']   # the first attempt's salvaged member stays usable
        audit_done = time.monotonic()
    result = {'fingerprint': fingerprint, 'status': status, 'group': group,
              'media_sha256': media_hashes(group), **({'retry': retry} if retry else {}),
              'seconds': {'blind_including_queue': blind_done-start,
                          'generation_including_queue': generation_done-blind_done,
                          'audit_including_queue': audit_done-generation_done,
                          'total': audit_done-start}}
    atomic_json(checkpoint, result)
    return result
