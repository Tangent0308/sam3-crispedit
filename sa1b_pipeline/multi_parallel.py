"""Parallel local edits with serial conflict components and verified pixel fusion.

All components start at the original photograph. Overlapping writable supports
(and explicit dependencies) are serialized in the planner's existing order.
Only accepted edits enter a component result; disjoint accepted supports are
copied once into the original. Crops provide context, never fusion ownership.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import combinations
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

from .geometry import load_mask
from .group_generation import multi_support_preflight, _union_support, _write_mask
from .multi_transactional import (run_committed_step, finalize_committed_group,
                                  _verify_hashes)
from .group_schema import validate_multi_record
from .production_state import atomic_json, immutable_json, file_hash, digest

POLICY = 'parallel-components-v5-current-context'


def conflict_components(group):
    """Connect writes, not crop boxes; connected targets keep planned order."""
    order = group['generation_order']
    units = {u['unit_id']: u for u in group['units']}
    size = Image.open(group['source_image']).size
    supports = {uid: load_mask({'mask_path': units[uid]['unit']['execution_mask_path']}, size)
                for uid in order}
    selectors = {uid: load_mask({'mask_path': units[uid].get('mask_v1') or units[uid]['mask']}, size)
                 for uid in order}
    neighbors = {uid: set() for uid in order}
    edges = []
    for a, b in combinations(order, 2):
        source_overlap = int(np.count_nonzero(selectors[a] & selectors[b]))
        overlap = int(np.count_nonzero(supports[a] & supports[b]))
        if source_overlap or overlap:
            neighbors[a].add(b); neighbors[b].add(a)
            edges.append({'units': [a, b],
                          'reason': 'source_mask_overlap' if source_overlap else 'writable_support_overlap',
                          'pixels': source_overlap or overlap, 'writable_overlap_pixels': overlap})
    # Optional explicit pairs are an extension to the plan, not inferred from nouns.
    for pair in group.get('edit_dependencies', []):
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(uid not in units for uid in pair) or pair[0] == pair[1]):
            raise ValueError('invalid edit dependency')
        a, b = pair
        if order.index(a) >= order.index(b):
            raise ValueError('generation order contradicts explicit dependency')
        neighbors[a].add(b); neighbors[b].add(a)
        details = [e for e in group.get('dependency_schedule', {}).get('evidence', [])
                   if e.get('before') == a and e.get('after') == b]
        edges.append({'units': [a, b], 'reason': 'content_dependency' if details else 'explicit_dependency',
                      'evidence': details})
    # Conditioning joins execution, but does not make the local edit depend
    # logically on accepting the surface edit. If it failed without writing,
    # later steps may safely use the original state and be audited normally.
    for a,b in group.get('dependency_schedule', {}).get('conditioning_edges', []):
        if a not in units or b not in units or a == b:
            raise ValueError('invalid conditioning edge')
        neighbors[a].add(b); neighbors[b].add(a)
        edges.append({'units':[a,b], 'reason':'shared_edited_context'})
    unseen = set(order); components = []
    for first in order:
        if first not in unseen: continue
        stack = [first]; found = set()
        while stack:
            uid = stack.pop()
            if uid in found: continue
            found.add(uid); unseen.discard(uid)
            stack.extend(neighbors[uid] - found)
        components.append([uid for uid in order if uid in found])
    return components, edges


def fuse_components(source, results, directory):
    """Copy already-feathered accepted pixels once; refuse competing ownership."""
    directory = Path(directory)
    original = np.asarray(Image.open(source).convert('RGB'))
    fused = original.copy(); owned = np.zeros(original.shape[:2], bool)
    evidence = []
    for i, result in enumerate(results):
        kept = result['kept']
        if not kept: continue
        support = _union_support(source, kept)
        if (owned & support).any():
            raise ValueError('parallel fusion has overlapping accepted supports')
        final = np.asarray(Image.open(result['current']).convert('RGB'))
        if final.shape != original.shape:
            raise ValueError('parallel component dimensions changed')
        # The compositor must preserve every source pixel outside accepted writes.
        if np.any(final[~support] != original[~support]):
            raise ValueError('parallel component changed pixels outside accepted support')
        fused[support] = final[support]
        owned |= support
        path = _write_mask(directory / f'component_{i:02d}_accepted_support.png', support)
        evidence.append({'component': i, 'kept': [u['unit_id'] for u in kept],
                         'image': result['current'], 'image_sha256': file_hash(result['current']),
                         'support': path, 'support_sha256': file_hash(path),
                         'copied_pixels': int(support.sum())})
    output = directory / 'fused.png'; output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fused).save(output)
    proof = {'version': 1, 'source_sha256': file_hash(source), 'output_sha256': file_hash(output),
             'disjoint_component_supports': True, 'outside_union_exactly_preserved': True,
             'no_second_alpha_blend': True, 'components': evidence}
    atomic_json(directory / 'fusion_proof.json', proof)
    return str(output), proof


def select_valid_component_subset(group, results):
    """Keep the largest valid set without rerendering or hiding dependencies.

    Each component can only retain a prefix of its accepted edits: a later
    output may already depend on every earlier accepted image in that chain.
    Disjoint components can be selected independently. No audit is overridden.
    """
    kept = [u for r in results for u in r['kept']]
    order = group['generation_order']
    ordered = sorted(kept, key=lambda u: order.index(u['unit_id']))
    if len(kept) < 2:
        return results, []
    for n in range(len(ordered), 1, -1):
        for candidate in combinations(ordered, n):
            ids = {u['unit_id'] for u in candidate}
            if any(r.get('audit_atomic') and 0 < len(ids & {u['unit_id'] for u in r['kept']}) < len(r['kept'])
                   for r in results):
                continue
            if any([u['unit_id'] for u in r['kept'] if u['unit_id'] in ids]
                   != [u['unit_id'] for u in r['kept']][:sum(u['unit_id'] in ids for u in r['kept'])]
                   for r in results):
                continue
            trial = {**group, 'units': list(candidate), 'n_targets': n,
                     'operations': [u['operation'] for u in candidate],
                     'generation_order': [uid for uid in order if uid in ids]}
            trial['adjacent_pairs'] = []
            try:
                validate_multi_record(trial)
            except (ValueError, KeyError, TypeError):
                continue
            if n == len(kept):
                return results, []
            selected = []
            for r in results:
                prefix = [u for u in r['kept'] if u['unit_id'] in ids]
                selected.append({**r, 'kept': prefix,
                                 'current': prefix[-1]['step_output_image'] if prefix else group['source_image']})
            return selected, [u['unit_id'] for u in ordered if u['unit_id'] not in ids]
    return results, []  # No valid multi group; preserve diagnostics and reject normally.


def generate_parallel_multi_group(plan, directory, editor_factory, auditor, *, seed=0,
                                  save_crop_evidence=False, audit_policy='step-and-joint'):
    """Generate once per target; serialize connected writes/content dependencies."""
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    if audit_policy not in ('step-and-joint', 'component-final-v2', 'group-final-v3', 'packed-components-v4'):
        raise ValueError('unknown parallel audit policy')
    component_only = audit_policy in ('component-final-v2','group-final-v3','packed-components-v4')
    grouped_audit = audit_policy in ('group-final-v3','packed-components-v4')
    if plan.get('commit_policy') or plan.get('joint_target'):
        raise ValueError('parallel generation requires an ungenerated plan')
    from .multi_execution_contract import schedule_content_dependencies
    plan = schedule_content_dependencies(deepcopy(plan))
    ids = [u['unit_id'] for u in plan['units']]; order = plan['generation_order']
    if len(set(ids)) != len(ids) or len(order) != len(ids) or set(order) != set(ids):
        raise ValueError('invalid parallel generation order')
    editor_config = editor_factory().config
    immutable_json(root / 'transaction_contract.json', {
        'policy': POLICY, 'audit_policy': audit_policy, 'plan': plan, 'seed': seed, 'editor': editor_config,
        'auditor': auditor.config, 'save_crop_evidence': save_crop_evidence,
        'source_sha256': file_hash(plan['source_image']),
        'mask_sha256': {u['unit_id']: file_hash(u.get('mask_v1') or u['mask']) for u in plan['units']},
        'code': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}})
    committed = root / 'committed.json'
    if committed.exists():
        saved = json.loads(committed.read_text()); _verify_hashes(saved['image_hashes'])
        if saved['group_sha256'] != digest(saved['group']):
            raise ValueError('parallel final record changed')
        for component in saved['group']['parallel']['fusion']['components']:
            _verify_hashes({component['image']: component['image_sha256'],
                            component['support']: component['support_sha256']})
        return saved['group']
    group = deepcopy(plan); group['commit_policy'] = POLICY
    preflight = multi_support_preflight(group, root)
    group['support_preflight'] = preflight
    from .multi_execution_contract import prepare_dependency_interfaces
    prepare_dependency_interfaces(group, root / 'dependency_interfaces')
    components, edges = conflict_components(group)
    atomic_json(root / 'schedule.json', {'components': components, 'edges': edges})
    by_id = {u['unit_id']: u for u in group['units']}
    original = np.asarray(Image.open(group['source_image']).convert('RGB'), dtype=np.int16)
    started = time.monotonic()

    def run_component(item):
        component_id, uids = item
        current = group['source_image']; kept = []; decisions = []; provisional = []
        editor = editor_factory()  # Thread-local adapter, shared remote GPU pool.
        for uid in uids:
            index = order.index(uid) + 1
            step_dir = root / f'step_{index:02d}_{uid}'
            from .multi_execution_contract import dependency_context
            member_input = deepcopy(by_id[uid])
            member_input.setdefault('unit', {})['dependency_context'] = dependency_context(
                group, uid, {u['unit_id'] for u in kept})
            decision = run_committed_step(group, member_input, current, original, step_dir,
                                          editor, auditor, index=index, seed=seed,
                                          save_crop_evidence=save_crop_evidence, defer_semantic=component_only)
            provisional.append(decision)
            member = decision['member']
            decisions.append({'unit_id': uid, 'operation': member['operation'],
                              'accepted': decision['accepted'], 'reasons': decision['reasons'],
                              'checkpoint': str(step_dir / 'decision.json')})
            if decision['accepted']:
                kept.append(member); current = member['step_output_image']
            atomic_json(root / f'component_{component_id:02d}_progress.json', {
                'attempted': len(decisions), 'planned': len(uids), 'current_image': current,
                'kept': len(kept), 'decisions': decisions})
            print(json.dumps({'group': group['multi_id'], 'component': component_id, 'step': uid,
                              'commit': decision['accepted'] if not component_only else None,
                              'provisional': decision['accepted'] if component_only else None,
                              'kept': len(kept)}), flush=True)
        if grouped_audit:
            return dict(current=current,kept=kept,decisions=decisions,provisional=provisional)
        if component_only:
            from .multi_component_audit import complete_component
            return complete_component(group, provisional, current, auditor, root / f'component_{component_id:02d}_audit')
        return {'current': current, 'kept': kept, 'decisions': decisions}

    with ThreadPoolExecutor(max_workers=len(components)) as executor:
        results = list(executor.map(run_component, enumerate(components)))
    shared_calls=0
    if grouped_audit:
        from .multi_component_audit import complete_components_together
        results,shared_calls=complete_components_together(group,results,auditor,root/'group_audit')
    results, contract_dropped = select_valid_component_subset(group, results)
    current, proof = fuse_components(group['source_image'], results, root / 'fusion')
    decisions = sorted([d for r in results for d in r['decisions']], key=lambda d: order.index(d['unit_id']))
    kept = [u for r in results for u in r['kept']]
    group['parallel'] = {'components': components, 'edges': edges, 'fusion': proof,
                         'audit_policy': audit_policy,
                         'component_semantic_calls': shared_calls+sum(r.get('semantic_calls', 0) for r in results) if component_only else None,
                         'contract_subset_dropped': contract_dropped,
                         'generation_and_step_audit_seconds': time.monotonic() - started,
                         'source_policy': 'original per component; committed previous step within component'}
    return finalize_committed_group(plan, group, kept, decisions, current, root, auditor, component_only=component_only)
