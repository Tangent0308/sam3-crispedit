"""Single-target generation from evidence-gated multi plans (single-target v2.1, opt-in, new module).

A multi plan holds 1-4 targets of one photograph. Multi-target generation edits connected targets
(overlapping writable supports, dependencies) in series and fuses the components into one image, so a target
edited after another one cannot be exported on its own: in single-target v2 (single_export) about one passing
target in five was lost that way. A single-target sample needs neither series nor fusion. Here every target is
edited from the ORIGINAL photograph as its own component, with the multi path's execution masks (the other
targets' selectors stay protected), deterministic checks, source-region resolution and semantic audit
(the packing of packed-components-v4: two targets whose supports are disjoint share one audit call on a
provisional image, overlapping ones are audited alone; verifier and Ref reconcile as configured on the
auditor). Every target that passes is exported as source -> its own output, which must differ from the source
only inside its own support (checked pixel by pixel). Content dependencies between targets do not apply: no
target sees another target's edit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import json
import time

import numpy as np
from PIL import Image

from .geometry import load_mask
from .group_generation import multi_support_preflight, _union_support
from .multi_component_audit import audit_component, complete_component
from .multi_parallel import fuse_components
from .multi_transactional import run_committed_step, member_passes, _evidence_hashes, _verify_hashes
from .production_state import atomic_json, immutable_json, file_hash, digest

POLICY = 'independent-singles-1'


def _independent_plan(plan):
    """The plan with every target on its own: no explicit dependencies, no conditioning edges."""
    group = deepcopy(plan)
    group['edit_dependencies'] = []
    group['dependency_schedule'] = {'version': POLICY, 'inferred': [], 'order': list(group['generation_order']),
                                    'conditioning_edges': [], 'evidence': []}
    return group


def audit_targets(group, results, auditor, directory):
    """One semantic audit per pack of independently generated targets, then every target's decision.

    Packing as in multi_component_audit.complete_components_together: a broad edited surface (>= 10% of the
    image) is shown together with up to two smaller targets inside its box, the rest in pairs. Each pack is
    audited once and the report is shared by its targets (an audit's Ref repair rewrites the members'
    instructions, so auditing the same pack again would see changed inputs). Targets whose supports overlap
    cannot share a provisional image and are audited alone. A target that failed its deterministic checks gets
    no semantic call."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    eligible = [dict(r, _index=i) for i, r in enumerate(results) if r['kept']]
    size = Image.open(group['source_image']).size
    geometry = {}
    for r in eligible:
        member = r['kept'][0]
        mask = load_mask({'mask_path': member.get('mask_v1') or member['mask']}, size)
        ys, xs = np.where(mask)
        if len(xs):
            geometry[r['_index']] = (int(mask.sum()), (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    packs, remaining = [], list(eligible)
    for anchor in sorted(eligible, key=lambda r: geometry.get(r['_index'], (0, ()))[0], reverse=True):
        if anchor not in remaining or anchor['_index'] not in geometry:
            continue
        area, box = geometry[anchor['_index']]
        if area < .1 * size[0] * size[1]:
            continue
        related = [o for o in remaining if o is not anchor and o['_index'] in geometry
                   and geometry[o['_index']][0] * 4 < area
                   and box[0] <= (geometry[o['_index']][1][0] + geometry[o['_index']][1][2]) / 2 <= box[2]
                   and box[1] <= (geometry[o['_index']][1][1] + geometry[o['_index']][1][3]) / 2 <= box[3]]
        if len(related) >= 2:
            pack = [anchor, *related[:2]]
            packs.append(pack)
            remaining = [r for r in remaining if r not in pack]
    packs += [remaining[i:i + 2] for i in range(0, len(remaining), 2)]
    calls = [0]

    def audit_pack(pack, out):
        final, _ = fuse_components(group['source_image'], pack, out / 'provisional')
        calls[0] += 1
        report = audit_component(group, [r['kept'][0] for r in pack], final, auditor, out / 'joint', aligned=True)
        return [(r['_index'], report, final) for r in pack]

    def review(item):
        index, pack = item
        supports = [_union_support(group['source_image'], r['kept']) for r in pack]
        if any((supports[i] & supports[j]).any() for i in range(len(pack)) for j in range(i)):
            return [row for j, r in enumerate(pack) for row in audit_pack([r], directory / f'pack_{index:02d}_{j}')]
        return audit_pack(pack, directory / f'pack_{index:02d}')
    reports = {}
    if packs:
        with ThreadPoolExecutor(max_workers=len(packs)) as executor:
            reports = {key: (report, final) for rows in executor.map(review, enumerate(packs))
                       for key, report, final in rows}
    output = []
    for i, r in enumerate(results):
        report, final = reports.get(i, (None, None))
        output.append(complete_component(group, r['provisional'], r['current'], auditor, directory / f'component_{i:02d}',
                                         joint_report=report, joint_image=final))
    return output, calls[0]


def single_proof(source, member):
    """Exported pixels: outside the support exactly the source, inside it exactly what the audit saw."""
    src = np.asarray(Image.open(source).convert('RGB'))
    out = np.asarray(Image.open(member['step_output_image']).convert('RGB'))
    if out.shape != src.shape:
        return {'pass': False, 'reason': 'dimensions_changed'}
    support = load_mask({'mask_path': member['support_mask']}, (src.shape[1], src.shape[0]))
    if np.any(out[~support] != src[~support]):
        return {'pass': False, 'reason': 'changed_outside_support'}
    image_sha256 = file_hash(member['step_output_image'])
    if member.get('component_audit_sha256') != image_sha256:
        return {'pass': False, 'reason': 'output_not_audited'}
    if member.get('joint_audit_image'):
        _verify_hashes({member['joint_audit_image']: member['joint_audit_sha256']})
        audited = np.asarray(Image.open(member['joint_audit_image']).convert('RGB'))
        if audited.shape != out.shape or not np.array_equal(audited[support], out[support]):
            return {'pass': False, 'reason': 'differs_from_audited_pixels'}
    return {'pass': True, 'outside_support_exact': True, 'audited_pixels_exact': True,
            'image_sha256': image_sha256, 'support_pixels': int(support.sum())}


def generate_independent_singles(plan, directory, editor_factory, auditor, *, seed=0, save_crop_evidence=False):
    """Edit every target of a multi plan from the original photograph; audit; return the group record."""
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    if plan.get('commit_policy') or plan.get('joint_target'):
        raise ValueError('single-target generation requires an ungenerated plan')
    ids = [u['unit_id'] for u in plan['units']]; order = plan['generation_order']
    if len(set(ids)) != len(ids) or len(order) != len(ids) or set(order) != set(ids):
        raise ValueError('invalid generation order')
    immutable_json(root / 'transaction_contract.json', {
        'policy': POLICY, 'plan': plan, 'seed': seed, 'editor': editor_factory().config, 'auditor': auditor.config,
        'save_crop_evidence': save_crop_evidence, 'source_sha256': file_hash(plan['source_image']),
        'mask_sha256': {u['unit_id']: file_hash(u.get('mask_v1') or u['mask']) for u in plan['units']},
        'code': {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}})
    committed = root / 'committed.json'
    if committed.exists():
        saved = json.loads(committed.read_text()); _verify_hashes(saved['image_hashes'])
        if saved['group_sha256'] != digest(saved['group']):
            raise ValueError('single-target final record changed')
        return saved['group']
    group = _independent_plan(plan); group['commit_policy'] = POLICY
    group['support_preflight'] = multi_support_preflight(group, root)
    by_id = {u['unit_id']: u for u in group['units']}
    original = np.asarray(Image.open(group['source_image']).convert('RGB'), dtype=np.int16)
    started = time.monotonic()

    def run_target(uid):
        index = order.index(uid) + 1
        step_dir = root / f'step_{index:02d}_{uid}'
        member = deepcopy(by_id[uid])
        member.setdefault('unit', {})['dependency_context'] = []
        decision = run_committed_step(group, member, group['source_image'], original, step_dir, editor_factory(),
                                      auditor, index=index, seed=seed, save_crop_evidence=save_crop_evidence,
                                      defer_semantic=True)
        out = decision['member']
        print(json.dumps({'group': group['multi_id'], 'step': uid, 'provisional': decision['accepted']}), flush=True)
        return {'current': out['step_output_image'] if decision['accepted'] else group['source_image'],
                'kept': [out] if decision['accepted'] else [],
                'decisions': [{'unit_id': uid, 'operation': out['operation'], 'accepted': decision['accepted'],
                               'reasons': decision['reasons'], 'checkpoint': str(step_dir / 'decision.json')}],
                'provisional': [decision]}

    with ThreadPoolExecutor(max_workers=len(order)) as executor:
        results = list(executor.map(run_target, order))
    results, calls = audit_targets(group, results, auditor, root / 'group_audit')
    decisions = [d for r in results for d in r['decisions']]
    members, exported = [], []
    for d in decisions:
        member = json.loads(Path(d['checkpoint']).read_text())['member']
        if d['accepted'] and member_passes(member.get('semantic_audit') or {}):
            proof = single_proof(group['source_image'], member)
            if proof['pass']:
                exported.append({'unit_id': d['unit_id'], 'image': member['step_output_image'],
                                 'image_sha256': proof['image_sha256'], 'proof': proof})
            else:
                d.update(accepted=False, reasons=['single_proof_failed: ' + proof['reason']])
        members.append(member)
    kept = {e['unit_id'] for e in exported}
    group['units'] = members
    group['n_targets'] = len(kept)
    group['operations'] = [m['operation'] for m in members if m['unit_id'] in kept]
    group['retention'] = {'parent_id': plan['multi_id'], 'planned_n_targets': len(ids),
                          'planned_operations': plan['operations'], 'decisions': decisions,
                          'kept_unit_ids': [uid for uid in order if uid in kept],
                          'dropped_unit_ids': [uid for uid in order if uid not in kept]}
    group['single_parallel'] = {'policy': POLICY, 'components': [[uid] for uid in order],
                                'source_policy': 'original photograph for every target',
                                'semantic_calls': calls, 'exported': exported,
                                'generation_and_audit_seconds': time.monotonic() - started}
    hashes = {p: h for m in members if m['unit_id'] in kept for p, h in _evidence_hashes(m).items()}
    hashes[group['source_image']] = file_hash(group['source_image'])
    atomic_json(root / 'group.json', group)
    atomic_json(committed, {'group': group, 'group_sha256': digest(group), 'image_hashes': hashes})
    return group


def singles_of(group: dict) -> list[dict]:
    """Single-target release records (single_salvage.record fields) of one generated group."""
    from .single_salvage import record
    units = {u['unit_id']: u for u in group['units']}
    out = []
    for item in group['single_parallel']['exported']:
        unit = units[item['unit_id']]
        rec = record('multi', group['multi_id'], group, unit, item['image'], unit['semantic_audit'],
                     reason='independent_target', others=[uid for uid in units if uid != item['unit_id']])
        rec['provenance'].update(export=POLICY, image_sha256=item['image_sha256'], single_proof=item['proof'])
        out.append(rec)
    return out
