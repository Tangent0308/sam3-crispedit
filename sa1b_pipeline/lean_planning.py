"""Minimal visual planning contract; public instruction/mask protocol stays at 2.0.

No synthetic visual evidence is added by normalization. Version 6 validators
check execution fields; semantic correctness still requires image-based audit.
"""
from copy import deepcopy
from collections import Counter
import json
from .instruction_protocol import BLIND_RULE

VERSION = 'lean-scene-v3'
OPERATIONS = ('add', 'remove', 'replace', 'attribute')
COMMON = '''Design a natural everyday photo edit for each marked source region using its ASSIGNED operation. Read the original scene and exact selected pixels together: include all selected pieces, exclude mask holes and neighbors, and never rename a part as its host. Small size, occlusion or clipping alone are not rejection reasons.
Prioritize edits a person would plausibly request for THIS photograph, with a clear visual purpose grounded in the visible scene and subject. Physical possibility alone is not enough: the result should feel intentional, useful to the desired photograph, and coherent with the subject's identity, function, material, scale and surroundings. Avoid arbitrary decoration or alteration merely to satisfy the assigned operation, a quota or novelty. Consider ordinary personal preferences without inventing an unseen story or changing an acceptable source just because editing is possible. Derive content from this image rather than a reusable default. If no such natural request of the assigned operation fits, set feasible=false with one short concrete reason; otherwise reason="". Keep this intent check internal; do not add rationale fields or explanatory instruction clauses.
Ref: "the" plus the shortest truthful noun phrase uniquely identifying the source among ALL similar instances in the ORIGINAL image. Add only necessary visible distinctions. No explanatory clause, IDs, coordinates, annotation colors or "selected region" in ref_phrase. entity is the physical noun for the exact selected extent. Do not guess hidden or technical identities.
add: retain existing content; add an absent item at a visible direct attachment/support site INSIDE the selected region, with plausible scale and room for its complete shape. No neighboring reference anchor. site is a short host-relative part noun without possessives or repeated host; empty when the selector itself is the contact site. contact_xy is the contact point in ORIGINAL-image coordinates [0,1].
remove: erase exactly the selected dispensable entity, leaving coherent anatomy, support and attachments outside it.
replace: a substitute fitting the location, function and mounting, with a visible physical form/construction difference stated in payload; appearance-only changes are attribute.
attribute: one visibly different, material-appropriate property value; preserve structure and unrequested properties. current_value records the original requested property for attribute only.
Output ONLY fields applicable to the assigned operation: add/replace payload is a concise new-content noun phrase with an article or explicit count; attribute uses one property/value in attributes; only add uses placement/contact_xy. Omit irrelevant fields. Do not output rendered instructions. ''' + BLIND_RULE + '\nReturn only the schema JSON, without explanatory observations or alternatives.\n'
DEPENDENCIES = '''For multi-target edits, keep requests compatible. Return dependencies only for real content interactions, not mere proximity: reveals_edited_surface means modify the surface BEFORE removing its occluder; uses_edited_host means modify the host BEFORE adding; changes_shared_context orders shared occlusion, shadows or reflections. Never delete an addition's host; reject the conflicting edit. Each dependency needs existing U IDs and brief visible evidence. Check every pair even when masks are disjoint; [] if independent.\n'''


def _obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def schema(kind, assignments=None, expected_ids=None):
    from .multi_plan_schema import multi_plan_schema
    old = multi_plan_schema(execution_contract=True)['properties']
    target = old['targets']['items']['properties']
    text = lambda n: dict(type='string', maxLength=n)
    # current_value is retained to detect deterministic no-op attribute requests.
    base = dict(unit_id=text(8), entity=text(65), ref_phrase=text(140),
                current_value=text(70))
    contact = target['grounding']['properties']['contact_xy']
    edit = dict(payload=text(130), attributes=target['attributes'], placement=target['placement'])
    if assignments is not None:
        wanted = list(expected_ids or assignments)
        def fields(uid, op, twin=False):
            values = dict(unit_id={'const':uid}, entity=text(65), ref_phrase=text(140))
            if op == 'attribute': values['current_value'] = text(70)
            if twin:
                if op == 'add': values.update(placement_site=text(60), contact_xy=contact)
            else:
                values.update(feasible=dict(type='boolean'), reason=text(180))
                if op in ('add','replace'): values['payload'] = edit['payload']
                if op == 'attribute': values['attributes'] = edit['attributes']
                if op == 'add': values.update(placement=target['placement']['anyOf'][0], contact_xy=contact)
            return _obj(values)
        variants = [fields(uid, assignments[uid], kind == 'twins') for uid in wanted]
        rows = dict(type='array', items={'anyOf':variants}, minItems=len(wanted), maxItems=len(wanted))
        if kind == 'multi':
            return _obj(dict(targets=rows, edit_dependencies=old['edit_dependencies']))
        op = next(iter(assignments.values()))
        group = dict(members=rows, feasible=dict(type='boolean'), reason=text(180))
        if op in ('add','replace'):group['payload'] = edit['payload']
        if op == 'attribute':group['attributes'] = edit['attributes']
        if op == 'add':group['placement'] = target['placement']['anyOf'][0]
        return _obj(group)
    if kind == 'twins':
        member = _obj(dict(**base, placement_site=text(60), contact_xy=contact))
        return _obj(dict(members=dict(type='array', items=member, minItems=2, maxItems=3),
                         feasible=dict(type='boolean'), reason=text(180), **edit))
    member = _obj(dict(**base, feasible=dict(type='boolean'), reason=text(180),
                       **edit, contact_xy=contact))
    return _obj(dict(targets=dict(type='array', items=member, minItems=1, maxItems=5),
                     edit_dependencies=old['edit_dependencies']))


def prompt(kind, assignments, *, locations=None, retry_targets=None, locked=None, previous_reasons=None):
    views = ('Image 1 is the original scene with outside yellow U locators. Image 2 rows are '
             + ('CONTEXT, BINARY MASK, SELECTED PIXELS on gray. ' if kind == 'twins' else
                'CONTEXT, SELECTED PIXELS on gray, BINARY MASK. ')
             + 'White mask pixels select; black/gray holes exclude. Scales differ. U tags bind rows to scene locations, not scene order.\n')
    special = ('''Twins: each output edits ONE member independently from the SAME source image. Choose ONE shared operation/content natural for EVERY member, without favoring one target. Categories may differ; object/part mixing is allowed (size ratio is checked by code). Payload is for ONE independent edit, not multiplied by member count. Each member has its own ref and placement_site/contact_xy; shared placement.site="". Reject the whole group if the shared request cannot fit all members. Return all member IDs even on rejection.\n'''
               if kind == 'twins' else DEPENDENCIES + 'Return each requested target once, including infeasible ones. Other feasible targets remain usable; code requires at least two.\n')
    result = views + COMMON + special + 'Assigned operations: ' + json.dumps(assignments) + '\n'
    if locations:
        result += 'Original-image boxes [left,top,right,bottom], in PERCENT [0,100] (not contact_xy scale): ' + json.dumps(locations) + '\n'
    if retry_targets:
        result += ('One operation-change retry. Return ONLY these target IDs: ' + json.dumps(retry_targets)
                   + '. Their previous assigned operations were unsuitable: ' + json.dumps(previous_reasons) + '\n')
    if locked:
        result += ('Already accepted edits are LOCKED; do not rewrite or return them. Check compatibility with them. '
                   'Return only dependencies touching retried targets; locked-to-locked dependencies are preserved by code. Locked edits: '
                   + json.dumps(locked) + '\n')
    shape = schema(kind, assignments, retry_targets)
    keys = 'members' if kind == 'twins' else 'targets'
    variants = shape['properties'][keys]['items']['anyOf']
    layout = {k: 'value' for k in shape['properties'] if k != keys}
    layout[keys] = [{k: (v['const'] if 'const' in v else 'value') for k,v in variant['properties'].items()} for variant in variants]
    result += 'Required JSON fields (names only; fill values from the image): '+json.dumps(layout)+'\n'
    return result


def validate_response(raw, kind, assignments, expected_ids=None):
    """Check correspondence before retrying or constructing downstream records."""
    if raw.get('_generation_metadata', {}).get('finish_reason') == 'length':
        raise ValueError('planner output truncated')
    rows = raw.get('members' if kind == 'twins' else 'targets')
    ids = [r.get('unit_id') for r in rows or []]
    expected = set(expected_ids or assignments)
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError('lean planner target IDs missing, duplicated or unknown')
    if kind == 'twins':
        if not isinstance(raw.get('feasible'), bool):
            raise ValueError('lean twin feasibility missing')
        if not raw['feasible'] and not raw.get('reason', '').strip():
            raise ValueError('rejection requires a concrete reason')
    else:
        for t in rows:
            if not isinstance(t.get('feasible'), bool):
                raise ValueError('lean target feasibility missing')
            if not t['feasible'] and not t.get('reason', '').strip():
                raise ValueError('rejection requires a concrete reason')
        from .multi_execution_contract import order_dependencies
        dependencies = raw.get('edit_dependencies', [])
        for d in dependencies:
            if (d.get('relation') not in {'reveals_edited_surface','uses_edited_host','changes_shared_context'}
                    or not str(d.get('evidence', '')).strip()):
                raise ValueError('invalid lean dependency')
        order_dependencies(list(assignments), list(assignments),
                           [[d.get('before'), d.get('after')] for d in dependencies])


def normalize(raw, kind, assignments, objects):
    validate_response(raw, kind, assignments)
    out = deepcopy(raw)
    out['grounding_version'] = 6
    out['planning_contract'] = VERSION
    ids = {f'U{i+1}': str(o['object_id']) for i, o in enumerate(objects)}
    if kind == 'twins':
        ops = set(assignments.values())
        if len(ops) != 1:
            raise ValueError('twins need one shared operation')
        out['operation'] = next(iter(ops))
        out.setdefault('payload', '')
        out.setdefault('attributes', [])
        out.setdefault('placement', None)
        for t in out['members']:
            t.setdefault('current_value', '')
            t.setdefault('placement_site', '')
            t.setdefault('contact_xy', [])
            t['object_id'] = ids[t['unit_id']]
            t['scope'] = t['entity']
            t['grounding'] = {k: t[k] for k in ('entity', 'current_value', 'contact_xy')}
    else:
        for t in out['targets']:
            t.setdefault('current_value', '')
            t.setdefault('contact_xy', [])
            t.setdefault('payload', '')
            t.setdefault('attributes', [])
            t.setdefault('placement', None)
            t.update(object_id=ids[t['unit_id']], operation=assignments[t['unit_id']],
                     scope=t['entity'], target_part='', qualifier='', remove_mode='whole')
            t['grounding'] = {k: t[k] for k in ('entity', 'current_value', 'contact_xy')}
            t['grounding'].update(edit_feasible=t['feasible'], reason=t['reason'])
        out['feasible'] = sum(t['feasible'] for t in out['targets']) >= 2
        out['generation_order'] = list(assignments)  # bind_dependencies topologically sorts this preference.
    return out


def validate_edit(item):
    """Execution consistency, not an assertion of visual correctness."""
    import re
    entity = item.get('scope', '')
    ref = item.get('ref_phrase', '')
    if (not entity.strip() or entity.lower() in {'object','region','selected region','the selected region'}
            or re.fullmatch(r'U\d+', entity, re.I)):
        raise ValueError('unresolved physical entity')
    if not ref.startswith('the ') or re.search(r'\b(selected|mask|region|U\d+)\b', ref, re.I):
        raise ValueError('explicit reference must name the source without selection markers')
    op = item['operation']
    payload, attrs, placement = item.get('payload'), item.get('attributes'), item.get('placement')
    if op in ('add', 'replace'):
        if not isinstance(payload, str) or not payload.strip():
            raise ValueError('missing edit payload')
    elif payload:
        raise ValueError('unexpected payload')
    if op == 'attribute':
        current = item['grounding'].get('current_value', '').strip().casefold()
        if not current or len(attrs or []) != 1 or attrs[0]['value'].strip().casefold() == current:
            raise ValueError('attribute missing original value or repeats it')
    elif attrs:
        raise ValueError('unexpected attributes')
    contact = item['grounding'].get('contact_xy')
    if op == 'add':
        if not placement or placement.get('anchor_mode') != 'support' or len(contact or []) != 2:
            raise ValueError('add requires direct placement and precise contact')
    elif placement is not None or contact != []:
        raise ValueError('unexpected placement/contact')


def choose_retry(assignments, rejected_ids, counts):
    """Deficit priority; never let the model silently switch to an easy type."""
    result = dict(assignments)
    projected = Counter(counts)
    for uid in rejected_ids:
        old = assignments[uid]
        alternatives = [op for op in OPERATIONS if op != old]
        # Rotate ties after the original operation instead of always choosing add.
        alternatives.sort(key=lambda op: (projected[op], (OPERATIONS.index(op)-OPERATIONS.index(old)) % 4))
        result[uid] = alternatives[0]
        projected[result[uid]] += 1
    return result


def merge_retry(first, second, retry_ids, assignments):
    """Keep accepted targets byte-for-byte; refresh only affected dependencies."""
    validate_response(second, 'multi', assignments, retry_ids)
    result = deepcopy(first)
    replacement = {t['unit_id']: t for t in second['targets']}
    result['targets'] = [replacement.get(t['unit_id'], t) for t in result['targets']]
    retry_ids = set(retry_ids)
    deps = [d for d in first['edit_dependencies'] if not retry_ids & {d['before'], d['after']}]
    for d in second['edit_dependencies']:
        if not retry_ids & {d['before'], d['after']}:
            raise ValueError('retry attempted to rewrite locked dependencies')
        deps.append(d)
    result['edit_dependencies'] = deps
    validate_response(result, 'multi', assignments)
    return result
