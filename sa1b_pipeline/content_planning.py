"""Small content-only planner contract; protocol 2.0 renders all public prose.

Ref + essential edit parameters only, with no generated rationale/observations.
This is opt-in: syntactic rendering is not evidence of correct grounding.
"""
from copy import deepcopy
import json
from .instruction_protocol import make_spec, render
from . import minimal_planning as minimal
from . import content_guard as guard
from .minimal_planning import validate_response, validate_edit

VERSION = 'scene-content-slots-1.4-attribute-value-guard'
SOURCE_HINT_VERSION = 'mask-annotation-hints-2'


def category_hints(objects):
    """Bind hints to the exact ordered objects used by group_views, never ID order.

    Missing/noisy descriptions are not a reason to discard a mask. Keep only
    bounded category strings; source captions, scores and prose are not hints.
    """
    ids = [str(o['object_id']) for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate source object IDs in planning hints')
    hints = {}
    for index, obj in enumerate(objects, 1):
        value = obj.get('category')
        if not isinstance(value, str):
            continue
        value = ' '.join(value.split())
        if value and len(value) <= 120 and value.lower() not in ('object', 'unknown', 'none', 'n/a'):
            hints[f'U{index}'] = value
    return hints


def annotation_hints(objects):
    """Bounded, fallible source evidence, never a verified identity or Ref."""
    categories = category_hints(objects)
    hints = {}
    for index, obj in enumerate(objects, 1):
        uid = f'U{index}'
        hint = {'category': categories[uid]} if uid in categories else {}
        description = obj.get('description')
        if isinstance(description, str) and description.strip():
            hint['description'] = ' '.join(description.split())[:240]
        if hint:
            hints[uid] = hint
    return hints

def obj(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)

def edit_schema(op):
 noun={'type':'string','minLength':1,'maxLength':120}
 if op=='remove':return obj({})
 if op=='replace':return obj({'object':noun})
 if op=='attribute':return obj({'property':{'enum':['color','material','finish','pattern','texture','text','tint','opacity']},'value':noun})
 return obj({'object':noun,'relation':{'enum':['on','in','inside','attached to','hanging from','resting on','around','along']},'site':{'type':'string','maxLength':50}})

def _schema(kind,a):
 ref={'type':'string','minLength':1,'maxLength':160};rej={'type':'string','minLength':1,'maxLength':120}
 if kind=='twins':
  members=dict(type='array',minItems=len(a),maxItems=len(a),items={'anyOf':[obj({'unit_id':{'const':u},'ref':ref}) for u in a]})
  return {'anyOf':[obj({'members':members,'edit':edit_schema(next(iter(a.values())))}),obj({'reject':rej})]}
 variants=[]
 for u,o in a.items():variants.extend([obj({'unit_id':{'const':u},'ref':ref,'edit':edit_schema(o)}),obj({'unit_id':{'const':u},'reject':rej})])
 return obj({'targets':dict(type='array',minItems=len(a),maxItems=len(a),items={'anyOf':variants})})

COMMON='''Design a photo edit someone would naturally request for this scene. Identify the exact selected content, not the object in a mask hole, its neighbor or its larger host. Choose an edit that fits its visible role and leaves plausible anatomy and support; do not force arbitrary decoration.
Ref is a complete English noun phrase starting with "the". Compare lookalikes in the FULL photo; keep only distinctions needed for uniqueness, then stop. Verify positions/ordinals; never invent landmarks or describe annotations.
Use the assigned operation. Reject with a concrete reason only if identification stays ambiguous or the operation has no natural edit; smallness/occlusion alone is not a rejection reason.
Return JSON only. Supply edit CONTENT, not an instruction sentence or explanations. The program writes the standard instruction.'''

RULES={
'remove':'remove: edit is empty {}. Delete the selected content with plausible background completion.',
'replace':'replace: object is a concise noun phrase for a concrete, different physical design suited to the same role/location, not merely a new color/material.',
'attribute':'attribute: property and value describe a meaningful visible change. Match the value to that property and avoid the original value.',
'add':'add: object is a concise noun phrase with an article/count. relation and optional site specify a direct attachment/support inside the visible anchor, with room and without hiding existing content. Empty site means the anchor itself.'}

def _prompt(r):
 s='Image 1: full photo with U locators. Image 2: aligned ORIGINAL CROP | OUTSIDE CONTOUR | BINARY MASK (white=selected), one row per target. Selected colors are unchanged. IDs are not scene order.\n'+COMMON+'\n'+'\n'.join(RULES[o] for o in RULES if o in r['assignments'].values())+'\n'
 if r['kind']=='twins':s+='Each member produces a SEPARATE edited copy of this same source; never edit them together. Members can differ in category. Return each member\'s own ref, plus ONE shared edit content equally natural for every member. The program applies that same edit to each member independently.\n'
 else:s+='Return each requested target once with its own ref/edit or reject. Edits must coexist; reject unsuitable ones without forcing a minimum count.\n'
 return s+'Assigned operations: '+json.dumps(r['assignments'])

def _convert(raw, kind, assignments):
    """Render per target; one invalid multi edit must not erase its neighbors."""
    meta = raw.get('_generation_metadata', {})
    if meta.get('finish_reason') == 'length':
        return {'_generation_metadata': meta, '_raw_output': raw}
    key = 'members' if kind == 'twins' else 'targets'
    rows, errors, specs = [], {}, {}
    try:
        if 'reject' in raw:
            return {'reject': guard.rejection(raw['reject']), '_generation_metadata': meta}
        for target in raw[key]:
            uid = target['unit_id']
            # IDs still fail closed at group level; they bind edits to masks.
            operation = assignments[uid]
            try:
                if 'reject' in target:
                    rows.append(dict(unit_id=uid, reject=guard.rejection(target['reject'])))
                    continue
                edit = raw['edit'] if kind == 'twins' else target['edit']
                kwargs = {}
                ref = guard.reference(target['ref'])
                if operation in ('add', 'replace'):
                    kwargs['payload'] = guard.complete_phrase(edit['object'], 'object')
                if operation == 'attribute':
                    # Literal text payloads may intentionally end with a preposition.
                    value = guard.attribute_value(edit['property'], edit['value'])
                    kwargs['attributes'] = [{'property': edit['property'], 'value': value}]
                if operation == 'add':
                    kwargs['placement'] = {'relation': edit['relation'], 'site': guard.site(edit['site'], ref=ref), 'anchor_mode': 'support'}
                spec = make_spec(operation, ref, ref.removeprefix('the '), **kwargs)
                specs[uid] = spec
                rows.append(dict(unit_id=uid, ref=ref, instruction=render(spec)))
            except (KeyError, ValueError, TypeError, AttributeError) as exc:
                if kind == 'twins':
                    raise
                # Empty instruction intentionally fails the ordinary per-item parser.
                # This is a format error, never a semantic refusal eligible for retry.
                rows.append(dict(unit_id=uid, ref=target.get('ref', ''), instruction=''))
                errors[uid] = str(exc)
        if kind == 'twins':
            if len({row['ref'].casefold() for row in rows}) != len(rows):
                raise ValueError('twin refs must distinguish separate targets; identical refs are ambiguous')
            template = rows[0]['instruction'].replace(rows[0]['ref'], '{ref}')
            if not all(row['instruction'] == template.replace('{ref}', row['ref']) for row in rows):
                raise ValueError('twin content does not render one shared instruction')
            return dict(members=[dict(unit_id=row['unit_id'], ref=row['ref']) for row in rows],
                        instruction=template, _content_specs=specs, _generation_metadata=meta)
        return dict(targets=rows, _content_specs=specs, _content_errors=errors, _generation_metadata=meta)
    except (KeyError, ValueError, TypeError, AttributeError, IndexError) as exc:
        return dict(_generation_metadata=meta, _parse_error=str(exc), _raw_output=raw)


def schema(kind, assignments=None, expected_ids=None):
    # The selector's initial container; every actual request binds its own IDs/operations.
    if not assignments:
        return {'type': 'object'}
    subset = {uid: assignments[uid] for uid in expected_ids} if expected_ids else assignments
    return _schema(kind, subset)


def prompt(kind, assignments, *, locations=None, retry_targets=None, locked=None, previous_reasons=None, view_style='aligned', source_hints=None):
    text = _prompt(dict(kind=kind, assignments=assignments))
    if view_style == 'clean-contours':
        from .full_contour_views import INTRO
        text = INTRO + text.split('\n', 1)[1]
    elif view_style == 'twin-crops':
        from .twin_crop_views import INTRO
        if kind != 'twins' or set(assignments) != {'U1', 'U2'}:
            raise ValueError('twin-crops requires a two-member twin group')
        text = INTRO + text.split('\n', 1)[1]
    elif view_style != 'aligned':
        raise ValueError('unknown content planner view style')
    if source_hints:
        if set(source_hints) - set(assignments):
            raise ValueError('source hints contain unassigned mask IDs')
        text += ('\nSource category hints (may be wrong): ' + json.dumps(source_hints, ensure_ascii=False)
                 + '\nUse hints only to help recognition; trust the photo and actual mask extent when they conflict. '
                   'A category is not a unique ref: distinguish the selected instance in the full photo without expanding a part to its host.\n')
    if locations:
        text += '\nOriginal-image boxes PERCENT [0,100] (not editable extents): ' + json.dumps(locations) + '\n'
    if retry_targets:
        text += '\nOne retry with reassigned operations. Return ONLY ' + json.dumps(retry_targets) + '. Previous rejection: ' + json.dumps(previous_reasons) + '\n'
    if locked:
        text += 'Keep these accepted edits unchanged and compatible: ' + json.dumps([
            dict(unit_id=t['unit_id'], ref=t['ref_phrase'], instruction=t['_minimal_instruction']) for t in locked]) + '\n'
    return text


def decode(raw, kind, assignments):
    converted = _convert(raw, kind, assignments)
    result = minimal.decode(converted, kind, assignments,
                            content_specs=converted.get('_content_specs'))
    if converted.get('_parse_error'):
        result['_minimal_error'] = converted['_parse_error']
        result['reason'] = converted['_parse_error']
        if kind == 'twins':
            result['feasible'] = False
    for target in result.get('targets', []):
        if target['unit_id'] in converted.get('_content_errors', {}):
            target['reason'] = converted['_content_errors'][target['unit_id']]
            target['_format_error'] = True
    result['_content_response'] = deepcopy(raw)
    result['_instruction_rendered_from_slots'] = True
    return result


def normalize(raw, kind, assignments, objects):
    out = minimal.normalize(raw, kind, assignments, objects)
    out['planning_contract'] = VERSION
    return out
