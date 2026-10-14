"""Experimental four-type proposals within ONE model response, never retries."""
from copy import deepcopy
import json

from .planning_candidates import grounded_neutral_prompt, expand_compact
from .planning_schema import typed_compact_schema

OPERATIONS = ('add', 'replace', 'remove', 'attribute')


def options_schema():
    branches = typed_compact_schema()['anyOf']
    changes = {b['properties']['op']['const']: b['properties']['change'] for b in branches}
    text = {'type': 'string', 'minLength': 1}
    nullable = {'type': ['string', 'null']}

    def obj(props):
        return {'type': 'object', 'properties': props, 'required': list(props),
                'additionalProperties': False}

    return obj({
        'kind': {'enum': ['object', 'part', 'surface', 'printed_motif', 'anatomy']},
        'full_ref': text, 'rival': nullable, 'scope': text, 'source_color': nullable,
        'options': obj({op: obj({
            'change': {'anyOf': [deepcopy(changes[op]), {'type': 'null'}]},
            'limitation': {'type': 'string'}, 'keep': {'type': 'string'},
        }) for op in OPERATIONS}),
    })


def options_prompt(obj, priorities):
    base = grounded_neutral_prompt(obj, priorities)
    rules = base.split('\nReturn JSON only, with kind first.', 1)[0]
    rules = rules.replace(
        'ref_core is a short noun phrase beginning with "the"; distinguish supplies the precise location needed to identify this instance in the unmarked full photograph. Their concatenation is the final reference.',
        'full_ref is ONE complete noun phrase beginning with "the", identifying this exact instance in the unmarked full photograph. Write its identity and distinguishing location together without repetition.')
    rules = rules.replace(
        'Choose the earliest NATURAL type from the priority list; fall back in this same response if necessary.',
        'Consider EACH of the four types independently, and provide one natural proposal for every feasible type. A simpler recolouring does not make addition, replacement or removal infeasible. Do not choose a winning type: software will select the earliest available proposal in the supplied priority list.')
    rules = rules.replace(
        'Only use no_edit if no combination of an accurate reference and a natural allowed edit can be produced; do not reject merely because the region is small or an initially preferred type is unsuitable.',
        'For an infeasible type, set change=null and give a brief factual limitation. For a feasible type, give its change and limitation="". Null is not justified merely by smallness, blur, the need for inpainting, or another type being easier. A removable detail may be erased with natural background reconstruction. A compatible component can receive a different functional design even if not broken. If the target cannot be accurately referenced, set every change to null and explain that reference limitation; never blame source-mask quality alone.')
    return rules + '''

Return ONE JSON response. The four proposals are alternatives for the SAME target, not four sequential edits. Each must be independently natural and preserve unrequested objects. No second request or revision will follow.
Use kind, full_ref, rival, scope, source_color, options. scope is a short target noun, rival is the closest confusable neighbour or null. options has add, replace, remove, attribute, each containing change, limitation, keep. keep is empty unless an essential unselected object needs explicit protection; otherwise it starts with "preserving".
change formats: add={"item":"a ...","relation":"on|in|around|along|at|attached to|sewn onto","site":""}; replace={"replacement":"a ...","design":"physical new form"}; remove={}; attribute={"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}. Use null for an infeasible type, never {} except for feasible removal.
Task: ''' + json.dumps({
        'type_priority': priorities,
        'fallible_mask_annotation': {'category': obj.category, 'description': obj.description},
        'target_in_full_scene': obj.metadata.get('full_scene_geometry'),
    }, ensure_ascii=False)


def select_option(raw, object_id, priorities):
    """Take first non-null proposal verbatim; do not repair or semantically audit."""
    if set(priorities) != set(OPERATIONS) or len(priorities) != len(OPERATIONS):
        raise ValueError('Expected each operation once in priority order')
    if not isinstance(raw, dict) or not isinstance(raw.get('options'), dict):
        raise ValueError('Missing four-type proposals')
    options = raw['options']
    if set(options) != set(OPERATIONS):
        raise ValueError('All four types must be evaluated in the same response')
    for op in OPERATIONS:
        option = options[op]
        if not isinstance(option, dict) or set(option) != {'change', 'limitation', 'keep'}:
            raise ValueError('Malformed type proposal')
        if not isinstance(option['limitation'], str) or not isinstance(option['keep'], str):
            raise ValueError('Malformed limitation or keep clause')
        if option['change'] is None:
            if not option['limitation'].strip():
                raise ValueError('Missing infeasibility explanation')
        elif not isinstance(option['change'], dict) or option['limitation'].strip():
            raise ValueError('Feasible proposal must have a change and no limitation')
    chosen = next((op for op in priorities if options[op]['change'] is not None), None)
    shared = {k: v for k, v in raw.items() if k != 'options'}
    compact = {**shared, 'op': chosen or 'no_edit',
               'change': options[chosen]['change'] if chosen else {
                   'reason': '; '.join(f"{op}: {options[op]['limitation']}" for op in priorities)},
               'keep': options[chosen]['keep'] if chosen else ''}
    result = expand_compact(compact, object_id)
    result['all_type_proposals'] = raw
    result['selection_rule'] = 'first_non_null_in_preassigned_priority'
    return result
