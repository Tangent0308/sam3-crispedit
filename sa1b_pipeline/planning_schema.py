"""Experimental decoding contract for the existing one-call compact planner.

This constrains representation and explicit type contradictions. It cannot
certify visual identity, reference accuracy, or naturalness.
"""
from copy import deepcopy


def typed_compact_schema(priorities=None, *, contrast_first=False, single_phrase=False,
                         skip_reason_max_chars=0):
    if skip_reason_max_chars < 0:
        raise ValueError('skip reason limit must be nonnegative')
    text = {'type': 'string', 'minLength': 1}
    nullable_text = {'type': ['string', 'null']}
    kinds = ['object', 'part', 'surface', 'printed_motif', 'anatomy']

    def obj(properties):
        return {'type': 'object', 'properties': properties,
                'required': list(properties), 'additionalProperties': False}

    changes = {
        'add': obj({'item': text, 'relation': {'enum': [
            'on', 'in', 'around', 'along', 'at', 'attached to', 'sewn onto',
            'beside', 'above', 'below', 'under']},
                    'site': {'type': 'string'}}),
        'replace': obj({'replacement': text, 'form_change' if single_phrase else 'design': text}),
        'remove': obj({}),
        'attribute': obj({'property': {'enum': [
            'color', 'material', 'finish', 'pattern', 'texture', 'text', 'tint', 'opacity']},
                          'value': text}),
        'no_edit': obj({'reason': text}),
    }
    if priorities is not None and (
        not priorities or len(set(priorities)) != len(priorities)
        or set(priorities) - (set(changes) - {'no_edit'})
    ):
        raise ValueError('priorities must be unique editing operations')
    branches = []
    for operation, change in changes.items():
        if priorities is not None and operation not in [*priorities, 'no_edit']:
            continue
        allowed_kinds = [k for k in kinds if not (
            operation == 'remove' and k == 'anatomy'
            or operation == 'replace' and k in {'printed_motif', 'anatomy'})]
        properties = {
            'kind': {'enum': allowed_kinds},
            **({'full_ref': text} if single_phrase else {'ref_core': text, 'distinguish': {'type': 'string'}}),
            'rival': nullable_text, 'scope': text, 'source_color': nullable_text,
            'op': {'const': operation},
        }
        if priorities is not None:
            skipped = priorities if operation == 'no_edit' else priorities[:priorities.index(operation)]
            reason = {**text, **({'maxLength': skip_reason_max_chars} if skip_reason_max_chars else {})}
            properties['skipped'] = obj({op: reason for op in skipped})
        properties.update(change=change, keep={'type': 'string'})
        if contrast_first:
            # Make the observed target/nearest competitor available before
            # decoding the final referring expression, without another call.
            order = ['kind', 'scope', 'source_color', 'rival', *(['full_ref'] if single_phrase else ['ref_core', 'distinguish']), 'op']
            properties = {**{key: properties[key] for key in order},
                          **{key: value for key, value in properties.items() if key not in order}}
        branch = obj(properties)
        branch['required'].remove('keep')
        branches.append(branch)
    return deepcopy({'anyOf': branches})
