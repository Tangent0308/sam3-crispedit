"""One bounded, same-response reference correction; never rewrite the edit goal."""
from copy import deepcopy
import re

from .instruction_protocol import render, validate_pair, clean

VERSION = 'audit-reference-revision-3-defect-only'
RULES = '''Instruction correction is allowed ONCE within this SAME audit response; no second model call.
First identify the exact SOURCE mask and inspect the actual edit independently. If the edit is correct and only the public reference is ambiguous, redundant or poorly worded, propose instruction_revision with mode="ref", ref="the shortest unique SOURCE noun phrase", reason="brief", same_target=true, unique=true, evidence="visible distinguishing cue and closest rival". Otherwise return mode="keep" with empty ref/reason/evidence and false flags.
A ref correction may add a verified position/host cue or remove redundant words, but MUST keep the same source entity, mask extent, attachment site, operation, payload and attributes. Do not rename an incorrectly edited target, change object identity, remove a required part/site, weaken the request or describe the generated object instead of the source. Incorrect generation, wrong masks/sites and incomplete edits remain failures. If the mask/entity identity itself was wrong, reject instead of relabeling it.
For an eligible correction, assess all reference checks/counts and ref_entity against the corrected ref; assess image checks against the ORIGINAL unchanged edit goal. Put the repaired textual problem in instruction_revision.reason, not serious_defects. Unrepaired defects still fail. Check the FULL corrected reference against the full SOURCE image: a similar neighbor is not ambiguous unless it satisfies all its qualifiers. Keep the noun phrase concise, no additional sentence, coordinates, mask IDs or annotation colors.
Code renders final Ref and no-ref from the unchanged semantic spec, replacing only ref; no-ref keeps "the selected region" and all edit content. Do not emit a free-form replacement instruction or change any semantic slot.
'''


def revision_schema():
    props = {'mode': {'enum': ['keep', 'ref']}, 'ref': {'type': 'string', 'maxLength': 240},
             'reason': {'type': 'string', 'maxLength': 240}, 'same_target': {'type': 'boolean'},
             'unique': {'type': 'boolean'}, 'evidence': {'type': 'string', 'maxLength': 240}}
    return dict(type='object', properties=props, required=list(props), additionalProperties=False)


FIELDS = ('ref', 'instruction', 'instruction_noref', 'instruction_spec', 'instruction_revision', 'unit')


def snapshot(member):
    return {k: deepcopy(member[k]) for k in FIELDS if k in member}


def revise(member, row, *, image_ok):
    """Mutate text only after validated, image-preserving evidence; return error or ''.

    Both rendered sentences are derived from the spec. Original text/spec remain
    in the revision ledger. No caller may turn a failed image check into a pass.
    """
    proposal = row.get('instruction_revision') or {'mode': 'keep'}
    if not isinstance(proposal, dict) or proposal.get('mode') not in ('keep', 'ref'):
        return 'invalid_instruction_revision'
    correcting = proposal['mode'] == 'ref'
    spec = deepcopy(member.get('instruction_spec'))
    if not spec:
        return 'missing_revision_spec' if correcting else ''
    if not image_ok:
        return 'instruction_revision_requires_valid_image' if correcting else ''
    try:
        original_noref = render(spec, False)
        if correcting:
            if member.get('instruction_revision', {}).get('applied'):
                raise ValueError('instruction already revised')
            allowed = set(revision_schema()['properties'])
            if set(proposal) != allowed:
                raise ValueError('unexpected revision fields')
            ref = clean(proposal['ref'])
            if (proposal['same_target'] is not True or proposal['unique'] is not True
                    or not clean(proposal.get('reason')) or not clean(proposal.get('evidence'))
                    or not ref.startswith('the ') or len(ref)>240 or len(ref.split())>32
                    or re.search(r'[.;\n\r!?]|\b(?:selected|mask|crop|U\d+)\b', ref, re.I)):
                raise ValueError('unverified or nonconcise reference')
            spec['ref'] = ref
        instruction, noref = render(spec), render(spec, False)
        validate_pair(spec, instruction, noref)
        if noref != original_noref:
            raise ValueError('revision changed blind edit semantics')
    except (ValueError, KeyError, TypeError):
        return 'invalid_instruction_revision'
    before = {k: deepcopy(member.get(k)) for k in ('ref','instruction','instruction_noref','instruction_spec')}
    after = dict(ref=spec['ref'], instruction=instruction, instruction_noref=noref, instruction_spec=spec)
    if before == after:
        return ''
    member.update(after)
    member['instruction_revision'] = dict(version=VERSION, applied=True,
        kind='reference' if correcting else 'canonical_format', before=before,
        after=deepcopy(after), evidence=deepcopy(proposal), extra_model_calls=0)
    if isinstance(member.get('unit'), dict):
        member['unit'].update(ref_phrase=spec['ref'], instruction_ref=instruction,
                              instruction_noref=noref, instruction_spec=deepcopy(spec))
    return ''
