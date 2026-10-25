"""Opt-in scene planner: compact output, original-color aligned visual inputs.

Semantic quality is evaluated separately from deterministic format validity.
This planner preserves protocol 2.0 and does not change production defaults.
"""
import json
import re
from . import minimal_planning as minimal
from .minimal_planning import decode, validate_response, validate_edit

VERSION = 'scene-ref-instruction-1'
COMMON = 'Plan edits a person could reasonably request for this particular photo. Choose a useful correction or coherent aesthetic change, not decoration merely to fill an assigned operation.\nIdentify each selected region independently before choosing edits: its exact content, not an occluder, neighboring item or larger host. Read colors from the original pixels. The completed scene must preserve plausible anatomy and support.\nRef: the shortest English noun phrase starting with "the" that identifies this target uniquely among ALL lookalikes in the full photo, including unmarked ones. Retain necessary distinctions; verify scene-wide ordering. Do not mention annotations or invent identifying details.\nThe assigned operation must produce a natural, visible change. Reject unresolved identity/ambiguity or an unsuitable operation; smallness and occlusion alone do not require rejection.\nReturn only the required JSON. Use exactly one sentence in the prescribed form. These constraints are NOT instruction content: do not append explanations, background-filling or preservation clauses.'
GRAMMAR = {'remove': 'Sentence: Remove <ref>.', 'replace': 'Sentence: Replace <ref> with <new item>. The new item needs a concrete different physical design that suits the same role/location; color/material-only changes are attribute.', 'attribute': 'Sentence: Change the <color|material|finish|pattern|surface texture> of <ref> to <value>. Or: Change the text on <ref> to "<text>". Or: Tint <ref> <color>. Or: Make <ref> <transparent|translucent|opaque>. Choose a concrete value of the named property, different from the visible original.', 'add': 'Sentence: Add <new item> <relation> <ref>. Or: Add <new item> <relation> the <site> of <ref>. Use an article/count. The anchor must directly support or contain the new item at a visible, available site, without hiding existing content. Relations: on | in | inside | attached to | hanging from | resting on | around | along.'}


def schema(kind, assignments=None, expected_ids=None):
    out = minimal.schema(kind, assignments, expected_ids)
    if not assignments:
        return out
    def sentence(operation, target):
        forms = {
            'remove': 'Remove ' + target,
            'replace': 'Replace ' + target + ' with .+',
            'add': 'Add .+ ' + target,
            'attribute': ('(?:Change the (?:color|material|finish|pattern|surface texture) of '
                          + target + ' to .+|Change the text on ' + target + ' to ".*"|Tint '
                          + target + ' .+|Make ' + target + ' (?:transparent|translucent|opaque))'),
        }
        return '^' + forms[operation] + r'\.$'
    if kind == 'twins':
        op = next(iter(assignments.values()))
        prop = out['anyOf'][0]['properties']['instruction']
        prop['pattern'] = sentence(op, re.escape('{ref}'))
        if op == 'remove':
            out['anyOf'][0]['properties']['instruction'] = {'const': 'Remove {ref}.'}
    else:
        for variant in out['properties']['targets']['items']['anyOf']:
            properties = variant['properties']
            if 'instruction' in properties:
                uid = properties['unit_id']['const']
                properties['instruction']['pattern'] = sentence(assignments[uid], 'the .+')
    return out


def prompt(kind, assignments, *, locations=None, retry_targets=None, locked=None, previous_reasons=None):
    text = ('Image 1 locates targets in the full photo. Image 2 has one row per ID: ORIGINAL CROP | SAME CROP with an OUTSIDE contour | SAME CROP binary mask (white=selected). All columns are aligned; selected colors are unchanged. IDs are not scene order.\n'
            + COMMON + '\n' + '\n'.join(GRAMMAR[o] for o in GRAMMAR if o in assignments.values()) + '\n')
    if kind == 'twins':
        text += "Create independent variants of ONE source: only one member is edited in each variant. Members may differ in category. The shared edit must be equally plausible for each target's role; do not give different content per member. Return unit_id/ref for each member and ONE instruction with literal {ref} exactly once. Otherwise return reject.\n"
    else:
        text += 'Design an independent edit for each target; proposed edits must coexist. Return unit_id/ref/instruction or unit_id/reject for each ID. Each instruction must use its exact ref once. Do not force unsuitable edits to reach a count.\n'
    text += 'Assigned operations: ' + json.dumps(assignments)
    if locations:
        text += '\nOriginal-image boxes PERCENT [0,100] (not editable extents): ' + json.dumps(locations) + '\n'
    if retry_targets:
        text += '\nOne retry with reassigned operations. Return ONLY ' + json.dumps(retry_targets) + '. Previous rejection: ' + json.dumps(previous_reasons) + '\n'
    if locked:
        text += 'Keep these accepted edits unchanged and compatible: ' + json.dumps([
            dict(unit_id=t['unit_id'], ref=t['ref_phrase'], instruction=t['_minimal_instruction']) for t in locked]) + '\n'
    return text


def normalize(raw, kind, assignments, objects):
    out = minimal.normalize(raw, kind, assignments, objects)
    out['planning_contract'] = VERSION
    return out
