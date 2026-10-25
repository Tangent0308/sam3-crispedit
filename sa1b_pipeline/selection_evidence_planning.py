"""Opt-in, single-call planning experiment; production defaults are unchanged.

The observation is model testimony, not a grounding verdict. It is diagnostic
metadata only and must never change the source mask or the training instruction.
"""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator, ValidationError

from . import per_mask_planning as baseline

VERSION = 'selection-evidence-ab-1'
COMMON = '''Identify ALL selected source pixels in Image 2 and find that same region in the clean full photo. Excluded pixels, including holes, are not selected. Do not substitute a nearby recognizable object. Distinguish an entire object from its visible surface or part; the request must match the actual selection, not a smaller convenient part or its larger host.
Design one change a person would plausibly request to improve this photo, suited to the selected content's function and surroundings. Judge the resulting scene, not just whether the words are grammatically possible. Keep the assigned operation; if no natural request of that type fits this selection, reject with a concrete reason. Do not force a different operation into the assigned label. Size, occlusion or being a part is not by itself a reason to reject.
Ref is one English noun phrase starting with "the". Locate it in the FULL unmarked photo, check competing instances, and use the shortest verified distinction that singles out this selection. Use full-photo directions; use an ordinal only with a clear group and verified count. Never include annotation labels, crop coordinates or uncertain details.
Return JSON content slots, not finished instruction sentences. Do not narrow or move the selection to make an edit fit.'''
RULES = {
    'add': '''add: keep the selected anchor and its existing contents. addition names new content with an article/count; relation and site place it directly on/in/attached to a visible part of this anchor. Use site="" if the selected surface itself is the site; otherwise name only that subpart, without repeating the anchor. The addition must fit, serve a plausible purpose, and not duplicate something already present. Distinguish a physical object from a printed/embroidered design.''',
    'remove': '''remove: edit={}. The selected content must actually disappear, with plausible background completion. Reject if this would leave the remaining subject anatomically broken, unsupported or unusable. Do not describe hiding, repainting or reconstructing the same content as removal.''',
    'replace': '''replace: replacement names a new physical object/design with an article/count. It must substitute for the selected entity and fit its surroundings while preserving the host. Changing the same item's color, material, finish or hairstyle is attribute, not replace; covering it with an accessory is add. If only those changes make sense, reject this operation. Do not copy source locating details into replacement.''',
    'attribute': '''attribute: property names the changed attribute; value contains only the desired result, not a command or source locator. Keep the selected entity and its structure. The change must be meaningful across the intended selected content and natural in the scene. For text, provide new written wording; for color, preserve unrelated text and motifs.''',
}
OBSERVATION = ('First output selected_content: one short factual phrase naming the full '
               'selected source content and its extent, before ref/edit or reject. '
               'Describe what is present, not the desired result. Do not invent a '
               'confident identity if it is unclear. This is not an extra editing request.')


def request_prompt(operation, image_paths, hint=None, *, evidence=True):
    if operation not in RULES:
        raise ValueError('unknown operation')
    # Reuse the exact baseline image/geometry/hint binding; do not add an image.
    original = baseline.request_prompt(operation, image_paths, hint)
    suffix = original[original.index('\nAssigned operation:'):]
    text = baseline.INTRO + COMMON + '\n' + RULES[operation]
    text += '\n' + (OBSERVATION if evidence else 'Return only ref/edit or reject JSON.')
    return text + suffix


def schema(operation, *, evidence=True):
    result = deepcopy(baseline.schema(operation))
    if evidence:
        for branch in result['anyOf']:
            branch['properties'] = {
                'selected_content': {'type': 'string', 'minLength': 1, 'maxLength': 120},
                **branch['properties'],
            }
            branch['required'] = ['selected_content', *branch['required']]
    return result


def combine(raws, kind, assignments, objects, *, evidence=True):
    if len(raws) != len(assignments):
        raise ValueError('per-mask response count mismatch')
    clean, observations, errors = [], {}, {}
    for (uid, operation), raw in zip(assignments.items(), raws):
        response = deepcopy(raw)
        metadata = response.pop('_generation_metadata', {})
        try:
            Draft202012Validator(schema(operation, evidence=evidence)).validate(response)
            if evidence:
                observation = response.pop('selected_content')
                if not observation.strip():
                    raise ValueError('selected_content is blank')
                observations[uid] = observation
        except (ValidationError, ValueError) as exc:
            errors[uid] = str(exc).split('\n')[0]
            response = {'reject': False}  # Baseline converts this to a format failure.
        response['_generation_metadata'] = metadata
        clean.append(response)
    plan = baseline.combine(clean, kind, assignments, objects)
    plan['planning_contract'] = VERSION + ('-observation' if evidence else '-concise')
    plan['_selection_observations'] = observations
    plan['_selection_observation_errors'] = errors
    plan['_experiment_raw_responses'] = deepcopy(raws)
    return plan
