"""Scene-grounded content choice, without examples or randomized content menus.

The historical preference API remains callable by existing entrypoints. It now
expresses operation constraints, never a preferred object, property value or family.
"""
VERSION = 'scene-grounded-no-examples-2'

PREFERENCE_RULE = """SCENE-GROUNDED CONTENT CHOICE, inside this SAME planning response:
Identify what the selected entity does and how it is currently used in this scene BEFORE choosing the edit. Choose a request with a plausible visual or practical purpose for that specific entity in its surroundings. Mere geometric fit or ease of rendering is insufficient. Judge the desired result against the original material, occupancy, physical scale, use and visible environment. A coherent personal preference is allowed; do not require proof of the photographer's intent.
Derive the content from these observations, not a default addition, property value or reusable content template. Variation should follow real differences between scenes and targets. Repetition is acceptable when independently justified by the image; forced novelty or an unnatural edit is not. Before finalizing content, write scene_role for the source entity and edit_purpose for the chosen result, each at most 10 words. scene_role is its actual visible use, not merely its shape or surface orientation. edit_purpose states what this particular change achieves in this scene. In fit/reason, check that the exact chosen content achieves that purpose while preserving required function. Geometric fit, generic plausibility and a bare claim of decoration are not substitutes for a scene-specific purpose. Do not put this explanation or alternatives in the published instruction.
Keep the assigned operation and exact selected extent. Within this response, discard an unsuitable idea and choose a better-fitting request; do not reject a valid region merely because the first idea failed. Reject only a concrete incompatibility when no natural request of the assigned operation fits. For twins the SAME content must be independently natural for EVERY member, without favoring one target in the blind instruction. No additional model calls or alternative plans are requested.
"""

_OPERATION_CONSTRAINTS = {
    'add': 'Choose new content whose presence makes sense for this entity and scene. It must have a visible direct contact/support site within the selected region, physically appropriate attachment and enough space for its complete shape. Preserve existing contents and function. Do not limit additions to decorative accessories or favor flatness merely for easier generation.',
    'remove': 'The selected entity must be dispensable in the intended scene. Its removal must leave coherent surroundings, support and remaining attachments without deleting excluded content.',
    'replace': 'Choose new content with a visibly different physical form or construction that is natural at this location and compatible with its function, scale and mounting. The new content must still make sense as a substitute for the selected entity in its scene; replacing a functional entity with a generic shape that cannot serve its role is unsuitable. A mere appearance change does not establish replacement.',
    'attribute': 'Choose a visibly different property value appropriate to the actual selected material and its use in this environment. Preserve physical structure and every unrequested property; do not select a value merely because it is easy to name.',
}


def content_preferences(operation, objects):
    """Compatibility API: IDs/categories never choose the requested content."""
    return _OPERATION_CONSTRAINTS.get(operation, '')


def preference_prompt(operations, objects):
    operations = sorted(set(operations) & _OPERATION_CONSTRAINTS.keys())
    if not operations:
        return ''
    return '\n' + PREFERENCE_RULE + '\n'.join(
        op + ': ' + content_preferences(op, objects) for op in operations)
