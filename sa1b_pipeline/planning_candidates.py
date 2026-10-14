"""Experimental one-call planner candidates; not the production default yet."""
from __future__ import annotations
import json
import re
from .instruction_style import INSTRUCTION_STYLE

CANDIDATE_VERSION = 'experimental-compact-v2.5-r4-concise2-directadd1'

COMPACT_PROMPT = '''Write one realistic, local photo-edit request for the EXACT supplied mask. Two images: full scene with a cyan target bounding rectangle, then one enlarged crop with the mask outlined in magenta/white. Only pixels inside the contour are selected; holes and surrounding pixels are NOT. The contour is outside the mask; target colors are original. Use the full scene to identify where the crop belongs. Do not mistake a nearby prominent object for the target. A background region can surround an excluded foreground person.

FIRST identify the actual selected object/part/surface/printed motif. A petal is not the whole flower; a thin trim is not the whole car panel; a printed flower is not a live flower. Blur, smallness, occlusion and incomplete borders alone are not reasons to reject. Name only visible facts, not guessed anatomy, driver/passenger side, or hidden function. Use image-left/right.

REFERENCE: return ref_core (the target noun with useful appearance) and distinguish (the spatial/shape description that makes THIS instance unique in the unmarked full scene). These two fields are concatenated verbatim into the final Ref. Compare the closest similar neighbor; the final phrase must exclude that neighbor. For repeated petals/leaves, clock direction or upper-left alone is insufficient: use a clearly defined counting group and direction, OR shape plus precise relation to a recognizable neighbor. Count only what you can see; never invent an ordinal. For two rails/chains, explicitly say upper/lower or left/right. For a tiny motif, use its nearest distinctive flower/button. Do not merely say 'above the belt' if several leaves match. Do not mention mask, contour, selected, crop, coordinates, or IDs in ref_core/distinguish. If already unique, distinguish may be empty. Keep the combined phrase concise but never omit the distinguishing relation. rival briefly names the most confusable neighbor (or null), not reasoning.

Before finalizing the reference, mentally remove all annotation from BOTH images. The user will not see the cyan rectangle or crop. 'Inside the cyan box' is never a distinguishing relation. rival must be the CLOSEST visually similar instance beside the target, not an easier distant object. If several petals occupy one clock direction, describe their order in a bounded visible group (state the start and counting direction) or a unique silhouette and immediate neighboring relation. A relation like 'left of the next petal' is circular. Do not invent a pocket, button ordinal, stem, or function to make the sentence sound specific. Prefer one accurate relation over a long speculative description.

EDIT: type priority is a balancing preference, subordinate to naturalness. First eliminate types that would be unnatural for this exact target, then choose the earliest remaining type IN THIS SAME RESPONSE. For a living petal or leaf, normally prefer a local color change, small dew drop, or appropriate insect over fabricating a different isolated organ shape. Replace is suitable for detachable objects, compatible components and printed designs with an obvious new form; it is not mandatory for every target. Attached human skin cannot be removed, but a small plausible tattoo, ring on a visible finger, or local skin-tone/texture correction can be valid. A failed remove proposal does not make the mask unusable. Only use no_edit if NONE of the allowed types permits an accurately referenced natural edit. Do not fabricate an unnatural edit for balance.
- add: keep the selected host; add a plausible item on/in/around it at a sensible scale. The added item may protrude. Printed additions must be called printed/embroidered. Do not add real dew to a printed flower, obstruct working parts, or use an unselected neighbor as support.
- replace: preserve the function and contacts; use a visibly different shape/design, not just color/material/tint. Give replacement as a noun phrase and design as the explicit new geometry/style (e.g. 'a button' + 'a square outline and four holes'). Code renders 'replacement with design'. If only a color/material change makes sense, use attribute. Do not invent a jagged/star-shaped single living petal or a different species of isolated leaf merely to satisfy replace: choose a natural allowed add/attribute/remove instead. Do not turn ventilation into a solid plate or glass into an opaque body panel. Keep unselected container contents/supports.
- remove: remove only the selected target; leave a natural scene. Never delete attached human anatomy or supports of unselected objects. Removing printed motifs erases the design, not the fabric. Remove a visible background object despite blur if its identity is clear. For attached skin choose another type or no_edit; do not mislabel it as an independent object to remove it.
- attribute: visibly change one plausible color/material/finish/pattern/texture/text/tint/opacity of exactly this target. Keep geometry and unrequested components. A motif's color may change without changing cloth. Do not recolor an entire person, make viewing glass opaque, or smooth already fully blurred skin (no visible change). Opacity values: transparent, translucent, opaque; tint preserves transparency. Do not request the current property value.

Output compact JSON only, no explanations or alternative plans. scope is a short target noun without position/color/article; kind is object|part|surface|printed_motif|anatomy. Schema:
{"ref_core":"the ...","distinguish":"...","rival":null,"scope":"...","kind":"...","source_color":null,"op":"add|replace|remove|attribute|no_edit","change":{...}}
change by op:
add: {"item":"a/an/count + new item","relation":"on|in|around|along|at|attached to|sewn onto","site":"optional selected host part, otherwise empty"}
replace: {"replacement":"a/an + new object","design":"visible new geometry or design"}
remove: {}
attribute: {"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}
no_edit: {"reason":"specific reference or edit limitation after considering ALL allowed types, never merely poor mask"}
Omit keep unless preserving unselected contents is essential; ordinary geometry/detail preservation is already implicit. If needed, keep starts with 'preserving'. Do not request a smaller unlocalized subpart. site should be a bare part name such as 'upper surface', 'rim', or 'middle finger', never 'the center of the petal' when scope is petal. Do not repeat ref_core in distinguish. Preserve existing detail unless explicitly changed. Final check within this response: target identity, distinction from closest neighbor, natural operation, and no annotation words in the final Ref.
'''


def compact_prompt(obj, priorities):
    return COMPACT_PROMPT+'\n'+INSTRUCTION_STYLE+'\nTask: '+json.dumps({
        'allowed_types_in_priority_order':priorities,
        'target_in_full_scene':obj.metadata.get('full_scene_geometry'),
        'resolved_execution_region_available':bool(obj.metadata.get('execution_mask_path')),
        'coordinates_are_context_only_not_reference_text':True,
    },ensure_ascii=False)


def binding_prompt(obj, priorities):
    """Experimental r4: local evidence first, fewer speculative descriptors."""
    prompt = compact_prompt(obj, priorities)
    prompt = prompt.replace(
        'Replace is suitable for detachable objects, compatible components and printed designs with an obvious new form;',
        'Replace is suitable for detachable objects and compatible physical components with an obvious new form;')
    prompt = prompt.replace('Removing printed motifs erases the design, not the fabric.',
                            'Removing printed motifs erases the design, preserving the existing substrate.')
    anchor = '\n\nREFERENCE:'
    rules = '''

LOCAL EVIDENCE BEFORE CONTEXT: The selected pixels in Image 2 retain their original colors and brightness; the surrounding context is dimmed. Identify these actual pixels before considering the host or scene. A blue outline does not make an object blue. If the target looks like stone, foliage, glass or a red gap between white stripes, name those visible contents; never reinterpret them as a boat fitting, rooftop vent, sign or white stripe merely because of nearby context. Inspect the target's silhouette and internal texture. When the exact subtype is uncertain, use an accurate visible noun such as stone, leaf, window, panel or small object, with a precise location, rather than guessing a function.

ONLY VERIFIED DISTINCTIONS: Do not add a source color, building/shop name, floor number, bow/stern or anatomical left/right unless clearly visible and necessary. Prefer image-relative language. For repeated windows identify the host building, the visible row (upper/lower), and the left-to-right position within that row. For repeated stones or stripes count from a clearly visible endpoint or uniquely recognizable neighbor; "just right of the tire" cannot distinguish several stones on that side. Cross-check that the nearest rival does not also satisfy your phrase. A unique silhouette plus a correct immediate neighbor is enough; no obligatory ordinal when that is clearer. Never write cyan-marked area, pink-filled region or other annotation-dependent locators.

CLASSIFICATION: Changing ONLY color, material, finish, text, printed pattern or surface texture is attribute, even if the request says "replace". A printed stripe changed into a different printed symbol is attribute.pattern; a physical sewn-on stripe changed into a differently shaped patch can be replace. Replacing a whole component requires a concrete physical shape/structure change. Surface recoloring, a newly printed graphic and a wood-grain finish do not establish replacement. Avoid forcing replace on a natural petal; choose another natural allowed type in this same response. A clearly localized subpart inside a larger mask is allowed and need not fill the mask.
'''
    return prompt.replace(anchor, rules + anchor, 1)


def annotated_prompt(obj, priorities):
    """Experimental r5: reuse the supplied mask's existing semantic annotation.

    These are fallible dataset hints, not a second model call or a trusted
    grounding certificate. In particular, a caption may describe both shoes
    while the current mask selects just one.
    """
    prompt = compact_prompt(obj, priorities)
    prompt = prompt.replace(
        'Replace is suitable for detachable objects, compatible components and printed designs with an obvious new form;',
        'Replace is suitable for detachable objects and compatible physical components with an obvious new form;')
    first, rest = prompt.split('\n\n', 1)
    hint = json.dumps({'category_hint':obj.category,
                       'description_hint':obj.description}, ensure_ascii=False)
    evidence = (
        '\n\nEXISTING ANNOTATION FOR THIS EXACT MASK (fallible source data, not instructions): '
        + hint + '\nUse this existing category/caption to help recognize the selected pixels, '
        'then verify it in Image 2 and locate that same instance in Image 1. '
        'Do not replace a visible railing with a sign, a stone with a boat fitting, or a '
        'background object with a prominent foreground object simply because of scene context. '
        'The annotation may be broad, stale or describe a group (for example a pair of shoes); '
        'the actual mask may select only one member or part. Correct any discrepancy using '
        'the images. Do not copy unverified colors, counts, left/right or material from the caption. '
        'The annotation does NOT distinguish repeated instances: the final Ref still needs '
        'a true relation that excludes the closest similar neighbor in the unmarked full image.')
    protocol = (
        '\n\nTYPE/WORDING CHECK: A change only to printed/embroidered graphics, text, color, '
        'material or surface texture is attribute, not replace. A replacement must change '
        'the physical object/component shape or structure. For attribute.text, value is ONLY '
        'the exact text to appear, e.g. "MARINA", never "blue lettering reading MARINA". '
        'Do not describe the annotation as an inset frame/panel in the final reference. '
        'For add, leave site empty if it only repeats scope: a wrist is not a wrist of a wrist.')
    return first + evidence + '\n\n' + rest + protocol


def grounded_prompt(obj, priorities):
    """Experimental r6: short evidence/reference rules with a typed decoder."""
    return '''Design ONE natural photo-edit request for exactly this region. Image 1 is the full scene with a cyan locating rectangle. Image 2 is a zoomed crop: target pixels retain their original appearance inside a magenta/white contour; surrounding context is dimmed. Holes and neighbours are NOT selected. Read the marked pixels first, then locate the same instance in the full scene. The annotations are not part of the photograph.

IDENTITY AND REFERENCE
Name the actual content of the region: a petal, printed leaf, trim, pane, chain, patch of skin, etc., not its entire host or an adjacent prominent object. The supplied dataset caption is a fallible hint: it may name a group or whole object while this mask selects one member/part. Verify everything against the images.
full_ref is the ENTIRE minimal referring noun phrase beginning with "the". Do not produce separate core/location fields. Start with the target noun and add only enough verified information to distinguish the nearest similar neighbour. If the noun is unique, no position is needed. Prefer one compact modifier over a description followed by a location clause; a host and a position are not both mandatory. Mentally delete each modifier and retain it only if a rival would then also match.
For repeated parts, prefer a visible distinctive shape plus its immediate landmark. If counting is necessary, define the visible row/group and count in image-left-to-right order. Count only visible instances; do not count the crop edge as a starting landmark. Do not invent a clock position, floor number, ordinal, hidden function, source colour or anatomy. In particular, "near the top of the flower" does not distinguish adjacent upright petals. A clearly visible notch plus its relation to the downward-facing petal can. A correct short reference is better than adding uncertain qualifiers. Never refer to a box, outline, inset, mask, crop, coordinates or IDs. Smallness, blur and imperfect borders alone are not failures.

EDIT AND TYPE
Choose the earliest NATURAL type from the priority list; fall back in this same response if necessary. The requested result should be an ordinary plausible retouch of this scene. Keep all unrequested components. A precisely named subpart within the region is allowed and need not occupy the whole region.
- add: a new physical accessory/object or a new printed/tattoo detail, attached to the selected host at a plausible scale and place. Preserve the host. Real droplets belong on a real petal, not a flower print. item is a noun phrase including article/count. site is a bare host part or empty; do not repeat scope.
- replace: a physical object/component changes to another compatible physical form. Include the actual geometry/structure difference ONCE in replacement (e.g. "a square four-hole button"). form_change is internal evidence only; it must not carry a requested feature missing from replacement. A printed logo or stripe becoming another printed design is attribute.pattern, never replace. Changing only colour, material or finish is also attribute. Do not invent an arbitrary species/shape for one living petal or leaf. Do not turn transparent viewing glass into an opaque plate.
- remove: delete a separable object or a natural removable detail, leaving a plausible scene. Do not remove attached anatomy or supports of retained contents. Do not remove part of a worn garment if that requires inventing the person's unselected body or leaves a broken garment. Erasing a printed motif preserves its substrate. A small background object may be removed despite blur if correctly identified.
- attribute: change one visible property: colour, material, finish, printed pattern, texture, text, tint or opacity. Preserve identity, geometry and unrequested markings. A new tattoo is add, not texture. text.value is ONLY the exact inscription. Tint keeps transparency. Never request the current property value, opaque paint on viewing glass, or a no-visible-change retouch of already blurred skin.
Only use no_edit if no combination of an accurate reference and a natural allowed edit can be produced; do not reject merely because the region is small or an initially preferred type is unsuitable.

Return JSON only, with kind first. kind: object|part|surface|printed_motif|anatomy. scope is a short noun without appearance or location. rival names the closest confusable neighbour, or null. source_color may be null. Do not output explanations.
{"kind":"...","full_ref":"the ...","rival":null,"scope":"...","source_color":null,"op":"...","change":{...}}
change by op: add={"item":"a ...","relation":"on|in|around|along|at|attached to|sewn onto","site":""}; replace={"replacement":"a complete new-object noun phrase INCLUDING the changed physical form","form_change":"brief internal evidence of the physical difference already expressed in replacement"}; remove={}; attribute={"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}; no_edit={"reason":"specific instruction limitation"}.
Optional keep starts with "preserving" and is only needed to protect essential unselected contents.
''' + INSTRUCTION_STYLE + '\nTask: ' + json.dumps({
        'allowed_types_in_priority_order': priorities,
        'fallible_mask_annotation': {'category': obj.category, 'description': obj.description},
        'target_in_full_scene': obj.metadata.get('full_scene_geometry'),
        'resolved_execution_region_available': bool(obj.metadata.get('execution_mask_path')),
        **({'selected_pixel_measurements': obj.metadata['selected_pixel_evidence'],
            'measurement_use': 'Check identity against these raw target-only RGB samples. They exclude annotation colours and neighbours. Lighting affects RGB; do not blindly turn samples into a semantic colour name.'}
           if obj.metadata.get('selected_pixel_evidence') else {}),
    }, ensure_ascii=False)


def grounded_neutral_prompt(obj, priorities):
    """r6 reference-only ablation without a development-scene landmark example."""
    prompt = grounded_prompt(obj, priorities)
    prompt = prompt.replace(
        'In particular, "near the top of the flower" does not distinguish adjacent upright petals. '
        'A clearly visible notch plus its relation to the downward-facing petal can.',
        'A location shared by several similar instances is insufficient. '
        'Use a distinguishing shape only if it is visible on THIS instance, and a landmark '
        'relation only if the full scene confirms that relation. Do not borrow a feature '
        'or position from a neighbouring instance.')
    return prompt.replace(
        'Never refer to a box, outline, inset, mask, crop, coordinates or IDs.',
        'Never refer to annotation boxes or outlines, an inset/crop view, the supplied mask, '
        'coordinates or IDs. Ordinary objects that actually exist in the photograph '
        'may still be named.')


def grounded_balanced_prompt(obj, priorities):
    """r9: make the preassigned type preference an explicit decision gate.

    This remains one response and one request.  It does not create candidate
    alternatives; it tells the model when a non-attribute edit is genuinely
    natural so that the balancing schedule is not defeated by a conservative
    attribute fallback.
    """
    prompt = grounded_neutral_prompt(obj, priorities)
    gate = '''

TYPE DECISION GATE (must be applied before writing op): The first item in
allowed_types_in_priority_order is a real balancing requirement. Try one
ordinary, concrete edit of that type for the selected pixels. Do not choose
attribute merely because recolouring is easiest. Use add for a plausible local
accessory, printed/tattoo detail or small item on a real host; use replace for
a same-function physical component or compatible design with a visible shape
change; use remove for a separable background object, litter, detachable detail
or printed motif whose substrate remains; use attribute only when all earlier
types would be unnatural, unsafe, or outside the selected region. If you fall
back, state the concrete semantic limitation only in internal `skipped` text
when requested; never put a fallback reason in `keep` or the instruction.
Never call an edit unnatural solely because the target is small, blurred or
partly occluded. Never choose a type just to satisfy a quota when it would
damage a person, garment, support, transparent pane or host.

REFERENCE GATE: `full_ref` alone must distinguish the selected instance.
When the noun already identifies it, stop. Otherwise use the shortest verified
modifier that excludes its closest rival. Keep necessary position INSIDE this
one noun phrase, not an appended explanation. `rival` is internal evidence,
not a substitute for an essential locator. A generic "upper-right petal" is
insufficient when two petals match; prefer a visible discriminating landmark
or a verified ordinal within a defined group. Never guess unreadable labels.

MASK SCOPE GATE: The mask is the authoritative extent. Never promote a small
window, door panel, leaf fragment, sleeve/skin patch, or partial car to its
whole vehicle, garment, person, plant, or parent object. If a background mask
surrounds a foreground hand/person, describe only the selected background
pixels; do not infer the surrounded object. Before choosing remove, explicitly
check that the named target is not a person's or animal's attached body part,
even when `kind` is uncertain. For a narrow or fragmented mask, use a narrow
visible noun and a verified local relation. The edit sentence must not ask for
anything outside that selected extent.
'''
    return prompt.replace('\n\nReturn JSON only, with kind first.', gate + '\n\nReturn JSON only, with kind first.', 1)


def assigned_operation_prompt(obj, priorities):
    """One requested operation or explicit infeasibility; never a type fallback."""
    if len(priorities) != 1:
        raise ValueError('assigned planning requires exactly one allowed operation')
    prompt = grounded_balanced_prompt(obj, priorities)
    start = prompt.index('TYPE DECISION GATE')
    end = prompt.index('REFERENCE GATE:', start)
    prompt = prompt[:start] + '''ASSIGNED TYPE CONTRACT: Only the single supplied operation is allowed.
Return that operation with one ordinary, natural, in-scope edit, or no_edit
with a short concrete reason why THIS operation cannot be done naturally or
referenced unambiguously. Do not return a different operation, relabel an
attribute edit as replace/add, broaden the mask to its host, or invent an
unnatural request to fill a quota. no_edit means this assigned operation is
unsuitable; it does NOT mean the source mask is bad. There is no second call.
Consider a plausible same-function replacement with a visible form change,
a directly supported/attached addition, or removal of a separable detail when
the assigned type allows it. An intact object need not be broken to replace
it, and a small or blurred object can still be removed when clearly named.
Keep body parts and supports intact. Removing an attached arm is infeasible.

''' + prompt[end:]
    prompt = prompt.replace(
        'Choose the earliest NATURAL type from the priority list; fall back in this same response if necessary.',
        'Design a natural edit of the SINGLE assigned type; no type fallback is permitted.')
    prompt = prompt.replace(
        'Only use no_edit if no combination of an accurate reference and a natural allowed edit can be produced; do not reject merely because the region is small or an initially preferred type is unsuitable.',
        'Use no_edit if THIS assigned type has no natural, unambiguously referenced edit. Smallness, blur or imperfect mask borders alone are not a reason.')
    return prompt


def grounded_supported_prompt(obj, priorities):
    """Candidate: use source semantics to resolve blur without trusting its extent.

    The annotation is already supplied for this exact mask.  No additional
    model request or derived label is introduced.  This is not promoted to the
    production planner until measured on the fixed cohort and new sources.
    """
    prompt = grounded_balanced_prompt(obj, priorities)
    old = ('The supplied dataset caption is a fallible hint: it may name a group or whole object '
           'while this mask selects one member/part. Verify everything against the images.')
    new = ('The supplied category and description are existing evidence about THIS mask, not '
           'a caption of the whole photograph. Use their semantic identity as a starting '
           'hypothesis and check it against the actual selected pixels. Override that identity '
           'only for a clearly visible contradiction in the selected pixels; a prominent '
           'neighbour, a hole in the mask, or an uncertain guess is not contrary evidence. '
           'When blur hides fine detail, prefer a supported broad noun to an invented precise '
           'identity. This does not make the annotation authoritative: its colour, instance '
           'count, spatial qualifiers and extent still require visual checking. If it names '
           'a group or entire host, narrow the noun to the visible member or part actually '
           'covered by this mask. Use the full image, not the caption, to distinguish neighbours.')
    if prompt.count(old) != 1:
        raise ValueError('grounding annotation anchor changed')
    return prompt.replace(old, new, 1)


def retouch_capability_prompt(obj, priorities):
    """Experimental r8: judge synthesized results, not physical disassembly."""
    prompt = grounded_prompt(obj, priorities)
    capability = '''This is DIGITAL PHOTO EDITING, not physical disassembly. Judge whether the finished photograph can look ordinary and coherent:
- Remove means erase the named target AND naturally reconstruct the background/surface it concealed. A removed box/tree does not leave a void; a removed button/logo can leave continuous cloth. A background bush may be removed for a cleaner composition. Do not skip removal merely because inpainting is required.
- Replace means synthesize another compatible design in place. It need not be a detachable real-world component, and the original need not be broken. A sound round button can become square; an ordinary window can receive a different pane layout. "No replacement is needed" or "recolouring is simpler" is not a valid fallback reason.
- Add may use an open surface as support: a buoy floats on water, a small droplet sits on a real leaf, a decal lies on a panel. No empty mounting socket is required. Preserve plausible scale and contacts.
The anatomy protection rule concerns attached HUMAN/ANIMAL anatomy, such as an arm or a hand. A plant part is not forbidden by that rule; judge its resulting appearance separately. Still never leave people maimed, a garment nonsensically broken, or retained objects unsupported. Background reconstruction does not authorize changes to neighbouring unselected objects.
Choose the FIRST type that admits an ordinary coherent result, not whichever is the easiest or most conservative. A natural alternative is sufficient; it need not be necessary maintenance. After choosing, keep only clauses consistent with it: removing an entire boat cannot also preserve its hull. For add, site is only a bare sub-location such as "center", never "center of the forearm" when the reference already names the forearm. For replace, replacement is one complete noun phrase including the new physical form; form_change records evidence internally and is not appended.\n\n'''
    return prompt.replace('EDIT AND TYPE\n', 'EDIT AND TYPE\n' + capability, 1)


def natural_priority_prompt(obj, priorities):
    """Experimental r7: positive edit choices and bounded reference checks."""
    prompt = grounded_prompt(obj, priorities)
    references = '''Before finalizing the reference, check its truth in the full photograph. A qualifier must both be true of this target and exclude its nearest rival. Do not name unreadable shop signs or borrow a distinctive notch from the target to describe its neighbour. For a short row, prefer leftmost/rightmost/middle when applicable; count within one visible surface, not across a corner onto another surface. Two matching windows on the same building require a left/right relation to one another or a unique landmark, not just the building's roof colour. Use only the minimum verified qualifiers.\n\n'''
    choices = '''The priority order is a real task requirement. Do not default to recolouring if an earlier type has a natural solution. Consider one ordinary concrete edit for that type before falling back; do this internally within this response, not as extra output. Positive possibilities, ONLY when compatible with the actual target:
- add: a small accessory on a person or product; a price tag on a shoe; a badge on clothing; lettering on a plain panel; a small bird on a branch; a subtle water droplet on a real plant. Specify scale and host placement, leaving the host intact.
- replace: a round fastening with a differently shaped compatible fastening; a shoe sole with a different tread/profile; a window with another transparent frame/pane layout; furniture or a vessel with a plausible same-function design. Printed surfaces are not physical objects to replace.
- remove: litter, an unnecessary cord, a background object, a logo/print, or a detachable ornament. Leaves may be removed when the remaining plant remains plausible. Anatomical parts remain protected.
- attribute: colour, visible finish, existing printed design or exact text. This is a valid fallback, but not a substitute for considering earlier natural types.
These are examples, not mandatory edits. Never force an implausible edit to meet a quota. New object/part geometry may differ from the source mask silhouette; the mask identifies the target/anchor, while the downstream execution support is recorded separately. A request still must preserve unselected objects.
For an add site, omit it if the target reference already names that exact part; avoid phrases like "the forearm of the forearm".\n\n'''
    prompt = prompt.replace('EDIT AND TYPE\n', references + 'EDIT AND TYPE\n', 1)
    prompt = prompt.replace('Only use no_edit if', choices + 'Only use no_edit if', 1)
    return prompt


def concise_prompt(obj, priorities):
    """Experimental complete-reference schema: fewer fields and no concatenation."""
    return '''Create ONE useful, natural photo-edit request for the target shown in two images: the full scene and a larger annotated crop. The crop marks exactly one mask. Identify its contents, then find the same instance in the full scene. Unselected objects and holes inside the outline are excluded.

Return a concise complete reference phrase that uniquely identifies this target in the UNMARKED full scene. Use its true visible identity plus a precise relation to a recognizable neighbor. If there are similar adjacent petals, leaves, buttons or cables, the phrase must distinguish the nearest one. A broad direction is not enough. You may count within a clearly defined visible group, stating the starting landmark and direction; never invent a count. The top of a crop is not the top of the original scene. Avoid uncertain anatomical left/right, hidden functions and guessed object parts. Do not mention annotations, pink, mask, crop, IDs or coordinates as locators. Smallness, blur, occlusion or an imperfect outline alone do not make the target unsuitable.

Choose the FIRST natural type in the supplied priority list, with any necessary fallback in this SAME response. Make an edit people would reasonably request for this object and scene. Preserve all unrequested objects. A clearly identified subpart inside the region is allowed; it need not fill the entire mask.
- add: add a plausible object/accessory/printed detail to this host. Keep the host. item includes its article/count. relation and optional site place it on the host. site is a bare part name, empty if the reference already names that part. Avoid placing a watch on a hand or real droplets on a printed flower.
- replace: choose a compatible object/component/design with a visible shape difference, preserving function, contacts and unselected contents. Supply a noun phrase replacement and an explicit design. Color/material alone is attribute. Avoid arbitrary changes to one living petal; restoring a visibly damaged part is reasonable.
- remove: remove the target, revealing a plausible background. Never remove attached anatomy or supports of retained objects. Erasing printed motifs preserves the fabric. If removal is unnatural, choose another allowed type.
- attribute: one visible, natural change of color, material, finish, texture, pattern, text, tint or opacity. Preserve shape and unrelated details. Do not request the current value. Tint is color while preserving transparency; opacity is transparent/translucent/opaque. Natural local skin retouching or a small tattoo (add) can be valid; deleting an arm is not.

Output JSON only. full_ref starts with "the" and contains the complete target phrase once, without an edit verb. scope is a short noun without article, position or color. No alternate drafts or rationale. no_edit is only for inability to produce an unambiguous reference or ANY natural allowed edit, not a poor-mask verdict.
{"full_ref":"the ...","scope":"...","kind":"object|part|surface|printed_motif|anatomy","source_color":null,"op":"add|replace|remove|attribute|no_edit","change":{...}}
change: add={"item":"a ...","relation":"on|in|around|along|at|attached to|sewn onto","site":""}; replace={"replacement":"a ...","design":"..."}; remove={}; attribute={"property":"...","value":"..."}; no_edit={"reason":"specific instruction limitation"}.
An optional keep clause starts with "preserving" and only protects essential unselected contents. Before answering, check that the final full_ref names the marked pixels, excludes the closest similar instance, and that the requested change is natural and local.
''' + INSTRUCTION_STYLE + '\nTask: '+json.dumps({'allowed_types_in_priority_order':priorities,'target_in_full_scene':obj.metadata.get('full_scene_geometry'),'coordinates_are_context_only_not_reference_text':True})


def expand_compact(raw, object_id):
    """Deterministic representation conversion, not a semantic repair or review."""
    if not isinstance(raw,dict) or raw.get('_parse_error'):
        return raw
    op=raw.get('op');change=raw.get('change')
    if not isinstance(change,dict):return {'_parse_error':'missing change object','compact_output':raw}
    if op=='no_edit':return {'feasible':False,'evidence':change.get('reason','unspecified instruction limitation'),'compact_output':raw}
    full_ref=raw.get('full_ref')
    core=raw.get('ref_core');distinction=raw.get('distinguish')
    if full_ref is not None:
        if not isinstance(full_ref,str) or not full_ref.strip():
            return {'_parse_error':'complete reference required','compact_output':raw}
        core=full_ref.strip().rstrip(' .');distinction=''
    elif not isinstance(core,str) or not isinstance(distinction,str) or not core.strip():
        return {'_parse_error':'reference core and distinction required','compact_output':raw}
    scope=raw.get('scope');kind=raw.get('kind')
    if kind not in {'object','part','surface','printed_motif','anatomy'}:
        return {'_parse_error':'invalid target kind','compact_output':raw}
    core=re.sub(r'^(?:a|an)\s+','the ',core.strip().rstrip(' .'),flags=re.I)
    distinction=distinction.strip().rstrip(' .')
    # A noun-headed apposition needs punctuation; relational clauses do not.
    ref=(core+' ('+distinction+')' if re.match(r'^(?:the|a|an)\s',distinction,re.I) else core+' '+distinction).strip()
    result=dict(feasible=True,object_id=object_id,category=scope,scope=scope,
        ref_phrase=ref,source_color=raw.get('source_color'),target_kind=kind,
        operation=op,payload='',attributes=[],placement=None,target_part='',remove_mode='whole',
        qualifier=raw.get('keep',''),selected_pixels=core,unselected_neighbors=raw.get('rival'),
        compact_output=raw,removal_is_natural=kind!='anatomy',
        removal_preserves_unselected=True)
    if op=='add':
        result['payload']=change.get('item','')
        site=change.get('site','')
        # Remove only literal host repetition. This does not infer a location
        # or repair a wrong referent; keep the original compact_output intact.
        if isinstance(site,str):
            # The renderer already supplies the selected host after "of".
            # A leading "its" would otherwise become "its surface of the
            # petal". Strip only this grammatical host pronoun, not anatomy,
            # position, or references to another object; retain raw output.
            site=re.sub(r'^its\s+', '', site.strip(), flags=re.I)
            bare=lambda s: re.sub(r'^(?:the|a|an)\s+','',s.strip(),flags=re.I)
            hosts=[bare(s) for s in [scope,core] if isinstance(s,str) and s.strip()]
            if bare(site).casefold() in {h.casefold() for h in hosts}:
                site=''
            else:
                for host in sorted(hosts,key=len,reverse=True):
                    shortened=re.sub(r'\s+of\s+(?:(?:the|a|an)\s+)?'+re.escape(host)+r'$', '', site, flags=re.I)
                    if shortened!=site:
                        site=shortened
                        break
        result['placement']={'relation':change.get('relation','on'),'site':site,'anchor_mode':change.get('anchor_mode', 'reference' if change.get('relation') in {'beside','near','above','below','under'} else 'support')}
    elif op=='replace':
        replacement=change.get('replacement','')
        if 'form_change' in change:
            # New contract: the public noun phrase is already complete.
            # Evidence is retained in compact_output, never appended to it.
            evidence=change['form_change']
            result['payload']=replacement
            result['replacement_changes_form']=isinstance(evidence,str) and bool(evidence.strip())
        else:
            # Preserve historical split-format decoding and its semantics.
            design=change.get('design','')
            result['payload']=replacement+' with '+design if replacement and design else ''
            if replacement and design.casefold().startswith((replacement+' with ').casefold()):
                result['payload']=design
            result['replacement_changes_form']=bool(design.strip())
    elif op=='attribute':
        result['attributes']=[{'property':change.get('property',''),'value':change.get('value','')}]
    if op=='remove' and kind=='printed_motif' and not result['qualifier']:
        # Motifs may be printed on cardboard, paint, glass, or cloth. The
        # structured kind does not identify the substrate's material.
        result['qualifier']='preserving the underlying surface'
    return result
