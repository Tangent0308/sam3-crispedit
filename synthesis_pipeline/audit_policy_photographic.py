"""Opt-in audit prompts derived from the v11 expansion's blind image reviews.

These prompts are not the frozen v4 policy. Compare them on a named development
run and then on disjoint new sources; never relabel the v4 measurements.
"""

PHOTOGRAPHIC_QUALITY = """Judge the actual local change for a photographic image-edit dataset. No requested instruction is available; do NOT invent one to excuse a defect.
Read BOTH images at matching coordinates. BEFORE's black/white contour is an annotation, not source color. AFTER is entirely clean: any line or halo there is output content. In full/detail panels, both views depict the same scene; use full views for distance and neighboring instances, details for seams.

First establish a useful semantic change: identify what was there BEFORE, what remains, and what is genuinely new AFTER. An exposed background item already present is not an insertion. Small incidental pixel redraw without a clearly identifiable object/property change is a no-op and fails. Removing the annotation, apparent boundary cleanup, tiny sharpening or generic shape refinement alone is NOT a useful edit. Verify the same feature in the clean full BEFORE view rather than hallucinating a change hidden by its detail outline.

Inspect the following evidence silently, then give a short concrete reason:
1. Footprint and completeness. Follow the ENTIRE old target, including legs, handles, thin edges, interior holes and held objects. Look for leftover body parts, ghost silhouettes, doubled edges, hard rectangular cuts, holes through new geometry, and newly damaged neighbors. Do not stop at recognizing the new central object.
2. Physical integration. Check actual contact/attachment, perspective, and scale RELATIVE TO objects at the same distance in the full scene. Trace supports and thin connecting parts. A new item must not float beside its intended support or intersect a solid closed surface. Ordinary occlusion can explain a hidden connection, but cannot explain a plainly disconnected segment.
3. Photographic integration. Compare the changed object with its immediate host and same-depth surroundings: depth of field, directional lighting, color cast, grain and shading. A razor-sharp insertion on a clearly defocused host, an unlit cutout in strongly colored light, or a flat fill that destroys pre-existing folds/texture is a defect. Native text, printed graphics, pins and deliberately flat surfaces can naturally be flat. A blurred/redacted FACE does not imply the person's sharp clothing must also be blurred. Saturation alone is not a defect when shading and texture remain.
4. Compare suspected defects back to BEFORE. Pre-existing blur, perspective, occlusion and compression are not new errors. Do not demand invisible bases, shadows or fine details beyond the image resolution. An intact partial recoloring can be usable; residual fragments after destructive removal cannot be reinterpreted as a clever partial edit.

PASS: an obvious useful local change and a broadly natural result, allowing minor rendering differences. FAIL: a definite new physical/photographic defect, recognizable target remnants, damaged neighbors, or no clear useful change. Do not fail just because a material is unusual or a hidden attachment is not visible. Do not pass just because some imaginable instruction could describe a bad picture.
Only JSON with three fields:
{"observed_change":"actual before -> after difference, at most 30 words", "reason":"decisive visible evidence with location, at most 70 words", "quality":"pass|fail"}.
"""

INSTANCE_RECONSTRUCTION = """
Before finalizing the command, mentally hide ALL outlines and search the complete BEFORE photograph for every object matching the proposed noun phrase. If more than one fits, include a stable neighboring landmark or a within-group ordinal/row. Broad left/right is not sufficient when several matching instances remain on that side. Do not use the newly generated appearance to identify the source. For addition, identify the actual host surface/body part and contact location, not just the nearest large object. A changed location on the SAME host can be described accurately; an insertion by an unrelated instance cannot be reassigned to the outlined host. Prefer plain visible object names over guessed categories or intricate material descriptions.
"""

PHOTOGRAPHIC_VERIFICATION = """
Also verify optical integration (local focus, light/color cast, grain), scale at the same scene depth, complete physical connections, and absence of unchanged target fragments. A sharp printed symbol can be normal on a sharp sign; a new razor-sharp ornament on a defocused host is not. Do not infer scene depth from privacy-blurred faces alone.
Hide the outline and test the instruction against ALL same-category objects in the full BEFORE photo. If multiple objects satisfy its wording, candidate_match=fail even if the crop makes the intended one obvious. For add, check where the NEW item really touches or belongs: the outlined host/part must be its actual anchor, not merely nearby. A different host is wrong scope. A subpart label on a whole-object mask requires subset, not same. An oversized/floating/poorly integrated insertion cannot be rescued by changing the wording to say it is large or hovering.
"""

# Short, defect-first alternative. Kept separate for reproducible ablations.
CRITICAL_QUALITY = """You are reviewing a localized photographic edit, not an illustration or a screen overlay. Decide from the pixels whether this pair is usable training data. No original instruction is given: changed identity, color or placement may be relabeled later, but bad pixels cannot be excused by relabeling.
The BEFORE outline is annotation only. AFTER has NO annotation. First inspect AFTER independently as a photograph, then compare the same positions in BEFORE. Use the clean full views to disambiguate anything covered by the outline.
Check in this order:
1. Trace the ENTIRE former object footprint, its interior holes, thin parts, shadows and held items. Is something left disconnected, chopped off, ghosted or unnaturally filled? Inspect the area the old object occupied, not only the convincing new center. A leftover silhouette filled with a uniform color is not a plausible new object or surface finish.
2. Inspect connections and borders literally. Where does each added object touch its actual host? A print must stay ON the visible host surface, not extend into empty space. A rod cannot pass through a solid closed surface. Judge size relative to objects at the same depth. Do not invent hidden supports for visibly disconnected parts.
3. Compare fine structure at the changed area: do fabric folds, material shading, grain, focus and scene color cast remain photographically plausible? A region that becomes a featureless paint bucket is a failure even if its silhouette is neat. A sharp icon on a blurred real-world surface is NOT acceptable as a deliberate digital overlay. Naturally flat signs can be flat, but must share the photographed surface's focus, lighting and perspective.
4. Verify a real semantic difference at the target. Removing the drawn annotation or merely sharpening an existing edge is not an edit. Compare suspected defects with BEFORE: old occlusions, compression and source defects are not new failures. Do not reject small harmless rendering differences or ordinary hidden contact.
PASS only when there is a clear useful change and no conspicuous new defect. FAIL for a visible remnant, ghost fill, broken physical connection, optical mismatch, lost three-dimensional texture, damaged neighbor or no meaningful change. Do not rationalize defects into an imaginative successful edit.
Return only {"observed_change":"concrete before -> after change, at most 30 words", "reason":"specific visible evidence, including location of any decisive defect, at most 70 words", "quality":"pass|fail"}.
"""

CRITICAL_VERIFY = """Check this proposed training label against the two photographs. The label is a hypothesis, not evidence.
Type: {candidate_type}
Instruction: {candidate}
Hide the outline mentally. Can someone locate exactly ONE target (or the explicitly named set) using this sentence in the FULL BEFORE picture? Compare every matching instance; left/right alone may not suffice. Check the action, identity, count, property, actual contact location and ALL substantial changes. No-op or a label that ignores another insertion/deletion fails candidate_match. Small wording differences are fine.
{type_checks}
Now inspect AFTER independently: reject clear new remnants, fill seams, chopped anatomy, impossible connections, lost shape/shading, pasted graphics or focus/light mismatch. Compare BEFORE to avoid calling old defects new ones. Never use a clever description to excuse bad image quality; do not demand perfection or invisible details.
Scope: same = the labeled target/host is the same marked object or already-marked part; subset = the label describes only an intact part of the marked object; wrong = a different instance/host; unclear = cannot establish. An intact partial recoloring can be good quality, but needs subset when its label is narrower than the mask. Target remnants are defects, not an intentional subset. For ADD specifically, the mask defines the HOST: naming a precise placement on that same host is still same, not subset. The marked object must really support/host the new item, not merely stand nearby; another host or placement outside the marked host/part is wrong.
Return only JSON: {{"reason":"specific visual evidence for quality, label and scope; at most 100 words", "quality":"pass|fail", "candidate_match":"pass|fail", "candidate_scope":"same|subset|wrong|unclear"}}.
"""

TYPE_CHECKS = {
    'add': 'Addition: locate the new item in AFTER and verify it did not exist in BEFORE. Trace its contact with the stated host. Printed marks must stay on that surface and share its focus/perspective; a sharp screen overlay on a blurred host is not a physical addition. The original host and other objects remain coherent.',
    'remove': 'Removal: compare the entire old object footprint, including thin parts, holes, its shadow/reflection and held objects. The named object must be gone, not merely recolored into a ghost. Reject recognizable leftover pieces or newly floating dependents. The exposed background must join its surroundings naturally; pre-existing background objects are not new insertions.',
    'replace': 'Replacement: verify that the original identity is gone and a genuinely different complete object occupies that location. Trace the old footprint outside the new silhouette, including legs, handles, shadows and reflections: old fragments do not become acceptable because the center looks convincing. Check the new object is physically connected, correctly scaled and photographically integrated.',
    'attribute': 'Attribute: verify the same instance remains and only the described property/scope changes. Inspect whether folds, material shading, surface detail and geometry remain plausible: uniform paint over a three-dimensional object is a quality failure. Unmentioned erased text/pattern, deleted parts or added items are substantial extra changes, not just a color change.',
}


TOPOLOGY_CHECKS = """
Before accepting an interaction, trace its actual support in AFTER. The words 'holding', 'attached', 'wearing' or 'sitting on' are conclusions that require visible geometry, not explanations that make it valid. For an articulated body, follow each visible appendage through its joint to its endpoint and check that the count and connections are anatomically possible. Two-dimensional overlap with the torso is not a grasp. For a surface-mounted item, trace its whole outline, holes and separated pieces: do they form one coherent intended physical object, or disconnected fragments? Distinguish a real occluded connection from a clearly impossible or duplicated one by comparing BEFORE. Do not invent an extra limb, fastening or structural part just to explain a leftover fragment. Do not penalize normal occlusion or demand invisible detail; reject only a concrete contradictory connection or broken form you can locate.
"""


def critical_verification_prompt(candidate_type, candidate):
    return CRITICAL_VERIFY.format(candidate_type=candidate_type, candidate=candidate,
        type_checks=TYPE_CHECKS.get(candidate_type, 'No valid edit type: candidate_match must fail.'))


GROUNDED_RECONSTRUCTION = """Describe the actual local edit, not an intended request. Compare the clean FULL BEFORE/AFTER photographs and their matching details. The outline only identifies the original target/host.
First establish the visible before -> after inventory and geometry. A complete new foreground object occupying the old object's place is a replacement unless the SAME object is visibly identifiable in BEFORE. Do not invent a previously hidden object to justify calling replacement removal. Conversely, if the old object remains and a new item attaches to it, this is addition, not replacement of the old object. Compare silhouette, thickness and components before calling a change merely color.
Choose the shortest accurate instruction that covers ALL substantial differences. Inspect which parts DID NOT change: if some remain the original color, name only the changed parts. If text or a structural component also disappears, a color-only instruction is incomplete. Never describe removal of unchanged neighbors or pretend an unsupported new object is attached to a nearby surface.
Identify the target in the original FULL photograph without an outline. Search all matching instances, including separate panels in a collage. Use a stable neighbor or group position when broad left/right is ambiguous. Do not identify a source using its new appearance. For add, identify its actual original host; do not relabel an edit to another instance to evade the mask.
Return one imperative English sentence, at most 25 words, no preservation clauses, speculative detail or fixed example objects. A physically bad edit cannot be rescued by description; return null if no clean useful instruction exists.
Only JSON: {"task_type":"add|remove|replace|attribute or null", "instruction":"command or null"}.
"""


def grounded_verification_prompt(candidate_type, candidate):
    return """Independently compare the pixels BEFORE assessing the proposed sentence below. It is only a hypothesis.
In your reason, first state the actual original target and its visible AFTER state. Check its entire silhouette, thickness, components, color, text and surroundings. Do not let the proposed verb decide the observed operation. A newly visible complete foreground object replacing the target is not automatically an old hidden background object: require visible correspondence in BEFORE. Mere shared broad category does not make a changed shape/identity a recoloring. For partial recoloring, check unchanged extremities rather than generalizing from the center.
Then check whether the sentence covers ALL substantial changes and uniquely selects the source in the FULL scene, including all panels in a collage. Extra removals, missing insertions, false contact, or ambiguous targets fail candidate_match. Paraphrases and harmless detail differences are fine.
Physical quality is independent of wording. Compare the complete old footprint, including shadows and fine extensions. Reject clear new remnants, ghost fill, impossible connections, lost three-dimensional shading, or local focus/light mismatch; do not reject old defects or minor rendering variation. AFTER is clean, with no annotation to ignore.
Scope: same when this is the marked target/host; subset when only an intact subpart of the marked object is described; wrong for another instance/host; unclear if not establishable. For ADD, a precise site on the SAME host is same, not subset, but proximity alone does not prove attachment. No label can rescue wrong-host edits with the original mask.
""" + TYPE_CHECKS.get(candidate_type,'Invalid type: candidate_match must fail.') + f"""
Now test the proposed label ({candidate_type}): {candidate}
Only JSON: {{"reason":"visible before -> after inventory FIRST, then decisive quality/label/scope evidence, at most 100 words", "quality":"pass|fail", "candidate_match":"pass|fail", "candidate_scope":"same|subset|wrong|unclear"}}.
"""


def selective_verification_prompt(original_type, original, candidate_type, candidate):
    return """Choose a correct training instruction by independently comparing the actual photographs. Neither proposed label is evidence. First establish the actual before -> after change, complete old footprint, all changed parts, and the original mask target/host.
Judge photographic quality separately: new ghost fragments, visibly disconnected parts, obvious fill/texture collapse or optical mismatch fail quality. A natural edit that differs from a proposed request is NOT a quality failure. Do not confuse untouched extremities of an intact partial recoloring with remnants after object removal. Old source defects do not count.
Check BOTH labels for truthful action/identity/color/count, complete coverage of substantial changes, and unambiguous source localization in the FULL photograph without an outline. A new foreground object cannot be called exposed old background without visible correspondence in BEFORE. A changed silhouette/thickness is not necessarily just a recoloring. Check unchanged parts before accepting a whole-object attribute command. Labels must not request removal of neighbors still present. In collages, locate the correct panel as well as the person/object. Minor paraphrases and harmless details are acceptable.
Choose ORIGINAL if it is fully accurate; do not replace a correct label just to improve its wording. Choose REWRITTEN only if the original is inaccurate and the rewrite actually fixes ALL errors without adding false claims. Choose NONE if neither is accurate or localized enough. An attractive picture alone cannot justify a false instruction.
Scope concerns the CHOSEN label: same = the same marked target/host; subset = only an intact smaller part of that target; wrong = a different instance/host; unclear = uncertain or no valid choice. ADD's mask defines its host, so precise placement on that same host is same. An item merely near a host is not attached to it. Wrong-host changes cannot be rescued by changing the target noun.
""" + f"""
ORIGINAL ({original_type}): {original}
REWRITTEN ({candidate_type}): {candidate}
Return only four fields: {{"reason":"actual visible change first, then separate quality and label-choice evidence, at most 100 words", "quality":"pass|fail", "label_choice":"original|rewritten|none", "chosen_scope":"same|subset|wrong|unclear"}}.
"""


def observed_verification_prompt(candidate_type, candidate, observation):
    """Challenge reconstruction using an instruction-blind reading, not its verdict.

    The observation can be wrong too: it is never a hard rule or another vote.
    No original generation request or earlier pass/fail judgment is supplied.
    """
    prompt=grounded_verification_prompt(candidate_type,candidate)
    if not observation:
        return prompt
    challenge=("A separate instruction-blind reading suggested the following possible pixel change: "
        + str(observation)
        + "\nThis reading may itself be wrong. Resolve any disagreement with the candidate by "
        "tracking the SAME original object and its entire footprint in both photographs. "
        "Neither text is evidence. Do not ignore a visible deletion while describing a smaller "
        "addition, or call a visibly pre-existing neighbor newly inserted. In your existing "
        "reason field, explain a substantive disagreement if there is one; verify from pixels "
        "rather than taking a majority vote.\n")
    return prompt.replace('\nNow test the proposed label', '\n'+challenge+'Now test the proposed label',1)


def observed_reconstruction_prompt(observation):
    if not observation:
        return GROUNDED_RECONSTRUCTION
    return GROUNDED_RECONSTRUCTION + (
        '\nAn earlier reading of these pixels, made WITHOUT the requested instruction, suggested: '
        + str(observation)
        + '\nThis is not a requested edit and may be inaccurate. Check it against both photos. '
        'Track the original target and already-visible neighbors before choosing the action. '
        'Use it only to detect a missed substantial change; correct it when the pixels disagree. '
        'Do not discard a visible deletion just to describe newly exposed hands or background. '
        'Still output only task_type and instruction, with the same 25-word limit.\n')
