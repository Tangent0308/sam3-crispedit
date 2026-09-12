"""ScaleEdit task-aware prompting and strict grounding-output parsing.

ScaleEdit category names describe dataset provenance more than mask geometry.
The paired images therefore decide whether a sample uses a full-image mask,
an inverse foreground mask, or a union of source/target local regions.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Sequence

from scaleedit import PROMPT_VERSION


SUPPORTED_TASKS = {
    "action_editing",
    "background_replacement",
    "building_surface_text_editing",
    "color_change",
    "compositional_editing",
    "count_change",
    "gui_interface_text_editing",
    "material_change",
    "movie_poster_text_editing",
    "object_addition",
    "object_removal",
    "object_replacement",
    "object_surface_text_editing",
    "part_extraction",
    "perceptual_reasoning",
    "scientific_reasoning",
    "size_change",
    "social_reasoning",
    "style_transfer",
    "symbolic_reasoning",
    "tone_adjustment",
    "viewpoint_transformation",
    "visual_beautification",
}

TEXT_TASKS = {
    "building_surface_text_editing",
    "gui_interface_text_editing",
    "movie_poster_text_editing",
    "object_surface_text_editing",
}

MASK_MODES = {"regions", "protect_foreground", "full_image"}
MASK_METHODS = {"sam", "box"}
REGION_MODES = {"object", "aggregate_region", "multi_instance"}
MASK_DENSITIES = {"object", "dense", "sparse"}
SELECTION_MODES = {"single", "all_matching", "compact_region"}
MASK_EXTENTS = {"whole_object", "whole_actor", "subpart", "surface_region"}
MAX_LOCALIZATION_ITEMS = 8
MAX_DETECTIONS_PER_CANDIDATE = 64
BBOX_LOCALIZATION_PROMPT_VERSION = "scaleedit_qwen_native_grounding_locator_v9_correspondence"


_COUNT_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_HUMAN_WORDS = (
    "boy|girl|man|woman|person|people|child|children|kid|adult|worker|player|"
    "student|teacher|chef|doctor|soldier|dancer|athlete"
)
_BODY_PART_WORDS = {
    "arm",
    "hand",
    "face",
    "head",
    "leg",
    "foot",
    "feet",
    "torso",
    "body",
    "mouth",
    "eye",
    "eyes",
    "shoulder",
    "waist",
}


def canonical_task(value: object) -> str:
    task = re.sub(r"[\s-]+", "_", str(value or "").strip().lower())
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"unknown ScaleEdit final_task: {value!r}")
    return task


def _task_guidance(task: str) -> str:
    if task in TEXT_TASKS:
        return (
            "This is a text edit. Treat the exact old/new glyph block as the edited region, not "
            "the whole sign, screen, poster, seal, or building. Use box masks because generic "
            "semantic segmentation is unreliable for glyphs. Include source and target text boxes. "
            "Visually verify the exact quoted old text in Image 1 and exact quoted new text in "
            "Image 2 before writing spatial_hint; a nearby word, line, or repeated carrier label "
            "is not a substitute for the requested glyph block."
        )
    guidance = {
        "background_replacement": (
            "Audit stable foreground entities. Usually protect those entities and invert their "
            "mask; use full_image only if no independent foreground survives."
        ),
        "part_extraction": (
            "Distinguish product-style extraction/recomposition from a local reveal. Product "
            "extraction normally recenters, rescales, or reconstructs the retained subject while "
            "replacing everything else, so use full_image. Use regions only for an in-place local "
            "operation such as removing a shell/cover to reveal an interior."
        ),
        "style_transfer": (
            "Do not assume style means full image. A wall, sign, facade, display, or phone-screen "
            "style change is local; only an explicit whole-image transformation is full_image. "
            "When many adjacent wall/pillar panels share one style edit, describe the complete "
            "architectural surface as one dense aggregate per image side instead of enumerating panels."
        ),
        "tone_adjustment": (
            "A whole-image filter, grayscale conversion, or relighting of the entire visible scene "
            "is full_image. Use regions only when the change is confined to a separable area such "
            "as the sky, background behind a stable subject, or one reflection."
        ),
        "viewpoint_transformation": (
            "A camera viewpoint/composition change is full_image. Rotation/view change of one "
            "isolated object against a stable background uses source+target object regions. "
            "A request for the front/rear/side view of a named object is object rotation, not a "
            "camera transformation, even if the generated object is recentered or rescaled."
        ),
        "compositional_editing": (
            "Decompose every realized addition, removal, replacement, and recolor; do not merge "
            "far-apart or semantically different regions. A shelf, stand, table, wall, or ground "
            "does not become edited source content merely because an added object is placed on it; "
            "a pure addition sub-edit must keep source_ref empty."
        ),
        "count_change": (
            "Ground only instances that appear or disappear, not unchanged comparison instances. "
            "Return one object item per changed instance. For each bbox, verify the top, bottom, "
            "left, and right visible contour; center the box on that instance and exclude the "
            "table, shelf, ground, shadow, and unchanged neighboring instances."
        ),
        "symbolic_reasoning": (
            "Ground the edited cells, symbols, lines, or path. Use box masks for glyphs, thin "
            "drawn lines, and tiny marks so that the label cannot lose them. For a maze/path, "
            "the target bbox must span the complete endpoint-to-endpoint route, including every "
            "turn; a box around only the middle segment is invalid."
        ),
        "perceptual_reasoning": (
            "Ground the complete repair/damage footprint visible in the pair, even when it is a "
            "sub-part such as an edge, crack, handle, tail tip, or missing cover."
        ),
        "scientific_reasoning": (
            "The category does not determine geometry. Decompose the actual paired-image changes; "
            "small dots/irritation use aggregate box regions, while environment-wide changes may "
            "use full_image only when they truly cover the entire scene."
        ),
        "social_reasoning": (
            "The category does not determine geometry. Resolve the actual action into local, "
            "background, or global changes using the image pair. When material is transferred "
            "between two locations, list only the portion removed at the origin and the portion "
            "added at the destination; do not mark retained material as newly added or removed."
        ),
        "visual_beautification": (
            "Face/skin restoration is local to the complete affected faces or people; object "
            "repair uses the damaged object's footprint; do not default to full image."
        ),
        "action_editing": (
            "Use source+target regions for the moved/rotated actor, object, body part, expression, "
            "drawer, or interaction object. If a person's action changes the pose, silhouette, scale, "
            "or placement of their body beyond one isolated limb, localize the complete person on "
            "both sides with mask_extent=whole_actor. Use subpart only when the rest of the actor is "
            "visibly unchanged. Avoid unrelated reconstruction drift. If the edited footprint "
            "is an empty cutout used as the rotation cue (for example a cookie bite moving around "
            "the cookie edge), label that cutout as negative_space and name its carrier object."
        ),
        "size_change": (
            "Use both the old and resized footprints of only the resized instance."
        ),
        "color_change": (
            "For an appearance-only recolor, localize the affected source content only because the "
            "final mask is in source coordinates; inspect the target to verify the realized color but "
            "do not create target items unless the intended footprint also moves or changes shape. "
            "Use the complete recolored object or explicitly recolored subpart. Plural or all-matching "
            "objects use one multi_instance query per semantic/location group, which may return many "
            "member boxes. A named facade or surface made uniform must include every source section "
            "whose original color changes, using mask_extent=surface_region. A large coherent adjacent "
            "section with a substantial paired-image appearance change is part of the realized edit "
            "footprint even when its source color, resulting shade, or architectural boundary was not "
            "named exactly; only small texture and reconstruction noise is unrelated drift. Liquid "
            "inside a glass is a dense subpart; "
            "exclude the glass, rim, stem, hands, and background."
        ),
        "material_change": (
            "For an appearance-only material edit, localize the affected source surface/object only "
            "because the final mask is in source coordinates; add target items only if the intended "
            "footprint moves or changes shape. Include the complete affected source surface, including "
            "thin structures such as rails and winding paths."
        ),
        "object_addition": "Use target regions for genuinely added content; do not box pre-existing peers.",
        "object_removal": "Use source regions for removed content; never box empty target space.",
        "object_replacement": (
            "Use source regions for the old content and target regions for the replacement."
        ),
    }
    return guidance.get(task, "Infer the complete realized edit footprint from the paired images.")


def build_observation_prompt(final_task: object, instruction: object) -> str:
    task = canonical_task(final_task)
    return f"""You are auditing one ScaleEdit source/result image pair before mask labeling.

Image 1 is the source. Image 2 is the edited result.
Final task: {task}
Corrected edit instruction: {str(instruction or '').strip()}

The images are the source of truth. Ignore resizing/JPEG noise and small unrelated generative drift,
but do not ignore a large coherent change on or immediately adjoining the edited object/surface.
Describe every intentional realized edit, decide the spatial mask route, and produce the FINAL
list of visible entities that a separate bbox-only pass must locate. That later pass will not
reinterpret the edit, add entities, or repair this plan, so make the item list complete and exact.

Task-specific guidance: {_task_guidance(task)}

Mask routes:
- regions: one or more local edits. Record visible source-side old/changed content and target-side
  new/changed content separately. Additions may have no source entity; removals may have no target.
- protect_foreground: a background is replaced/removed while one or more independent foreground
  subjects remain. List every retained subject that the editable background mask must exclude.
- full_image: essentially the whole canvas/composition is intentionally transformed. Do not choose
  this merely because task={task}; local style, lighting, screen, facade, or object edits are regions.

Geometry choices:
- semantic_object: a coherent object/surface that SAM can segment.
- dense_region: an irregular but spatially dense footprint.
- sparse_marks: text, glyphs, dots, cracks, thin lines/paths, tiny repeated marks. These later use a
  conservative filled box so mask recall is not lost.
- negative_space=true only when the intended pixels are an empty hole, gap, opening, cutout, or
  missing bite defined by a foreground object's contour. Set carrier_ref to that concrete foreground
  object. Dark marks, shadows, transparent material, and ordinary pale objects are not negative space.

Planning rules:
- regions: emit one localization item for every source-side old/changed entity and every target-side
  new/changed entity, except appearance-only color/material edits use source items only. Pure
  additions emit target items only; pure removals emit source items only.
- A stand, shelf, table, wall, floor, ground, platform, shadow, or empty space is not a source edit
  merely because an added object is placed there. Put that information only in spatial_hint.
- A transfer or move from one location to another uses remove items for content disappearing at the
  origin and add items for content appearing at the destination. Do not mechanically list the
  unchanged/retained content at both locations as edited entities.
- HARD LIMIT: localization_items contains at most {MAX_LOCALIZATION_ITEMS} semantic queries. A single
  specifically selected instance uses region_mode=object and selection_mode=single. When same-class
  instances have different explicit roles or positions (for example one left and one right), emit one
  object/single item per position. Use one region_mode=multi_instance, selection_mode=all_matching item
  only for an undifferentiated all/every/plural set sharing one referring expression. Its ref and
  spatial_hint must describe the complete set, never only one member. The bbox pass may return one box
  per visible member with the same candidate ID. Only a spatially continuous compact footprint uses
  region_mode=aggregate_region, selection_mode=compact_region and one union box. Never describe a
  distributed row/crowd as aggregate_region, and never omit members merely to shorten the list.
- For all_matching, set expected_count to the number of clearly visible matching members when it can
  be counted reliably, otherwise null. expected_count is an audit hint and does not cap detections.
- Choose topology from the visible pixels, not grammar alone. Separate countable objects use
  multi_instance/all_matching. Many small touching items or material filling one named container,
  compartment, or bounded patch use dense_region + aggregate_region/compact_region and one union
  box spanning the outermost changed content. Never encode plural contained content as one ordinary
  semantic object: that makes the locator and SAM retain only a few central members.
- mask_extent=whole_actor when an action changes a person's overall pose/silhouette; whole_object for
  a complete object; subpart for liquid, a face, arm, or other explicit part; surface_region for the
  complete affected wall/facade/material footprint. A uniform-color surface includes every original
  non-target-color section of that named surface, even when sections have different apparent colors.
  If several coherent surface sections visibly undergo the requested appearance change, describe all
  of them explicitly; do not discard a major adjacent section as reconstruction drift.
- protect_foreground: emit only complete stable foreground entities, on image_side=source with
  role=protected_foreground and edit_op=protect.
- full_image: localization_items must be empty.
- mask_method=box for exact text/glyph blocks, thin paths/lines/cracks, dots and sparse marks;
  otherwise use sam. Use aggregate_region only for one continuous compact region, and
  multi_instance for separate repeated members.
- For negative space, ref names the empty region, carrier_ref names the foreground object whose
  silhouette defines it, geometry=dense_region, mask_method=sam, region_mode=object, and
  mask_density=dense. The bbox-only pass will locate the empty region rather than the whole carrier.

Return compact JSON only. Do not output coordinates in this pass:
{{"realized_edit":"one precise sentence","mask_mode":"regions|protect_foreground|full_image","localization_items":[{{"image_side":"source|target","role":"edit_region|protected_foreground","edit_op":"add|remove|change|protect","ref":"concrete visible entity or matching set, 2-12 words","spatial_hint":"concise location or set boundary","geometry":"semantic_object|dense_region|sparse_marks","mask_method":"sam|box","region_mode":"object|aggregate_region|multi_instance","selection_mode":"single|compact_region|all_matching","mask_extent":"whole_object|whole_actor|subpart|surface_region","expected_count":null,"mask_density":"object|dense|sparse","negative_space":false,"carrier_ref":"empty unless negative_space is true"}}],"confidence":"high|medium|low"}}
"""


def localization_requires_image_pair(observation: Dict[str, Any]) -> bool:
    """Whether bbox grounding needs source/target comparison in one call."""

    items = list(observation.get("localization_items", []))
    sides = {str(item.get("image_side")) for item in items}
    if len(sides & {"source", "target"}) > 1:
        return True
    # Surface completion is the one single-side route whose referring
    # expression is defined by a paired-image appearance change. Ordinary
    # object/text grounding follows Qwen's preferred single-image setup.
    return any(str(item.get("mask_extent")) == "surface_region" for item in items)


def build_grounding_prompt(
    final_task: object, instruction: object, observation: Dict[str, Any] | str
) -> str:
    """Build a compact Qwen-native grounding request from the semantic plan.

    Qwen's official grounding examples use one short locate instruction and a
    JSON output request. Keep only the referring expression, an optional image
    index, and the minimum geometry qualifier that changes what must be boxed.
    Routing and mask metadata remain in the parsed plan and are joined back by
    candidate ID after inference.
    """

    task = canonical_task(final_task)
    del instruction
    if not isinstance(observation, dict):
        raise ValueError("bbox localization requires a parsed edit plan")
    items = [
        item
        for item in observation.get("localization_items", [])
        # Optional appearance-surface completion is derived from aligned pair
        # evidence after grounding the required primary surface. Asking Qwen
        # to guess an unnamed residual creates ambiguous neighboring boxes.
        if not (
            bool(item.get("optional"))
            and str(item.get("mask_extent")) == "surface_region"
        )
    ]
    image_sides = [
        side
        for side in ("source", "target")
        if any(str(item.get("image_side")) == side for item in items)
    ]
    pair_required = localization_requires_image_pair(observation)
    single_image = not pair_required
    realized_edit = " ".join(str(observation.get("realized_edit", "")).split())
    queries = []
    output_slots = []
    for item in items:
        candidate_id = int(item["candidate_id"])
        image_side = str(item["image_side"])
        image_number = 1 if image_side == "source" else 2
        ref = " ".join(str(item["ref"]).split())
        spatial_hint = " ".join(str(item.get("spatial_hint", "")).split())
        geometry = str(item.get("geometry", "semantic_object"))
        region_mode = str(item.get("region_mode", "object"))
        selection_mode = str(item.get("selection_mode", "single"))
        mask_extent = str(item.get("mask_extent", "whole_object"))
        expected_count = item.get("expected_count")
        optional = bool(item.get("optional", False))
        negative_space = bool(item.get("negative_space", False))
        carrier_ref = " ".join(str(item.get("carrier_ref", "")).split())
        location = "" if not spatial_hint else f"; at {spatial_hint}"
        if optional and mask_extent == "surface_region":
            description = (
                f'optional changed surface "{ref}"{location}; omit if absent'
            )
        elif negative_space:
            description = (
                f'complete empty region "{ref}"{location} inside "{carrier_ref}"'
            )
        elif mask_extent == "whole_actor":
            description = f'complete actor "{ref}"{location}, including all visible body parts'
        elif mask_extent == "surface_region":
            description = f'complete changed surface "{ref}"{location}'
        elif geometry == "dense_region" or mask_extent == "subpart":
            description = f'complete named material or subpart "{ref}"{location}'
        elif geometry == "sparse_marks":
            description = f'exact glyphs or marks "{ref}"{location}'
        else:
            description = f'complete object "{ref}"{location}'
        if region_mode == "multi_instance" or selection_mode == "all_matching":
            if expected_count is not None:
                count = int(expected_count)
                context = f'; changed-set context: "{realized_edit}"' if realized_edit else ""
                description = (
                    f'exactly {count} different members of {description}{context}'
                )
                output_slots.extend(
                    (candidate_id, member_index)
                    for member_index in range(count)
                )
            else:
                description = f'every visible instance matching {description}'
                output_slots.append((candidate_id, None))
        elif region_mode == "aggregate_region":
            description = f"one box enclosing {description}"
            output_slots.append((candidate_id, None))
        else:
            output_slots.append((candidate_id, None))
        image_prefix = "" if single_image else f"Image {image_number} | "
        queries.append(f"candidate_id={candidate_id} | {image_prefix}{description}")
    if not queries:
        raise ValueError("bbox localization requires at least one candidate")
    query_text = "\n".join(queries)
    if single_image:
        image_context = (
            "source image" if image_sides[0] == "source" else "edited image"
        )
        header = f"Locate these targets in this {image_context} and report bbox coordinates in JSON format:"
    else:
        header = (
            "Image 1 is the source and Image 2 is the edited result. Locate each target in its "
            "specified image and report bbox coordinates in JSON format:"
        )
    text_rule = ""
    if task in TEXT_TASKS:
        text_rule = (
            " Box only the requested glyphs, not nearby text or the carrier. For a replacement, "
            "locate new text at the position corresponding to the old text; ignore pre-existing "
            "copies elsewhere."
        )
    output_example = ",".join(
        f'{{"bbox_2d":[x1,y1,x2,y2],"candidate_id":{candidate_id}'
        + ("" if member_index is None else f',"member_index":{member_index}')
        + "}"
        for candidate_id, member_index in output_slots
    )
    return f"""{header}
{query_text}

Report one tight bbox per target or per requested member.{text_rule}
Use normalized 0-1000 coordinates. Keep candidate_id numeric and return JSON only:
[{output_example}]
"""


def build_observation_correction_prompt(error: object) -> str:
    """Ask for a complete bounded plan after a syntactically invalid answer."""

    return f"""Your previous edit-plan answer could not be parsed: {str(error)}
Return the complete edit-plan JSON object again, with no prose or markdown. Do not continue the
previous text. It must contain mask_mode and localization_items, and localization_items has a hard
maximum of {MAX_LOCALIZATION_ITEMS} entries. If necessary, merge nearby same-side objects with the
same edit into multi_instance semantic query groups; do not omit changed members. Close every array
and object. Use exactly the schema and rules from the original request."""


def build_bbox_correction_prompt(
    expected_candidate_ids: Sequence[int],
    error: object,
    multi_candidate_ids: Sequence[int] | None = None,
    aggregate_candidate_ids: Sequence[int] | None = None,
    optional_candidate_ids: Sequence[int] | None = None,
) -> str:
    """Correct only the locator's output contract, without replanning the edit."""

    expected = [int(value) for value in expected_candidate_ids]
    multi_ids = [int(value) for value in (multi_candidate_ids or [])]
    aggregate_ids = [int(value) for value in (aggregate_candidate_ids or [])]
    optional_ids = [int(value) for value in (optional_candidate_ids or [])]
    required_ids = [value for value in expected if value not in set(optional_ids)]
    return f"""Your previous visual-grounding answer could not be bound to the requested candidates:
{str(error)}
Locate the same entities from the original request; do not reinterpret them. Return only one
complete Qwen-native JSON array. Every required ID below must occur at least once, in candidate-group
order. Optional IDs may be omitted only when the requested residual entity does not exist:
required={required_ids}; optional={optional_ids}
Multi-instance IDs {multi_ids} may occur multiple times: return one separate box for every visible
member and repeat that candidate_id. Aggregate IDs {aggregate_ids} return one union box enclosing
their continuous compact region. Every other ID returns exactly one box. Every box must have four
non-degenerate normalized 0-1000 coordinates. No prose or markdown."""


def build_object_viewpoint_retry_prompt(
    final_task: object,
    instruction: object,
    observation: Dict[str, Any] | str,
    object_ref: str,
) -> str:
    """Correct a false global route while preserving bbox-only pass separation."""

    del final_task, observation
    return f"""Correct the edit plan for this paired source/target image.

Instruction: {str(instruction or '').strip()}
`{object_ref}` is one isolated transformed object, not a camera or whole-scene transformation.
Return mask_mode=regions with exactly two localization_items: the complete old `{object_ref}` on
image_side=source and the complete new `{object_ref}` on image_side=target. Use edit_op=change,
role=edit_region, mask_method=sam, region_mode=object and mask_density=object. Include a concise
spatial_hint that distinguishes the intended instance. Do not output coordinates; a separate
bbox-only pass will localize the two items.

Return compact JSON only:
{{"realized_edit":"one precise sentence","mask_mode":"regions","localization_items":[{{"image_side":"source","role":"edit_region","edit_op":"change","ref":"{object_ref}","spatial_hint":"source location","geometry":"semantic_object","mask_method":"sam","region_mode":"object","mask_density":"object"}},{{"image_side":"target","role":"edit_region","edit_op":"change","ref":"{object_ref}","spatial_hint":"target location","geometry":"semantic_object","mask_method":"sam","region_mode":"object","mask_density":"object"}}],"confidence":"high|medium|low"}}
"""


def _json_candidates(text: str) -> Iterable[Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        yield value


def _first_json_object(text: str) -> Dict[str, Any]:
    fallback: Dict[str, Any] | None = None
    for value in _json_candidates(text):
        if isinstance(value, dict):
            if "mask_mode" in value:
                return value
            if fallback is None:
                fallback = value
    if fallback is not None:
        raise ValueError(
            "no complete edit-plan JSON object found (output may be truncated)"
        )
    raise ValueError("no edit-plan JSON object found")


def _plan_defaults(geometry: object) -> tuple[str, str, str]:
    geometry_value = str(geometry or "semantic_object").strip().lower()
    if geometry_value == "sparse_marks":
        return "box", "aggregate_region", "sparse"
    if geometry_value == "dense_region":
        return "sam", "object", "dense"
    return "sam", "object", "object"


def _plan_scope_defaults(
    geometry: str, region_mode: str
) -> tuple[str, str]:
    if region_mode == "multi_instance":
        return "all_matching", "whole_object"
    if region_mode == "aggregate_region":
        return "compact_region", (
            "subpart" if geometry in {"dense_region", "sparse_marks"} else "whole_object"
        )
    return "single", (
        "subpart" if geometry in {"dense_region", "sparse_marks"} else "whole_object"
    )


def _expected_count(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str) and value.strip().lower() in {"null", "unknown", "all"}:
        return None
    if isinstance(value, bool):
        raise ValueError("expected_count must be a positive integer or null")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_count must be a positive integer or null") from exc
    if count <= 0 or count > MAX_DETECTIONS_PER_CANDIDATE:
        raise ValueError(
            f"expected_count must be in 1..{MAX_DETECTIONS_PER_CANDIDATE} or null"
        )
    return count


def _boolean_field(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{field} must be a boolean")


def _normalize_localization_items(
    raw_items: object, mode: str
) -> List[Dict[str, Any]]:
    if not isinstance(raw_items, list):
        raise ValueError("localization_items must be a list")
    result: List[Dict[str, Any]] = []
    seen = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise ValueError(f"localization item must be an object: {raw!r}")
        side = str(raw.get("image_side", "")).strip().lower()
        role = str(raw.get("role", "edit_region")).strip().lower()
        edit_op = str(raw.get("edit_op", "change")).strip().lower()
        ref = str(raw.get("ref", "")).strip()
        spatial_hint = str(raw.get("spatial_hint", raw.get("extent", ""))).strip()
        geometry = str(raw.get("geometry", "semantic_object")).strip().lower()
        default_method, default_region_mode, default_density = _plan_defaults(geometry)
        method = str(raw.get("mask_method", default_method)).strip().lower()
        region_mode = str(raw.get("region_mode", default_region_mode)).strip().lower()
        density = str(raw.get("mask_density", default_density)).strip().lower()
        default_selection, default_extent = _plan_scope_defaults(
            geometry, region_mode
        )
        selection_mode = str(
            raw.get("selection_mode", default_selection)
        ).strip().lower()
        mask_extent = str(raw.get("mask_extent", default_extent)).strip().lower()
        expected_count = _expected_count(raw.get("expected_count"))
        optional = _boolean_field(raw.get("optional", False), "optional")
        negative_space = _boolean_field(
            raw.get("negative_space", False), "negative_space"
        )
        carrier_ref = str(raw.get("carrier_ref", "")).strip()
        if side not in {"source", "target"}:
            raise ValueError(f"invalid localization image_side: {side!r}")
        if role not in {"edit_region", "protected_foreground"}:
            raise ValueError(f"invalid localization role: {role!r}")
        if edit_op not in {"add", "remove", "change", "protect"}:
            raise ValueError(f"invalid localization edit_op: {edit_op!r}")
        if not ref:
            raise ValueError("localization ref must be non-empty")
        if method not in MASK_METHODS:
            raise ValueError(f"invalid localization mask_method: {method!r}")
        if region_mode not in REGION_MODES:
            raise ValueError(f"invalid localization region_mode: {region_mode!r}")
        if density not in MASK_DENSITIES:
            raise ValueError(f"invalid localization mask_density: {density!r}")
        if selection_mode not in SELECTION_MODES:
            raise ValueError(f"invalid localization selection_mode: {selection_mode!r}")
        if mask_extent not in MASK_EXTENTS:
            raise ValueError(f"invalid localization mask_extent: {mask_extent!r}")
        if negative_space and not carrier_ref:
            raise ValueError("negative-space localization item requires carrier_ref")

        if region_mode == "multi_instance":
            selection_mode = "all_matching"
        elif selection_mode == "all_matching":
            region_mode = "multi_instance"
        elif region_mode == "aggregate_region":
            selection_mode = "compact_region"
        else:
            selection_mode = "single"

        if mode == "full_image":
            continue
        if mode == "protect_foreground":
            if role != "protected_foreground" or side != "source":
                continue
            edit_op, method, density = "protect", "sam", "object"
            # Preserve an explicit repeated-foreground topology. Collapsing
            # "all students" or "all desks" to a singleton makes the bbox
            # parser reject Qwen's otherwise correct member boxes and leaves
            # holes after foreground-mask inversion.
            if region_mode == "multi_instance" or selection_mode == "all_matching":
                region_mode, selection_mode = "multi_instance", "all_matching"
            else:
                region_mode, selection_mode, expected_count = "object", "single", None
            mask_extent = (
                "whole_actor" if mask_extent == "whole_actor" else "whole_object"
            )
            negative_space, carrier_ref, optional = False, "", False
        else:
            if role != "edit_region":
                continue
            # Item-level edit semantics prevent an unchanged support/revealed
            # background from reaching the locator in mixed compositional edits.
            if (edit_op == "add" and side == "source") or (
                edit_op == "remove" and side == "target"
            ):
                continue
        if negative_space:
            geometry, method, region_mode, density = (
                "dense_region",
                "sam",
                "object",
                "dense",
            )
            selection_mode, mask_extent, expected_count = "single", "subpart", None
            optional = False
        else:
            carrier_ref = ""
        identity = (
            side,
            role,
            edit_op,
            ref.lower(),
            spatial_hint.lower(),
            negative_space,
            carrier_ref.lower(),
            selection_mode,
            mask_extent,
            optional,
        )
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            {
                "candidate_id": len(result),
                "image_side": side,
                "role": role,
                "edit_op": edit_op,
                "ref": ref,
                "spatial_hint": spatial_hint,
                "geometry": geometry,
                "mask_method": method,
                "region_mode": region_mode,
                "selection_mode": selection_mode,
                "mask_extent": mask_extent,
                "expected_count": expected_count,
                "optional": optional,
                "mask_density": density,
                "negative_space": negative_space,
                "carrier_ref": carrier_ref,
            }
        )
    if len(result) > MAX_LOCALIZATION_ITEMS:
        raise ValueError(
            f"localization_items exceeds hard limit: {len(result)}>{MAX_LOCALIZATION_ITEMS}"
        )
    return result


def parse_observation(text: str) -> Dict[str, Any]:
    value = _first_json_object(text)
    mode = str(value.get("mask_mode", "")).strip().lower()
    if mode not in MASK_MODES:
        raise ValueError(f"invalid mask_mode: {mode!r}")
    localization_items = _normalize_localization_items(
        value.get("localization_items"), mode
    )
    if mode != "full_image" and not localization_items:
        raise ValueError(f"{mode} observation requires localization_items")
    return {
        "realized_edit": str(value.get("realized_edit", "")).strip(),
        "mask_mode": mode,
        "localization_items": localization_items,
        "confidence": str(value.get("confidence", "")).strip().lower(),
    }


def _singular_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 3:
        return token[:-2]
    if token.endswith("s") and len(token) > 2:
        return token[:-1]
    return token


def _explicit_count_for_ref(instruction: object, ref: object) -> int | None:
    """Return only counts stated by the user and tied to this query.

    Vision-language models are useful at proposing a repeated-object query, but
    their free-form counts are not reliable enough to reject a mask.  Bind an
    audit count only when an explicit cardinal in the instruction describes a
    noun that also occurs in the referring expression.
    """

    text = re.sub(r"\s+", " ", str(instruction or "").strip().lower())
    ref_tokens = {
        _singular_token(token)
        for token in re.findall(r"[a-z0-9]+", str(ref or "").lower())
        if len(token) > 1
    }
    number_pattern = "|".join(_COUNT_WORDS)
    for match in re.finditer(
        rf"\b(?P<count>\d+|{number_pattern})\b(?!\s*-?\s*tone\b)"
        rf"(?P<phrase>(?:\s+[a-z][a-z-]*){{1,5}})",
        text,
    ):
        phrase_tokens = {
            _singular_token(token)
            for token in re.findall(r"[a-z]+", match.group("phrase"))
            if len(token) > 1
        }
        if not (ref_tokens & phrase_tokens):
            continue
        raw_count = match.group("count")
        count = int(raw_count) if raw_count.isdigit() else _COUNT_WORDS[raw_count]
        if 0 < count <= MAX_DETECTIONS_PER_CANDIDATE:
            return count
    return None


def _human_owner(ref: object, instruction: object) -> str:
    text = re.sub(r"\s+", " ", str(ref or "").strip().lower())
    possessive = re.search(rf"\b((?:young\s+)?(?:{_HUMAN_WORDS})s?)['’]s\b", text)
    if possessive:
        return possessive.group(1)
    direct = re.search(rf"\b((?:young\s+)?(?:{_HUMAN_WORDS})s?)\b", text)
    if direct:
        return direct.group(1)
    instruction_text = re.sub(r"\s+", " ", str(instruction or "").strip().lower())
    direct = re.search(rf"\b((?:young\s+)?(?:{_HUMAN_WORDS})s?)\b", instruction_text)
    return direct.group(1) if direct else ""


def _body_parts(ref: object) -> set[str]:
    tokens = {
        _singular_token(token)
        for token in re.findall(r"[a-z]+", str(ref or "").lower())
    }
    return tokens & {_singular_token(token) for token in _BODY_PART_WORDS}


def apply_observation_plan_policy(
    final_task: object, instruction: object, observation: Dict[str, Any]
) -> Dict[str, Any]:
    """Normalize high-value semantic invariants before the bbox-only pass.

    This is deterministic and does not add an MLLM call.  It deliberately
    addresses only facts already stated by the instruction or redundantly
    present in the parsed plan: explicit counts, complete edited surfaces, and
    a human actor whose multiple body parts changed.
    """

    task = canonical_task(final_task)
    if observation.get("mask_mode") != "regions":
        return observation
    items = [dict(item) for item in observation.get("localization_items", [])]
    overrides: List[Dict[str, Any]] = []

    # Model-estimated counts (for example, counting people instead of visible
    # backpacks) create false failures and bias the locator.  Retain a count
    # only when the instruction explicitly quantifies this particular noun.
    for item in items:
        if item.get("selection_mode") != "all_matching":
            continue
        explicit_count = _explicit_count_for_ref(instruction, item.get("ref"))
        old_count = item.get("expected_count")
        item["expected_count"] = explicit_count
        if old_count != explicit_count:
            overrides.append(
                {
                    "rule": "only_instruction_bound_counts_are_auditable_v1",
                    "candidate_id": item.get("candidate_id"),
                    "model_count": old_count,
                    "instruction_count": explicit_count,
                }
            )

    # A uniform appearance edit may affect several differently colored pieces
    # of one named facade/surface.  Query those changed sections from the pair,
    # rather than allowing a color adjective invented by round 1 to narrow the
    # locator to just one piece.
    if task in {"color_change", "material_change"}:
        residual_surface_items: List[Dict[str, Any]] = []
        for item in items:
            if item.get("mask_extent") != "surface_region":
                continue
            old_ref = str(item.get("ref", ""))
            # Retain the model's concrete changed-color anchor. Replacing it
            # with the whole instruction subject (for example, "right
            # building facade") makes a union locator include unchanged
            # sections and can still exclude an ambiguous adjacent section.
            # The paired-image completion cue below expands from this trusted
            # changed anchor to coherent spillover only.
            item["ref"] = old_ref
            item["spatial_hint"] = (
                "compare Image 1 with Image 2 and include every separate visible section "
                "of this named surface whose appearance changes, even when source colors differ"
            )
            item.update(
                {
                    "geometry": "dense_region",
                    "mask_method": "sam",
                    "region_mode": "aggregate_region",
                    "selection_mode": "compact_region",
                    "expected_count": None,
                    "mask_density": "dense",
                    "optional": False,
                }
            )
            overrides.append(
                {
                    "rule": "appearance_surface_uses_all_changed_sections_v1",
                    "candidate_id": item.get("candidate_id"),
                    "model_ref": old_ref,
                    "locator_ref": item["ref"],
                }
            )
            if len(items) + len(residual_surface_items) < MAX_LOCALIZATION_ITEMS:
                residual_surface_items.append(
                    {
                        **item,
                        "candidate_id": -1,
                        "ref": f"additional changed surface adjoining {old_ref}",
                        "spatial_hint": (
                            "immediately adjoining the primary section; present only when a large "
                            "coherent extra section visibly changes in the paired image"
                        ),
                        "optional": True,
                    }
                )
        if residual_surface_items:
            items.extend(residual_surface_items)
            overrides.append(
                {
                    "rule": "appearance_surface_optional_residual_in_same_locator_call_v1",
                    "added_candidates": len(residual_surface_items),
                }
            )

    # If round 1 independently reports two or more body parts for the same
    # person in an action edit, the edit is no longer an isolated-limb case.
    # Collapse those redundant subpart queries to one complete actor per side.
    if task == "action_editing":
        owner_parts: Dict[str, set[str]] = {}
        owner_has_whole_actor: set[str] = set()
        for item in items:
            owner = _human_owner(item.get("ref"), instruction)
            if not owner:
                continue
            owner_parts.setdefault(owner, set()).update(_body_parts(item.get("ref")))
            if item.get("mask_extent") == "whole_actor":
                owner_has_whole_actor.add(owner)
        collapse_owners = {
            owner
            for owner, parts in owner_parts.items()
            if len(parts) >= 2 or owner in owner_has_whole_actor
        }
        if collapse_owners:
            emitted = set()
            normalized: List[Dict[str, Any]] = []
            for item in items:
                owner = _human_owner(item.get("ref"), instruction)
                if owner not in collapse_owners:
                    normalized.append(item)
                    continue
                key = (item.get("image_side"), owner)
                if key in emitted:
                    continue
                emitted.add(key)
                explicit_count = _explicit_count_for_ref(instruction, owner)
                plural = bool(
                    explicit_count is not None and explicit_count > 1
                    or re.search(r"\b(people|children|men|women|boys|girls)\b", owner)
                )
                item.update(
                    {
                        "edit_op": "remove" if item.get("image_side") == "source" else "add",
                        "ref": f"complete visible {owner}",
                        "spatial_hint": "the full seated or standing actor specified by the edit",
                        "geometry": "semantic_object",
                        "mask_method": "sam",
                        "region_mode": "multi_instance" if plural else "object",
                        "selection_mode": "all_matching" if plural else "single",
                        "mask_extent": "whole_actor",
                        "expected_count": explicit_count if plural else 1,
                        "mask_density": "object",
                        "negative_space": False,
                        "carrier_ref": "",
                    }
                )
                normalized.append(item)
            items = normalized
            overrides.append(
                {
                    "rule": "multi_body_part_action_uses_whole_actor_v1",
                    "actors": sorted(collapse_owners),
                }
            )

    for candidate_id, item in enumerate(items):
        item["candidate_id"] = candidate_id
    result = dict(observation)
    result["localization_items"] = items
    if overrides:
        result["plan_policy_overrides"] = overrides
    return result


def parse_bbox_localization(
    text: str,
    expected_candidate_ids: Sequence[int] | None = None,
    aggregate_candidate_ids: Sequence[int] | None = None,
    multi_candidate_ids: Sequence[int] | None = None,
    optional_candidate_ids: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Parse Qwen-native ``label`` + ``bbox_2d`` grounding safely.

    Explicit candidate ids are authoritative and may be returned out of order.
    Qwen also commonly emits semantic labels without ids; because the prompt
    requires one result per query in request order, a complete same-length
    array can be bound positionally. Multi-instance IDs may occur repeatedly;
    those member boxes remain separate so SAM receives one anchor per object.
    Repeated sub-boxes for an otherwise complete singleton checklist are
    conservatively unioned and marked for semantic QC.
    Equal coordinates remain distinct across different candidate IDs because
    source and target candidates can legitimately occupy the same geometry.
    """

    expected = (
        None
        if expected_candidate_ids is None
        else [int(value) for value in expected_candidate_ids]
    )
    aggregate_ids = {int(value) for value in (aggregate_candidate_ids or [])}
    multi_ids = {int(value) for value in (multi_candidate_ids or [])}
    optional_ids = {int(value) for value in (optional_candidate_ids or [])}
    if aggregate_ids & multi_ids:
        raise ValueError("candidate cannot be both aggregate_region and multi_instance")
    if expected is not None and not optional_ids.issubset(set(expected)):
        raise ValueError("optional candidate IDs must be part of the expected checklist")
    required_ids = [] if expected is None else [value for value in expected if value not in optional_ids]
    errors: List[str] = []
    found_array = False
    decoded_array_lengths: List[int] = []
    for value in _json_candidates(text):
        if not isinstance(value, list):
            continue
        # ``_json_candidates`` also sees each nested coordinate vector. Only
        # an outer array containing grounding objects is a response candidate.
        if value and not any(isinstance(item, dict) for item in value):
            continue
        found_array = True
        decoded_array_lengths.append(len(value))
        try:
            if len(value) > max(
                MAX_DETECTIONS_PER_CANDIDATE,
                len(expected or []) * MAX_DETECTIONS_PER_CANDIDATE,
            ):
                raise ValueError(f"too many bbox detections: {len(value)}")
            if expected is not None and len(value) < len(required_ids):
                raise ValueError(
                    f"bbox candidate count mismatch: required={len(required_ids)} actual={len(value)}"
                )
            parsed_items = []
            for raw in value:
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"bbox localization item must be an object: {raw!r}"
                    )
                candidate_value = raw.get("candidate_id")
                if candidate_value is None:
                    label = str(raw.get("label", ""))
                    label_match = re.search(
                        r"(?:candidate(?:_id)?\s*[:=#]?\s*|^\s*)(\d+)"
                        r"\s*(?:\||member\s*[:=#]|$)",
                        label,
                        flags=re.IGNORECASE,
                    )
                    if label_match:
                        candidate_value = label_match.group(1)
                parsed_items.append(
                    {
                        "candidate_id": (
                            None if candidate_value is None else int(candidate_value)
                        ),
                        "ref": str(raw.get("ref", "")).strip(),
                        "bbox_2d": _normalize_box(raw.get("bbox_2d")),
                    }
                )

            ids = [item["candidate_id"] for item in parsed_items]
            if expected is None:
                if any(value is None for value in ids):
                    raise ValueError("bbox candidate_id missing without an expected order")
                if len(set(ids)) != len(ids):
                    raise ValueError(f"duplicate bbox candidate_id: {ids}")
                return parsed_items

            if all(value is not None for value in ids):
                actual_set = set(ids)
                duplicate_ids = {
                    candidate_id for candidate_id in actual_set if ids.count(candidate_id) > 1
                }
                repeatable_ids = aggregate_ids | multi_ids
                if not set(required_ids).issubset(actual_set) or not actual_set.issubset(
                    set(expected)
                ):
                    if duplicate_ids and not duplicate_ids.issubset(repeatable_ids):
                        raise ValueError(f"duplicate bbox candidate_id: {ids}")
                    raise ValueError(
                        f"bbox candidate mismatch: required={required_ids} "
                        f"optional={sorted(optional_ids)} actual={ids}"
                    )
                grouped = {
                    candidate_id: [
                        item
                        for item in parsed_items
                        if item["candidate_id"] == candidate_id
                    ]
                    for candidate_id in actual_set
                }
                normalized = []
                for candidate_id in expected:
                    candidates = grouped.get(candidate_id, [])
                    if not candidates:
                        continue
                    if len(candidates) > MAX_DETECTIONS_PER_CANDIDATE:
                        raise ValueError(
                            f"too many boxes for candidate_id={candidate_id}: "
                            f"{len(candidates)}>{MAX_DETECTIONS_PER_CANDIDATE}"
                        )
                    if (
                        candidate_id in aggregate_ids
                        or candidate_id not in multi_ids
                    ) and len(candidates) > 1:
                        # Qwen sometimes decomposes one logical query (a pair
                        # of shoes, a building silhouette, an articulated
                        # object) into several valid sub-boxes despite being
                        # asked for one box.  Once every required candidate is
                        # present, union those pieces instead of failing the
                        # complete sample. Multi-instance queries deliberately
                        # keep one box per physical member.
                        boxes = [item["bbox_2d"] for item in candidates]
                        merged = {
                                "candidate_id": candidate_id,
                                "ref": candidates[0]["ref"],
                                "bbox_2d": [
                                    min(box[0] for box in boxes),
                                    min(box[1] for box in boxes),
                                    max(box[2] for box in boxes),
                                    max(box[3] for box in boxes),
                                ],
                            }
                        if candidate_id not in aggregate_ids:
                            merged["bbox_parse_recovery"] = (
                                "single_candidate_subbox_union"
                            )
                        candidates = [merged]
                    elif candidate_id in multi_ids:
                        # Remove exact duplicate detections within one image/query,
                        # while retaining nearby/overlapping physical instances.
                        unique = []
                        seen_boxes = set()
                        for candidate in candidates:
                            box_key = tuple(candidate["bbox_2d"])
                            if box_key in seen_boxes:
                                continue
                            seen_boxes.add(box_key)
                            unique.append(candidate)
                        candidates = unique
                    for member_index, candidate in enumerate(candidates):
                        normalized.append(
                            (
                                {**candidate, "member_index": member_index}
                                if candidate_id in multi_ids
                                else candidate
                            )
                        )
                return normalized

            # Native Qwen labels are often semantic-only (for example
            # ``Image 1 | sky``). The locator contract makes exact-length
            # arrays positional. Mixed explicit ids are accepted only when
            # every supplied id already agrees with its expected position.
            if (
                len(expected) == 1
                and expected[0] in multi_ids
                and all(value is None or value == expected[0] for value in ids)
            ):
                return [
                    {
                        **item,
                        "candidate_id": expected[0],
                        "member_index": member_index,
                    }
                    for member_index, item in enumerate(parsed_items)
                ]
            positional_ids = (
                expected
                if len(parsed_items) == len(expected)
                else required_ids
                if len(parsed_items) == len(required_ids)
                else None
            )
            if positional_ids is None:
                raise ValueError(
                    f"bbox candidate count mismatch: required={len(required_ids)} "
                    f"with_optional={len(expected)} actual={len(parsed_items)}"
                )
            for position, item in enumerate(parsed_items):
                supplied = item["candidate_id"]
                if supplied is not None and supplied != positional_ids[position]:
                    raise ValueError(
                        "partial bbox candidate ids conflict with request order: "
                        f"position={position} expected={positional_ids[position]} actual={supplied}"
                    )
                item["candidate_id"] = positional_ids[position]
            return parsed_items
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    # Recover Qwen's occasional repeated bbox_2d keys only when the raw count
    # exactly matches the positional checklist. Do not deduplicate coordinates:
    # the same normalized box is valid for corresponding source/target regions.
    repeated_key_recovery_allowed = not found_array or any(
        length < len(expected or []) for length in decoded_array_lengths
    )
    if expected is not None and repeated_key_recovery_allowed:
        # A long all-matching response can be cut after several complete
        # grounding objects. Recover those intact objects, preserving their
        # explicit candidate IDs, and let the normal grouping/count checks
        # validate them. Mark the recovery so downstream QC never reports the
        # result as silently equivalent to a complete JSON response.
        recovered_objects = []
        for match in re.finditer(r"\{[^{}]*\}", str(text or ""), flags=re.DOTALL):
            fragment = match.group(0)
            if '"bbox_2d"' not in fragment:
                continue
            try:
                value = json.loads(fragment)
            except json.JSONDecodeError:
                continue
            candidate_value = value.get("candidate_id")
            if candidate_value is None:
                label_match = re.search(
                    r"(?:candidate(?:_id)?\s*[:=#]?\s*|^\s*)(\d+)"
                    r"\s*(?:\||member\s*[:=#]|$)",
                    str(value.get("label", "")),
                    flags=re.IGNORECASE,
                )
                if label_match:
                    candidate_value = label_match.group(1)
            if candidate_value is None:
                continue
            try:
                recovered_objects.append(
                    {
                        "candidate_id": int(candidate_value),
                        "ref": str(value.get("ref", "")).strip(),
                        "bbox_2d": _normalize_box(value.get("bbox_2d")),
                        "bbox_parse_recovery": "partial_json_objects",
                    }
                )
            except (TypeError, ValueError):
                continue
        recovered_ids = {item["candidate_id"] for item in recovered_objects}
        raw_bbox_count = len(
            re.findall(r'"bbox_2d"\s*:\s*(\[[^\]]+\])', str(text or ""))
        )
        incomplete_single_multi_tail = bool(
            len(expected) == 1
            and expected[0] in multi_ids
            and raw_bbox_count > len(recovered_objects)
        )
        if (
            recovered_objects
            and set(required_ids).issubset(recovered_ids)
            and recovered_ids.issubset(set(expected))
            and not incomplete_single_multi_tail
        ):
            recovered_text = json.dumps(
                [
                    {
                        "candidate_id": item["candidate_id"],
                        "ref": item["ref"],
                        "bbox_2d": item["bbox_2d"],
                    }
                    for item in recovered_objects
                ]
            )
            recovered = parse_bbox_localization(
                recovered_text,
                expected,
                aggregate_candidate_ids=sorted(aggregate_ids),
                multi_candidate_ids=sorted(multi_ids),
                optional_candidate_ids=sorted(optional_ids),
            )
            return [
                {**item, "bbox_parse_recovery": "partial_json_objects"}
                for item in recovered
            ]
        raw_boxes = []
        for match in re.finditer(r'"bbox_2d"\s*:\s*(\[[^\]]+\])', str(text or "")):
            try:
                raw_boxes.append(_normalize_box(json.loads(match.group(1))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        sole_multi_id = (
            expected[0]
            if len(expected) == 1 and expected[0] in multi_ids and raw_boxes
            else None
        )
        positional_ids = (
            [sole_multi_id] * len(raw_boxes)
            if sole_multi_id is not None
            else expected
            if len(raw_boxes) == len(expected)
            else required_ids
            if len(raw_boxes) == len(required_ids)
            else None
        )
        if positional_ids is not None:
            return [
                {
                    "candidate_id": candidate_id,
                    "ref": "",
                    "bbox_2d": bbox,
                    **(
                        {"member_index": member_index}
                        if candidate_id in multi_ids
                        else {}
                    ),
                    **(
                        {"bbox_parse_recovery": "partial_bbox_coordinates"}
                        if sole_multi_id is not None
                        else {}
                    ),
                }
                for member_index, (candidate_id, bbox) in enumerate(
                    zip(positional_ids, raw_boxes)
                )
            ]

    detail = errors[-1] if errors else "no JSON bbox array found"
    if not found_array and expected is not None:
        detail = (
            f"{detail}; raw bbox count did not match required={len(required_ids)} "
            f"or with_optional={len(expected)}"
        )
    raise ValueError(detail)


def grounding_from_localization(
    observation: Dict[str, Any], localized_boxes: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """Join first-pass semantics with second-pass boxes deterministically."""

    mode = str(observation.get("mask_mode", "")).strip().lower()
    if mode not in MASK_MODES:
        raise ValueError(f"invalid planned mask_mode: {mode!r}")
    items = list(observation.get("localization_items", []))
    if mode == "full_image":
        if localized_boxes:
            raise ValueError("full_image plan cannot have localized boxes")
        return {
            "prompt_version": PROMPT_VERSION,
            "mask_mode": mode,
            "source": [],
            "target": [],
            "protected_foreground": [],
        }
    boxes_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for localized in localized_boxes:
        candidate_id = int(localized["candidate_id"])
        boxes_by_id.setdefault(candidate_id, []).append(dict(localized))
    all_expected = {int(item["candidate_id"]) for item in items}
    required = {
        int(item["candidate_id"]) for item in items if not bool(item.get("optional", False))
    }
    actual = set(boxes_by_id)
    if not required.issubset(actual) or not actual.issubset(all_expected):
        raise ValueError(
            f"bbox localization candidate mismatch: required={sorted(required)} "
            f"optional={sorted(all_expected - required)} actual={sorted(actual)}"
        )
    source: List[Dict[str, Any]] = []
    target: List[Dict[str, Any]] = []
    protected: List[Dict[str, Any]] = []
    semantic_qc_flags: List[str] = []
    for plan in items:
        candidate_id = int(plan["candidate_id"])
        candidates = boxes_by_id.get(candidate_id, [])
        if not candidates and bool(plan.get("optional", False)):
            continue
        expected_count = plan.get("expected_count")
        if expected_count is not None and len(candidates) != int(expected_count):
            semantic_qc_flags.append(
                f"COUNT_MISMATCH:candidate_id={candidate_id}:"
                f"expected={int(expected_count)}:actual={len(candidates)}"
            )
        if (
            str(plan.get("selection_mode", "single")) == "all_matching"
            and expected_count is not None
            and int(expected_count) > 1
            and len(candidates) == 1
        ):
            semantic_qc_flags.append(
                f"MULTI_INSTANCE_SINGLETON:candidate_id={candidate_id}"
            )
        for member_index, localized in enumerate(candidates):
            output = {
                "candidate_id": candidate_id,
                "member_index": int(localized.get("member_index", member_index)),
                "ref": str(plan["ref"]),
                "bbox_2d": _normalize_box(localized["bbox_2d"]),
                "mask_method": str(plan.get("mask_method", "sam")),
                "region_mode": str(plan.get("region_mode", "object")),
                "selection_mode": str(plan.get("selection_mode", "single")),
                "mask_extent": str(plan.get("mask_extent", "whole_object")),
                "expected_count": expected_count,
                "mask_density": str(plan.get("mask_density", "object")),
                "negative_space": bool(plan.get("negative_space", False)),
                "carrier_ref": str(plan.get("carrier_ref", "")),
            }
            if localized.get("bbox_parse_recovery"):
                recovery = str(localized["bbox_parse_recovery"])
                output["bbox_parse_recovery"] = recovery
                qc_prefix = (
                    "BBOX_PARTIAL_JSON_RECOVERY"
                    if recovery in {"partial_json_objects", "partial_bbox_coordinates"}
                    else "BBOX_SINGLE_QUERY_UNION_RECOVERY"
                )
                semantic_qc_flags.append(f"{qc_prefix}:candidate_id={candidate_id}")
            if localized.get("bbox_refinement"):
                output["bbox_refinement"] = dict(localized["bbox_refinement"])
            if mode == "protect_foreground":
                protected.append(
                    {
                        **output,
                        "mask_method": "sam",
                        "mask_density": "object",
                    }
                )
            elif plan.get("image_side") == "source":
                source.append(output)
            else:
                target.append(output)

    # Catch internally contradictory left-to-right localization without an
    # additional model call. Restrict this to explicit ordinal hints so phrases
    # such as "right side of the boy's body" are not treated as global order.
    horizontal_ranks = {
        "far left": 0,
        "leftmost": 0,
        "center-left": 1,
        "centre-left": 1,
        "center": 2,
        "centre": 2,
        "center-right": 3,
        "centre-right": 3,
        "far right": 4,
        "rightmost": 4,
    }

    def hint_rank(value: object) -> int | None:
        text = str(value or "").lower()
        for phrase, rank in horizontal_ranks.items():
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                return rank
        return None

    for side in ("source", "target"):
        ranked = []
        for plan in items:
            if plan.get("image_side") != side:
                continue
            rank = hint_rank(plan.get("spatial_hint"))
            candidate_id = int(plan["candidate_id"])
            candidates = boxes_by_id.get(candidate_id, [])
            if rank is None or len(candidates) != 1:
                continue
            box = candidates[0]["bbox_2d"]
            ranked.append((rank, (float(box[0]) + float(box[2])) / 2.0, candidate_id))
        ranked.sort()
        for left, right in zip(ranked, ranked[1:]):
            if left[0] < right[0] and left[1] >= right[1]:
                semantic_qc_flags.append(
                    f"SPATIAL_ORDER_MISMATCH:{side}:"
                    f"candidate_ids={left[2]},{right[2]}"
                )
    semantic_qc_flags = list(dict.fromkeys(semantic_qc_flags))
    payload = {
        "prompt_version": PROMPT_VERSION,
        "mask_mode": mode,
        "source": source,
        "target": target,
        "protected_foreground": protected,
        "semantic_qc_flags": semantic_qc_flags,
    }
    if mode == "regions" and not (source or target):
        raise ValueError("regions plan produced no grounded items")
    if mode == "protect_foreground" and not protected:
        raise ValueError("protect_foreground plan produced no protected items")
    return payload


def _normalize_box(value: Any) -> List[float]:
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(part, str) for part in value)
    ):
        coordinate_pairs = [part.split(",") for part in value]
        if all(len(pair) == 2 for pair in coordinate_pairs):
            value = [
                coordinate.strip()
                for pair in coordinate_pairs
                for coordinate in pair
            ]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"invalid bbox_2d: {value!r}")
    box = [float(number) for number in value]
    if not all(math.isfinite(number) for number in box):
        raise ValueError(f"non-finite bbox_2d: {box!r}")
    box = [min(1000.0, max(0.0, number)) for number in box]
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox_2d: {box!r}")
    return [round(number, 3) for number in box]


def grounding_status(payload: Dict[str, Any]) -> str:
    if payload.get("runtime_error"):
        return "RUNTIME_ERROR"
    if not payload.get("ground_parse_ok"):
        return "PARSE_ERROR"
    mode = payload.get("mask_mode")
    if mode == "full_image":
        return "FULL_IMAGE"
    if mode == "protect_foreground" and payload.get("protected_foreground"):
        return "PROTECT_FOREGROUND"
    if mode == "regions" and (payload.get("source") or payload.get("target")):
        return "OK"
    return "GROUND_FAIL"


def apply_task_post_policy(
    final_task: object, instruction: object, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply narrow deterministic invariants after image-conditioned grounding.

    Product-style extraction is a canvas recomposition in ScaleEdit: the kept
    subject is normally recentered/rescaled and the old subject footprint must
    also be erased. An inverse-background mask would therefore leave a false
    hole at precisely the old subject location. This invariant is deliberately
    narrow and does not affect in-place reveals that share the same task name.
    """

    task = canonical_task(final_task)
    instruction_text = re.sub(r"\s+", " ", str(instruction or "").strip().lower())
    product_extraction = task == "part_extraction" and (
        "product photography" in instruction_text
        or "product mockup" in instruction_text
        or ("extract " in instruction_text and "white background" in instruction_text)
    )
    if product_extraction and payload.get("mask_mode") != "full_image":
        result = dict(payload)
        result["route_override"] = {
            "rule": "product_extraction_is_full_canvas_recomposition_v1",
            "original_mask_mode": payload.get("mask_mode"),
            "original_source": payload.get("source", []),
            "original_target": payload.get("target", []),
            "original_protected_foreground": payload.get("protected_foreground", []),
        }
        result["mask_mode"] = "full_image"
        result["source"] = []
        result["target"] = []
        result["protected_foreground"] = []
        return result

    # Do not let an MLLM reinterpret a pure add/remove task as a replacement by
    # labeling the unchanged support or the revealed background on the opposite
    # image side.  Keep a non-empty fallback side when the expected side is
    # missing so a model error cannot turn the contract into an empty mask.
    pure_side = {
        "object_addition": ("target", "source"),
        "object_removal": ("source", "target"),
    }.get(task)
    if payload.get("mask_mode") == "regions" and pure_side is not None:
        keep_side, drop_side = pure_side
        kept_items = payload.get(keep_side, [])
        dropped_items = payload.get(drop_side, [])
        if kept_items and dropped_items:
            result = dict(payload)
            result[drop_side] = []
            result["side_override"] = {
                "rule": f"pure_{task}_uses_{keep_side}_only_v1",
                "dropped_side": drop_side,
                "dropped_items": dropped_items,
            }
            return result

    # ScaleEdit masks are stored in source coordinates. For an appearance-only
    # color/material edit, a target-side reconstruction is evidence that the
    # edit occurred, but mapping that generative drift back into source space
    # adds false regions. The planner is instructed to emit source items only;
    # this deterministic guard handles otherwise valid legacy/noncompliant plans.
    if (
        payload.get("mask_mode") == "regions"
        and task in {"color_change", "material_change"}
        and payload.get("source")
        and payload.get("target")
    ):
        result = dict(payload)
        dropped_items = payload.get("target", [])
        result["target"] = []
        result["side_override"] = {
            "rule": "appearance_only_edit_uses_source_coordinates_v1",
            "dropped_side": "target",
            "dropped_items": dropped_items,
        }
        return result

    if task == "symbolic_reasoning" and "maze" in instruction_text and "path" in instruction_text:
        # A recurrent ScaleEdit failure mode is a bbox around only the central
        # maze segment. The true route runs between distant illustrated
        # endpoints. Use a conservative maze-interior box; target->source
        # mapping adds the normal small raster dilation afterwards.
        changed = False
        result = dict(payload)
        for side in ("source", "target"):
            items = []
            for raw_item in payload.get(side, []):
                item = dict(raw_item)
                ref = str(item.get("ref", "")).lower()
                if item.get("mask_method") == "box" and ("path" in ref or "line" in ref):
                    item["bbox_2d"] = [50.0, 100.0, 950.0, 900.0]
                    changed = True
                items.append(item)
            result[side] = items
        if changed:
            result["box_override"] = {
                "rule": "maze_path_must_cover_both_endpoints_v1",
                "bbox_2d": [50.0, 100.0, 950.0, 900.0],
            }
            return result

    return payload


def object_viewpoint_ref(task: object, instruction: object) -> str:
    """Extract an explicit isolated object from common ScaleEdit templates.

    This intentionally returns nothing for camera/scene viewpoint language so
    those samples retain their image-conditioned full-image route.
    """

    task = canonical_task(task)
    instruction_text = re.sub(r"\s+", " ", str(instruction or "").strip().lower())
    if task not in {"action_editing", "viewpoint_transformation"}:
        return ""
    if re.search(r"\b(camera|view from|standing at|vantage point|entire scene)\b", instruction_text):
        return ""
    patterns = (
        r"^(?:rotate|turn)\s+(?:the\s+)?(.+?)\s+(?:clockwise|counterclockwise|to face)\b",
        r"^(?:draw|show|render)\s+(?:the\s+)?(?:front|rear|back|side|top|bottom|profile|three-quarter)(?:\s+profile)?\s+(?:view\s+)?of\s+(?:the\s+)?(.+?)(?:,|\.|$)",
        r"^(?:draw|show|render)\s+(?:the\s+)?(.+?)\s+from\s+(?:a|the)\s+(?:front|rear|back|side|top|bottom|profile|three-quarter)\b",
        r"^zoom\s+(?:in|out)\s+(?:on|from)\s+(?:the\s+)?(.+?)(?:,|\.|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, instruction_text)
        if match:
            ref = re.sub(r"\s+", " ", match.group(1)).strip(" .,'\"")
            if ref:
                return ref
    return ""
