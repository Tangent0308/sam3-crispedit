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
REGION_MODES = {"object", "aggregate_region"}
MASK_DENSITIES = {"object", "dense", "sparse"}
BBOX_LOCALIZATION_PROMPT_VERSION = "scaleedit_qwen_grounding_locator_current"


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
            "semantic segmentation is unreliable for glyphs. Include source and target text boxes."
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
            "Use source+target regions for the moved/rotated object, body part, expression, drawer, "
            "or interaction object. Avoid unrelated reconstruction drift. If the edited footprint "
            "is an empty cutout used as the rotation cue (for example a cookie bite moving around "
            "the cookie edge), label that cutout as negative_space and name its carrier object."
        ),
        "size_change": (
            "Use both the old and resized footprints of only the resized instance."
        ),
        "color_change": (
            "Use the complete recolored object or explicitly recolored sub-part on both sides, "
            "including every clearly changed repeated instance. Nearby repeated sub-parts with "
            "the same edit, such as a stack of book spines, may be one complete aggregate cluster."
        ),
        "material_change": (
            "Use the complete surface/object whose material changes on both sides, including thin "
            "structures such as rails and winding paths."
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

The images are the source of truth. Ignore resizing/JPEG noise and unrelated generative drift.
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
  new/changed entity. Pure additions emit target items only; pure removals emit source items only.
- A stand, shelf, table, wall, floor, ground, platform, shadow, or empty space is not a source edit
  merely because an added object is placed there. Put that information only in spatial_hint.
- A transfer or move from one location to another uses remove items for content disappearing at the
  origin and add items for content appearing at the destination. Do not mechanically list the
  unchanged/retained content at both locations as edited entities.
- Keep localization_items concise, normally at most 8 items total. Repeated objects that must be
  addressed independently (especially count changes or separated instances) must be separate items
  with left/right/top/bottom discriminators. If many adjacent elements receive the same edit and form
  one collective footprint, use one complete item per compact spatial cluster with
  region_mode=aggregate_region; never omit pixels merely to shorten the list.
- protect_foreground: emit only complete stable foreground entities, on image_side=source with
  role=protected_foreground and edit_op=protect.
- full_image: localization_items must be empty.
- mask_method=box for exact text/glyph blocks, thin paths/lines/cracks, dots and sparse marks;
  otherwise use sam. Use aggregate_region only for a nearby group that should share one box.
- For negative space, ref names the empty region, carrier_ref names the foreground object whose
  silhouette defines it, geometry=dense_region, mask_method=sam, region_mode=object, and
  mask_density=dense. The bbox-only pass will locate the empty region rather than the whole carrier.

Return compact JSON only. Do not output coordinates in this pass:
{{"realized_edit":"one precise sentence","mask_mode":"regions|protect_foreground|full_image","localization_items":[{{"image_side":"source|target","role":"edit_region|protected_foreground","edit_op":"add|remove|change|protect","ref":"concrete visible entity, 2-10 words","spatial_hint":"concise instance-disambiguating location","geometry":"semantic_object|dense_region|sparse_marks","mask_method":"sam|box","region_mode":"object|aggregate_region","mask_density":"object|dense|sparse","negative_space":false,"carrier_ref":"empty unless negative_space is true"}}],"confidence":"high|medium|low"}}
"""


def build_grounding_prompt(
    final_task: object, instruction: object, observation: Dict[str, Any] | str
) -> str:
    """Build a Qwen-style visual-grounding-only request from the semantic plan.

    Deliberately keep routing and mask metadata out of this pass.  Qwen only
    needs a short referring expression, an image index, and a stable id to do
    the coordinate prediction; the caller deterministically joins the boxes
    back to the richer first-pass items.
    """

    del final_task, instruction
    if not isinstance(observation, dict):
        raise ValueError("bbox localization requires a parsed edit plan")
    queries = []
    for item in observation.get("localization_items", []):
        candidate_id = int(item["candidate_id"])
        image_side = str(item["image_side"])
        image_number = 1 if image_side == "source" else 2
        ref = " ".join(str(item["ref"]).split())
        spatial_hint = " ".join(str(item.get("spatial_hint", "")).split())
        geometry = str(item.get("geometry", "semantic_object"))
        negative_space = bool(item.get("negative_space", False))
        carrier_ref = " ".join(str(item.get("carrier_ref", "")).split())
        location = "" if not spatial_hint else f"; location: {spatial_hint}"
        if negative_space:
            description = (
                f'complete visible empty/white negative-space region of "{ref}"{location}; '
                f'its boundary is defined by the foreground carrier "{carrier_ref}". '
                "Box the full cutout or concavity from its deepest interior edge through its open "
                "mouth at the carrier's expected outer contour, even when it blends into the "
                "surrounding background; exclude carrier material"
            )
        elif geometry == "dense_region":
            description = (
                f'complete visible material/region pixels of "{ref}"{location}; '
                "exclude its container, support, and surrounding background"
            )
        elif geometry == "sparse_marks":
            description = (
                f'complete visible marks of "{ref}"{location}; '
                "exclude the carrier object or surface"
            )
        else:
            description = f'complete visible object "{ref}"{location}'
        queries.append(f"{candidate_id} | Image {image_number} | {description}")
    if not queries:
        raise ValueError("bbox localization requires at least one candidate")
    query_text = "\n".join(queries)
    return f"""Visual grounding only. The edit analysis is complete. Do not reinterpret the edit.
Image 1 is the source. Image 2 is the edited result.

Detect and locate exactly these referred entities in their named full image:
{query_text}

For each line, output one tight bounding box around the complete visible entity. Include all visible
parts (such as head, ears, limbs, handles, edges, and thin extensions) with a small margin. Exclude
support surfaces, shadows, and neighboring objects.

Return only a JSON array with exactly {len(queries)} objects in the same candidate order. Every
object must retain the candidate_id at the start of its input line:
[{{"candidate_id": 0, "bbox_2d": [x1, y1, x2, y2]}}]
bbox_2d is the normalized 0-1000 top-left and bottom-right coordinates on the named full image.
"""


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
    for value in _json_candidates(text):
        if isinstance(value, dict):
            return value
    raise ValueError("no JSON object found")


def _plan_defaults(geometry: object) -> tuple[str, str, str]:
    geometry_value = str(geometry or "semantic_object").strip().lower()
    if geometry_value == "sparse_marks":
        return "box", "aggregate_region", "sparse"
    if geometry_value == "dense_region":
        return "sam", "object", "dense"
    return "sam", "object", "object"


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
        if negative_space and not carrier_ref:
            raise ValueError("negative-space localization item requires carrier_ref")

        if mode == "full_image":
            continue
        if mode == "protect_foreground":
            if role != "protected_foreground" or side != "source":
                continue
            edit_op, method, region_mode, density = "protect", "sam", "object", "object"
            negative_space, carrier_ref = False, ""
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
                "mask_density": density,
                "negative_space": negative_space,
                "carrier_ref": carrier_ref,
            }
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


def parse_bbox_localization(
    text: str, expected_candidate_ids: Sequence[int] | None = None
) -> List[Dict[str, Any]]:
    """Parse the compact full-image candidate_id-to-bbox response."""

    parse_error: ValueError | None = None
    try:
        for value in _json_candidates(text):
            if not isinstance(value, list):
                continue
            result = []
            seen = set()
            for raw in value:
                if not isinstance(raw, dict):
                    raise ValueError(f"bbox localization item must be an object: {raw!r}")
                candidate_value = raw.get("candidate_id")
                if candidate_value is None:
                    label_match = re.match(
                        r"\s*(\d+)\s*\|", str(raw.get("label", ""))
                    )
                    if label_match:
                        candidate_value = label_match.group(1)
                candidate_id = int(candidate_value)
                if candidate_id in seen:
                    raise ValueError(f"duplicate bbox candidate_id: {candidate_id}")
                seen.add(candidate_id)
                result.append(
                    {
                        "candidate_id": candidate_id,
                        "ref": str(raw.get("ref", "")).strip(),
                        "bbox_2d": _normalize_box(raw.get("bbox_2d")),
                    }
                )
            if expected_candidate_ids is not None:
                expected = [int(item) for item in expected_candidate_ids]
                actual = [item["candidate_id"] for item in result]
                if actual != expected:
                    raise ValueError(
                        f"bbox candidate mismatch: expected={expected} actual={actual}"
                    )
            return result
        raise ValueError("no JSON bbox array found")
    except (TypeError, ValueError) as exc:
        parse_error = ValueError(str(exc))

    try:
        # Qwen occasionally emits the requested boxes in order but collapses
        # them into one JSON object with repeated ``bbox_2d`` keys.  A normal
        # JSON decoder keeps only the last key.  Since the locator contract is
        # explicitly positional, recover only when the raw box count matches
        # the full expected checklist exactly; otherwise preserve the failure.
        if expected_candidate_ids is None:
            raise parse_error
        raw_boxes = []
        for match in re.finditer(r'"bbox_2d"\s*:\s*(\[[^\]]+\])', str(text or "")):
            try:
                raw_boxes.append(_normalize_box(json.loads(match.group(1))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        # Some malformed answers repeat one of the same boxes under another
        # duplicate key. Collapse only exact repeats; recovery is still
        # accepted only when the remaining count exactly matches the complete
        # positional checklist.
        unique_boxes = []
        for bbox in raw_boxes:
            if bbox not in unique_boxes:
                unique_boxes.append(bbox)
        expected = [int(value) for value in expected_candidate_ids]
        if len(unique_boxes) != len(expected):
            raise parse_error
        return [
            {"candidate_id": candidate_id, "ref": "", "bbox_2d": bbox}
            for candidate_id, bbox in zip(expected, unique_boxes)
        ]
    except ValueError:
        raise parse_error


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
    box_by_id = {int(item["candidate_id"]): item["bbox_2d"] for item in localized_boxes}
    expected = {int(item["candidate_id"]) for item in items}
    if set(box_by_id) != expected:
        raise ValueError(
            f"bbox localization candidate mismatch: expected={sorted(expected)} "
            f"actual={sorted(box_by_id)}"
        )
    source: List[Dict[str, Any]] = []
    target: List[Dict[str, Any]] = []
    protected: List[Dict[str, Any]] = []
    for plan in items:
        output = {
            "ref": str(plan["ref"]),
            "bbox_2d": _normalize_box(box_by_id[int(plan["candidate_id"])]),
            "mask_method": str(plan.get("mask_method", "sam")),
            "region_mode": str(plan.get("region_mode", "object")),
            "mask_density": str(plan.get("mask_density", "object")),
            "negative_space": bool(plan.get("negative_space", False)),
            "carrier_ref": str(plan.get("carrier_ref", "")),
        }
        if mode == "protect_foreground":
            protected.append({**output, "mask_method": "sam", "region_mode": "object", "mask_density": "object"})
        elif plan.get("image_side") == "source":
            source.append(output)
        else:
            target.append(output)
    payload = {
        "prompt_version": PROMPT_VERSION,
        "mask_mode": mode,
        "source": source,
        "target": target,
        "protected_foreground": protected,
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
