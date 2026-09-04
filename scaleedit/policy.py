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
            "style change is local; only an explicit whole-image transformation is full_image."
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
            "far-apart or semantically different regions."
        ),
        "count_change": (
            "Ground only instances that appear or disappear, not unchanged comparison instances."
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
            "background, or global changes using the image pair."
        ),
        "visual_beautification": (
            "Face/skin restoration is local to the complete affected faces or people; object "
            "repair uses the damaged object's footprint; do not default to full image."
        ),
        "action_editing": (
            "Use source+target regions for the moved/rotated object, body part, expression, drawer, "
            "or interaction object. Avoid unrelated reconstruction drift."
        ),
        "size_change": (
            "Use both the old and resized footprints of only the resized instance."
        ),
        "color_change": (
            "Use the complete recolored object or explicitly recolored sub-part on both sides, "
            "including every clearly changed repeated instance."
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
Describe every intentional realized edit, not merely the nouns in the instruction. Decide the
spatial mask route from actual geometry; ScaleEdit task names are not reliable route aliases.

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

Return compact JSON only:
{{"realized_edit":"one precise sentence","mask_mode":"regions|protect_foreground|full_image","changes":[{{"source_ref":"visible old/changed entity or empty","target_ref":"visible new/changed entity or empty","change":"before to after","geometry":"semantic_object|dense_region|sparse_marks","source_extent":"source location/extent or empty","target_extent":"target location/extent or empty"}}],"protected_foreground":[{{"ref":"retained concrete entity","extent":"complete source location/extent"}}],"confidence":"high|medium|low"}}
"""


def build_grounding_prompt(
    final_task: object, instruction: object, observation: Dict[str, Any] | str
) -> str:
    task = canonical_task(final_task)
    observation_text = (
        json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        if isinstance(observation, dict)
        else str(observation)
    )
    return f"""Convert the paired-image audit below into the final ScaleEdit mask contract.

Image 1 is the source. Image 2 is the edited result. All coordinates are [x1,y1,x2,y2] on a
0..1000 scale in the NAMED full image. The two images can have different pixel dimensions.
Final task: {task}
Corrected instruction: {str(instruction or '').strip()}
Prior audit: {observation_text}

First verify the audit against both images. Then obey these rules:
1. Preserve its mask_mode unless direct visual evidence proves it wrong.
2. regions: source contains each visible old/changed footprint; target contains each visible
   new/changed footprint. Recolor/material/size/pose/repair/text generally need both sides. Pure
   addition needs target only; pure removal needs source only. Cover every realized sub-edit.
3. protect_foreground: source and target must be empty. protected_foreground contains complete
   SOURCE-image boxes for all stable foreground subjects, including thin limbs/edges. Do not protect
   sky, water, terrain, road, generic vegetation, or environmental content being replaced.
4. full_image: all three item lists must be empty.
5. Each ref is a concrete 2-10 word visible noun phrase suitable for SAM, never an action or absence.
6. Bboxes are conservative recall-first search regions with a small margin. Do not use a whole-image
   box to hide uncertainty. Split distant or semantically different entities; combine a nearby group
   of repeated tiny elements into one aggregate_region bbox.
7. mask_method=box for exact text/glyph blocks, thin lines/paths/cracks, dots, tiny marks, and other
   sparse content SAM could erase. The box must tightly cover the complete changed block, not its
   carrier sign/screen/object. A path bbox must include the entire path from its start endpoint to
   its destination endpoint, not merely its center. Use mask_method=sam for coherent objects and
   surfaces.
8. mask_density=sparse for separated marks; dense for a filled local area; object for normal objects.
9. For text tasks ({', '.join(sorted(TEXT_TASKS))}), locate the changed old and new text itself and
   use box. For a local style edit, box/segment only the styled surface, never default to full_image.
10. Output JSON only. Do not include commentary or markdown.

Required schema:
{{"prompt_version":"{PROMPT_VERSION}","mask_mode":"regions|protect_foreground|full_image","source":[{{"ref":"old or changed visible entity","bbox_2d":[x1,y1,x2,y2],"mask_method":"sam|box","region_mode":"object|aggregate_region","mask_density":"object|dense|sparse"}}],"target":[{{"ref":"new or changed visible entity","bbox_2d":[x1,y1,x2,y2],"mask_method":"sam|box","region_mode":"object|aggregate_region","mask_density":"object|dense|sparse"}}],"protected_foreground":[{{"ref":"stable foreground entity","bbox_2d":[x1,y1,x2,y2],"mask_method":"sam","region_mode":"object","mask_density":"object"}}]}}
"""


def build_object_viewpoint_retry_prompt(
    final_task: object,
    instruction: object,
    observation: Dict[str, Any] | str,
    object_ref: str,
) -> str:
    """Request real object boxes after a false full-image viewpoint route."""

    base = build_grounding_prompt(final_task, instruction, observation)
    return f"""{base}

MANDATORY ROUTE CORRECTION: `{object_ref}` is one isolated transformed object, not a camera or
whole-scene transformation. mask_mode MUST be regions. Return a conservative source bbox around
the complete old `{object_ref}` footprint and a target bbox around the complete new `{object_ref}`
footprint. The boxes must be real object-localized coordinates; do not use [0,0,1000,1000].
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


def parse_observation(text: str) -> Dict[str, Any]:
    value = _first_json_object(text)
    mode = str(value.get("mask_mode", "")).strip().lower()
    if mode not in MASK_MODES:
        raise ValueError(f"invalid mask_mode: {mode!r}")
    changes = value.get("changes", [])
    protected = value.get("protected_foreground", [])
    if not isinstance(changes, list) or not isinstance(protected, list):
        raise ValueError("changes and protected_foreground must be lists")
    return {
        "realized_edit": str(value.get("realized_edit", "")).strip(),
        "mask_mode": mode,
        "changes": [item for item in changes if isinstance(item, dict)],
        "protected_foreground": [item for item in protected if isinstance(item, dict)],
        "confidence": str(value.get("confidence", "")).strip().lower(),
    }


def _normalize_box(value: Any) -> List[float]:
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


def _normalize_items(value: Any, *, protected: bool = False) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("grounding item collection must be a list")
    result = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError(f"grounding item must be an object: {raw!r}")
        ref = str(raw.get("ref", "")).strip()
        if not ref:
            raise ValueError("grounding ref must be non-empty")
        method = "sam" if protected else str(raw.get("mask_method", "sam")).strip().lower()
        if method not in MASK_METHODS:
            raise ValueError(f"invalid mask_method: {method!r}")
        region_mode = str(raw.get("region_mode", "object")).strip().lower()
        if region_mode not in REGION_MODES:
            raise ValueError(f"invalid region_mode: {region_mode!r}")
        density = str(raw.get("mask_density", "object")).strip().lower()
        if density not in MASK_DENSITIES:
            raise ValueError(f"invalid mask_density: {density!r}")
        result.append(
            {
                "ref": ref,
                "bbox_2d": _normalize_box(raw.get("bbox_2d")),
                "mask_method": method,
                "region_mode": region_mode,
                "mask_density": density,
            }
        )
    return result


def parse_grounding(text: str) -> Dict[str, Any]:
    value = _first_json_object(text)
    mode = str(value.get("mask_mode", "")).strip().lower()
    if mode not in MASK_MODES:
        raise ValueError(f"invalid mask_mode: {mode!r}")
    source = _normalize_items(value.get("source", []))
    target = _normalize_items(value.get("target", []))
    protected = _normalize_items(value.get("protected_foreground", []), protected=True)
    if mode == "regions" and not (source or target):
        raise ValueError("regions mode requires at least one source or target box")
    if mode == "protect_foreground" and not protected:
        raise ValueError("protect_foreground mode requires a protected foreground box")
    if mode == "full_image":
        source, target, protected = [], [], []
    elif mode == "regions":
        protected = []
    else:
        source, target = [], []
    return {
        "prompt_version": PROMPT_VERSION,
        "mask_mode": mode,
        "source": source,
        "target": target,
        "protected_foreground": protected,
    }


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
