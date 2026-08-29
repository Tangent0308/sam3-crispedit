"""Shared policy and parsing for the CrispEdit MLLM grounding stage.

The grounding coordinate system is always ``[0, 1000]`` in the image named by
``grounding_image``.  Pixel conversion intentionally happens only in stage 2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


SINGLE_PASS_PROMPT_VERSION = "qwen35_grounding_v2"
TWO_PASS_PROMPT_VERSION = "qwen35_realized_edit_region_ground_v7"
OBSERVATION_PROMPT_VERSION = "qwen35_realized_edit_spec_v4"
BBOX_REFINEMENT_PROMPT_VERSION = "qwen35_local_bbox_refinement_v1"
# Two-pass is the default on the mask-improved branch.  The runner still
# exposes v2 single-pass for controlled A/B and backwards-compatible runs.
PROMPT_VERSION = TWO_PASS_PROMPT_VERSION

TYPE_ALIASES = {
    "add": "add",
    "remove": "remove",
    "replace": "replace",
    "color": "color",
    "motion": "motion",
    "motion change": "motion",
    "background": "background",
    "background change": "background",
    "style": "style",
}

GROUNDING_ROUTES = {
    "add": ("target",),
    "remove": ("source",),
    "replace": ("source", "target"),
    "color": ("source",),
    "motion": ("source", "target"),
    "background": ("source",),
    "style": (),
}

_NON_VISUAL_REF_RE = re.compile(r"^(?:the\s+)?(?:absence|lack)\s+of\b", re.IGNORECASE)
_SUBJECT_SURFACE_RE = re.compile(
    r"\b(?:skin|complexion|fur|feathers?|hair|scales?)\b", re.IGNORECASE
)
_LOCAL_BODY_EDIT_RE = re.compile(
    r"\b(?:arms?|hands?|faces?|heads?|necks?|ears?|legs?|feet|foot|bodies|"
    r"skin|complexion|fur|feathers?|hair|scales?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GroundingRequest:
    grounding_image: str
    prompt: str


def prompt_version_for_mode(mode: str) -> str:
    if mode == "two-pass":
        return TWO_PASS_PROMPT_VERSION
    if mode == "single":
        return SINGLE_PASS_PROMPT_VERSION
    raise ValueError(f"unknown grounding mode: {mode!r}")


def canonicalize_type(raw_type: object) -> str:
    value = re.sub(r"\s+", " ", str(raw_type or "").strip().lower())
    if value not in TYPE_ALIASES:
        raise ValueError(f"unknown CrispEdit type: {raw_type!r}")
    return TYPE_ALIASES[value]


def grounding_images(raw_type: object) -> Sequence[str]:
    return GROUNDING_ROUTES[canonicalize_type(raw_type)]


def _visible_ref(value: object) -> str:
    ref = str(value or "").strip()
    if ref.lower() in {
        "empty",
        "none",
        "nothing",
        "n/a",
        "na",
        "not present",
        "no object",
        "absent",
        "removed",
        "gone",
        "no longer present",
    }:
        return ""
    return ref


def _task_rule(etype: str, grounding_image: str, has_observation: bool = False) -> str:
    if etype == "add":
        if grounding_image == "source":
            return (
                "Find source-side content that the prior comparison says was also removed or "
                "replaced during this add edit. Do not box empty space."
            )
        return (
            "Find the complete spatial region occupied by newly added content in Image 2. "
            "Do not include similar content already present in Image 1."
        )
    if etype == "remove":
        if grounding_image == "target":
            return (
                "Find target-side content that the prior comparison says was also added or "
                "replaced during this remove edit. Do not box empty space."
            )
        return (
            "Find the complete spatial region occupied in Image 1 by content removed from "
            "Image 2. Include all clearly removed elements."
        )
    if etype == "replace":
        side = "old/replaced content" if grounding_image == "source" else "new replacement content"
        image_name = "Image 1" if grounding_image == "source" else "Image 2"
        return f"Find the {side} on {image_name}; cover the complete replacement footprint."
    if etype == "color":
        if has_observation:
            return (
                "Find every complete object or sub-part on Image 1 whose color/material/texture "
                "actually changed. Independently recheck every visible same-material subject "
                "surface before boxing: for a skin-tone edit this explicitly includes face/head, "
                "neck, arms/hands, and exposed legs even if pass 1 omitted or marked one unchanged. "
                "Use matching detail tiles as evidence, and add a part only when its source/result "
                "appearance changes in the same direction. Emit separate part-level refs and cover "
                "the complete extent of each spatially separated changed region."
            )
        return (
            "Find the complete object whose color/material/texture is changed. Box the "
            "whole object, not only the most visibly changed patch. If the instruction "
            "explicitly names a body part or sub-part, box every changed instance of that part."
        )
    if etype == "motion":
        if has_observation:
            side = "Image 1 before the change" if grounding_image == "source" else "Image 2 after the change"
            return (
                f"Find on {side} the body part, object, and immediate interaction footprint that "
                "the prior visual comparison identifies as actually edited. Do not expand a local "
                "pose edit to the whole person merely because pixels were regenerated."
            )
        return (
            "Find the body parts and directly interacted objects whose pose or position changes "
            "on the selected image. Prefer the local interaction region (for example hand + pen "
            "+ clipboard), not the whole person, unless the person's whole-body pose changes."
        )
    if etype == "background":
        return (
            "Find every foreground subject/object in Image 1 that must stay unchanged while the "
            "background is replaced. Do NOT box the background. Include attached foreground "
            "details and important foreground props that must be protected."
        )
    raise ValueError(f"no grounding task for type {etype}")


def build_change_observation_prompt(raw_type: object, instruction: object) -> str:
    """Ask pass 1 for a precise, image-grounded specification of the realized edit."""

    etype = canonicalize_type(raw_type)
    type_audit = ""
    if etype == "color":
        type_audit = (
            "\nMandatory independent color/material audit: temporarily ignore the instruction's "
            "claimed spatial scope and compare matching source/result regions made of the same "
            "subject material. Use the enlarged matching detail tiles when supplied. For a "
            "person/animal, explicitly check visible face/head, neck, each arm/hand, and exposed "
            "legs/feet; do not invent torso skin hidden by clothing. For every checked region, "
            "write source_appearance and target_appearance before deciding changed. A region may "
            "be changed=false only after stating what color/material is visibly observed on both "
            "sides. List a clear same-direction spillover even if the instruction omits it, but "
            "do not promote weak global lighting drift. If multiple disconnected exposed surfaces "
            "changed, emit separate change items for face/head, arms/hands, and legs/feet. NEVER "
            "replace these part-level items with a whole-person sam_ref.\n"
        )
    return f"""You are comparing a source image and its edited result before spatial grounding.

Image 1 is the source image. Image 2 is the edited result.
Edit type: {etype}
Instruction: {str(instruction or '').strip()}

Inspect both images carefully and rewrite the instruction into a precise specification of what
actually changed in the realized result. The instruction is a useful hypothesis about intent, but
the paired images are the source of truth. Your description will be used by another model to draw
complete edit-region boxes, so resolve vague or image-inconsistent wording now.
{type_audit}

Rules:
- State the concrete visual entity or region, its before/after appearance, and its complete spatial
  extent. The edit_summary must describe the realized edit rather than copy the instruction.
- Include every clear edited object or local region, including a clear instruction-external change.
  Example: if arms and the face both become darker, list both even if only the arms were requested.
- Distinguish one object/region, one nearby group, and genuinely separate regions. Many small,
  scattered but nearby points (piercings, petals, stars, spots, flowers, debris, etc.) are ONE
  nearby group; describe the whole group's layout and outer extent instead of enumerating dots.
- Separate items only when they form spatially well-separated edit regions or require different
  visual noun phrases. Do not split one contiguous object into incidental sub-parts.
- Different semantic categories that need different segmentation phrases remain separate even when
  they touch (for example a hand and a tablet, or an arm and a tool). For color/material edits,
  also separate visibly disconnected changed surfaces such as face/neck versus arms/hands; clothing
  or background between them must not be absorbed into one whole-person change.
- For additions use an empty source_ref; for removals use an empty target_ref. For replacement,
  motion, color, material, and texture edits describe both sides when visible.
- For motion, describe the actually moved body/object and immediate interaction footprint. Do not
  label the whole person as changed solely because pose-conditioned reconstruction differs subtly.
- Ignore JPEG artifacts, alignment/resampling noise, weak reconstruction drift, and weak global
  illumination drift unless that global change is the requested edit.
- source_ref and target_ref identify what is visible on each side. sam_ref is a concise 2-8 word
  visual noun phrase suitable for segmentation. It must include distinguishing visible attributes
  such as color/material and the concrete head noun (for example "rows of small white boats" or
  "pink roses and white blooms"). Avoid context/location phrases such as "in marina", "on arch",
  or "around woman", and avoid vague labels such as "mixed objects". region_description gives
  precise location, layout, and extent. Do not output coordinates or boxes.
- Each sam_ref must name exactly one segmentable semantic category. A combined phrase such as
  "hands holding tablet", "hand and pen", or "person with bicycle" is invalid: create separate
  change items/refs for the body part and prop.
- Populate checked_regions before changes. It must include the named edit region and plausible
  same-subject spillover regions you visually compared. Set changed from the images, not from the
  instruction. Every checked region with changed=true must also appear in changes.
- Prefer a complete description of clear changes, but do not manufacture edits from subtle unrelated
  generative differences. A response that merely restates the instruction is wrong.
- Output JSON only, with no markdown or explanation.

Required schema:
{{"edit_summary":"precise realized edit in one sentence","checked_regions":[{{"ref":"visual region checked","source_appearance":"literal source appearance","target_appearance":"literal result appearance","changed":true}}],"changes":[{{"source_ref":"old visual entity/region or empty","target_ref":"new visual entity/region or empty","sam_ref":"one specific segmentable category","region_description":"exact location, layout, and complete extent","region_layout":"single|nearby_group|separate_regions","change":"specific before-to-after visual change","instruction_aligned":true}}]}}
"""


def _observation_context(observation: Any) -> str:
    if observation is None:
        return ""
    if isinstance(observation, str):
        return observation.strip()
    return json.dumps(observation, ensure_ascii=False, separators=(",", ":"))


def _subject_surface_family(instruction: object, observation: Any) -> str:
    # This high-recall audit is intentionally narrow.  Pass-1 observations for
    # an ordinary whole-object recolor often mention incidental body-like parts
    # (for example an alien's head and hands); allowing that generated text to
    # select the route can shrink a correct whole-object box to those parts.
    # Route selection therefore comes only from the user's requested surface.
    instruction_text = str(instruction or "")
    if not _LOCAL_BODY_EDIT_RE.search(instruction_text):
        return ""
    matches = [
        match.group(0).lower()
        for match in _SUBJECT_SURFACE_RE.finditer(instruction_text)
    ]
    return matches[0] if matches else ""


def _build_subject_surface_grounding_prompt(surface_family: str, observation: Any) -> str:
    known_changes = []
    if isinstance(observation, dict):
        known_changes = observation.get("changes", [])
    known_refs = []
    for change in known_changes:
        if not isinstance(change, dict):
            continue
        ref = str(
            change.get("sam_ref")
            or change.get("source_ref")
            or change.get("target_ref")
            or ""
        ).strip()
        if ref and ref not in known_refs:
            known_refs.append(ref)
    known_text = json.dumps(known_refs, ensure_ascii=False, separators=(",", ":"))
    return f"""Independently ground a same-subject surface/material recoloring in Image 1.

Image 1 is the source full image. Image 2 is the edited result full image.
Surface/material family to compare: {surface_family}
REQUIRED known changed refs from pass 1: {known_text}

Do not assume the original instruction's named spatial scope is complete, and do not inherit a
previous unchanged verdict. First output a complete bbox for EVERY required known ref above; none
may be dropped. Then independently compare matching source/result appearances to ADD any omitted
same-surface region, using the enlarged paired detail tiles as evidence. The required refs are a
recall floor, not a complete list. For a person or animal, check every VISIBLE region made of
this same surface/material: face/head, ears, neck, left and right arms/hands, and exposed legs/feet.
Do not invent skin/fur hidden by clothing.

Output one item for each spatially disconnected part whose attribute clearly changes in the same
direction. Use part-level SAM phrases such as "man's face and neck" or "man's bare arms and hands";
never box the whole person. Each conservative bbox must contain the part's complete visible extent,
including fingers and thin extremities, with a small safety margin. Coordinates always refer to the
FULL Image 1 on the 0-1000 scale, never to a detail tile. If only one part truly changes, return only
that part, except that every REQUIRED known ref must still be returned. Before finalizing, verify that
each required ref has a corresponding output item. Output JSON only, no explanation.

Required schema:
[{{"ref":"man's face and neck","bbox_2d":[x1,y1,x2,y2],"region_mode":"object","mask_density":"dense"}}]
"""


def build_grounding_prompt(
    raw_type: object,
    instruction: object,
    grounding_image: str,
    observation: Any = None,
) -> str:
    etype = canonicalize_type(raw_type)
    if grounding_image not in {"source", "target"} or (
        grounding_image not in GROUNDING_ROUTES[etype] and observation is None
    ):
        raise ValueError(f"{grounding_image!r} is not routed for {etype}")
    selected = "Image 1 (source)" if grounding_image == "source" else "Image 2 (result)"
    supplemental_side = grounding_image not in GROUNDING_ROUTES[etype]
    instruction_label = (
        "Original instruction (context only; do NOT ground its absent opposite-side object)"
        if supplemental_side
        else "Instruction"
    )
    observation_text = _observation_context(observation)
    surface_family = _subject_surface_family(instruction, observation)
    if etype == "color" and grounding_image == "source" and surface_family:
        return _build_subject_surface_grounding_prompt(surface_family, observation)
    observation_block = (
        "\nPrior source/result comparison (a realized-edit specification):\n"
        f"{observation_text}\n"
        "Use its concrete entity, region_description, grouping, and sam_ref, while verifying them "
        "against the two images. Ground clear realized changes relevant to the selected image; do "
        "not revert to vague instruction wording or invent changes absent from the images. Only "
        "the change items present in THIS specification are applicable to the selected side. An "
        "earlier conversational item omitted here because it exists only on the opposite image "
        "must not be boxed.\n"
        if observation_text
        else ""
    )
    return f"""You are grounding the spatial footprint of a successful image edit.

Image 1 is the source image. Image 2 is the edited result.
Edit type: {etype}
{instruction_label}: {str(instruction or '').strip()}
{observation_block}

Only produce boxes in {selected}. Use the other image only as visual evidence for what changed.
Task: {_task_rule(etype, grounding_image, bool(observation_text))}

Rules:
- Return one JSON item per SPATIALLY SEPARATED EDIT REGION, not per object instance or tiny point.
- If multiple small edited elements are scattered but spatially near or form one local pattern,
  ALWAYS emit ONE aggregate_region box enclosing the entire group, regardless of count. Examples:
  facial piercings, a spray of petals, a flower halo, stars/spots, crumbs, or small removed marks.
  Never draw one box per dot in such a group.
- Use separate boxes only for clearly well-separated clusters/objects with substantial empty space
  between them. Repeated elements that form one band, halo, patch, row, or local area are one region.
- Spatially touching DIFFERENT semantic categories still need separate boxes and refs when one SAM
  phrase cannot name both precisely. In particular, do not combine hands/arms with a tablet, bowl,
  clipboard, pen, drill, or other interacted prop. Do not combine disconnected face/neck skin with
  arms/hands into a whole-person box.
- Each ref must name exactly ONE segmentable semantic category. Phrases mixing categories with
  "and", "with", or "holding" (for example "hands holding tablet") are invalid. Emit overlapping
  hand and tablet boxes with separate refs when both changed. Color/material body edits must use
  part refs such as "man's face and neck" and "man's bare arms and hands", never "man/person".
- A bbox is a conservative SAM search region, not the final mask. It MUST be a superset of the full
  edited footprint. Inspect the leftmost, rightmost, topmost, and bottommost edited pixels/elements,
  then add a small safety margin. Missing any edited part is worse than modest extra context.
- For a normal object or body part, include its complete contour, thin extremities, and any clearly
  co-changing shadow/reflection. Do not crop at the image edge; use 0 or 1000 when appropriate.
- Each ref must be a short, SAM-friendly visual noun phrase of 2-8 words. Name only the visual
  target (for example "rows of small white boats", "pink roses and white blooms", or "facial
  piercings"). Include visible color/material/category attributes that distinguish the edited target.
  Never use context/location nouns such as marina/arch/garden/person to substitute for appearance,
  and do not repeat spatial wording such as "surrounding the girl". Prefer a specific prior sam_ref.
- Set region_mode="aggregate_region" for a nearby multi-element group; otherwise use "object".
- Set mask_density="sparse" when separate small elements have visible gaps between their masks
  (piercings, petals, stars, small flowers, birds, etc.); set it to "dense" for a contiguous or
  overlapping cluster/bed/solid region; use "object" for a normal single object.
- Avoid nested duplicate boxes for the same edit. The region set should cover every described clear
  edit while remaining faithful to the selected image.
- Coordinates are relative to the selected image on a 0-1000 scale, independent of image resize.
- If the requested edit target truly is not visible in the selected image, return [].
- On a supplemental opposite-side request, return ONLY concrete entities named on that side of the
  provided specification. A removed source entity is absent from Image 2 and must never be boxed
  there; likewise a target-only addition does not exist in Image 1.
- Output JSON only. No markdown and no explanation.

Required schema:
[{{"ref":"facial piercings","bbox_2d":[x1,y1,x2,y2],"region_mode":"aggregate_region","mask_density":"sparse"}}]
"""


def build_grounding_requests(
    raw_type: object,
    instruction: object,
    observation: Any = None,
) -> List[GroundingRequest]:
    etype = canonicalize_type(raw_type)
    default_images = list(grounding_images(etype))
    images = list(default_images)
    # Add/remove samples occasionally realize a clear collateral operation in
    # the opposite direction (e.g. remove piercings but add earrings).  Route
    # only explicit source-only/target-only or instruction-external evidence,
    # avoiding redundant target grounding for ordinary color/motion changes.
    if etype in {"add", "remove"} and isinstance(observation, dict):
        for change in observation.get("changes", []):
            if not isinstance(change, dict):
                continue
            source_ref = _visible_ref(change.get("source_ref"))
            target_ref = _visible_ref(change.get("target_ref"))
            if source_ref and not target_ref and "source" not in images:
                images.append("source")
            if target_ref and not source_ref and "target" not in images:
                images.append("target")
    requests = []
    for image in images:
        context = observation
        if isinstance(observation, dict) and isinstance(observation.get("changes"), list):
            side_key = f"{image}_ref"
            other_key = "target_ref" if image == "source" else "source_ref"
            supplemental = image not in default_images
            applicable = []
            for change in observation["changes"]:
                if not isinstance(change, dict) or not _visible_ref(change.get(side_key)):
                    continue
                if supplemental and _visible_ref(change.get(other_key)) and bool(
                    change.get("instruction_aligned", True)
                ):
                    continue
                applicable.append(change)
            context = {
                "edit_summary": observation.get("edit_summary", ""),
                "changes": applicable,
            }
        requests.append(
            GroundingRequest(
                image,
                build_grounding_prompt(raw_type, instruction, image, context),
            )
        )
    return requests


def _candidate_json_objects(text: str) -> Iterable[str]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    yield cleaned
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(cleaned)):
            char = cleaned[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield cleaned[start : end + 1]
                    break


def parse_change_observation(text: str) -> Dict:
    """Parse and normalize the first-pass realized-change checklist."""

    parsed = None
    last_error: Exception | None = None
    for candidate in _candidate_json_objects(text):
        try:
            value = json.loads(candidate)
            # A malformed outer object may still contain valid nested objects.
            # Do not mistake one checked-region/change item for the response.
            if isinstance(value, dict) and isinstance(value.get("changes"), list):
                parsed = value
                break
        except (TypeError, ValueError) as exc:
            last_error = exc
    if parsed is None:
        # Qwen can omit a brace inside the verbose checked_regions list while
        # still emitting a complete, valid changes array.  The latter is the
        # trusted input to pass 2, so recover it and discard only the damaged
        # audit section.  Restrict the search to text after the `changes` key so
        # a valid checked_regions array is never misinterpreted as changes.
        changes_match = re.search(r'"changes"\s*:', str(text or ""))
        if changes_match:
            suffix = str(text or "")[changes_match.end() :]
            for candidate in _candidate_json_arrays(suffix):
                try:
                    value = json.loads(candidate)
                    if isinstance(value, list) and value and all(
                        isinstance(item, dict) for item in value
                    ):
                        summary = ""
                        summary_match = re.search(
                            r'"(?:edit_summary|summary)"\s*:\s*'
                            r'("(?:\\.|[^"\\])*")',
                            str(text or ""),
                        )
                        if summary_match:
                            summary = str(json.loads(summary_match.group(1))).strip()
                        parsed = {
                            "edit_summary": summary,
                            "checked_regions": [],
                            "changes": value,
                        }
                        break
                except (TypeError, ValueError) as exc:
                    last_error = exc
        if parsed is None:
            raise ValueError(f"no JSON observation object found: {last_error}")
    summary = str(parsed.get("edit_summary", parsed.get("summary", ""))).strip()
    checked_regions = []
    raw_checks = parsed.get("checked_regions", [])
    if raw_checks is not None and not isinstance(raw_checks, list):
        raise ValueError("observation checked_regions must be a list")
    for index, item in enumerate(raw_checks or []):
        if not isinstance(item, dict):
            raise ValueError(f"checked region {index} is not an object")
        ref = str(item.get("ref", "")).strip()
        if not ref:
            raise ValueError(f"checked region {index} has an empty ref")
        changed = item.get("changed", False)
        if isinstance(changed, str):
            changed = changed.strip().lower() in {"true", "yes", "1"}
        checked = {"ref": ref, "changed": bool(changed)}
        if "source_appearance" in item:
            checked["source_appearance"] = str(item.get("source_appearance", "")).strip()
        if "target_appearance" in item:
            checked["target_appearance"] = str(item.get("target_appearance", "")).strip()
        checked_regions.append(checked)
    changes = parsed.get("changes")
    if not isinstance(changes, list):
        raise ValueError("observation changes must be a list")
    normalized = []
    for index, item in enumerate(changes):
        if not isinstance(item, dict):
            raise ValueError(f"observation change {index} is not an object")
        source_ref = _visible_ref(item.get("source_ref", ""))
        target_ref = _visible_ref(item.get("target_ref", ""))
        sam_ref = str(item.get("sam_ref", source_ref or target_ref)).strip()
        region_description = str(
            item.get("region_description", item.get("spatial_extent", ""))
        ).strip()
        raw_layout = str(item.get("region_layout", "single")).strip().lower()
        layout_aliases = {
            "single": "single",
            "single_object": "single",
            "object": "single",
            "nearby_group": "nearby_group",
            "aggregate": "nearby_group",
            "aggregate_region": "nearby_group",
            "separate_regions": "separate_regions",
        }
        region_layout = layout_aliases.get(raw_layout, "single")
        change = str(item.get("change", item.get("description", ""))).strip()
        if not source_ref and not target_ref:
            raise ValueError(f"observation change {index} has no visible source/target ref")
        if not change:
            raise ValueError(f"observation change {index} has an empty change description")
        aligned = item.get("instruction_aligned", True)
        if isinstance(aligned, str):
            aligned = aligned.strip().lower() in {"true", "yes", "1"}
        normalized.append(
            {
                "source_ref": source_ref,
                "target_ref": target_ref,
                "sam_ref": sam_ref,
                "region_description": region_description,
                "region_layout": region_layout,
                "change": change,
                "instruction_aligned": bool(aligned),
            }
        )
    if not normalized:
        raise ValueError("observation contains no realized changes")
    return {
        "edit_summary": summary,
        "checked_regions": checked_regions,
        "changes": normalized,
    }


def _candidate_json_arrays(text: str) -> Iterable[str]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    yield cleaned
    starts = [index for index, char in enumerate(cleaned) if char == "["]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(cleaned)):
            char = cleaned[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    yield cleaned[start : end + 1]
                    break


def _salvage_truncated_grounding(text: str) -> List[Dict]:
    """Recover complete boxes from a response cut off by max-new-tokens.

    Qwen occasionally tries to express many instances by repeating ``bbox_2d``
    keys in one object.  For the recall-first mask policy, their aggregate
    extent is still useful and strictly better than dropping the sample.
    """

    bbox_pattern = re.compile(
        r'"bbox_2d"\s*:\s*\[\s*'
        r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
        r'(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    )
    ref_pattern = re.compile(r'"ref"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
    label_pattern = re.compile(r'"label"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')
    box_matches = list(bbox_pattern.finditer(text))
    ref_matches = list(ref_pattern.finditer(text))
    labels_precede_boxes = bool(ref_matches)
    label_matches = ref_matches or list(label_pattern.finditer(text))
    if not box_matches or not label_matches:
        return []
    decoded_labels = [
        (match.start(), json.loads(f'"{match.group(1)}"').strip())
        for match in label_matches
    ]
    raw_items = []
    for box_match in box_matches:
        preceding = [item for item in decoded_labels if item[0] < box_match.start()]
        # Repeated bbox keys in one object inherit its single preceding ref.
        # If a malformed response puts the label after the bbox, fall back to
        # the first following label instead of discarding a recoverable box.
        if labels_precede_boxes and preceding:
            ref = preceding[-1][1]
        else:
            following = [item for item in decoded_labels if item[0] > box_match.start()]
            if following:
                ref = following[0][1]
            elif preceding:
                ref = preceding[-1][1]
            else:
                continue
        raw_items.append((ref, [float(value) for value in box_match.groups()]))
    results = []
    for ref, bbox in raw_items:
        clipped = [max(0.0, min(1000.0, float(value))) for value in bbox]
        if (
            clipped[0] < clipped[2]
            and clipped[1] < clipped[3]
            and not _NON_VISUAL_REF_RE.search(ref)
        ):
            results.append({"ref": ref, "bbox_2d": [round(value, 3) for value in clipped]})
    return results


def parse_grounding_output(text: str) -> List[Dict]:
    """Parse, validate, clip and normalize one model response."""

    parsed = None
    last_error: Exception | None = None
    for candidate in _candidate_json_arrays(text):
        try:
            value = json.loads(candidate)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                parsed = value
                break
        except (TypeError, ValueError) as exc:
            last_error = exc
    if parsed is None:
        salvaged = _salvage_truncated_grounding(text)
        if salvaged:
            return salvaged
        raise ValueError(f"no JSON object array found: {last_error}")

    # JSON permits duplicate object keys and silently keeps only the last one.
    # Qwen uses repeated bbox_2d keys surprisingly often for same-class
    # instances, so prefer the positional recovery whenever it finds more
    # complete boxes than the normal parse retained.
    salvaged = _salvage_truncated_grounding(text)
    if len(salvaged) > len(parsed):
        return salvaged

    results: List[Dict] = []
    for index, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} is not an object")
        ref = str(item.get("ref", item.get("label", ""))).strip()
        bbox = item.get("bbox_2d")
        if not ref:
            raise ValueError(f"item {index} has an empty ref")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"item {index} has invalid bbox_2d")
        try:
            coords = [max(0.0, min(1000.0, float(value))) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"item {index} has non-numeric bbox_2d") from exc
        x1, y1, x2, y2 = coords
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"item {index} has a degenerate bbox_2d: {coords}")
        # Replacement prompts occasionally coax the model into boxing an
        # abstract non-entity such as "absence of hands".  It has no pixels on
        # that side; the opposite-side replacement footprint remains valid.
        if _NON_VISUAL_REF_RE.search(ref):
            continue
        result = {"ref": ref, "bbox_2d": [round(value, 3) for value in coords]}
        if "region_mode" in item:
            raw_mode = str(item.get("region_mode", "object")).strip().lower()
            result["region_mode"] = (
                "aggregate_region"
                if raw_mode in {"aggregate", "aggregate_region", "nearby_group", "group"}
                else "object"
            )
        if "mask_density" in item:
            raw_density = str(item.get("mask_density", "object")).strip().lower()
            result["mask_density"] = (
                raw_density if raw_density in {"sparse", "dense", "object"} else "object"
            )
        results.append(result)
    return results


def box_needs_local_refinement(box: Dict, threshold: float = 220.0) -> bool:
    """Return whether a normalized grounding box is too small to trust globally.

    Small edit targets receive relatively few vision tokens when two wide full
    images are shown together.  The semantic noun is usually correct, but the
    resulting box is often shifted or clips a thin extremity.  Those candidates
    are sent through a high-resolution crop-localized verification pass.
    """

    bbox = box.get("bbox_2d") if isinstance(box, dict) else None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return min(x2 - x1, y2 - y1) < float(threshold)


def local_bbox_refinement_enabled(raw_type: object) -> bool:
    """Return whether boxes for an edit route represent editable regions."""

    return canonicalize_type(raw_type) not in {"background", "style"}


def bbox_refinement_crop(
    bbox: Sequence[float],
    min_context: float = 120.0,
    context_scale: float = 1.0,
) -> List[float]:
    """Build a context-rich full-image crop in normalized coordinates."""

    x1, y1, x2, y2 = (float(value) for value in bbox)
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"cannot refine degenerate bbox: {bbox}")
    x_margin = max(float(min_context), width * float(context_scale))
    y_margin = max(float(min_context), height * float(context_scale))
    return [
        round(max(0.0, x1 - x_margin), 3),
        round(max(0.0, y1 - y_margin), 3),
        round(min(1000.0, x2 + x_margin), 3),
        round(min(1000.0, y2 + y_margin), 3),
    ]


def build_bbox_refinement_prompt(candidates: Sequence[Dict]) -> str:
    """Ask Qwen to re-ground initial proposals in enlarged candidate crops."""

    checklist = []
    for candidate in candidates:
        checklist.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "ref": str(candidate["ref"]),
            }
        )
    checklist_text = json.dumps(checklist, ensure_ascii=False, separators=(",", ":"))
    return f"""Verify and correct small edit-region bounding boxes using enlarged image crops.

Each candidate image shown above is an enlarged crop for exactly one checklist item, in the same
order as the checklist.  Treat each crop as an independent image with its own [0,1000] coordinate
system.  Each crop was proposed by a noisy detector, so the requested entity may be off-center and
the proposal may originally have contained only its upper/lower half.  No full-image coordinates
are provided here; derive every returned number solely from the visible candidate crop.

Checklist: {checklist_text}

For every candidate, locate the concrete visual entity named by ref inside that candidate's crop.
Return a conservative crop-relative bbox that contains its COMPLETE visible contour.  Explicitly
check top, bottom, left and right extremes.  Include thin extremities, the full crown/brim of
headwear, the full hand including fingers, the complete mouth/lips rather than the nose, and the
entire held object rather than only its most salient half.  Modest surrounding context is better
than clipping any edited pixel.  Do not infer or reuse coordinates from any other image.

Return exactly one item for each candidate_id, using coordinates relative to that candidate's CROP,
not the full image.  Output JSON only, with no markdown or explanation.

Required schema:
[{{"candidate_id":0,"ref":"black cap","bbox_2d":[x1,y1,x2,y2]}}]
"""


def parse_bbox_refinement_output(text: str) -> List[Dict]:
    """Parse crop-relative bbox refinements while retaining candidate ids."""

    parsed = None
    last_error: Exception | None = None
    for candidate in _candidate_json_arrays(text):
        try:
            value = json.loads(candidate)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                parsed = value
                break
        except (TypeError, ValueError) as exc:
            last_error = exc
    if parsed is None:
        raise ValueError(f"no bbox refinement JSON array found: {last_error}")

    results = []
    seen = set()
    for index, item in enumerate(parsed):
        try:
            candidate_id = int(item.get("candidate_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"refinement item {index} has invalid candidate_id") from exc
        if candidate_id in seen:
            raise ValueError(f"duplicate refinement candidate_id: {candidate_id}")
        seen.add(candidate_id)
        bbox = item.get("bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"refinement item {index} has invalid bbox_2d")
        try:
            coords = [max(0.0, min(1000.0, float(value))) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"refinement item {index} has non-numeric bbox_2d") from exc
        if not (coords[0] < coords[2] and coords[1] < coords[3]):
            raise ValueError(f"refinement item {index} has a degenerate bbox_2d: {coords}")
        results.append(
            {
                "candidate_id": candidate_id,
                "ref": str(item.get("ref", "")).strip(),
                "bbox_2d": [round(value, 3) for value in coords],
            }
        )
    return results


def map_crop_bbox_to_full(
    crop_bbox: Sequence[float], crop_relative_bbox: Sequence[float]
) -> List[float]:
    """Map a [0,1000] crop-relative box back into full-image coordinates."""

    crop_x1, crop_y1, crop_x2, crop_y2 = (float(value) for value in crop_bbox)
    x1, y1, x2, y2 = (float(value) for value in crop_relative_bbox)
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    return [
        round(crop_x1 + x1 * crop_width / 1000.0, 3),
        round(crop_y1 + y1 * crop_height / 1000.0, 3),
        round(crop_x1 + x2 * crop_width / 1000.0, 3),
        round(crop_y1 + y2 * crop_height / 1000.0, 3),
    ]


def conservative_refined_bbox(
    initial_bbox: Sequence[float],
    refined_bbox: Sequence[float] | None,
    min_padding: float = 12.0,
    max_padding: float = 24.0,
    union_iou_threshold: float = 0.25,
    union_containment_threshold: float = 0.75,
    fallback_min_padding: float = 60.0,
    fallback_max_padding: float = 80.0,
) -> List[float]:
    """Return a recall-first envelope around initial and locally refined boxes.

    A successful local answer receives only a modest edge margin.  The initial
    and refined boxes are unioned when they overlap enough to represent
    compatible edge estimates, or when one substantially contains the other.
    The containment check prevents a crop pass that sees only a salient
    subpart (for example a hammer head) from discarding a recall-safe initial
    box.  A clearly shifted proposal (for example a nose box corrected to a
    mouth) must not contaminate the final region.  A larger recall-first
    expansion is reserved for refinement failure.
    """

    x1, y1, x2, y2 = (float(value) for value in initial_bbox)
    if refined_bbox is None:
        min_extent = min(x2 - x1, y2 - y1)
        padding = min(
            float(fallback_max_padding),
            max(float(fallback_min_padding), 0.5 * min_extent),
        )
        envelope = [x1 - padding, y1 - padding, x2 + padding, y2 + padding]
    else:
        rx1, ry1, rx2, ry2 = (float(value) for value in refined_bbox)
        intersection_width = max(0.0, min(x2, rx2) - max(x1, rx1))
        intersection_height = max(0.0, min(y2, ry2) - max(y1, ry1))
        intersection = intersection_width * intersection_height
        initial_area = (x2 - x1) * (y2 - y1)
        refined_area = (rx2 - rx1) * (ry2 - ry1)
        union_area = initial_area + refined_area - intersection
        iou = intersection / union_area if union_area > 0 else 0.0
        smaller_area = min(initial_area, refined_area)
        containment = intersection / smaller_area if smaller_area > 0 else 0.0
        if (
            iou >= float(union_iou_threshold)
            or containment >= float(union_containment_threshold)
        ):
            base = [min(x1, rx1), min(y1, ry1), max(x2, rx2), max(y2, ry2)]
        else:
            base = [rx1, ry1, rx2, ry2]
        min_extent = min(base[2] - base[0], base[3] - base[1])
        padding = min(float(max_padding), max(float(min_padding), 0.15 * min_extent))
        envelope = [
            base[0] - padding,
            base[1] - padding,
            base[2] + padding,
            base[3] + padding,
        ]
    return [round(max(0.0, min(1000.0, value)), 3) for value in envelope]


def grounding_is_complete(etype: str, boxes_by_image: Dict[str, List[Dict]]) -> bool:
    """Return whether a sample has enough grounding to enter stage 2.

    Replacement is intentionally asymmetric: edits such as "replace the absence of
    hands with hands" have no source entity to box, but a valid target footprint.
    """

    if etype == "style":
        return True
    if etype == "replace":
        return bool(boxes_by_image.get("source") or boxes_by_image.get("target"))
    return all(bool(boxes_by_image.get(image)) for image in GROUNDING_ROUTES[etype])
