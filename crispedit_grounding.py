"""Shared policy and parsing for the CrispEdit MLLM grounding stage.

The grounding coordinate system is always ``[0, 1000]`` in the image named by
``grounding_image``.  Pixel conversion intentionally happens only in stage 2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


PROMPT_VERSION = "qwen35_grounding_v2"

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


@dataclass(frozen=True)
class GroundingRequest:
    grounding_image: str
    prompt: str


def canonicalize_type(raw_type: object) -> str:
    value = re.sub(r"\s+", " ", str(raw_type or "").strip().lower())
    if value not in TYPE_ALIASES:
        raise ValueError(f"unknown CrispEdit type: {raw_type!r}")
    return TYPE_ALIASES[value]


def grounding_images(raw_type: object) -> Sequence[str]:
    return GROUNDING_ROUTES[canonicalize_type(raw_type)]


def _task_rule(etype: str, grounding_image: str) -> str:
    if etype == "add":
        return (
            "Find every newly added object or decoration in Image 2. Do not box a "
            "similar object that was already present in Image 1."
        )
    if etype == "remove":
        return (
            "Find every object or decoration in Image 1 that is removed in Image 2. "
            "Include all repeated or dispersed removed instances."
        )
    if etype == "replace":
        side = "old/replaced content" if grounding_image == "source" else "new replacement content"
        image_name = "Image 1" if grounding_image == "source" else "Image 2"
        return f"Find the {side} on {image_name}; cover the complete replacement footprint."
    if etype == "color":
        return (
            "Find the complete object whose color/material/texture is changed. Box the "
            "whole object, not only the most visibly changed patch. If the instruction "
            "explicitly names a body part or sub-part, box every changed instance of that part."
        )
    if etype == "motion":
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


def build_grounding_prompt(raw_type: object, instruction: object, grounding_image: str) -> str:
    etype = canonicalize_type(raw_type)
    if grounding_image not in GROUNDING_ROUTES[etype]:
        raise ValueError(f"{grounding_image!r} is not routed for {etype}")
    selected = "Image 1 (source)" if grounding_image == "source" else "Image 2 (result)"
    return f"""You are grounding the spatial footprint of a successful image edit.

Image 1 is the source image. Image 2 is the edited result.
Edit type: {etype}
Instruction: {str(instruction or '').strip()}

Only produce boxes in {selected}. Use the other image only as visual evidence for what changed.
Task: {_task_rule(etype, grounding_image)}

Rules:
- Return one JSON item per distinct relevant instance. Include every relevant repeated instance.
- Use a loose aggregate box only when there are more than 8 tiny/dense instances or the instances
  truly cannot be counted. Do not replace a small set of distinct instances with one large region.
- Sparse or hollow edits need special care: for borders/frames, piercings, tattoos, scattered
  petals, and decorations around a subject, box each visible instance when there are at most 8.
  When an aggregate is unavoidable, tightly enclose the complete set, including its extremes.
- Each ref must be a short, SAM-friendly visual noun phrase of 2-8 words. Name only the visual
  target (for example "pastel blue flowers" or "facial piercings"); do not repeat spatial wording
  such as "surrounding the girl" because the bbox already expresses location.
- Boxes must be loose enough to include the full object, thin extremities, and any clearly
  co-changing shadow/reflection. Coverage is more important than tightness.
- Coordinates are relative to the selected image on a 0-1000 scale, independent of image resize.
- If the requested edit target truly is not visible in the selected image, return [].
- Output JSON only. No markdown and no explanation.

Required schema:
[{{"ref":"front black chair","bbox_2d":[x1,y1,x2,y2]}}]
"""


def build_grounding_requests(raw_type: object, instruction: object) -> List[GroundingRequest]:
    return [
        GroundingRequest(image, build_grounding_prompt(raw_type, instruction, image))
        for image in grounding_images(raw_type)
    ]


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
        results.append({"ref": ref, "bbox_2d": [round(value, 3) for value in coords]})
    return results


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
