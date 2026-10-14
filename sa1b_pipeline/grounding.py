"""Deterministic checks for spatial phrases against source mask geometry.

These checks do not replace visual review. They catch contradictions that are
cheap to detect before diffusion, especially when several same-category
instances share one image and a plan uses ``leftmost``/``rightmost`` or
``right of center`` language.
"""
from __future__ import annotations

import re
from typing import Any, Iterable


_LEFTMOST = re.compile(r"\b(?:leftmost|far\s+left|extreme\s+left)\b", re.I)
_RIGHTMOST = re.compile(r"\b(?:rightmost|far\s+right|extreme\s+right)\b", re.I)
_RIGHT_OF_CENTER = re.compile(r"\b(?:to\s+the\s+)?right\s+of\s+(?:the\s+)?center\b", re.I)
_LEFT_OF_CENTER = re.compile(r"\b(?:to\s+the\s+)?left\s+of\s+(?:the\s+)?center\b", re.I)


def _bbox(obj: Any) -> list[float] | None:
    value = obj.get("bbox") if isinstance(obj, dict) else getattr(obj, "bbox", None)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _object_id(obj: Any) -> str:
    value = obj.get("object_id") if isinstance(obj, dict) else getattr(obj, "object_id", "")
    return str(value)


def _category(obj: Any) -> str:
    value = obj.get("category") if isinstance(obj, dict) else getattr(obj, "category", "")
    # Dense SA-1B labels often contain aliases separated by commas. The first
    # alias is the stable comparison key; the full phrase remains in prompts.
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split(" ", 1)[0]


def _center_x(obj: Any) -> float | None:
    box = _bbox(obj)
    return None if box is None else (box[0] + box[2]) / 2.0


def _text(unit: Any) -> str:
    values = []
    for name in ("instruction_ref", "ref_phrase", "scope"):
        value = unit.get(name) if isinstance(unit, dict) else getattr(unit, name, "")
        if value:
            values.append(str(value))
    return " ".join(values)


def _claim(text: str) -> str | None:
    if _LEFTMOST.search(text):
        return "leftmost"
    if _RIGHTMOST.search(text):
        return "rightmost"
    if _RIGHT_OF_CENTER.search(text):
        return "right_of_center"
    if _LEFT_OF_CENTER.search(text):
        return "left_of_center"
    return None


def validate_spatial_grounding(units: Iterable[Any], objects: Iterable[Any]) -> list[str]:
    """Return hard grounding errors for high-confidence spatial claims.

    The function intentionally emits no error when a phrase has no directional
    claim or when geometry is unavailable. It only rejects a claim when there
    are at least two comparable instances and the target's relative order is
    unambiguous from their source bboxes.
    """
    by_id = {_object_id(obj): obj for obj in objects if _object_id(obj)}
    positions: dict[str, tuple[str, float]] = {}
    for object_id, obj in by_id.items():
        x = _center_x(obj)
        category = _category(obj)
        if x is not None and category:
            positions[object_id] = (category, x)

    claims: list[tuple[str, str, str, float]] = []
    for unit in units:
        object_id = str(unit.get("object_id") if isinstance(unit, dict) else getattr(unit, "object_id", ""))
        claim = _claim(_text(unit))
        if claim and object_id in positions:
            category, x = positions[object_id]
            claims.append((object_id, category, claim, x))

    errors: list[str] = []
    for object_id, category, claim, x in claims:
        peers = [(other_id, other_x) for other_id, (other_cat, other_x) in positions.items()
                 if other_cat == category and other_id != object_id]
        if not peers:
            continue
        all_x = [other_x for _, other_x in peers] + [x]
        tolerance = max(8.0, (max(all_x) - min(all_x)) * 0.04)
        if claim == "leftmost" and x > min(all_x) + tolerance:
            errors.append(f"{object_id}:leftmost_target_is_not_leftmost_{category}")
        elif claim == "rightmost" and x < max(all_x) - tolerance:
            errors.append(f"{object_id}:rightmost_target_is_not_rightmost_{category}")
        elif claim == "right_of_center" and x <= sorted(all_x)[len(all_x) // 2] + tolerance:
            errors.append(f"{object_id}:right_of_center_target_is_not_right_of_center_{category}")
        elif claim == "left_of_center" and x >= sorted(all_x)[len(all_x) // 2] - tolerance:
            errors.append(f"{object_id}:left_of_center_target_is_not_left_of_center_{category}")
    return errors
