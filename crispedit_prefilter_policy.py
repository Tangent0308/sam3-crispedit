#!/usr/bin/env python3
"""Evidence normalization and deterministic decisions for the CrispEdit fact prefilter.

The vision-language model is deliberately not asked for a keep/drop verdict.  It
only extracts instruction slots and reports factual observations.  This module
normalizes those observations and applies the per-edit-type predicates described
in ``PREFILTER_FALSE_KEEP_OPTIMIZATION.md``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PREFILTER_METHOD = "fact"
EVIDENCE_SCHEMA = "fact_evidence"

VALID_EDIT_TYPES = {
    "add",
    "remove",
    "replace",
    "color",
    "motion",
    "background",
    "style",
}

TYPE_ALIASES = {
    "addition": "add",
    "background change": "background",
    "background_change": "background",
    "color change": "color",
    "colour": "color",
    "colour change": "color",
    "motion change": "motion",
    "motion_change": "motion",
    "removal": "remove",
    "replacement": "replace",
    "style change": "style",
}

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "dark",
    "darker",
    "gold",
    "gray",
    "green",
    "grey",
    "light",
    "lighter",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "teal",
    "white",
    "yellow",
}

TRUE = "TRUE"
FALSE = "FALSE"
UNKNOWN = "UNKNOWN"


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_edit_type(raw_type: object) -> str:
    value = str(raw_type or "").strip().lower().replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    value = TYPE_ALIASES.get(value, value)
    return value if value in VALID_EDIT_TYPES else "unknown"


def _choice(value: object, allowed: Sequence[str], default: str) -> str:
    normalized = str(value or "").strip().upper()
    aliases = {
        "TRUE": "YES",
        "FALSE": "NO",
        "N/A": "UNCLEAR",
        "UNKNOWN": "UNCLEAR",
        "LOCAL": "CONTROLLED_EDIT",
        "LOCAL_EDIT": "CONTROLLED_EDIT",
        "GLOBAL": "GLOBAL_REGEN",
        "REGENERATED": "GLOBAL_REGEN",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in allowed else default


def _confidence(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not math.isfinite(result):
        result = default
    return max(0.0, min(1.0, result))


def _optional_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, result)


def _short_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_bbox(value: object) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        coords = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    # Qwen-VL commonly emits its documented 0..1000 coordinate convention
    # even when a prompt asks for normalized values.  Preserve those boxes
    # instead of clamping every upper coordinate to 1 and discarding the box.
    if max(coords) > 1.0:
        if min(coords) < 0.0 or max(coords) > 1000.0:
            return None
        coords = [item / 1000.0 for item in coords]
    coords = [max(0.0, min(1.0, item)) for item in coords]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None
    return coords


def normalize_slots(parsed: Dict, raw_type: object) -> Dict:
    """Normalize Step 0 output into a stable, auditable schema."""

    if not isinstance(parsed, dict):
        parsed = {}
    raw_canonical = canonical_edit_type(raw_type)
    status = _choice(
        parsed.get("instruction_status"),
        ("NORMAL", "NO_OP", "UNJUDGEABLE"),
        "UNJUDGEABLE",
    )
    raw_subgoals = parsed.get("subgoals")
    if not isinstance(raw_subgoals, list):
        raw_subgoals = []
    if not raw_subgoals and any(
        parsed.get(key) for key in ("object_a", "object_b", "attribute", "part")
    ):
        raw_subgoals = [parsed]

    subgoals: List[Dict] = []
    for index, item in enumerate(raw_subgoals):
        if not isinstance(item, dict):
            continue
        edit_type = canonical_edit_type(item.get("edit_type") or raw_canonical)
        if edit_type == "unknown":
            edit_type = raw_canonical
        object_a = _short_text(item.get("object_a"))
        object_b = _short_text(item.get("object_b"))
        attribute = _short_text(item.get("attribute"))
        # Qwen occasionally calls an attire-color change a replacement because
        # the wording says "red outfit".  Keep this correction deliberately
        # narrow: the raw row must be color and the proposed B slot must contain
        # an explicit color word.
        if (
            raw_canonical == "color"
            and edit_type == "replace"
            and COLOR_WORDS.intersection(_normalized_words(object_b))
        ):
            edit_type = "color"
            attribute = attribute or object_b
            object_b = ""
        subgoals.append(
            {
                "subgoal_index": index,
                "edit_type": edit_type,
                "object_a": object_a,
                "object_b": object_b,
                "attribute": attribute,
                "part": _short_text(item.get("part")),
                "count": _optional_int(item.get("count")),
                "location": _short_text(item.get("location")),
            }
        )

    if status == "NORMAL" and not subgoals:
        subgoals = [
            {
                "subgoal_index": 0,
                "edit_type": raw_canonical,
                "object_a": "",
                "object_b": "",
                "attribute": "",
                "part": "",
                "count": None,
                "location": "",
            }
        ]

    return {
        "schema": EVIDENCE_SCHEMA,
        "instruction_status": status,
        "raw_canonical_type": raw_canonical,
        "subgoals": subgoals,
        "confidence": _confidence(parsed.get("confidence")),
        "notes": _short_text(parsed.get("notes")),
    }


def deterministic_slot_conflict(slots: Dict, deterministic: Dict) -> bool:
    """Return a conservative cross-check flag without overriding MLLM slots."""

    if not deterministic or not deterministic.get("parse_ok", False):
        return False
    subgoals = slots.get("subgoals") or []
    if not subgoals:
        return False
    first = subgoals[0]
    checks = (
        (_short_text(deterministic.get("source")), _short_text(first.get("object_a"))),
        (_short_text(deterministic.get("target")), _short_text(first.get("object_b"))),
    )
    for left, right in checks:
        if left and right and not _texts_overlap(left, right):
            return True
    return False


def normalize_single_image_evidence(parsed: Dict) -> Dict:
    if not isinstance(parsed, dict):
        parsed = {}
    facts: List[Dict] = []
    raw_facts = parsed.get("facts")
    if not isinstance(raw_facts, list):
        raw_facts = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        raw_boxes = item.get("bboxes")
        if not isinstance(raw_boxes, list):
            raw_boxes = []
        boxes = [box for box in (_normalize_bbox(value) for value in raw_boxes) if box]
        facts.append(
            {
                "subgoal_index": _optional_int(item.get("subgoal_index")),
                "role": _choice(item.get("role"), ("OBJECT_A", "OBJECT_B", "PART"), "OBJECT_A"),
                "query": _short_text(item.get("query")),
                "present": _choice(item.get("present"), ("YES", "NO", "UNCLEAR"), "UNCLEAR"),
                "count": _optional_int(item.get("count")),
                "bboxes": boxes,
                "attribute_value": _short_text(item.get("attribute_value")),
                "pose": _short_text(item.get("pose")),
                "confidence": _confidence(item.get("confidence")),
            }
        )
    return {
        "scene_description": _short_text(parsed.get("scene_description")),
        "scene_label": _short_text(parsed.get("scene_label")),
        "style_label": _short_text(parsed.get("style_label")),
        "subject_identity": _short_text(parsed.get("subject_identity")),
        "subject_bbox": _normalize_bbox(parsed.get("subject_bbox")),
        "facts": facts,
        "crop_observation": _short_text(parsed.get("crop_observation")),
        "confidence": _confidence(parsed.get("confidence")),
    }


def normalize_pair_evidence(parsed: Dict) -> Dict:
    if not isinstance(parsed, dict):
        parsed = {}
    differences: List[Dict] = []
    raw_differences = parsed.get("visible_differences")
    if not isinstance(raw_differences, list):
        raw_differences = []
    for item in raw_differences:
        if isinstance(item, str):
            description = _short_text(item)
            significance = "MEDIUM"
        elif isinstance(item, dict):
            description = _short_text(item.get("description"))
            significance = _choice(item.get("significance"), ("LOW", "MEDIUM", "HIGH"), "MEDIUM")
        else:
            continue
        if description:
            differences.append({"description": description, "significance": significance})
    return {
        "visible_differences": differences,
        "same_subject": _choice(parsed.get("same_subject"), ("YES", "NO", "UNCLEAR"), "UNCLEAR"),
        "composition_preserved": _choice(
            parsed.get("composition_preserved"), ("YES", "NO", "UNCLEAR"), "UNCLEAR"
        ),
        "unrelated_regions_preserved": _choice(
            parsed.get("unrelated_regions_preserved"), ("YES", "NO", "UNCLEAR"), "UNCLEAR"
        ),
        "edit_scope": _choice(
            parsed.get("edit_scope"),
            ("CONTROLLED_EDIT", "GLOBAL_REGEN", "UNCLEAR"),
            "UNCLEAR",
        ),
        "confidence": _confidence(parsed.get("confidence")),
    }


def normalize_text_match(parsed: Dict, subgoal_count: int) -> Dict:
    if not isinstance(parsed, dict):
        parsed = {}
    matches: Dict[int, Dict] = {}
    raw_matches = parsed.get("subgoal_matches")
    if not isinstance(raw_matches, list):
        raw_matches = []
    for item in raw_matches:
        if not isinstance(item, dict):
            continue
        index = _optional_int(item.get("subgoal_index"))
        if index is None or index >= subgoal_count:
            continue
        matches[index] = {
            "subgoal_index": index,
            "match": _choice(
                item.get("match"),
                ("MATCH", "PARTIAL", "MISMATCH", "NOT_MENTIONED"),
                "NOT_MENTIONED",
            ),
            "reason": _short_text(item.get("reason")),
            "confidence": _confidence(item.get("confidence")),
        }
    normalized = []
    for index in range(subgoal_count):
        normalized.append(
            matches.get(
                index,
                {
                    "subgoal_index": index,
                    "match": "NOT_MENTIONED",
                    "reason": "No match result returned.",
                    "confidence": 0.0,
                },
            )
        )
    return {
        "subgoal_matches": normalized,
        "overall_match": _choice(
            parsed.get("overall_match"),
            ("MATCH", "PARTIAL", "MISMATCH", "NOT_MENTIONED"),
            "NOT_MENTIONED",
        ),
        "confidence": _confidence(parsed.get("confidence")),
    }


def _fact(evidence: Dict, subgoal_index: int, role: str) -> Optional[Dict]:
    for item in evidence.get("facts") or []:
        if item.get("subgoal_index") == subgoal_index and item.get("role") == role:
            return item
    return None


def _present(fact: Optional[Dict]) -> str:
    if not fact or fact.get("present") == "UNCLEAR":
        return UNKNOWN
    return TRUE if fact.get("present") == "YES" else FALSE


def _tri_and(values: Iterable[str]) -> str:
    values = list(values)
    if any(value == FALSE for value in values):
        return FALSE
    if values and all(value == TRUE for value in values):
        return TRUE
    return UNKNOWN


def _tri_not(value: str) -> str:
    if value == TRUE:
        return FALSE
    if value == FALSE:
        return TRUE
    return UNKNOWN


def _normalized_words(text: object) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _texts_overlap(left: object, right: object) -> bool:
    a, b = set(_normalized_words(left)), set(_normalized_words(right))
    if not a or not b:
        return False
    return bool(a & b) or " ".join(a) in " ".join(b) or " ".join(b) in " ".join(a)


def _different(left: object, right: object) -> str:
    a, b = _normalized_words(left), _normalized_words(right)
    if not a or not b:
        return UNKNOWN
    if a == b:
        return FALSE
    sa, sb = set(a), set(b)
    similarity = len(sa & sb) / max(len(sa | sb), 1)
    return FALSE if similarity >= 0.8 else TRUE


STATE_TOKEN_ALIASES = {
    "lower": "lower",
    "lowered": "lower",
    "lowering": "lower",
    "down": "lower",
    "downward": "lower",
    "lap": "lower",
    "raise": "raise",
    "raised": "raise",
    "raising": "raise",
    "upward": "raise",
    "straight": "upright",
    "straighten": "upright",
    "straightened": "upright",
    "upright": "upright",
    "erect": "upright",
    "lean": "lean",
    "leaned": "lean",
    "leaning": "lean",
    "bent": "lean",
    "slouch": "lean",
    "slouched": "lean",
    "tilt": "tilt",
    "tilted": "tilt",
    "tilting": "tilt",
    "angle": "tilt",
    "angled": "tilt",
    "turn": "turn",
    "turned": "turn",
    "rotate": "turn",
    "rotated": "turn",
    "dark": "dark",
    "darker": "dark",
    "darkened": "dark",
    "light": "light",
    "lighter": "light",
    "lightened": "light",
    "star": "star",
    "stars": "star",
    "starry": "star",
    "painted": "painting",
    "paint": "painting",
}

STATE_STOP_WORDS = {
    "a",
    "an",
    "and",
    "appears",
    "are",
    "artistic",
    "attire",
    "background",
    "body",
    "color",
    "current",
    "currently",
    "dress",
    "hand",
    "has",
    "having",
    "in",
    "is",
    "main",
    "of",
    "outfit",
    "person",
    "pose",
    "position",
    "posture",
    "scene",
    "skin",
    "style",
    "the",
    "to",
    "tone",
    "visible",
    "woman",
}

STATE_OPPOSITES = {
    "lower": {"raise"},
    "raise": {"lower"},
    "upright": {"lean"},
    "lean": {"upright"},
    "left": {"right"},
    "right": {"left"},
    "dark": {"light"},
    "light": {"dark"},
}


def _canonical_state_words(text: object) -> List[str]:
    return [
        STATE_TOKEN_ALIASES.get(word, word)
        for word in _normalized_words(text)
        if word not in STATE_STOP_WORDS
    ]


def _state_text(evidence: Dict, subgoal_index: int, edit_type: str) -> str:
    if edit_type == "style":
        return " ".join(
            value
            for value in (evidence.get("style_label"), evidence.get("scene_description"))
            if value
        )
    if edit_type == "background":
        fact = _fact(evidence, subgoal_index, "OBJECT_A")
        return " ".join(
            value
            for value in (
                fact.get("attribute_value") if fact else "",
                evidence.get("scene_label"),
                evidence.get("scene_description"),
            )
            if value
        )
    preferred_roles = ("PART", "OBJECT_A") if edit_type == "motion" else ("OBJECT_A",)
    fact = next(
        (_fact(evidence, subgoal_index, role) for role in preferred_roles if _fact(evidence, subgoal_index, role)),
        None,
    )
    if not fact:
        return ""
    return " ".join(
        value for value in (fact.get("attribute_value"), fact.get("pose")) if value
    )


def _state_matches_goal(goal: object, state: object, edit_type: str) -> str:
    goal_words = _canonical_state_words(goal)
    state_words = _canonical_state_words(state)
    if not goal_words or not state_words:
        return UNKNOWN
    overlap = set(goal_words) & set(state_words)
    if edit_type in {"color", "motion"}:
        return TRUE if overlap else UNKNOWN
    required = 1 if len(set(goal_words)) == 1 else 2
    return TRUE if len(overlap) >= required else UNKNOWN


def _review_goal_match(
    subgoal: Dict, review_source: Dict, review_target: Dict, review_change: str
) -> str:
    """Derive target alignment from independent state facts, never a model verdict."""

    if review_change == FALSE:
        return review_change
    edit_type = canonical_edit_type(subgoal.get("edit_type"))
    if edit_type in {"add", "remove", "replace"}:
        return TRUE
    goal = subgoal.get("attribute") or subgoal.get("object_b")
    if not goal:
        return UNKNOWN
    source_state = _state_text(
        review_source, int(subgoal.get("subgoal_index", 0)), edit_type
    )
    target_state = _state_text(
        review_target, int(subgoal.get("subgoal_index", 0)), edit_type
    )
    target_match = _state_matches_goal(goal, target_state, edit_type)
    if target_match == TRUE:
        return TRUE
    goal_words = set(_canonical_state_words(goal))
    source_words = set(_canonical_state_words(source_state))
    target_words = set(_canonical_state_words(target_state))
    opposites = set().union(*(STATE_OPPOSITES.get(word, set()) for word in goal_words))
    if review_change == TRUE and opposites & source_words and not opposites & target_words:
        # Example: source is forward-leaning, target is merely described as
        # seated.  The observed removal of the opposite state supports the
        # requested straightening without asking the model for that conclusion.
        return TRUE
    if opposites & target_words:
        return FALSE
    if _state_matches_goal(goal, source_state, edit_type) == TRUE:
        return FALSE
    return UNKNOWN


def _subgoal_change(subgoal: Dict, source: Dict, target: Dict) -> str:
    index = int(subgoal.get("subgoal_index", 0))
    edit_type = canonical_edit_type(subgoal.get("edit_type"))
    source_a, target_a = _fact(source, index, "OBJECT_A"), _fact(target, index, "OBJECT_A")
    source_b, target_b = _fact(source, index, "OBJECT_B"), _fact(target, index, "OBJECT_B")

    if edit_type == "add":
        source_present, target_present = _present(source_b), _present(target_b)
        if source_present == FALSE and target_present == TRUE:
            return TRUE
        if target_present == FALSE or (source_present == FALSE and target_present == FALSE):
            return FALSE
        if source_present == TRUE and target_present == TRUE:
            source_count = source_b.get("count") if source_b else None
            target_count = target_b.get("count") if target_b else None
            if source_count is not None and target_count is not None:
                expected = subgoal.get("count")
                delta = target_count - source_count
                if expected is not None:
                    return TRUE if delta >= int(expected) else FALSE
                return TRUE if delta > 0 else FALSE
        return UNKNOWN

    if edit_type == "remove":
        return _tri_and((_present(source_a), _tri_not(_present(target_a))))

    if edit_type == "replace":
        source_a_present = _present(source_a)
        source_b_present = _present(source_b)
        target_a_present = _present(target_a)
        target_b_present = _present(target_b)
        a_removed = _tri_not(target_a_present)
        if source_a_present == TRUE and target_a_present == TRUE:
            source_count = source_a.get("count") if source_a else None
            target_count = target_a.get("count") if target_a else None
            if source_count is not None and target_count is not None:
                a_removed = TRUE if target_count < source_count else FALSE
        b_added = target_b_present
        if source_b_present == TRUE and target_b_present == TRUE:
            source_count = source_b.get("count") if source_b else None
            target_count = target_b.get("count") if target_b else None
            if source_count is not None and target_count is not None:
                b_added = TRUE if target_count > source_count else FALSE
        return _tri_and(
            (
                source_a_present,
                _tri_not(source_b_present),
                a_removed,
                b_added,
            )
        )

    if edit_type == "color":
        present = _tri_and((_present(source_a), _present(target_a)))
        if present != TRUE:
            return present
        return _different(
            source_a.get("attribute_value") if source_a else "",
            target_a.get("attribute_value") if target_a else "",
        )

    if edit_type == "motion":
        source_part = _fact(source, index, "PART") or source_a
        target_part = _fact(target, index, "PART") or target_a
        present = _tri_and((_present(source_part), _present(target_part)))
        if present != TRUE:
            return present
        return _different(
            source_part.get("pose") if source_part else "",
            target_part.get("pose") if target_part else "",
        )

    if edit_type == "background":
        if source_a and target_a:
            attribute_change = _different(
                source_a.get("attribute_value"), target_a.get("attribute_value")
            )
            if attribute_change != UNKNOWN:
                return attribute_change
        return _different(
            source.get("scene_label") or source.get("scene_description"),
            target.get("scene_label") or target.get("scene_description"),
        )

    if edit_type == "style":
        return _different(source.get("style_label"), target.get("style_label"))

    return UNKNOWN


def _yes_no_unknown(value: object) -> str:
    normalized = str(value or "").upper()
    if normalized == "YES":
        return TRUE
    if normalized == "NO":
        return FALSE
    return UNKNOWN


def _match_truth(value: object) -> str:
    normalized = str(value or "").upper()
    if normalized in {"MATCH", "PARTIAL"}:
        return TRUE
    if normalized in {"MISMATCH", "NOT_MENTIONED"}:
        return FALSE
    return UNKNOWN


def _all_confidences(
    slots: Dict, source: Dict, target: Dict, paired: Dict, text_match: Dict
) -> List[float]:
    values = [
        _confidence(slots.get("confidence")),
        _confidence(source.get("confidence")),
        _confidence(target.get("confidence")),
        _confidence(paired.get("confidence")),
        _confidence(text_match.get("confidence")),
    ]
    values.extend(_confidence(item.get("confidence")) for item in source.get("facts") or [])
    values.extend(_confidence(item.get("confidence")) for item in target.get("facts") or [])
    values.extend(
        _confidence(item.get("confidence")) for item in text_match.get("subgoal_matches") or []
    )
    return values


def adjudicate_evidence(
    slots: Dict,
    source: Dict,
    target: Dict,
    paired: Dict,
    text_match: Dict,
    confidence_threshold: float = 0.6,
    review: Optional[Dict] = None,
) -> Dict:
    """Combine model facts into the final code-owned verdict and decision."""

    status = slots.get("instruction_status", "UNJUDGEABLE")
    subgoals = slots.get("subgoals") or []
    predicates: Dict[str, str] = {}
    review_conflict = False

    if status == "NO_OP":
        predicates["instruction_actionable"] = FALSE
    elif status == "NORMAL" and subgoals:
        predicates["instruction_actionable"] = TRUE
    else:
        predicates["instruction_actionable"] = UNKNOWN

    review_source = (review or {}).get("source") or {}
    review_target = (review or {}).get("target") or {}
    has_review_states = bool(review_source and review_target)

    change_values: List[str] = []
    match_values: List[str] = []
    for index, subgoal in enumerate(subgoals):
        change = _subgoal_change(subgoal, source, target)
        reviewed = (
            _subgoal_change(subgoal, review_source, review_target)
            if has_review_states
            else UNKNOWN
        )
        review_match = (
            _review_goal_match(subgoal, review_source, review_target, reviewed)
            if has_review_states
            else UNKNOWN
        )
        if has_review_states:
            predicates[f"subgoal_{index}_review_change"] = reviewed
            predicates[f"subgoal_{index}_review_target_match"] = review_match
        if reviewed != UNKNOWN:
            if change == UNKNOWN:
                change = reviewed
            elif reviewed != change:
                # Step 5 now consists of two independent, focused single-image
                # observations.  Their code-computed transition can refine the
                # noisier full-image states for every edit type.
                change = reviewed
                predicates[f"subgoal_{index}_review_refined_change"] = TRUE
        matches = text_match.get("subgoal_matches") or []
        match_value = matches[index].get("match") if index < len(matches) else "NOT_MENTIONED"
        match = _match_truth(match_value)
        if review_match == TRUE and match_value in {"NOT_MENTIONED", "MISMATCH"}:
            # Step 3 is deliberately blind and can omit subtle changes.  A
            # targeted Step 5 source/target state comparison supplies the
            # missing mention without asking the model for a support verdict.
            match = TRUE
        elif review_match == FALSE and match == TRUE:
            review_conflict = True
        elif reviewed == FALSE and match == TRUE:
            review_conflict = True
        if has_review_states and change == UNKNOWN and reviewed == UNKNOWN and match == TRUE:
            # A matched instruction-blind difference is itself positive change
            # evidence.  Use it only when the focused single-image states do not
            # contradict it; this recovers cropped/occluded parts and uncountable
            # additions without reintroducing a model-authored support label.
            change = TRUE
            predicates[f"subgoal_{index}_review_blind_change_fallback"] = TRUE
        predicates[f"subgoal_{index}_change"] = change
        change_values.append(change)
        predicates[f"subgoal_{index}_blind_match"] = match
        match_values.append(match)

    predicates["change_happened"] = _tri_and(change_values)
    predicates["blind_description_matches"] = _tri_and(match_values)
    predicates["same_subject"] = _yes_no_unknown(paired.get("same_subject"))
    predicates["composition_preserved"] = _yes_no_unknown(paired.get("composition_preserved"))
    predicates["unrelated_regions_preserved"] = _yes_no_unknown(
        paired.get("unrelated_regions_preserved")
    )
    if paired.get("edit_scope") == "CONTROLLED_EDIT":
        predicates["not_global_regeneration"] = TRUE
    elif paired.get("edit_scope") == "GLOBAL_REGEN":
        predicates["not_global_regeneration"] = FALSE
    else:
        predicates["not_global_regeneration"] = UNKNOWN

    confidences = _all_confidences(slots, source, target, paired, text_match)
    nonzero_confidences = [value for value in confidences if value > 0]
    min_confidence = min(nonzero_confidences) if nonzero_confidences else 0.0
    predicates["confidence_sufficient"] = (
        TRUE if nonzero_confidences and min_confidence >= confidence_threshold else UNKNOWN
    )
    if review and _confidence(review.get("confidence")) >= confidence_threshold:
        predicates["confidence_sufficient"] = TRUE

    predicates["review_consistent"] = FALSE if review_conflict else TRUE

    edit_types = {canonical_edit_type(item.get("edit_type")) for item in subgoals}
    if edit_types == {"style"} and _tri_and(
        (
            predicates.get("same_subject", UNKNOWN),
            predicates.get("composition_preserved", UNKNOWN),
            predicates.get("unrelated_regions_preserved", UNKNOWN),
        )
    ) == TRUE:
        # A content-preserving whole-image style transfer is not the harmful
        # kind of regeneration that this predicate is meant to reject.
        predicates["not_global_regeneration"] = TRUE
    required = [
        "instruction_actionable",
        "change_happened",
        "blind_description_matches",
        "same_subject",
        "composition_preserved",
        "not_global_regeneration",
        "confidence_sufficient",
        "review_consistent",
    ]
    if not edit_types.issubset({"background", "style"}):
        required.append("unrelated_regions_preserved")

    failed = [key for key in required if predicates.get(key) == FALSE]
    unresolved = [key for key in required if predicates.get(key) == UNKNOWN]
    subgoal_combined = [
        _tri_and(
            (
                predicates.get(f"subgoal_{index}_change", UNKNOWN),
                predicates.get(f"subgoal_{index}_blind_match", UNKNOWN),
            )
        )
        for index in range(len(subgoals))
    ]
    mixed_compound = (
        len(subgoal_combined) > 1
        and TRUE in subgoal_combined
        and (FALSE in subgoal_combined or UNKNOWN in subgoal_combined)
    )
    mixed_compound = mixed_compound or (
        len(match_values) > 1
        and TRUE in match_values
        and (FALSE in match_values or UNKNOWN in match_values)
    )
    evidence_conflict = any(
        predicates.get(f"subgoal_{index}_change") in {TRUE, FALSE}
        and predicates.get(f"subgoal_{index}_blind_match") in {TRUE, FALSE}
        and predicates.get(f"subgoal_{index}_change")
        != predicates.get(f"subgoal_{index}_blind_match")
        for index in range(len(subgoals))
    )
    conflict_is_reviewable = evidence_conflict
    hard_failed = [
        key
        for key in failed
        if key not in {"change_happened", "blind_description_matches"}
    ]
    if failed and not hard_failed and review is None and (
        mixed_compound or conflict_is_reviewable
    ):
        verdict, decision = "UNSURE", "drop"
    elif failed:
        verdict, decision = "FAIL", "drop"
    elif not unresolved:
        verdict, decision = "PASS", "keep"
    else:
        verdict, decision = "UNSURE", "drop"

    if verdict == "UNSURE" and failed:
        reason = "Conflicting/partial evidence needs review: " + ", ".join(failed) + "."
    elif failed:
        reason = "Code predicates failed: " + ", ".join(failed) + "."
    elif unresolved:
        reason = "Code predicates unresolved: " + ", ".join(unresolved) + "."
    else:
        reason = "All required evidence predicates passed."

    review_reasons = list(unresolved)
    if evidence_conflict:
        review_reasons.append("evidence_conflict")
    if mixed_compound:
        review_reasons.append("compound_partial")
    if review_conflict:
        review_reasons.append("review_answer_flip")
    if predicates.get("confidence_sufficient") == UNKNOWN:
        review_reasons.append("low_confidence")

    change_truth = predicates.get("change_happened", UNKNOWN)
    match_truth = predicates.get("blind_description_matches", UNKNOWN)
    if change_truth == FALSE:
        failure_mode = "NO_RELEVANT_CHANGE"
    elif match_truth == FALSE:
        failure_mode = "WRONG_TARGET"
    elif verdict != "PASS":
        failure_mode = "AMBIGUOUS"
    else:
        failure_mode = "OTHER"

    return {
        "verdict": verdict,
        "decision": decision,
        "confidence": min_confidence,
        "reason": reason,
        "reason_codes": failed or unresolved,
        "predicates": predicates,
        "change_presence": TRUE if change_truth == TRUE else FALSE if change_truth == FALSE else UNKNOWN,
        "instruction_achievement": (
            "YES" if match_truth == TRUE else "NO" if match_truth == FALSE else "UNCLEAR"
        ),
        "failure_mode": failure_mode,
        "review_needed": verdict == "UNSURE" and review is None,
        "review_reasons": sorted(set(review_reasons)),
    }
