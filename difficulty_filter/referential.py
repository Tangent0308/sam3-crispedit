"""Policy for same-class, subset-referential image-edit filtering.

This module intentionally contains no model initialization or dataset writes.  The
CLI in ``scripts/filter_referential_edits.py`` supplies the Qwen and SAM3 evidence;
the functions here keep screening, parsing, de-duplication, and fusion testable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MLLM_PROMPT_VERSION = "qwen35_referential_subset_audit_v4"
SAM_COUNT_POLICY_VERSION = "sam3_open_vocab_instance_count_v2_edit_overlap"
FILTER_POLICY_VERSION = "same_class_subset_fusion_v3"

# These tasks are global by construction or overwhelmingly global in the two
# source datasets.  ``mask_mode != regions`` is screened independently, so a
# newly introduced global edit type also fails closed when labeling routed it to
# a full-image/protected-foreground mask.
GLOBAL_EDIT_TYPES = frozenset(
    {
        "background_replacement",
        "style_transfer",
        "tone_adjustment",
        "visual_beautification",
        "viewpoint_transformation",
        "part_extraction",
    }
)

# A pure addition may use a spatial reference, but it does not edit a selected
# source-image instance. Mixed add+change requests live under compositional
# editing in ScaleEdit and remain eligible for semantic review.
NON_SUBSET_EDIT_TYPES = frozenset({"object_addition"})

_CUE_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "spatial",
        re.compile(
            r"\b(?:left|right|top|bottom|upper|lower|middle|center|front|back|"
            r"foreground|background|nearest|closest|farthest|behind|beside|"
            r"next to|between|above|below|under|over|on the .*? side)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ordinal",
        re.compile(
            r"\b(?:first|second|third|fourth|fifth|last|leftmost|rightmost|"
            r"topmost|bottommost|\d+(?:st|nd|rd|th))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cardinality",
        re.compile(
            r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|single|"
            r"pair|both|another|each|several|some|\d+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "relation",
        re.compile(
            r"\b(?:holding|wearing|carrying|attached to|inside|outside|on the|"
            r"in the|at the|belonging to|next to|adjacent to|surrounding)\b",
            re.IGNORECASE,
        ),
    ),
)

_ALLOWED_JUDGMENTS = {"yes", "likely", "no"}
_ALLOWED_SUBSET = {"yes", "likely", "no", "uncertain"}
_ALLOWED_CUES = {
    "spatial",
    "ordinal",
    "cardinality",
    "relation",
    "appearance",
    "identity",
    "none",
}


@dataclass(frozen=True)
class ScreenResult:
    eligible: bool
    reason: str


def deterministic_screen(edit_type: str, mask_mode: str) -> ScreenResult:
    """Discard known global edits before either expensive model is called."""

    normalized_type = str(edit_type or "").strip().lower()
    normalized_mode = str(mask_mode or "").strip().lower()
    if normalized_mode != "regions":
        return ScreenResult(False, f"non_region_mask_mode:{normalized_mode or 'missing'}")
    if normalized_type in GLOBAL_EDIT_TYPES:
        return ScreenResult(False, f"global_edit_type:{normalized_type}")
    if normalized_type in NON_SUBSET_EDIT_TYPES:
        return ScreenResult(False, f"no_edited_source_subset:{normalized_type}")
    return ScreenResult(True, "eligible_local_edit")


def instruction_cues(instruction: str) -> List[str]:
    """Return cheap lexical cues used only for sampling and audit, not fusion."""

    text = str(instruction or "")
    return [name for name, pattern in _CUE_PATTERNS if pattern.search(text)]


def extract_grounding_refs(ground_json: str, limit: int = 8) -> List[str]:
    """Extract source-first semantic phrases from either labeling contract."""

    try:
        payload = json.loads(str(ground_json or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    refs: List[str] = []
    for side in ("source", "target", "protected_foreground"):
        values: Any
        boxes = payload.get("boxes")
        if isinstance(boxes, Mapping):
            values = boxes.get(side, [])
        else:
            values = payload.get(side, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            ref = str(item.get("ref", "")).strip()
            if ref and ref not in refs:
                refs.append(ref)
                if len(refs) >= limit:
                    return refs
    return refs


def build_mllm_prompt(
    instruction: str,
    edit_type: str,
    metadata_refs: Sequence[str],
) -> str:
    """Build the sole MLLM request used by the filter.

    The model supplies both the semantic subset judgment and a category-only SAM
    query.  The category must omit disambiguating attributes because SAM3 is
    meant to count *all* same-class instances, not just the selected subset.
    """

    hints = json.dumps(list(metadata_refs), ensure_ascii=False)
    return f"""You are screening an image-edit training example for a specific difficult task.

You see ONLY the source image. The edit instruction is:
{instruction}

Dataset edit type: {edit_type}
Optional phrases from prior mask-label metadata: {hints}

Target definition:
- The source image visibly contains at least TWO distinct, countable instances of one semantic
  object class.
- The instruction edits only ONE or A PROPER SUBSET of those same-class instances, rather than all
  of them.
- The selected subset is identified by fine-grained reference language such as position,
  left/right/top/bottom, ordinal, number, relation to another object/person, appearance, or identity.

Reject these as `no`: a single visible instance; editing every instance of the class; global style,
background, filter, or whole-image edits; selecting a part when there are not multiple peer objects;
purely ADDING a new instance even when peer objects already exist; and instructions whose target
cannot be determined from the source image. Be recall-oriented: use
`likely` when the image plausibly satisfies the definition but small/occluded instances or wording
make the count uncertain.

Before counting, copy into `edited_subject_phrase` the shortest exact phrase from the instruction
that names an EXISTING SOURCE entity that is actually edited, removed, moved, or replaced. Do this
before considering how many objects are visible. If there are multiple edits, choose the edited
entity whose same-class subset status you are evaluating.

Derive `object_category` ONLY from `edited_subject_phrase`. It must be a short singular common-noun
visual class for counting ALL peer instances with open-vocabulary segmentation (examples:
`backpack`, `plate`, `person`, `car`). Never choose an unchanged reference anchor, container,
support, destination, or nearby object merely because the instruction uses it to locate the edit.
For example, in "move her right hand to the surface of a sink", `edited_subject_phrase` is `right
hand` and `object_category` is `hand`, never `sink`. If only one edited-subject object is visible,
answer `no` even when several reference anchors are visible. Remove color, size, ownership, state,
number, ordinal, and location. Do not output an edited subpart if the reference selects among whole
objects. Metadata phrases are hints only; verify everything against the source image and instruction.

Return exactly one JSON object and no markdown:
{{
  "edited_subject_phrase": "exact phrase copied from the instruction or empty string",
  "object_category": "singular category or empty string",
  "visible_same_class_count": 0,
  "selected_instance_count": null,
  "subset_relation": "yes|likely|no|uncertain",
  "reference_cues": ["spatial|ordinal|cardinality|relation|appearance|identity|none"],
  "fine_grained_referential": "yes|likely|no",
  "confidence": 0.0,
  "reason": "one short evidence-based sentence"
}}
"""


def _json_object_candidates(text: str) -> Iterable[str]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    yield stripped
    starts = [index for index, char in enumerate(stripped) if char == "{"]
    for start in starts:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(stripped)):
            char = stripped[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield stripped[start : index + 1]
                    break


def parse_mllm_response(text: str) -> Dict[str, Any]:
    """Parse and strictly normalize the Qwen JSON response."""

    payload: Optional[Dict[str, Any]] = None
    last_error = "no JSON object"
    seen = set()
    for candidate in _json_object_candidates(text):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            value = json.loads(candidate)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            continue
        if isinstance(value, dict):
            payload = value
            break
    if payload is None:
        raise ValueError(f"invalid MLLM JSON: {last_error}")

    subject_phrase = str(payload.get("edited_subject_phrase", "")).strip()
    if "edited_subject_phrase" not in payload:
        raise ValueError("missing edited_subject_phrase")
    category = str(payload.get("object_category", "")).strip().lower()
    category = re.sub(r"\s+", " ", category)
    judgment = str(payload.get("fine_grained_referential", "")).strip().lower()
    subset = str(payload.get("subset_relation", "")).strip().lower()
    if judgment not in _ALLOWED_JUDGMENTS:
        raise ValueError(f"invalid fine_grained_referential={judgment!r}")
    if subset not in _ALLOWED_SUBSET:
        raise ValueError(f"invalid subset_relation={subset!r}")

    try:
        visible_count = int(payload.get("visible_same_class_count", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("visible_same_class_count must be an integer") from exc
    if visible_count < 0:
        raise ValueError("visible_same_class_count must be non-negative")
    selected_raw = payload.get("selected_instance_count")
    selected_count = None if selected_raw is None else int(selected_raw)
    if selected_count is not None and selected_count < 0:
        raise ValueError("selected_instance_count must be non-negative or null")
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    confidence = min(1.0, max(0.0, confidence))

    raw_cues = payload.get("reference_cues", [])
    if not isinstance(raw_cues, list):
        raise ValueError("reference_cues must be an array")
    cues = []
    for value in raw_cues:
        cue = str(value).strip().lower()
        if cue in _ALLOWED_CUES and cue not in cues:
            cues.append(cue)
    if not cues:
        cues = ["none"]

    return {
        "edited_subject_phrase": subject_phrase,
        "object_category": category,
        "visible_same_class_count": visible_count,
        "selected_instance_count": selected_count,
        "subset_relation": subset,
        "reference_cues": cues,
        "fine_grained_referential": judgment,
        "confidence": confidence,
        "reason": str(payload.get("reason", "")).strip()[:1000],
    }


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(intersection / max(area_a + area_b - intersection, 1e-8))


def _box_intersection(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )


def deduplicate_sam_instances(
    candidates: Sequence[Mapping[str, Any]],
    mask_iou_threshold: float = 0.80,
    containment_threshold: float = 0.92,
    box_iou_threshold: float = 0.90,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Suppress duplicate SAM proposals while retaining an auditable reject list."""

    ordered = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for original in ordered:
        candidate = dict(original)
        mask = np.asarray(candidate.pop("_mask"), dtype=bool)
        duplicate_of = None
        duplicate_metrics: Dict[str, float] = {}
        for kept_index, existing in enumerate(kept):
            # Instance proposals with disjoint boxes cannot have overlapping
            # masks. This guard avoids allocating million-pixel boolean arrays
            # for every pair when a broad prompt (for example ``text`` or
            # ``person``) produces dozens of legitimate instances.
            overlap_area = _box_intersection(
                candidate["bbox_xyxy"], existing["bbox_xyxy"]
            )
            if overlap_area <= 0.0:
                continue
            existing_mask = np.asarray(existing["_mask"], dtype=bool)
            intersection = int(np.logical_and(mask, existing_mask).sum())
            union = int(np.logical_or(mask, existing_mask).sum())
            minimum = min(int(mask.sum()), int(existing_mask.sum()))
            mask_iou = intersection / max(union, 1)
            containment = intersection / max(minimum, 1)
            overlap = box_iou(candidate["bbox_xyxy"], existing["bbox_xyxy"])
            if (
                mask_iou >= mask_iou_threshold
                or containment >= containment_threshold
                or overlap >= box_iou_threshold
            ):
                duplicate_of = kept_index
                duplicate_metrics = {
                    "mask_iou": round(mask_iou, 6),
                    "containment": round(containment, 6),
                    "box_iou": round(overlap, 6),
                }
                break
        if duplicate_of is None:
            candidate["_mask"] = mask
            kept.append(candidate)
        else:
            candidate["duplicate_of"] = duplicate_of
            candidate["duplicate_metrics"] = duplicate_metrics
            rejected.append(candidate)
    return kept, rejected


def fuse_evidence(
    *,
    deterministic_eligible: bool,
    deterministic_reason: str,
    mllm: Optional[Mapping[str, Any]],
    sam_count: Optional[int],
    sam_selected_count: Optional[int] = None,
    all_selected_fraction: float = 0.90,
) -> Dict[str, Any]:
    """Fuse independent semantic and instance-count evidence.

    ``keep`` is high-confidence automatic acceptance. ``review`` is deliberately
    recall-oriented and is included by ``loose_keep``; downstream users can use
    only ``keep`` when they prefer precision.  Excluded task types never invoke
    either model and always remain ``drop``.
    """

    if not 0.0 < all_selected_fraction <= 1.0:
        raise ValueError("all_selected_fraction must be in (0, 1]")

    if not deterministic_eligible:
        return {
            "decision": "drop",
            "loose_keep": False,
            "reason": deterministic_reason,
        }
    if mllm is None:
        return {"decision": "drop", "loose_keep": False, "reason": "missing_mllm_evidence"}
    if sam_count is None:
        return {"decision": "review", "loose_keep": True, "reason": "missing_sam_evidence"}

    judgment = str(mllm.get("fine_grained_referential", "no"))
    subset = str(mllm.get("subset_relation", "uncertain"))
    category = str(mllm.get("object_category", "")).strip()
    visible_count = int(mllm.get("visible_same_class_count", 0) or 0)
    if not category:
        return {"decision": "drop", "loose_keep": False, "reason": "empty_object_category"}

    semantic_strong = judgment == "yes" and subset in {"yes", "likely"}
    semantic_plausible = judgment in {"yes", "likely"} and subset != "no"
    if sam_count >= 2 and sam_selected_count is not None:
        selected_fraction = sam_selected_count / sam_count
        if selected_fraction >= all_selected_fraction:
            return {
                "decision": "drop",
                "loose_keep": False,
                "reason": "edit_mask_covers_nearly_all_sam_instances",
            }
        if sam_selected_count == 0:
            if semantic_strong and visible_count >= 2:
                return {
                    "decision": "review",
                    "loose_keep": True,
                    "reason": "mllm_subset_but_edit_mask_matches_no_sam_instance",
                }
            return {
                "decision": "drop",
                "loose_keep": False,
                "reason": "edit_mask_matches_no_sam_instance",
            }
    if judgment == "no" or subset == "no":
        # The MLLM is intentionally conservative about ambiguous instance
        # boundaries. Preserve a disagreement for manual review only when SAM3
        # independently finds multiple peers and the existing training mask
        # overlaps a non-empty proper subset. This never enters the strict
        # selected manifest.
        if (
            sam_count >= 2
            and sam_selected_count is not None
            and 0 < sam_selected_count < sam_count * all_selected_fraction
        ):
            return {
                "decision": "review",
                "loose_keep": True,
                "reason": "mllm_reject_but_sam_mask_subset",
            }
        return {
            "decision": "drop",
            "loose_keep": False,
            "reason": f"mllm_reject:{judgment}:subset_{subset}",
        }
    if sam_count >= 2 and semantic_strong:
        return {
            "decision": "keep",
            "loose_keep": True,
            "reason": (
                "mllm_subset_and_sam_mask_subset"
                if sam_selected_count is not None
                else "mllm_subset_and_sam_multiple"
            ),
        }
    if sam_count >= 2 and semantic_plausible:
        return {
            "decision": "review",
            "loose_keep": True,
            "reason": "sam_multiple_mllm_likely",
        }
    if semantic_strong and visible_count >= 2:
        return {
            "decision": "review",
            "loose_keep": True,
            "reason": "mllm_multiple_but_sam_undercount",
        }
    return {
        "decision": "drop",
        "loose_keep": False,
        "reason": f"insufficient_multiple_instance_evidence:sam_{sam_count}:mllm_{visible_count}",
    }


def stable_priority(sample_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
