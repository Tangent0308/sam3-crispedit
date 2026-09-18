"""One-call, type-aware image-pair quality policy for CrispEdit."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from PIL import Image

from crispedit.prefilter.policy import canonical_edit_type


PREFILTER_METHOD = "pair_quality"
EVIDENCE_SCHEMA = "pair_quality_dimensions"
PROMPT_VERSION = "crispedit_pair_quality"

QUALITY_DIMENSIONS = (
    "source_reference",
    "instruction_meaningfulness",
    "edit_completion",
    "source_integrity",
    "target_integrity",
    "content_preservation",
)
QUALITY_STATUSES = {"PASS", "FAIL", "UNSURE"}
CHANGE_STATUSES = {"CLEAR", "NONE", "UNSURE"}
MATCH_STATUSES = {"PASS", "FAIL", "UNSURE"}
REASON_CODES = {
    "MISSING_SOURCE_REFERENT",
    "AMBIGUOUS_SOURCE_REFERENT",
    "NO_OP_ALREADY_SATISFIED",
    "UNJUDGEABLE_INSTRUCTION",
    "EDIT_NOT_COMPLETED",
    "WRONG_REFERENT_EDITED",
    "WRONG_ATTRIBUTE_OR_ACTION",
    "INCOMPLETE_COMPOUND_EDIT",
    "INCOMPLETE_OBJECT_REMOVAL",
    "COUNT_OR_LOCATION_MISMATCH",
    "SOURCE_GENERATION_ARTIFACT",
    "TARGET_GENERATION_ARTIFACT",
    "IMPLAUSIBLE_PLACEMENT_OR_GEOMETRY",
    "UNRELATED_CONTENT_CHANGED",
    "PAIR_MISALIGNED",
    "INSUFFICIENT_VISUAL_EVIDENCE",
    "OTHER",
}
REASON_DIMENSIONS = {
    "MISSING_SOURCE_REFERENT": ("source_reference",),
    "AMBIGUOUS_SOURCE_REFERENT": ("source_reference",),
    "NO_OP_ALREADY_SATISFIED": ("instruction_meaningfulness",),
    "UNJUDGEABLE_INSTRUCTION": ("instruction_meaningfulness",),
    "EDIT_NOT_COMPLETED": ("edit_completion",),
    "WRONG_REFERENT_EDITED": ("edit_completion",),
    "WRONG_ATTRIBUTE_OR_ACTION": ("edit_completion",),
    "INCOMPLETE_COMPOUND_EDIT": ("edit_completion",),
    "INCOMPLETE_OBJECT_REMOVAL": ("edit_completion",),
    "COUNT_OR_LOCATION_MISMATCH": (
        "source_reference",
        "instruction_meaningfulness",
        "edit_completion",
    ),
    "SOURCE_GENERATION_ARTIFACT": ("source_integrity",),
    "TARGET_GENERATION_ARTIFACT": ("target_integrity",),
    "IMPLAUSIBLE_PLACEMENT_OR_GEOMETRY": (
        "edit_completion",
        "target_integrity",
    ),
    "UNRELATED_CONTENT_CHANGED": ("content_preservation",),
    "PAIR_MISALIGNED": ("content_preservation",),
}


TYPE_GUIDANCE = {
    "add": (
        "The requested new object should be absent at the requested location/count in "
        "SOURCE and visibly present in TARGET. Existing similar objects elsewhere do not "
        "make an addition a no-op. The added object may become the main subject; that alone "
        "is not unrelated change. Natural occlusion, contact shadows, and local relighting "
        "around the insertion are allowed."
    ),
    "remove": (
        "The requested object must be visible in SOURCE and absent in TARGET, including "
        "integral contents or dependent traces. Removing the main subject is allowed when "
        "the instruction asks for it. Judge preservation only outside the removed region "
        "and its necessary inpainting."
    ),
    "replace": (
        "The old object must be visible in SOURCE and the requested replacement must be "
        "visible in TARGET in the intended role/location. A clear paired old-to-new visual "
        "transition is sufficient; do not require a separate statement that the new object "
        "was absent in SOURCE. Changing the main subject's category, identity, silhouette, "
        "or geometry is expected when that subject is the replacement target."
    ),
    "color": (
        "The requested object/part must be present and its visible target color or material "
        "must change as requested. If SOURCE already has the requested categorical color, "
        "minor darkening, contrast, lighting, or highlight drift is a no-op rather than a "
        "successful recoloring. A subtle edit is valid only when the before/after color "
        "difference is directly identifiable. Preserve semantics and geometry outside the "
        "recolored area."
    ),
    "motion": (
        "The requested person/object/part must visibly change pose, orientation, action, or "
        "relative position as requested. Changes to directly interacting objects are allowed. "
        "Before passing, identify the exact requested object or body part in SOURCE and state "
        "its concrete before/after geometry or relation. Do not invent an instructed object "
        "that is not visible. Framing, rescaling, camera drift, or two effectively identical "
        "poses do not demonstrate motion, and a different action does not satisfy the edit."
    ),
    "background": (
        "The background, environment, ground, sky, weather, time of day, and their lighting "
        "may all change. Do not call those expected changes composition failure or global "
        "regeneration. Require the independent foreground subject's identity, pose, rough "
        "scale, and placement to remain usable unless the instruction also changes them."
    ),
    "style": (
        "The entire rendering medium, palette, texture, shading, and local contours may "
        "change. This is not unrelated change or global regeneration by itself. Require the "
        "same semantic content and roughly corresponding spatial layout, not pixel alignment "
        "or an unchanged photographic appearance."
    ),
}


QUALITY_PROMPT = """You are a strict but fair quality auditor for a CrispEdit image-editing dataset.

Image 1 is the SOURCE before editing. Image 2 is the TARGET edited result.
Dataset edit type: __EDIT_TYPE__
Instruction: __INSTRUCTION__

Judge the actual pixels in the two images jointly. Do not assume that the edit succeeded
merely because TARGET looks plausible or already resembles the instruction. First inspect
SOURCE without trusting the claimed outcome, then inspect TARGET, and identify the concrete
before-to-after difference. The goal is to reject concrete no-ops, wrong edits, damaged
results, and unrelated semantic changes while retaining visibly valid edits. Do not reject
an edit merely because it is large, becomes visually salient, changes the requested main
object, or comes from a generative editor. Do not demand exact pixel alignment.

Type-specific intended edit scope and completion rule:
__TYPE_GUIDANCE__

Before judging the six dimensions, fill edit_observation. source_state and target_state must
each directly describe the requested object's/property's/action's visible state in that
image, including requested noun, count, location, and relation when applicable. If the exact
requested object is absent, say so; never infer it from the instruction. visible_change is:
- CLEAR only when a concrete requested before-to-after difference can be named;
- NONE when the requested state is effectively the same, already satisfied, or only the
  framing/resampling/lighting changed;
- UNSURE only when resolution or ambiguity prevents comparison.
instruction_match is PASS only when that concrete difference realizes every requested
subgoal; otherwise use FAIL or UNSURE. A statement such as "the pose/color changed" without
describing how SOURCE and TARGET differ is not sufficient evidence.

Judge all six dimensions independently:

1. source_reference: For remove/replace/color/motion, the exact source referent and any
   stated location/count/relation must be present in SOURCE. For add, validate the scene or
   location anchor rather than requiring the new object to exist. For background/style,
   SOURCE only needs to be a coherent identifiable basis for the edit.
2. instruction_meaningfulness: The requested end state must not already be fully satisfied
   in SOURCE. Respect location and count: a similar object elsewhere does not make a
   location-specific add a no-op. The instruction must define a visually checkable change.
3. edit_completion: TARGET must visibly realize every requested subgoal on the correct
   referent, including requested attribute/action/location/count. A clear paired transition
   is valid evidence even if one image alone would be ambiguous.
4. source_integrity: Fail only for a conspicuous source defect that makes the training pair
   unusable, such as severe smearing, malformed structure, duplicated fragments, or major
   corruption. Ordinary synthetic aesthetics are not a defect.
5. target_integrity: The edited result must be coherent, without obvious holes, remnants,
   malformed anatomy/geometry, floating fragments, or severe insertion artifacts.
6. content_preservation: Judge only outside the type-specific intended edit scope above.
   Expected changes inside that scope must never be counted as unrelated changes. Ignore
   harmless resampling, texture, lighting, and small generative drift. Fail only for concrete,
   substantial unrelated semantic or geometric changes, or pair misalignment.

Use PASS when a dimension is visibly satisfied, FAIL when concrete pixels show a defect,
and UNSURE only when the pixels genuinely cannot resolve it. When the requested transition
is genuinely visible and no concrete evidence contradicts it, use PASS rather than UNSURE.
Do not invent hidden states.
Return exactly one JSON object and no markdown:
{
  "edit_observation": {
    "source_state": "requested visible state in SOURCE",
    "target_state": "requested visible state in TARGET",
    "visible_change": "CLEAR|NONE|UNSURE",
    "instruction_match": "PASS|FAIL|UNSURE"
  },
  "source_reference": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "instruction_meaningfulness": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "edit_completion": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "source_integrity": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "target_integrity": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "content_preservation": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "reason_codes": ["zero or more allowed codes"],
  "summary": "one concise explanation grounded in both images",
  "confidence": 0.0
}

Allowed reason codes:
MISSING_SOURCE_REFERENT, AMBIGUOUS_SOURCE_REFERENT, NO_OP_ALREADY_SATISFIED,
UNJUDGEABLE_INSTRUCTION, EDIT_NOT_COMPLETED, WRONG_REFERENT_EDITED,
WRONG_ATTRIBUTE_OR_ACTION, INCOMPLETE_COMPOUND_EDIT, INCOMPLETE_OBJECT_REMOVAL,
COUNT_OR_LOCATION_MISMATCH, SOURCE_GENERATION_ARTIFACT, TARGET_GENERATION_ARTIFACT,
IMPLAUSIBLE_PLACEMENT_OR_GEOMETRY, UNRELATED_CONTENT_CHANGED, PAIR_MISALIGNED,
INSUFFICIENT_VISUAL_EVIDENCE, OTHER.
"""


JSON_CORRECTION_PROMPT = """The previous answer was not a valid complete JSON object.
Return the same assessment using exactly the requested schema. Include edit_observation,
all six dimensions, reason_codes, summary, and confidence. Output JSON only."""


def type_guidance(raw_type: object) -> str:
    edit_type = canonical_edit_type(raw_type)
    return TYPE_GUIDANCE.get(
        edit_type,
        (
            "Use the instruction itself to define the intended edit scope and judge only "
            "visible facts."
        ),
    )


def build_quality_prompt(raw_type: object, instruction: object) -> str:
    edit_type = canonical_edit_type(raw_type)
    return (
        QUALITY_PROMPT.replace("__EDIT_TYPE__", edit_type)
        .replace("__INSTRUCTION__", str(instruction or ""))
        .replace("__TYPE_GUIDANCE__", type_guidance(edit_type))
    )


def build_quality_conversation(
    source: Image.Image,
    target: Image.Image,
    raw_type: object,
    instruction: object,
) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 (SOURCE before editing):"},
                {"type": "image", "image": source.convert("RGB")},
                {"type": "text", "text": "Image 2 (TARGET edited result):"},
                {"type": "image", "image": target.convert("RGB")},
                {
                    "type": "text",
                    "text": build_quality_prompt(raw_type, instruction),
                },
            ],
        }
    ]


def extract_json_object(text: object) -> Dict[str, Any]:
    value = str(text or "").strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", value):
            try:
                parsed, _ = decoder.raw_decode(value[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                break
        else:
            raise ValueError("no complete JSON object in model response") from direct_error
    if not isinstance(parsed, dict):
        raise ValueError("quality response must be a JSON object")
    return parsed


def normalize_quality_assessment(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dimension evidence and derive the final code-owned verdict."""

    observation = payload.get("edit_observation")
    if not isinstance(observation, dict):
        raise ValueError("missing edit_observation")
    source_state = re.sub(
        r"\s+", " ", str(observation.get("source_state", ""))
    ).strip()
    target_state = re.sub(
        r"\s+", " ", str(observation.get("target_state", ""))
    ).strip()
    if not source_state or not target_state:
        raise ValueError("edit_observation requires source_state and target_state")
    visible_change = str(observation.get("visible_change", "")).strip().upper()
    instruction_match = str(observation.get("instruction_match", "")).strip().upper()
    if visible_change not in CHANGE_STATUSES:
        raise ValueError(f"invalid visible_change status: {visible_change!r}")
    if instruction_match not in MATCH_STATUSES:
        raise ValueError(f"invalid instruction_match status: {instruction_match!r}")

    result: Dict[str, Any] = {
        "edit_observation": {
            "source_state": source_state,
            "target_state": target_state,
            "visible_change": visible_change,
            "instruction_match": instruction_match,
        }
    }
    statuses = []
    for name in QUALITY_DIMENSIONS:
        dimension = payload.get(name)
        if not isinstance(dimension, dict):
            raise ValueError(f"missing quality dimension: {name}")
        status = str(dimension.get("status", "")).strip().upper()
        if status not in QUALITY_STATUSES:
            raise ValueError(f"invalid {name} status: {status!r}")
        evidence = re.sub(r"\s+", " ", str(dimension.get("evidence", ""))).strip()
        if not evidence:
            raise ValueError(f"missing {name} evidence")
        result[name] = {"status": status, "evidence": evidence}

    # The explicit pair comparison is canonical for completion. This prevents
    # an all-PASS verdict from contradicting a no-change or mismatch report.
    completion = result["edit_completion"]
    if visible_change == "NONE" or instruction_match == "FAIL":
        completion["status"] = "FAIL"
    elif (
        visible_change == "UNSURE" or instruction_match == "UNSURE"
    ) and completion["status"] == "PASS":
        completion["status"] = "UNSURE"
    statuses = [result[name]["status"] for name in QUALITY_DIMENSIONS]

    raw_codes = payload.get("reason_codes", [])
    if not isinstance(raw_codes, list):
        raise ValueError("reason_codes must be a list")
    reason_codes = []
    for value in raw_codes:
        code = str(value).strip().upper()
        if code not in REASON_CODES:
            code = "OTHER"
        dimensions = REASON_DIMENSIONS.get(code, ())
        if dimensions and not any(
            result[name]["status"] in {"FAIL", "UNSURE"} for name in dimensions
        ):
            continue
        if code not in reason_codes:
            reason_codes.append(code)

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    confidence = min(1.0, max(0.0, confidence))
    summary = re.sub(r"\s+", " ", str(payload.get("summary", ""))).strip()
    if not summary:
        raise ValueError("missing summary")

    if "FAIL" in statuses:
        verdict = "FAIL"
    elif "UNSURE" in statuses:
        verdict = "UNSURE"
    else:
        verdict = "PASS"
    result.update(
        {
            "reason_codes": reason_codes,
            "summary": summary,
            "confidence": confidence,
            "verdict": verdict,
            "keep": verdict == "PASS",
        }
    )
    return result


def failed_dimensions(assessment: Dict[str, Any]) -> Sequence[str]:
    return tuple(
        name
        for name in QUALITY_DIMENSIONS
        if assessment.get(name, {}).get("status") == "FAIL"
    )


def unresolved_dimensions(assessment: Dict[str, Any]) -> Sequence[str]:
    return tuple(
        name
        for name in QUALITY_DIMENSIONS
        if assessment.get(name, {}).get("status") == "UNSURE"
    )
