"""Image-pair quality prefilter for native RefEdit samples.

The auditor makes one image-conditioned call per sample and records factual
judgements for independent quality dimensions.  The final PASS/FAIL/UNSURE
decision is derived in code so that a model cannot silently waive a failed
dimension in a free-form overall verdict.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from PIL import Image


QUALITY_PROMPT_VERSION = "refedit_pair_quality_qwen38_v1"
QUALITY_DIMENSIONS = (
    "source_reference",
    "instruction_meaningfulness",
    "edit_completion",
    "source_integrity",
    "target_integrity",
    "content_preservation",
)
QUALITY_STATUSES = {"PASS", "FAIL", "UNSURE"}
REASON_CODES = {
    "MISSING_SOURCE_REFERENT",
    "WRONG_SOURCE_ATTRIBUTE",
    "AMBIGUOUS_REFERENT",
    "NO_OP_ALREADY_SATISFIED",
    "EDIT_NOT_COMPLETED",
    "WRONG_REFERENT_EDITED",
    "WRONG_ATTRIBUTE_OR_ACTION",
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
    "WRONG_SOURCE_ATTRIBUTE": ("source_reference",),
    "AMBIGUOUS_REFERENT": ("source_reference",),
    "NO_OP_ALREADY_SATISFIED": ("instruction_meaningfulness",),
    "EDIT_NOT_COMPLETED": ("edit_completion",),
    "WRONG_REFERENT_EDITED": ("edit_completion",),
    "WRONG_ATTRIBUTE_OR_ACTION": ("edit_completion",),
    "INCOMPLETE_OBJECT_REMOVAL": ("edit_completion",),
    "COUNT_OR_LOCATION_MISMATCH": ("source_reference", "edit_completion"),
    "SOURCE_GENERATION_ARTIFACT": ("source_integrity",),
    "TARGET_GENERATION_ARTIFACT": ("target_integrity",),
    "IMPLAUSIBLE_PLACEMENT_OR_GEOMETRY": (
        "edit_completion",
        "target_integrity",
    ),
    "UNRELATED_CONTENT_CHANGED": ("content_preservation",),
    "PAIR_MISALIGNED": ("content_preservation",),
}


QUALITY_PROMPT = """You are a strict but fair quality auditor for an image-editing dataset.

Image 1 is the SOURCE before editing. Image 2 is the TARGET edited result.
Editing task: __TASK__
Instruction: __INSTRUCTION__

Judge the actual pixels in both images. Do not assume that the edit succeeded merely
because the target looks plausible. Check all six dimensions independently:

1. source_reference: The exact object(s) described by the instruction, including
   location, count, order, appearance, and relations, must exist in Image 1.
2. instruction_meaningfulness: The requested change must not already be clearly true
   in Image 1. For a move, the selected object must not already be at the requested
   destination. The instruction must define a visually checkable edit.
3. edit_completion: Image 2 must perform the requested operation on the correct
   referent, with the requested material/color/action/location/count. A removal must
   also remove integral contents or dependent traces that cannot remain on their own;
   for example, removing a fountain while leaving its water suspended is incomplete.
4. source_integrity: Image 1 must be a coherent usable source, without conspicuous
   blur, smearing, missing structures, duplicated fragments, or pre-existing
   generation/edit artifacts in the relevant scene.
5. target_integrity: The edited object and its placement/geometry must look coherent,
   without obvious holes, smears, malformed shapes, floating remnants, or implausible
   insertion artifacts.
6. content_preservation: Unrelated scene content should remain substantially
   consistent. Ignore harmless JPEG/resizing differences and small texture or lighting
   drift, but fail large unrelated semantic or geometric changes.

Use PASS when the dimension is visibly satisfied, FAIL when there is concrete visual
evidence of a defect, and UNSURE only when the pixels genuinely do not resolve it.
Do not penalize an edit just for being subtle. Do not invent hidden states.

Return exactly one JSON object and no markdown:
{
  "source_reference": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "instruction_meaningfulness": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "edit_completion": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "source_integrity": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "target_integrity": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "content_preservation": {"status": "PASS|FAIL|UNSURE", "evidence": "short image-grounded evidence"},
  "reason_codes": ["zero or more codes from the allowed list below"],
  "summary": "one concise explanation grounded in both images",
  "confidence": 0.0
}

Allowed reason codes:
MISSING_SOURCE_REFERENT, WRONG_SOURCE_ATTRIBUTE, AMBIGUOUS_REFERENT,
NO_OP_ALREADY_SATISFIED, EDIT_NOT_COMPLETED, WRONG_REFERENT_EDITED,
WRONG_ATTRIBUTE_OR_ACTION, INCOMPLETE_OBJECT_REMOVAL, COUNT_OR_LOCATION_MISMATCH,
SOURCE_GENERATION_ARTIFACT, TARGET_GENERATION_ARTIFACT,
IMPLAUSIBLE_PLACEMENT_OR_GEOMETRY, UNRELATED_CONTENT_CHANGED, PAIR_MISALIGNED,
INSUFFICIENT_VISUAL_EVIDENCE, OTHER.
"""


JSON_CORRECTION_PROMPT = """The previous answer was not a valid complete JSON object.
Return the same assessment using exactly the requested schema. Include all six
dimensions, reason_codes, summary, and confidence. Output JSON only."""


def build_quality_prompt(task: object, instruction: object) -> str:
    """Build the single-call pair-quality prompt."""

    return QUALITY_PROMPT.replace("__TASK__", str(task)).replace(
        "__INSTRUCTION__", str(instruction)
    )


def build_quality_conversation(
    source: Image.Image,
    target: Image.Image,
    task: object,
    instruction: object,
) -> List[Dict[str, Any]]:
    """Build a Qwen/vLLM multimodal conversation with explicit image roles."""

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
                    "text": build_quality_prompt(task, instruction),
                },
            ],
        }
    ]


def extract_json_object(text: object) -> Dict[str, Any]:
    """Extract the first complete JSON object without greedy fence matching."""

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
    """Validate evidence fields and derive a deterministic final verdict."""

    result: Dict[str, Any] = {}
    statuses = []
    for name in QUALITY_DIMENSIONS:
        dimension = payload.get(name)
        if not isinstance(dimension, dict):
            raise ValueError(f"missing quality dimension: {name}")
        status = str(dimension.get("status", "")).strip().upper()
        if status not in QUALITY_STATUSES:
            raise ValueError(f"invalid {name} status: {status!r}")
        evidence = str(dimension.get("evidence", "")).strip()
        if not evidence:
            raise ValueError(f"missing {name} evidence")
        result[name] = {"status": status, "evidence": evidence}
        statuses.append(status)

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
            # The dimension evidence is the canonical judgement.  Discard a
            # contradictory auxiliary code instead of polluting aggregate
            # failure statistics (for example NO_OP beside a PASS
            # instruction_meaningfulness judgement).
            continue
        if code not in reason_codes:
            reason_codes.append(code)

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    confidence = min(1.0, max(0.0, confidence))
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise ValueError("missing summary")

    # Only an all-PASS assessment is admitted into a strict training subset.
    # Every failed dimension is terminal; unresolved visual evidence is kept
    # separate from both PASS and FAIL for auditable policy changes later.
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
