"""Source-only binary filter for difficult local referential edits."""

from __future__ import annotations

from typing import Any, Dict

from PIL import Image

from crispedit.prefilter.pair_quality import extract_json_object
from crispedit.prefilter.policy import canonical_edit_type


FILTER_METHOD = "difficult_local_edit"
EVIDENCE_SCHEMA = "source_binary_judgment"
PROMPT_VERSION = "crispedit_difficult_local_edit"

ELIGIBLE_TYPES = frozenset({"add", "remove", "replace", "color", "motion"})
VERDICTS = frozenset({"PASS", "DROP"})


SCENE_PROMPT = """Select samples for a difficult local image-editing benchmark.

You see only the SOURCE image.
Edit type: __EDIT_TYPE__
Instruction: __INSTRUCTION__

PASS when finding the exact edit region is genuinely difficult: the source has multiple comparable objects,
parents, or anchors, and the instruction selects an instance, local subset/group, part of a selected parent,
or precise placement relative to them. Comparable candidates must share the target's semantic role; nearby
different roles do not count. Spatial, ordinal, and relative cues are valid. Recognizable small or
background objects count. "Lone" can select a locally isolated object when same-category objects exist
elsewhere. A compact spatial or attribute-selected group of separate objects can PASS even when it is the
whole matching group if localizing its multiple regions is difficult. A local addition passes when tied to
named source anchors, such as between two people.

DROP a unique obvious target; a global, background, or style edit; a plain all/every-object edit without local
selection; an ambiguous target; an addition located only by an absolute image area; or a broad uncounted group
addition. Repeated parts within one parent never qualify even with a spatial cue, such as a hair section, one
tire, or one hand. A part qualifies only after its parent is selected among comparable parents, such as the
eyes of the middle dog. For an existing edit, mentally remove the identifying phrase: if the target would
still be the only obvious candidate, DROP.
Unrelated landmarks do not make a unique target difficult, such as a unique hammock between two umbrellas.
An unqualified "one of" is ambiguous. The target noun alone is not a selector among same-role candidates.
Visual prominence or centrality cannot replace a selector missing from the instruction. Do not invent a
relation. Do not segment, draw boxes, enumerate, or count.

Return one JSON object and no markdown:
{"verdict":"PASS or DROP","target":"short edit target","reference":"identifying phrase from the instruction, or NONE","reason":"one short sentence"}"""


JSON_CORRECTION_PROMPT = """Return only one valid JSON object with exactly these fields:
{"verdict":"PASS or DROP","target":"short edit target","reference":"exact phrase or NONE","reason":"one short sentence"}
Use PASS or DROP only. Do not add markdown."""


def build_scene_prompt(edit_type: object, instruction: object) -> str:
    """Build the concise source-only scene qualification prompt."""

    return (
        SCENE_PROMPT.replace("__EDIT_TYPE__", canonical_edit_type(edit_type))
        .replace("__INSTRUCTION__", str(instruction or "").strip())
    )


def build_scene_conversation(
    source_image: Image.Image,
    edit_type: object,
    instruction: object,
) -> list[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_image.convert("RGB")},
                {"type": "text", "text": build_scene_prompt(edit_type, instruction)},
            ],
        }
    ]


def deterministic_screen(edit_type: object) -> Dict[str, Any]:
    """Skip edit families that cannot be a local referential edit."""

    canonical = canonical_edit_type(edit_type)
    eligible = canonical in ELIGIBLE_TYPES
    return {
        "eligible": eligible,
        "canonical_type": canonical,
        "reason": "model_required" if eligible else "ineligible_edit_type",
    }


def _required_text(payload: Dict[str, Any], field: str) -> str:
    value = str(payload.get(field) or "").strip()
    if not value:
        raise ValueError(f"{field} is required")
    return value


def normalize_scene_assessment(payload: object) -> Dict[str, Any]:
    """Validate the model's direct binary scene judgment."""

    if not isinstance(payload, dict):
        raise ValueError("scene assessment must be a JSON object")
    verdict = str(payload.get("verdict") or "").strip().upper()
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be PASS or DROP, got {verdict!r}")
    target = _required_text(payload, "target")
    reference = _required_text(payload, "reference")
    reason = _required_text(payload, "reason")
    if verdict == "PASS" and reference.upper() == "NONE":
        raise ValueError("PASS requires an explicit identifying reference")
    return {
        "verdict": verdict,
        "target": target,
        "reference": reference,
        "reason": reason,
    }


def parse_scene_response(text: object) -> Dict[str, Any]:
    return normalize_scene_assessment(extract_json_object(text))
