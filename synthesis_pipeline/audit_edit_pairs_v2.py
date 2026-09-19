"""Audit an edit from only its outlined source and clean edited image.

The original instruction is never overwritten.  A visually sound but
instruction-mismatched edit may receive a proposed rewrite for separate
review; the proposal does not turn the original audit into a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.vlm_utils as vlm
from synthesis_pipeline.audit_edit_pairs import (
    file_sha256,
    load_jsonl,
    locality_metrics,
    mask_array,
    normalized_task_type,
    parse_json_object,
    refer_object_text,
    write_jsonl,
)
from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs


AUDIT_VERSION = "two_image_model_specific_audit_v6"
TYPE_RULES = {
    "add": "A genuinely new item must appear; the original anchor must not be replaced.",
    "remove": "The selected source object must disappear, with plausible background fill.",
    "replace": "The selected source object must be replaced by a recognizable new identity.",
    "attribute": "The selected source object must keep its identity while the requested property changes.",
}
HUMAN_TYPE_RULES = {
    "add": (
        "Identify the requested new item in IMAGE 2 at the specified anchor. "
        "It must not already be present in IMAGE 1. The original anchor must remain "
        "recognizable. Reject if the item is absent, attached to a different instance, "
        "replaces the anchor, or hangs unsupported when it should be attached."
    ),
    "remove": (
        "Locate the selected source object in IMAGE 1, then actively search its old "
        "position in IMAGE 2. Reject if that object or a recognizable part remains, "
        "or if its removal leaves an obvious hole, smear, or impossible unsupported item. "
        "Independent nearby objects may remain."
    ),
    "replace": (
        "The selected old identity must stop being recognizable and the requested new "
        "identity must be recognizable in its place. Reject if the old object remains "
        "with a new object merely added beside/behind it, or if the replacement is "
        "indistinct or attached to a different instance."
    ),
    "attribute": (
        "The selected object must keep its identity while the requested property "
        "visibly changes. Compare that exact property in both images; do not claim a "
        "change just because the instruction asks for it. Reject if it is unchanged, "
        "a different instance changes, or the object is replaced/deformed instead."
    ),
}
SEMANTIC_CHECKS = {
    "add": (
        "Final semantic check: the new item must be genuinely absent in IMAGE 1 "
        "and attached to the EXACT outlined anchor, not a similar nearby one. "
        "The anchor must remain identifiable; an addition that covers or replaces "
        "it is not successful. Do not infer support or attachment if none is visible."
    ),
    "remove": (
        "Final semantic check: look at the old object's entire footprint in "
        "IMAGE 2. A recognizable old part OR a newly invented person/object in that "
        "footprint is not natural background reconstruction. Do not explain away "
        "an unexpected new subject as one that was previously hidden."
    ),
    "replace": (
        "Final semantic check: the requested NEW identity must actually be "
        "recognizable at the outlined position and the OLD identity must no longer "
        "be recognizable. A second object appearing beside or behind the old one "
        "is an addition, not a replacement."
    ),
    "attribute": (
        "Final semantic check: compare the exact requested property in IMAGE 1 "
        "and IMAGE 2. If IMAGE 1 already has the requested final property, merely "
        "ending with that property is not evidence of an edit. The property's "
        "actual change must be visible, and unrelated distinctive details of the "
        "same object should remain."
    ),
}
REWRITE_VERBS = {
    "add": ("add ",),
    "remove": ("remove ", "erase "),
    "replace": ("replace ",),
    "attribute": ("change ", "make ", "recolor "),
}

PROMPT = """Audit a localized image edit from exactly TWO aligned views. IMAGE 1 is the SOURCE with an external black/white contour marking the original target mask; pixels inside the contour are still photographic. IMAGE 2 is the clean EDITED view at the same scale and location. The views may be context crops, so do not invent content outside them. For an ADD, the mask can mark an existing placement anchor, not the future item.

Inspect the two images BEFORE interpreting the instruction. Name what is really inside the source contour and describe every visible object or part that appears, disappears, or changes at that location. Compare nearby people, objects, logos, hands, supports, and background details. Do not call an object by the requested name unless its distinctive shape is visible. If no change is visible, say so plainly.

Original instruction: <<<INSTRUCTION>>>
Original edit type: <<<TASK_TYPE>>>
Source-target descriptor to verify, not assume: <<<TARGET_HINT>>>
Type rule: <<<TYPE_RULE>>>

Apply these hard checks; do not rationalize contradictions as a plausible interpretation:
- target_match is false if the edit changes a different instance or attaches an addition to a different anchor than requested.
- unexpected_change must name any extra new/deleted object or lost source detail not required by the edit, even if that object is mentioned in the source descriptor. A new person after a removal, an added toy after a color change, or a vanished logo is NOT harmless.
- artifact must name visible residual shapes, black holes, blur patches, cutout borders, disconnected/floating attachments, malformed anatomy, or incoherent background. If too small or ambiguous to verify at this scale, report that as an artifact/uncertainty.
- visual_quality is fail if there is no recognizable edit, unexpected_change, artifact, or an implausible result. Otherwise it can pass even when the requested result was different.
- instruction_match is fail if the original instruction's instance/anchor, edit type, requested identity, extent, count, or attribute is not actually achieved. A round ring is not automatically a pretzel; a statue-like silhouette is not automatically a recognizable deity.

When the actual edit is visually sound and on the same source target, write one concise observed_instruction describing the ACTUAL visible operation with the same edit type and unambiguous original target locator. Do this even if the original instruction also matches. If there is no real edit, the target is wrong, or the result is visually bad, use JSON null. This is a proposed data label, not permission to change the original verdict.

Return one compact JSON object only with exactly these keys: `source_target` (string), `observed_edit` (string), `target_match` (boolean), `unexpected_change` (string or null), `artifact` (string or null), `visual_quality` (`pass` or `fail`), `instruction_match` (`pass` or `fail`), `observed_instruction` (string or null), and `reason` (string). Use visible evidence, not the instruction's wording, for observed_edit."""

LEGACY_ARTIFACT_RULE = "- artifact must name visible residual shapes, black holes, blur patches, cutout borders, disconnected/floating attachments, malformed anatomy, or incoherent background. If too small or ambiguous to verify at this scale, report that as an artifact/uncertainty."
TOLERANT_ARTIFACT_RULE = "- artifact must name CONSPICUOUS residual shapes, black holes, blur/smear patches, cutout borders, disconnected/floating attachments, malformed anatomy, or incoherent background. A faint shadow, slight soft edge, or subtle texture/reflection mismatch can be acceptable if the result remains natural and the requested operation is clear; use null for these minor imperfections."
PROMPT_STRICT_TOLERANT = PROMPT.replace(
    LEGACY_ARTIFACT_RULE, TOLERANT_ARTIFACT_RULE
).replace(
    "- instruction_match is fail if the original instruction's instance/anchor, edit type, requested identity, extent, count, or attribute is not actually achieved.",
    "- instruction_match is fail if the original instruction's instance/anchor, edit type, requested identity, extent, count, or attribute is not actually achieved. Compare requested properties between IMAGE 1 and IMAGE 2: if the property was already present, final-state similarity alone does not prove a change.",
)
PROMPT_SEVERITY = PROMPT.replace(
    LEGACY_ARTIFACT_RULE,
    LEGACY_ARTIFACT_RULE + " Report `artifact_severity` as `none` when there is no "
    "artifact, `minor` only for a faint imperfection that does not harm the "
    "edit's overall realism, or `major` for a conspicuous defect. Do not use "
    "`minor` for an unremoved object, malformed identity, floating attachment, "
    "or an obvious smear/hole.",
).replace(
    "- visual_quality is fail if there is no recognizable edit, unexpected_change, artifact, or an implausible result. Otherwise it can pass even when the requested result was different.",
    "- visual_quality is fail if there is no recognizable edit, unexpected_change, a MAJOR artifact, or an implausible result. A truly minor artifact may coexist with pass if the edit remains broadly natural. Otherwise it can pass even when the requested result was different.",
).replace(
    "`artifact` (string or null), `visual_quality`",
    "`artifact` (string or null), `artifact_severity` (`none`, `minor`, or `major`), `visual_quality`",
)

HUMAN_RUBRIC_PROMPT = """Judge this image-edit training pair using the same practical checks as a careful human reviewer. IMAGE 1 is the SOURCE; a black/white outline marks the original mask without recoloring its contents. IMAGE 2 is the aligned, clean EDITED image. Both can be context crops. <<<MASK_NOTE>>>

Original instruction: <<<INSTRUCTION>>>
Edit type: <<<TASK_TYPE>>>
Target hint (may be inaccurate; check the pixels): <<<TARGET_HINT>>>
<<<PIXEL_NOTE>>>

Review in this order, using what you can actually SEE rather than what the instruction predicts:
1. In IMAGE 1, identify the outlined object/anchor and distinguish it from similar nearby instances. In IMAGE 2, compare the SAME location and neighboring objects. State the actual visible change, or explicitly say "No visible edit." Do not assume an object disappeared merely because it looks different.
2. Apply this edit-type test: <<<TYPE_RULE>>>
3. Check usability: reject conspicuous residual pieces, floating or unsupported additions, malformed objects, obvious blur/smear/fill seams, or substantial unrelated changes. Do not reject a broadly natural edit for tiny texture differences, a faint shadow, or imperfect but plausible reconstruction.
<<<SEMANTIC_CHECKS>>>

Decision gates: `instruction_match=pass` only when the requested operation, exact instance/anchor, identity/attribute, and count visibly match. `target_match=false` for a different instance or addition anchor. `visual_quality=pass` only when a real edit is recognizable, the result is broadly natural, and no conspicuous defect or unrelated change exists. If the claimed change is not visible or you cannot verify it at this scale, fail that criterion; do not invent evidence. Keep `unexpected_change` and `artifact` null for trivial differences, but name any material defect you see. In `reason`, cite concrete before/after evidence and the decisive pass/fail check, not a generic assurance.

Only when the actual edit is visually good, on the correct source target, but differs from the original instruction, propose one short `observed_instruction` for the ACTUAL operation with the same edit type. Otherwise use null. A rewrite must never excuse a wrong instance, missing edit, or visual artifact.

Return one JSON object only: {"source_target": string, "observed_edit": string, "target_match": boolean, "unexpected_change": string or null, "artifact": string or null, "visual_quality": "pass" or "fail", "instruction_match": "pass" or "fail", "observed_instruction": string or null, "reason": string}."""

CHECKLIST_PROMPT = """You are inspecting a potentially FAILED image-edit pair. IMAGE 1 is the source; the black/white outline marks the original target mask without recoloring it. IMAGE 2 is the clean edited view at the same coordinates. Both may be crops. <<<MASK_NOTE>>>

Fill two independent visual inventories BEFORE deciding whether the requested edit happened:
- `source_target`: the visible object or anchor inside the outline in IMAGE 1, including a distinguishing feature.
- `edited_target_area`: what is actually visible at the same location in IMAGE 2, including any surviving part of the old object. Do not write the requested result here unless it is visibly there.
- `old_target_visible`: true if the specific old object's recognizable appearance still survives at that location; a changed object of the same category need not count as the OLD appearance. For a removal this is a crucial veto.

Now compare with the instruction:
Instruction: <<<INSTRUCTION>>>
Edit type: <<<TASK_TYPE>>>
Target hint to verify, not assume: <<<TARGET_HINT>>>
Type-specific check: <<<TYPE_RULE>>>

Set `requested_change_visible` true ONLY if the requested instance, operation, count, and result are actually visible in IMAGE 2. If the old object is still there after a claimed removal or replacement, set it false even if another detail changed. An addition on another person/anchor is false. For attribute, compare the exact requested property in the two images. If you cannot verify the result, set false.

Then inspect usability: a conspicuous remnant, floating attachment, malformed object, obvious blur/smear/fill seam, or substantial unrelated change is a failure. Ignore tiny texture differences, faint shadows, and imperfect but plausible reconstruction. Name material defects in `artifact` or `unexpected_change`; otherwise use null. `visual_quality` can pass for a natural actual edit even if the original instruction is wrong. `instruction_match` passes only if requested_change_visible and target_match are both true. The `reason` must explain the decisive visible evidence, including whether the old target survives; never claim something vanished without checking IMAGE 2.

Only if the result is visually good, on the correct source target, but the original instruction is wrong, propose a concise `observed_instruction` for the ACTUAL operation with the same edit type. Never rewrite away a wrong instance, no-op, or artifact.

Return one JSON object only: {"source_target": string, "edited_target_area": string, "old_target_visible": boolean, "requested_change_visible": boolean, "observed_edit": string, "target_match": boolean, "unexpected_change": string or null, "artifact": string or null, "visual_quality": "pass" or "fail", "instruction_match": "pass" or "fail", "observed_instruction": string or null, "reason": string}."""

EVIDENCE_GATE_PROMPT = """Audit one image-edit pair. IMAGE 1 is the source with a thin black/white outline around the original mask, leaving its photographic pixels visible. IMAGE 2 is the edited image at the same coordinates. Both images may be context crops. <<<MASK_NOTE>>>

Your first obligation is to report visible BEFORE/AFTER evidence, even if it contradicts the instruction. In `source_target`, name the outlined object or anchor and a distinctive visible detail. In `edited_target_area`, name what is actually at that SAME location in IMAGE 2, including any surviving old part and the resulting color, shape, and attachment. For an attribute, state the requested property's before and after appearance explicitly. If the two appearances look alike, report no verified change. Do not infer an object has vanished just because the instruction asks for removal.

Instruction: <<<INSTRUCTION>>>
Edit type: <<<TASK_TYPE>>>
Target hint to check against pixels, not assume: <<<TARGET_HINT>>>
Edit-type test: <<<TYPE_RULE>>>

Decide independently whether (1) the requested operation on the exact selected instance/anchor is visibly complete and (2) the actual result is a usable, broadly natural image edit. Wrong instance, wrong count, missing new identity, surviving old target, or an attribute already present in IMAGE 1 fails instruction match. A conspicuous remnant, floating addition, deformed subject, obvious blur/hole/seam, or substantial unrelated change fails visual quality. Tiny texture differences or a faint natural shadow alone do not fail. If a required detail is too small to verify, fail instruction match rather than inventing evidence. Cite the decisive before/after detail in `reason`.

An instruction mismatch does not automatically fail visual quality. Only for a visually good real edit on the same source target, propose a short `observed_instruction` naming the actual operation and unambiguous target; never let that rewrite change the original verdict. Otherwise use null.

Return exactly one JSON object: {"source_target": string, "edited_target_area": string, "observed_edit": string, "target_match": boolean, "unexpected_change": string or null, "artifact": string or null, "visual_quality": "pass" or "fail", "instruction_match": "pass" or "fail", "observed_instruction": string or null, "reason": string}."""

CONSERVATIVE_GATE_PROMPT = """You are a skeptical visual verifier of ONE image-edit pair, not a captioner of the instruction. IMAGE 1 is the original with a thin black/white contour marking the selected mask; the photographic target is visible inside. IMAGE 2 is the result at the SAME coordinates. Both may be context crops. <<<MASK_NOTE>>>

Complete the visual comparison in this order, before deciding:
1. source_target: What exactly is outlined in IMAGE 1? Include its original color, shape, and relation to nearby instances. Do not copy an inaccurate target hint.
2. edited_target_area: At those same coordinates in IMAGE 2, describe the actual object/background, its color and shape, any surviving original part, and any physical support or attachment. For an attribute, explicitly contrast the requested property's BEFORE and AFTER appearance. Do not assert a requested color/object that cannot be seen.
3. Look ACROSS THE ENTIRE TWO VIEWS for a substantial unrelated insertion, deletion, deformation, edge smear, hole, or missing support. Name such changes in unexpected_change or artifact; otherwise use null. An item suspended without visible support is a defect even if it is in the requested area.

Original instruction: <<<INSTRUCTION>>>
Edit type: <<<TASK_TYPE>>>
Target hint (fallible): <<<TARGET_HINT>>>
Required type test: <<<TYPE_RULE>>>

Treat the instruction as a hypothesis to DISPROVE. instruction_match=pass only if the outlined source target, exact referent, operation, count, and requested final property/identity are all supported by your BEFORE/AFTER descriptions. If the source already had the claimed final property, or only an adjacent part changes, fail. A visually plausible operation at the wrong instance or with a materially false locator still fails. A removal fails if any recognizable target piece remains; an attribute fails if the object's identity is lost.

Independently, visual_quality=pass only if a real edit is visible, the resulting object/background and its scale/support are plausible, and no conspicuous residual, blur, deformation, or unrelated scene change exists. A small texture difference or faint plausible shadow is fine. If evidence is insufficient to establish a required change, fail that criterion rather than guessing. Explain one decisive positive or negative BEFORE/AFTER observation in reason. Do not let a fluent instruction override contradictory pixels.

If the actual edit is visually good and on the SAME outlined target but the original instruction is wrong, propose one short observed_instruction of the same edit type describing only the actual visible operation; otherwise use null. A rewrite never changes the original verdict.

Return one JSON object ONLY: {"source_target": string, "edited_target_area": string, "observed_edit": string, "target_match": boolean, "unexpected_change": string or null, "artifact": string or null, "visual_quality": "pass" or "fail", "instruction_match": "pass" or "fail", "observed_instruction": string or null, "reason": string}."""

NO_EDIT_PATTERN = re.compile(
    r"^\s*(?:no\s+(?:visible\s+)?(?:changes?|edits?|additions?)\b|"
    r"(?:the\s+)?(?:source|target|area|image|stairs)\b.{0,80}\bunchanged\b)",
    flags=re.IGNORECASE,
)


def parse_audit_json(raw: str) -> Optional[dict[str, Any]]:
    """Recover a JSON-fields-only response without accepting free-form prose.

    Some Qwen3.8 responses contain the required JSON key/value pairs but omit
    the outer braces, or omit commas between one-key-per-line fields. This
    recovery is deliberately restricted to that exact shape and is followed
    by the same schema validation as an ordinary response.
    """
    parsed = parse_json_object(raw)
    if parsed is not None:
        return parsed
    stripped = str(raw).strip()
    if not stripped.startswith('"source_target"') or '"reason"' not in stripped:
        return None
    lines = [line.strip().rstrip(",") for line in stripped.splitlines() if line.strip()]
    candidate = "{" + ",".join(lines) + "}" if len(lines) > 1 else "{" + stripped + "}"
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_prompt(
    row: dict[str, Any], prompt_variant: str = "human_rubric",
    metrics: Optional[dict[str, float]] = None,
) -> str:
    task_type = normalized_task_type(row)
    if prompt_variant not in {"human_rubric", "human_rubric_pixel_cue", "human_rubric_semantic", "checklist", "evidence_gate", "conservative_gate", "legacy", "legacy_tolerant", "legacy_severity"}:
        raise ValueError(f"Unknown audit prompt variant: {prompt_variant}")
    template = (
        PROMPT if prompt_variant == "legacy"
        else PROMPT_STRICT_TOLERANT if prompt_variant == "legacy_tolerant"
        else PROMPT_SEVERITY if prompt_variant == "legacy_severity"
        else CHECKLIST_PROMPT if prompt_variant == "checklist"
        else EVIDENCE_GATE_PROMPT if prompt_variant == "evidence_gate"
        else CONSERVATIVE_GATE_PROMPT if prompt_variant == "conservative_gate"
        else HUMAN_RUBRIC_PROMPT
    )
    type_rules = (
        TYPE_RULES if prompt_variant in {"legacy", "legacy_tolerant", "legacy_severity"}
        else HUMAN_TYPE_RULES
    )
    pixel_note = ""
    if prompt_variant == "human_rubric_pixel_cue" and task_type != "add":
        fraction = (metrics or {}).get("inside_changed_fraction")
        percentage = f"{fraction * 100:.1f}%" if fraction is not None else "unavailable"
        pixel_note = (
            "Mechanical cross-check (not a semantic verdict): "
            f"{percentage} of pixels inside the original mask changed by at least "
            "0.05 mean RGB intensity. A low value is a warning that the old target "
            "may still be present; inspect IMAGE 2 carefully before claiming removal, "
            "replacement, or property change. Fine-detail edits can legitimately "
            "change few pixels, so never fail solely from this number."
        )
    return (
        template.replace("<<<INSTRUCTION>>>", str(row.get("editing_instruction", "")))
        .replace("<<<TASK_TYPE>>>", task_type)
        .replace("<<<TARGET_HINT>>>", refer_object_text(row) or "not provided")
        .replace("<<<TYPE_RULE>>>", type_rules[task_type])
        .replace(
            "<<<MASK_NOTE>>>",
            "For this addition the outline marks an existing placement anchor, not the future item."
            if task_type == "add" else "",
        )
        .replace("<<<PIXEL_NOTE>>>\n", f"{pixel_note}\n" if pixel_note else "")
        .replace(
            "<<<SEMANTIC_CHECKS>>>\n",
            SEMANTIC_CHECKS[task_type] + "\n"
            if prompt_variant == "human_rubric_semantic" else "",
        )
    )


def audit_messages(
    outlined_source: Image.Image, edited_image: Image.Image, row: dict[str, Any],
    prompt_variant: str = "human_rubric",
    metrics: Optional[dict[str, float]] = None,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": outlined_source},
                {"type": "image", "image": edited_image},
                {"type": "text", "text": build_prompt(row, prompt_variant, metrics)},
            ],
        }
    ]


def normalize_result(
    value: Optional[dict[str, Any]], task_type: str,
    require_checklist: bool = False,
    require_artifact_severity: bool = False,
    require_edited_area: bool = False,
) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict):
        return None
    result = dict(value)
    for key in ("source_target", "reason"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            return None
        result[key] = result[key].strip()
    observed_edit = result.get("observed_edit")
    if observed_edit is None and str(result.get("instruction_match", "")).lower() == "fail":
        # Some models encode a plainly absent edit as null. Treat it as a
        # semantic failure instead of losing the case to a parse error.
        observed_edit = "No visible edit."
    if not isinstance(observed_edit, str) or not observed_edit.strip():
        return None
    result["observed_edit"] = observed_edit.strip()
    for key in ("visual_quality", "instruction_match"):
        verdict = result.get(key, "")
        result[key] = (
            "pass" if verdict else "fail"
        ) if isinstance(verdict, bool) else str(verdict).strip().lower()
        if result[key] not in {"pass", "fail"}:
            return None
    if not isinstance(result.get("target_match"), bool):
        return None
    if require_edited_area:
        if not isinstance(result.get("edited_target_area"), str) or not result["edited_target_area"].strip():
            return None
        result["edited_target_area"] = result["edited_target_area"].strip()
    if require_checklist:
        if not isinstance(result.get("edited_target_area"), str) or not result["edited_target_area"].strip():
            return None
        result["edited_target_area"] = result["edited_target_area"].strip()
        if not isinstance(result.get("old_target_visible"), bool):
            return None
        if not isinstance(result.get("requested_change_visible"), bool):
            return None
    for key in ("unexpected_change", "artifact"):
        evidence = result.get(key)
        if evidence is not None and not isinstance(evidence, str):
            return None
        result[key] = (evidence.strip() or None) if isinstance(evidence, str) else None
    if require_artifact_severity:
        severity = str(result.get("artifact_severity", "")).strip().lower()
        if severity not in {"none", "minor", "major"}:
            return None
        if bool(result["artifact"]) != (severity != "none"):
            return None
        result["artifact_severity"] = severity
    candidate = result.get("observed_instruction")
    if isinstance(candidate, str):
        candidate = candidate.strip() or None
    if candidate is not None and not isinstance(candidate, str):
        return None
    no_edit = bool(NO_EDIT_PATTERN.search(result["observed_edit"]))
    severe_artifact = result["artifact"] and (
        not require_artifact_severity or result["artifact_severity"] == "major"
    )
    if result["unexpected_change"] or severe_artifact or no_edit:
        result["visual_quality"] = "fail"
    if not result["target_match"] or no_edit:
        result["instruction_match"] = "fail"
    if require_checklist:
        vetoes = []
        if not result["requested_change_visible"]:
            vetoes.append("requested_change_not_visible")
        if task_type in {"remove", "replace"} and result["old_target_visible"]:
            vetoes.append("old_target_still_visible")
        if task_type in {"add", "attribute"} and not result["old_target_visible"]:
            vetoes.append("anchor_or_identity_lost")
        if vetoes:
            result["instruction_match"] = "fail"
        result["checklist_vetoes"] = vetoes
    valid_observed_instruction = (
        candidate is not None
        and candidate.lower().startswith(REWRITE_VERBS[task_type])
        and len(candidate.split()) <= 35
        and result["target_match"]
        and result["visual_quality"] == "pass"
        and not result.get("checklist_vetoes")
    )
    result["observed_instruction"] = (
        candidate if valid_observed_instruction else None
    )
    eligible = (
        valid_observed_instruction
        and result["instruction_match"] == "fail"
    )
    result["rewrite_candidate"] = candidate if eligible else None
    result["salvage_status"] = "manual_review_candidate" if eligible else "none"
    result["quality"] = (
        "pass"
        if result["visual_quality"] == result["instruction_match"] == "pass"
        else "fail"
    )
    return result


def apply_low_change_veto(
    result: Optional[dict[str, Any]], task_type: str,
    metrics: dict[str, float], threshold: float,
    attribute_threshold: float = 0.0,
) -> Optional[dict[str, Any]]:
    """Reject likely no-op remove/replace pairs without another VLM request.

    This is deliberately limited to operations that should replace most of
    the original mask footprint. Additions and fine-grained attribute edits
    can legitimately change only a small part of the anchor mask.
    """
    active_threshold = (
        attribute_threshold if task_type == "attribute"
        else threshold if task_type in {"remove", "replace"}
        else 0.0
    )
    if result is None or active_threshold <= 0:
        return result
    changed = metrics["inside_changed_fraction"]
    if changed >= active_threshold:
        return result
    result["metric_veto"] = {
        "code": "low_mask_change_for_attribute" if task_type == "attribute"
        else "low_mask_change_for_remove_or_replace",
        "inside_changed_fraction": changed,
        "threshold": active_threshold,
    }
    result["instruction_match"] = "fail"
    result["quality"] = "fail"
    result["observed_instruction"] = None
    result["rewrite_candidate"] = None
    result["salvage_status"] = "none"
    result["reason"] = (
        result["reason"].rstrip(". ") + ". "
        f"Mechanical veto: only {changed:.1%} of the source mask changed "
        f"(below {active_threshold:.0%}); a visible {task_type} is not established."
    )
    return result


def fingerprint(
    row: dict[str, Any], source_path: Path, edited_path: Path, args: argparse.Namespace
) -> str:
    payload = {
        "version": AUDIT_VERSION,
        "prompt": build_prompt(row, args.prompt_variant),
        "mask": row.get("mask"),
        "source_sha256": file_sha256(source_path),
        "edited_sha256": file_sha256(edited_path),
        "vlm": args.vlm,
        "vlm_model_id": args.vlm_model_id,
        "vlm_dtype": args.vlm_dtype,
        "max_new_tokens": args.max_new_tokens,
        "longest_side": args.longest_side,
        "image_scope": args.image_scope,
        "guard_pixels": args.guard_pixels,
        "low_change_veto_threshold": args.low_change_veto_threshold,
        "attribute_low_change_veto_threshold": args.attribute_low_change_veto_threshold,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--edited-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vlm", choices=("qwen8b-vllm", "qwen38-vllm"), required=True)
    parser.add_argument("--vlm-model-id", default=None)
    parser.add_argument("--vlm-device", default="cuda:0")
    parser.add_argument("--vlm-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--longest-side", type=int, default=1280)
    parser.add_argument(
        "--image-scope", choices=("full", "context_crop"), default="context_crop"
    )
    parser.add_argument("--guard-pixels", type=int, default=24)
    parser.add_argument(
        "--low-change-veto-threshold", type=float, default=0.30,
        help="Conservative remove/replace no-op veto; 0 disables it.",
    )
    parser.add_argument(
        "--attribute-low-change-veto-threshold", type=float, default=0.0,
        help="Optional attribute no-op veto; calibrate on reviewed cases before enabling.",
    )
    parser.add_argument("--save-input-previews", action="store_true")
    parser.add_argument(
        "--prompt-variant", choices=("auto", "human_rubric", "human_rubric_pixel_cue", "human_rubric_semantic", "checklist", "evidence_gate", "conservative_gate", "legacy", "legacy_tolerant", "legacy_severity"),
        default="auto",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.prompt_variant == "auto":
        # The broader human rubric helps 8B on the 52-case pilot, whereas
        # 27B loses precision on the separate 20-case set with that rubric.
        args.prompt_variant = "legacy" if args.vlm == "qwen38-vllm" else "human_rubric"
    if args.longest_side < 256:
        raise ValueError("--longest-side must be at least 256")
    if not 0 <= args.low_change_veto_threshold <= 1:
        raise ValueError("--low-change-veto-threshold must be within [0, 1]")
    if not 0 <= args.attribute_low_change_veto_threshold <= 1:
        raise ValueError("--attribute-low-change-veto-threshold must be within [0, 1]")
    rows = load_jsonl(args.annotations_jsonl)
    if args.max_items is not None:
        rows = rows[: args.max_items]
    existing_path = args.out_dir / "edit_audit.jsonl"
    existing = (
        {str(row["image"]): row for row in load_jsonl(existing_path)}
        if args.resume and existing_path.exists()
        else {}
    )
    tasks: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["image"])
        source_path = args.source_dir / str(row.get("source_image") or name)
        edited_path = args.edited_dir / name
        signature = fingerprint(row, source_path, edited_path, args)
        if name in existing and existing[name].get("input_fingerprint") == signature:
            results[name] = existing[name]
            continue
        with Image.open(source_path) as handle:
            source = handle.convert("RGB")
        with Image.open(edited_path) as handle:
            edited = handle.convert("RGB")
        mask = mask_array(source.size, row.get("mask", []))
        before, after = audit_two_image_inputs(
            source,
            edited,
            mask,
            longest_side=args.longest_side,
            scope=args.image_scope,
        )
        if args.save_input_previews:
            preview_dir = args.out_dir / "inputs"
            preview_dir.mkdir(parents=True, exist_ok=True)
            before.save(preview_dir / f"{Path(name).stem}_source.png")
            after.save(preview_dir / f"{Path(name).stem}_edited.png")
        tasks.append(
            {
                "row": row,
                "before": before,
                "after": after,
                "metrics": locality_metrics(source, edited, mask, args.guard_pixels),
                "input_fingerprint": signature,
            }
        )

    load_seconds = 0.0
    inference_seconds = 0.0
    if tasks:
        vlm.configure_backend(
            name=args.vlm,
            model_id=args.vlm_model_id,
            device=args.vlm_device,
            dtype=args.vlm_dtype,
        )
        load_started = time.perf_counter()
        backend = vlm.get_backend()
        load_seconds = time.perf_counter() - load_started
        for offset in tqdm(range(0, len(tasks), args.batch_size), desc="two-image audit"):
            batch = tasks[offset : offset + args.batch_size]
            request_started = time.perf_counter()
            outputs = backend.chat_batch(
                [audit_messages(task["before"], task["after"], task["row"],
                                args.prompt_variant, task["metrics"]) for task in batch],
                max_new_tokens=args.max_new_tokens,
            )
            inference_seconds += time.perf_counter() - request_started
            for task, raw in zip(batch, outputs):
                row = task["row"]
                name = str(row["image"])
                parsed = normalize_result(
                    parse_audit_json(raw), normalized_task_type(row),
                    require_checklist=args.prompt_variant == "checklist",
                    require_artifact_severity=args.prompt_variant == "legacy_severity",
                    require_edited_area=args.prompt_variant in {"evidence_gate", "conservative_gate"},
                )
                parsed = apply_low_change_veto(
                    parsed, normalized_task_type(row), task["metrics"],
                    args.low_change_veto_threshold,
                    args.attribute_low_change_veto_threshold,
                )
                results[name] = {
                    "image": name,
                    "task_type": normalized_task_type(row),
                    "editing_instruction": row.get("editing_instruction"),
                    "quality": parsed["quality"] if parsed else "parse_error",
                    "rewrite_candidate": parsed.get("rewrite_candidate") if parsed else None,
                    "salvage_status": parsed.get("salvage_status") if parsed else "none",
                    "audit": parsed,
                    "locality_metrics": task["metrics"],
                    "raw_response": raw,
                    "input_fingerprint": task["input_fingerprint"],
                }
        vlm.shutdown_backend()

    ordered = [results[str(row["image"])] for row in rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(existing_path, ordered)
    summary = {
        "audit_version": AUDIT_VERSION,
        "prompt_variant": args.prompt_variant,
        "low_change_veto_threshold": args.low_change_veto_threshold,
        "attribute_low_change_veto_threshold": args.attribute_low_change_veto_threshold,
        "low_change_veto_count": sum(
            bool((row.get("audit") or {}).get("metric_veto")) for row in ordered
        ),
        "cases": len(ordered),
        "quality_counts": dict(Counter(row["quality"] for row in ordered)),
        "visual_quality_counts": dict(
            Counter(
                (row.get("audit") or {}).get("visual_quality", "parse_error")
                for row in ordered
            )
        ),
        "salvage_candidate_count": sum(
            row["salvage_status"] == "manual_review_candidate" for row in ordered
        ),
        "vlm": args.vlm,
        "vlm_model_id": args.vlm_model_id,
        "vlm_dtype": args.vlm_dtype,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "images_per_call": 2,
        "image_scope": args.image_scope,
        "vlm_calls": len(tasks),
        "reused_cases": len(ordered) - len(tasks),
        "backend_load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "inference_cases_per_minute": (
            round(len(tasks) * 60 / inference_seconds, 3)
            if inference_seconds > 0
            else 0.0
        ),
        "audit_wall_seconds": round(time.perf_counter() - started, 3),
    }
    temporary = args.out_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.out_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
