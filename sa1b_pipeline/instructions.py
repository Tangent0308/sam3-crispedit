"""Benchmark-v2 instruction contracts and deterministic normalization."""
from __future__ import annotations

import re
from typing import Any

from .schema import EditUnit


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().rstrip(" .;"))


def _lower_article(value: str) -> str:
    """Match benchmark-v2's sentence-internal referring phrases."""
    for article in ("the ", "a ", "an "):
        if value.lower().startswith(article):
            return value[:1].lower() + value[1:]
    return value


def _scope_phrase(raw_scope: str, category: str) -> str:
    """Return the short noun phrase used by v2 no-ref instructions."""
    category = _clean(category)
    if category and category.lower() not in {"object", "thing", "part", "unknown", "item"}:
        # Dense SA-1B categories sometimes contain comma-separated aliases.
        return _lower_article(category.split(",", 1)[0].strip())
    value = _clean(raw_scope)
    value = re.sub(r"^(?:the|a|an)\s+", "", value, flags=re.I)
    value = re.sub(r"^(?:entire|whole|visible|selected)\s+", "", value, flags=re.I)
    return _lower_article(value or "object")


def _payload_for_operation(operation: str, payload: str, target: str) -> str:
    """Normalize an MLLM that returned a full instruction instead of payload."""
    value = _clean(payload)
    lower = value.lower()
    if operation == "attribute":
        match = re.search(r"\bto\s+(.+)$", value, re.I)
        if lower.startswith("change ") and match:
            return _clean(match.group(1))
    if operation == "replace":
        match = re.search(r"\bwith\s+(.+)$", value, re.I)
        if lower.startswith("replace ") and match:
            return _clean(match.group(1))
    if operation == "add":
        match = re.match(r"add\s+(.+?)\s+to\s+.+$", value, re.I)
        if lower.startswith("add ") and match:
            return _clean(match.group(1))
    return value


def render_instruction(operation: str, target: str, payload: str, attribute_kind: str | None = None) -> str:
    target, payload = _clean(target), _clean(payload)
    if operation == "add":
        return f"Add {payload} to {target}."
    if operation == "replace":
        return f"Replace {target} with {payload}."
    if operation == "remove":
        return f"Remove {target}."
    if operation == "attribute":
        kind = "material" if attribute_kind == "material" else "color"
        return f"Change the {kind} of {target} to {payload}."
    raise ValueError(f"unknown operation: {operation}")


def build_unit(raw: dict[str, Any], obj: dict[str, Any], mask_path: str, geometry: dict[str, Any],
               protection: list[str]) -> EditUnit:
    operation = _clean(raw.get("operation")).lower()
    if operation not in {"add", "replace", "remove", "attribute"}:
        raise ValueError(f"invalid MLLM operation: {operation}")
    ref = _lower_article(_clean(raw.get("ref_phrase") or raw.get("target") or obj.get("description") or obj.get("category")))
    scope = _scope_phrase(_clean(raw.get("scope") or ""), _clean(obj.get("category") or ""))
    payload = _payload_for_operation(operation, _clean(raw.get("payload") or raw.get("new_content")), ref)
    if operation == "add" and not re.match(r"^(?:a|an|one|two|three)\s+", payload, flags=re.I):
        raise ValueError("add payload must introduce a new physical item")
    if operation == 'remove' and not payload:
        payload = 'remove the selected target and reconstruct the revealed background'
    if not payload:
        raise ValueError(f"MLLM omitted payload for {obj.get('object_id')}")
    attribute_kind = _clean(raw.get("attribute_kind")) or None
    if operation == "attribute" and attribute_kind not in {"color", "material"}:
        attribute_kind = "color"
    ref_instruction = render_instruction(operation, ref, payload, attribute_kind)
    # The benchmark contract uses type-specific region language.  Keeping the
    # replacement phrase explicit is important: a mask must follow a complete
    # referring phrase, never an article or a guessed scope description.
    kind = "material" if attribute_kind == "material" else "color"
    no_ref_instruction = {
        "add": f"Add {payload} to the selected {scope}.",
        "replace": f"Replace the selected {scope} with {payload}.",
        "remove": f"Remove the selected {scope}.",
        "attribute": f"Change the {kind} of the selected {scope} to {payload}.",
    }[operation]
    object_id = str(obj.get("object_id", obj.get("id")))
    completion = {
        "add": "The source object remains, and the added item must have the specified attachment.",
        "replace": "The old target must be absent; do not leave fragments or add a duplicate beside it.",
        "remove": "The old target must be absent; preserve the rest of the scene and neighboring instances.",
        "attribute": "Change the requested appearance while retaining the selected object geometry and identity.",
    }[operation]
    return EditUnit(
        unit_id=f"U{raw.get('unit_index', 1)}", object_id=object_id, operation=operation,
        ref_phrase=ref, scope=scope, payload=payload,
        instruction_ref=ref_instruction, instruction_noref=no_ref_instruction,
        mask_path=mask_path, bbox=[int(x) for x in geometry["bbox"]], point=[int(x) for x in geometry["point"]],
        attribute_kind=attribute_kind, locator="mask", has_ref=True,
        completion_requirement=ref_instruction + " " + completion,
        preserve=list(protection), mllm_evidence=_clean(raw.get("evidence")),
    )


def validate_instruction(unit: EditUnit) -> list[str]:
    errors = []
    if unit.operation == "add" and not re.search(r"\bto\b", unit.instruction_ref, re.I):
        errors.append("add_missing_attachment_relation")
    if unit.operation == "replace" and not re.search(r"\bwith\b", unit.instruction_ref, re.I):
        errors.append("replace_missing_new_identity")
    if unit.operation == "attribute" and not re.search(r"\b(color|material)\b", unit.instruction_ref, re.I):
        errors.append("attribute_missing_property")
    if unit.instruction_ref.lower() == unit.instruction_noref.lower():
        errors.append("ref_and_noref_are_identical")
    if len(unit.instruction_ref.split()) > 26 or len(unit.instruction_noref.split()) > 16:
        errors.append("instruction_too_long")
    return errors
