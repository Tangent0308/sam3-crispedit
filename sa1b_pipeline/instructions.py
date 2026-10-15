"""Benchmark-v2 instruction contracts and deterministic normalization."""
from __future__ import annotations

import re
from typing import Any

from .schema import EditUnit
from .instruction_protocol import VERSION, make_spec, render, validate_pair, length_advice, needs_local_execution, mask_contract


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
    value = _clean(raw_scope)
    if not value:
        value = _clean(category).split(',', 1)[0]
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
    return render(make_spec(operation, _clean(target), 'object', _clean(payload),
                            attribute_kind=attribute_kind))


def build_unit(raw: dict[str, Any], obj: dict[str, Any], mask_path: str, geometry: dict[str, Any],
               protection: list[str]) -> EditUnit:
    operation = _clean(raw.get("operation")).lower()
    if operation not in {"add", "replace", "remove", "attribute"}:
        raise ValueError(f"invalid MLLM operation: {operation}")
    ref = _lower_article(_clean(raw.get("ref_phrase") or raw.get("target") or obj.get("description") or obj.get("category")))
    scope = _scope_phrase(_clean(raw.get("scope") or ""), _clean(obj.get("category") or ""))
    attributes = raw.get('attributes') or []
    attribute_kind = _clean(raw.get('attribute_kind')) or 'color'
    payload = _payload_for_operation(operation, _clean(raw.get('payload') or raw.get('new_content')), ref)
    if operation == 'attribute' and attribute_kind == 'text':
        payload = str(raw.get('payload', raw.get('new_content', ''))).strip()
    if operation == 'add' and not re.match(r'^(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|[1-9][0-9]*)\s+', payload, flags=re.I):
        raise ValueError('add payload must introduce a new physical item')
    part = _clean(raw.get('target_part'))
    # Some planners use ``target_part`` for an add host's attachment site
    # even though protocol 1.0 stores that information under placement.site.
    # Promote this equivalent representation before applying executability
    # gates; the add mask remains the host anchor and no extra region is
    # invented. A genuine add subpart mask is therefore never required.
    placement_raw = raw.get('placement')
    if operation == 'add' and part:
        placement_raw = {**(placement_raw if isinstance(placement_raw, dict) else {}), 'site': part}
        part = ''
    mode = raw.get('remove_mode') or ('part' if operation == 'remove' and part else 'whole')
    spec = make_spec(operation, ref, scope, payload, target_part=part, remove_mode=mode,
                     attribute_kind=attribute_kind, attributes=attributes,
                     placement=placement_raw, qualifier=raw.get('qualifier', ''))
    execution = obj.get('metadata', {}).get('execution_mask_path')
    # Geometry is supplied by a trusted preparation step, never a path invented
    # by the planning VLM. A host selector alone cannot resolve a subpart.
    if needs_local_execution(spec) and not execution:
        raise ValueError('subpart task requires a resolved execution_mask_path in object metadata')
    if spec['placement'].get('anchor_mode') == 'reference' and not execution:
        raise ValueError('reference add requires a resolved execution_mask_path in object metadata')
    if operation == 'attribute':
        category = _clean(obj.get('category')).lower().split(',')[0].strip()
        if category in {'person', 'people', 'man', 'woman', 'boy', 'girl', 'human', 'pedestrian'} and not part:
            raise ValueError('attribute requires the actual surface/part mask, not a whole-person mask')
        surface = _clean(raw.get('edit_surface')).lower()
        if category in {'window', 'windows', 'car', 'cars', 'sedan', 'suv', 'side mirror', 'mirror'} and surface and not part:
            raise ValueError('mixed-component target requires a separately selected surface/part mask for attribute')
        for attr in spec['attributes']:
            value = attr['value']
            if attr['property'] == 'material' and re.fullmatch(
                r'(?:(?:matte|glossy|bright|dark|light|deep)\s+)*(?:black|white|red|blue|green|grey|gray|yellow|orange|pink|purple)', value.lower()):
                raise ValueError('material payload is only a color/finish, not a material')
            if attr['property'] in {'color', 'tint'} and _clean(raw.get('source_color')).lower() == value.lower():
                raise ValueError('requested color is already present in the clean source')
            # Reject planner no-ops such as "change the red SUV to red".
            # The reference and scope are trusted source selectors, so a
            # repeated visible descriptor is evidence that the requested
            # property is already present even when source_color is absent.
            if attr['property'] in {'color', 'tint', 'material', 'finish', 'pattern', 'texture'}:
                value_norm = re.sub(r'\s+', ' ', value).strip().lower()
                source_descriptors = ' '.join(
                    _clean(raw.get(key)) for key in ('ref_phrase', 'scope', 'source_color')
                ).lower()
                if value_norm and re.search(
                        r'(?<![a-z0-9])' + re.escape(value_norm) + r'(?![a-z0-9])',
                        source_descriptors):
                    raise ValueError('requested attribute value is already present in the source selector')
        attribute_kind = spec['attributes'][0]['property'] if len(spec['attributes']) == 1 else 'multiple'
    else:
        attribute_kind = None
    ref_instruction, no_ref_instruction = render(spec), render(spec, False)
    completion = {
        'add': 'Keep the source anchor; realize exactly the requested count and placement relation.',
        'replace': 'Replace only the named object or part; retain the rest of its host and scene.',
        'remove': 'Remove exactly the named extent and contents if requested; preserve the remaining host.',
        'attribute': 'Change exactly the requested properties; retain identity and all unrequested structure and appearance.',
    }[operation]
    return EditUnit(
        unit_id=f"U{raw.get('unit_index', 1)}", object_id=str(obj.get('object_id', obj.get('id'))),
        operation=operation, ref_phrase=ref, scope=scope, payload=payload,
        instruction_ref=ref_instruction, instruction_noref=no_ref_instruction,
        mask_path=mask_path, bbox=[int(x) for x in geometry['bbox']], point=[int(x) for x in geometry['point']],
        attribute_kind=attribute_kind, locator='mask', has_ref=True,
        edit_surface=_clean(raw.get('edit_surface')) or None, source_color=_clean(raw.get('source_color')) or None,
        completion_requirement=ref_instruction + ' ' + completion, preserve=list(protection),
        mllm_evidence=_clean(raw.get('evidence')), mask_role=mask_contract(spec)['source_role'], mask_semantics=mask_contract(spec),
        support_policy={'add':'anchor_plus_attachment','replace':'target_plus_collar','remove':'target_plus_collar','attribute':'target_only'}[operation],
        execution_mask_path=execution, protected_mask_path=obj.get("metadata", {}).get("protected_mask_path"),
        instruction_protocol_version=VERSION, instruction_spec=spec,
        instruction_length_advice=length_advice(ref_instruction, no_ref_instruction),
    )


def validate_instruction(unit: EditUnit) -> list[str]:
    errors = []
    if unit.instruction_spec:
        try:
            validate_pair(unit.instruction_spec, unit.instruction_ref, unit.instruction_noref)
            if unit.instruction_protocol_version != VERSION or unit.operation != unit.instruction_spec['operation']:
                errors.append('instruction_protocol_mismatch')
        except ValueError:
            errors.append('instruction_semantic_pair_mismatch')
    elif unit.instruction_ref.lower() == unit.instruction_noref.lower():
        errors.append('ref_and_noref_are_identical')
    return errors
