"""Current edit-unit prompts and strict observation normalization."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence
from crispedit.common import canonical_edit_type
from crispedit.mask.checklist import observation_prompt, grounding_prompt, flatten_edit_units, strict_json

TWO_PASS_PROMPT_VERSION = "edit-region-checklist"
PROMPT_VERSION = TWO_PASS_PROMPT_VERSION
OBSERVATION_PROMPT_VERSION = "instruction-edit-units"
GROUNDING_ROUTES = {'add': ('target',), 'remove': ('source',), 'replace': ('source',),
                    'color': ('source',), 'motion': ('source',)}

@dataclass(frozen=True)
class GroundingRequest:
    grounding_image: str
    prompt: str


def prompt_version_for_mode(mode):
    if mode != 'two-pass':
        raise ValueError('Only the current two-pass method is supported')
    return PROMPT_VERSION


def canonicalize_type(raw_type):
    value = canonical_edit_type(raw_type)
    if value not in GROUNDING_ROUTES:
        raise ValueError(f'Unsupported CrispEdit mask type: {raw_type!r}')
    return value


def grounding_images(raw_type):
    return GROUNDING_ROUTES[canonicalize_type(raw_type)]


def build_change_observation_prompt(raw_type, instruction):
    return observation_prompt(canonicalize_type(raw_type), str(instruction or '').strip())


def build_grounding_prompt(raw_type, instruction, grounding_image, observation=None):
    if grounding_image not in grounding_images(raw_type):
        raise ValueError('Incorrect segmentation canvas')
    if observation is None:
        raise ValueError('Grounding requires the first-pass observation')
    return grounding_prompt(observation, grounding_image)


def build_grounding_requests(raw_type, instruction, observation=None):
    requests = []
    for side in grounding_images(raw_type):
        context = {'edit_summary': observation.get('edit_summary', ''), 'changes': [
            {**c, 'change_id': c.get('change_id', i)} for i, c in enumerate(observation['changes'])
            if _visible_ref(c.get(f'{side}_ref'))]}
        requests.append(GroundingRequest(side, build_grounding_prompt(raw_type, instruction, side, context)))
    return requests


def grounding_is_complete(etype, boxes_by_image):
    return all(bool(boxes_by_image.get(side)) for side in grounding_images(etype))


def _visible_ref(value: object) -> str:
    ref = str(value or "").strip()
    if ref.lower() in {
        "empty",
        "none",
        "nothing",
        "n/a",
        "na",
        "not present",
        "no object",
        "absent",
        "removed",
        "gone",
        "no longer present",
    }:
        return ""
    return ref


def parse_change_observation(text: str) -> Dict:
    """Accept complete JSON edit units; normalize persisted checklist records."""
    parsed = strict_json(text)
    if isinstance(parsed, list):
        parsed = {'edits': parsed}
    if not isinstance(parsed, dict):
        raise ValueError('observation must be an object or complete event array')
    if 'edits' in parsed:
        parsed = {'changes': flatten_edit_units(parsed)}
    summary = str(parsed.get("edit_summary", parsed.get("summary", ""))).strip()
    checked_regions = []
    raw_checks = parsed.get("checked_regions", [])
    if raw_checks is not None and not isinstance(raw_checks, list):
        raise ValueError("observation checked_regions must be a list")
    for index, item in enumerate(raw_checks or []):
        if not isinstance(item, dict):
            raise ValueError(f"checked region {index} is not an object")
        ref = str(item.get("ref", "")).strip()
        if not ref:
            raise ValueError(f"checked region {index} has an empty ref")
        changed = item.get("changed", False)
        if isinstance(changed, str):
            changed = changed.strip().lower() in {"true", "yes", "1"}
        checked = {"ref": ref, "changed": bool(changed)}
        if "source_appearance" in item:
            checked["source_appearance"] = str(item.get("source_appearance", "")).strip()
        if "target_appearance" in item:
            checked["target_appearance"] = str(item.get("target_appearance", "")).strip()
        checked_regions.append(checked)
    changes = parsed.get("changes")
    if not isinstance(changes, list):
        raise ValueError("observation changes must be a list")
    normalized = []
    for index, item in enumerate(changes):
        if not isinstance(item, dict):
            raise ValueError(f"observation change {index} is not an object")
        source_ref = _visible_ref(item.get("source_ref", ""))
        target_ref = _visible_ref(item.get("target_ref", ""))
        sam_ref = str(item.get("sam_ref", source_ref or target_ref)).strip()
        region_description = str(
            item.get("region_description", item.get("spatial_extent", ""))
        ).strip()
        raw_layout = str(item.get("region_layout", "single")).strip().lower()
        layout_aliases = {
            "single": "single",
            "single_object": "single",
            "object": "single",
            "nearby_group": "nearby_group",
            "aggregate": "nearby_group",
            "aggregate_region": "nearby_group",
            "separate_regions": "separate_regions",
        }
        region_layout = layout_aliases.get(raw_layout, "single")
        change = str(item.get("change", item.get("description", ""))).strip()
        if not source_ref and not target_ref:
            raise ValueError(f"observation change {index} has no visible source/target ref")
        if not change:
            raise ValueError(f"observation change {index} has an empty change description")
        aligned = item.get("instruction_aligned", True)
        if isinstance(aligned, str):
            aligned = aligned.strip().lower() in {"true", "yes", "1"}
        normalized.append(
            {
                "source_ref": source_ref,
                "target_ref": target_ref,
                "sam_ref": sam_ref,
                "region_description": region_description,
                "region_layout": region_layout,
                "change": change,
                "instruction_aligned": bool(aligned),
                **({key: item[key] for key in ("edit_id", "source_location", "target_location")}
                   if "edit_id" in item else {}),
            }
        )
    # An explicitly empty, valid checklist is a visual no-edit finding, not a
    # JSON failure to retry until the model invents an object.
    return {
        "edit_summary": summary,
        "checked_regions": checked_regions,
        "changes": normalized,
    }
