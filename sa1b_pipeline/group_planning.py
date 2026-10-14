"""One-call planners for twin and adjacent multi-target groups."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import load_mask
from .group_geometry import adjacent_pairs, twin_pair_gate, object_features, precise_point
from .instruction_protocol import make_spec, render
from .instructions import build_unit, validate_instruction
from .mllm import parse_json


COLORS = [(255, 66, 66), (54, 180, 255), (255, 190, 40), (144, 88, 255), (35, 205, 120)]


TWIN_PROMPT = r'''Plan one counterfactual twin edit group. The original scene is the first image; the second image is a compact contact sheet of the same candidate instances. Each candidate has a different solid outline/color and its ID is printed in the layout only. The colors and labels are localization aids, never source content.

The candidates are already prefiltered as two or three distinct visible instances of the same noun. Use the original pixels and the masks to verify the noun and write a short, unique natural reference for each candidate. One shared edit must be natural for every member. The operation is assigned and cannot be changed or silently rejected as another operation. If the assigned operation is not natural for all members, return feasible=false.

The no-ref instruction must be identical for all members and be rendered from one shared structured specification. References are concise noun phrases that distinguish members in the unmarked original image. Do not mention masks, colors, IDs, crops, annotations or image coordinates. For add, select the directly supporting host/attachment site; use a reference anchor only when direct attachment is clearly unnatural. For remove, never remove human anatomy or an essential support. For replace, change visible form/design, not only color. For attribute, change one plausible requested property that is visibly different on every member.

Return JSON only:
{"feasible":true,"operation":"...","scope":"short noun phrase","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","members":[{"object_id":"...","ref_phrase":"the ...","evidence":"brief"},...],"evidence":"brief"}
The operation must be the assigned operation. attributes is [{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}] only for attribute. placement for add is {"relation":"on|in|around|to|...","site":"optional noun","anchor_mode":"support|reference"}; direct support is preferred. remove_mode is whole, part, contents, contents_only, lowerlegs or hem. Keep the edit one concise sentence after rendering.
'''


MULTI_PROMPT = r'''Plan one joint editing group from the original scene. The first image marks all candidates with distinct colors and printed IDs; the second is a contact sheet for the preselected adjacent candidates. Markings and labels are localization aids only. Select every provided candidate exactly once and give each a natural, concise instruction grounded in the unmarked original image.

Operations and target order are assigned by the caller. Do not change an operation, merge targets, add fallback targets, or refer to a target created by an earlier step. Each ref must uniquely identify its own original target. At least one preselected pair is adjacent. Avoid conflicting edits such as removing a host while editing its part. Add uses the directly supporting host first. Remove never removes human anatomy or essential support; replace changes form/design; attribute changes one visible property.

Return JSON only:
{"feasible":true,"generation_order":["U1",...],"targets":[{"unit_id":"U1","object_id":"...","operation":"...","scope":"...","ref_phrase":"the ...","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","evidence":"brief"}],"evidence":"brief"}
Use exactly the assigned operation for each target and exactly the assigned target IDs. No masks, colors, labels or crop references in instructions.
'''


def _overlay(source: Image.Image, objects: list[dict[str, Any]], masks: list[np.ndarray], *, labels: bool = True) -> Image.Image:
    canvas = source.convert("RGB").copy()
    arr = np.asarray(canvas).copy()
    for mask, color in zip(masks, COLORS):
        arr[mask] = (0.55 * arr[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    canvas = Image.fromarray(arr)
    draw = ImageDraw.Draw(canvas)
    for obj, mask, color in zip(objects, masks, COLORS):
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=max(2, round(max(canvas.size) / 700)))
        if labels:
            label = f"{obj['object_id']} / {obj.get('category','object')}"
            draw.rectangle((x0, max(0, y0 - 20), min(canvas.width, x0 + 12 * len(label) + 8), y0), fill=color)
            draw.text((x0 + 4, max(0, y0 - 18)), label, fill="black", font=ImageFont.load_default())
    canvas.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    return canvas


def _crop_sheet(source: Image.Image, objects: list[dict[str, Any]], masks: list[np.ndarray]) -> Image.Image:
    tiles = []
    for i, (obj, mask) in enumerate(zip(objects, masks), 1):
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        margin = max(24, int(extent * .45))
        box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
               min(source.width, int(xs.max()) + margin + 1), min(source.height, int(ys.max()) + margin + 1))
        tile = source.crop(box).convert("RGB")
        local = mask[box[1]:box[3], box[0]:box[2]]
        ar = np.asarray(tile).copy()
        color = COLORS[i - 1]
        ar[local] = (0.45 * ar[local] + 0.55 * np.asarray(color)).astype(np.uint8)
        tile = Image.fromarray(ar)
        tile.thumbnail((420, 300), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (440, 340), "white")
        panel.paste(tile, ((440 - tile.width) // 2, 32 + (300 - tile.height) // 2))
        ImageDraw.Draw(panel).text((10, 8), f"{obj['object_id']}  {obj.get('category','object')}", fill="black")
        tiles.append(panel)
    if not tiles:
        return Image.new("RGB", (440, 340), "white")
    cols = min(2, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 440, rows * 340), (235, 235, 235))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 440, (i // cols) * 340))
    return sheet


def group_views(candidate: dict[str, Any], objects: list[dict[str, Any]], directory: str | Path,
                *, image_size: tuple[int, int] | None = None) -> tuple[list[str], list[str], str]:
    """Build two stable images plus binary mask files for a single group call."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(candidate["image_path"]).convert("RGB")
    masks = [load_mask(obj, source.size) for obj in objects]
    key = json.dumps({"source": candidate.get("image_sha256"), "ids": [o.get("object_id") for o in objects]}, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    root = directory / digest
    root.mkdir(exist_ok=True)
    full = root / "marked_scene.png"
    sheet = root / "member_crops.png"
    mask_paths = []
    if not full.exists():
        _overlay(source, objects, masks).save(full, compress_level=1)
    if not sheet.exists():
        _crop_sheet(source, objects, masks).save(sheet, compress_level=1)
    for obj, mask in zip(objects, masks):
        path = root / f"mask_{obj['object_id']}.png"
        if not path.exists():
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, compress_level=1)
        mask_paths.append(str(path))
    return [str(full), str(sheet)], mask_paths, digest


def twin_prompt(assigned_operation: str, objects: list[dict[str, Any]]) -> str:
    return TWIN_PROMPT + "\nAssigned operation: " + assigned_operation + "\nCandidates:\n" + "\n".join(
        f"- object_id={o['object_id']}, category={o.get('category','object')}" for o in objects)


def multi_prompt(assignments: dict[str, str], objects: list[dict[str, Any]], order: list[str]) -> str:
    return MULTI_PROMPT + "\nAssigned operations:\n" + "\n".join(
        f"- object_id={o['object_id']}, unit_id=U{i+1}, operation={assignments[o['object_id']]}" for i, o in enumerate(objects)
    ) + "\nDefault generation order: " + ", ".join(order)


def _shared_raw(plan: dict[str, Any], ref: str, object_id: str, unit_index: int) -> dict[str, Any]:
    return {"feasible": True, "object_id": object_id, "unit_index": unit_index,
            "operation": plan["operation"], "ref_phrase": ref,
            "category": plan.get("scope", "object"), "scope": plan["scope"],
            "payload": plan.get("payload", ""), "target_part": plan.get("target_part", ""),
            "remove_mode": plan.get("remove_mode", "whole"), "placement": plan.get("placement"),
            "attributes": plan.get("attributes", []), "qualifier": plan.get("qualifier", ""),
            "evidence": plan.get("evidence", "")}


def build_twin_group(candidate: dict[str, Any], objects: list[dict[str, Any]], plan: dict[str, Any],
                     mask_paths: list[str], *, split: str, scope_level: str, group_id: str,
                     geometry: list[dict[str, Any]]) -> dict[str, Any]:
    if plan.get("feasible") is not True or plan.get("operation") not in {"add", "replace", "remove", "attribute"}:
        raise ValueError("infeasible or invalid twin plan")
    members = plan.get("members")
    if not isinstance(members, list) or len(members) != len(objects):
        raise ValueError("twin plan member count mismatch")
    by_id = {str(m.get("object_id")): m for m in members}
    units = []
    for index, (obj, mask_path, geom) in enumerate(zip(objects, mask_paths, geometry), 1):
        item = by_id.get(str(obj["object_id"]))
        if not item or not str(item.get("ref_phrase", "")).strip():
            raise ValueError("missing twin reference")
        raw = _shared_raw(plan, item["ref_phrase"], str(obj["object_id"]), index)
        unit = build_unit(raw, {**obj, "metadata": obj.get("metadata", {})}, mask_path, geom, [])
        errors = validate_instruction(unit)
        if errors:
            raise ValueError("invalid twin instruction: " + ",".join(errors))
        units.append(unit)
    spec = dict(units[0].instruction_spec)
    no_ref = units[0].instruction_noref
    if any(u.instruction_noref != no_ref for u in units):
        raise ValueError("shared spec did not render one no-ref instruction")
    member_rows = []
    for u, geom in zip(units, geometry):
        member_rows.append({"unit_id": u.unit_id, "object_id": u.object_id, "ref": u.ref_phrase,
                            "instruction": u.instruction_ref, "instruction_noref": no_ref,
                            "instruction_spec": u.instruction_spec, "mask": u.mask_path,
                            "bbox": u.bbox, "point": u.point, "support_mask": None,
                            "support_policy": u.support_policy, "edited_image": None,
                            "generation_context": {}, "seed": 0, "deterministic": {},
                            "semantic_audit": {}, "area_fraction": geom["area_fraction"],
                            "operation": u.operation, "scope": u.scope, "payload": u.payload,
                            "mask_role": u.mask_role, "mask_semantics": u.mask_semantics,
                            "unit": u.to_dict()})
    return {"schema_version": "sa1b.samtok.group.v1", "twin_id": group_id,
            "source_id": candidate["source_id"], "source_image": candidate["image_path"],
            "source_provenance": candidate.get("provenance", {}), "operation": units[0].operation,
            "instruction_spec": {**spec, "ref": ""}, "instruction_noref": no_ref,
            "scope_level": scope_level, "members": member_rows, "split": split,
            "group_checks": {"plan": {"status": "pass", "model_evidence": plan.get("evidence", "")},
                             "group_gate": "pending_generation"}, "merged_target": None}


def build_multi_group(candidate: dict[str, Any], objects: list[dict[str, Any]], plan: dict[str, Any],
                      mask_paths: list[str], *, split: str, group_id: str,
                      geometry: list[dict[str, Any]], assignments: dict[str, str]) -> dict[str, Any]:
    if plan.get("feasible") is not True:
        raise ValueError("infeasible multi plan")
    targets = plan.get("targets")
    if not isinstance(targets, list) or len(targets) != len(objects):
        raise ValueError("multi plan target count mismatch")
    by_id = {str(t.get("object_id")): t for t in targets}
    units = []
    for index, (obj, mask_path, geom) in enumerate(zip(objects, mask_paths, geometry), 1):
        item = by_id.get(str(obj["object_id"]))
        if not item or item.get("operation") != assignments.get(str(obj["object_id"])):
            raise ValueError("multi operation or object mismatch")
        raw = {"feasible": True, "object_id": str(obj["object_id"]), "unit_index": index,
               "operation": item["operation"], "ref_phrase": item["ref_phrase"],
               "category": item.get("scope", "object"), "scope": item["scope"],
               "payload": item.get("payload", ""), "target_part": item.get("target_part", ""),
               "remove_mode": item.get("remove_mode", "whole"), "placement": item.get("placement"),
               "attributes": item.get("attributes", []), "qualifier": item.get("qualifier", ""),
               "evidence": item.get("evidence", "")}
        unit = build_unit(raw, {**obj, "metadata": obj.get("metadata", {})}, mask_path, geom, [])
        errors = validate_instruction(unit)
        if errors:
            raise ValueError("invalid multi instruction: " + ",".join(errors))
        units.append(unit)
    order_ids = plan.get("generation_order") or [u.unit_id for u in sorted(units, key=lambda u: (u.operation != "remove", u.operation != "replace", u.operation != "attribute"))]
    if set(order_ids) != {u.unit_id for u in units}:
        raise ValueError("invalid generation order")
    member_rows = []
    for u, geom in zip(units, geometry):
        member_rows.append({"unit_id": u.unit_id, "object_id": u.object_id, "ref": u.ref_phrase,
                            "instruction": u.instruction_ref, "instruction_noref": u.instruction_noref,
                            "instruction_spec": u.instruction_spec, "mask": u.mask_path,
                            "bbox": u.bbox, "point": u.point, "support_mask": None,
                            "area_fraction": geom["area_fraction"], "step_input_image": None,
                            "step_output_image": None, "generation_context": {}, "step_audit": {},
                            "operation": u.operation, "scope": u.scope, "payload": u.payload,
                            "mask_role": u.mask_role, "mask_semantics": u.mask_semantics,
                            "unit": u.to_dict()})
    return {"schema_version": "sa1b.samtok.group.v1", "multi_id": group_id,
            "source_id": candidate["source_id"], "source_image": candidate["image_path"],
            "source_provenance": candidate.get("provenance", {}), "n_targets": len(units),
            "operations": [u.operation for u in units], "generation_order": order_ids,
            "adjacent_pairs": [], "units": member_rows, "joint_target": None,
            "joint_audit": {"status": "pending_generation"}, "split": split}
