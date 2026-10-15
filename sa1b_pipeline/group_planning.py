"""One-call planners for twin and adjacent multi-target groups."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .geometry import load_mask
from .group_geometry import adjacent_pairs, twin_pair_gate, object_features, precise_point
from .instruction_protocol import make_spec, render
from .instructions import build_unit, validate_instruction
from .mllm import parse_json


COLORS = [(255, 66, 66), (54, 180, 255), (255, 190, 40), (144, 88, 255), (35, 205, 120)]


TWIN_PROMPT = r'''Plan one counterfactual twin edit group. Image 1 is the clean original scene. Image 2 has one row per candidate: a clean source crop on the LEFT and its aligned black-and-white mask on the RIGHT. White identifies the target; black is excluded. Unit and object IDs are printed above each row. Judge appearance only from clean photographs; the mask and headers are localization aids.

The candidates are provisional geometric/category matches, not verified semantic instances. Dense labels may be wrong; verify that all white masks actually cover distinct instances of the same noun. Use the original pixels and the masks to verify the noun and write a short, unique natural reference for each candidate. One shared edit must be natural for every member. The operation is assigned and cannot be changed or silently rejected as another operation. If the assigned operation is not natural for all members, return feasible=false.

The no-ref instruction must be identical for all members and be rendered from one shared structured specification. References are concise noun phrases that distinguish members in the unmarked original image. Do not mention annotation colors, IDs, masks or crops. Genuine source colors and concise image-left/image-right references are allowed when needed for unique identification. For twin add, use only a directly supporting host/attachment site shared by every member; reference anchors are not allowed. Reject the group if no natural directly attached addition works for all members. For remove, never remove human anatomy or an essential support. For replace, change visible form/design, not only color. For attribute, change one plausible requested property that is visibly different on every member.

Return JSON only:
{"feasible":true,"operation":"...","scope":"short shared noun","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","members":[{"object_id":"...","ref_phrase":"the ...","source_color":"","evidence":"brief"},...],"evidence":"brief"}
The operation must be the assigned operation. attributes is [{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}] only for attribute. placement for add is {"relation":"on|in|around|to|...","site":"optional noun","anchor_mode":"support"}; direct support is required for twins. remove_mode is whole, part, contents, contents_only, lowerlegs or hem. Keep the edit one concise sentence after rendering.
FIELD CONTRACT: scope is ONE shared singular object/part noun, e.g. wheel rim, never a full sentence or metadata alias list. It must describe the visible white mask itself, not an imagined subpart or the whole host of a part mask. target_part MUST be "": these candidates have no additional resolved execution sub-mask. If the mask is already a rim, use scope="wheel rim" and target_part="". Do not reinterpret a building mask as only its facade or a person mask as only their clothes.
For remove: payload="", target_part="", attributes=[], placement=null, qualifier="". Example shared fields: scope="street lamp", operation="remove". Never put the instruction sentence in payload.
For attribute: payload="", attributes=[{"property":"color","value":"blue"}], target_part="", placement=null, qualifier="". The property must naturally apply to the selected mask, preserve other surfaces, and differ from EVERY member's original appearance.
For replace: payload names a different physical object/design (e.g. "a five-spoke alloy wheel"), never just "magenta" or a different paint color. target_part="", attributes=[], placement=null, qualifier="".
For add: payload names a conventional new physical item, placement describes its direct support/attachment to each selected host; attributes=[], target_part="", qualifier="". Do not plant a shrub directly in solid rock or attach items to a painted line. No vague "small"/"identical" qualifiers: encode necessary size/design in payload.
Use natural edit content. A plausible refusal is better than inventing a source object, narrowing the mask, or requesting an operation that does not fit. Return feasible=false with a concrete reason if any member cannot satisfy the assigned operation.
'''


MULTI_PROMPT = r'''Plan one joint editing group from the original scene. Image 1 is the clean original scene. Image 2 has one row per candidate: a clean source crop on the LEFT and its aligned black-and-white mask on the RIGHT. White identifies the target; black is excluded. Unit and object IDs are printed above each row. Read all appearance from clean photographs only. Select every provided candidate exactly once and give each a natural, concise instruction grounded in the unmarked original image.

Operations and target order are assigned by the caller. Do not change an operation, merge targets, add fallback targets, or refer to a target created by an earlier step. Each ref must uniquely identify its own original target. Verify color words against the unmarked pixels; never call a black object red because a small detail is red. If color is uncertain, use a stable position or host instead. At least one preselected pair is adjacent. Avoid conflicting edits such as removing a host while editing its part. Add uses the directly supporting host first. Remove never removes human anatomy or essential support; replace changes form/design; attribute changes one visible property.

Return JSON only:
{"feasible":true,"generation_order":["U1",...],"targets":[{"unit_id":"U1","object_id":"...","operation":"...","scope":"...","ref_phrase":"the ...","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","evidence":"brief"}],"evidence":"brief"}
Use exactly the assigned operation for each target and exactly the assigned target IDs. No annotation-color, mask, ID or crop references in instructions; genuine source colors are allowed. scope is only the short object/part noun, without articles or instance/location qualifiers; put those in ref_phrase.
Field requirements are strict: every target must have a nonempty scope and ref_phrase. For
attribute, set exactly one nonempty attributes item with property and value that is
visibly different from the selected source (never repeat a color, material, finish or
pattern already used in its ref or scope), and leave payload empty. For add, set a concrete new-object noun in payload and a non-null
placement with relation and anchor_mode; choose anchor_mode=support whenever the new
object can sit on or attach directly to the masked host. For replace, set a concrete
replacement object/design in payload and leave attributes empty. For remove, leave
payload and attributes empty and use a valid remove_mode. Never emit an empty object,
empty list, or null value for a field required by its operation.
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


def _crop_sheet(source: Image.Image, objects: list[dict[str, Any]], masks: list[np.ndarray], *, locator_outline: bool = False) -> Image.Image:
    """Aligned clean crop and binary locator; never tint source appearance."""
    width, height, header = 512, 320, 32
    sheet = Image.new("RGB", (width * (3 if locator_outline else 2), (height + header) * len(objects)), (35, 35, 35))
    draw = ImageDraw.Draw(sheet)
    for index, (obj, mask) in enumerate(zip(objects, masks)):
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError("empty group target mask")
        extent = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        margin = max(24, int(extent * .45))
        box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
               min(source.width, int(xs.max()) + margin + 1), min(source.height, int(ys.max()) + margin + 1))
        clean = source.crop(box).convert("RGB")
        locator = Image.fromarray(mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8) * 255).convert("RGB")
        scale = min(width / clean.width, height / clean.height)
        size = (max(1, round(clean.width * scale)), max(1, round(clean.height * scale)))
        top = index * (height + header)
        for column, (tile, label) in enumerate(((clean, "SOURCE"), (locator, "MASK: WHITE = TARGET"))):
            draw.text((column * width + 8, top + 8),
                      f"U{index+1} / object {obj['object_id']} - {label}", fill="white")
            resized = tile.resize(size, Image.Resampling.LANCZOS if column == 0 else Image.Resampling.NEAREST)
            sheet.paste(resized, (column * width + (width - size[0]) // 2,
                                 top + header + (height - size[1]) // 2))
        if locator_outline:
            # The optional third column adds a contour outside the selector;
            # the first column remains the untouched appearance reference.
            photo = clean.resize(size, Image.Resampling.LANCZOS)
            binary = locator.convert("L").resize(size, Image.Resampling.NEAREST)
            expanded = np.asarray(binary.filter(ImageFilter.MaxFilter(7))) > 0
            inside = np.asarray(binary) > 0
            arr = np.asarray(photo).copy()
            arr[expanded & ~inside] = (255, 220, 30)
            draw.text((2 * width + 8, top + 8), f"U{index+1} - OUTSIDE CONTOUR: LOCATION ONLY", fill="white")
            sheet.paste(Image.fromarray(arr), (2 * width + (width-size[0])//2,
                                              top+header+(height-size[1])//2))
    return sheet


def group_views(candidate: dict[str, Any], objects: list[dict[str, Any]], directory: str | Path,
                *, image_size: tuple[int, int] | None = None, locator_outline: bool = False) -> tuple[list[str], list[str], str]:
    """Build two stable images plus binary mask files for a single group call."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(candidate["image_path"]).convert("RGB")
    masks = [load_mask(obj, source.size) for obj in objects]
    key = json.dumps({"version": "clean-binary-outline-v3" if locator_outline else "clean-binary-v2",
                      "source": hashlib.sha256(source.tobytes()).hexdigest(),
                      "ids": [o.get("object_id") for o in objects],
                      "masks": [hashlib.sha256(mask.tobytes()).hexdigest() for mask in masks]}, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:20]
    root = directory / digest
    root.mkdir(exist_ok=True)
    full = root / "clean_scene.png"
    marked = root / "marked_scene.png"
    sheet = root / "member_crops.png"
    mask_paths = []
    if not full.exists():
        clean = source.copy()
        clean.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        clean.save(full, compress_level=1)
    if not marked.exists():
        _overlay(source, objects, masks).save(marked, compress_level=1)
    if not sheet.exists():
        _crop_sheet(source, objects, masks, locator_outline=locator_outline).save(sheet, compress_level=1)
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


def _shared_raw(plan: dict[str, Any], ref: str, object_id: str, unit_index: int,
                *, scope: str | None = None, source_color: str = "") -> dict[str, Any]:
    return {"feasible": True, "object_id": object_id, "unit_index": unit_index,
            "operation": plan["operation"], "ref_phrase": ref,
            "category": scope or plan.get("scope", "object"), "scope": scope or plan["scope"],
            "payload": plan.get("payload", ""), "target_part": plan.get("target_part", ""),
            "remove_mode": plan.get("remove_mode", "whole"), "placement": plan.get("placement"),
            "attributes": plan.get("attributes", []), "qualifier": plan.get("qualifier", ""),
            "source_color": source_color, "evidence": plan.get("evidence", "")}


def build_twin_group(candidate: dict[str, Any], objects: list[dict[str, Any]], plan: dict[str, Any],
                     mask_paths: list[str], *, split: str, scope_level: str, group_id: str,
                     geometry: list[dict[str, Any]], assigned_operation: str | None = None) -> dict[str, Any]:
    if assigned_operation is not None and plan.get("operation") != assigned_operation:
        raise ValueError("twin assigned operation mismatch")
    if split == "eval" and len(objects) != 2:
        raise ValueError("evaluation twins must have exactly two members")
    if plan.get("operation") == "add" and (plan.get("placement") or {}).get("anchor_mode") != "support":
        raise ValueError("twin add requires a direct support anchor")
    if plan.get("feasible") is not True or plan.get("operation") not in {"add", "replace", "remove", "attribute"}:
        raise ValueError("infeasible or invalid twin plan")
    members = plan.get("members")
    if not isinstance(members, list) or len(members) != len(objects):
        raise ValueError("twin plan member count mismatch")
    by_id = {str(m.get("object_id")): m for m in members}
    if len(by_id) != len(members):
        raise ValueError("duplicate twin object ids")
    # The planner verifies one noun for all masks. Dense categories are hints,
    # not authoritative scope names (e.g. corrugated slat versus a full shutter).
    shared_scope = str(plan.get("scope") or "").strip()
    if not shared_scope:
        raise ValueError("missing shared twin scope")
    if plan.get("operation") == "attribute":
        attrs = plan.get("attributes") or []
        target_values = [str(a.get("value", "")).strip().lower() for a in attrs if isinstance(a, dict)]
        for item in members:
            evidence = f"{item.get('ref_phrase', '')} {item.get('source_color', '')}".lower()
            if any(value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", evidence)
                   for value in target_values):
                raise ValueError("attribute value is already present in a twin member description")
    if len(objects) not in (2, 3) or len(mask_paths) != len(objects) or len(geometry) != len(objects):
        raise ValueError("twin mask/geometry count mismatch")
    areas = [float(g.get("area_fraction", 0)) for g in geometry]
    if min(areas) <= 0 or max(areas) > 3 * min(areas):
        raise ValueError("twin exact mask area ratio exceeds 3")
    units = []
    for index, (obj, mask_path, geom) in enumerate(zip(objects, mask_paths, geometry), 1):
        item = by_id.get(str(obj["object_id"]))
        if not item or not str(item.get("ref_phrase", "")).strip():
            raise ValueError("missing twin reference")
        raw = _shared_raw(plan, item["ref_phrase"], str(obj["object_id"]), index,
                          scope=shared_scope, source_color=str(item.get("source_color", "")))
        unit = build_unit(raw, {**obj, "metadata": obj.get("metadata", {})}, mask_path, geom, [])
        errors = validate_instruction(unit)
        if errors:
            raise ValueError("invalid twin instruction: " + ",".join(errors))
        units.append(unit)
    twin_refs = [u.ref_phrase.strip().lower() for u in units]
    if len(set(twin_refs)) != len(twin_refs):
        raise ValueError("twin references are not unique")
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
    if len(by_id) != len(targets):
        raise ValueError("duplicate multi object ids")
    units = []
    for index, (obj, mask_path, geom) in enumerate(zip(objects, mask_paths, geometry), 1):
        item = by_id.get(str(obj["object_id"]))
        expected_unit_id = f"U{index}"
        if (not item or str(item.get("unit_id")) != expected_unit_id
                or item.get("operation") != assignments.get(str(obj["object_id"]))):
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
    multi_refs = [u.ref_phrase.strip().lower() for u in units]
    if len(set(multi_refs)) != len(multi_refs):
        raise ValueError("multi references are not unique")
    order_ids = plan.get("generation_order") or [u.unit_id for u in sorted(units, key=lambda u: (u.operation != "remove", u.operation != "replace", u.operation != "attribute"))]
    if (not isinstance(order_ids, list) or len(order_ids) != len(units)
            or set(order_ids) != {u.unit_id for u in units}):
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
