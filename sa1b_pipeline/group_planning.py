"""One-call planners for twin and adjacent multi-target groups."""
from __future__ import annotations

from .group_geometry import candidate_category

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .geometry import load_mask
from .group_geometry import adjacent_pairs, twin_pair_gate, object_features, precise_point
from .instruction_protocol import make_spec, render, BLIND_RULE
from .instruction_style import INSTRUCTION_STYLE
from .instructions import build_unit, validate_instruction
from .mllm import parse_json


COLORS = [(255, 66, 66), (54, 180, 255), (255, 190, 40), (144, 88, 255), (35, 205, 120)]


TWIN_PROMPT = r'''Plan ONE counterfactual twin group in ONE response. Image 1 is the clean original scene. Image 2 has aligned source crops and binary masks, one row per candidate. White selects the source region, black excludes it. IDs only connect rows to records; never put IDs or annotation colors in instructions.

First identify EACH mask independently from pixels. Candidate labels are deliberately withheld: the geometric preselection does not prove semantic sameness. Describe the selected physical entity, its visible extent and one excluded neighboring part in at most 16 words. Infer a mechanical part from its attachment/function, not just its color or rectangular outline. Do not call every member the same noun merely to create a group. A strap alone, a bag plus strap and a pouch are not interchangeable extents. A partial colored patch cannot authorize deleting the whole object. Occlusion, image clipping and small size alone are NOT rejection reasons when the visible target and request are clear.

Then decide whether ALL masks represent distinct editable instances at COMPARABLE granularity (all whole objects or all parts), possibly DIFFERENT categories, and whether the assigned operation is natural for ALL. Reject a mismatched group with feasible=false and one concrete reason. Never change the assigned operation or omit a member. Do not select new targets.

Before choosing words, find the intersection of practical edits appropriate to EVERY member in its current scene. Optimize for an ordinary photo-editing request, not a novel transformation merely possible to render. Different categories are welcome; shared FUNCTION, physical support and current state matter more than names. Reject only a concrete incompatibility, not small size, occlusion, clipping, unconventional but plausible taste, or a different category. Do not force a payload just to satisfy a quota.
For feasible groups choose ONE shared edit:
- add: a conventional object directly supported by every host, at an actually visible shared attachment site. scope/ref name the INITIAL mask entity; placement.site names its necessary subpart (e.g. seat). The pipeline later derives the source site mask from target difference intersected with host. Do not call the whole host its subpart. No distant/adjacent reference anchors. Keep site empty only when the selected entity itself is the direct site.
  Choose the shared PHYSICAL AFFORDANCE before the item: a visible attachable surface, hanging point, pocket/opening, or stable support plane. Prefer a modest attached accessory when both surfaces support it; free-standing objects require a visible level support with enough room in BOTH members. A chair seat and table are not automatically equally good places for drinking glasses. Printed/embroidered decorations must be named as such; do not confuse real accessories with printed motifs. A selected dark clothing panel is not a strap merely because a strap is nearby. If the source masks already select the attachment parts, leave site empty but still identify those parts accurately. Keep the new item easily visible at the source resolution without making it oversized. Do not restrict all scenes to stickers; choose an accessory appropriate to their actual shared use.
- replace: a compatible new physical form/design at the same extent, not merely another color/material. The new object must fit the function, scale and mounting of EACH original location. Name the functional object explicitly rather than generic hardware. Include the new design in payload; replacement_form is a short internal explanation.
- remove: erase the selected independent object/part naturally. Do not remove human anatomy, essential supports or only a carrying strap while leaving its bag unsupported. Do not broaden a fragment mask into the whole object.
- attribute: one plausible property with a new value different from EVERY member's current value. Prefer an everyday appearance variation suitable for the material and use (e.g. paintable products or fabric); do not recolor raw food/skin as paint or turn transparent windows into opaque paint. Preserve unrequested shape, markings and other components. Do not reinterpret a person as clothes or an entire mixed object as one surface.

Each ref is the shortest noun phrase that uniquely identifies its mask in the ORIGINAL scene. Prefer one verified distinguishing modifier; use an ordinal only with a clear counting set. For several similar objects on the same side, left/right alone is insufficient. No relative clauses, explanatory afterthoughts or invented landmarks. All members share the exact same no-ref sentence; each member keeps its own truthful scope. Never rename different objects as one class. The shared payload/value must be equally natural at EVERY member: chair and stool can share a bar-stool replacement; a car and bag cannot share leather material; removing an occupied chair leaves a person unsupported. Reject requests that clearly imply only one member. Actual source-mask areas must stay within a factor of three. Payload names new content explicitly, without 'a matching one' or 'like it'.
Compare the ref against the CLOSEST similar neighbor, not an easier distant object. In a dense repeated row, bound the counting set and verify the ordinal, or use one uniquely visible local relation; 'front row, right of center' is not enough if several objects match. For add/replace the payload describes ONE independent edit, never pluralize it merely because the group has two members. Default to a/an + singular item; multiple items require an explicit count or a natural collective such as a bunch of flowers.

Return valid JSON. Put members FIRST so each visual identification precedes the common edit. For every supplied ID include exactly one entry:
{"members":[{"object_id":"ID","scope":"own singular category","ref_phrase":"the ...","placement_site":"own attachment site or empty","grounding":{"entity":"independently identified noun","kind":"object|part","host":"host noun or empty","current_value":"source property or empty","distinction":"brief visible unique fact","mask_description":"white includes ...; excludes ...","site_visible":true,"edit_feasible":true}}],"feasible":true,"grounding_version":4,"operation":"ASSIGNED","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","replacement_form":"","shared_fit":[{"object_id":"ID","natural":true,"reason":"specific fit of chosen edit in this scene, at most 12 words"}],"evidence":"brief"}

Fields: target_part and qualifier are empty because these source candidates have no resolved additional part mask. remove_mode=whole. For add/replace payload is the concrete new object; for remove/attribute payload is empty. For attribute attributes=[{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}], otherwise attributes=[]. Only add has placement={"relation":"on|in|attached to|...","site":"part or empty","anchor_mode":"support"}; otherwise null. For non-add site_visible=true. Each grounding.entity must agree with its OWN member.scope. Different scopes are allowed. All grounding.kind values must agree. For add, each member.placement_site may differ; only relation/payload are shared. Do not force identical site nouns onto different hosts. After choosing the shared payload/value, provide shared_fit for EVERY ID. Check each member against that EXACT edit, including existing items, occupancy, support, installation, and unchanged materials. A generic statement such as "both can be edited" is not evidence. Payload is a concise noun phrase with an article; omit optional stylistic specifications. Keep all internal evidence brief. A rejected response needs only {"feasible":false,"reason":"specific visible incompatibility"}.
'''


MULTI_PROMPT = r'''Plan ONE multi-target edit group in ONE response. Image 1 is the clean original scene. Image 2 has THREE aligned columns per candidate row: clean SOURCE, binary MASK, and SELECTED PIXELS ONLY on neutral gray. The third column contains ONLY pixels inside that source mask, without changing their original color. Gray outside it is padding, not scene content. White selects the exact source entity; black excludes it. Read appearance from the clean photographs only. IDs connect rows to records, never enter instructions.

Identify EVERY mask independently before designing its edit. Name what is visible in the SELECTED PIXELS ONLY column, verifying its extent against the binary mask and clean source. Black holes exclude objects even when the outer contour surrounds them. Name its actual physical extent, not a convenient adjacent object or the most salient part of a larger mask. A complete car is not just its trunk; a canopy is not the whole boat. Dense labels are not evidence. Occlusion, clipping, size and unusual but plausible taste alone are not rejection reasons.
Operations and target IDs are assigned. Use every candidate exactly once without changing its operation. All refs and masks refer to the ORIGINAL scene, never a newly created object. Use the shortest unique noun phrase. Compare against the nearest similar neighbor in the full scene; use a verified ordinal or short host relation when left/right is ambiguous. No trailing explanatory clauses. Do not depend on the color/shape of another target that an earlier step will change. Prefer generation order remove, replace, attribute, add, while respecting dependencies. Reject destructive conflicts: deleting a host while editing its part, removing essential supports, or a replacement that occupies another target's location.

Choose an ordinary, physically plausible photo-editing request for each target:
- add: choose a visible direct support/attachment surface BEFORE a modest new item. Check current occupancy: do not add an accessory already present at that site. Printed decorations must be explicitly named as printed. A free-standing item needs a stable visible plane; attachable surfaces can receive a suitable accessory. Select a concrete singular item, not a distant object. scope/ref name the initial mask; placement.site is its necessary attachment subpart, or empty when the selected entity itself is that site. The pipeline derives the precise site mask from image differences. No remote/neighbor reference anchors.
- remove: erase an independently removable entity without leaving unsupported people, objects or body remnants. No human anatomy or essential functional supports. Do not broaden a fragment mask to an entire object.
- replace: choose a new physical form/design that fits the same location, mounting and scale. A new color/material alone is attribute, not replace. Do not replace only a part of a whole-object mask.
- attribute: one visible property, new value different from the source, appropriate to its material and use. Preserve shape, markings and unrelated materials. Do not paint skin/food or make transparent glass opaque. Do not reinterpret a whole person as clothing.

Return JSON only. Every target includes concise independent grounding BEFORE its edit fields:
{"feasible":true,"targets":[{"unit_id":"U1","object_id":"ID","grounding":{"entity":"exact scope noun","mask_description":"includes ...; excludes ...","current_value":"original attribute value or empty","edit_feasible":true,"reason":"specific physical fit, at most 12 words"},"scope":"singular object or part noun","ref_phrase":"the ...","operation":"ASSIGNED","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","evidence":"brief"}],"generation_order":["U1"],"evidence":"brief"}
Fields target_part and qualifier must be empty; remove_mode=whole. For add/replace payload is an explicit concise new-object noun phrase with an article; otherwise empty. Only attribute has attributes=[{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}]. Only add has placement={"relation":"on","site":"","anchor_mode":"support"}; all others null.
For add, relation must be EXACTLY one of: on, in, inside, to, around, across, along, over, at, attached to, clipped to, hanging from, resting on, mounted on, sewn onto, tied around, pinned to, hooked over, strapped over. A site goes in site, NEVER inside relation: use relation="on", site="the rear corner", not relation="on the rear corner of". Site must not repeat scope. grounding.entity must equal scope. Use feasible=false with one concrete reason if any assigned edit is incompatible; do not force a bad edit for a quota.
'''




def validate_twin_grounding(plan: dict[str, Any]) -> None:
    """Validate explicit contradictions, never treat self-reported fields as visual proof."""
    if plan.get("grounding_version") not in (2, 3, 4):
        raise ValueError("missing twin visual grounding")
    if plan.get("grounding_version") == 4:
        fit = plan.get("shared_fit")
        expected = {str(m.get("object_id")) for m in plan.get("members", [])}
        if (not isinstance(fit, list) or len(fit) != len(expected)
                or not all(isinstance(x, dict) for x in fit)
                or {str(x.get("object_id")) for x in fit} != expected
                or not all(x.get("natural") is True and str(x.get("reason", "")).strip() for x in fit)):
            raise ValueError("shared edit lacks per-member naturalness evidence")
    kinds = set()
    operation = plan.get("operation")
    for member in plan.get("members", []):
        binding = member.get("grounding") or {}
        scope = str(member.get("scope") or plan.get("scope", "")).strip().casefold()
        if not scope or str(binding.get("entity", "")).strip().casefold() != scope:
            raise ValueError("mask entity differs from member scope")
        if binding.get("kind") not in {"object", "part"}:
            raise ValueError("twin selector is not one object or part")
        if plan.get("grounding_version") == 4 and binding.get("host") and binding["kind"] == "object":
            # Explicit attached component nouns cannot become whole objects
            # merely because the model labels both members "object".
            from .group_geometry import candidate_granularity
            if candidate_granularity({"category": scope}) == "part":
                raise ValueError("attached component labeled as whole object")
        kinds.add(binding["kind"])
        if not str(binding.get("distinction", "")).strip():
            raise ValueError("missing full-scene distinction")
        if binding.get("edit_feasible") is not True:
            raise ValueError("shared edit infeasible for member")
        if operation == "add" and binding.get("site_visible") is not True:
            raise ValueError("shared add site is not visible on every member")
        if operation == "attribute":
            current = str(binding.get("current_value", "")).strip().casefold()
            if not current:
                raise ValueError("missing original attribute value")
            for attr in plan.get("attributes", []):
                target = str(attr.get("value", "")).strip().casefold()
                if target and (target == current or re.search(r"(?<!\w)" + re.escape(target) + r"(?!\w)", current)):
                    raise ValueError("attribute value already present in grounded source")
    if len(kinds) != 1:
        raise ValueError("twin instance granularity mismatch")
    if operation == "replace" and not str(plan.get("replacement_form", "")).strip():
        raise ValueError("missing replacement physical design difference")
    if operation == "replace":
        # Catch explicit material/color-only rewrites of the SAME noun. This
        # cannot certify a design change; remaining cases still need vision.
        property_words = set("a an the new different modern polished matte glossy shiny smooth rough "
            "wood wooden metal metallic steel stainless plastic ceramic glass marble stone concrete "
            "brass copper gold golden silver aluminum aluminium alloy leather fabric black white red "
            "blue green yellow orange purple pink gray grey brown beige bright dark light solid".split())
        payload_words = set(re.findall(r"[a-z]+", str(plan.get("payload", "")).casefold()))
        if payload_words and payload_words <= property_words | {'chrome','finish','texture','coating','paint','surface'}:
            raise ValueError('replacement payload names only an attribute, not a new physical object')
        for member in plan.get("members", []):
            scope_words = set(re.findall(r"[a-z]+", str(member.get("scope") or plan.get("scope", "")).casefold()))
            if payload_words and not (payload_words - scope_words - property_words):
                raise ValueError("replacement changes only material/color/finish of the same noun")


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


def _crop_sheet(source: Image.Image, objects: list[dict[str, Any]], masks: list[np.ndarray], *, locator_outline: bool = False, locator_cutout: bool = False) -> Image.Image:
    """Aligned clean crop and binary locator; never tint source appearance."""
    width, height, header = (512, 272, 32) if locator_cutout else (640, 400, 32)
    sheet = Image.new("RGB", (width * (3 if locator_outline or locator_cutout else 2), (height + header) * len(objects)), (35, 35, 35))
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
        if locator_cutout:
            # Retain the selected pixels' real color/texture. A contour alone
            # can ambiguously outline both a foreground object and its background.
            selected = Image.composite(clean, Image.new("RGB", clean.size, (100, 100, 100)), locator.convert("L"))
            draw.text((2 * width + 8, top + 8), f"U{index+1} - SELECTED PIXELS ONLY", fill="white")
            sheet.paste(selected.resize(size, Image.Resampling.LANCZOS),
                        (2*width+(width-size[0])//2, top+header+(height-size[1])//2))
        elif locator_outline:
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


def _located_scene(source: Image.Image, masks: list[np.ndarray]) -> Image.Image:
    """Link crop IDs to the full scene without tinting selected pixels."""
    canvas = source.convert("RGB").copy()
    canvas.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    scaled = [np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(
        canvas.size, Image.Resampling.NEAREST)) > 0 for mask in masks]
    union = np.logical_or.reduce(scaled)
    array = np.asarray(canvas).copy()
    for mask in scaled:
        expanded = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0
        array[expanded & ~union] = (255, 220, 30)
    canvas = Image.fromarray(array)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, mask in enumerate(scaled, 1):
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)
        # Place each tag outside all selectors, not over a neighboring target.
        for x, y in ((x0, y0-30), (x1+4, y0), (x0, y1+4), (x0-60, y0)):
            if x >= 0 and y >= 0 and x+56 <= canvas.width and y+28 <= canvas.height \
                    and not union[y:y+28,x:x+56].any():
                draw.rectangle((x, y, x+55, y+27), fill="black")
                draw.text((x+4, y+1), f"U{index}", font=font, fill=(255,220,30))
                break
    return canvas


def group_views(candidate: dict[str, Any], objects: list[dict[str, Any]], directory: str | Path,
                *, image_size: tuple[int, int] | None = None, locator_outline: bool = False,
                locate_in_scene: bool = False, locator_cutout: bool = False) -> tuple[list[str], list[str], str]:
    """Build two stable images plus binary mask files for a single group call."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(candidate["image_path"]).convert("RGB")
    masks = [load_mask(obj, source.size) for obj in objects]
    key = json.dumps({"version": "clean-binary-cutout-v5" if locator_cutout else ("clean-binary-outline-v4" if locator_outline else "clean-binary-v3"),
                      **({"scene_locator_version": 1} if locate_in_scene else {}),
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
        _crop_sheet(source, objects, masks, locator_outline=locator_outline, locator_cutout=locator_cutout).save(sheet, compress_level=1)
    for obj, mask in zip(objects, masks):
        path = root / f"mask_{obj['object_id']}.png"
        if not path.exists():
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path, compress_level=1)
        mask_paths.append(str(path))
    if locate_in_scene:
        located = root / "located_scene.png"
        if not located.exists():
            _located_scene(source, masks).save(located, compress_level=1)
        full = located
    return [str(full), str(sheet)], mask_paths, digest


def twin_prompt(assigned_operation: str, objects: list[dict[str, Any]]) -> str:
    return TWIN_PROMPT + "\n" + BLIND_RULE + "\nAssigned operation: " + assigned_operation + "\nCandidates:\n" + "\n".join(
        f"- object_id={o['object_id']} (must appear exactly once)" for o in objects)


def multi_prompt(assignments: dict[str, str], objects: list[dict[str, Any]], order: list[str]) -> str:
    return MULTI_PROMPT + "\n" + BLIND_RULE + "\n" + INSTRUCTION_STYLE + "\nAssigned operations:\n" + "\n".join(
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


def _validate_group_edit_fields(plan: dict[str, Any]) -> None:
    """Catch observed malformed fields before spending editor time.

    This catches explicit color/material-only replacements, not all semantic mismatches;
    source appearance and physical plausibility still require visual review.
    """
    operation = plan.get("operation")
    if operation == "add":
        site = str((plan.get("placement") or {}).get("site") or "").strip().lower()
        if site in {"empty", "none", "null", "optional", "optional noun", "n/a"}:
            raise ValueError("add placement site must be a location noun or empty string")
        scope = str(plan.get("scope") or "").strip().lower()
        if site and site.removeprefix("the ") == scope.removeprefix("the "):
            raise ValueError("add placement site repeats the host scope")
    if operation == "replace":
        colors = r"red|green|blue|yellow|orange|purple|pink|black|white|grey|gray|brown|silver|gold|beige|cyan|magenta"
        materials = r"wood|wooden|metal|metallic|steel|plastic|glass|ceramic|marble|leather|fabric|brass|copper|stone|concrete|matte|glossy|polished"
        modifiers = re.compile(r"\b(?:a|an|the|solid|bright|dark|light|" + colors + "|" + materials + r")\b", re.I)
        def without_color(value: Any) -> str:
            return " ".join(modifiers.sub(" ", str(value or "")).lower().split())
        scope = without_color(plan.get("scope"))
        payload = without_color(plan.get("payload"))
        reference = without_color(plan.get("ref_phrase"))
        if payload and payload in {scope, reference}:
            raise ValueError("replace needs a different physical form, not only a color or material")


def build_twin_group(candidate: dict[str, Any], objects: list[dict[str, Any]], plan: dict[str, Any],
                     mask_paths: list[str], *, split: str, scope_level: str, group_id: str,
                     geometry: list[dict[str, Any]], assigned_operation: str | None = None,
                     require_grounding: bool = False) -> dict[str, Any]:
    if "malformed_mllm_json" in plan.get("failure_reasons", []):
        raise ValueError("malformed planner JSON: " + str(plan.get("evidence", "")))
    if plan.get("feasible") is False:
        raise ValueError("infeasible twin plan: " + str(plan.get("reason") or plan.get("evidence") or "no shared natural edit"))
    if assigned_operation is not None and plan.get("operation") != assigned_operation:
        raise ValueError("twin assigned operation mismatch")
    if split == "eval" and len(objects) != 2:
        raise ValueError("evaluation twins must have exactly two members")
    if plan.get("operation") == "add" and (plan.get("placement") or {}).get("anchor_mode") != "support":
        raise ValueError("twin add requires a direct support anchor")
    if plan.get("feasible") is not True or plan.get("operation") not in {"add", "replace", "remove", "attribute"}:
        raise ValueError("infeasible or invalid twin plan")
    if require_grounding or plan.get("grounding_version") in (2, 3, 4):
        validate_twin_grounding(plan)
    _validate_group_edit_fields(plan)
    members = plan.get("members")
    if not isinstance(members, list) or len(members) != len(objects):
        raise ValueError("twin plan member count mismatch")
    by_id = {str(m.get("object_id")): m for m in members}
    if len(by_id) != len(members):
        raise ValueError("duplicate twin object ids")
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
                          scope=str(item.get("scope") or plan.get("scope") or ""), source_color=str(item.get("source_color", "")))
        if not raw["scope"]:
            raise ValueError("missing member scope")
        if raw["operation"] == "add" and "placement_site" in item:
            raw["placement"] = {**(raw["placement"] or {}), "site": item["placement_site"]}
        _validate_group_edit_fields({**plan, "scope": raw["scope"], "placement": raw["placement"]})
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
        grounding = by_id[str(u.object_id)].get("grounding") or {}
        member_rows.append({"unit_id": u.unit_id, "object_id": u.object_id, "ref": u.ref_phrase,
                            "instruction": u.instruction_ref, "instruction_noref": no_ref,
                            "instruction_spec": u.instruction_spec, "mask": u.mask_path,
                            "bbox": u.bbox, "point": u.point, "support_mask": None,
                            "support_policy": u.support_policy, "edited_image": None,
                            "generation_context": {}, "seed": 0, "deterministic": {},
                            "semantic_audit": {}, "area_fraction": geom["area_fraction"],
                            "operation": u.operation, "scope": u.scope, "payload": u.payload,
                            "granularity": grounding.get("kind") or ("object" if scope_level == "object" else "part"),
                            "grounding": grounding,
                            "mask_role": u.mask_role, "mask_semantics": u.mask_semantics,
                            "unit": u.to_dict()})
    return {"schema_version": "sa1b.samtok.group.v1",
            "instruction_protocol_version": "2.0", "mask_semantics_version": "2.0",
            "instruction_style_version": "benchmark-concise-3-blind-1",
            "twin_contract_version": "blind-cross-category-1",
            "category_relation": "same" if len({candidate_category({"category": u.scope}) for u in units}) == 1 else "cross",
            "twin_id": group_id,
            "source_id": candidate["source_id"], "source_image": candidate["image_path"],
            "source_provenance": candidate.get("provenance", {}), "operation": units[0].operation,
            "instruction_spec": {**spec, "ref": "", "scope": ""}, "instruction_noref": no_ref,
            "scope_level": scope_level, "members": member_rows, "split": split,
            "group_checks": {"plan": {"status": "pass", "model_evidence": plan.get("evidence", "")},
                             "group_gate": "pending_generation"}, "merged_target": None}


def build_multi_group(candidate: dict[str, Any], objects: list[dict[str, Any]], plan: dict[str, Any],
                      mask_paths: list[str], *, split: str, group_id: str,
                      geometry: list[dict[str, Any]], assignments: dict[str, str],
                      require_grounding: bool = False) -> dict[str, Any]:
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
        if require_grounding:
            grounding = item.get("grounding") or {}
            if (grounding.get("entity") != item.get("scope")
                    or grounding.get("edit_feasible") is not True
                    or not grounding.get("mask_description") or not grounding.get("reason")):
                raise ValueError("multi target lacks consistent visual grounding")
            if item.get("operation") == "attribute":
                current = str(grounding.get("current_value", "")).strip().casefold()
                if not current or any(str(a.get("value", "")).strip().casefold() == current
                                      for a in item.get("attributes", [])):
                    raise ValueError("multi attribute is missing source value or repeats it")
            if item.get("target_part") or item.get("qualifier") or item.get("remove_mode", "whole") != "whole":
                raise ValueError("multi edit exceeds the preselected source extent")
            if item.get("operation") == "add" and (item.get("placement") or {}).get("anchor_mode") != "support":
                raise ValueError("multi add requires a direct support anchor")
        _validate_group_edit_fields(item)
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
    return {"schema_version": "sa1b.samtok.group.v1",
            "instruction_protocol_version": "2.0", "mask_semantics_version": "2.0",
            "instruction_style_version": "benchmark-concise-3-blind-1", "multi_id": group_id,
            "source_id": candidate["source_id"], "source_image": candidate["image_path"],
            "source_provenance": candidate.get("provenance", {}), "n_targets": len(units),
            "operations": [u.operation for u in units], "generation_order": order_ids,
            "adjacent_pairs": [], "units": member_rows, "joint_target": None,
            "joint_audit": {"status": "pending_generation"}, "split": split}
