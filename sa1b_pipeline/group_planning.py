"""One-call planners for twin and adjacent multi-target groups."""
from __future__ import annotations

from .group_geometry import candidate_category
from .multi_semantics import MULTI_PLAN_VERSION, validate_multi_semantics, same_scope_identity

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from .fonts import load_font
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


MULTI_PROMPT = r'''Plan one multi-target edit group in ONE response. Image 1 is the clean original scene. Image 2 has three aligned columns per numbered target: SOURCE photograph, binary MASK, and SELECTED PIXELS ONLY on gray. White selects; black and gray padding exclude. Use the photograph for appearance and context, the mask/cutout for exact ownership. Row order is not scene order. Source labels are fallible hints.

Identify each selected entity before choosing its modification. Name its actual extent: clothing is not the wearer, a part is not its host, and a hole in a mask excludes its contents. Occlusion, small size and source blur alone are not reasons to reject an identifiable region.
Operations and target IDs are assigned. Keep every target and its assigned operation. Ref names the ORIGINAL entity with the shortest unique noun phrase: normally a noun plus one verified distinguishing modifier. Compare with similar neighbors, not only selected targets. Never add an explanatory clause, IDs or coordinates. Clothing itself does not walk or hold things: use a brief wearer phrase when needed. Avoid incidental left/right details unless needed for uniqueness; when needed distinguish image-relative direction from anatomical direction. Never guess body-relative left/right from screen position. Ref must not depend on another target's property that will change.

Design an ordinary edit that fits this exact region. First record visible_before (at most 12 words), then decide the edit. Prefer one modest, clearly visible change with a simple physical fit over an intricate installation. In reason state the specific fit, not a generic claim of naturalness.
- add: keep the host AND its existing objects, lettering and decorations. Choose an unoccupied visible surface within the selected entity, then a new item that can directly attach there. The item must be absent at that site BEFORE editing. Rewriting an existing sign/logo/design or recoloring an accessory is not add. A new printed motif is allowed only on a visibly free patch without replacing existing content. Avoid stacking a second functional accessory on an occupied site. Use a simple attachment that is fully visible; do not invent an occluded body zone. Prefer a side-free attachment when equally natural. placement.site is a short part noun without her/his/its/their or a repeated host; empty if the selected entity is already the site. The source site mask will be refined from the actual change. No remote or neighboring anchors.
- remove: remove a separable entity while leaving coherent remaining structure. Check attached contents, supported objects and the host first. Do not remove anatomy, core worn garments or a support that leaves remnants hanging. A vessel with upright flowers is not independently removable. If this exact operation is incompatible, mark infeasible; do not rename the mask or erase its neighbors to make it work.
- replace: choose a concrete different form/construction fitting the same footprint, mounting and function. It must visibly replace the selected entity, not add a second one. State a concrete structural difference, such as slats replacing a solid back or shorts replacing trousers. A compatible panel design is enough; radical geometry changes are unnecessary. Color/material alone is attribute. Preserve adjacent parts and the required support.
- attribute: change one visible property to a different plausible value. Preserve geometry and unrequested markings/components; requested text/pattern changes may change those markings. An opaque region cannot become opaque as a no-op. Do not reinterpret a whole person or head as clothing or hair.
Prefer remove, replace, attribute, add order unless dependencies require otherwise. Reject conflicts such as deleting the host of another edit. All requests must still make sense together. Never force a physically wrong operation just to meet a quota.

Return JSON only. Every target has independent grounding BEFORE edit fields:
{"grounding_version":3,"feasible":true,"targets":[{"unit_id":"U1","object_id":"ID","grounding":{"entity":"actual selected entity","mask_description":"selected extent, at most 12 words","visible_before":"visible pre-edit state or occupancy, at most 12 words","current_value":"original attribute value, else empty","edit_feasible":true,"reason":"specific physical fit, at most 12 words"},"ref_phrase":"the ...","operation":"ASSIGNED","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":"","evidence":"brief"}],"generation_order":["U1"],"evidence":"brief"}
Include every assigned target exactly once. grounding.entity is the sole source category; do not repeat a scope field. target_part and qualifier are empty; remove_mode is whole. Only add/replace has payload: a concise concrete noun phrase with an article. Only attribute has attributes=[{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}]. Opacity is transparent, translucent or opaque; frosted is a finish. Only add has placement={"relation":"on","site":"","anchor_mode":"support"}; otherwise null.
Add relation is one of: on, in, inside, to, around, across, along, over, at, attached to, clipped to, hanging from, resting on, mounted on, sewn onto, tied around, pinned to, hooked over, strapped over. Put a necessary attachment part in site, not relation. If an assigned operation cannot naturally fit, retain its target entry with edit_feasible=false, a concrete reason and feasible=false. Do not call a valid mask defective because its assigned operation is unsuitable.
'''




MULTI_GROUNDING_FIRST_PROMPT = r'''Plan ONE multi-target edit group in ONE response. Image 1 is the clean ORIGINAL scene. Image 2 is a numbered grid of SELECTED PIXELS ONLY crops. Every tile retains only its own mask's original pixels; uniform gray is excluded padding, including holes. Tile order is NOT scene order. Use the supplied full-scene boxes to locate each tile in Image 1. The boxes are locators, NOT editable regions. Never infer that an object inside a box or surrounded by selected pixels is selected.

Write the complete regions array FIRST, identifying ALL masks before choosing any edits. For each tile identify the selected material/entity itself, its extent, and the most easily confused EXCLUDED neighboring entity or part. Use the scene to understand context, but the cutout to decide ownership. A visible surface or clothing item is not its whole host. Background visible between foreground parts is background, not those foreground parts. Name a precise part when only that part is selected; do not rename it to satisfy its assigned operation. Clipping/occlusion alone does not invalidate an otherwise identifiable entity. If ownership is genuinely unclear, mark extent=uncertain and reject instead of guessing. Ref must name exactly this extent, using the shortest distinguishing phrase in the ORIGINAL scene. No extra explanatory clause. Do not write crop coordinates, IDs or annotation terms in refs.

After ALL regions, design the assigned edits. Never change the identified scope to accommodate a payload. Keep every target, including when infeasible; set its edit_feasible=false with a short reason. A group is feasible only if every assigned edit is natural at its exact selected extent. Rejecting an incompatible operation does not mean the source mask is defective.
''' + MULTI_PROMPT[MULTI_PROMPT.index('Operations and target IDs'):MULTI_PROMPT.index('Return JSON only.')] + r'''
Return JSON only, in this key order:
{"grounding_version":2,"regions":[{"unit_id":"U1","object_id":"ID","entity":"precise singular noun","ref_phrase":"the ...","extent":"object|part|surface|background|uncertain","mask_description":"selected visible material and extent, at most 12 words","excluded":"nearest excluded entity or part, or none"}],"targets":[{"unit_id":"U1","object_id":"ID","operation":"ASSIGNED","current_value":"value of the requested property BEFORE editing, or empty for non-attribute","edit_feasible":true,"reason":"specific physical fit or incompatibility, at most 12 words","payload":"","target_part":"","remove_mode":"whole","placement":null,"attributes":[],"qualifier":""}],"generation_order":["U1"],"feasible":true}
Include EVERY supplied ID exactly once in regions and targets. Region entity/ref are authoritative: do not duplicate or override scope/ref in targets. Do not omit targets just because one is incompatible. For add/replace payload is one concise explicit new-object noun phrase with an article; otherwise empty. Only attribute has attributes=[{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}]. Only add has placement={"relation":"on","site":"","anchor_mode":"support"}; otherwise null. target_part and qualifier are empty, remove_mode=whole.
Add relation must be exactly one of: on, in, inside, to, around, across, along, over, at, attached to, clipped to, hanging from, resting on, mounted on, sewn onto, tied around, pinned to, hooked over, strapped over. Necessary attachment subparts go in site, never in relation. Leave site empty if the selected entity itself is the attachment site. Do not repeat the host noun as site. Existing items and excluded neighbors cannot become the selected entity. An already present accessory cannot be an addition; changing only material/color cannot be replacement. Keep evidence concise.
'''


def normalize_multi_grounding(plan: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind grounding-first regions to edits; reject explicit contradictions.

    These are schema/consistency checks, not independent visual verification.
    Legacy cached plans remain readable without being silently upgraded.
    """
    # Version 3 emits the noun once. Legacy template placeholders are not
    # semantic labels and can be bound to the independently identified noun
    # without another model call; real conflicting labels still fail below.
    placeholders = {"singular noun", "singular object or part noun", "scope noun"}
    targets = plan.get("targets")
    if (plan.get("grounding_version") == 3 or
            (plan.get("grounding_version") != 2 and isinstance(targets, list) and
             any(isinstance(t, dict) and t.get("scope") in placeholders for t in targets))):
        if not isinstance(targets, list):
            raise ValueError("multi grounding targets missing")
        bound = []
        for item in targets:
            entity = (item.get("grounding") or {}).get("entity", "")
            if not isinstance(entity, str) or not entity.strip() or entity in placeholders:
                raise ValueError("multi source entity unresolved")
            scope = item.get("scope")
            if scope and scope not in placeholders and not same_scope_identity(entity, scope):
                raise ValueError("multi edit overrides independently grounded extent")
            bound.append({**item, "scope": entity if not scope or scope in placeholders else scope})
        return {**plan, "targets": bound}
    if plan.get("grounding_version") != 2:
        return plan
    expected = {str(o["object_id"]): f"U{i+1}" for i, o in enumerate(objects)}
    regions, targets = plan.get("regions"), plan.get("targets")
    for name, rows in (("regions", regions), ("targets", targets)):
        if (not isinstance(rows, list) or len(rows) != len(expected)
                or any(not isinstance(row, dict) for row in rows)
                or {str(row.get("object_id")) for row in rows} != set(expected)):
            raise ValueError(f"multi grounding {name} must cover every target exactly once")
        if any(row.get("unit_id") != expected[str(row["object_id"])] for row in rows):
            raise ValueError("multi grounding unit/object binding mismatch")
    by_id = {str(r["object_id"]): r for r in regions}
    bound = []
    for item in targets:
        region = by_id[str(item["object_id"])]
        if (region.get("extent") not in {"object", "part", "surface", "background"}
                or any(not isinstance(region.get(k), str) or not region[k].strip()
                       for k in ("entity", "ref_phrase", "mask_description", "excluded"))):
            raise ValueError("multi mask ownership unresolved")
        if (("scope" in item and item["scope"] != region["entity"])
                or ("ref_phrase" in item and item["ref_phrase"] != region["ref_phrase"])):
            raise ValueError("multi edit overrides independently grounded extent")
        bound.append({**item, "scope": region["entity"], "ref_phrase": region["ref_phrase"],
                      "grounding": {**region, "current_value": item.get("current_value", ""),
                                    "edit_feasible": item.get("edit_feasible"), "reason": item.get("reason", "")}})
    return {**plan, "targets": bound}


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
    font = load_font(22)
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


def _selected_grid(source: Image.Image, objects: list[dict[str, Any]], masks: list[np.ndarray]) -> Image.Image:
    """One tile per mask, with no excluded scene pixels competing for attention."""
    width, height, header = 640, 400, 32
    columns = min(2, len(objects))
    sheet = Image.new("RGB", (columns * width, ((len(objects)+columns-1)//columns)*(height+header)), (35, 35, 35))
    draw = ImageDraw.Draw(sheet)
    for i, (obj, mask) in enumerate(zip(objects, masks)):
        ys, xs = np.where(mask)
        if not len(xs):
            raise ValueError("empty group target mask")
        margin = max(8, round(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)*.12))
        box = (max(0, int(xs.min())-margin), max(0, int(ys.min())-margin),
               min(source.width, int(xs.max())+margin+1), min(source.height, int(ys.max())+margin+1))
        crop = source.crop(box).convert("RGB")
        binary = Image.fromarray(mask[box[1]:box[3], box[0]:box[2]].astype(np.uint8)*255)
        scale = min(width/crop.width, height/crop.height)
        size = (max(1, round(crop.width*scale)), max(1, round(crop.height*scale)))
        # Composite AFTER resizing: interpolation cannot expose excluded pixels.
        cutout = Image.composite(crop, Image.new("RGB", crop.size, (100, 100, 100)), binary)
        selected = Image.composite(cutout.resize(size, Image.Resampling.LANCZOS),
                                   Image.new("RGB", size, (100, 100, 100)),
                                   binary.resize(size, Image.Resampling.NEAREST))
        x, y = (i % columns)*width, (i // columns)*(height+header)
        draw.text((x+8, y+8), f"U{i+1} / object {obj['object_id']} - SELECTED PIXELS ONLY", fill="white")
        sheet.paste(selected, (x+(width-size[0])//2, y+header+(height-size[1])//2))
    return sheet


def group_views(candidate: dict[str, Any], objects: list[dict[str, Any]], directory: str | Path,
                *, image_size: tuple[int, int] | None = None, locator_outline: bool = False,
                locate_in_scene: bool = False, locator_cutout: bool = False,
                selected_only: bool = False, context_cutout: bool = False) -> tuple[list[str], list[str], str]:
    """Build two stable images plus binary mask files for a single group call."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source = Image.open(candidate["image_path"]).convert("RGB")
    masks = [load_mask(obj, source.size) for obj in objects]
    if context_cutout:
        view_version = "context-selected-v1"
    elif selected_only:
        view_version = "selected-only-grid-v1"
    elif locator_cutout:
        view_version = "clean-binary-cutout-v5"
    else:
        view_version = "clean-binary-outline-v4" if locator_outline else "clean-binary-v3"
    key = json.dumps({"version": view_version,
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
        if context_cutout:
            three = _crop_sheet(source, objects, masks, locator_cutout=True)
            paired = Image.new("RGB", (1024, three.height))
            paired.paste(three.crop((0, 0, 512, three.height)), (0, 0))
            paired.paste(three.crop((1024, 0, 1536, three.height)), (512, 0))
            paired.save(sheet, compress_level=1)
        else:
            (_selected_grid(source, objects, masks) if selected_only else
             _crop_sheet(source, objects, masks, locator_outline=locator_outline, locator_cutout=locator_cutout)).save(sheet, compress_level=1)
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


def multi_prompt(assignments: dict[str, str], objects: list[dict[str, Any]], order: list[str], *, grounding_first: bool = False, context_cutout: bool = False) -> str:
    prompt = MULTI_GROUNDING_FIRST_PROMPT if grounding_first else MULTI_PROMPT
    if context_cutout:
        prompt = prompt.replace("Image 2 is a numbered grid of SELECTED PIXELS ONLY crops. Every tile retains only its own mask's original pixels; uniform gray is excluded padding, including holes. Tile order is NOT scene order.",
            "Image 2 has one row per target with TWO aligned columns: LEFT is clean local CONTEXT, RIGHT contains SELECTED PIXELS ONLY. "
            "Use the left crop to understand attachment and anatomy; ONLY the right crop defines the editable entity. "
            "Uniform gray in the right crop is excluded padding, including holes. Row order is NOT scene order.")
    return prompt + "\n" + BLIND_RULE + "\n" + INSTRUCTION_STYLE + "\nAssigned operations:\n" + "\n".join(
        f"- object_id={o['object_id']}, unit_id=U{i+1}, operation={assignments[str(o['object_id'])]}, source_label_hint={json.dumps(str(o.get('category', 'unknown'))[:120])}" for i, o in enumerate(objects)
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
    plan = normalize_multi_grounding(plan, objects)
    if plan.get("feasible") is not True:
        targets = plan.get("targets")
        reasons = [str(t.get("reason", "")) for t in (targets if isinstance(targets, list) else [])
                   if isinstance(t, dict) and t.get("edit_feasible") is False]
        raise ValueError("infeasible multi plan: " + "; ".join(reasons or [str(plan.get("reason", ""))]))
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
            if (not same_scope_identity(grounding.get("entity", ""), item.get("scope", ""))
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
        validate_multi_semantics(item, obj.get("category", ""))
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
                            "grounding": by_id[str(u.object_id)].get("grounding", {}),
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
