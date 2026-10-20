"""Plan: one Qwen3.8-27B call per pair (quality, units, protocol 2.0 slots, boxes).

It replaces four calls of the old flow: the quality filter, the observation and grounding passes, and the slot call
of the protocol adaptation.  The scene-difficulty filter is gone (the user's decision, 2026-10-20); whether S has a
confusable neighbor is only recorded (`comparable`).  The unit rules are the slot prompt of protocol-adapt-4 with the
labeled outlines replaced by the changed areas the precheck found, and the writing rules of benchmark 2.6.0.
"""
from __future__ import annotations

from PIL import Image

from . import instruction_protocol as ip
from . import masks as mk
from . import slots as sl
from . import views as vw
from .instruction_style import ADD_ANCHOR_POLICY, INSTRUCTION_STYLE
from .llm import parse_json

# The shared writing rules without the benchmark's add authoring policy: that policy chooses a natural item for a
# host, while a training pair already shows its item.
assert INSTRUCTION_STYLE.endswith("\n\n" + ADD_ANCHOR_POLICY)
WRITING_RULES = INSTRUCTION_STYLE[: -len("\n\n" + ADD_ANCHOR_POLICY)]
ATTEMPTS = 3
MAX_UNITS = 4
PROBLEMS = ("no_change", "global", "partial", "defect", "unrelated_change", "not_local", "unclear", "too_many")
TYPE_LABELS = {"text": "text (a change to written text)", "attribute": "attribute (color or material change)"}

PLAN_PROMPT = """You turn one example of an image-editing dataset into structured training records.
Image 1 is BEFORE (the original photo); boxes refer to this image. Image 2 is BEFORE with the areas whose pixels changed outlined and numbered. Image 3 is AFTER (the edited photo, aligned to BEFORE). Image 4 is a close-up around the changed areas, BEFORE on the left and AFTER on the right.

Original instruction (written before the edit; it may be vague, wrong or only partly carried out): "{instruction}"
Original type label: {label}
Changed areas in image 2, found by comparing pixels (they can include shadows, reflections, revealed background or faint noise): {legend}

Describe the edit as AFTER shows it; never choose a different or better edit. Read colors from image 1, never from the outline colors.

1. observed: say in one sentence what actually changed from BEFORE to AFTER.

2. quality, judged on the images:
- "clean": every change is a complete, realistic local edit: a new or changed object looks natural (no distortion, smearing, ghosting, duplicated or missing parts, garbled text), a replaced or recolored object is changed as a whole (no half old, half new object), a removed object leaves a plausible background, and nothing else in the image changed except what the edit itself causes (its shadow, reflection, contact, revealed background).
- "reject" otherwise, with one problem: "no_change" (nothing visible changed), "global" (style, lighting, color grading or most of the image changed), "partial" (the edit is only half done), "defect" (artifacts, distortion, an unnatural or broken result), "unrelated_change" (other objects or areas changed as well), "not_local" (moving, re-posing, flipping, zooming or another viewpoint), "unclear" (cannot tell what changed), "too_many" (more than {max_units} edits).

3. units: the edits, one per source object; usually exactly one. A unit acts on ONE source object, or one part of it, with ONE operation; several new items of the same kind at one anchor are one unit (give their number). Separate objects are separate units: a group of people, several animals, a row of buildings, cars, bottles or chairs, "the two flagpoles". Counted as ONE object: a pile, heap, bunch, stack or cluster of same-kind items lying together (a pile of sweet potatoes, a bunch of bananas, a stack of plates, a bouquet), and a matching pair worn or used by one person (shoes, gloves, sleeves). Every changed area that is part of the edit belongs to a unit; areas of faint noise belong to none.

4. operation of a unit, judged by what AFTER shows, not by the instruction's verb:
- add: a new physical object or attached item appears; what was there stays.
- replace: an object or part becomes a different kind of object or part (its structure or category changes). A change of only color, pattern or material is not replace.
- remove: an object, a part of it, or its contents disappear.
- attribute: the object keeps its identity and shape and only an appearance property changes: color, material, finish (glossy or matte), pattern, texture, text (what written text says), tint (color of a transparent object) or opacity.
- Written text: deleting text or a logo is remove; changing what text says is attribute with property "text"; writing new text where there was none is add.
- No unit for moving, re-posing, resizing, style or whole-image changes: then quality is "reject".

5. The source object S of a unit, which must be visible in BEFORE:
- add: S is the anchor, what the new item directly stands, sits, lies or rests on, is in, hangs from, is attached, mounted, sewn, pinned or printed on, or is worn or held by in AFTER. Name S at the finest level that has a name (granularity):
  "object": a whole object small enough to say where the item goes, e.g. a cup, a candle, a sign, a hat.
  "part": a named part of a larger object, e.g. the head or hand of a person, the seat of a sofa, the top of a cake, the top of a tower, the back of a zebra, the handle of a bag, a branch of a tree.
  "surface": a stretch of a large surface, e.g. a road, street, sidewalk, ground, floor, grass, sand, snow, water, the sea, a wall, the sky, a large table top, a counter, a shelf or a bed.
  A dog added on the road next to a man has S "the road next to the man", not the man; a boat on the sea has S "the sea ...", not a rock on the shore; a cross on a tower has S "the top of the tower", not the whole church.
  Water carries what floats on it. The sky is S only for something that flies or floats in the air by itself (a balloon, a kite, a bird in flight, an aircraft, fireworks, the moon, a rainbow); a bird on a wire has S the wire. Text or a graphic added to the picture is printed on what carries it (a wall, a sign, a poster, or the background of a poster or graphic).
  One exception: when the new item holds S up or holds it (a bench added under a sitting person, a tray under a fish, a bowl under fruit, a pedestal under a statue), S is the object held, granularity "object", relation "under", anchor_mode "reference". No other reference is used: S is never an object the new item is merely near, above, below or beside. If nothing visible carries an added item and it does not fly or float by itself, the example cannot be used: quality "reject", problem "defect".
  S is never the new item or empty space.
- remove, replace, attribute: S is the object or part the edit acts on in BEFORE (if only a part of an object changes, such as a lid or a sleeve, S is that part).
- written text: when only some lettering changes, S is that lettering area; name it with a noun such as "lettering", "logo", "label", "number" or "banner", never with the word "text" (the sentence already says "the text on ..."); otherwise S is the object that carries it.

6. Slots of a unit. Reuse the instruction's wording where possible and never add details that neither the instruction nor AFTER shows. The sentences are rendered from these slots; follow the writing rules shared with the benchmark:
{writing_rules}

- ref: the shortest noun phrase starting with "the" that picks out S in BEFORE (see REFERENCE above), written from BEFORE only, never from the edit result. Say "on the left" or "in the center", not "of the image". For add: an object by itself ("the white candle"); a part as "the <part> of <host>" ("the top of the unicorn cake"); a stretch of a surface by the surface and one landmark ("the road next to the monk", "the sand to the right of the yellow surfboard").
- scope: the kind of S itself at the same granularity as S, with no article, color, position, owner or number, e.g. "jacket", "chair seat", "suitcase handle", "sign". For add: the object ("candle"), the part ("cake top", "sofa seat", "tower top", "hand") or the surface ("road", "sand", "sky").
- host_ref, host_scope: for add, the whole object or surface S belongs to, and its kind: "the unicorn cake" and "cake" for S "the top of the unicorn cake"; "the road" and "road" for S "the road next to the monk"; the same as ref and scope when S is a whole object. For other operations "".
- part: for an add with granularity "part", the part as it would follow "the" and precede "of": "top", "seat", "back", "handle", "left sleeve"; otherwise "".
- granularity: for add "object", "part" or "surface"; otherwise "".
- target_part: "" (S already is the part that changes).
- payload: for add the new item, for replace the replacement, as AFTER shows it, in the instruction's own words where they are true (count, size, color, kind), e.g. "a small red identification tag"; do not add colors, materials or other details that neither the instruction states nor AFTER clearly shows. Name only what is new in AFTER, and drop or correct parts of the instruction that AFTER contradicts or that already existed in BEFORE. Placement words say where S is: they belong in the ref, not in the payload, and the payload never refers to S with "its" or "it". For replace, one complete noun phrase for the new object including its changed form, e.g. "a square four-hole button". For remove and attribute "".
- remove_mode: for remove one of "whole", "contents" (a container such as a basket, box or bag removed with what it holds), "contents_only" (only what a container holds), "lowerlegs" (trousers cut to shorts), "hem" (skirt or dress shortened); otherwise "whole". Vehicles, furniture and people are not containers.
- attributes: for attribute a list of {{"property": ..., "value": ...}} with property one of color, material, finish, pattern, texture, text, tint, opacity, as AFTER shows it. For text the value is the exact new text without quotes, as written in AFTER. tint and opacity stand alone; opacity is transparent, translucent or opaque. Otherwise [].
- placement: for add {{"relation": ..., "site": ..., "anchor_mode": ...}}, otherwise {{}}. relation: how the new item meets S, true in AFTER, one of on, in, inside, around, across, along, over, at, attached to, clipped to, pinned to, hanging from, hooked over, resting on, mounted on, sewn onto, strapped over, tied around; never the bare "to". Use "on" for a surface the item stands or lies on and "in" for the sky, water, sand, snow or a container it is in; "over" only for something draped over or covering S, never for something above S; "under" only for the exception above. anchor_mode: "reference" for that exception, otherwise "support". site: "" unless S is a whole object and the item goes on one spot of it (e.g. "side" for a sticker on the side of a candle); never repeat the part.
- qualifier: normally "". Only a short clause the result needs, e.g. "including its pedestal" when an attached item is removed together with S.

7. Locations of a unit:
- areas: the numbers of the changed areas in image 2 that belong to this unit.
- s_box: a tight box around S itself in image 1, [x1, y1, x2, y2] normalized to 0-1000; for a person include head and limbs. For add with granularity "part", the box of that part; with "surface", the box of the stretch of surface right around and under the new item.
- host_box: for add with granularity "part" or "surface", a tight box around the whole host (host_ref) in image 1; otherwise [].
- item_box: for add and replace, a tight box around the new item in image 3 (AFTER); otherwise [].
- comparable: true when BEFORE shows another object of the same kind as S that the ref has to tell apart from S, else false.

Return only JSON:
{{"observed": "...", "quality": "clean", "problem": "", "reason": "<at most 15 words>", "units": [{{"areas": [1], "operation": "add", "ref": "the ...", "scope": "...", "host_ref": "the ...", "host_scope": "...", "part": "", "granularity": "surface", "target_part": "", "payload": "...", "remove_mode": "whole", "attributes": [], "placement": {{"relation": "on", "site": "", "anchor_mode": "support"}}, "qualifier": "", "s_box": [0, 0, 0, 0], "host_box": [0, 0, 0, 0], "item_box": [0, 0, 0, 0], "comparable": false}}]}}
For a rejected example return {{"observed": "...", "quality": "reject", "problem": "...", "reason": "...", "units": []}}."""


def legend(areas, more):
    items = [f'[{a["n"]}] {vw.PALETTE[(a["n"] - 1) % len(vw.PALETTE)][0]}, {100 * a["area_frac"]:.2g}% of the image'
             for a in areas]
    if more:
        items.append(f"{more} more tiny pieces not outlined")
    return "; ".join(items) if items else "none found"


def plan_prompt(row, areas, more):
    return PLAN_PROMPT.format(instruction=row["instruction"].replace('"', "'"),
                              label=TYPE_LABELS.get(row["native_op"], row["native_op"]), legend=legend(areas, more),
                              max_units=MAX_UNITS, writing_rules=WRITING_RULES)


def plan_parts(source, target, areas):
    """Image 1 BEFORE (box view), 2 BEFORE with the numbered changed areas, 3 AFTER, 4 close-up of all areas."""
    masks = [(mk.decode(a), vw.PALETTE[(a["n"] - 1) % len(vw.PALETTE)][1]) for a in areas]
    views = [("Image 1, BEFORE:", vw.resized(source, vw.BOX_PIXELS)),
             ("Image 2, BEFORE with the changed areas outlined and numbered:", vw.outlined(source, masks, vw.FULL_PIXELS)),
             ("Image 3, AFTER:", vw.resized(target, vw.FULL_PIXELS))]
    box = vw.union_box([a["box_1000"] for a in areas])
    if box:
        views.append(("Image 4, close-up, BEFORE left and AFTER right:",
                      vw.close_up(source, target, masks, box, numbers=True)))
    return vw.chat_parts(views)


def as_box(value):
    """A 0-1000 [x1, y1, x2, y2] box, or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        box = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    box = [min(1000, max(0, v)) for v in box]
    return box if box[0] < box[2] and box[1] < box[3] else None


def parse_unit(raw, n_areas):
    """(unit, problems): normalized slots, protocol 2.0 spec and boxes of one unit, or the reasons to ask again."""
    slots, fixes = sl.normalize_slots(raw)
    slots["target_part"] = slots.get("target_part", "")
    problems = sl.slot_problems(slots) if slots["operation"] in ip.OPERATIONS else [
        f'operation "{slots["operation"]}" is not one of add, replace, remove, attribute']
    s_box, host_box, item_box = as_box(raw.get("s_box")), as_box(raw.get("host_box")), as_box(raw.get("item_box"))
    if s_box is None:
        problems.append("s_box must be a box [x1, y1, x2, y2] in 0-1000 with x1 < x2 and y1 < y2")
    if slots["operation"] == "add" and slots.get("granularity") in ("part", "surface") and host_box is None:
        problems.append("host_box must box the whole host for granularity part or surface")
    areas = []
    for a in raw.get("areas") or []:
        try:
            a = int(a)
        except (TypeError, ValueError):
            continue
        if 1 <= a <= n_areas and a not in areas:
            areas.append(a)
    spec = None
    if not problems:
        try:
            spec = sl.to_spec(slots)
            if spec["target_part"] and spec["operation"] != "add":
                spec = sl.part_as_s(spec)
                fixes.append("part_as_s")
            ip.render(spec, True)
            ip.render(spec, False)
        except (ValueError, KeyError, IndexError) as exc:
            problems.append(str(exc))
            spec = None
    return {"slots": slots, "fixes": fixes, "spec": spec, "s_box": s_box, "host_box": host_box, "item_box": item_box,
            "areas": areas, "comparable": bool(raw.get("comparable"))}, problems


def parse_plan(text, n_areas):
    raw = parse_json(text)
    quality = ip.clean(raw.get("quality")).lower()
    if quality not in ("clean", "reject"):
        raise ValueError(f'quality must be "clean" or "reject", not {quality!r}')
    out = {"observed": ip.clean(raw.get("observed")), "quality": quality, "reason": ip.clean(raw.get("reason")),
           "problem": ip.clean(raw.get("problem")).lower() if quality == "reject" else "", "units": []}
    if quality == "reject":
        if out["problem"] not in PROBLEMS:
            out["problem"] = "other:" + out["problem"]
        return out, []
    units = raw.get("units") or []
    if not units:
        raise ValueError('a "clean" example needs at least one unit')
    if len(units) > MAX_UNITS:
        return {**out, "quality": "reject", "problem": "too_many", "units": []}, []
    problems = []
    for k, unit in enumerate(units, 1):
        if not isinstance(unit, dict):
            raise ValueError(f"unit {k} is not an object")
        parsed, unit_problems = parse_unit(unit, n_areas)
        out["units"].append(parsed)
        problems += [f"unit {k}: {p}" for p in unit_problems]
    return out, problems


async def plan_pair_with_parts(client, row, parts, areas, more):
    """The plan call, asked again with the problems stated until the units form valid specs."""
    base = plan_prompt(row, areas, more)
    prompt, record = base, {"id": row["id"], "attempts": [], "tokens": 0}
    for seed in range(ATTEMPTS):
        response = await client.call(parts, prompt, "plan", seed=seed)
        record["tokens"] += response["tokens"]
        try:
            plan, problems = parse_plan(response["text"], len(areas))
            if problems:
                raise ValueError("; ".join(problems))
            record["attempts"].append({"seed": seed, "ok": True})
            record["plan"] = plan
            return record
        except Exception as exc:  # noqa: BLE001 - malformed or invalid: ask again, saying what was wrong
            error = f"{type(exc).__name__}: {exc}"[:600]
            record["attempts"].append({"seed": seed, "ok": False, "error": error, "finish": response["finish"],
                                       "text": response["text"][-2000:]})
            prompt = base + f"\n\nYour previous answer was not usable ({error}). Answer again, fixing it."
    record["plan"] = None
    return record


def open_images(run, row, align):
    from .pixels import apply_alignment, load_rgb
    src = load_rgb(run / row["source_image"])
    tgt = apply_alignment(load_rgb(run / row["target_image"]), src.shape[:2], align)
    return Image.fromarray(src), Image.fromarray(tgt)
