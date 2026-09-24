"""Compact edit evidence and identity-preserving, single-image grounding."""

import json
import math
import re


def observation_prompt(edit_type, instruction):
    """Describe realized edits, then list independently segmentable instances."""
    return f'''Image 1 is SOURCE; image 2 is TARGET.
Edit type: {edit_type}
Instruction: {instruction}

Compare the WHOLE pair before listing the substantive edits actually visible. The instruction
and type describe intended editing, not the limit of the changed region. Include all clearly
co-edited instances, including ones elsewhere in the image. Describe concrete before/after
appearance, not just an operation name. Ignore minor resampling, lighting noise and unchanged
neighbors. Do not invent changes.

For each edit, list the units to segment. One unit is one distinct object instance or edited part.
Separate different instances and separate left/right limbs; keep each unit's full visible extent.
Whole-object scope requires removal or replacement of the object's identity/overall shape.
Recoloring skin/hair/clothes and changing a face or limb pose are LOCAL edits, not replacement
of the person. Select only the changed parts; preserved clothes/torso must stay outside them.
An arm motion includes its hand and sleeve; list a moved prop separately.
Explicitly list objects that disappear/change together with the subject: contents, carried or
ridden objects, supports. A removed bowl of fruit needs a bowl unit AND a fruit unit;
removed riders AND horses need separate person and horse units, not only "riders".
For skin, list each exposed face/arm/leg and any changed hair separately; no whole-person ref.
A nearby cluster of tiny particles or a continuous light strand can be one unit; split distant groups.

Refs are short visual nouns naming WHAT to segment in that image, with useful appearance attributes.
Locations identify WHICH instance using its owner, image-relative position and nearby anchors.
Use an empty ref only if that unit is absent on that side, including newly added contents of an
unchanged container. No coordinates or masks. Return edits=[] if no substantive edit is visible.
Concise JSON only:
{{"edits":[{{"change":"concrete before-to-after operation","units":[{{"source_ref":"object or part","target_ref":"object or part","source_location":"instance identity in SOURCE","target_location":"instance identity in TARGET","layout":"single or nearby_group"}}]}}]}}
'''


def strict_json(text):
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text).strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def flatten_edit_units(payload):
    """Adapt the event/unit schema to the downstream checklist without merging units."""
    if "changes" in payload or not isinstance(payload.get("edits"), list):
        raise ValueError("observation requires an edits array, not mixed schemas")
    changes = []
    for edit_id, edit in enumerate(payload["edits"]):
        if not isinstance(edit, dict) or not isinstance(edit.get("change"), str) or not edit["change"].strip():
            raise ValueError("each edit requires a concrete change description")
        if not isinstance(edit.get("units"), list) or not edit["units"]:
            raise ValueError("each edit requires at least one segmentation unit")
        for unit in edit["units"]:
            if not isinstance(unit, dict):
                raise ValueError("segmentation unit must be an object")
            for side in ("source", "target"):
                if not isinstance(unit.get(f"{side}_ref"), str):
                    raise ValueError(f"unit needs a {side}_ref string; use empty string for absence")
                location = unit.get(f"{side}_location", "")
                if not isinstance(location, str) or (unit[f"{side}_ref"].strip() and not location.strip()):
                    raise ValueError(f"visible {side} unit needs its own instance location")
            if not unit['source_ref'].strip() and not unit['target_ref'].strip():
                raise ValueError("unit cannot be absent in both images")
            layout = unit.get("layout", "single")
            if layout not in {"single", "nearby_group"}:
                raise ValueError("unit layout must be single or nearby_group")
            changes.append({"edit_id": edit_id, "change": edit["change"].strip(),
                            "source_ref": unit['source_ref'].strip(), "target_ref": unit['target_ref'].strip(),
                            "source_location": unit.get('source_location', '').strip(),
                            "target_location": unit.get('target_location', '').strip(),
                            "region_description": (unit.get('source_location') or unit.get('target_location', '')).strip(),
                            "region_layout": layout})
    return changes


def segmentation_ref(ref):
    """Normalize identity/state words without inventing another object category.

    Do not ask the model to name the same subject twice: it can turn 'bird head'
    into 'bird' or omit changed cushions. Non-person object attributes and
    attachments are deliberately preserved, not reduced to a head noun.
    """
    phrase = str(ref).strip()
    # Possessive owners disambiguate grounding, not the part to segment.
    phrase = re.sub(r"^(?:(?:the|left|right|middle|central)\s+)*"
                    r"(?:man|woman|person|boy|girl|child|character|monkey|rider)(?:'s|’s)\s+",
                    "", phrase, flags=re.I)
    phrase = re.sub(r"^(?:man|woman|person|boy|girl|child|character|monkey)\s+"
                    r"(?=(?:face|arms?|hands?|legs?)\b)", "", phrase, flags=re.I)
    whole_person = re.match(r"^(?:(?:the|a|left|right|middle|young|old|elderly|two|three)\s+)*"
                            r"(man|woman|person|people|men|women|boy|girl|child|children|horseback rider|rider)\b(?!['’])",
                            phrase, flags=re.I)
    mixed_subjects = re.search(r"\band\b", phrase, re.I) and not re.search(r"\b(?:in|wearing)\b.*\band\b", phrase, re.I)
    if whole_person and not mixed_subjects and not re.search(r"\b(?:head|face|eyes?|arms?|hands?|legs?|feet|neck)\b", phrase, re.I):
        return 'people' if whole_person[1].lower() in {'people','men','women','children'} else 'person'
    phrase = re.sub(r"\b(?:smiling|surprised|frowning|neutral|closed|open|raised|lowered|bent|extended)\s+"
                    r"(?=(?:face|eyes?|arms?|hands?)\b)", "", phrase, flags=re.I)
    if re.search(r"\b(?:arms?|hands?)\b", phrase, re.I):
        phrase = re.sub(r"\s+(?:high[- ]fiving|clasped together)$", "", phrase, flags=re.I)
    return phrase


def grounding_checklist(observation, side):
    return [
        {"change_id": item.get("change_id", i), "ref": segmentation_ref(item[f"{side}_ref"]),
         "grounding_ref": item[f"{side}_ref"],
         "location": (item[f"{side}_ref"] + "; " + item.get(f"{side}_location", item.get("region_description", ""))).strip("; "),
         "change": item.get("change", ""),
         **({"edit_id": item["edit_id"]} if "edit_id" in item else {}),
         "layout": item.get("region_layout", "single")}
        for i, item in enumerate(observation.get("changes", []))
        if item.get(f"{side}_ref")
    ]


def grounding_prompt(observation, side):
    checklist = grounding_checklist(observation, side)
    return f"""Locate each listed object or part in the shown {side} image.
Checklist: {json.dumps(checklist, ensure_ascii=False, separators=(',', ':'))}
Use the appearance and location in THIS image; change explains the editing scope, not an instruction
to find an absent opposite-side appearance. ref defines WHAT to enclose, location identifies WHICH.
Use tight boxes enclosing the complete visible contour, including head/limbs for a whole person.
Do not substitute the whole person for a local part. Crop context will be added by code.
Return EVERY change_id exactly once. Do not merge different IDs or rewrite their refs.
Coordinates are [x1,y1,x2,y2], normalized to [0,1000] in this image. JSON only:
[{{"change_id":0,"bbox_2d":[100,200,400,500]}}]
If an item cannot be located, return its ID with bbox_2d=null and a short reason.
"""


def checked_bbox(values):
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("bbox_2d must contain four coordinates")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("bbox coordinates must be finite numbers")
    if not all(0 <= x <= 1000 for x in values) or not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError("bbox must be ordered and within [0,1000]")
    return [float(x) for x in values]


def parse_checklist_grounding(text, observation, side):
    """Validate coverage; refs are assigned by code, never re-invented by pass 2."""
    records = strict_json(text)
    if isinstance(records, dict) and set(records) == {"changes"}:
        records = records["changes"]  # Complete wrapper only; IDs and contents are still validated.
    if not isinstance(records, list):
        raise ValueError("grounding must be a complete JSON array")
    expected = {item['change_id']: item for item in grounding_checklist(observation, side)}
    seen, boxes, unresolved = set(), [], []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("grounding item must be an object")
        identity = item.get("change_id")
        if type(identity) is not int or identity not in expected or identity in seen:
            raise ValueError(f"unexpected or duplicate change_id: {identity}")
        seen.add(identity)
        if "bbox_2d" in item and "boxes" not in item:
            item = {**item, "status": "not_visible" if item["bbox_2d"] is None else "located",
                    "boxes": [] if item["bbox_2d"] is None else [{"bbox_2d": item["bbox_2d"]}]}
        regions = item.get("boxes")
        if not isinstance(regions, list):
            raise ValueError(f"change_id {identity}: boxes must be an array")
        if item.get("status") == "not_visible" and not regions:
            if not str(item.get("reason", "")).strip():
                raise ValueError("not_visible requires a reason")
            unresolved.append({"change_id": identity, "reason": item["reason"]})
            continue
        if item.get("status") != "located" or not regions:
            raise ValueError(f"change_id {identity}: invalid status/boxes")
        for region in regions:
            if isinstance(region, list):
                region = {"bbox_2d": region, "region_mode": item.get("region_mode"),
                          "mask_density": item.get("mask_density")}
            if not isinstance(region, dict):
                raise ValueError("each box must be an object or four-coordinate array")
            mode = region.get("region_mode", item.get("region_mode"))
            density = region.get("mask_density", item.get("mask_density"))
            if mode is not None and mode not in {"object", "aggregate_region"}:
                raise ValueError("invalid region_mode")
            if density is not None and density not in {"object", "dense", "sparse"}:
                raise ValueError("invalid region_mode or mask_density")
            ref = expected[identity]['ref']
            if len(regions) > 1:
                ref = re.sub(r'\barms\b', 'arm and hand', ref, flags=re.I)
            box = {"change_id": identity, "ref": ref,
                   **({"edit_id": expected[identity]["edit_id"]} if "edit_id" in expected[identity] else {}),
                   "change": expected[identity].get("change", ""),
                   "region_layout": expected[identity]["layout"],
                   "grounding_ref": expected[identity]["grounding_ref"],
                   "location": expected[identity]["location"],
                   "bbox_2d": checked_bbox(region.get("bbox_2d"))}
            # These are optional morphology hints, not object identity. If omitted,
            # the SAM pipeline infers them from the ref just as for legacy boxes.
            if mode is not None:
                box["region_mode"] = mode
            elif expected[identity]["layout"] == "nearby_group":
                box["region_mode"] = "aggregate_region"
            if density is not None:
                box["mask_density"] = density
            boxes.append(box)
    if seen != set(expected):
        raise ValueError(f"missing change_ids: {sorted(set(expected)-seen)}")
    return boxes, unresolved
