"""Audit: the protocol 2.0 sentences of every unit and the second 27B call that checks them against the images.

Sentences come from the vendored protocol module: render(spec, True) is the ref sentence and render(spec, False) the
blind no-ref sentence, checked by validate_noref (protocol_v2_adapt.sentences); a qualifier the module cannot
classify leaves the unit with its ref sentence only (noref_ineligible).  The audit prompt and its rules are those of
protocol-adapt-4 with the no-ref point stated for "the selected region", and one more point, result, for the C3
failure modes (half-done replacements and recolors, broken add contact).  With several votes every vote must pass.
"""
from __future__ import annotations

from PIL import Image

from . import instruction_protocol as ip
from . import masks as mk
from . import views as vw
from .llm import parse_json

ATTEMPTS = 3
AUDIT_KEYS = ("edit", "relation", "target", "outline", "noref", "wording", "result")

AUDIT_PROMPT = """You check one rewritten training example of an image-editing dataset.
Image 1 is BEFORE with the selected source object S outlined in green{orange}. Image 2 is AFTER. Image 3 is a close-up, BEFORE with the outlines on the left and AFTER on the right.{others}

Instruction with a reference: "{ref}"
The same instruction without the reference: "{noref}"
Edit type: {op}. {role}

Judge each point "pass" or "fail":
1. edit: the instruction describes what actually changed from BEFORE to AFTER at S: the right operation, the right object, the requested new item or value (its kind, count, color, material or exact text), with nothing important wrong or missing. Check every phrase: anything the instruction says to add must be new in AFTER (not already in BEFORE), and every placement phrase (on, next to, near, under, ...) must be true in AFTER. Changes that only follow from the edit (shadows, revealed background, a carried bag removed with a person) are fine.
2. relation: {relation_rule}
3. target: {target_rule}
4. outline: {outline_rule}
5. noref: using the green outline as the selected region, the instruction without the reference asks for the same edit.
6. wording: both instructions are clear, natural, concise English, with no word that neither identifies the target nor defines the result.
7. result: AFTER shows the edit completely and cleanly: a replaced object is entirely the new object with no part of the old one left, a recolored or re-textured object is changed as a whole, a removed object leaves a plausible background, an added item looks natural and really rests on or is attached to S, and there are no artifacts, smears, seams or cut-off edges.

Return only JSON: {{"edit": "pass", "relation": "pass", "target": "pass", "outline": "pass", "noref": "pass", "wording": "pass", "result": "pass", "problems": "<at most 25 words; empty if everything passes>"}}"""
AUDIT_RULES = {
    "add_spot": {
        "relation": "the stated relation of the new item to S ({relation}) is true in AFTER, and the new item touches "
                    "S: it stands, rests, hangs or is attached on it, holds it up, or flies or floats inside the "
                    "outlined sky or water. S across open space from the new item fails.",
        "target": "the green outline is on what the new item directly stands, rests, hangs, is attached or printed "
                  "on in AFTER (for something that flies or floats, the sky or water around it; for a new item that "
                  "holds S up, the object it holds), and it is what the instruction's target phrase names. A "
                  "different object, or one with open space between it and the new item, fails.",
        "outline": "the green outline is one compact area of that object, part or surface at the spot where the new "
                   "item goes; it may be only the part of a larger object or surface around that spot. It fails when "
                   "it covers a large whole object or surface of which the new item takes only a small spot, marks a "
                   "place away from the new item, takes in other objects, or traces marks, text or specks instead of "
                   "the object or surface."},
    "add_whole": {
        "relation": "the stated relation of the new item to S ({relation}) is true in AFTER, at the spot of S the "
                    "instruction names if it names one, and the new item touches S there (or holds S up).",
        "target": "the green outline marks the whole object that the instruction's target phrase names, and the new "
                  "item is on it (at the spot the instruction names, e.g. \"the top of\", if any) in AFTER.",
        "outline": "the green outline covers the visible extent of that object, with no large missing piece and no "
                   "large area of something else; a whole object is right even when the new item takes only a small "
                   "spot of it."},
    "other": {
        "relation": 'answer "pass" (only for add).',
        "target": "the green outline marks the object or part that the instruction's target phrase names, not a "
                  "different instance and not a much larger or smaller thing.",
        "outline": "the green outline covers the visible extent of that object or part, with no large missing piece "
                   "and no large area of something else."},
}


def sentences(spec):
    """(ref, no-ref or None, reason)."""
    ref = ip.render(spec, True)
    noref = ip.render(spec, False)
    try:
        ip.validate_noref(spec, noref)
    except ValueError as error:
        return ref, None, "unclassified_qualifier" if "qualifier" in str(error) else str(error)
    return ref, noref, None


def audit_prompt(spec, ref, noref, whole, has_exec, others):
    add = spec["operation"] == "add"
    if add and whole:
        role = ("For add, S is the whole existing object the new item goes on (or, for \"under\", holds up); the "
                "instruction names the spot on it when one is needed. The new item does not exist in BEFORE.")
    elif add:
        role = ("For add, S is the existing object, part or surface at the spot where the new item goes: what it "
                "stands, rests or hangs on, is attached or printed on, or the sky or water it flies or floats in; the "
                "outline may be just the part of a larger object or surface around that spot. The new item does not "
                "exist in BEFORE.")
    else:
        role = "S is the object or part the edit acts on."
    rules = AUDIT_RULES[("add_whole" if whole else "add_spot") if add else "other"]
    orange = ", and, where it differs from S, the area that changes outlined in orange" if has_exec else ""
    note = (" AFTER also contains other edits elsewhere in the image; they belong to other instructions, so judge "
            "only the edit at S.") if others else ""
    return AUDIT_PROMPT.format(orange=orange, others=note, ref=ref, noref=noref or "(none)", op=spec["operation"],
                               role=role, target_rule=rules["target"], outline_rule=rules["outline"],
                               relation_rule=rules["relation"].format(relation=spec["placement"].get("relation", "to"))
                               if add else rules["relation"])


def audit_parts(source, target, s_mask, exec_mask):
    masks = []
    if exec_mask is not None and (exec_mask & s_mask).sum() < 0.95 * (exec_mask | s_mask).sum():
        masks.append((exec_mask, vw.EXEC_COLOR))
    masks.append((s_mask, vw.S_COLOR))  # last, so the green outline is never hidden
    box = vw.union_box([mk.box_1000(m) for m, _ in masks])
    views = [("Image 1, BEFORE with outlines:", vw.outlined(source, masks, vw.FULL_PIXELS, numbers=False)),
             ("Image 2, AFTER:", vw.resized(target, vw.FULL_PIXELS)),
             ("Image 3, close-up, BEFORE left and AFTER right:", vw.close_up(source, target, masks, box))]
    return vw.chat_parts(views)


async def audit_vote(client, parts, prompt, vote):
    out = {"tokens": 0}
    for attempt in range(ATTEMPTS):
        response = await client.call(parts, prompt, "audit", seed=100 * vote + attempt)
        out["tokens"] += response["tokens"]
        try:
            value = parse_json(response["text"])
            verdicts = {k: ip.clean(value.get(k)).lower() for k in AUDIT_KEYS}
            if any(v not in ("pass", "fail") for v in verdicts.values()):
                raise ValueError(f"bad verdicts {verdicts}")
            return {**out, **verdicts, "problems": ip.clean(value.get("problems")),
                    "passed": all(v == "pass" for v in verdicts.values()), "attempts": attempt + 1}
        except Exception as exc:  # noqa: BLE001 - next seed
            out["error_text"] = f"{type(exc).__name__}: {exc}"[:300]
    return {**out, "passed": None, "attempts": ATTEMPTS}


async def audit_unit(client, source, target, spec, s_mask, exec_mask, whole, others, votes=2):
    ref, noref, reason = sentences(spec)
    prompt = audit_prompt(spec, ref, noref, whole, exec_mask is not None, others)
    parts = audit_parts(source, target, s_mask, exec_mask)
    results = [await audit_vote(client, parts, prompt, v) for v in range(votes)]
    merged = {"instruction_ref": ref, "instruction_noref": noref, "noref_ineligible": reason,
              "tokens": sum(r["tokens"] for r in results), "votes": results}
    for k in AUDIT_KEYS:
        values = [r.get(k) for r in results]
        merged[k] = "fail" if "fail" in values else "pass" if all(v == "pass" for v in values) else None
    merged["passed"] = None if any(r["passed"] is None for r in results) else all(r["passed"] for r in results)
    merged["problems"] = " | ".join(r.get("problems") or "" for r in results if r.get("problems"))
    return merged
