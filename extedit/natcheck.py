"""Naturalness check of the CrispEdit add pairs: does the BEFORE image show traces of an erased object?

CrispEdit makes its add pairs from real photos by erasing an object: the photo is the AFTER, the erased image the
BEFORE.  QA of the first production batch (2026-10-22) found erasing traces in 2 of 6 sampled add rows that had passed
the audit: a woman whose dress was erased shows the background through her body, an erased tractor left its wheels
floating.  A model trained on such pairs learns to add objects only where something is visibly missing.
Qwen3.8-27B (thinking, two votes in one request) gets the BEFORE image with the edited area outlined, a close-up of
the area and the same close-up of the AFTER, and says whether the BEFORE looks like an untouched photo there.  A row is
rejected when both votes see traces.  Calibration on the 565 CrispEdit add rows of the first batch (EXTEDIT_PROTOCOL2.md
§2.7): 0 / 1 / 2 "not natural" votes for 349 / 127 / 89 rows; of 15 reviewed 2-vote rows 12 show erasing traces and 3
are borderline, while most 1-vote rows are fine, so one vote does not reject.  Only CrispEdit add rows are checked: in
the other source types the BEFORE is the real photo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import masks as mk
from . import views as vw
from .llm import parse_json

ATTEMPTS = 3
VOTES = 2
REJECT_VOTES = 2
OUTLINE = (235, 50, 45)
FULL_PIXELS = 1024 * 1024
# Close-up: the box of the edited area grown by half its size on each side, at least a quarter of the image's long
# side, scaled to a long side of 1024 px (at most 3x).
CROP = {"pad": 0.5, "min_frac": 0.25, "long_side": 1024, "max_scale": 3.0}

PROMPT = """Image 1 is a photo; a red outline marks an area of it. Image 2 is a close-up of the outlined area of image 1, and image 3 is the same close-up after an object was added there.

Pairs like this one are made from a real photo (image 3) by erasing an object, so image 1 may show traces of the erasing. Judge image 1 as it is, using images 1 and 2 (image 3 only shows what was erased): does the outlined area look like a natural, intact part of an untouched photo?

Traces of erasing are, for example:
- parts of the erased object left behind with nothing to attach to: wheels without a vehicle; legs, feet or hands without a body; straps, handles or cables that end in the air; a shadow or reflection of something that is not there;
- a person or object that looks hollow, cut open or see-through, with background showing where a body or object should be;
- smeared, blurred, warped or patchy texture, repeated patterns, color blotches, or lines and edges that break off;
- an empty spot that is clearly unnatural for the scene, such as a person-shaped gap in a crowd.
An empty spot that is natural for the scene (plain wall, floor, road, sky, table, water or grass) counts as natural, even though image 3 shows an object there.

Return only JSON: {"traces": "the traces you see in image 1, or an empty string", "natural": true}"""


def needs_check(row):
    """CrispEdit add rows: their BEFORE image is the erased one."""
    return row["dataset"] == "crispedit" and row["native_type"] == "add"


def edited_area(gate):
    """Union of the units' edited regions (where the AFTER differs): the place of the added object."""
    area = None
    for unit in gate["units"]:
        mask = mk.decode(unit["edited_region"])
        area = mask if area is None else area | mask
    return area


def crop_box(mask, size, p=CROP):
    W, H = size
    ys, xs = np.nonzero(mask)
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = p["min_frac"] * max(W, H)
    bw = min(W, max((x2 - x1) * (1 + 2 * p["pad"]), side))
    bh = min(H, max((y2 - y1) * (1 + 2 * p["pad"]), side))
    left = min(max(0.0, (x1 + x2) / 2 - bw / 2), W - bw)
    top = min(max(0.0, (y1 + y2) / 2 - bh / 2), H - bh)
    return round(left), round(top), round(left + bw), round(top + bh)


def views(source, target, area, p=CROP):
    """The BEFORE with the area outlined, and enlarged close-ups of the area in the BEFORE and the AFTER."""
    box = crop_box(area, source.size, p)
    scale = min(p["max_scale"], p["long_side"] / max(box[2] - box[0], box[3] - box[1]))
    size = (max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale)))
    sub = area[box[1]:box[3], box[0]:box[2]]
    close = [vw.outlined(image.convert("RGB").crop(box).resize(size, Image.Resampling.LANCZOS), [(sub, OUTLINE)],
                         10 ** 9, numbers=False) for image in (source, target)]
    return vw.outlined(source, [(area, OUTLINE)], FULL_PIXELS, numbers=False), close[0], close[1]


def request(run, row, gate):
    """Chat parts of one row; a top-level function so that a process pool draws the views off the event loop."""
    run = Path(run)
    source = Image.open(run / row["source_image"]).convert("RGB")
    target = Image.open(run / gate["target_pasted"]).convert("RGB")
    full, before, after = views(source, target, edited_area(gate))
    return vw.chat_parts([("Image 1, the photo:", full), ("Image 2, close-up of the outlined area of image 1:", before),
                          ("Image 3, the same close-up after the object was added:", after)])


def parse_vote(text):
    value = parse_json(text)
    natural = value.get("natural")
    if isinstance(natural, str):
        natural = {"true": True, "false": False}.get(natural.strip().lower())
    if not isinstance(natural, bool):
        raise ValueError("natural must be true or false")
    return {"natural": natural, "traces": str(value.get("traces") or "")}


def decide(votes, reject_votes=REJECT_VOTES):
    """(passed, reason) from the votes."""
    if len(votes) < VOTES:
        return False, "unreadable_answer"
    if sum(not v["natural"] for v in votes) >= reject_votes:
        return False, "before_unnatural"
    return True, None


async def check_row(client, row_id, parts):
    """Two votes in one request; asked again (with the problem stated) for the votes that could not be parsed."""
    prompt, votes, tokens = PROMPT, [], 0
    for attempt in range(ATTEMPTS):
        response = await client.call(parts, prompt, "natcheck", seed=attempt, n=VOTES - len(votes))
        tokens += response["tokens"]
        choices = response["choices"] if "choices" in response else [response]
        problem = None
        for choice in choices:
            try:
                votes.append(parse_vote(choice["text"]))
            except Exception as exc:  # noqa: BLE001 - malformed answer: ask again
                problem = f"{type(exc).__name__}: {exc}"
        if len(votes) >= VOTES:
            break
        prompt = PROMPT + f"\n\nYour previous answer was not usable ({problem}). Answer again."
    passed, reason = decide(votes)
    return {"id": row_id, "votes": votes, "passed": passed, "reason": reason, "attempts": attempt + 1,
            "tokens": tokens}


def summary_of(records):
    from collections import Counter
    return {"rows": len(records), "passed": sum(r["passed"] for r in records),
            "not_natural_votes": dict(Counter(sum(not v["natural"] for v in r["votes"]) for r in records)),
            "reasons": dict(Counter(r["reason"] for r in records if not r["passed"]))}
