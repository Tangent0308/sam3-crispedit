"""Text check: the written text of every text-change unit is read back from the images and compared with the text the
instruction asks for.

Validation of the first production batch (2026-10-22, 30 sampled text rows) found what the audit misses: AFTER shows
other text than the instruction ("ELÉCTION" for "ELECCIÓN", "ANDLE" with the H cut off for "HANDLE"), or the text did
not change.  Here Qwen3.8-27B (greedy, no thinking) reads the text of the target area in a BEFORE and an AFTER
close-up without being told what to expect, and the comparison is made in code, on letters and digits only (case,
spaces, punctuation and accents ignored):
  ocr_mismatch      the AFTER reading does not contain the requested text;
  no_text_change    BEFORE and AFTER read the same, letter case included ("PIRATES" -> "Pirates" is a change);
  old_text_remains  the old text is still there next to the new one (unless it is part of the new text, as
                    "ALEX" of "ALEXA").
"Contains" rather than "equals", because the model often reads a neighboring word along ("THE BALM" for "BALM").  The
model's legibility and garbled judgments are recorded but not used: on the calibration set they rejected correct
stylized and made-up words ("HOLLYWOOD", "LEENS") and caught nothing the comparison missed.
A unit is a text change when its spec is an attribute change of property "text".
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

from PIL import Image

from . import masks as mk
from . import views as vw
from .llm import parse_json

ATTEMPTS = 3
# Close-up: the target box grown by 60 % of its width on each side and 40 % of its height above and below (new text
# can be longer than the old), at least 10 % of the image's long side, scaled to a long side of 1024 px (at most 4x).
CROP = {"pad_x": 0.6, "pad_y": 0.4, "min_frac": 0.1, "long_side": 1024, "max_scale": 4.0}

TEXT_PROMPT = """Image 1 is a close-up of a photo before an edit, image 2 the same close-up after the edit. In both images a green outline marks the same area.

In each image, read the written text in the outlined area: the complete words that lie inside the outline or are crossed by it, character by character, exactly as written. Keep the letter case, do not correct spelling, do not fill in hidden or cut-off letters, and join several lines with a single space. Ignore words that lie entirely outside the outline. Write "" when the area holds no readable text.

Then judge the text in the outlined area of image 2:
- after_legible: true when every character can be read with certainty; false when any character is blurred, broken, cut off by the image border or ambiguous.
- after_garbled: true when it is gibberish, a misspelling of real words, or has malformed letters; a real name, brand, number, or a cleanly written made-up word is not garbled.

Return only JSON: {"before": "...", "after": "...", "after_legible": true, "after_garbled": false}"""


def text_units(plan):
    """[(unit index, requested text)] of the text-change units of a plan."""
    out = []
    for k, unit in enumerate(plan["units"]):
        spec = unit["spec"]
        if spec["operation"] == "attribute" and spec["attributes"][0]["property"] == "text":
            out.append((k, spec["attributes"][0]["value"]))
    return out


def normalize(text, keep_case=False):
    """Letters and digits only, without accents, case-folded unless keep_case: "ELÉCTION!" and "election" compare
    equal."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    out = "".join(ch for ch in decomposed if not unicodedata.combining(ch) and ch.isalnum())
    return out if keep_case else out.casefold()


def crop_box(box_1000, size, p=CROP):
    W, H = size
    x1, y1, x2, y2 = box_1000[0] * W / 1000, box_1000[1] * H / 1000, box_1000[2] * W / 1000, box_1000[3] * H / 1000
    side = p["min_frac"] * max(W, H)
    bw = min(W, max((x2 - x1) * (1 + 2 * p["pad_x"]), side))
    bh = min(H, max((y2 - y1) * (1 + 2 * p["pad_y"]), side))
    left = min(max(0.0, (x1 + x2) / 2 - bw / 2), W - bw)
    top = min(max(0.0, (y1 + y2) / 2 - bh / 2), H - bh)
    return round(left), round(top), round(left + bw), round(top + bh)


def close_ups(source, target, s_mask, p=CROP):
    """BEFORE and AFTER close-ups of the target area, enlarged, the area outlined just outside its edge."""
    box = crop_box(mk.box_1000(s_mask), source.size, p)
    scale = min(p["max_scale"], p["long_side"] / max(box[2] - box[0], box[3] - box[1]))
    size = (max(1, round((box[2] - box[0]) * scale)), max(1, round((box[3] - box[1]) * scale)))
    mask = s_mask[box[1]:box[3], box[0]:box[2]]
    out = []
    for image in (source, target):
        big = image.convert("RGB").crop(box).resize(size, Image.Resampling.LANCZOS)
        out.append(vw.outlined(big, [(mask, vw.S_COLOR)], 10 ** 9, numbers=False))
    return out


def text_requests(run, row, plan, seg, gate):
    """Per text unit of one pair: the requested text and the chat parts; a top-level function so that a process pool
    draws the close-ups off the event loop that sends the requests."""
    run = Path(run)
    source = Image.open(run / row["source_image"]).convert("RGB")
    target = Image.open(run / gate["target_pasted"]).convert("RGB")
    out = []
    for k, expected in text_units(plan):
        before, after = close_ups(source, target, mk.decode(seg["units"][k]["selection"]))
        out.append({"unit": k, "expected": expected,
                    "parts": vw.chat_parts([("Image 1, before the edit:", before), ("Image 2, after the edit:", after)])})
    return out


def parse_reading(text):
    value = parse_json(text)
    out = {"before": str(value.get("before") or ""), "after": str(value.get("after") or "")}
    for key in ("after_legible", "after_garbled"):
        flag = value.get(key)
        if isinstance(flag, str):
            flag = {"true": True, "false": False}.get(flag.strip().lower())
        if not isinstance(flag, bool):
            raise ValueError(f"{key} must be true or false")
        out[key] = flag
    return out


def decide(expected, reading):
    """(passed, reason) of one unit from the model's reading."""
    after, before, want = normalize(reading["after"]), normalize(reading["before"]), normalize(expected)
    if not want or want not in after:
        return False, "ocr_mismatch"
    if normalize(reading["before"], keep_case=True) == normalize(reading["after"], keep_case=True):
        return False, "no_text_change"
    if before and before not in want and before in after:
        return False, "old_text_remains"
    return True, None


async def check_unit(client, request):
    """Read one unit (asked again, with the problem stated, while the answer cannot be parsed) and decide."""
    prompt, tokens = TEXT_PROMPT, 0
    for attempt in range(ATTEMPTS):
        response = await client.call(request["parts"], prompt, "ocr", seed=attempt)
        tokens += response["tokens"]
        try:
            reading = parse_reading(response["text"])
        except Exception as exc:  # noqa: BLE001 - malformed answer: ask again
            prompt = TEXT_PROMPT + f"\n\nYour previous answer was not usable ({type(exc).__name__}: {exc}). Answer again."
            continue
        passed, reason = decide(request["expected"], reading)
        return {"unit": request["unit"], "expected": request["expected"], **reading, "passed": passed,
                "reason": reason, "attempts": attempt + 1, "tokens": tokens}
    return {"unit": request["unit"], "expected": request["expected"], "passed": False, "reason": "unreadable_answer",
            "attempts": ATTEMPTS, "tokens": tokens}


def summary_of(records):
    """Units checked, passed and the rejection reasons, from textcheck records."""
    from collections import Counter
    units = [u for r in records for u in r["units"]]
    return {"rows": len(records), "rows_passed": sum(r["passed"] for r in records), "units": len(units),
            "units_passed": sum(u["passed"] for u in units),
            "reasons": dict(Counter(u["reason"] for u in units if not u["passed"]).most_common())}

