"""Unified text re-check before the training data are merged (2026-10-24; docs/EXTEDIT_TEXT_RECHECK.md).

Every text-related unit of the five protocol-2.0 edit sources (SA-1B 50k "D1", cleaned v1 "D3" / "D4", extedit
batches "D10" / "D11"):
* text change: an attribute change of property "text"; the requested text is the spec value;
* text rendering: an add or replace whose new object may carry written text (a quoted string or a word such as
  "reading", "text", "letters" in the new-object description); Qwen3.8-27B takes the requested text from the
  description and may find none ("a child reading a book"), which makes the unit no text rendering.
Two checks; they are recorded only, the decision is made in code at the merge after a calibration:
* request: the instruction alone, two votes (thinking low) giving the requested text and whether it is garbled,
  i.e. gibberish or a misspelling of real words ("Stadiiom", "IVECCO RACINGTEAMS"), as the source instructions of
  ScaleEdit often are; names, brands, codes, numbers, other languages and cleanly made-up names are not;
* reading: the written text read back from BEFORE / AFTER close-ups exactly as the extedit text check reads it
  (textcheck.py: greedy, no thinking, not told the requested text) and compared in code: around S for a text
  change, around the edited region for a text rendering.  The D10 / D11 text changes went through that check when
  they were labeled, so they get the request check only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import masks as mk
from . import textcheck as tc
from . import views as vw
from .llm import parse_json
from .sources import DATASETS

ATTEMPTS = 3
VOTES = 2
QUOTED = re.compile(r'["“”]|(?<![A-Za-z])\'[^\']+\'(?![A-Za-z])')
TEXT_WORDS = re.compile(r"\b(reading|reads|saying|says|labell?ed|lettered|inscribed|printed|text|words?|letters|"
                        r"lettering|slogan|caption|headline|title|numbers?|digits|name)\b", re.I)

# The five protocol-2.0 edit sources (SAMTokEdit v3 doc 01); D10 / D11 text changes are not read again.
DEFAULT_SOURCES = {
    "D1": {"kind": "release", "root": f"{DATASETS}/sa1b_samtok_edit_50k_20261014/release",
           "files": [("annotations/train_protocol_v2.jsonl", False)]},
    "D3": {"kind": "manifest", "root": f"{DATASETS}/SAMTokEdit_Cleaned_20261009",
           "files": [("protocol_adapt_qwen38_27b_v1/protocol_v2_blind/manifest_protocol_v2.jsonl", False)]},
    "D4": {"kind": "manifest", "root": f"{DATASETS}/SAMTokEdit_Cleaned_20261009",
           "files": [("protocol_multi_qwen38_27b_v1/protocol_v2_blind/manifest_multi_protocol_v2.jsonl", True)]},
    "D10": {"kind": "manifest", "root": f"{DATASETS}/ExtEdit_Protocol2_20261021", "read_changes": False,
            "files": [("manifest_protocol_v2.jsonl", False), ("manifest_multi_protocol_v2.jsonl", True)]},
    "D11": {"kind": "manifest", "root": f"{DATASETS}/ExtEdit_Protocol2_20261022", "read_changes": False,
            "files": [("manifest_protocol_v2.jsonl", False), ("manifest_multi_protocol_v2.jsonl", True)]},
}

REQUEST_PROMPT = """An image-editing instruction may ask for written text in the edited image.

Instruction: "{instruction}"
{subject}

Answer two questions.
1. text: the exact text the edited image must show there, character by character as the instruction gives it: keep letter case and spelling, join several lines with one space. Write "" when the instruction asks for no written text, for example "a child reading a book", "a number of birds", or a logo or emblem without letters.
2. garbled: true when that text looks broken: random letters, or a misspelling of common words (for example "Stadiiom" for "Stadium", "RESTARANT" for "RESTAURANT"). False for correctly spelled words, names, brands, abbreviations, codes and numbers, text in other languages, and cleanly made-up product or place names. False when text is "".

Return only JSON: {{"text": "...", "garbled": false}}"""
SUBJECT = {"change": 'The text it asks for: "{expected}".',
           "render": 'The new object it asks for: "{payload}".'}


def text_kind(spec):
    """"change", "render" (a candidate: the request check decides) or None."""
    op = spec["operation"]
    if op == "attribute" and spec.get("attributes") and spec["attributes"][0].get("property") == "text":
        return "change"
    payload = spec.get("payload") or ""
    if op in ("add", "replace") and (QUOTED.search(payload) or TEXT_WORDS.search(payload)):
        return "render"
    return None


def _rle(region):
    if region and region.get("rle_counts") and region.get("rle_size"):
        return {"rle_size": list(region["rle_size"]), "rle_counts": region["rle_counts"]}
    return None


def _png_rle(path):
    return mk.encode(np.asarray(Image.open(path).convert("L")) > 0)


def build_units(sources, out_path):
    """Write the units file: one line per text-related unit of the given sources.

    sources: {name: {"kind": "release" | "manifest", "root": dataset dir, "files": [(relative path, multi)],
    "read_changes": bool}}.  "release" is the SA-1B 50k annotation format (S as a PNG mask), "manifest" the D3 / D4
    format (regions as RLE).  Image paths are written absolute.  Returns the counts per source and kind."""
    from collections import Counter
    counts = Counter()
    with open(out_path, "w") as stream:
        for name, src in sources.items():
            root = Path(src["root"])
            for rel, multi in src["files"]:
                for line in open(root / rel):
                    record = json.loads(line)
                    units = record["units"] if multi else [record]
                    for k, unit in enumerate(units):
                        spec = unit["instruction_spec"]
                        kind = text_kind(spec)
                        if kind is None:
                            continue
                        rid = record.get("id") or record.get("sample_id")
                        if src["kind"] == "release":
                            s_area = {"png": str(root / record["mask"])}
                            target = record["edited_image"]
                            ref = record.get("instruction") or record.get("instruction_noref")
                        else:
                            s_area = _rle(unit.get("selection"))
                            target = record["target_image"]
                            ref = unit.get("instruction_ref") or unit.get("instruction_noref")
                        changed = (_rle(unit.get("edited_region")) or _rle(unit.get("execution_region")) or s_area)
                        area = s_area if kind == "change" else changed
                        read = kind == "render" or src.get("read_changes", True)
                        stream.write(json.dumps({
                            "id": f"{name}:{rid}#{k}", "source": name, "record": rid, "unit": k, "kind": kind,
                            "instruction": ref, "payload": spec.get("payload") or "",
                            "expected": spec["attributes"][0].get("value") if kind == "change" else None,
                            "source_image": str(root / record["source_image"]), "target_image": str(root / target),
                            "area": area, "read": read}, ensure_ascii=False) + "\n")
                        counts[f"{name}:{kind}"] += 1
    return dict(counts)


def area_mask(area, shape):
    """The unit's area as a boolean mask of the source image's shape."""
    if area is None:
        return None
    mask = (np.asarray(Image.open(area["png"]).convert("L")) > 0) if "png" in area else mk.decode(area)
    if mask.shape != shape:
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0
    return mask


def read_parts(unit):
    """Chat parts of the read-back (BEFORE and AFTER close-ups, the area outlined), or None without an area; a
    top-level function so that a process pool draws them off the event loop."""
    source = Image.open(unit["source_image"]).convert("RGB")
    target = Image.open(unit["target_image"]).convert("RGB")
    if target.size != source.size:
        target = target.resize(source.size, Image.Resampling.LANCZOS)
    mask = area_mask(unit["area"], (source.height, source.width))
    if mask is None or not mask.any():
        return None
    before, after = tc.close_ups(source, target, mask)
    return vw.chat_parts([("Image 1, before the edit:", before), ("Image 2, after the edit:", after)])


def parse_request(text):
    value = parse_json(text)
    garbled = value.get("garbled")
    if isinstance(garbled, str):
        garbled = {"true": True, "false": False}.get(garbled.strip().lower())
    if not isinstance(garbled, bool):
        raise ValueError("garbled must be true or false")
    return {"text": str(value.get("text") or ""), "garbled": garbled}


async def request_votes(client, unit):
    """Two votes on the requested text (one request; asked again for the votes that could not be parsed)."""
    prompt = REQUEST_PROMPT.format(instruction=unit["instruction"],
                                   subject=SUBJECT[unit["kind"]].format(expected=unit["expected"], payload=unit["payload"]))
    base, votes, tokens, problem = prompt, [], 0, None
    for attempt in range(ATTEMPTS):
        response = await client.call([], prompt, "request", seed=attempt, n=VOTES - len(votes))
        tokens += response["tokens"]
        for choice in response["choices"] if "choices" in response else [response]:
            try:
                votes.append(parse_request(choice["text"]))
            except Exception as exc:  # noqa: BLE001 - malformed answer: ask again
                problem = f"{type(exc).__name__}: {exc}"
        if len(votes) >= VOTES:
            break
        prompt = base + f"\n\nYour previous answer was not usable ({problem}). Answer again."
    return votes, tokens


def expected_text(unit, votes):
    """The text to read back: the spec value of a text change; for a rendering the text the first vote that found
    one took from the description ("" when none did)."""
    if unit["kind"] == "change":
        return unit["expected"]
    return next((v["text"] for v in votes if v["text"].strip()), "")


async def check_unit(client, unit, get_parts):
    """Request votes, then (when the unit is read and there is text to read) the read-back and its decision;
    `get_parts()` is awaited for the read-back views only when they are needed."""
    votes, tokens = await request_votes(client, unit)
    record = {"id": unit["id"], "source": unit["source"], "kind": unit["kind"], "votes": votes, "tokens": tokens,
              "expected": expected_text(unit, votes), "reading": None, "read_passed": None, "read_reason": None}
    if len(votes) < VOTES:
        record["error_votes"] = "unreadable_answer"
    if unit["read"] and record["expected"].strip():
        parts = await get_parts()
        if parts is None:
            record["read_reason"] = "no_area"
            return record
        result = await tc.check_unit(client, {"unit": unit["unit"], "expected": record["expected"], "parts": parts})
        record["tokens"] += result.pop("tokens", 0)
        record["reading"] = {k: result.get(k) for k in ("before", "after", "after_legible", "after_garbled")}
        record["read_passed"], record["read_reason"] = result["passed"], result["reason"]
    return record


def summary_of(records):
    from collections import Counter
    out = {"units": len(records), "by_source_kind": dict(Counter(f"{r['source']}:{r['kind']}" for r in records))}
    out["renderings_with_text"] = sum(r["kind"] == "render" and bool(r["expected"].strip()) for r in records)
    out["garbled_votes"] = dict(Counter(sum(v["garbled"] for v in r["votes"]) for r in records))
    out["read"] = dict(Counter("not_read" if r["read_passed"] is None and r["read_reason"] is None else
                               ("passed" if r["read_passed"] else r["read_reason"]) for r in records))
    out["unreadable_votes"] = sum("error_votes" in r for r in records)
    return out
