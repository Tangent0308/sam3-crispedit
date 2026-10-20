"""Protocol 2.0 slots: deterministic repairs, problems the model is asked to fix, the spec, SAM3 prompts, flags.

normalize_slots, normalize_anchor, slot_problems, anchor_problems, part_as_s, sam_prompt and the word lists are ported
from samtok_edit v2 src/samtok_edit21/preparation/protocol_adapt.py ("protocol-adapt-4"); the spec is built with the
vendored protocol 2.0 module, whose fields are those of 1.0.
"""
from __future__ import annotations

import re

from . import instruction_protocol as ip

GRANULARITIES = ("object", "part", "surface")
RELATION_START = re.compile(r"(on|onto|in|into|inside|at|to|of|from|below|above|under|beside|near|next|around|across|"
                            r"along|over)\b")
HOST_PRONOUN = re.compile(r"\b(on|in|to|onto|into|around|from|at|of|over|under|across|along|inside)\s+(its|it)\b", re.I)
RELATION_ALIASES = {"onto": "on", "on top of": "on", "upon": "on", "into": "in", "within": "in", "next to": "beside",
                    "by": "beside", "alongside": "beside", "underneath": "under", "beneath": "under",
                    "attached": "attached to", "hanging on": "hanging from", "close to": "near", "nearby": "near"}
SIDE_RELATIONS = {"to the left of": "left side", "left of": "left side", "to the right of": "right side",
                  "right of": "right side"}
QUOTED = re.compile(r"(?<![A-Za-z0-9])['\"‘“]([^'\"‘’“”]+?)['\"’”](?![A-Za-z0-9])")
COUNT_WORDS = r"one|two|three|four|five|six|seven|eight|nine|ten|single|pair|few|several|\d+"
COLOR_WORDS = (r"red|blue|green|yellow|black|white|pink|purple|orange|brown|gr[ae]y|gold|golden|silver|beige|navy|"
               r"teal|violet|maroon|cyan|magenta|turquoise|crimson|ivory|cream|tan|olive|lavender|amber|bronze|"
               r"burgundy|khaki|coral|mint|lime|indigo|copper")
MATERIAL_WORDS = (r"wood|wooden|metal|metallic|steel|iron|copper|brass|leather|glass|marble|stone|plastic|fabric|denim|"
                  r"velvet|silk|wool|woolen|ceramic|porcelain|concrete|brick|rubber|paper|cardboard|bamboo|wicker|"
                  r"chrome|gold|silver|bronze|knitted|straw|linen|cotton|suede|granite|crystal")
VALUE_WORD = re.compile(rf"\b({COUNT_WORDS}|{COLOR_WORDS}|{MATERIAL_WORDS})\b", re.I)
SCOPE_LOCATOR = re.compile(rf"\b(left|right|leftmost|rightmost|center|centre|central|middle|foreground|background|"
                           rf"upper|lower|nearest|farthest|first|second|third|last|{COUNT_WORDS}|{COLOR_WORDS})\b", re.I)
IMAGE_POSITION = re.compile(r"\b(positioned in|occupying|of the (image|photo|picture|frame|scene)|quadrant|"
                            r"portion of|section of the image|area of the image)\b", re.I)


def clean_phrase(value):
    return ip.clean(value).strip(" .,;:")


def value_words(text):
    return {m.group(1).lower() for m in VALUE_WORD.finditer(text or "")}


def normalize_slots(slots):
    """Deterministic repairs of near-miss slots; every repair is listed so reviews can count them."""
    fixes, s = [], dict(slots)
    op = s["operation"] = ip.clean(s.get("operation")).lower()
    s["ref"] = clean_phrase(s.get("ref"))
    if s["ref"] and not re.match(r"(?i)^(the|a|an)\b", s["ref"]):
        s["ref"] = "the " + s["ref"]
        fixes.append("ref_article")
    s["scope"] = re.sub(r"(?i)^(the|a|an|selected)\s+", "", clean_phrase(s.get("scope")))
    s["target_part"] = re.sub(r"(?i)^(the|its)\s+", "", clean_phrase(s.get("target_part")))
    s["payload"] = clean_phrase(s.get("payload")) if op in ("add", "replace") else ""
    s["qualifier"] = clean_phrase(s.get("qualifier"))
    s["remove_mode"] = (ip.clean(s.get("remove_mode")).lower() or "whole") if op == "remove" else "whole"
    if op == "remove" and s["target_part"] and s["remove_mode"] == "whole":
        s["remove_mode"] = "part"
        fixes.append("remove_mode_part")
    s["attributes"] = []
    for item in (slots.get("attributes") or []) if op == "attribute" else []:
        if not isinstance(item, dict):
            continue
        prop = ip.clean(item.get("property")).lower().replace("colour", "color")
        prop = {"surface texture": "texture", "transparency": "opacity"}.get(prop, prop)
        value = str(item.get("value") or "")
        value = value.strip().strip("\"'“”‘’") if prop == "text" else clean_phrase(value)
        s["attributes"].append({"property": prop, "value": value})
    s["placement"] = {}
    if op == "add":
        raw = slots.get("placement") or {}
        relation = ip.clean(raw.get("relation")).lower() or "to"
        site = re.sub(r"(?i)^(the|its)\s+", "", clean_phrase(raw.get("site")))
        if relation in SIDE_RELATIONS:
            relation, site = "beside", site or SIDE_RELATIONS[relation]
            fixes.append("side_relation")
        if relation in RELATION_ALIASES:
            relation = RELATION_ALIASES[relation]
            fixes.append("relation_alias")
        if relation == "to" and site:
            relation = "on"
            fixes.append("to_with_site")
        mode = "reference" if relation in ip.REFERENCE_RELATIONS else "support"
        if ip.clean(raw.get("anchor_mode")).lower() not in ("", mode):
            fixes.append("anchor_mode")
        s["placement"] = {"relation": relation, "anchor_mode": mode, **({"site": site} if site else {})}
        normalize_anchor(slots, s, fixes)
    else:
        s.update(granularity="", host_ref="", host_scope="", part="")
    return s, fixes


def normalize_anchor(raw, s, fixes):
    """An add's S at one of three granularities and the host it belongs to; a part's ref follows from its host."""
    gran = ip.clean(raw.get("granularity")).lower()
    host_ref = clean_phrase(raw.get("host_ref"))
    host_scope = re.sub(r"(?i)^(the|a|an)\s+", "", clean_phrase(raw.get("host_scope")))
    part = re.sub(r"(?i)^(the|its)\s+", "", clean_phrase(raw.get("part")))
    if s["placement"]["anchor_mode"] == "reference" and gran != "object":
        gran = "object"  # the exception: the object the new item holds up, as a whole
        fixes.append("reference_object")
    if gran == "object":
        host_ref, host_scope, part = s["ref"], s["scope"], ""
    elif gran in ("part", "surface"):
        host_ref = ip.noun(host_ref) if host_ref else ""
        if gran == "part" and part and host_ref:
            ref = f"{ip.noun(part)} of {host_ref}"
            if ref != s["ref"]:
                s["ref"] = ref
                fixes.append("part_ref")
            if not s["scope"] and host_scope:
                s["scope"] = f"{host_scope} {part.split()[-1]}"
        if gran == "surface":
            host_scope = host_scope or s["scope"]
            part = ""
            if s["placement"].pop("site", None):
                fixes.append("surface_site")
    s.update(granularity=gran, host_ref=host_ref, host_scope=host_scope, part=part)


def to_spec(slots):
    return ip.make_spec(slots["operation"], slots["ref"], slots["scope"], slots["payload"],
                        target_part=slots["target_part"], remove_mode=slots["remove_mode"],
                        attributes=slots["attributes"] or None, placement=slots["placement"] or None,
                        qualifier=slots["qualifier"])


def slot_problems(slots):
    """Slot errors the protocol module does not catch; any of them sends the call back with feedback."""
    problems = []
    site = (slots.get("placement") or {}).get("site", "").lower()
    scope = slots.get("scope", "").lower()
    if slots.get("operation") == "add" and site and (scope == site or scope.endswith(" " + site)):
        problems.append(f'scope "{slots["scope"]}" names the spot on S; scope must name the kind of S itself')
    if IMAGE_POSITION.search(slots.get("ref", "")):
        problems.append(f'ref "{slots["ref"]}" describes the image frame; say "on the left" or "in the center" instead')
    if RELATION_START.match(site):
        problems.append(f'placement.site "{site}" starts with a relation word; site names only the spot on S, '
                        "e.g. \"seat\", and the relation goes in placement.relation")
    if slots.get("operation") in ("add", "replace") and re.match(r"(?i)(the|this|that|its|his|her|their)\b",
                                                                 slots.get("payload", "")):
        problems.append(f'payload "{slots["payload"]}" names something new: start it with "a", "an", "some" or a '
                        'number, never with "the" or a possessive')
    if slots.get("operation") == "add":
        problems += anchor_problems(slots)
    return problems


def anchor_problems(slots):
    """An add's anchor must be what the new item touches, named at a granularity the segment step can cut."""
    problems = []
    gran, relation = slots.get("granularity"), slots["placement"].get("relation", "to")
    if gran not in GRANULARITIES:
        problems.append('granularity must be "object", "part" or "surface"')
    if gran in ("part", "surface") and not slots.get("host_ref"):
        problems.append("host_ref must name the whole object or surface S belongs to")
    if gran == "part" and (not slots.get("part") or RELATION_START.match(slots["part"].lower())):
        problems.append('part must name the part of the host that S is, e.g. "top" or "seat", without a relation word')
    if relation == "to":
        problems.append('relation "to" says nothing: name how the new item meets S, e.g. "on", "in", "attached to", '
                        '"around" or "hanging from"; if it only stands next to or behind S, S is what it stands on')
    if relation in ("near", "above", "below", "beside"):
        problems.append(f'relation "{relation}" is not used: S is what the new item stands, rests or hangs on (an '
                        "object, a part of one, or a stretch of a surface such as the ground, floor, water or table "
                        'top); "under" only when the new item holds S up; if nothing visible carries the item, the '
                        "unit cannot be used")
    if re.search(r"\bsky\b", slots.get("host_scope", "") + " " + slots.get("scope", ""), re.I) and \
            relation not in ("in", "across"):
        problems.append('for the sky the relation is "in" (or "across" for a rainbow)')
    if HOST_PRONOUN.search(slots.get("payload", "")):
        problems.append(f'payload "{slots["payload"]}" refers to S with its/it; the payload names only the new item')
    return problems


def part_as_s(spec):
    """A named part of S becomes S itself (mask semantics 2.0: S is the actual edited source region)."""
    part = ip.noun(spec["target_part"])
    head = part.split()[-1].lower()
    ref = spec["ref"] if re.search(rf"\b{re.escape(head)}s?\b", spec["ref"], re.I) else f"{part} of {spec['ref']}"
    return ip.make_spec(spec["operation"], ref, re.sub(r"^the\s+", "", part), spec["payload"],
                        attributes=spec["attributes"] or None, qualifier=spec["qualifier"],
                        remove_mode="whole" if spec["remove_mode"] == "part" else spec["remove_mode"])


def sam_prompt(spec, host_ref="", part=False):
    """SAM3's text prompt: the ref without its article; for add the whole object or surface S belongs to (SAM3
    segments whole objects reliably and the spot is cut later); for a named part that became S, the part's kind
    (a ref such as "the shirt of the man on the boat" makes SAM3 take the man)."""
    text = (host_ref or spec["scope"]) if spec["operation"] == "add" else spec["scope"] if part else spec["ref"]
    return re.sub(r"(?i)^(the|a|an)\s+", "", text)


def text_flags(instruction, native_op, spec, ref, noref):
    """Review flags on the wording: values the original never mentioned, a retyped operation, scope or ref that
    carry locators, lengths over the style budget."""
    flags = []
    new_values = {spec["payload"].lower()} | {a["value"].lower() for a in spec["attributes"]}
    if is_text(spec):
        quoted = QUOTED.findall(instruction or "")
        if quoted and spec["attributes"][0]["value"] not in quoted:
            flags.append("text_value_not_quoted_in_original")
    invented = sorted(value_words(" ".join(new_values)) - value_words(instruction) - {"one", "single"})
    if invented:
        flags.append("new_values:" + ",".join(invented))
    if SCOPE_LOCATOR.search(spec["scope"]) or re.search(r"\d|'s\b", spec["scope"]):
        flags.append("scope_not_plain")
    if IMAGE_POSITION.search(spec["ref"]):
        flags.append("ref_image_position")
    if native_op and native_op != spec["operation"] and not (native_op == "text" and is_text(spec)):
        flags.append(f"retyped:{native_op}->{spec['operation']}")
    return flags + ip.length_advice(ref, noref)


def is_text(spec):
    return spec["operation"] == "attribute" and spec["attributes"][0]["property"] == "text"
