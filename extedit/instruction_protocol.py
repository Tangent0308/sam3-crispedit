"""Unified instruction protocol 2.0; dependency-free canonical implementation.

Vendored byte-for-byte in sa1b_pipeline/instruction_protocol.py. Change both
copies together; cross-repository conformance is checked during migration.
"""

from __future__ import annotations

import json
import re

VERSION = "2.0"
MASK_SEMANTICS_VERSION = "2.0"
OPERATIONS = {"add", "replace", "remove", "attribute"}
PROPERTIES = {"color", "material", "finish", "pattern", "texture", "text", "tint", "opacity"}
REMOVE_MODES = {"whole", "part", "contents", "contents_only", "lowerlegs", "hem"}
RELATIONS = {
    "to",
    "on",
    "in",
    "inside",
    "around",
    "across",
    "along",
    "over",
    "under",
    "above",
    "below",
    "beside",
    "near",
    "at",
    "attached to",
    "clipped to",
    "hanging from",
    "resting on",
    "mounted on",
    "sewn onto",
    "tied around",
    "pinned to",
    "hooked over",
    "strapped over",
}
REFERENCE_RELATIONS = {"under", "above", "below", "beside", "near"}


def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def noun(value):
    value = clean(value)
    return value if re.match(r"^(the|a|an|its|each|every|both)\b", value, re.I) else "the " + value


def make_spec(
    operation,
    ref,
    scope,
    payload="",
    *,
    target_part="",
    remove_mode="whole",
    attribute_kind=None,
    attributes=None,
    placement=None,
    qualifier="",
):
    spec = dict(
        version=VERSION,
        operation=clean(operation),
        ref=clean(ref),
        scope=clean(scope),
        payload=clean(payload),
        target_part=clean(target_part),
        remove_mode=remove_mode,
        attributes=attributes or [],
        placement=placement or {},
        qualifier=clean(qualifier),
    )
    if operation == "attribute" and not attributes:
        spec["attributes"] = [
            {
                "property": attribute_kind or "color",
                "value": str(payload) if attribute_kind == "text" else clean(payload),
            }
        ]
    validate_spec(spec)
    return spec


def validate_spec(s):
    allowed = {
        "version",
        "operation",
        "ref",
        "scope",
        "payload",
        "target_part",
        "remove_mode",
        "attributes",
        "placement",
        "qualifier",
    }
    if set(s) != allowed or s["version"] != VERSION:
        raise ValueError("invalid instruction protocol version or fields")
    op = s["operation"]
    if op not in OPERATIONS or not clean(s["ref"]) or not clean(s["scope"]):
        raise ValueError("operation, source ref and verified scope are required")
    if s["scope"].startswith("the selected "):
        raise ValueError("scope must be a noun phrase, without the selected prefix")
    if s["remove_mode"] not in REMOVE_MODES:
        raise ValueError("unknown remove_mode")
    if op != "remove" and s["remove_mode"] != "whole":
        raise ValueError("remove_mode is only valid for remove")
    if op in {"add", "replace"} and not s["payload"]:
        raise ValueError("new item/replacement is required")
    if op == "remove" and s["payload"]:
        raise ValueError("remove payload must be empty; use target_part/remove_mode/qualifier")
    if op == "remove" and s["remove_mode"] == "part" and not s["target_part"]:
        raise ValueError("part removal requires target_part")
    if op == "remove" and s["target_part"] and s["remove_mode"] != "part":
        raise ValueError("target_part removal requires remove_mode=part")
    if op == "add" and s["target_part"]:
        raise ValueError("add uses placement.site, not target_part")
    p = s["placement"]
    if not isinstance(p, dict) or set(p) - {"relation", "site", "anchor_mode"}:
        raise ValueError("invalid placement fields")
    if p:
        if op != "add" or p.get("relation", "to") not in RELATIONS:
            raise ValueError("invalid placement relation/operation")
        mode = p.get("anchor_mode", "support")
        if p.get("relation", "to") == "to" and p.get("site"):
            raise ValueError("a placement site needs an explicit spatial relation")
        if mode not in {"support", "reference"}:
            raise ValueError("unknown anchor_mode")
        if p.get("relation") in REFERENCE_RELATIONS and mode != "reference":
            raise ValueError("relative placement requires anchor_mode=reference")
    attrs = s["attributes"]
    if not isinstance(attrs, list) or (op == "attribute") != bool(attrs):
        raise ValueError("attributes are required only for attribute operations")
    seen = set()
    for a in attrs:
        if (
            not isinstance(a, dict)
            or set(a) != {"property", "value"}
            or a["property"] not in PROPERTIES
            or not isinstance(a["value"], str)
            or not clean(a["value"])
        ):
            raise ValueError("unknown or empty attribute property/value")
        if a["property"] in seen:
            raise ValueError("duplicate attribute property")
        seen.add(a["property"])
    if "tint" in seen and len(seen) > 1:
        raise ValueError("tint must be a standalone color edit")
    if "opacity" in seen and len(seen) > 1:
        raise ValueError("opacity must be a standalone edit")
    if seen == {"opacity"} and attrs[0]["value"] not in {"transparent", "translucent", "opaque"}:
        raise ValueError("opacity value must be transparent, translucent or opaque")


def render(s, has_ref=True, *, selector_token=""):
    """Render semantic slots; an optional token binds only the selector, never payload."""
    validate_spec(s)
    source = s["ref"] if has_ref else "the selected region"
    if selector_token and not has_ref:
        source += " " + selector_token
    target = noun(s["target_part"]) + " of " + source if has_ref and s["target_part"] else source
    op, payload = s["operation"], s["payload"]
    if op == "add":
        p = s["placement"]
        anchor = noun(p["site"]) + " of " + source if has_ref and p.get("site") else source
        if selector_token and has_ref:
            anchor += " " + selector_token
        text = f"Add {payload} {p.get('relation', 'to')} {anchor}"
    elif op == "replace":
        if selector_token and has_ref:
            target += " " + selector_token
        text = f"Replace {target} with {payload}"
    elif op == "remove":
        mode = s["remove_mode"]
        text = {
            "whole": f"Remove {source}",
            "part": f"Remove {noun(s['target_part'])} from {source}",
            "contents": f"Remove {source}, including its contents",
            "contents_only": f"Remove the contents from {source}",
            "lowerlegs": f"Remove the lower legs of {source} to make knee-length shorts",
            "hem": f"Remove the lower section of {source} to make it knee-length",
        }[mode]
        if not has_ref:
            text = {
                "whole": f"Remove {source}", "part": f"Remove {source}",
                "contents_only": f"Remove {source}",
                "contents": f"Remove {source}, including its contents",
                "lowerlegs": f"Remove {source} to make knee-length shorts",
                "hem": f"Remove {source} to make it knee-length",
            }[mode]
        elif selector_token:
            # Preserve legacy Ref wording; bind the full explicitly edited part.
            locator = (noun(s["target_part"]) + " from " + source if mode == "part" else
                       "the contents from " + source if mode == "contents_only" else
                       "the lower legs of " + source if mode == "lowerlegs" else
                       "the lower section of " + source if mode == "hem" else source)
            text = text.replace(locator, locator + " " + selector_token, 1)
    else:
        if selector_token and has_ref:
            target += " " + selector_token
        attrs = s["attributes"]
        prop, value = attrs[0]["property"], attrs[0]["value"]
        if prop == "tint":
            text = f"Tint {target} {value}"
        elif prop == "opacity":
            text = f"Make {target} {value}"
        else:

            def expression(a):
                name = "surface texture" if a["property"] == "texture" else a["property"]
                value = (
                    json.dumps(a["value"], ensure_ascii=False)
                    if a["property"] == "text"
                    else a["value"]
                )
                return name, value

            name, value = expression(attrs[0])
            text = f"Change the {name} {'on' if prop == 'text' else 'of'} {target} to {value}"
            for a in attrs[1:]:
                name, value = expression(a)
                text += f" and its {name} to {value}"
    qualifier = s["qualifier"] if has_ref else blind_qualifier(s)
    if qualifier:
        text += ", " + qualifier
    return text + "."



BLIND_RULE = ('Split the instruction into "where" and "what": replace the entire "where" '
              'with "the selected region", whose source mask selects exactly that region; '
              'keep the "what" unchanged.')


def blind_qualifier(spec):
    """Drop protections naming scene objects; retain generic edit requirements.

    Unknown qualifiers are kept for explicit deterministic rejection, not silently
    discarded as if their semantics were irrelevant.
    """
    q = clean(spec.get("qualifier", ""))
    if q.lower().startswith("preserving "):
        generic = {"preserving the underlying surface", "preserving its shape",
                   "preserving the original shape", "preserving the surface texture"}
        if q.lower() in generic:
            return q
        # Only discard a protection when it demonstrably points back to the
        # scene. Unknown edit requirements must not silently disappear.
        source_words = set(re.findall(r"[a-z]+", " ".join(
            str(spec.get(k, "")) for k in ("ref", "scope", "target_part")
        ).lower())) - {"the", "a", "an", "of", "on", "in", "with", "its", "and"}
        named_source = any(re.search(r"\b" + re.escape(word) + r"\b", q, re.I)
                           for word in source_words)
        scene_relation = re.search(
            r"\b(?:left|right|upper|lower|front|back|behind|near|adjacent|neighboring)\b"
            r"|\b(?:on either side|growing in it|next to)\b", q, re.I)
        if named_source or scene_relation:
            return ""
    return q


def validate_noref(spec, text):
    """Check canonical blind slots, exempting payload/value/result/relation slots.

    Exact slot rendering prevents leaked source nouns, adjectives and position
    words without incorrectly banning those same words in a requested payload.
    Relations (including reference-add near/under) are allowed by protocol §3.
    """
    validate_spec(spec)
    if text != render(spec, False):
        raise ValueError("no-ref differs from canonical blind semantic slots")
    q = blind_qualifier(spec)
    if q and q.lower() not in {"preserving the underlying surface", "preserving its shape",
                              "preserving the original shape", "preserving the surface texture"}:
        raise ValueError("unclassified no-ref qualifier; resolve location versus edit requirement")
    for value in [spec.get("payload", "")] + [a["value"] for a in spec["attributes"]
                                                if a["property"] != "text"]:
        if re.search(r"\b(?:a matching one|another one like it|same as (?:it|the)|like the original)\b", value, re.I):
            raise ValueError("payload must name the requested content without a source reference")
    return True

def validate_pair(spec, instruction_ref, instruction_noref):
    validate_noref(spec, instruction_noref)
    if render(spec, True) != instruction_ref or render(spec, False) != instruction_noref:
        raise ValueError("instruction pair differs from the shared semantic specification")


def length_advice(ref, noref):
    return [
        f"{mode}_length_review:{len(text.split())}"
        for mode, text, budget in [("ref", ref, 26), ("noref", noref, 16)]
        if len(text.split()) > budget
    ]


def needs_local_execution(spec):
    return bool(spec.get("target_part")) or spec.get("remove_mode") in {
        "contents",
        "contents_only",
        "lowerlegs",
        "hem",
    }


def mask_contract(spec):
    """One semantic contract for benchmark locators and training source masks.

    Does not manufacture an execution region or claim visual correctness.
    """
    validate_spec(spec)
    add = spec["operation"] == "add"
    reference = add and spec["placement"].get("anchor_mode") == "reference"
    return {
        "version": MASK_SEMANTICS_VERSION,
        "source_role": "anchor" if add else "target",
        "selector": "actual edited source region; add selects its direct attachment site or support anchor",
        "locator_equivalence": "ref/point/box/mask identify the same source selector",
        "output_difference_ground_truth": False,
        "anchor_mode": spec["placement"].get("anchor_mode", "support") if add else None,
        "direct_support_first": add,
        "reference_exception": (
            "direct additions must be unnatural; adjacent unique site with visible support; "
            "no remote landmark; No-ref retains relation but hides host and site identity"
        )
        if reference
        else None,
        "execution_region_required": needs_local_execution(spec) or reference,
        "source_region_resolution_required": needs_local_execution(spec) or bool(add and spec["placement"].get("site")),
        "unrequested_source_content_preserved": True,
        "allowed_effects": "only requested edit and physically necessary local contact, occlusion, revealed background, edges, lighting or shadows",
    }
