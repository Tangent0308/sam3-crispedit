"""Cheap semantic exclusions before quota assignment; never a visual verifier.

These gates exclude an operation, not a source mask. Other operations can still
use the region. Dense category hints are advisory and are not copied into refs.
"""
from __future__ import annotations

import re

MULTI_PLAN_VERSION = "multi-visible-fit-7-physical-contrast"
_PEOPLE = {"person", "people", "man", "woman", "boy", "girl", "child", "customer"}
_ANATOMY = {"head", "face", "arm", "hand", "leg", "foot", "finger", "ear", "eye", "nose", "mouth", "neck", "torso", "body"}
_CORE_CLOTHES = {"jeans", "trousers", "pants", "shorts", "shirt", "t-shirt", "blouse", "jacket", "coat", "dress", "skirt", "sleeve", "sweater", "clothing", "garment", "hijab"}
_MANUFACTURED_HOSTS = {"chair", "table", "lamp", "doll", "statue", "mannequin", "robot", "toy", "machine", "awning", "crane", "clock", "sofa"}


def _heads(value):
    heads = {p.strip().split()[-1] for p in re.split(r"[,;/]", str(value).lower().replace("_", " ")) if p.strip()}
    plurals = {word + "s": word for word in _ANATOMY | _PEOPLE | _CORE_CLOTHES | _MANUFACTURED_HOSTS}
    plurals.update({"feet": "foot", "men": "man", "women": "woman", "children": "child", "bodies": "body"})
    return {plurals.get(head, head) for head in heads}


def operation_exclusion(category: str, operation: str) -> str:
    """Conservative triage for unqualified source labels and exact plan scopes."""
    heads = _heads(category)
    words = set(re.findall(r"[a-z]+", str(category).lower()))
    anatomy = bool(heads & _ANATOMY) and not words & _MANUFACTURED_HOSTS
    if anatomy and operation in {"remove", "replace", "attribute"}:
        return "anatomy requires a resolved editable subpart, not whole-part erasure/replacement/recoloring"
    if heads & _PEOPLE and operation == "attribute":
        return "whole-person mask cannot silently become clothing or hair"
    if heads & _CORE_CLOTHES and operation == "remove":
        return "core clothing removal has unresolved body/content reconstruction"
    return ""


def validate_multi_semantics(item, source_category=""):
    # A clothing mask must never silently become the wearer, even for add.
    if _heads(source_category) & _CORE_CLOTHES and _heads(item.get("scope", "")) & (_ANATOMY | _PEOPLE):
        raise ValueError("multi garment mask reinterpreted as anatomy/person")
    # Direct anchors cannot authorize a body zone outside the named garment.
    # This is a narrow contradiction gate, not a complete attachment ontology.
    if item.get("operation") == "add":
        site = str((item.get("placement") or {}).get("site", "")).lower()
        if re.match(r"^(?:the\s+)?(?:his|her|its|their)\s+", site):
            raise ValueError("multi attachment site must be a host-relative noun without a possessive")
        if (_heads(item.get("scope", "")) & {"jeans", "trousers", "pants", "shorts", "skirt"}
                and re.search(r"\b(?:breast|chest|shoulder|collar|lapel)\b", site)):
            raise ValueError("multi direct attachment site lies outside the lower garment")
    reason = operation_exclusion(item.get("scope", ""), item.get("operation"))
    if reason:
        raise ValueError("multi operation incompatible with source extent: " + reason)


MULTI_PHYSICAL_RULES = """Before approving an edit, verify the selected entity, its host and what must remain.
Removing worn clothing must NOT erase the covered person: a missing leg, cut-off lower body or floating torso is a severe failure even if background inpainting is seamless. Removing a glove may pass when a plausible bare hand remains. Removing an independent complete person may pass when no body fragments remain. Never use the instruction to excuse physically impossible remnants.
The source mask of a shirt/sleeve is fabric, not the person/arm; a head mask is not a hair-only mask. Recoloring clothes while saying 'change the person blue' is a scope failure. A new object added next to an unchanged original is not replacement. Add needs an actually visible new item with the requested contact relationship; a chin strap drawn on top of a hat is not correctly installed. Minor texture variation, source blur or unconventional plausible style alone is not a failure."""


def same_scope_identity(entity: str, scope: str) -> bool:
    """Allow noun-preserving modifiers, never a host/part or location rewrite.

    This only checks duplicate label consistency. It is not visual grounding.
    Keep the original strings in the record for later visual audits.
    """
    def words(value):
        return re.sub(r"^(?:the|a|an)\s+", "", str(value).strip().casefold()).split()
    left, right = words(entity), words(scope)
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = sorted((left, right), key=len)
    if long[-len(short):] != short:
        return False
    prefix = set(long[:-len(short)])
    # Relational phrases and explicit partial extents can change the entity.
    if prefix & {"of", "in", "on", "at", "with", "without", "behind", "under",
                 "above", "below", "part", "portion", "section", "half", "upper", "lower"}:
        return False
    return True
