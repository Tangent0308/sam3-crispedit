"""Match a singular semantic proposal to its intended localization box."""

import re


def container_object_prompt(ref, edit_type, region_mode='object', density='object', layout='single'):
    """A whole replaced/removed vessel keeps its contents in the grounding identity.

    Only explicit container heads qualify. Do not turn a part, a group, or added
    contents into its owner, and do not simplify arbitrary compound objects.
    """
    if edit_type not in {'remove', 'replace'} or region_mode != 'object' or density != 'object' or layout != 'single':
        return ''
    match = re.fullmatch(
        r'(?:(?:the|a|an|small|large|white|black|red|blue|green|brown|ceramic|wooden|metal|glass)\s+)*'
        r'(bowl|plate|cup|mug|basket|tray)\s+(?:with|containing|filled with|of)\s+.+',
        str(ref).strip(), re.I)
    return match[1].lower() if match else ''


def added_attachment_ref(change):
    """Resolve explicit unchanged-owner -> owner-with-added-part contradictions.

    Owner matching ignores articles and an explicit empty/bare state, but leaves
    synonyms/other transformations untouched. Never reduce a newly added complete
    object with its attachments.
    """
    source = str(change.get('source_ref', '')).strip()
    target = str(change.get('target_ref', '')).strip()
    description = str(change.get('change', ''))
    parts = re.split(r'\s+with\s+', target, maxsplit=1, flags=re.I)
    normalize = lambda value: re.sub(r'^(?:(?:a|an|the|empty|bare|unoccupied)\s+)+', '', value.strip().lower())
    if not source or len(parts) != 2 or normalize(source) != normalize(parts[0]):
        return ''
    explicitly_empty = bool(re.match(r'^(?:(?:a|an|the)\s+)?(?:empty|bare|unoccupied)\b', source, re.I))
    addition = bool(re.search(r'\b(?:added|addition|appeared)\b', description, re.I))
    contents = explicitly_empty and bool(re.search(r'\b(?:filled|populated)\b', description, re.I))
    if not (addition or contents):
        return ''
    if re.search(r'\b(?:recolo\w*|replaced|moved|reshaped|redesigned|turned|became)\b', description, re.I):
        return ''
    attachment = parts[1].strip(' .;')
    return attachment if 1 <= len(attachment.split()) <= 8 else ''


def atomic_body_refs(ref):
    match = re.fullmatch(r'(face|head) and (arms|hands)',ref.strip(),re.I)
    return [match[1].lower(),match[2].lower()] if match else [ref]


def ordinary_object_group(item):
    """Multiple solid instances are not particle/line-like sparse regions."""
    if item.get('region_layout') != 'nearby_group':
        return False
    return bool(re.match(r'^(?:(?:the|a|all)\s+)*(?:(?:group|pair|row|set|cluster)\s+of\s+)?'
                         r'(?:people|men|women|children|boys|girls|desks?|tables?|chairs?|armchairs?|'
                         r'sofas?|cars?|vehicles?|boats?|horses?|cookies?|plates?|bottles?)\b',
                         item.get('ref',''),re.I))


def single_object_prompt(ref, region_mode='object', density='object', layout='single'):
    if region_mode != 'object' or density != 'object' or layout != 'single':
        return False
    if re.search(r'\b(?:and|with|group|cluster|stack|pair|row|many|several|multiple|skin)\b',ref,re.I):
        return False
    # Conservative singular heads. Unknown nouns and plural/multipart concepts
    # retain multi-instance behavior; do not infer counts from mask components.
    return bool(re.search(r'\b(?:person|man|woman|boy|girl|child|car|vehicle|horse|cat|dog|bird|'
                          r'tent|balloon|chair|armchair|shirt|jacket|hoodie|robe|dress|arm|hand|'
                          r'face|head|phone|smartphone|jar|vase|tree|cake|plate|cup|cookie|boat|'
                          r'ship|eye|guitar|lamp|box|bottle|book|snake|mirror|seat|knife|stand|'
                          r'tassel|tray|bowl|body|top|topper|pedestal|sofa|umbrella|desk|table|cloak)\s*$',ref,re.I))


def match_single_candidate(candidates):
    """Use box overlap for identity, not semantic score alone or a blind union."""
    if not candidates:
        return []
    return [max(candidates,key=lambda item:(item['box_iou'],item['concept_score']))]
