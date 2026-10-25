"""Conservative contradictions explicitly stated by independently planned Refs.

This is not a physical relationship detector. It abstains on proximity, vague
ownership and ambiguous host names. Visual auditing remains necessary.
"""
import re

VERSION = 'explicit-host-conflicts-3-source-owned-parts'
_RELATION = re.compile(
    r'\b(?:of|attached to|hanging(?: \w+)? from|dropping(?: \w+)? from|'
    r'suspended from|worn by|carried by|held by|mounted on|fixed to|'
    r'fastened to|supported by)\s+(the .+)', re.I)
_DEPICTION = re.compile(r'\b(?:photo|picture|painting|portrait|statue|model|drawing|image) of\b', re.I)
_LOCATORS = {'left', 'right', 'leftmost', 'rightmost', 'upper', 'lower',
             'topmost', 'bottommost', 'foreground', 'background'}
_OPPOSITES = ({'left', 'right'}, {'leftmost', 'rightmost'},
              {'upper', 'lower'}, {'topmost', 'bottommost'},
              {'foreground', 'background'})


def _words(text):
    return re.findall(r"[a-z]+", text.lower())


def _head(text):
    # Match the head before a relational clause; do not confuse a rotor with
    # the helicopter named in its locating phrase.
    phrase = re.split(r'\b(?:of|on|in|at|with|behind|beside|flying|standing|'
                      r'hanging|dropping|held|worn|carried|attached|suspended|'
                      r'mounted|fixed|fastened|supported|owned|by|that|which)\b', text, maxsplit=1)[0]
    words = _words(phrase)
    return words[-1] if words else ''


def _same_host(anchor, ref):
    if _head(anchor) != _head(ref):
        return False
    a, b = set(_words(anchor)), set(_words(ref))
    if any((a & pair) and (b & pair) and (a & pair) != (b & pair)
           for pair in _OPPOSITES):
        return False
    # Exact Ref is sufficient. Otherwise require an explicit shared locator,
    # and all identity adjectives in the attachment's anchor to be present.
    if ' '.join(_words(anchor)) == ' '.join(_words(ref)):
        return True
    return bool(a & b & _LOCATORS) and a <= b


def find_conflicts(targets, assignments):
    eligible = [t for t in targets if t.get('feasible')]
    conflicts = []
    for child in eligible:
        if assignments[child['unit_id']] == 'remove':
            continue
        ref = child.get('ref_phrase', '')
        if _DEPICTION.search(ref):
            continue
        for match in _RELATION.finditer(ref):
            anchor = match.group(1).rstrip('., ')
            # Uniqueness among all selected potential hosts, not only removes.
            hosts = [t for t in eligible if t is not child
                     and _same_host(anchor, t.get('ref_phrase', ''))]
            if len(hosts) != 1 or assignments[hosts[0]['unit_id']] != 'remove':
                continue
            conflicts.append(dict(
                removed_host=hosts[0]['unit_id'], retained_attachment=child['unit_id'],
                host_ref=hosts[0]['ref_phrase'], attachment_ref=ref,
                relation=match.group(0),
                reason='explicit_host_removal_conflicts_with_retained_attachment'))
    return conflicts


def part_site_conflicts(targets, assignments, objects):
    """Explicit named attachment site plus source-mask containment.

    A roof deletion and a rooftop addition on the same selected building are
    contradictory even when the addition's Ref names the whole building.
    No proximity inference; unknown masks or unspecified sites abstain.
    """
    import numpy as np
    from PIL import Image
    from .geometry import load_mask
    eligible = [t for t in targets if t.get('feasible')]
    masks = {}

    def mask(uid):
        if uid in masks:
            return masks[uid]
        index = int(uid.removeprefix('U')) - 1
        if not 0 <= index < len(objects):
            return None
        obj = objects[index]
        path = obj.get('mask_path') or obj.get('mask')
        rle = obj.get('rle') or obj.get('segmentation')
        if path:
            with Image.open(path) as im:
                size = im.size
        elif isinstance(rle, dict) and len(rle.get('size', [])) == 2:
            size = tuple(reversed(rle['size']))
        else:
            return None
        value = load_mask(obj, size)
        if value.ndim != 2 or not value.any():
            return None
        masks[uid] = value
        return value

    conflicts = []
    for removed in eligible:
        if assignments[removed['unit_id']] != 'remove':
            continue
        ref = removed.get('ref_phrase', '')
        if _DEPICTION.search(ref):
            continue
        for child in eligible:
            if child is removed or assignments[child['unit_id']] == 'remove':
                continue
            child_ref = child.get('ref_phrase', '')
            if _DEPICTION.search(child_ref):
                continue
            # Inspect nested ownership too: "sole of the shoe of the man".
            anchors = []
            start = 0
            while match := _RELATION.search(child_ref, start):
                anchors.append(match[1].rstrip('., '))
                start = match.start() + 1
            # Stop at a second determiner, rather than treating the entire
            # "shoe on the runner's ..." phrase as its own host.
            anchors += [m[0][:-2].strip() for m in re.finditer(
                r"\bthe (?:(?!\bthe\b)[^.;])*?[’']s\b", child_ref, re.I)]
            identity = [a for a in anchors if _head(a) == _head(ref)
                        and set(_words(a)) <= set(_words(ref))]
            if not identity:
                continue
            host, region = mask(removed['unit_id']), mask(child['unit_id'])
            if host is None or region is None or host.shape != region.shape:
                continue
            coverage = float(np.count_nonzero(region & host) / np.count_nonzero(region))
            fraction = float(np.count_nonzero(region) / np.count_nonzero(host))
            if coverage >= .9 and fraction <= .7:
                conflicts.append(dict(removed_host=removed['unit_id'],
                    retained_attachment=child['unit_id'], host_ref=ref,
                    attachment_ref=child_ref, declared_owner=identity[0],
                    source_coverage=coverage, source_part_fraction=fraction,
                    reason='source_owned_part_conflicts_with_host_removal'))
        owner = re.search(r'\bof\s+(the .+)', ref, re.I)
        if not owner:
            continue
        part = _head(ref)
        for added in eligible:
            if assignments[added['unit_id']] != 'add':
                continue
            anchor = added.get('ref_phrase', '')
            if _DEPICTION.search(anchor) or _head(owner[1]) != _head(anchor):
                continue
            a, b = set(_words(owner[1])), set(_words(anchor))
            if any(a & pair and b & pair and a & pair != b & pair for pair in _OPPOSITES):
                continue
            site = (added.get('placement') or {}).get('site', '')
            explicit = bool(site and _head(site) == part)
            # A rooftop payload is an explicit location, even with site="".
            explicit |= part == 'roof' and bool(re.search(r'\brooftop\b', added.get('payload', ''), re.I))
            if not explicit:
                continue
            region, host = mask(removed['unit_id']), mask(added['unit_id'])
            if region is None or host is None or region.shape != host.shape:
                continue
            coverage = float(np.count_nonzero(region & host) / np.count_nonzero(region))
            fraction = float(np.count_nonzero(region) / np.count_nonzero(host))
            if coverage >= .9 and fraction <= .7:
                conflicts.append(dict(removed_host=removed['unit_id'],
                    retained_attachment=added['unit_id'], host_ref=ref, attachment_ref=anchor,
                    site=site, payload=added.get('payload', ''), source_coverage=coverage,
                    source_part_fraction=fraction,
                    reason='removed_attachment_site_conflicts_with_add'))
    return conflicts


def apply_conflicts(targets, assignments, objects=()):
    """Mark only incompatible host deletes infeasible; never enlarge masks."""
    conflicts = find_conflicts(targets, assignments)
    conflicts += part_site_conflicts(targets, assignments, objects)
    by_id = {t['unit_id']: t for t in targets}
    for item in conflicts:
        target = by_id[item['removed_host']]
        target['feasible'] = False
        target['reason'] = item['reason']
        target.setdefault('host_conflict_evidence', []).append(item)
    return conflicts
