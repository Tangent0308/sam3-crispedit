"""Metadata-first group selection from SA-1B companion dense labels (opt-in).

The companion annotation already names every dense mask (category aliases +
detector score). Selection uses it BEFORE any planner call:

* per-category operation affordance, judged once per category string by a
  text-only model call and cached (no image, no per-mask cost);
* only `natural` (mask, operation) pairs are eligible; geometry must also pass
  the existing operation quality gate;
* masks with a low label score, a same-category parent (fragment), more than
  four similar-size same-category competitors, a person/object mixed alias
  list, or an extreme area are skipped.

Twins may over-generate a third same-operation candidate (`twin_keep=2`):
post-planning evidence gates then keep two passing members. Multi targets use
distinct operations, most-needed first. Selection never edits masks.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

VERSION = 'evidence-selection-3-support-gap'
OPERATIONS = ('add', 'remove', 'replace', 'attribute')
AFFORD_KEY = {'add': 'add_anchor', 'remove': 'remove', 'replace': 'replace', 'attribute': 'attribute'}
PERSON = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'people', 'kid', 'player', 'runner', 'rider',
          'pedestrian', 'tourist', 'worker', 'crowd', 'soldier', 'lady', 'guy', 'spectator', 'band'}
STOP = {'the', 'a', 'an', 'of', 'and', 'with', 'in', 'on', 'object', 'objects', 'part', 'parts', 'item', 'items',
        'thing', 'things', 'surface', 'area', 'region', 'piece', 'section'}
DEFAULTS = {'min_score': .6, 'min_area': .0015, 'max_area': .3, 'max_similar_competitors': 4,
            'twin_max_area_ratio': 3.0, 'separation': .05,
            # Reviewed history: gated removals of masks >=2% of the image were 0/7 acceptable.
            'max_area_by_operation': {'remove': .02}}

AFFORD = '''An image region was annotated with the category (aliases separated by commas): "{category}".
Example description of such a region: "{description}"
For a realistic photo-editing dataset, rate how naturally each operation applies to the WHOLE annotated thing in an ordinary photo:
- remove: delete it entirely and fill in the background. Natural for separable objects (a parked car, a sign, a bag, a bird, a cup). Unnatural for clothing currently worn, body parts, structural or load-bearing parts, ground/sky/walls/surfaces that other things rest on, or parts whose absence leaves a broken host.
- replace: substitute it with a DIFFERENT kind of object or a clearly different design that fills the same place and role. Natural for discrete objects with plausible alternatives (a fork -> a spoon, a hat -> a helmet, a sign -> a different sign). Unnatural when only its color or material could change (walls, panels, arches, a vase's handle, grass) and for body parts.
- attribute: change its color, material, texture or pattern while keeping the thing itself.
- add_anchor: place a new physical item directly on or attached to it.
Use "natural", "conditional" (depends strongly on the specific instance), or "unnatural".
Return JSON only: {{"remove": "...", "replace": "...", "attribute": "...", "add_anchor": "..."}}'''
_LEVEL = {'type': 'string', 'enum': ['natural', 'conditional', 'unnatural']}
AFFORD_SCHEMA = {'type': 'object', 'properties': {k: _LEVEL for k in AFFORD_KEY.values()},
                 'required': list(AFFORD_KEY.values()), 'additionalProperties': False}


def lemma(word: str) -> str:
    w = word.lower()
    irregular = {'people': 'person', 'men': 'man', 'women': 'woman', 'children': 'child', 'feet': 'foot',
                 'teeth': 'tooth', 'leaves': 'leaf', 'shelves': 'shelf', 'knives': 'knife', 'glasses': 'glasses',
                 'trousers': 'trousers', 'pants': 'pants', 'jeans': 'jeans', 'shorts': 'shorts', 'clothes': 'clothes',
                 'buses': 'bus', 'boxes': 'box', 'benches': 'bench', 'branches': 'branch', 'dresses': 'dress',
                 'arches': 'arch', 'dishes': 'dish', 'brushes': 'brush', 'watches': 'watch', 'lenses': 'lens'}
    if w in irregular:
        return irregular[w]
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us', 'is')):
        return w[:-1]
    return w


def alias_heads(category) -> set[str]:
    heads = set()
    for alias in re.split(r'[,;/]', str(category or '').lower()):
        words = [lemma(w) for w in re.findall(r'[a-z]+', alias)]
        words = [w for w in words if w not in STOP]
        if words:
            heads.add(words[-1])
    return heads


# Activity tags ride along with person aliases ("man,person,run,runner"); they
# do not indicate a second, non-person entity inside the mask.
ACTIVITY = {'run', 'walk', 'hike', 'ride', 'stand', 'sit', 'play', 'dance', 'swim', 'jog', 'climb', 'surf', 'ski',
            'skate', 'race', 'march', 'parade', 'wear', 'pose', 'smile', 'shop', 'work', 'sport', 'athlete'}


def mixed_person_category(category) -> bool:
    heads = alias_heads(category) - ACTIVITY
    return bool(heads & PERSON) and bool(heads - PERSON)


def _decode_quarter(obj):
    from pycocotools import mask as coco
    rle = obj['rle']
    if isinstance(rle.get('counts'), list):
        rle = coco.frPyObjects(rle, *rle['size'])
    return np.ascontiguousarray(coco.decode(rle)[::4, ::4]).astype(bool)


def image_features(candidate: dict) -> tuple[dict, dict]:
    """Per-object score/area/competitor/fragment features; quarter-resolution overlaps."""
    masks, info = {}, {}
    for obj in candidate['objects']:
        oid = str(obj['object_id'])
        mask = _decode_quarter(obj)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        masks[oid] = mask
        info[oid] = {'obj': obj, 'area': int(mask.sum()),
                     'bbox': (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                     'heads': alias_heads(obj.get('category')),
                     'score': float((obj.get('metadata') or {}).get('score') or 0.0)}
    if not masks:
        return {}, {}
    height, width = next(iter(masks.values())).shape

    def overlap(a, b):
        A, B = info[a]['bbox'], info[b]['bbox']
        if A[2] <= B[0] or B[2] <= A[0] or A[3] <= B[1] or B[3] <= A[1]:
            return 0
        return int((masks[a] & masks[b]).sum())

    for tid, t in info.items():
        competitors, parent = [], False
        for oid, o in info.items():
            if oid == tid or not (t['heads'] & o['heads']):
                continue
            inter = overlap(tid, oid)
            if inter / t['area'] >= .8 and o['area'] > t['area']:
                parent = True
            if inter / t['area'] < .3 and inter / o['area'] < .3:
                competitors.append(oid)
        distinct = []
        for c in sorted(competitors, key=lambda c: -info[c]['area']):
            if any(overlap(c, d) / max(1, min(info[c]['area'], info[d]['area'])) > .5 for d in distinct):
                continue
            distinct.append(c)
        t.update(n_competitors=len(distinct),
                 n_similar_competitors=sum(.25 <= info[c]['area'] / t['area'] <= 4 for c in distinct),
                 same_category_parent=parent, area_fraction=t['area'] / float(height * width))
    return info, masks


def eligible(feature: dict, operation: str, affordance: dict, *, image_size, limits=DEFAULTS) -> bool:
    from .geometry import load_mask, mask_geometry, operation_quality_gate
    max_area = min(limits['max_area'], (limits.get('max_area_by_operation') or {}).get(operation, 1.0))
    if (feature['score'] < limits['min_score'] or not limits['min_area'] <= feature['area_fraction'] <= max_area
            or feature['same_category_parent'] or feature['n_similar_competitors'] > limits['max_similar_competitors']
            or mixed_person_category(feature['obj'].get('category'))):
        return False
    if (affordance.get(str(feature['obj'].get('category'))) or {}).get(AFFORD_KEY[operation]) != 'natural':
        return False
    return bool(operation_quality_gate(mask_geometry(load_mask(feature['obj'], image_size)), operation)['eligible'])


def priority(feature: dict) -> float:
    """Confident, sparsely repeated masks first; large stuff regions are not rewarded."""
    return feature['score'] - .08 * feature['n_similar_competitors'] - (.15 if feature['area_fraction'] > .15 else 0)


def _evidence(info, chosen, dependents=None):
    keys = ('score', 'area_fraction', 'n_competitors', 'n_similar_competitors', 'same_category_parent')
    out = {c: {k: info[c][k] for k in keys} for c in chosen}
    for c in chosen:
        if dependents is not None and c in dependents:
            out[c]['support_dependents'] = dependents[c]
    return out


class _Image:
    """Per-image features plus cached eligibility, including support dependents."""
    def __init__(self, candidate, affordance, limits, support_check=True):
        self.candidate, self.affordance, self.limits = candidate, affordance, limits
        self.info, self.masks = image_features(candidate)
        self.size = tuple(reversed(candidate['objects'][0]['rle']['size'])) if candidate['objects'] else (0, 0)
        self.order = sorted(self.info, key=lambda k: -priority(self.info[k]))
        self.support_check = support_check
        self.worn_check = bool(limits.get('worn_check'))
        self._ok, self.dependents, self._ground, self.wearers = {}, {}, None, {}

    def ok(self, oid, operation):
        key = (oid, operation)
        if key not in self._ok:
            good = eligible(self.info[oid], operation, self.affordance, image_size=self.size, limits=self.limits)
            if good and self.support_check and operation in ('remove', 'replace'):
                # Removing/replacing a supporter must not leave a segmented neighbour floating.
                from . import support_graph
                if oid not in self.dependents:
                    if self._ground is None:
                        self._ground = support_graph.ground_mask(self.info, self.masks)
                    self.dependents[oid] = support_graph.dependents(self.info, self.masks, oid, ground=self._ground)
                good = not self.dependents[oid]
            if good and self.worn_check and operation == 'remove':
                # Removing a worn shoe/helmet/glove/garment exposes a body the generator renders poorly.
                from . import support_graph
                if oid not in self.wearers:
                    self.wearers[oid] = support_graph.worn_on_body(self.info, self.masks, oid)
                good = not self.wearers[oid]
            if good and self.limits.get('covering_check') and operation == 'remove':
                # Removing snow/moss/a tarp lying on an object must invent that object's hidden top.
                from . import support_graph
                good = not support_graph.covered_objects(self.info, self.masks, oid)
            if good and self.limits.get('surface_recolor_check') and operation == 'attribute':
                # Recolouring sky/water/ground floods a ragged mask against the scene light.
                from . import support_graph
                good = not support_graph.scene_surface(self.info[oid])
            self._ok[key] = good
        return self._ok[key]


def select_twin(candidate: dict, affordance: dict, deficit: Counter, *, members: int = 3,
                limits=DEFAULTS, support_check=True, saturated_ok=True,
                members_by_operation: dict | None = None, image: _Image | None = None) -> dict | None:
    """One operation with `members` mutually separate eligible masks (pairwise area ratio <= 3).

    Operations are tried in decreasing ``deficit``; with ``saturated_ok=False``
    an operation whose deficit is already <= 0 is never chosen.
    ``members_by_operation`` over-selects a low-yield operation (e.g. {'add': 4}).
    ``image``: features already computed for this candidate with the same settings (``menu``).
    """
    im = image or _Image(candidate, affordance, limits, support_check)
    info, masks = im.info, im.masks
    if not info:
        return None

    from scipy.ndimage import binary_dilation

    def separate(a, b, operation):
        """Mirror twin_support_preflight: the two WRITABLE supports (each mask grown
        by the operation's support radius, editor._support_radius) must keep a gap
        of max(16px, 10% of the smaller bbox short side) at full resolution (with
        50% margin); quarter-resolution masks, so /4."""
        if int((masks[a] & masks[b]).sum()):
            return False
        sides = {c: min(info[c]['bbox'][2] - info[c]['bbox'][0], info[c]['bbox'][3] - info[c]['bbox'][1]) * 4 for c in (a, b)}
        short = min(sides.values())
        growth = sum(_support_growth(operation, sides[c]) for c in (a, b))
        radius = max(1, int(round((growth + 1.5 * max(16.0, 0.1 * short)) / 4)))
        x0, y0 = max(0, min(info[a]['bbox'][0], info[b]['bbox'][0]) - radius), max(0, min(info[a]['bbox'][1], info[b]['bbox'][1]) - radius)
        x1, y1 = max(info[a]['bbox'][2], info[b]['bbox'][2]) + radius, max(info[a]['bbox'][3], info[b]['bbox'][3]) + radius
        grown = binary_dilation(masks[a][y0:y1, x0:x1], iterations=radius)
        return not bool((grown & masks[b][y0:y1, x0:x1]).any())

    for operation in sorted(OPERATIONS, key=lambda op: (-deficit[op], op)):
        if not saturated_ok and deficit[operation] <= 0:
            continue
        pool = [oid for oid in im.order if im.ok(oid, operation)]
        size = int((members_by_operation or {}).get(operation, members))
        for want in list(range(size, 1, -1)):
            chosen = []
            for oid in pool:
                trial = chosen + [oid]
                areas = [info[c]['area'] for c in trial]
                if max(areas) > limits['twin_max_area_ratio'] * min(areas):
                    continue
                # Add twins are compared on the RESOLVED attachment sites (blind
                # protocol area ratio <= 3), so pair anchors of the same kind and
                # similar size; heterogeneous anchors receive incomparable items.
                if operation == 'add' and chosen and (
                        not all(info[oid]['heads'] & info[c]['heads'] for c in chosen)
                        or max(areas) > limits.get('add_twin_area_ratio', 2.0) * min(areas)):
                    continue
                if all(separate(oid, c, operation) for c in chosen):
                    chosen = trial
                if len(chosen) == want:
                    break
            if len(chosen) == want:
                objects = [info[c]['obj'] for c in chosen]
                deficit[operation] -= 2
                parts = sum(_is_part(o) for o in objects)
                return {'candidate': candidate, 'objects': objects, 'operation': operation,
                        'scope_level': 'part_on_different_hosts' if parts == len(objects) else 'object',
                        'gate': {'member_count': len(objects), 'pairs': [], 'eligible': True, 'geometry_exact': True,
                                 'version': VERSION},
                        **({'twin_keep': 2} if len(objects) > 2 else {}),
                        'evidence_selection': {'version': VERSION, 'limits': dict(limits), 'support_check': support_check,
                                               'members': _evidence(info, chosen, im.dependents)}}
    return None


def _support_growth(operation: str, short_side: float) -> float:
    """Full-resolution support radius of editor._support_radius for this operation."""
    if operation == 'add':
        return float(np.clip(round(short_side * .28), 12, 64))
    if operation == 'replace':
        return float(np.clip(round(short_side * .24), 12, 56))
    if operation == 'remove':
        return float(np.clip(round(short_side * .14), 8, 40))
    return 0.0


def _is_part(obj) -> bool:
    from .group_geometry import candidate_granularity
    return candidate_granularity(obj) == 'part'


def select_multi(candidate: dict, affordance: dict, deficit: Counter, *, targets: int = 4,
                 limits=DEFAULTS, support_check=True, saturated_ok=True, image: _Image | None = None) -> dict | None:
    """Up to four distinct operations, most-needed first; at least two targets.

    Operations with a positive deficit are filled first; saturated operations
    are only used to reach the two-target minimum (or freely if ``saturated_ok``).
    """
    im = image or _Image(candidate, affordance, limits, support_check)
    info, masks = im.info, im.masks
    if not info:
        return None
    chosen, operations = [], []

    def take(operation):
        for oid in im.order:
            if oid in chosen:
                continue
            if any(int((masks[oid] & masks[c]).sum()) / max(1, min(info[oid]['area'], info[c]['area'])) >= limits['separation']
                   for c in chosen):
                continue
            if im.ok(oid, operation):
                chosen.append(oid)
                operations.append(operation)
                return True
        return False

    ranked = sorted(OPERATIONS, key=lambda op: (-deficit[op], op))
    for operation in ranked:
        if len(chosen) == targets:
            break
        if deficit[operation] > 0 or saturated_ok:
            take(operation)
    if len(chosen) < 2 and not saturated_ok:
        for operation in ranked:
            if len(chosen) >= 2:
                break
            if operation not in operations:
                take(operation)
    if len(chosen) < 2:
        return None
    for operation in operations:
        deficit[operation] -= 1
    width, height = im.size
    return {'candidate': candidate, 'objects': [info[c]['obj'] for c in chosen], 'operations': operations,
            'adjacent_pairs': [], 'image_size': [width, height],
            'evidence_selection': {'version': VERSION, 'limits': dict(limits), 'support_check': support_check,
                                   'members': _evidence(info, chosen, im.dependents)}}


def twin_allocation(limit: int, deficit: Counter) -> dict:
    """Twin groups per operation for this wave: ``limit`` split in proportion to the remaining selection
    units (deficits count two targets per group), never more than an operation needs."""
    need = {op: math.ceil(max(0.0, deficit[op]) / 2) for op in OPERATIONS}
    total = sum(need.values())
    if total <= limit:
        return need
    share = {op: limit * need[op] / total for op in OPERATIONS}
    out = {op: int(share[op]) for op in OPERATIONS}
    for op in sorted(OPERATIONS, key=lambda op: (-(share[op] - out[op]), op))[:limit - sum(out.values())]:
        out[op] += 1
    return out


def selection_settings(*, twin_members=3, twin_members_add=None, worn_check=False, covering_check=False,
                       no_surface_recolor=False, support_check=True, skip_grayscale=False) -> dict:
    """Selection settings as used by scripts/select_evidence_groups.py; menus record them."""
    extra = {k: True for k, v in (('worn_check', worn_check), ('covering_check', covering_check),
                                  ('surface_recolor_check', no_surface_recolor)) if v}
    return {'version': VERSION, 'members': int(twin_members),
            'members_by_operation': {'add': int(twin_members_add)} if twin_members_add else None,
            'limits': dict(DEFAULTS, **extra), 'support_check': bool(support_check),
            'skip_grayscale': bool(skip_grayscale)}


def settings_fingerprint(settings: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def image_menu(candidate: dict, affordance: dict, settings: dict) -> dict:
    """``menu`` with ``selection_settings``; black-and-white sources are skipped when the settings say so."""
    if settings['skip_grayscale'] and is_grayscale(candidate['image_path']):
        return {'skip': 'grayscale_skipped'}
    return menu(candidate, affordance, members=settings['members'], limits=settings['limits'],
                support_check=settings['support_check'], members_by_operation=settings['members_by_operation'])


def menu(candidate: dict, affordance: dict, *, members: int = 3, limits=DEFAULTS, support_check=True,
         members_by_operation: dict | None = None) -> dict:
    """What one image can serve, for the production selection cache (``scripts/select_evidence_groups.py
    --menu-dir``): the twin member count per operation when only that operation is needed (0: none), and
    the operations of a multi group when all four are needed.

    The twin entry is exact: ``select_twin`` tries operations one at a time, so with any deficits it
    returns the first needed operation whose entry is >= 2. The multi entry is a prediction (other
    deficit orders can pick other masks); rows are always built by ``select_twin`` / ``select_multi``.
    """
    im = _Image(candidate, affordance, limits, support_check)
    twin = {}
    for operation in OPERATIONS:
        row = select_twin(candidate, affordance, Counter({operation: 1}), members=members, limits=limits,
                          support_check=support_check, saturated_ok=False,
                          members_by_operation=members_by_operation, image=im)
        twin[operation] = len(row['objects']) if row else 0
    row = select_multi(candidate, affordance, Counter(dict.fromkeys(OPERATIONS, 1)), limits=limits,
                       support_check=support_check, saturated_ok=False, image=im)
    return {'twin': twin, 'multi': list(row['operations']) if row else []}


def affordance_requests(categories: dict[str, str]) -> list[tuple[str, str]]:
    return [(category, AFFORD.format(category=category, description=(description or '')[:300]))
            for category, description in categories.items()]


def load_affordance_cache(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def update_affordance_cache(path: Path, categories: dict[str, str], selector) -> dict:
    """Judge each NEW category string once (text-only, no reasoning) and persist."""
    cache = load_affordance_cache(path)
    pending = {c: d for c, d in categories.items() if c not in cache}
    requests = affordance_requests(pending)
    if requests:
        responses = selector.generate_multimage([([], prompt) for _, prompt in requests],
                                                schemas=[AFFORD_SCHEMA] * len(requests),
                                                token_limits=[60] * len(requests),
                                                thinking=[False] * len(requests))
        for (category, _), response in zip(requests, responses):
            if isinstance(response, dict) and all(response.get(k) in ('natural', 'conditional', 'unnatural')
                                                  for k in AFFORD_KEY.values()):
                cache[category] = {k: response[k] for k in AFFORD_KEY.values()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + '.tmp')
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1, sort_keys=True))
        tmp.replace(path)
    return cache


def is_grayscale(image_path, *, tolerance: int = 6, share: float = .995) -> bool:
    """Black-and-white photo (stored as RGB): nearly every pixel has equal channels.

    In such photos the editor paints new content in colour (full run 4: a brown beam in a grey
    gondola photo) and colour edits are meaningless. About 0.5% of the source pool.
    """
    import numpy as np
    from PIL import Image
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        image.thumbnail((256, 256))
        a = np.asarray(image).astype(np.int16)
    return float(((a.max(axis=2) - a.min(axis=2)) <= tolerance).mean()) >= share

