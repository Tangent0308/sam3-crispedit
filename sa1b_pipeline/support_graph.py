"""Support/attachment dependents of a target, from SA-1B dense masks only.

Removing (or replacing) a target must not leave another visible object floating:
cups on a removed table, a sign on a removed pole, plants in a removed planter,
a rope hanging from a removed helicopter. The companion annotation already
segments those neighbours, so this is a cheap geometric test on
quarter-resolution masks, without model calls:

* rests_on   - much of O's bottom outline sits directly on the target;
* hangs_from - much of O's top outline hangs directly from the target;
* embedded   - O is mostly enclosed by the target (a sticker on a box);
* attached   - the target is pole-like and O touches it.

An object whose own bottom outline rests on the ground (or the photo's bottom
edge) has its own support and is exempt, except for hangs_from. Background
"stuff" (sky, walls, ground, ...) is never a dependent, and a part or whole of
the target is not one either (it is removed or kept together with it).
Image-plane adjacency can also come from occlusion, so a flag means "do not
remove/replace this target", never "the neighbour is physically attached".
"""
from __future__ import annotations

import numpy as np

VERSION = 'support-graph-5-worn-covering-surface'
STUFF = {'sky', 'cloud', 'wall', 'building', 'facade', 'ground', 'floor', 'road', 'street', 'grass', 'lawn', 'field',
         'water', 'sea', 'ocean', 'river', 'lake', 'mountain', 'hill', 'ceiling', 'sidewalk', 'pavement', 'path',
         'pathway', 'walkway', 'sand', 'beach', 'snow', 'dirt', 'soil', 'earth', 'terrain', 'forest', 'tree',
         'foliage', 'vegetation', 'bush', 'hedge', 'shrub', 'background', 'shadow', 'reflection', 'light', 'sunlight',
         'roof', 'asphalt', 'concrete', 'plaza', 'square', 'courtyard', 'land', 'landscape', 'rock', 'cliff',
         'riverbank', 'shore', 'horizon', 'city', 'cityscape', 'skyline', 'crowd', 'people', 'carpet', 'rug', 'tile',
         'brick', 'stone', 'gravel', 'mud'}
GROUND = {'ground', 'floor', 'road', 'street', 'grass', 'lawn', 'field', 'sidewalk', 'pavement', 'path', 'pathway',
          'walkway', 'sand', 'beach', 'snow', 'dirt', 'soil', 'earth', 'terrain', 'asphalt', 'plaza', 'square',
          'courtyard', 'land', 'carpet', 'rug', 'tile', 'gravel', 'mud', 'water', 'sea', 'river', 'lake'}
BODY = {'hand', 'arm', 'finger', 'thumb', 'leg', 'foot', 'knee', 'toe', 'head', 'face', 'hair', 'ear', 'eye', 'nose',
        'mouth', 'lip', 'neck', 'shoulder', 'elbow', 'wrist', 'torso', 'body', 'skin', 'beard'}
ARCHITECTURE = {'window', 'door', 'doorway', 'column', 'pillar', 'arch', 'archway', 'balcony', 'railing', 'balustrade',
                'baluster', 'stair', 'staircase', 'step', 'bridge', 'tower', 'chimney', 'gable', 'dome', 'shutter',
                'gate', 'fence', 'wall', 'building', 'house', 'roof', 'facade', 'storefront', 'porch', 'frame', 'pane',
                'glass', 'awning', 'canopy', 'ledge', 'sill', 'cornice', 'spire', 'steeple'}
# Never a dependent of another object: people, what they wear, vehicles and
# supports themselves. Adjacency with them is occlusion, not attachment.
NOT_DEPENDENT = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'people', 'player', 'runner', 'rider',
                 'pedestrian', 'tourist', 'worker', 'spectator', 'crowd', 'shirt', 't-shirt', 'top', 'blouse', 'jeans',
                 'pants', 'trousers', 'shorts', 'skirt', 'dress', 'jacket', 'coat', 'sweater', 'hoodie', 'vest', 'cuff',
                 'sleeve', 'sock', 'shoe', 'boot', 'sandal', 'slipper', 'glove', 'scarf', 'uniform', 'clothing',
                 'garment', 'fabric', 'denim', 'car', 'truck', 'bus', 'van', 'vehicle', 'bicycle', 'bike', 'motorcycle',
                 'boat', 'ship', 'train', 'tram', 'airplane', 'trunk', 'tree', 'column', 'pillar', 'pole', 'post',
                 'mast', 'lamppost', 'signpost', 'flagpole', 'animal', 'dog', 'cat', 'horse', 'bird', 'cow'}
# Only these can carry loads or attachments; for any other target, image-plane
# adjacency is occlusion and no dependent is inferred.
SUPPORTERS = {'table', 'desk', 'counter', 'bench', 'chair', 'sofa', 'couch', 'stool', 'bed', 'shelf', 'cabinet',
              'cart', 'trolley', 'tray', 'plate', 'platter', 'bowl', 'basket', 'crate', 'box', 'pallet', 'stand',
              'rack', 'easel', 'planter', 'pot', 'vase', 'pedestal', 'plinth', 'podium', 'tripod', 'pole', 'post',
              'lamppost', 'signpost', 'flagpole', 'mast', 'bracket', 'hook', 'rope', 'line', 'wire', 'cable', 'branch',
              'bicycle', 'bike', 'motorcycle', 'scooter', 'horse', 'boat', 'swing', 'hammock', 'wagon', 'carriage',
              'stall', 'kiosk', 'trellis', 'railing', 'fence', 'ledge', 'sill', 'windowsill', 'mantel', 'cupboard'}
# Supporters that people sit on or ride: then people/clothing can be dependents.
SEATS = {'bench', 'chair', 'sofa', 'couch', 'stool', 'bed', 'bicycle', 'bike', 'motorcycle', 'scooter', 'horse',
         'boat', 'swing', 'hammock', 'wagon', 'carriage'}
PEOPLE = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'people', 'player', 'runner', 'rider',
          'pedestrian', 'tourist', 'worker', 'spectator', 'shirt', 't-shirt', 'top', 'blouse', 'jeans', 'pants',
          'trousers', 'shorts', 'skirt', 'dress', 'jacket', 'coat', 'sweater', 'hoodie', 'vest', 'sock', 'shoe',
          'boot', 'sandal', 'slipper', 'uniform', 'clothing', 'garment', 'leg', 'foot'}
POLE_LIKE = {'pole', 'post', 'stand', 'bracket', 'mast', 'rod', 'stake', 'rope', 'chain', 'wire', 'cable', 'string',
             'hook', 'holder', 'tripod', 'easel', 'hanger', 'rack', 'branch', 'stem', 'stick', 'beam', 'lamppost',
             'signpost', 'flagpole', 'trellis'}
# Worn items whose removal exposes skin or body shape (head, face, feet, hands,
# main garments): the generator renders the revealed body poorly (a helmet left a
# rubber-like bald head, a sneaker an undefined foot). Outerwear and carried or
# strapped items (jacket, backpack, bag, race bib) reveal clothing and stay allowed.
BODY_WEAR = {'helmet', 'hat', 'cap', 'beanie', 'hood', 'headscarf', 'hijab', 'turban', 'crown', 'tiara', 'wig',
             'headband', 'headphone', 'headphones', 'earmuff', 'earmuffs', 'mask', 'balaclava', 'glasses',
             'sunglasses', 'goggles', 'spectacles', 'eyeglasses', 'shoe', 'sneaker', 'boot', 'sandal', 'slipper',
             'sock', 'stocking', 'glove', 'mitten', 'gauntlet', 'shirt', 't-shirt', 'top', 'blouse', 'dress',
             'jeans', 'pants', 'trousers', 'shorts', 'skirt', 'leggings', 'tights', 'swimsuit', 'bikini'}
WEARER = PEOPLE | BODY | {'human', 'athlete', 'soldier', 'officer', 'student', 'graduate', 'performer', 'dancer',
                          'skier', 'skater', 'surfer', 'cyclist', 'biker', 'driver', 'vendor', 'guard', 'police',
                          'policeman', 'baby', 'toddler', 'lady', 'gentleman', 'adult', 'teenager', 'army'}
# A covering layer lying on an object: removing it must invent the hidden top of
# that object (snow cap removal also erased a stone lantern's roof).
COVERINGS = {'snow', 'ice', 'frost', 'moss', 'tarp', 'tarpaulin', 'blanket', 'tablecloth', 'cover', 'covering'}
# Scene-scale surfaces: a colour edit floods a ragged, holed mask and ignores the
# scene light (sky halos, flat painted pavement or water; 5 of 5 reviewed were flawed).
SCENE_SURFACES = {'sky', 'cloud', 'clouds', 'water', 'sea', 'ocean', 'river', 'lake', 'pond', 'ground', 'floor',
                  'road', 'street', 'sidewalk', 'pavement', 'asphalt', 'concrete', 'path', 'pathway', 'walkway',
                  'sand', 'beach', 'dirt', 'soil', 'earth', 'terrain', 'mud', 'gravel'}
LIMITS = {'contact_px': 3, 'rests_on': .35, 'hangs_from': .35, 'embedded': .6, 'attached_min_px': 4,
          'min_area_px': 24, 'own_support': .2, 'max_area_ratio': 4.0, 'tiny_ratio': .03, 'tiny_image_fraction': .0005,
          # 'ground': only ground/photo edge counts as a neighbour's own support.
          # 'ground+objects': any other object below it also counts (fewer flags, lower recall).
          'own_support_mode': 'ground'}


def _bottom_outline(mask):
    below = np.zeros_like(mask)
    below[:-1] = mask[1:]
    return mask & ~below


def _top_outline(mask):
    above = np.zeros_like(mask)
    above[1:] = mask[:-1]
    return mask & ~above


def _outline(mask):
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask)


def _near_below(target, k):
    """Pixels that have a ``target`` pixel within 1..k rows directly below them."""
    near = np.zeros_like(target)
    for d in range(1, k + 1):
        near[:-d] |= target[d:]
    return near


def _near_above(target, k):
    near = np.zeros_like(target)
    for d in range(1, k + 1):
        near[d:] |= target[:-d]
    return near


def ground_mask(info: dict, masks: dict) -> np.ndarray:
    shape = next(iter(masks.values())).shape
    ground = np.zeros(shape, dtype=bool)
    for oid, o_info in info.items():
        if o_info['heads'] & GROUND:
            ground |= masks[oid]
    ground[-1:, :] = True  # the photo's bottom edge also supports
    return ground


def dependents(info: dict, masks: dict, target_id: str, *, ground=None, limits=LIMITS) -> list[dict]:
    """Objects that would lose support or attachment if ``target_id`` disappeared.

    ``info``/``masks`` are the quarter-resolution per-object features used by
    evidence selection (keys: obj, area, bbox, heads). Returns a list of
    {object_id, category, relation, fraction, area_ratio}; an empty list means no
    evidence of a dependent, not proof of physical plausibility.
    """
    from scipy.ndimage import binary_dilation
    t = masks[target_id]
    t_info = info[target_id]
    k = limits['contact_px']
    x0, y0, x1, y1 = t_info['bbox']
    pad = 2 * k + 2
    t_near = binary_dilation(t, iterations=k)
    below_t = _near_below(t, k)   # pixel has the target just below it
    above_t = _near_above(t, k)   # pixel has the target just above it
    if not (t_info['heads'] & SUPPORTERS):
        return []
    excluded = (STUFF | BODY | NOT_DEPENDENT) - (PEOPLE if t_info['heads'] & SEATS else set())
    pole_like = bool(t_info['heads'] & POLE_LIKE) and not (t_info['heads'] & STUFF)
    t_architecture = bool(t_info['heads'] & ARCHITECTURE)
    image_px = t.size
    others = None  # union of every other non-stuff object, for the own-support exemption
    out = []
    for oid, o_info in info.items():
        if oid == target_id or o_info['area'] < limits['min_area_px'] or o_info['heads'] & excluded:
            continue
        if o_info['heads'] & ARCHITECTURE and not t_architecture:
            continue  # a window cannot rest on a palm tree; adjacency is occlusion
        if (o_info['area'] < limits['tiny_ratio'] * t_info['area']
                and o_info['area'] < limits['tiny_image_fraction'] * image_px):
            continue
        ox0, oy0, ox1, oy1 = o_info['bbox']
        if ox1 < x0 - pad or ox0 > x1 + pad or oy1 < y0 - pad or oy0 > y1 + pad:
            continue
        o = masks[oid]
        inter = int((o & t).sum())
        if inter / o_info['area'] >= .5 or inter / t_info['area'] >= .5:
            continue  # part/whole of the target: removed or kept together with it
        if not (o & t_near).any():
            continue
        own = o & ~t
        bottom, top, outline = _bottom_outline(own), _top_outline(own), _outline(own)
        n_bottom, n_top, n_outline = int(bottom.sum()), int(top.sum()), int(outline.sum())
        rests = float((bottom & below_t).sum() / n_bottom) if n_bottom else 0.0
        hangs = float((top & above_t).sum() / n_top) if n_top else 0.0
        enclosed = float((outline & t_near).sum() / n_outline) if n_outline else 0.0
        not_huge = o_info['area'] <= limits['max_area_ratio'] * t_info['area']
        relation, fraction = None, 0.0
        if hangs >= limits['hangs_from'] and not_huge and hangs > rests:
            relation, fraction = 'hangs_from', hangs
        elif rests >= limits['rests_on'] and not_huge:
            relation, fraction = 'rests_on', rests
        elif enclosed >= limits['embedded'] and o_info['area'] <= t_info['area']:
            relation, fraction = 'embedded', enclosed
        elif pole_like and int((outline & t_near).sum()) >= limits['attached_min_px']:
            relation, fraction = 'attached', float((outline & t_near).sum() / max(1, n_outline))
        if relation and relation != 'embedded' and n_bottom:
            if ground is None:
                ground = ground_mask(info, masks)
            support = ground
            if limits['own_support_mode'] == 'ground+objects':
                if others is None:
                    others = np.zeros_like(t)
                    for xid, x_info in info.items():
                        if xid != target_id and not (x_info['heads'] & STUFF):
                            others |= masks[xid]
                # Another object (not the target) directly below also supports it.
                support = ground | (others & ~o & ~t)
            own = float((bottom & _near_below(support, k)).sum() / n_bottom)
            if own >= limits['own_support']:
                relation = None
        if relation:
            out.append({'object_id': oid, 'category': o_info['obj'].get('category'), 'relation': relation,
                        'fraction': round(fraction, 3), 'area_ratio': round(o_info['area'] / t_info['area'], 3)})
    return out


def worn_on_body(info, masks, target_id, inside=.5) -> list[str]:
    """Wearers (people/body parts) of a body-wear target: its removal would expose skin or body shape.

    A body-wear category (BODY_WEAR) counts as worn when at least ``inside`` of its
    mask lies within one person/body mask, or it touches one (a shoe on a foot).
    """
    target = info[target_id]
    if not (target['heads'] & BODY_WEAR) or target['heads'] & {'backpack', 'bag'}:
        return []
    from scipy.ndimage import binary_dilation
    mask = masks[target_id]
    area = max(1, int(mask.sum()))
    near = binary_dilation(mask, iterations=LIMITS['contact_px'])
    wearers = []
    for oid, other in info.items():
        if oid == target_id or not (other['heads'] & WEARER) or other['heads'] & BODY_WEAR:
            continue
        overlap = int((mask & masks[oid]).sum())
        if overlap >= inside * area or (near & masks[oid]).sum() >= LIMITS['attached_min_px']:
            wearers.append(str(oid))
    return wearers


def covered_objects(info, masks, target_id, limits=LIMITS, ground=None) -> list[str]:
    """What a covering-layer target (COVERINGS) lies on, unless it lies on the ground.

    Returns the non-ground objects below it, or ['unsegmented'] when its bottom
    outline rests neither on ground/water nor on a segmented object (the covered
    object is often not segmented: a snow cap on a stone lantern). [] means removal
    reveals ground (a snow patch on a lawn) or the target is no covering.
    """
    target = info[target_id]
    if not (target['heads'] & COVERINGS):
        return []
    mask = masks[target_id]
    bottom = _bottom_outline(mask)
    n_bottom = int(bottom.sum())
    if not n_bottom:
        return []
    k = limits['contact_px']
    if ground is None:
        ground = ground_mask(info, masks)
    if (bottom & _near_below(ground & ~mask, k)).sum() >= limits['rests_on'] * n_bottom:
        return []
    covered = []
    for oid, other in info.items():
        if oid == target_id or other['heads'] & (GROUND | STUFF | COVERINGS):
            continue
        if (bottom & _near_below(masks[oid] & ~mask, k)).sum() >= limits['rests_on'] * n_bottom:
            covered.append(str(oid))
    return covered or ['unsegmented']


def scene_surface(target_info) -> bool:
    """A sky/water/ground-type region, not a bounded object."""
    return bool(target_info['heads'] & SCENE_SURFACES)
