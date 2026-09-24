"""Plan and visually review edits against immutable dataset masks, without source SAM.

Both MLLM rounds share one vLLM engine. Reports retain every input and attempt.
The mask is an allowed edit region; a local instruction need not change it all.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.generate_samtok_plan import (
    TYPE_ACTION_PATTERNS,
    REPLACEMENT_GENERIC_WORDS,
    mask_geometry_hint,
    normalized_area,
    parse_json_object,
    replacement_text_from_instruction,
    same_replacement_category,
)
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
from synthesis_pipeline.reference_binding import bind_reference
from utils.context_edit import protected_neighbors
import utils.vlm_utils as vlm

VERSION = 'dataset_mask_v6_28_semantic_surface_and_assembly_guard'
RULES = {
    'add': 'Choose one concrete addition appropriate to this scene, completely absent from the clean source photo, small relative to its host but clearly visible and judgeable at full-image resolution. Novelty is visual rather than merely lexical: a differently named subtype is not new when it would be indistinguishable from something already visible at full-image scale. Search the entire source before accepting it; adding another instance, another visually equivalent item, or increasing the quantity of an existing item is not a valid add task. Reject large, giant, oversized, microscopic, single-pixel-scale, or solitary crumb/particle additions. When the selected host is already tiny in the full photo, do not propose an even smaller surface detail or attached object on it. Anchor the addition to an actually visible surface or attachment on the selected instance. The visible body of the new item must fit substantially inside the selected pixels: do not place a freestanding object on top of the target, perch an object on a thin selected structure, span holes between sparse slats or wires, or attach an item that needs to protrude into excluded pixels. Prefer a surface-applied or closely overlapping addition when the mask follows the host silhouette. The item, attachment method, and placement must form a common physically credible relationship; do not balance a loose item on a host, or place one against fur/skin without a visible, ordinary attachment method, merely because pixels are available. Give a clear local placement relation that identifies one exact site; an indefinite member of many repeated local parts is not exact even with a broad quadrant. Never leave the editor to choose among repeated parts. A broad wall, facade, ground plane, forest, foliage, or building owner alone is not a placement site. Do not invent a supporting person/hand, an unseen garment feature, or a front-side body site on a rear view. Avoid anatomical left/right; use an unambiguous visible action or image-relative locator. Let the scene determine the new item; vary the choice across scenes.',
    'remove': 'Remove the selected visible object or selected coherent group. Include all named members, but never name a larger group than selected. Check thin extensions and objects physically supported by, worn by, carried by, or attached to it: if a required dependent object is outside the mask and would be left suspended, reject. Independent background supports and another person\'s hand are not dependent objects merely because they touch the target. Reject removal of a large permanent scene structure, a non-interchangeable section of a continuous built structure, or a broad background surface when it would require synthesizing a major portion of the scene; a small self-contained architectural component remains eligible. Also reject a collection of integrated mechanical components whose removal would leave a visibly incomplete vehicle or machine; a genuinely detachable accessory can remain eligible. A visibly detached portion resting independently is not a continuous-owner conflict merely because it was cut from a larger item. Ask only for removal; background reconstruction is implicit.',
    'replace': 'Replace the selected old instance with a concrete, visibly different, scene-plausible instance of compatible apparent scale, footprint, pose and support. Do not describe the replacement as larger, wider, taller, multi-level, multi-section, elongated, upright, standing, oversized, or expanded when the selected extent cannot contain that change. Preserve a compact crouching, seated, kneeling or lying footprint rather than expanding it to a standing pose. The replacement must change object category or create a clearly recognizable new identity/form at full-image resolution; a nearly indistinguishable sibling category is not useful. A pure color, material, clothing, pattern, style, text, advertised content, emblem, decoration, or depicted artwork change to the same physical carrier belongs to attribute, not replace. A shirt remains a shirt, a coat/rain jacket remains outerwear, and a bottle remains a bottle when only material, color, print, or named subtype changes. Rewording the same object with an alias does not make a replacement. A named subtype is useful only when it is visibly different and does not already describe the source. People and animals ARE valid replaceable instances. A person can be replaced by another person only when the new identity is visibly distinguished by a physical characteristic, demographic category, or genuinely different role/form; changing only uniform or clothing color is attribute, not replace. A homogeneous group of the same category can be replaced together, but reject a heterogeneous host-plus-occupant, container-plus-content, or multi-object mini-scene; never replace a mixed assembly with a newly invented mixed assembly. Reject replacement of a large permanent scene structure, a non-interchangeable section of a continuous built structure, or a broad background surface when it would remake a major portion of the scene; a small self-contained architectural component remains eligible. All VISIBLE parts of the old target must be selected; hidden parts behind occluders or beyond the photo need no mask. Check held/carried accessories outside the mask, riders/passengers on selected transport, externally mounted equipment, and contents enclosed by a selected container: reject if the new instance would leave unsupported content or an impossible interaction. A visibly detached portion resting independently can be replaced as one object. Preserve the actual foreground occluder. Do not invent extra supporting objects.',
    'attribute': 'Change one visually evidenced, scene-plausible property of a specific surface inside the mask. Identify that surface in the instruction, even if the mask is the entire owner: never ask to change the color of a person, animal, or other multi-surface owner as a whole. Reject an indistinct, blurred or semantically unresolved background patch: uncertainty between a person, barrier, backdrop or generic surface is not a meaningful attribute target. Do not assign ordinary human or animal skin an implausible bright synthetic color merely to maximize contrast. The other selected surfaces remain unchanged by default. Keep identity, silhouette, markings, pattern, weathering and texture unless the requested property itself is pattern or texture. Use a target value with obvious visual contrast from the source; subtle near-black to dark-color changes are not useful. For a color-only change on a multicolored or patterned surface, name exactly one visible constituent such as its base/background or a specific existing marking; never recolor the whole patterned surface to one color. Do not request a uniform, solid, plain or smooth result on a multicolored, patterned, mottled or weathered surface. Do not infer skin, fabric, fur, or another material when the pixels show only an unresolved silhouette. Avoid guessing specialized object or garment categories: when construction is unclear, identify the visible material/appearance and its location on the owner. Call a lower-body garment trousers only if separated trouser legs are visually evident; otherwise use a factual lower-garment/fabric description.',
}


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def topology_hint(mask):
    """Disambiguate inside/outside using dataset pixels, not a segmenter."""
    h,w=mask.shape
    import cv2
    _,components=cv2.connectedComponents((~mask.astype(bool)).astype(np.uint8))
    center_label=components[h//2,w//2]
    edge_labels=set(np.concatenate([components[0],components[-1],components[:,0],components[:,-1]]))
    if center_label and center_label not in edge_labels:
        return ('IMPORTANT membership evidence: the photo center is EXCLUDED in a closed hole of the mask. '
                'This is an outer region around an excluded interior; '
                'do not mistake the central object enclosed by the inner boundary for the selected region. ')
    return ''


def component_hint(mask, source=None):
    """Describe mask connectivity without pretending components are objects."""
    import cv2
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    total = int(mask.sum())
    minimum = max(16, int(round(total * 0.02)))
    selected_y, selected_x = np.nonzero(mask)
    selected_span = max(
        int(selected_x.max() - selected_x.min() + 1),
        int(selected_y.max() - selected_y.min() + 1),
    )
    substantial_indices = [
        index for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum
        or (
            int(stats[index, cv2.CC_STAT_AREA]) >= 16
            and max(
                int(stats[index, cv2.CC_STAT_WIDTH]),
                int(stats[index, cv2.CC_STAT_HEIGHT]),
            ) >= max(24, round(0.22 * selected_span))
        )
    ]
    substantial = len(substantial_indices)
    if substantial < 1 and count > 1:
        substantial = 1
        substantial_indices = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    base = (
        f'Binary membership evidence: {substantial} substantial or elongated visible mask '
        f'fragment(s) contain selected pixels. Smaller compact fragments may be noise or '
        'thin disconnected parts. Pixel fragments are not object counts: several '
        'fragments can belong to one occluded instance. Do not infer another '
        'selected instance solely from a tiny fragment.'
    )
    if source is None or substantial <= 1:
        return base

    # Objective appearance evidence makes a small but distant selected
    # component harder to overlook without assigning an object label to it.
    source_array = np.asarray(source.convert('RGB'))
    height, width = mask.shape
    details = []
    ordered = sorted(
        substantial_indices,
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
        reverse=True,
    )[:3]
    for number, index in enumerate(ordered, start=1):
        x, y, component_width, component_height, area = map(int, stats[index])
        center_x = x + component_width / 2
        center_y = y + component_height / 2
        horizontal = 'left' if center_x < width / 3 else 'right' if center_x > 2 * width / 3 else 'center'
        vertical = 'upper' if center_y < height / 3 else 'lower' if center_y > 2 * height / 3 else 'middle'
        pixels = source_array[components == index]
        median_rgb = np.median(pixels, axis=0).round().astype(int).tolist()
        details.append(
            f'fragment {number}: {100.0 * area / max(total, 1):.1f}% of selected pixels, '
            f'{vertical}-{horizontal}, median source RGB {tuple(median_rgb)}'
        )
    return base + ' Objective fragment evidence: ' + '; '.join(details) + (
        '. Match each numbered fragment marker to the clean photo independently; '
        'RGB is only a visual cross-check, not an object label.'
    )


def enclosed_hole_hint(mask):
    """Report meaningful enclosed excluded islands without changing the mask."""
    import cv2
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (~mask.astype(bool)).astype(np.uint8), connectivity=8
    )
    edge_labels = set(np.concatenate([
        labels[0], labels[-1], labels[:, 0], labels[:, -1]
    ]).tolist())
    minimum = max(32, int(round(int(mask.sum()) * 0.002)))
    holes = [
        index for index in range(1, count)
        if index not in edge_labels
        and int(stats[index, cv2.CC_STAT_AREA]) >= minimum
    ]
    if not holes:
        return ''
    height, width = mask.shape
    locations = []
    for index in sorted(
        holes, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True
    )[:3]:
        x, y = centroids[index]
        horizontal = 'left' if x < width / 3 else 'right' if x > 2 * width / 3 else 'center'
        vertical = 'upper' if y < height / 3 else 'lower' if y > 2 * height / 3 else 'middle'
        locations.append(vertical + '-' + horizontal)
    return (
        f'IMPORTANT binary topology: {len(holes)} meaningful enclosed EXCLUDED '
        f'hole(s) occur near {", ".join(locations)}. Any person, accessory, '
        'or surface visible inside such a hole is outside the target even though '
        'selected pixels surround it.'
    )


def canonical_target(text):
    """Make a locator easy to embed naturally without losing any content words."""
    value = re.sub(r'\s+', ' ', str(text).strip()).rstrip('.,;:')
    value = re.sub(r'^(?:the|a|an)\s+', '', value, flags=re.I)
    return value


def phrase_words(text):
    return ' '.join(re.findall(r"[a-z0-9]+", canonical_target(text).lower()))


def locator_coverage(target, instruction):
    """Return soft lexical coverage while allowing natural inflection/reordering."""
    stop = {'a','an','the','of','in','on','at','to','with','from','by','and','or'}
    target_words=[word for word in phrase_words(target).split() if word not in stop]
    instruction_words=[word for word in phrase_words(instruction).split() if word not in stop]
    if not target_words:
        return 0.0

    def related(left, right):
        if left == right:
            return True
        # Covers ordinary inflection and close derivations such as
        # hold/holding/held or upholstery/upholstered without a heavy NLP dep.
        irregular={'held':'hold','holding':'hold','holds':'hold'}
        left=irregular.get(left,left);right=irregular.get(right,right)
        if left == right:
            return True
        common=0
        for a,b in zip(left,right):
            if a != b:
                break
            common += 1
        return common >= 5 and common >= min(len(left),len(right)) - 2

    matched=sum(any(related(word, candidate) for candidate in instruction_words)
                for word in target_words)
    return matched / len(target_words)


def locator_is_preserved(target, instruction):
    """Require a compact identifying core, not near-verbatim long phrasing.

    A natural command may omit redundant appearance details while still naming
    the only possible instance.  Three matching content words are enough for a
    long frozen locator; short locators retain a proportional requirement.
    """
    stop = {'a','an','the','of','in','on','at','to','with','from','by','and','or'}
    content = [word for word in phrase_words(target).split() if word not in stop]
    if not content:
        return False
    required = min(2, max(1, int(np.ceil(0.6 * len(content)))))
    return locator_coverage(target, instruction) * len(content) + 1e-9 >= required


def grounding_marker_error(value):
    """Return a high-confidence inconsistency in numbered point evidence.

    The numbered dots are generated from selected pixels.  Requiring the model
    to account for them in order catches cases where it latches onto an
    attractive neighbour and explains away a selected extension.  Lexical
    coherence is deliberately weak: one concrete word shared with the frozen
    target/surface evidence is enough.
    """
    observations = value.get('marker_observations')
    if not isinstance(observations, list):
        return 'missing_marker_observations'
    reason = str(value.get('reason', ''))
    # The VLM sometimes correctly notices that selected pixels from another
    # instance leaked into the mask, then still accepts the dominant instance.
    # Such an answer contradicts the fixed-mask contract and must not advance.
    foreign_fragment_patterns = (
        r'\b(?:excluded|outside|unselected)\b[^.!?]{0,160}'
        r'\b(?:fragment|portion|part|pixels?)\b[^.!?]{0,90}'
        r'\b(?:included|inside|selected|within)\b',
        r'\b(?:erroneous|unrelated|foreign|different[- ]instance)\b[^.!?]{0,90}'
        r'\b(?:fragment|portion|part|pixels?)\b[^.!?]{0,90}'
        r'\b(?:included|inside|selected|within)\b',
    )
    if any(re.search(pattern, reason, re.I) for pattern in foreign_fragment_patterns):
        return 'grounding_admits_foreign_selected_fragment'
    for expected, observation in enumerate(observations, start=1):
        numbers = re.findall(r'\bTARGET\s*(\d+)\b', str(observation), re.I)
        if (not numbers or numbers[0] != str(expected)
                or any(number != str(expected) for number in numbers)):
            return 'marker_observations_not_numbered_in_order'
        if re.search(
            r'\b(?:outside\s+(?:the\s+)?(?:main\s+)?(?:subject|target|selection|mask|boundary)|'
            r'not\s+part\s+of\s+(?:the\s+)?(?:target|selection|mask)|unrelated\s+to\s+(?:the\s+)?target|'
            r'different\s+(?:object|subject))\b',
            str(observation),
            re.I,
        ):
            return 'marker_explicitly_outside_claimed_target'

    # Repeating the same visual claim for two distant numbered dots is a
    # common failure mode on crowded same-category scenes: the model reads the
    # first instance correctly and then silently assigns every other dot to
    # it.  Force a retry so each dot is actually inspected.  Position-only
    # differences remain valid and therefore are retained by normalization.
    normalized_observations = [
        re.sub(r'\btarget\s*\d+\b|[^a-z0-9]+', ' ', str(item).lower()).strip()
        for item in observations
    ]
    target_text = str(value.get('target_description', ''))
    explicitly_plural = bool(re.search(
        r'\b(?:group|cluster|pair|network|canopy|branches|twigs|foliage|'
        r'two|three|four|several|multiple|both)\b',
        target_text,
        re.I,
    ))
    if (not explicitly_plural
            and len(normalized_observations) != len(set(normalized_observations))):
        return 'duplicate_marker_observations'

    # Reject an explicit self-contradiction such as describing a numbered dot
    # as the ``partial sheep on the left`` while listing that same partial
    # sheep as outside.  Require both a distinguishing word and another shared
    # content word so ordinary anatomy repeated across neighbouring instances
    # (for example, two horses' necks) does not trigger this rule.
    outside_words = set(re.findall(
        r'[a-z]+', str(value.get('outside_description', '')).lower()
    ))
    distinguishing = {
        'partial', 'other', 'another', 'adjacent', 'foreground', 'background',
        'nearest', 'farthest', 'separate', 'different',
    }
    generic = {
        'the', 'a', 'an', 'on', 'in', 'at', 'of', 'to', 'with', 'from', 'by',
        'and', 'or', 'area', 'section', 'surface', 'body', 'part', 'edge',
        'upper', 'lower', 'top', 'bottom', 'middle', 'near', 'visible',
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'black', 'white', 'grey', 'gray', 'gold', 'silver',
    }
    for observation in observations:
        observation_words = set(re.findall(r'[a-z]+', str(observation).lower()))
        shared = (observation_words & outside_words) - generic
        if shared & distinguishing and len(shared) >= 2:
            return 'marker_observation_matches_outside_item'

    # A selected-pixel marker cannot land on a person when the target and all
    # selected surfaces are inanimate, while people are explicitly listed as
    # outside. A location tail such as "in the bleachers" is not membership.
    human = r'\b(?:person|people|man|woman|boy|girl|child|pedestrian)s?\b'
    selected_identity = ' '.join([
        str(value.get('target_description', '')),
        *[str(item) for item in value.get('selected_surfaces', [])],
    ])
    outside_text = str(value.get('outside_description', ''))
    if (not re.search(human, selected_identity, re.I)
            and re.search(human, outside_text, re.I)
            and any(re.search(human, str(item), re.I) for item in observations)):
        return 'marker_observation_matches_outside_item'

    stop = {
        'target', 'lies', 'located', 'positioned', 'placed', 'points', 'point',
        'is', 'are', 'was', 'on', 'in', 'at', 'of', 'the', 'a', 'an', 'with',
        'and', 'or', 'to', 'from', 'near', 'just', 'specifically', 'section',
        'area', 'part', 'surface', 'upper', 'lower', 'top', 'bottom', 'middle',
        'mid', 'foreground', 'background', 'left', 'right', 'side', 'source',
        'pixels', 'pixel', 'visible', 'selected', 'selection', 'region',
        'interior', 'extension', 'actual', 'real', 'dark', 'light', 'color',
        'material', 'object', 'serving', 'same', 'main', 'which', 'for',
    }

    def concrete_words(text):
        words = set()
        for word in re.findall(r'[a-z]+', str(text).lower()):
            if len(word) < 3 or word in stop:
                continue
            if word.endswith('ies') and len(word) > 4:
                word = word[:-3] + 'y'
            elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
                word = word[:-1]
            words.add(word)
        return words

    evidence_words = concrete_words(' '.join([
        str(value.get('target_description', '')),
        *[str(item) for item in value.get('selected_surfaces', [])],
    ]))
    if evidence_words:
        for observation in observations:
            if not concrete_words(observation) & evidence_words:
                return 'marker_observation_incoherent_with_target'
    return None


def unsampled_held_target(grounding):
    """Catch an associated held object that was never sampled as selected.

    A person mask is often visually adjacent to a carried object.  The VLM can
    then broaden ``person`` into ``person holding object`` even though no
    numbered selected-pixel point lands on the object.  This guard is limited
    to explicit holding/carrying relations, where the consequence for remove
    and replace is severe; ordinary clothing descriptions are unaffected.
    """
    target = str(grounding.get('target_description', ''))
    match = re.search(
        r'\b(?:holding|carrying|pulling|pushing)\s+(?:(?:a|an|the)\s+)?([^,;]+)',
        target,
        re.I,
    )
    if not match:
        return False
    relation_object = match.group(1)
    stop = {
        'a', 'an', 'the', 'on', 'in', 'at', 'to', 'with', 'from', 'by',
        'left', 'right', 'front', 'rear', 'foreground', 'background',
        'small', 'large', 'dark', 'light', 'red', 'blue', 'green', 'yellow',
        'orange', 'purple', 'pink', 'brown', 'black', 'white', 'grey', 'gray',
        'striped', 'patterned', 'print',
    }
    object_words = {
        word.rstrip('s') for word in re.findall(r'[a-z]+', relation_object.lower())
        if len(word) >= 3 and word not in stop
    }
    marker_words = {
        word.rstrip('s') for word in re.findall(
            r'[a-z]+', ' '.join(map(str, grounding.get('marker_observations', []))).lower()
        )
        if len(word) >= 3 and word not in stop
    }
    return bool(object_words) and not bool(object_words & marker_words)


def unsampled_relational_surface(grounding):
    """Catch support/accessory nouns promoted into membership without evidence.

    Locators such as ``person sitting on the sofa`` are useful even when the
    sofa is outside the mask.  They become unsafe only when the grounding also
    lists that related object as a selected surface despite no numbered point
    sampling it.  Ordinary ``with`` phrases are not checked here because a
    mask may legitimately include equipment whose interior background is an
    excluded hole and whose surface was not chosen for a numbered point.
    """
    target = str(grounding.get('target_description', '')).lower()
    selected = ' '.join(map(str, grounding.get('selected_surfaces', []))).lower()
    markers = ' '.join(map(str, grounding.get('marker_observations', []))).lower()
    candidates = []
    for match in re.finditer(
        r'\b(?:sitting|seated|standing|lying|resting|perched)\s+'
        r'(?:directly\s+)?(?:on|in)\s+(?:(?:a|an|the|his|her)\s+)?([^,;.]+)',
        target,
    ):
        candidates.append(match.group(1))
    stop = {
        'a', 'an', 'the', 'his', 'her', 'its', 'on', 'in', 'at', 'to',
        'and', 'with', 'of', 'left', 'right', 'front', 'rear', 'foreground',
        'background', 'small', 'large', 'dark', 'light', 'red', 'blue',
        'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'black',
        'white', 'grey', 'gray', 'patterned', 'striped', 'visible',
    }

    def words(text):
        return {
            word.rstrip('s') for word in re.findall(r'[a-z]+', text)
            if len(word) >= 3 and word not in stop
        }

    selected_words = words(selected)
    marker_words = words(markers)
    for candidate in candidates:
        # Check coordinated related objects separately so sampling one lid does
        # not silently validate a different unsampled hose.
        for part in re.split(r'\s+and\s+|\s*&\s*', candidate):
            relation_words = words(part)
            if (relation_words & selected_words
                    and not relation_words & marker_words):
                return True
    return False


def target_promotes_unsampled_outside_noun(grounding):
    """Reject a coordinated target noun supported only by outside evidence.

    This catches grounding phrases that correctly identify the main selected
    unit but append a neighbouring object's noun after ``and``/``with`` even
    though no numbered target point or selected-surface phrase samples it.
    """
    target = str(grounding.get('target_description', '')).lower()
    outside = str(grounding.get('outside_description', '')).lower()
    sampled = ' '.join([
        *[str(item) for item in grounding.get('marker_observations', [])],
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ]).lower()
    stop = {
        'a', 'an', 'the', 'its', 'his', 'her', 'and', 'with', 'including',
        'on', 'in', 'at', 'to', 'from', 'by', 'of', 'for', 'left', 'right',
        'front', 'rear', 'foreground', 'background', 'small', 'large', 'dark',
        'light', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'black', 'white', 'grey', 'gray', 'gold', 'silver', 'metal',
        'wooden', 'visible', 'surface', 'part', 'section', 'side', 'upper',
        'lower', 'top', 'bottom', 'middle', 'central', 'center',
    }

    def words(text):
        result = set()
        for word in re.findall(r'[a-z]+', text):
            if len(word) < 3 or word in stop:
                continue
            if word.endswith('ies') and len(word) > 4:
                word = word[:-3] + 'y'
            elif word.endswith('s') and not word.endswith('ss') and len(word) > 3:
                word = word[:-1]
            result.add(word)
        return result

    outside_words = words(outside)
    sampled_words = words(sampled)
    for clause in re.split(r'\b(?:and|with|including)\b', target)[1:]:
        candidate = words(clause)
        if candidate and candidate & outside_words and not candidate & sampled_words:
            return True
    return False


def explicit_outside_dependency(grounding):
    """Read only unambiguous target-attached excluded content.

    This narrow guard complements, rather than replaces, the MLLM relation
    decision.  It intentionally ignores generic contact and positional text.
    """
    outside = str(grounding.get('outside_description', '')).lower()
    if re.search(
        r'\b(?:on|over|around|across)\s+(?:the\s+)?(?:selected|target)'
        r"(?:['’]s|\s+(?:person|man|woman|boy|girl|child|animal)(?:['’]s)?)?"
        r'(?:\s+\w+){0,2}\s+(?:shoulder|back|waist|neck|arm|body|head)\b',
        outside,
    ):
        return 'held_attached'
    if re.search(
        r'\b(?:worn|carried|held|pulled)\s+by\s+(?:the\s+)?(?:selected|target)\b',
        outside,
    ):
        return 'held_attached'
    if re.search(
        r'\b(?:hanging|dangling|suspended)\s+(?:from|on)\s+(?:the\s+)?'
        r'(?:selected|target|person|man|woman|boy|girl|child)(?:[\'’]s)?\s+'
        r'(?:collar|neck|shoulder|waist|belt|arm|hand|body)\b',
        outside,
    ):
        return 'held_attached'
    if re.search(
        r'\b(?:baby|infant|child|person|animal|object|item)\b[^,;]{0,70}'
        r'\b(?:sitting|resting|lying|standing|leaning)\s+(?:directly\s+)?'
        r'(?:on|against|across)\s+(?:(?:the\s+)?(?:selected|target)\b|'
        r'(?:the\s+)?(?:person|man|woman|adult)(?:[\'’]s)?\s+'
        r'(?:lap|torso|body|arm|legs?)\b)',
        outside,
    ):
        return 'supported_by_target'
    target = str(grounding.get('target_description', '')).lower()
    # Removing or replacing a selected container while excluded visible
    # contents remain inside would leave those contents unsupported. Keep the
    # direction narrow: an item selected inside an excluded container is fine.
    containers = r'(?:pot|planter|vase|basket|container|box|bowl|cup|bag|cart)'
    contents = r'(?:plant|flower|foliage|vegetation|leaves|soil|liquid|food|fruit|object|item|contents?)'
    if (re.search(rf'\b{containers}s?\b', target)
            and re.search(
                rf'\b{contents}s?\b[^,;]{{0,70}}\b(?:inside|within|in)\s+'
                rf'(?:the\s+|that\s+|its\s+)?(?:selected\s+|target\s+)?{containers}s?\b',
                outside,
            )):
        return 'supported_by_target'
    return None


def edit_conflicts_with_reflection(grounding):
    """Reject an edit when its optical counterpart lies outside the mask.

    A reflection is not an independent scene object. Editing only a reflected
    target, or editing the physical target while retaining its visible
    reflection, creates an inconsistency a fixed single-region mask cannot
    repair.

    The outside-side check requires lexical agreement with the frozen target,
    so an unrelated reflection elsewhere in the scene is not rejected.
    """
    target = canonical_target(str(grounding.get('target_description', ''))).lower()
    if re.search(
        r'\b(?:reflection|reflected\s+(?:image|figure|object|person|animal))\b',
        target,
    ):
        return True

    outside = str(grounding.get('outside_description', '')).lower()
    reflected = re.search(
        r'\b(?:reflection|reflected\s+(?:image|figure|object|person|animal))\b'
        r'[^,;.]{0,100}',
        outside,
    )
    if not reflected:
        return False

    # Use the head of the frozen referent rather than appearance words. For
    # example, ``black and white cat standing ...`` contributes ``cat``.
    head = re.split(
        r'\b(?:on|near|beside|behind|between|in|at|under|above|by|of|holding|'
        r'carrying|wearing|standing|sitting|facing|located|positioned|parked)\b',
        target,
        maxsplit=1,
    )[0]
    ignored = set(REPLACEMENT_GENERIC_WORDS) | {
        'and', 'with', 'the', 'this', 'that', 'left', 'right', 'front',
        'foreground', 'background', 'small', 'large', 'dark', 'light',
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'black', 'white', 'grey', 'gray', 'gold', 'silver',
    }
    terms = [
        word.rstrip('s')
        for word in re.findall(r'[a-z]+', head)
        if len(word) >= 3 and word not in ignored
    ]
    reflected_words = {
        word.rstrip('s') for word in re.findall(r'[a-z]+', reflected.group(0))
    }
    return bool(terms and terms[-1] in reflected_words)


def selected_external_rigging_risk(grounding):
    """Flag targets visibly equipped for attachment to outside equipment."""
    evidence = ' '.join([
        str(grounding.get('target_description', '')),
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ])
    if re.search(
        r'\b(?:harness|bridle|reins?|tether|tow(?:ing)?\s+(?:strap|cable|bar)|'
        r'trace\s+chain)s?\b',
        evidence,
        re.I,
    ):
        return True

    # A vehicle can be a perfectly coherent mask while still being an
    # incoherent replacement unit: the protected rider/passenger or an
    # externally mounted accessory would have to remain aligned to a newly
    # generated frame.  Require explicit relation evidence so an unrelated
    # nearby person or basket does not trigger this guard.
    target = str(grounding.get('target_description', ''))
    outside = str(grounding.get('outside_description', ''))
    transport = r'(?:bi(?:cycle|ke)|motorcycle|motorbike|scooter|skateboard|sled|sledge)'
    if (re.search(rf'\b{transport}\b[^.;]{{0,70}}\b(?:ridden|occupied|used)\s+by\b', target, re.I)
            and re.search(
                r'\b(?:rider|passenger|person|man|woman|boy|girl|child|legs?|feet|shoes?|'
                r'basket|carrier|pannier|seat)\b',
                outside,
                re.I,
            )):
        return True
    return False


def broad_permanent_scene_edit(row, target):
    """Reject broad or non-interchangeable built-scene remove/replacements."""
    if row.get('task_type') not in {'remove', 'replace'}:
        return False
    text = canonical_target(target).lower()
    if re.search(r'\b(?:toy|model|miniature)\b', text):
        return False
    # Reinforcement emerging from concrete is not a detachable prop even when
    # its visible mask is compact and clean.
    if re.search(r'\brebar\b|\breinforcement\s+(?:cage|rods?|bars?)\b', text):
        return True
    head = re.split(
        r'\b(?:on|near|beside|behind|between|in|at|under|above|by|against|'
        r'holding|carrying|wearing|standing|sitting|facing|located|positioned|'
        r'moored|docked|parked)\b',
        text,
        maxsplit=1,
    )[0]
    # Whole landmark-scale structures are unsuitable even when distant and
    # therefore small in pixel area.
    if re.search(
        r'\b(?:building|skyscraper|high[- ]rise|apartment\s+block|castle|'
        r'fortress|palace|stadium|mosque|church|minaret|tower|bridge)s?\b',
        head,
    ):
        return True
    if re.search(
        r'\b(?:stone|concrete|brick|masonry|structural|load[- ]bearing|temple)\b'
        r'[^,;.]{0,35}\b(?:pillar|column)s?\b|'
        r'\b(?:pillar|column)s?\b[^,;.]{0,50}\b(?:temple|entrance|facade|building)\b',
        text,
    ):
        return True
    # Repainting a local patch of vegetation is useful; regenerating a whole
    # landmark-scale tree is not a fine-grained fixed-mask edit and routinely
    # produces conspicuous canopy boundaries.
    if (re.search(r'\b(?:tree|tree\s+canopy)\b', head)
            and normalized_area(row['mask']) >= 0.12):
        return True
    if re.search(
        r'\b(?:architectural\s+)?panel\b[^,;.]{0,70}'
        r'\bbetween\s+(?:the\s+)?(?:two\s+)?(?:fluted\s+)?(?:columns|pillars)\b',
        text,
    ):
        return True
    grounding = row.get('visual_grounding', {})
    selected_evidence = ' '.join(
        str(item) for item in grounding.get('selected_surfaces', [])
    ).lower()
    # Selected-surface prose often contains component words such as "cabin
    # walls" on an otherwise movable object. Only use that secondary evidence
    # when the target head itself is a generic structural unit.
    structural_evidence = head
    if re.search(
        r'\b(?:structure|architectural\s+element|construction|section|surface|'
        r'area|portion)\b',
        head,
    ):
        structural_evidence += ' ' + selected_evidence
    continuous = re.search(
        r'\b(?:wall|railing|fence|facade|ceiling|floor|roof|road|street|'
        r'ground|lawn|field|river|canal|colonnade)s?\b',
        structural_evidence,
    )
    if not continuous:
        return False
    if re.search(r'\b(?:section|part|portion|stretch|segment)\b', head):
        return True
    if re.search(r'\band\b', head) or re.search(r'\b(?:walls|railings|fences)\b', head):
        return True
    return normalized_area(row['mask']) >= 0.08


def partial_integrated_mechanical_assembly(target):
    """Reject edits that delete several integral parts of one machine.

    A detachable accessory is still a useful local target.  This guard is
    deliberately narrower: it requires an explicit vehicle/machine owner and
    either a named partial section or at least two different integral
    components.  Such a removal would otherwise leave an obviously amputated
    host even though every selected pixel was grounded correctly.
    """
    text = canonical_target(target).lower()
    if not re.search(
        r'\b(?:car|sedan|suv|truck|van|bus|motorcycle|motorbike|scooter|'
        r'bicycle|bike|vehicle|machine|engine)\b',
        text,
    ):
        return False
    if re.search(
        r'\b(?:front|rear|upper|lower)\s+(?:section|assembly|portion)\b',
        text,
    ):
        return True
    components = {
        match.group(1).lower()
        for match in re.finditer(
            r'\b(handlebars?|gauges?|instrument\s+clusters?|fenders?|forks?|'
            r'brake\s+levers?|control\s+assembl(?:y|ies)|bumpers?|headlights?|'
            r'windshields?|wheels?|doors?|hoods?)\b',
            text,
        )
    }
    return len(components) >= 2


def attribute_targets_unresolved_background(grounding):
    """Reject color edits on a background region with no stable identity."""
    target = str(grounding.get('target_description', '')).lower()
    evidence = ' '.join([
        target,
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ]).lower()
    generic_background = bool(re.search(
        r'\b(?:background|backdrop)\s+(?:surface|area|region|patch)\b|'
        r'\b(?:surface|area|region|patch)\b[^.;]{0,35}\bin\s+the\s+background\b',
        target,
    ))
    unresolved = bool(re.search(
        r'\b(?:blurred|indistinct|unidentified|unresolved|unclear|unknown)\b',
        evidence,
    ))
    admitted_guess = bool(re.search(
        r'\b(?:likely|possibly|perhaps|either)\b[^.;]{0,55}'
        r'\b(?:spectator|person|clothing|barrier|backdrop|surface)\b',
        evidence,
    ))
    return generic_background and (unresolved or admitted_guess)


def addition_site_is_grounded(instruction, grounding):
    """Check only high-confidence garment construction claims."""
    claimed = set(re.findall(
        r'\b(?:pocket|buttonhole|belt\s+loop)s?\b',
        instruction.lower(),
    ))
    evidence = ' '.join([
        str(grounding.get('target_description', '')),
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ]).lower()
    if not all(re.search(r'\b' + re.escape(term.rstrip('s')) + r's?\b', evidence)
               for term in claimed):
        return False

    # A body-site noun is not automatically visible merely because the whole
    # person is selected.  Require that the visual grounding actually saw the
    # claimed site (``torso`` is sufficient evidence for an unsided chest).
    body_sites = {
        'chest': r'\b(?:chest|breast|front(?:\s+of\s+the)?\s+torso|front\s+torso)\b',
        'breast': r'\b(?:chest|breast|front(?:\s+of\s+the)?\s+torso|front\s+torso)\b',
        'lapel': r'\blapels?\b',
        'wrist': r'\bwrists?\b',
        'shoulder': r'\bshoulders?\b',
        'sleeve': r'\bsleeves?\b',
    }
    instruction_lower = instruction.lower()
    for site, evidence_pattern in body_sites.items():
        if re.search(r'\b' + site + r's?\b', instruction_lower):
            if not re.search(evidence_pattern, evidence):
                return False
    return True


def addition_uses_ambiguous_body_side(instruction):
    """Reject anatomical left/right sites that cannot be read from the photo.

    ``left chest`` silently alternates between the subject's and viewer's side,
    and is especially unreliable for rear or oblique views.  Image-relative
    wording remains available when a side is genuinely useful.
    """
    if re.search(r"\b(?:viewer['’]?s|image[- ]relative|image)\s+(?:left|right)\b",
                 instruction, re.I):
        return False
    return bool(re.search(
        r"\b(?:left|right)\s+(?:chest|breast|lapel|wrist|forearm|arm|sleeve|"
        r"shoulder|hand|hip|leg|ankle|shoe|foot)\b|"
        r"\b(?:chest|breast|lapel|wrist|forearm|arm|sleeve|shoulder|hand|hip|"
        r"leg|ankle|shoe|foot)\s+on\s+(?:the\s+)?(?:person|man|woman|boy|girl|"
        r"player|soldier|worker)['’]?s?\s+(?:left|right)\b|"
        r"\b(?:upper\s+|lower\s+)?(?:left|right)\s+side\s+of\s+(?:the\s+)?"
        r"(?:person|man|woman|boy|girl|player|soldier|worker)['’]?s?\s+"
        r"(?:\w+\s+){0,2}(?:top|shirt|jersey|jacket|coat|dress|garment)\b",
        instruction,
        re.I,
    ))


def addition_is_microscopic(instruction):
    """Reject additions that cannot be judged in the full training image."""
    return bool(re.search(
        r'\b(?:single|one|small|tiny|minute|individual)\b[^.;]{0,35}'
        r'\b(?:flake|crumb|grain|speck|particle|sprinkle)\b|'
        r'\b(?:flake|crumb|grain|speck|particle)\b',
        instruction,
        re.I,
    ))


def addition_surface_detail_is_unjudgeable(row, instruction):
    """A detail or complex attachment on a tiny host is not auditable."""
    mask = row.get('mask')
    if not isinstance(mask, dict) or 'size' not in mask or 'counts' not in mask:
        return False
    area = normalized_area(mask)
    if re.search(
        r'\b(?:sticker|decal|patch|badge|emblem|label|symbol|logo|mark)\b',
        instruction,
        re.I,
    ):
        return area < 0.002
    if re.search(
        r'\b(?:padlock|locket|pendant|brooch|wristwatch|watch|hair\s+clip)\b',
        instruction,
        re.I,
    ):
        return area < 0.005
    return False


def addition_is_oversized(instruction):
    """The requested corpus is local; an explicitly oversized add is not."""
    return bool(re.search(
        r'\b(?:add|attach|place|put|apply|install|mount)\b[^.;]{0,35}'
        r'\b(?:large|giant|huge|oversized|massive)\b',
        instruction,
        re.I,
    ))


def addition_uses_underspecified_repeated_site(instruction, grounding):
    """Reject an indefinite local part when the visual evidence has many.

    A quadrant does not identify one member of a dense repeated structure.
    Require a stable neighbouring landmark or an ordinal/uniqueness cue.
    """
    placement = re.split(
        r'\b(?:to|on|onto|around|within|inside)\b',
        instruction,
        maxsplit=1,
        flags=re.I,
    )
    if len(placement) < 2:
        return False
    tail = placement[1].lower()
    evidence = ' '.join([
        str(grounding.get('target_description', '')),
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ]).lower()
    # Extract plural evidence conservatively and compare it with the head of
    # an indefinite placement site; avoid any fixed menu of scene objects.
    plural_terms = {
        word[:-3] + 'y' if word.endswith('ies') else word[:-2] if word.endswith('ches') else word[:-1]
        for word in re.findall(r'\b[a-z]{4,}s\b', evidence)
        if not word.endswith(('ss', 'us'))
    }
    site = re.search(
        r'\b(?:a|an)\s+([a-z-]+(?:\s+[a-z-]+){0,4}?)'
        r'(?=\s+(?:in|at|near|beside|behind|between|above|below|from|of)\b|[,.;]|$)',
        tail,
    )
    site_noun = re.findall(r'[a-z]+', site.group(1))[-1] if site else ''
    if not site_noun or site_noun.rstrip('s') not in plural_terms:
        return False
    return not bool(re.search(
        r'\b(?:only|single|nearest|closest|central|center|middle|first|second|'
        r'third|last|beside|next\s+to|adjacent\s+to|immediately|between|'
        r'directly\s+(?:above|below|behind|in\s+front\s+of))\b',
        tail,
        re.I,
    ))


def addition_host_is_broad_natural_background(grounding):
    """Reject a dense scene-wide natural mask with no unique local support."""
    target = str(grounding.get('target_description', ''))
    surfaces = ' '.join(str(item) for item in grounding.get('selected_surfaces', []))
    evidence = f'{target} {surfaces}'
    broad_owner = re.search(
        r'\b(?:forest|woodland|foliage|vegetation|tree\s+canopy|natural\s+background)\b',
        evidence,
        re.I,
    )
    repeated_parts = re.search(
        r'\b(?:branches|twigs|needles|leaves|trees|shrubs|bushes)\b',
        evidence,
        re.I,
    )
    return bool(broad_owner and repeated_parts)


def addition_reason_has_novelty_check(reason):
    """Require the scope pass to expose its image-wide novelty comparison."""
    return bool(re.search(r'\bnovelty\s+check\s*:', str(reason), re.I))


def addition_reason_admits_existing_item(instruction, reason):
    """Reject an add rationale that explicitly sees the same item elsewhere."""
    match = re.search(
        r'\b(?:add|attach|place|put|apply|install|mount)\s+(.*?)'
        r'(?=\s+(?:to|on|onto|at|in|inside|within|around|near|beside)\b|[.;]|$)',
        instruction,
        re.I,
    )
    if not match:
        return False
    ignored = {
        'a','an','the','single','one','small','tiny','compact','thin','round',
        'rectangular','square','red','blue','green','yellow','orange','purple',
        'pink','brown','black','white','grey','gray','gold','silver','bright',
        'dark','light','fresh','new',
    }

    def terms(text):
        result=set()
        for word in re.findall(r'[a-z]+',text.lower()):
            if word in ignored or len(word)<3:continue
            if word.endswith('ies') and len(word)>4:word=word[:-3]+'y'
            elif word.endswith('s') and not word.endswith('ss') and len(word)>3:word=word[:-1]
            result.add(word)
        return result

    item_terms=terms(match.group(1))
    if not item_terms:
        return False
    for sentence in re.split(r'(?<=[.!?])\s+',str(reason)):
        if not re.search(r'\b(?:source|photo|image|scene)\b',sentence,re.I):continue
        if not re.search(r'\b(?:contains?|shows?|has|already|visible|present|existing)\b',sentence,re.I):continue
        if re.search(r'\b(?:no|none|not|without|absent|doesn[’\']?t|does\s+not)\b',sentence,re.I):continue
        if item_terms <= terms(sentence):return True
    return False


def addition_requires_unsupported_material_claim(instruction, grounding):
    """A mounting mechanism must be supported by visible host material."""
    if not re.search(r'\bmagnets?\b',instruction,re.I):return False
    target_head=re.split(
        r'\b(?:located|positioned|beside|near|next\s+to|left\s+of|right\s+of|behind|in\s+front\s+of)\b',
        str(grounding.get('target_description','')),
        maxsplit=1,
        flags=re.I,
    )[0]
    evidence=' '.join([
        target_head,
        *[str(item) for item in grounding.get('selected_surfaces',[])],
    ])
    return not bool(re.search(
        r'\b(?:metal|metallic|steel|iron|magnetic|refrigerator|fridge)\b',
        evidence,
        re.I,
    ))


def instruction_uses_ambiguous_member_reference(instruction):
    """Do not delegate the choice among repeated local parts to the editor."""
    return bool(re.search(
        r'\bone\s+of\s+(?:the\s+|these\s+|those\s+|several\s+|multiple\s+)?',
        instruction,
        re.I,
    ))


def addition_site_is_sufficiently_local(instruction, grounding):
    """Require a sub-location when the selected host is a broad scene plane."""
    target = str(grounding.get('target_description', ''))
    target_head = re.split(
        r'\b(?:on|near|beside|behind|between|in|at|under|above|by|located|positioned)\b',
        target,
        maxsplit=1,
        flags=re.I,
    )[0]
    evidence = ' '.join([
        target_head,
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ])
    if not re.search(
        r'\b(?:building|facade|wall|ground|floor|field|lawn|road|street|path|'
        r'pavement|ceiling|roof)\b',
        evidence,
        re.I,
    ):
        return True
    # A quadrant or "central facade" still leaves a large surface with many
    # valid pixels. Require a concrete landmark, an exact corner, or a stable
    # neighbour relation rather than a coarse direction alone.
    return bool(re.search(
        r'\b(?:corner|entrance|door|window|roofline|base|ground[- ]floor|'
        r'logo|lettering|clock|arch|balcony)\b|'
        r'\b(?:near|beside|above|below|beneath|next\s+to|between|adjacent\s+to|'
        r'directly\s+(?:left|right)\s+of)\b',
        instruction,
        re.I,
    ))


def partial_living_body_part_edit(target):
    """Reject remove/replace plans that amputate only part of a living body."""
    text = canonical_target(target).lower()
    if re.search(r'\b(?:statue|mannequin|doll|toy|sculpture)\b', text):
        return False
    body = r'(?:hand|finger|arm|forearm|wrist|leg|foot|feet|head|torso|body)'
    living = r'(?:person|man|woman|boy|girl|child|baby|adult|human|animal|dog|cat|horse)'
    leading_appearance = (
        r'(?:(?:dark|shadowed|visible|cropped|partial|bare|covered|clothed|'
        r'lower|upper|left|right|foreground|background)[,\s-]+){0,5}'
    )
    return bool(re.search(
        rf'^(?:the\s+)?{leading_appearance}{body}s?\b|'
        rf'\b{living}(?:[\'’]s\s+|\s+)(?:visible\s+)?{body}s?\b|'
        rf'\b{body}s?\s+of\s+(?:the\s+)?{living}\b',
        text,
        re.I,
    ))


def whole_living_target_has_visible_part_outside(grounding):
    """Reject a whole-owner edit when its own visible anatomy is excluded."""
    target=str(grounding.get('target_description',''))
    outside=str(grounding.get('outside_description',''))
    living=r'(?:person|man|woman|boy|girl|child|adult|player|worker|skier|animal|dog|cat|horse|zebra|bear)'
    if not re.search(rf'\b{living}s?\b',target,re.I):return False
    part=r'(?:head|face|neck|torso|body|arm|hand|leg|foot|feet|tail)'
    return bool(re.search(
        rf'\b{part}s?\s+of\s+(?:the\s+)?(?:selected|target)\s+{living}\b|'
        rf'\b(?:selected|target)\s+{living}(?:[’\']s)?\s+{part}s?\b',
        outside,
        re.I,
    ))


def unresolved_tiny_fragment(target):
    """Reject semantically unresolved slivers that are not judgeable objects."""
    return bool(re.search(
        r'\b(?:thin|tiny|small|narrow|unidentified|unknown)\b[^.;]{0,45}'
        r'\b(?:fragment|sliver|speck|strip)\b|'
        r'\b(?:fragment|sliver)\s+or\s+(?:fragment|sliver)\b',
        target,
        re.I,
    ))


def replacement_has_incompatible_footprint(target, replacement, mask_area=None):
    """Catch explicit footprint/aspect expansions the fixed mask cannot hold."""
    if bool(
        re.search(r'\b(?:armchair|dining\s+chair|chair)\b', target, re.I)
        and re.search(r'\b(?:sofa|couch|loveseat|sectional|bench)\b', replacement, re.I)
    ):
        return True
    if (re.search(r'\b(?:high[- ]back(?:ed)?\s+)?(?:dining\s+)?chair\b',target,re.I)
            and re.search(r'\bstool\b',replacement,re.I)):
        return True
    if (re.search(r'\b(?:dining\s+chair|side\s+chair)\b',target,re.I)
            and re.search(r'\b(?:armchair|recliner|lounge\s+chair)\b',replacement,re.I)):
        return True
    if (re.search(r'\b(?:monitor|display|screen)\b', target, re.I)
            and re.search(r'\bultra[- ]?wide\b', replacement, re.I)
            and not re.search(r'\bultra[- ]?wide\b', target, re.I)):
        return True
    if (re.search(r'\b(?:crouching|crouched|seated|sitting|kneeling|lying|prone)\b', target, re.I)
            and re.search(r'\b(?:standing|upright)\b', replacement, re.I)):
        return True
    if (re.search(r'\b(?:bus|coach)\b', target, re.I)
            and re.search(r'\bdouble[- ]decker\b', replacement, re.I)
            and not re.search(r'\bdouble[- ]decker\b', target, re.I)):
        return True
    if (re.search(r'\b(?:bus|coach|truck|van)\b', target, re.I)
            and re.search(r'\b(?:articulated|bendy|multi[- ]section)\b', replacement, re.I)
            and not re.search(r'\b(?:articulated|bendy|multi[- ]section)\b', target, re.I)):
        return True
    if (re.search(r'\b(?:hot\s*dog|sausage)\b', target, re.I)
            and re.search(r'\b(?:sandwich|burger)\b', replacement, re.I)):
        return True
    if (re.search(r'\b(?:whole|entire|charred|round)?\s*pizza\b',target,re.I)
            and not re.search(r'\b(?:slice|piece|portion)\s+of\s+pizza\b',target,re.I)
            and re.search(r'\b(?:slice|piece|portion)\s+of\s+(?:\w+\s+){0,2}pizza\b',replacement,re.I)):
        return True
    if (re.search(r'\b(?:letterbox|mailbox|mail\s+slot)\b', target, re.I)
            and re.search(r'\b(?:doorbell|knocker|button)\b', replacement, re.I)):
        return True
    if (re.search(r'\bumbrellas?\b',target,re.I)
            and re.search(r'\b(?:bouquet|floral\s+arrangement|bunch\s+of\s+flowers?)\b',replacement,re.I)):
        return True
    if (mask_area is not None and mask_area < 0.005
            and re.search(r'\b(?:large|giant|oversized|expanded)\b', replacement, re.I)
            and not re.search(r'\b(?:large|giant|oversized)\b', target, re.I)):
        return True
    return False


def addition_places_loose_item_on_bare_body(instruction, grounding):
    """Reject props merely balanced on a selected bare torso."""
    evidence = ' '.join([
        str(grounding.get('target_description', '')),
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ])
    if not re.search(r'\b(?:bare|shirtless)\b', evidence, re.I):
        return False
    return bool(re.search(
        r'\b(?:on|to)\s+(?:the\s+)?(?:person(?:[’\']s)?|man(?:[’\']s)?|'
        r'woman(?:[’\']s)?|his|her)?\s*(?:bare\s+)?'
        r'(?:back|chest|torso|shoulder|skin|body)\b',
        instruction,
        re.I,
    ))


def addition_places_unfastened_item_on_fur(instruction):
    """A loose decoration cannot simply adhere to animal fur or a coat."""
    if re.search(r'\b(?:attach|clip|pin|tuck|weave|fasten|tie)\b', instruction, re.I):
        return False
    return bool(re.search(
        r'\b(?:add|place|put|set)\b[^.;]{0,55}\b(?:flower|blossom|sprig|leaf|leaves)\b'
        r'[^.;]{0,60}\b(?:to|on|onto|against)\b[^.;]{0,35}\b(?:fur|coat)\b',
        instruction,
        re.I,
    ))


def addition_uses_unstable_narrow_support(instruction):
    """Reject a weighted loose item merely balanced on a narrow support."""
    return bool(
        re.search(
            r'\b(?:potted|pot|planter|vase|bottle|glass|cup|mug|bowl|book|box|'
            r'case|basket|candle|lamp)\b[^.;]{0,80}\b(?:on|onto|atop)\b'
            r'[^.;]{0,40}\b(?:rail|railing|rim|edge|spindle|wire|rope|handle)\b',
            instruction,
            re.I,
        )
    )


def addition_requires_pixels_outside_mask(instruction):
    """Reject explicit placements whose visible body must cross the mask edge.

    This is deliberately limited to strong linguistic evidence.  Surface
    decals, pins, watches, ribbons and ordinary overlapping wearables remain
    eligible, including on small targets.
    """
    patterns = (
        r'\b(?:place|put|set)\b[^.;]{0,100}\b(?:on|onto)\s+(?:the\s+)?'
        r'top(?:\s+surface)?\s+of\b',
        r'\b(?:add|place|put)\b[^.;]{0,100}\bperched\s+on\b',
        r'\b(?:add|attach)\b[^.;]{0,70}\b(?:flag|banner)\b[^.;]{0,70}'
        r'\b(?:to|on)\b[^.;]{0,35}\b(?:pole|post|staff|rod|handle|rail|bar|'
        r'branch|stem|antenna)\b',
        r'\b(?:add|attach|place)\b[^.;]{0,70}\b(?:flag|banner)\b[^.;]{0,70}'
        r'\b(?:to|on|onto|atop)\b[^.;]{0,35}\b(?:top|upper\s+edge|corner)\b',
        r'\b(?:add|attach|place)\b[^.;]{0,70}\b(?:flag|banner|pennant)\b[^.;]{0,70}'
        r'\b(?:to|on|onto)\b[^.;]{0,40}\b(?:frame|gate|fence|mesh|grille)\b',
        r'\b(?:add|place|put)\b[^.;]{0,70}\b(?:top\s+hat|crown|antlers?)\b'
        r'[^.;]{0,70}\b(?:head|top)\b',
        r'\b(?:add|attach|place|install|mount)\b[^.;]{0,70}'
        r'\b(?:antenna|aerial|spire|chimney|weather\s+vane|mast)\b'
        r'[^.;]{0,70}\b(?:roof|roofline|rooftop|top)\b',
        # A volumetric loose object sitting on a selected planar support rises
        # into background pixels that the immutable host mask cannot edit.
        r'\b(?:add|place|put|set)\b[^.;]{0,45}\b(?:cup|mug|glass|bottle|vase|'
        r'bowl|box|book|pot|planter|figurine|statue|cone|lamp|candle|toy|apple|'
        r'fruit|ball|ornament|device|helmet|hard\s+hat)\b'
        r'[^.;]{0,65}\b(?:on|onto|atop)\b[^.;]{0,55}\b(?:table|desk|counter|'
        r'shelf|stand|brochures?|tray|plate|seat|bench|floor|ground|surface|'
        r'roof|hood|lid|bags?|stack|pallet)\b',
        # A solid plate mounted across a sparse support necessarily fills the
        # gaps, which are excluded pixels in a slat/wire mask.
        r'\b(?:add|attach|mount|place)\b[^.;]{0,50}\b(?:sign|signboard|plaque|'
        r'nameplate|plate|panel)\b[^.;]{0,55}\b(?:to|on|onto)\b[^.;]{0,55}'
        r'\b(?:picket|slatted|wire|mesh|chain[- ]link)\s+(?:fence|railing)\b',
    )
    return any(re.search(pattern, instruction, re.I) for pattern in patterns)


def addition_conflicts_with_excluded_contact(instruction, grounding):
    """Use explicit outside evidence to prevent adding onto an occupied site."""
    outside = str(grounding.get('outside_description', ''))
    if not re.search(
        r'\b(?:resting|sitting|lying|placed|mounted|attached|touching|covering|'
        r'perched)\s+(?:directly\s+)?(?:on|onto|against|around|over)\b|'
        r'\bon\s+(?:the\s+)?(?:selected|target|its)\b',
        outside,
        re.I,
    ):
        return False
    placement = re.split(r'\b(?:on|onto|atop|to|around)\b', instruction, maxsplit=1, flags=re.I)
    if len(placement) < 2:
        return False
    stop = {
        'the', 'a', 'an', 'selected', 'target', 'its', 'left', 'right',
        'upper', 'lower', 'top', 'bottom', 'small', 'large', 'visible',
        'person', 'man', 'woman', 'boy', 'girl', 'child', 'animal', 'dog',
        'cat', 'horse', 'sheep', 'elephant', 'vehicle', 'car', 'truck',
    }
    site_words = {
        word.rstrip('s') for word in re.findall(r'[a-z]+', placement[1].lower())
        if len(word) >= 3 and word not in stop
    }
    contact_complements = re.findall(
        r'\b(?:resting|sitting|lying|placed|mounted|attached|touching|covering|'
        r'perched)\s+(?:directly\s+)?(?:on|onto|against|around|over)\s+'
        r'([^,;.]{1,90})',
        outside,
        re.I,
    )
    if not contact_complements:
        return False
    stop.update({
        'stand', 'table', 'desk', 'counter', 'fence', 'railing', 'chair',
        'vehicle', 'bicycle', 'surface', 'side', 'front', 'rear', 'back',
    })
    outside_words = {
        word.rstrip('s')
        for phrase in contact_complements
        for word in re.findall(r'[a-z]+', phrase.lower())
        if len(word) >= 3 and word not in stop
    }
    return bool(site_words & outside_words)


def multicolored_or_patterned(text):
    if re.search(
        r'\b(?:patterned|pattern|striped|floral|plaid|checked|printed|spotted|'
        r'camouflage|camo|multicolou?red|multi-colou?red|two-tone|bicolou?red)\b',
        text,
        re.I,
    ):
        return True
    colors = r'(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey|gray|teal|gold|silver)'
    return bool(re.search(rf'\b{colors}\s+(?:and|/)\s+{colors}\b', text, re.I))


def textured_or_weathered(text):
    return bool(re.search(
        r'\b(?:weathered|mottled|variegated|patina|patinated|stained|'
        r'distressed|textured|marbled)\b',
        text,
        re.I,
    ))


def attribute_color_change_is_low_contrast(instruction, grounding):
    """Reject a narrow set of source/target colors that remain near-black."""
    evidence = ' '.join([
        str(grounding.get('target_description', '')),
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ])
    return bool(
        re.search(r'\b(?:black|near[- ]black|very\s+dark\s+(?:grey|gray|blue))\b', evidence, re.I)
        and re.search(r'\b(?:navy(?:\s+blue)?|dark\s+blue|near[- ]black)\b', instruction, re.I)
    )


def attribute_uses_implausible_living_tissue_color(instruction, grounding):
    """Keep ordinary photographic skin edits within plausible appearance."""
    if not re.search(r'\bskin(?:\s+tone|\s+colou?r)?\b', instruction, re.I):
        return False
    evidence = ' '.join([
        str(grounding.get('target_description', '')),
        str(grounding.get('reason', '')),
        *[str(item) for item in grounding.get('selected_surfaces', [])],
    ])
    if not re.search(
        r'\b(?:person|people|human|man|woman|boy|girl|child|baby|adult|'
        r'animal|mammal|elephant|giraffe|horse|dog|cat|bear|monkey)s?\b',
        evidence,
        re.I,
    ) and not re.search(r'\bskin\s+tone\b', instruction, re.I):
        return False
    return bool(re.search(
        r'\b(?:bright|vivid|neon|solid)\b[^.;]{0,20}'
        r'\b(?:blue|green|purple|violet|orange|yellow|teal|gold|silver)\b',
        instruction,
        re.I,
    ))


def attribute_uses_implausible_intrinsic_material_color(instruction):
    """Catch a narrow high-confidence material/color contradiction."""
    return bool(
        re.search(r'\btomato\s+sauce\b',instruction,re.I)
        and re.search(r'\b(?:bright\s+)?(?:blue|green|teal|purple|violet)\b',instruction,re.I)
    )


def attribute_recolors_whole_pattern(instruction, grounding_text=''):
    """Require a named constituent for an explicitly multi-color pattern."""
    colors = set(re.findall(
        r'\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|'
        r'grey|gray|teal|gold|silver)\b',
        grounding_text,
        re.I,
    ))
    if len({color.lower().replace('gray', 'grey') for color in colors}) < 2:
        return False
    if not re.search(
        r'\b(?:change|set|make|turn)\b[^.;]{0,100}'
        r'\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|'
        r'grey|gray|teal|gold|silver)\b',
        instruction,
        re.I,
    ):
        return False
    prefix = re.split(r'\b(?:to|into)\b', instruction, maxsplit=1, flags=re.I)[0]
    prefix_colors = {
        color.lower().replace('gray', 'grey')
        for color in re.findall(
            r'\b(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|'
            r'grey|gray|teal|gold|silver)\b',
            prefix,
            re.I,
        )
    }
    grounding_colors = {color.lower().replace('gray', 'grey') for color in colors}
    if prefix_colors & grounding_colors:
        return False
    return not bool(re.search(
        r'\b(?:stripes?|spots?|dots?|checks?|lines?|motifs?|flowers?|'
        r'pattern|background|base)\b',
        instruction,
        re.I,
    ))


def unresolved_silhouette(text):
    """Distinguish genuinely unresolved silhouettes from ordinary shape prose."""
    return bool(re.search(
        r'\b(?:only|mere(?:ly)?|entirely|pure(?:ly)?|solid|dark)\b'
        r'[^.;]{0,35}\bsilhouette\b|'
        r'\bsilhouette\b[^.;]{0,35}\b(?:only|unresolved|without\s+visible\s+detail)\b|'
        r'\bseen\s+(?:only\s+)?(?:as|in)\s+(?:a\s+)?silhouette\b',
        text,
        re.I,
    ))


def replacement_is_merely_surface_variant(target, replacement):
    """Narrow same-category rejection to color/material/style-only variants.

    A distinct constituent, form, breed, or human identity can still make a
    useful replacement.  The scope VLM must judge whether that distinction is
    actually visible; this guard handles only the deterministic no-op cases.
    """
    same_category = same_replacement_category(target, replacement)

    def head_segment(text):
        return re.split(
            r'\b(?:on|near|beside|behind|between|in|at|under|above|by|of|'
            r'holding|carrying|worn|wearing|standing|sitting|facing|located|'
            r'positioned|mounted)\b',
            text,
            maxsplit=1,
            flags=re.I,
        )[0]

    old_head = head_segment(target)
    new_head = head_segment(replacement)
    # Selected surface carriers do not become semantic replacements merely by
    # changing material, style, advertised use, or a close lexical subtype.
    # Apply this only to the referent head, so a whole person described as
    # wearing one of these items can still be replaced by a new person.
    head_carrier_groups = (
        r'\b(?:shirt|t-?shirt|tee|flannel\s+shirt)\b',
        r'\b(?:coat|jacket|raincoat|rain\s+jacket|cardigan)\b',
        r'\bbottles?\b',
        r'\b(?:letterbox|mailbox|mail\s+slot)\b',
        r'\b(?:hot\s*dog|sausage|bratwurst|frankfurter)\b',
        r'\bumbrellas?\b',
    )
    if any(re.search(group, old_head, re.I) and re.search(group, new_head, re.I)
           for group in head_carrier_groups):
        return True
    # A hanging banner/standard is routinely used as a lexical escape hatch
    # for a flag whose color or emblem merely changed.  A genuinely different
    # silhouette such as a pennant or windsock remains eligible.
    flag_banner_alias = bool(
        re.search(r'\bflag\b', target, re.I)
        and re.search(r'\b(?:banner|standard)\b', replacement, re.I)
    )
    if flag_banner_alias:
        return True

    # These are surface carriers: changing their print, advertised purpose,
    # color or pattern leaves the same physical object category.  Include
    # common lexical aliases because the model otherwise escapes by changing
    # ``necktie`` to ``tie`` or ``sign`` to ``signboard``.
    carrier_aliases = (
        (r'\b(?:neck)?tie\b', r'\b(?:neck)?tie\b'),
        (r'\b(?:sign|signboard|billboard|placard)\b',
         r'\b(?:sign|signboard|billboard|placard|menu\s+board|notice\s+board|'
         r'advertising\s+board|display\s+board)\b'),
        (r'\b(?:poster|advertisement|advert|display\s+panel)\b',
         r'\b(?:poster|advertisement|advert|display\s+panel)\b'),
        (r'\b(?:mural|fresco|wall\s+painting|artwork)\b',
         r'\b(?:mural|fresco|painting|wall\s+art|artwork|art\s+panel)\b'),
        (r'\b(?:computer\s+)?(?:monitor|display|screen|television|tv)\b',
         r'\b(?:computer\s+)?(?:monitor|display|screen|television|tv)\b'),
        (r'\b(?:pillar|column)\b', r'\b(?:pillar|column)\b'),
        # High-confidence near-synonyms whose proposed visual difference is
        # only upholstery/material, livery, or foliage color. Distinct forms
        # such as a swivel chair, biplane, canoe, or tree remain eligible.
        (r'\barmchair\b', r'\b(?:wingback\s+chair|armchair)\b'),
        (r'\b(?:aircraft|airplane|aeroplane|commercial\s+jet|passenger\s+jet|airliner)\b',
         r'\b(?:aircraft|airplane|aeroplane|commercial\s+jet|passenger\s+jet|airliner)\b'),
        (r'\b(?:wooden\s+)?(?:fishing\s+)?boat\b',
         r'\b(?:wooden\s+|aluminum\s+|aluminium\s+|red\s+|blue\s+)*'
         r'(?:fishing\s+)?(?:boat|skiff)\b'),
        (r'\bbush(?:es)?\b', r'\bshrubs?\b'),
        (r'\bshrubs?\b', r'\bbush(?:es)?\b'),
    )
    if any(re.search(old, target, re.I) and re.search(new, replacement, re.I)
           for old, new in carrier_aliases):
        return True
    # Different culinary labels do not make a useful replacement when the
    # source and result share the same elongated green visual form.
    if (re.search(r'\bcucumber\b', target, re.I)
            and re.search(r'\b(?:zucchini|courgette)\b', replacement, re.I)):
        return True
    style_word = bool(re.search(r'\b(?:style|styled|look|design)\b', replacement, re.I))
    if style_word:
        ignored = {
            'the', 'and', 'with', 'worn', 'wearing', 'standing', 'sitting',
            'foreground', 'background', 'left', 'right', 'center', 'central',
            'wide', 'small', 'large', 'style', 'styled', 'look', 'design',
        }
        old_words = {
            word.rstrip('s') for word in re.findall(r'[a-z]+', target.lower())
            if len(word) >= 4 and word not in ignored
        }
        new_words = {
            word.rstrip('s') for word in re.findall(r'[a-z]+', replacement.lower())
            if len(word) >= 4 and word not in ignored
        }
        if old_words & new_words:
            return True
    # ``group of powdered donuts`` is headed by the generic word ``group`` in
    # the legacy parser.  Recover the actual repeated item so changing only a
    # coating cannot masquerade as an object replacement.
    if not same_category:
        group_prefix = r'\b(?:group|cluster|collection|set)\s+of\b'
        old_group = re.search(group_prefix + r'[^,;]*?\b([a-z]+)\b(?:\s+(?:in|on|at|near|beside|inside)\b|[.!]?\s*$)', target, re.I)
        new_group = re.search(group_prefix + r'[^,;]*?\b([a-z]+)\b(?:\s+(?:in|on|at|near|beside|inside)\b|[.!]?\s*$)', replacement, re.I)
        if old_group and new_group:
            same_category = old_group.group(1).rstrip('s').lower() == new_group.group(1).rstrip('s').lower()
    if not same_category:
        return False
    human = {
        'person', 'man', 'woman', 'boy', 'girl', 'child', 'adult', 'player',
        'worker', 'pedestrian', 'rider', 'cyclist', 'surfer', 'skier',
    }
    target_tokens = set(re.findall(r'[a-z]+', target.lower()))
    replacement_tokens = set(re.findall(r'[a-z]+', replacement.lower()))
    if style_word:
        return True
    generic = set(REPLACEMENT_GENERIC_WORDS) | {
        'brick', 'painted', 'paint', 'matte', 'glossy', 'gloss', 'plain',
        'solid', 'smooth', 'rough', 'bright', 'pale', 'deep',
        'chocolate', 'frosted', 'powdered', 'glazed', 'coated', 'topped',
        'icing', 'frosting', 'sugar', 'rubber', 'group', 'cluster',
        'collection', 'set', 'featuring', 'feature', 'emblem', 'logo',
        'symbol', 'cross', 'stripe', 'striped', 'pattern', 'motif', 'letter',
        'number', 'text', 'graphic', 'print',
    }

    # A new person is a valid identity replacement, but changing only their
    # clothes/uniform is likely to be executed as an attribute edit.  Accept a
    # same-supercategory human replacement only when the language establishes
    # some new identity, physical characteristic, demographic, role, or form.
    if target_tokens & human and replacement_tokens & human:
        attire = {
            'uniform', 'jersey', 'shirt', 'tshirt', 'jacket', 'coat', 'suit',
            'dress', 'skirt', 'trouser', 'trousers', 'pants', 'shorts', 'sock',
            'socks', 'shoe', 'shoes', 'boot', 'boots', 'hat', 'cap', 'clothes',
            'clothing', 'wearing', 'worn', 'raincoat', 'hoodie', 'sweater',
            'cardigan', 'apron', 'jeans', 'denim',
        }
        colors = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'brown', 'black', 'white', 'grey', 'gray', 'teal', 'gold', 'silver',
        }
        ignored_human = generic | attire | colors | human | {
            'in', 'with', 'and', 'the', 'a', 'an', 'another', 'different',
        }
        old_identity = target_tokens - ignored_human
        new_identity = replacement_tokens - ignored_human
        explicit_identity = {
            'male', 'female', 'elderly', 'older', 'young', 'teenage', 'bald',
            'blonde', 'brunette', 'bearded', 'chef', 'referee', 'doctor',
            'nurse', 'firefighter', 'police', 'clown', 'mascot', 'robot',
        }
        if ((replacement_tokens & explicit_identity) - (target_tokens & explicit_identity)
                or (new_identity - old_identity)):
            return False
        return True

    def content(tokens):
        result = set()
        for token in tokens:
            if token in generic:
                continue
            if token.endswith('ies') and len(token) > 4:
                token = token[:-3] + 'y'
            elif token.endswith('s') and not token.endswith('ss') and len(token) > 3:
                token = token[:-1]
            result.add(token)
        return result

    return not (content(replacement_tokens) - content(target_tokens))


def dependency_from_grounding(grounding):
    """High-precision relation policy over the VLM's own visual evidence.

    This deliberately uses relation phrases rather than object-name examples,
    so it generalizes across scene content and does not bias prompt diversity.
    Only relations explicitly attached to *excluded* content are considered.
    A previous implementation scanned the whole grounding explanation and
    therefore confused ``person wearing a shirt`` or ``girl sitting on the
    right`` with an outside dependency.
    """
    target = str(grounding.get('target_description', '')).lower()
    outside = str(grounding.get('outside_description', '')).lower()
    reason = str(grounding.get('reason', '')).lower()

    # The explanation is usable only when the dependency phrase occurs in the
    # same explicit exclusion clause. Do not infer direction merely because an
    # excluded neighbour is held by somebody else or shares a noun with the
    # selected target.
    exclusion_clause = re.compile(
        r'\b(?:exclude[sd]?|outside(?:\s+the)?\s+(?:mask|boundary)|not selected)\b'
        r'[^.!?]{0,140}\b(?:held|carried|worn|attached|being\s+(?:pulled|carried|ridden))\b'
    )
    reverse_exclusion_clause = re.compile(
        r'\b(?:held|carried|worn|attached|being\s+(?:pulled|carried|ridden))\b'
        r'[^.!?]{0,140}\b(?:exclude[sd]?|outside(?:\s+the)?\s+(?:mask|boundary)|not selected)\b'
    )
    if exclusion_clause.search(reason) or reverse_exclusion_clause.search(reason):
        return 'held_attached'

    # Use support direction only when the excluded item is explicitly the
    # grammatical subject and contact is explicit.  Bare "standing on the
    # right" and "sitting on the left" are positional locators, not support.
    if re.search(
        r'^(?:the\s+)?(?:person|people|child|animal|object|item|box|bag)\b'
        r'[^,;]{0,60}\b(?:sitting|resting|standing|lying|placed)\s+'
        r'(?:directly\s+)?on\s+(?:top\s+of\s+)?(?:the\s+)?selected\b',
        outside,
    ):
        return 'supported_by_target'
    if re.search(
        r'\b(?:structure|component|object|part)\s+(?:directly\s+)?above\b|'
        r'\b(?:rests?|resting|mounted)\s+(?:directly\s+)?(?:on top of|above)\b|'
        r'\bsupported\s+by\b',
        outside,
    ):
        return 'structural_support'
    if re.search(
        r'\b(?:remaining|rest|continuation)\b.{0,60}\b(?:extending|continues?|connected|same)\b|'
        r'\b(?:same|continuous)\b.{0,30}\b(?:owner|object|structure|bicycle|vehicle|bench|wall|rail|fence)\b',
        outside,
    ):
        return 'continuous_owner'
    return None


def prompt_for(row, stage, feedback=''):
    binding=row.get('reference_binding',{})
    # Shared/group hints routinely cause whole-group instructions for one instance.
    reference = binding.get('label','') if binding.get('status')=='bound_mention' else ''
    siblings = [bind_reference(row.get('answer',''),i,row.get('num_masks',1)).get('label')
                for i in range(row.get('num_masks',1)) if i!=row.get('mask_index')]
    siblings = [s for s in siblings if s and s!=reference and s!=binding.get('label')]
    common = f'''Two images: (1) the clean full source photo; (2) a context crop of that same photo with a black/white boundary and one or more numbered external TARGET pointers. Every numbered blue dot labelled TARGET 1/2/3 lies on selected pixels; extension dots call attention to distant thin parts of the same connected region. Only pixels inside the boundary are selected; holes are excluded. The boundary, pointer, blue dots, numbers and text are annotations, never scene colors. Use the full photo to distinguish similar instances.
The supplied dataset mask is fixed and usable; no later source segmentation or mask expansion will occur.
{row.get('mask_geometry_hint','')}
{row.get('mask_topology_hint','')}
{row.get('mask_component_hint','')}
{row.get('mask_hole_hint','')}
Dataset referring hint: {json.dumps(reference, ensure_ascii=True)}. Empty means a shared/group hint was deliberately withheld. A supplied hint is optional context, NOT evidence that the named whole object or group is selected; identify the actually outlined member/part from the pixels.
Other separately annotated regions in this source: {json.dumps(siblings,ensure_ascii=True)}. These are other targets, not permission to edit them together with this mask. Do not merge an adjacent surface into the selected object description.
Judge only visible selected pixels. Occlusion and photo-boundary truncation do not make a mask incomplete. A background surface, body part, coherent group, or several disconnected visible fragments of one occluded instance can be valid. Do not invent invisible anatomy or attachments. Read appearance from actual photo pixels. If a category is uncertain, use a factual visual description. Use photo-relative positions, not ambiguous anatomical left/right.
'''
    if stage == 'ground':
        return common + '''VISUAL GROUNDING ONLY. Do not propose, imagine, or discuss an edit.
First identify which real visible pixels are inside the boundary. Inspect the entire boundary systematically, not just the center. Describe the real source appearance directly underneath every numbered blue TARGET dot before naming the target. Start each observation with its exact label in numeric order (TARGET 1, TARGET 2, TARGET 3 as present); never skip, merge, renumber, or explain away a dot. Objects on the other side of the boundary are excluded even when they touch, overlap, are held by, rest on, or visually belong to the selected owner. Trace disconnected selected fragments back to their visible owner before deciding whether they are one instance or several. When objective fragment evidence reports two or more substantial fragments, compare each fragment's stated full-photo location and median source RGB with the clean photo before answering. Do not assign a fragment to the dominant instance merely because the dataset query mentions both. Two fragments may share one occluded owner only when their visible appearance and intervening occlusion make that ownership plausible. If a fragment or numbered dot lands on a different visible instance, even another instance of the same category, decision must be reject.
For outside_description, prioritize boundary-contact dependencies in this order: (1) items in hands or mouth, (2) bags/clothes/accessories attached to a person, (3) excluded contents visibly enclosed by a selected container, (4) any excluded item resting on, covering, touching, tethered to, or occupying a selected surface/contact point, (5) people or objects supported by the selected surface, (6) a visible mirror/glossy-surface reflection of the selected target or the real counterpart of a selected reflection, (7) structure supported above/below it, (8) an unselected continuation of the same continuous object, then (9) the nearest similar instance. Trace visible straps, cables, ropes, hoses and rigging to the object at their other end. When an excluded person, animal or object overlaps the target's body, lap, support surface or silhouette, state the direction of support/contact explicitly instead of calling it merely nearby. List up to three relevant excluded items rather than substituting a distant neighbor.
Test every noun separately before using words such as "and", "including", "holding", "carrying", or "with" in target_description. Such a noun may appear in the target phrase only when its own visible pixels are inside. If the boundary cuts through a longer bench, train, wall, bed, or other continuous structure, describe only the selected section/surface/cars, never the whole owner.
The target_description must uniquely identify the selected unit after all annotations are removed. Mentally test it against the clean full photo: if two visible candidates satisfy the phrase, add one concise position, appearance, or neighbour relation. Do not copy annotation terminology into it. Never broaden a selected person or animal to an object it is holding, pulling, pushing, harnessed to, or touching unless at least one numbered TARGET dot visibly lands on that second object. A visible association is not mask membership.
Resolve clothing and object categories from visible construction, not expectation. When seams, openings, transparency, or shape do not establish a specialized category, use a factual appearance phrase such as outer garment or visible fabric rather than guessing.
List only concrete surfaces or components visibly inside the boundary in selected_surfaces. This list is evidence for later local attribute edits, so never include an item merely because it is normally attached to the owner or visible in an excluded hole. A relationship word such as "on" or "with" is not membership evidence: trace the boundary around each related item, use numbered TARGET dots when present, and keep an item only when its own visible surface is selected. Re-check every orange EXCLUDED HOLE pointer: content visible at that pointer is outside even if it is enclosed by the target outline.
Reject only if the selected pixels cannot be described as a coherent visible object, part, surface, or group.
Return ONLY JSON with six fields:
reason: specific visual evidence for the inside/outside distinction, at most three short sentences;
decision: accept or reject;
marker_observations: a list with one short entry for each numbered blue TARGET dot, in numeric order, naming the actual source pixels under that dot;
target_description: one concise source object/part phrase (1-24 words) that uniquely identifies only the selected unit in the FULL photo, using a short position or neighbor relation when similar instances exist;
selected_surfaces: a list of 1-5 concise visible surface/component phrases definitely inside the boundary;
outside_description: up to three relevant excluded contacts/dependencies or look-alikes, prioritizing contacts as specified above, or "none".
''' + feedback

    task = row['task_type']
    grounding = row.get('visual_grounding', {})
    frozen_target = grounding.get('target_description', '')
    frozen_outside = grounding.get('outside_description', 'none')
    frozen_evidence = grounding.get('reason', '')
    frozen_surfaces = grounding.get('selected_surfaces', [])
    request = f'''The first visual pass is now frozen:
selected target: {json.dumps(frozen_target, ensure_ascii=True)}
selected visible surfaces: {json.dumps(frozen_surfaces, ensure_ascii=True)}
nearest relevant excluded content: {json.dumps(frozen_outside, ensure_ascii=True)}
visual evidence summary: {json.dumps(frozen_evidence, ensure_ascii=True)}
Independently verify this frozen grounding against both images. If it is wrong, reject; never silently switch to another object or broaden/narrow it. If correct, design exactly one feasible {task} edit for that selected unit.
The frozen text is a claim to test, not permission. Check every noun joined by "and", "including", "holding", "carrying", "with", or "on" separately. Reject if any claimed target noun lies across the boundary or in an excluded hole. Also reject if excluded contact/support content makes this task physically incoherent.
Task type: {task}
{RULES[task]}
The mask is the maximum allowed edit region, not necessarily an instruction to change every selected pixel. The command must identify the intended instance even after annotations are removed. Locate a local part within its owner. Include a short full-photo position or neighbor relation when another instance could match.
For every task, reject a selected reflection whose real counterpart is outside, and reject a physical target whose corresponding visible reflection is outside: a fixed single-region edit cannot keep the pair optically consistent. For removal, inspect hands, mouth, supports, container interiors and contact points for separately visible dependent content outside the boundary; reject if removal would leave it floating. Also inspect straps, cables, ropes, reins, harnesses and other rigging that connect the selected unit to outside equipment. For replacement, inspect held/carried accessories, enclosed contents, supported people/items, structural attachments, and unselected continuation of the same object; reject an incoherent partial replacement. Treat excluded content enclosed by a selected container, or an excluded living being or object overlapping and resting on the target's body/lap, as supported_by_target even though it is a separate semantic instance. A clearly detached portion resting independently is not continuous_owner merely because it came from a larger item. For addition, first search the clean full photo for the proposed item and its closest visible lookalike. A new name for a visually equivalent subtype does not establish absence. If any matching or visually indistinguishable item is already visible anywhere, choose a visibly distinct absent item. Begin the reason with "Novelty check:" and state the closest lookalike (or that none is visible) and why the proposal remains visually distinguishable at full-image scale. Use an actually visible mounting surface on the frozen target, never an excluded accessory. Mentally draw the new item's complete visible silhouette: accept only when it fits substantially inside selected pixels. A selected host is not permission to draw above it or beside it in excluded background. The proposed item must be stably supported under ordinary gravity, fit the support width, remain judgeable at full-image scale, and must not compete with an excluded object already touching or occupying that exact site. Reject a physically or socially implausible attachment. For a color-only attribute, name the exact visibly evidenced material/surface, use obvious visual contrast, preserve every existing marking, weathering and texture unless that property itself is the requested edit, and keep living tissue visually plausible for an ordinary photograph.
Write a concise English command, normally 8-24 words (maximum 32), with only the action, unambiguous target and desired result. Do not include preservation clauses, reconstruction recipes, coordinates, or annotation language.
Classify the most consequential relation between the frozen target and excluded content as exactly one of: none, independent, held_attached, supported_by_target, structural_support, continuous_owner. Every conflict class is directional and applies only to content visibly OUTSIDE the boundary. "held_attached" means an excluded item is held, worn, or carried by the target; clothing or an accessory already inside the boundary is not a conflict. "supported_by_target" means excluded content rests, sits, or stands on top of the target; a selected person merely sitting on an excluded chair is the opposite direction. "structural_support" means the selected target supports excluded structure above/below it; a self-contained ladder, lamp, or module merely mounted on a larger structure is not structural support. "continuous_owner" means the mask selects only a non-interchangeable part of the same continuous object. When claiming a conflict, identify the exact excluded item visible across the boundary; never infer one from pose alone.
HARD DECISION CONTRACT: for remove, held_attached/supported_by_target/structural_support/continuous_owner requires reject; a self-contained detachable object is not continuous_owner. For replace, held_attached/supported_by_target/structural_support/continuous_owner requires reject unless the selected part is visibly a self-contained interchangeable component and the instruction names only that component. Do not describe such a conflict and then accept it.
Return ONLY JSON with five fields:
reason: state whether the frozen target matches the selected pixels, name the relevant excluded content, and explain feasibility; at most three short sentences;
decision: accept or reject;
dependency_class: exactly one class from the list above;
target_description: copy all words of the frozen selected target in the same order, or an empty string if rejecting;
editing_instruction: a concise command that preserves the same distinctive category and full-photo locator. Natural inflection and word order are allowed; avoid awkward possessives or duplicated descriptions. Return an empty string if rejecting.
'''
    return common + request + feedback


def validate(value, row, stage):
    if not isinstance(value,dict): return None,'invalid_json'
    if value.get('decision') not in {'accept','reject'}: return None,'invalid_decision'
    if not isinstance(value.get('reason'),str) or not value['reason'].strip(): return None,'missing_reason'
    if value['decision']=='reject': return None,'model_rejected'
    if stage == 'ground':
        target = value.get('target_description')
        outside = value.get('outside_description')
        marker_observations = value.get('marker_observations')
        selected_surfaces = value.get('selected_surfaces')
        if not isinstance(target, str) or not isinstance(outside, str):
            return None, 'missing_grounding_text'
        if (not isinstance(marker_observations, list)
                or not 1 <= len(marker_observations) <= 3
                or not all(isinstance(item, str) and item.strip() for item in marker_observations)):
            return None, 'missing_marker_observations'
        if (not isinstance(selected_surfaces, list)
                or not 1 <= len(selected_surfaces) <= 5
                or not all(isinstance(item, str) and item.strip() for item in selected_surfaces)):
            return None, 'missing_selected_surfaces'
        marker_error = grounding_marker_error(value)
        if marker_error:
            return None, marker_error
        if unsampled_held_target(value):
            return None, 'held_target_not_sampled_by_marker'
        if unsampled_relational_surface(value):
            return None, 'associated_surface_not_sampled_by_marker'
        if target_promotes_unsampled_outside_noun(value):
            return None, 'grounding_promotes_unsampled_outside_noun'
        if not target.isascii() or not 1 <= len(target.split()) <= 32:
            return None, 'invalid_grounding_target'
        target = canonical_target(target)
        grounding = {
            'version': VERSION,
            'decision': 'accept',
            'reason': value['reason'],
            'marker_observations': [item.strip() for item in marker_observations],
            'target_description': target,
            'selected_surfaces': [item.strip() for item in selected_surfaces],
            'outside_description': outside.strip() or 'none',
        }
        return {
            **row,
            'editing_instruction': '',
            'new_instruction': '',
            'visual_grounding': grounding,
            'planning_policy': VERSION,
        }, 'accepted'
    dependency = value.get('dependency_class')
    dependency_classes = {
        'none', 'independent', 'held_attached', 'supported_by_target',
        'structural_support', 'continuous_owner',
    }
    if dependency not in dependency_classes:
        return None, 'missing_or_invalid_dependency_class'
    hard_conflicts = {
        'remove': {'held_attached', 'supported_by_target', 'structural_support', 'continuous_owner'},
        'replace': {'held_attached', 'supported_by_target', 'structural_support', 'continuous_owner'},
        'add': set(),
        'attribute': set(),
    }
    if dependency in hard_conflicts[row['task_type']]:
        return None, 'contradictory_dependency_accept'
    outside_dependency = explicit_outside_dependency(
        row.get('visual_grounding', {})
    )
    if outside_dependency in hard_conflicts[row['task_type']]:
        return None, 'policy_rejected_explicit_outside_dependency'
    instruction=value.get('editing_instruction');target=value.get('target_description')
    if not isinstance(instruction,str) or not isinstance(target,str): return None,'missing_text'
    # The command has the hard brevity limit. Do not discard an otherwise valid
    # short command solely because its unambiguous target phrase exceeds 18 words.
    if not instruction.isascii() or not 3<=len(instruction.split())<=32 or not 1<=len(target.split())<=32:
        return None,'invalid_text_length_or_language'
    # A real face/medical mask is scene content, not an annotation reference.
    annotation_text=re.sub(r'\b(?:face|surgical|medical|protective|dust|respirator|costume|purple) masks?\b','face covering',instruction,flags=re.I)
    if re.search(r'\b(?:dataset|binary|segmentation|target|provided|supplied|outlined?) masks?\b|\bmasks?(?:ed)?\s+(?:region|area|pixels?|boundary)\b|\b(?:in|inside|within)\s+the\s+masks?\b|\b(?:overlay|outline|outlined|crop|coordinates?|bbox|annotation)\b',annotation_text,re.I):
        return None,'annotation_language_in_instruction'
    if instruction_uses_ambiguous_member_reference(instruction):
        return None, 'instruction_uses_ambiguous_member_reference'
    if row['task_type']=='attribute':
        # Whole-person color changes are ambiguous and frequently recolor skin,
        # hair and clothing together. Require the command to name the surface.
        if re.search(
            r'^\s*change\s+(?:(?:the\s+)?colou?r\s+of\s+)?(?:the\s+)?'
            r'(?:person|man|woman|boy|girl|child|animal|dog|cat|horse|elephant)\b'
            r"(?!['’]s\b)",
            instruction,
            re.I,
        ):
            return None,'attribute_missing_specific_surface'
        grounding_text=' '.join([
            str(row.get('visual_grounding',{}).get('target_description','')),
            str(row.get('visual_grounding',{}).get('reason','')),
            ' '.join(row.get('visual_grounding',{}).get('selected_surfaces',[])),
        ])
        if (multicolored_or_patterned(grounding_text)
                and re.search(r'\b(?:solid|plain|unpatterned)\b',instruction,re.I)):
            return None,'attribute_would_erase_visible_pattern'
        if (multicolored_or_patterned(grounding_text)
                and attribute_recolors_whole_pattern(instruction, grounding_text)):
            return None, 'attribute_recolors_whole_pattern'
        if (textured_or_weathered(grounding_text)
                and re.search(r'\b(?:uniform|plain|smooth|untextured)\b', instruction, re.I)):
            return None, 'attribute_would_erase_visible_texture'
        if attribute_color_change_is_low_contrast(
                instruction, row.get('visual_grounding', {})
        ):
            return None, 'attribute_color_change_not_visually_distinct'
        if attribute_uses_implausible_living_tissue_color(
                instruction, row.get('visual_grounding', {})
        ):
            return None, 'attribute_living_tissue_color_not_scene_plausible'
        if attribute_uses_implausible_intrinsic_material_color(instruction):
            return None, 'attribute_intrinsic_material_color_not_scene_plausible'
        if (unresolved_silhouette(grounding_text)
                and not re.search(r'\bsilhouette\b', instruction, re.I)):
            return None, 'attribute_material_not_visually_grounded'
    action_ok = bool(TYPE_ACTION_PATTERNS[row['task_type']].search(instruction))
    if (row['task_type'] == 'add'
            and re.search(r'\bapply\b[^.;]{0,50}\b(?:sticker|decal|patch|label)\b',
                          instruction, re.I)):
        action_ok = True
    if not action_ok:
        return None,'wrong_action_type'
    if row['task_type']=='replace' and not re.search(r'\bwith\b',instruction,re.I): return None,'missing_replacement'
    frozen = row.get('visual_grounding', {}).get('target_description')
    if not frozen: return None,'missing_visual_grounding'
    if phrase_words(target) != phrase_words(frozen):
        return None,'target_changed_from_visual_grounding'
    target = frozen
    if edit_conflicts_with_reflection(row.get('visual_grounding', {})):
        return None, 'policy_rejected_reflection_dependency'
    if broad_permanent_scene_edit(row, target):
        return None, 'policy_rejected_broad_permanent_scene_edit'
    if unresolved_tiny_fragment(target):
        return None, 'policy_rejected_unresolved_tiny_fragment'
    if (row['task_type'] in {'remove', 'replace'}
            and partial_living_body_part_edit(target)):
        return None, 'policy_rejected_partial_living_body_part'
    if (row['task_type'] in {'remove', 'replace'}
            and partial_integrated_mechanical_assembly(target)):
        return None, 'policy_rejected_partial_integrated_mechanical_assembly'
    if (row['task_type'] in {'remove', 'replace'}
            and whole_living_target_has_visible_part_outside(
                row.get('visual_grounding', {})
            )):
        return None, 'policy_rejected_incomplete_living_owner'
    if (row['task_type'] in {'remove', 'replace'}
            and selected_external_rigging_risk(row.get('visual_grounding', {}))):
        return None, 'policy_rejected_external_rigging_risk'
    if not locator_is_preserved(target,instruction):
        return None,'instruction_lost_distinctive_target_locator'
    if (row['task_type']=='add'
            and not addition_site_is_grounded(instruction, row.get('visual_grounding', {}))):
        return None, 'addition_site_not_grounded_inside_mask'
    if (row['task_type']=='add'
            and addition_uses_ambiguous_body_side(instruction)):
        return None, 'addition_uses_ambiguous_anatomical_side'
    if row['task_type']=='add' and addition_is_oversized(instruction):
        return None, 'addition_is_not_fine_grained'
    if row['task_type']=='add' and addition_is_microscopic(instruction):
        return None, 'addition_is_microscopic_at_full_image_scale'
    if (row['task_type']=='add'
            and addition_surface_detail_is_unjudgeable(row, instruction)):
        return None, 'addition_surface_detail_unjudgeable_at_full_image_scale'
    if (row['task_type']=='add'
            and not addition_site_is_sufficiently_local(
                instruction, row.get('visual_grounding', {})
            )):
        return None, 'addition_site_not_locally_specified'
    if (row['task_type']=='add'
            and addition_uses_underspecified_repeated_site(
                instruction, row.get('visual_grounding', {})
            )):
        return None, 'addition_site_not_unique_among_repeated_parts'
    if (row['task_type']=='add'
            and addition_host_is_broad_natural_background(
                row.get('visual_grounding', {})
            )):
        return None, 'addition_host_is_broad_natural_background'
    if (row['task_type']=='add'
            and not addition_reason_has_novelty_check(value.get('reason', ''))):
        return None, 'addition_missing_visual_novelty_check'
    if (row['task_type']=='add'
            and addition_reason_admits_existing_item(
                instruction,value.get('reason','')
            )):
        return None, 'addition_item_already_visible_in_source'
    if (row['task_type']=='add'
            and addition_requires_unsupported_material_claim(
                instruction,row.get('visual_grounding',{})
            )):
        return None, 'addition_attachment_material_not_grounded'
    if (row['task_type']=='add'
            and addition_places_loose_item_on_bare_body(
                instruction, row.get('visual_grounding', {})
            )):
        return None, 'addition_places_loose_item_on_bare_body'
    if (row['task_type']=='add'
            and addition_places_unfastened_item_on_fur(instruction)):
        return None, 'addition_places_unfastened_item_on_fur'
    if (row['task_type']=='add'
            and addition_uses_unstable_narrow_support(instruction)):
        return None, 'addition_uses_unstable_narrow_support'
    if (row['task_type']=='add'
            and addition_requires_pixels_outside_mask(instruction)):
        return None, 'addition_requires_pixels_outside_mask'
    if (row['task_type']=='add'
            and addition_conflicts_with_excluded_contact(
                instruction, row.get('visual_grounding', {})
            )):
        return None, 'addition_conflicts_with_excluded_contact'
    if row['task_type']=='replace':
        replacement=replacement_text_from_instruction(instruction)
        if re.fullmatch(
            r'\s*(?:a|an|the)?\s*(?:bright|dark|light|deep|pale)?\s*'
            r'(?:red|blue|green|yellow|orange|purple|pink|brown|black|white|grey|gray|teal)\s*'
            r'(?:one|version|model)\s*[.!]?\s*',
            replacement,
        ):
            return None,'replace_target_is_only_a_color_variant'
        if replacement_is_merely_surface_variant(target, replacement):
            return None, 'replace_keeps_same_object_category'
        mask_area = None
        if isinstance(row.get('mask'), dict) and {'size', 'counts'} <= set(row['mask']):
            mask_area = normalized_area(row['mask'])
        if replacement_has_incompatible_footprint(target, replacement, mask_area):
            return None, 'replacement_has_incompatible_footprint'
    if row['task_type']=='attribute':
        if attribute_targets_unresolved_background(
                row.get('visual_grounding', {})):
            return None, 'attribute_target_is_unresolved_background'
        instruction_words=phrase_words(instruction)
        target_words=phrase_words(target)
        target_start=instruction_words.find(target_words)
        prefix=instruction_words[:target_start] if target_start >= 0 else ''
        stop_words={
            'change','set','make','turn','color','colour','the','a','an','of','on',
            'in','at','worn','by','to','into','bright','dark','light','deep','pale',
            'red','blue','green','yellow','orange','purple','pink','brown','black',
            'white','grey','gray','teal','gold','silver','solid','plain',
        }
        surface_terms={word.rstrip('s') for word in prefix.split()
                       if len(word)>=3 and word not in stop_words}
        grounding_words={word.rstrip('s') for word in phrase_words(' '.join([
            target,
            str(row.get('visual_grounding',{}).get('reason','')),
            ' '.join(row.get('visual_grounding',{}).get('selected_surfaces',[])),
        ])).split()}
        if surface_terms and not surface_terms & grounding_words:
            return None,'attribute_surface_not_grounded_inside_mask'
    return {**row,'editing_instruction':instruction,'new_instruction':instruction,
            'refer_object':[target], 'masked_content':target,'segmentation_target':target,
            'mask_refinement':'original','structural_parts':[],
            'mask_compatibility':'compatible','compatibility_reason':value['reason'],
            'planning_policy':VERSION,
            'planning_revision':{
                'version':VERSION,'decision':'accept','reason':value['reason'],
                'dependency_class':dependency,
                'original_instruction':row.get('editing_instruction',''),
                'verification':'mllm_grounded_source_plan'}},'accepted'


def fixed_region(row, source_size, protected):
    """Retain the exact dataset RLE, without semantic filtering or refinement."""
    encoded=coco_mask.encode(np.asfortranarray(protected.astype(np.uint8)))
    encoded['counts']=encoded['counts'].decode('ascii')
    return {**row, 'sam_target_mask':row['mask'],
            'region_contract':{'version':VERSION,'status':'original','policy':'original',
                'provenance':'dataset_mask_no_source_sam','source_sam_called':False,
                'verification':'mllm_only_no_semantic_segmentation','source_size':list(source_size),
                'segmentation_target':row['segmentation_target'],'protected_mask':encoded,
                'original_instruction':row['editing_instruction'],'original_task_type':row['task_type']}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True,help='Parent of plan/, scope/ and regions/')
    p.add_argument('--ids',default='')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--attempts',type=int,default=2)
    p.add_argument('--model-id',default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    original=read(a.data_root/'annotations.jsonl');rows=original
    if a.ids:
        ids={int(x) for x in a.ids.split(',')};rows=[r for r in rows if int(r['image'].split('_')[0]) in ids]
        if len(rows)!=len(ids):raise ValueError('Missing or duplicate IDs')
    jobs=[]
    for row in rows:
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        mask=mask_array(source.size,row['mask'])
        jobs.append(dict(row={**row,'mask_geometry_hint':mask_geometry_hint(mask),
                              'mask_topology_hint':topology_hint(mask),
                              'mask_component_hint':component_hint(mask, source),
                              'mask_hole_hint':enclosed_hole_hint(mask)},source=source,
                         crop=instruction_target_crop(source,mask,target_pointer=True)))
    started=time.perf_counter()
    vlm.configure_backend('qwen38-vllm',model_id=a.model_id,device='cuda:0',dtype='bf16')
    backend=vlm.get_backend();load=time.perf_counter()-started;stats={}
    try:
        for stage, directory in [('ground','plan'),('scope','scope')]:
            root=a.out_root/directory;root.mkdir(exist_ok=False);(root/'inputs').mkdir()
            (root/'sources').symlink_to((a.data_root/'sources').resolve())
            write(root/'input_annotations.jsonl',original)
            for j in jobs:j['crop'].save(root/'inputs'/j['row']['image'])
            accepted=[];responses=[];pending=jobs;inference=0.;stage_start=time.perf_counter()
            for attempt in range(1,a.attempts+1):
                again=[]
                for offset in range(0,len(pending),a.batch_size):
                    batch=pending[offset:offset+a.batch_size]
                    messages=[[{'role':'user','content':[{'type':'image','image':j['source']},
                        {'type':'image','image':j['crop']},{'type':'text','text':prompt_for(j['row'],stage,j.get('feedback',''))}]}] for j in batch]
                    t=time.perf_counter();outputs=backend.chat_batch(messages,max_new_tokens=384);inference+=time.perf_counter()-t
                    if len(outputs)!=len(batch):raise ValueError('Incomplete backend batch')
                    for job,raw,message in zip(batch,outputs,messages):
                        parsed=parse_json_object(raw);updated,status=validate(parsed,job['row'],stage)
                        record=dict(image=job['row']['image'],stage=stage,attempt=attempt,raw_response=raw,
                                    parsed=parsed,status=status,prompt=message[0]['content'][-1]['text'])
                        responses.append(record)
                        if updated is not None:accepted.append({**job,'row':updated,'feedback':''})
                        elif (status not in {'model_rejected', 'contradictory_dependency_accept'}
                              and not status.startswith('policy_rejected_')):
                            if status=='annotation_language_in_instruction':
                                help_text=(' If the target is a real wearable mask, call it a face covering. '
                                           'Otherwise remove references to annotation masks.')
                            elif status=='attribute_surface_not_grounded_inside_mask':
                                help_text=(' Name the edited surface with factual words already present in the '
                                           'frozen target or visual evidence summary; do not introduce a new '
                                           'surface or a synonym that cannot be checked against that evidence.')
                            elif status=='instruction_lost_distinctive_target_locator':
                                help_text=(' Preserve the frozen target\'s distinctive category and full-photo '
                                           'locator, using natural grammar without changing the referent.')
                            elif status=='instruction_uses_ambiguous_member_reference':
                                help_text=(' Identify one exact visible site with a concise position or neighbour '
                                           'relation; never leave the editor to choose one of repeated parts.')
                            elif status=='replace_target_is_only_a_color_variant':
                                help_text=(' Name a concrete replacement instance with a visibly different '
                                           'identity or form, not only a color plus a vague pronoun.')
                            elif status=='replace_keeps_same_object_category':
                                help_text=(' Choose a replacement with a genuinely different object category '
                                           'or recognizable identity; move a mere subtype, material, style, '
                                           'or color change to an attribute task instead.')
                            elif status=='addition_site_not_grounded_inside_mask':
                                help_text=(' The claimed attachment feature is absent from the frozen visual '
                                           'evidence. Use only an actually visible selected surface or reject.')
                            elif status=='addition_uses_ambiguous_anatomical_side':
                                help_text=(' Anatomical left/right is ambiguous in an image. Use a visible action '
                                           'or image-relative locator at an actually visible selected site.')
                            elif status=='addition_is_not_fine_grained':
                                help_text=(' The proposed addition is too large for this fine-grained corpus. '
                                           'Choose a compact, clearly visible absent item at one exact selected site.')
                            elif status=='addition_is_microscopic_at_full_image_scale':
                                help_text=(' The proposed item is too small to judge in the full image. Choose a '
                                           'clearly visible addition of meaningful extent on the same host.')
                            elif status=='addition_surface_detail_unjudgeable_at_full_image_scale':
                                help_text=(' The selected host is already tiny in the full photo, so a smaller '
                                           'surface detail would not be judgeable. Choose a meaningful addition '
                                           'that occupies more of the selected extent, or reject.')
                            elif status=='addition_site_not_locally_specified':
                                help_text=(' The selected host is broad. Name one visible sub-location or nearby '
                                           'landmark so the placement is locally determined without annotations.')
                            elif status=='addition_site_not_unique_among_repeated_parts':
                                help_text=(' The placement leaves several repeated local parts as candidates. '
                                           'Use a stable neighbour relation or another uniqueness cue that selects '
                                           'one exact visible site after annotations are hidden.')
                            elif status=='addition_host_is_broad_natural_background':
                                help_text=(' The selected mask is a dense scene-wide natural background rather '
                                           'than one uniquely grounded local support. Reject this add plan; '
                                           'positional adjectives cannot make that host fine-grained.')
                            elif status=='addition_missing_visual_novelty_check':
                                help_text=(' Reinspect the entire clean photo. Begin the reason with "Novelty '
                                           'check:" and compare the proposal with its closest visible lookalike; '
                                           'choose another item or reject if they are not visibly distinct.')
                            elif status=='addition_item_already_visible_in_source':
                                help_text=(' Your own novelty reason says that the proposed item is already '
                                           'visible elsewhere in the clean photo. Choose a genuinely absent, '
                                           'visually distinct item; adding another instance is invalid.')
                            elif status=='addition_attachment_material_not_grounded':
                                help_text=(' The proposed attachment mechanism requires a host material that is '
                                           'not visually established. Use an ordinary attachment supported by '
                                           'the frozen evidence, choose another item, or reject.')
                            elif status=='addition_places_loose_item_on_bare_body':
                                help_text=(' Do not balance a loose prop on bare skin. Choose a physically '
                                           'credible, normally fastened placement on another visibly selected '
                                           'surface, or reject if none exists.')
                            elif status=='addition_places_unfastened_item_on_fur':
                                help_text=(' A loose decoration cannot simply adhere to fur. Use a visibly '
                                           'credible ordinary attachment method at a selected site, or choose '
                                           'another feasible addition.')
                            elif status=='addition_uses_unstable_narrow_support':
                                help_text=(' The proposed loose item is not stably supported by that narrow '
                                           'surface. Choose a smaller normally attached addition at another '
                                           'visible selected site, or reject.')
                            elif status=='addition_requires_pixels_outside_mask':
                                help_text=(' The complete new-item silhouette would cross into excluded pixels. '
                                           'Choose a surface-applied or closely overlapping addition whose '
                                           'visible body fits inside the selected region, or reject.')
                            elif status=='addition_conflicts_with_excluded_contact':
                                help_text=(' The proposed attachment site is already occupied by excluded '
                                           'content. Choose another visibly free selected site, or reject.')
                            elif status=='replacement_has_incompatible_footprint':
                                help_text=(' The replacement requires a substantially different footprint. Choose '
                                           'a visibly different object that fits the selected target extent.')
                            elif status=='attribute_material_not_visually_grounded':
                                help_text=(' The pixels establish only a silhouette, not the claimed material. '
                                           'Request a visible silhouette/appearance property or reject.')
                            elif status=='attribute_would_erase_visible_pattern':
                                help_text=(' Preserve the visible pattern. Change one existing pattern color '
                                           'without making the surface solid, or edit another clearly visible '
                                           'unpatterned selected surface.')
                            elif status=='attribute_recolors_whole_pattern':
                                help_text=(' The selected surface is visibly multicolored or patterned. Name one '
                                           'specific visible constituent color, base, or marking to change rather '
                                           'than recoloring the whole patterned surface.')
                            elif status=='attribute_would_erase_visible_texture':
                                help_text=(' Preserve the visibly weathered or mottled texture. Change one '
                                           'property without requesting a uniform, plain, or smooth result.')
                            elif status=='attribute_color_change_not_visually_distinct':
                                help_text=(' Choose a target color with obvious contrast from the visibly dark '
                                           'source surface while keeping the same selected surface.')
                            elif status=='attribute_living_tissue_color_not_scene_plausible':
                                help_text=(' Keep ordinary visible skin plausible for the photographic scene. '
                                           'Choose another clearly evidenced property or a natural-looking value.')
                            elif status=='attribute_intrinsic_material_color_not_scene_plausible':
                                help_text=(' The requested color contradicts the named material itself. Choose a '
                                           'scene-plausible property change on the same selected surface.')
                            elif status in {
                                'marker_observations_not_numbered_in_order',
                                'marker_explicitly_outside_claimed_target',
                                'marker_observation_incoherent_with_target',
                                'duplicate_marker_observations',
                                'marker_observation_matches_outside_item',
                                'grounding_admits_foreign_selected_fragment',
                                'held_target_not_sampled_by_marker',
                                'associated_surface_not_sampled_by_marker',
                                'grounding_promotes_unsampled_outside_noun',
                            }:
                                help_text=(' Account for every numbered blue dot in exact numeric order. If a '
                                           'dot lies on a different semantic object, decision must be reject; '
                                           'never reinterpret it as an ignorable extension. Do not include a '
                                           'held, supported, or attached object unless a numbered dot visibly '
                                           'samples its own pixels.')
                            else:
                                help_text=''
                            keep = (' Keep the same edit type and frozen visual target.' if stage == 'scope'
                                    else ' Re-read only the selected source pixels; do not propose an edit.')
                            again.append({**job,'feedback':
                                '\nCorrect this output error: '+status+'.'+help_text+
                                keep+' Previous answer: '+raw})
                        print(json.dumps({k:v for k,v in record.items() if k!='prompt'},ensure_ascii=False),flush=True)
                    write(root/'responses.jsonl',responses)
                pending=again
                if not pending:break
            accepted.sort(key=lambda j:j['row']['image'])
            write(root/'annotations.jsonl',[j['row'] for j in accepted])
            stats[stage]=dict(input_cases=len(jobs),accepted=len(accepted),calls=len(responses),inference_seconds=inference,
                wall_seconds=time.perf_counter()-stage_start,version=VERSION)
            (root/'summary.json').write_text(json.dumps(stats[stage],indent=2));jobs=accepted
    finally:vlm.shutdown_backend()
    root=a.out_root/'regions';root.mkdir(exist_ok=False);(root/'sources').symlink_to((a.data_root/'sources').resolve())
    regions=[]
    for j in jobs:
        row=j['row'];target=mask_array(j['source'].size,row['mask']).astype(bool)
        protect=protected_neighbors(a.data_root,row,j['source'].size)&~target
        regions.append(fixed_region(row,j['source'].size,protect))
    write(root/'annotations.jsonl',regions);write(root/'input_annotations.jsonl',original)
    (root/'summary.json').write_text(json.dumps(dict(cases=len(regions),unresolved=0,source_sam_calls=0,mask_policy='original_dataset'),indent=2))
    summary=dict(version=VERSION,input_cases=len(rows),accepted=len(regions),load_seconds=load,stages=stats,
                 source_sam_calls=0,wall_seconds=time.perf_counter()-started)
    (a.out_root/'dataset_planning_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
