"""Select existing SAM proposals; never fabricate pixels to improve topology.

PCS scores measure concept confidence, not mask IoU. Shape is only a warning
signal: thin/sparse targets keep their existing policy and disconnected objects
are not penalized merely for having multiple components.
"""

import re

import cv2
import numpy as np


_THIN = re.compile(
    r"\b(?:eyes?|eyelids?|eyebrows?|lashes|mouth|smile|lips?|wires?|strings?|"
    r"lights?|chains?|ribbons?|confetti|petals?|flowers?|branches?|tentacles?|"
    r"fences?|nets?|mesh|lace|glasses|earrings?|necklaces?|piercings?|studs?|"
    r"beads?|spots?|dots?|ladders?|lines?|outlines?)\b", re.I
)


def use_object_selection(ref, region_mode, density):
    return region_mode == "object" and density != "sparse" and not _THIN.search(ref)


def mask_iou(a, b):
    return float(np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1))


def topology(mask):
    mask = (mask > 0).astype(np.uint8)
    area = int(mask.sum())
    if not area:
        return dict(area=0, holes=1., perforation=1., hull_fill=0., interior=0., largest=0.)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    envelope = np.zeros_like(mask)
    cv2.drawContours(envelope, contours, -1, 1, cv2.FILLED)
    _, _, hole_stats, _ = cv2.connectedComponentsWithStats(envelope-mask, connectivity=8)
    hole_areas = hole_stats[1:, cv2.CC_STAT_AREA]
    # A hand occluding a shirt, or the opening between chair arms, is a real
    # large hole. Only dispersed small holes are evidence of perforation.
    perforation = float(hole_areas[hole_areas < area*.01].sum()/area)
    hull = np.zeros_like(mask)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(mask)), 1)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return dict(area=area, holes=float((envelope.sum()-area)/max(envelope.sum(), 1)),
                perforation=perforation,
                hull_fill=float(area/max(hull.sum(), 1)),
                interior=float(cv2.erode(mask, np.ones((3, 3), np.uint8)).sum()/area),
                largest=float(stats[1:, cv2.CC_STAT_AREA].max()/area))


def _defect(f):
    # Fixed-crop alternatives have the same scale. This is not an accuracy score.
    return 2*f['perforation'] + max(0., .94-f['interior']) + max(0., .25-f['hull_fill'])


def _fragmented(f):
    return f['area'] >= 100 and ((f['hull_fill'] < .40 and f['interior'] < .80)
                                 or f['interior'] < .35)


def select_object_candidate(pvs_candidates, text, joint, allow_extent_recovery=False):
    """Return (mask, metadata, source, audit), or None if nothing is available.

    Each input is (mask, metadata). PVS inputs MUST already pass the model-IoU,
    bounding-box and containment gates. PCS inputs pass semantic/containment
    gates. Never compare PCS confidence numerically with PVS predicted IoU.
    """
    pool = {f'pvs_{i}': (mask, meta, 'pvs') for i, (mask, meta) in enumerate(pvs_candidates)}
    for name, value in [('text', text), ('joint', joint)]:
        if value[0] is not None:
            pool[name] = (*value, 'pcs')
    if not pool:
        return None
    features = {name: topology(value[0]) for name, value in pool.items()}
    semantic = [n for n in ('text', 'joint') if n in pool]
    extent_degenerate = False
    reason = 'OBJECT_SEMANTIC_ONLY'
    if semantic:
        chosen = semantic[-1]
        if len(semantic) == 2:
            delta = _defect(features['joint'])-_defect(features['text'])
            if delta > .025:
                chosen, reason = 'text', 'OBJECT_TEXT_MORE_COMPLETE'
            elif delta < -.025:
                chosen, reason = 'joint', 'OBJECT_JOINT_MORE_COMPLETE'
            else:
                reason = 'OBJECT_JOINT_STABLE'
                # Independent visual support resolves ambiguous semantic masks.
                support = {n: max((mask_iou(pool[n][0], m) for m, _ in pvs_candidates), default=0.)
                           for n in semantic}
                best = max(support, key=support.get)
                if support[best] >= .65 and support[best]-support[chosen] > .20:
                    chosen, reason = best, 'OBJECT_INDEPENDENT_SUPPORT'
        visual = [n for n in pool if n.startswith('pvs_')]
        # A fragmented semantic result can be background specks despite a high
        # concept score. Use a coherent high-IoU box proposal, or flag review.
        reliable = [n for n in visual if pool[n][1].get('predicted_iou', 0) >= .85
                    and features[n]['holes'] < .03 and features[n]['hull_fill'] > .70
                    and features[n]['interior'] > .90 and features[n]['largest'] > .97]
        extent_degenerate = allow_extent_recovery and features[chosen]['hull_fill'] < .10
        extent_candidates = [n for n in visual if extent_degenerate
            and pool[n][1].get('predicted_iou',0) >= .75
            and features[n]['holes'] < .03 and features[n]['hull_fill'] > .70
            and features[n]['largest'] > .97
            and 3 < features[n]['area']/max(features[chosen]['area'],1) < 50
            and np.logical_and(pool[n][0],pool[chosen][0]).sum()/max(features[chosen]['area'],1) > .90]
        if extent_candidates:
            chosen = max(extent_candidates,key=lambda n: pool[n][1]['predicted_iou'])
            reason = 'OBJECT_RECOVER_COLLAPSED_EXTENT'
            extent_degenerate = False
        elif _fragmented(features[chosen]) and reliable:
            chosen = max(reliable, key=lambda n: pool[n][1]['predicted_iou']-_defect(features[n]))
            reason = 'OBJECT_RECOVER_FRAGMENTED_SEMANTIC'
        elif len(semantic) == 2 and abs(_defect(features['text'])-_defect(features['joint'])) > .05:
            # A smooth PVS outline may cover both a rim-only semantic mask and
            # a perforated whole wheel. Require support from BOTH PCS modes,
            # bounded expansion and a single coherent silhouette.
            complete = [n for n in reliable if features[n]['largest'] >= .995
                        and features[n]['area'] <= 1.5*features[chosen]['area']
                        and features[n]['area'] >= features[chosen]['area']*1.05
                        and all(mask_iou(pool[n][0], pool[s][0]) >= .60 and
                                np.logical_and(pool[n][0], pool[s][0]).sum()/features[s]['area'] >= .88
                                for s in semantic)]
            if complete:
                chosen = min(complete, key=lambda n: _defect(features[n]))
                reason = 'OBJECT_COMPLETE_WITH_TWO_PROMPT_SUPPORT'
    else:
        chosen = max(pool, key=lambda n: pool[n][1].get('predicted_iou', 0)-_defect(features[n]))
        reason = 'OBJECT_VISUAL_ONLY'
    mask, metadata, source = pool[chosen]
    metadata = {**metadata, 'selection_reason': reason,
                'selection_review': bool(_fragmented(features[chosen]) or extent_degenerate),
                'pcs_fusion': chosen if source == 'pcs' else ''}
    if source == 'pvs':
        metadata.update(selected_count=1, candidate_count=len(pvs_candidates))
    audit = {'selected': chosen, 'reason': reason, 'topology': features,
             'review': metadata['selection_review']}
    return mask, metadata, source, audit
