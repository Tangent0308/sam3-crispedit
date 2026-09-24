"""Fill internal omissions only when independent attachment masks support them."""

import re

import cv2
import numpy as np

from crispedit.mask.candidates import mask_iou


def attachment_phrase(ref):
    parts = re.split(r'\bwith\b', ref, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        # Complete seats include cushions. This only proposes a concept query;
        # absent/unsupported cushions add no pixels and real openings survive.
        return 'cushions' if re.search(r'\b(?:chairs?|armchairs?|sofas?|couches)\s*$',ref,re.I) else ''
    phrase = parts[1].strip(' .;')
    return phrase if 1 <= len(phrase.split()) <= 8 else ''


def silhouette_and_holes(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    silhouette = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(silhouette, contours, -1, 1, cv2.FILLED)
    return silhouette, silhouette & (mask == 0)


def complete_supported_holes(mask, text, joint, *, envelope=None):
    """Do not fill a hole merely because it exists or expand the outer contour."""
    audit = {'added_pixels': 0, 'reason': 'ATTACHMENT_UNSUPPORTED'}
    if text is None or joint is None:
        return mask, audit
    agreement = mask_iou(text, joint)
    audit['agreement'] = agreement
    common = (text > 0) & (joint > 0)
    area = int(common.sum())
    containment = area/max(min(int((text > 0).sum()), int((joint > 0).sum())), 1)
    audit['smaller_proposal_supported'] = containment
    # Joint prompting can add the enclosing chair to valid cushion instances.
    # Use only their intersection, never those additional enclosing pixels.
    if agreement < .70 and containment < .95:
        # A group query can return different *extra* cushions in the two modes.
        # Match coherent components, not the global union; keep only pixels
        # that both prompts actually support.
        supported = np.zeros_like(common)
        for proposal, other in ((text, joint),(joint,text)):
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                (proposal > 0).astype(np.uint8),connectivity=8)
            for identity in range(1,count):
                component = labels == identity
                component_area = int(stats[identity,cv2.CC_STAT_AREA])
                if component_area >= 16 and (component & (other > 0)).sum()/component_area >= .95:
                    supported |= component & common
        if not supported.any():
            return mask, audit
        common = supported
        area = int(common.sum())
        audit['component_matching'] = True
    silhouette, holes = silhouette_and_holes(mask)
    if envelope is not None:
        silhouette = (envelope > 0).astype(np.uint8)
        holes = silhouette & (mask == 0)
    inside = float((common & (silhouette > 0)).sum()/max(area, 1))
    # An attachment must not degenerate into the enclosing object/background.
    # Contents such as cutlery may cross the rim. With an explicit container
    # envelope, retain only the supported pixels INSIDE it; never expand it.
    min_inside = .50 if envelope is not None else .90
    audit['inside_silhouette'] = inside
    if area < 8 or inside < min_inside or area > .70*int(silhouette.sum()):
        audit['reason'] = 'ATTACHMENT_GEOMETRY_REJECTED'
        return mask, audit
    addition = common & (holes > 0)
    audit.update(added_pixels=int(addition.sum()), inside_silhouette=inside,
                 reason='ATTACHMENT_SUPPORTED_HOLES')
    return (mask | addition).astype(np.uint8), audit


def container_contents(ref):
    """Only call after validating an explicit whole-container source ref."""
    parts = re.split(r'\s+(?:filled with|with|containing|of)\s+', ref, maxsplit=1, flags=re.I)
    if len(parts) != 2:
        return []
    phrases = [p.strip(' ,.;') for p in re.split(r'\s+and\s+|,', parts[1])]
    return [p for p in phrases if 1 <= len(p.split()) <= 6][:3]


def container_envelope(mask):
    """Limit content proposals spatially; this envelope is never a final mask."""
    if int(mask.sum()) < 100:
        return None
    hull = np.zeros_like(mask)
    cv2.fillConvexPoly(hull, cv2.convexHull(cv2.findNonZero(mask.astype(np.uint8))), 1)
    fill = mask.sum() / max(hull.sum(), 1)
    return hull if .45 <= fill < .94 else None


def close_content_seams(mask, supported_content, envelope):
    """Close pixel-scale seams beside recovered contents, not arbitrary holes."""
    points = cv2.findNonZero(envelope.astype(np.uint8))
    if points is None or not supported_content.any():
        return mask, 0
    _, _, width, height = cv2.boundingRect(points)
    radius = max(1, min(3, round(min(width, height) * .01)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    nearby = cv2.dilate(supported_content.astype(np.uint8), kernel)
    addition = (closed > 0) & (mask == 0) & (nearby > 0) & (envelope > 0)
    area = int(addition.sum())
    if area > .05 * int(mask.sum()):
        return mask, 0
    return (mask | addition).astype(np.uint8), area
