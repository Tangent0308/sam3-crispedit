"""Construction-only contact geometry and calibrated visual audit requirements.

These fields never enter published ref/no-ref strings or change source masks.
"""
from __future__ import annotations

import numpy as np

VERSION = 'contact-core-quality-1'

PLAN_RULES = """EXECUTION FIT, in this same planning response:
For add, return contact_xy=[x,y] in normalized ORIGINAL FULL image coordinates, on the exact visible contact site within WHITE selected pixels. This point must match placement.site and preserve occupied content. The entire added object must fit the available editable space at a plausible scale, without crossing protected targets. Choose its attachment according to physical support and the scene, not a universal preferred shape. For non-add use contact_xy=[]. This is internal metadata and never part of the published instruction.
Consider what remains after every edit. Removing the selected entity must not strand excluded loads or attachments. Preserve required neighboring structure and unrequested contents. Attribute edits preserve instance count, construction and unrequested optical properties. Context outside the selected region may explain physical dependencies but cannot be erased to hide an unsuitable plan.
Use a truthful entity noun supported by visible evidence rather than guessing a technical identity. Distinguish repeated instances by a verified local relation in the original scene. A brief unambiguous reference is preferable to a shorter ambiguous one.
"""

AUDIT_RULES = '''QUALITY THRESHOLD: basic usable photo editing, not pixel-perfect restoration.
PASS with warnings for modest edge softness/halos, texture/reflection variation, a slightly different plausible replacement shape, source blur, minor local shadows, or unconventional but coherent taste. Do not invent requested precision or require unselected/occluded pixels to change. A plausible tint can change highlights; visible reflections are not proof of opacity. Judge at the source photo's effective resolution.
FAIL serious visible defects: wrong object/contact site, a missing edit, a substantially clipped new item, changed instance count for an attribute edit, erased old contents beyond necessary occlusion, floating loads/remnants, broken anatomy/support, or changes to unrelated targets. Count source and result instances explicitly for material/attribute changes; metallic highlights must not hide additional rods or duplicate parts. Trace an added item's COMPLETE silhouette and its actual contact, including the lower edge: a tag sliced by a straight support boundary is not a complete tag. Compare original logos/markings before accepting a new decoration. A shadow must still transmit the surface texture, not become a flat painted shape.
Return serious_defects as a list of short concrete visible contradictions, empty when none; put only nonfatal imperfections in warnings. Any serious_defects entry makes the result fail. Do not turn a mild warning into natural=false. Never accept a serious defect merely to improve acceptance. If the core edit is unjudgeable, fail with a specific reason; uncertainty about tiny texture details is only a warning.
'''


def contact_metadata(values, mask):
    """Validate a planner point; allow only small numerical localization error."""
    if values is None or values == []:
        return None  # Old plans remain readable; new decoder requires the field.
    if (not isinstance(values, list) or len(values) != 2
            or any(not isinstance(v, (float, int)) or isinstance(v, bool)
                   or not np.isfinite(v) or not 0 <= v <= 1 for v in values)):
        raise ValueError('invalid add contact coordinates')
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty contact host')
    h, w = mask.shape
    x, y = values[0] * (w - 1), values[1] * (h - 1)
    distances = (xs - x) ** 2 + (ys - y) ** 2
    index = int(np.argmin(distances))
    tolerance = max(5., min(48., min(np.ptp(xs) + 1, np.ptp(ys) + 1) * .15))
    distance = float(np.sqrt(distances[index]))
    if distance > tolerance:
        raise ValueError('add contact point lies outside the selected host')
    return {'version': VERSION, 'point': [int(xs[index]), int(ys[index])],
            'predicted_xy': values, 'snap_distance': distance}


def contact_support(mask, contact):
    """A bounded output envelope around contact, distinct from the source site.

    Thin hosts need surrounding room for the new item. Other source selectors
    are still subtracted by preflight; this function never overrides protection.
    """
    mask = np.asarray(mask, bool)
    ys, xs = np.nonzero(mask)
    x, y = contact['point']
    radius = int(np.clip(max(np.sqrt(mask.sum()) * .32,
                             min(np.ptp(xs) + 1, np.ptp(ys) + 1) * .55), 24, 224))
    result = np.zeros_like(mask)
    result[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1] = True
    return result


def paste_diagnostics(source, raw, support):
    """Reject a dominant addition cut by the compositor; ignore small drift.

    A boundary-crossing component alone is only a warning. A hard gate also
    requires >=256 retained pixels and >=50% of all visible change belonging
    to that component. This avoids vetoing tiny background/texture fragments.
    """
    from scipy.ndimage import binary_opening, label
    delta = np.max(np.abs(np.asarray(source, dtype=np.int16) -
                          np.asarray(raw, dtype=np.int16)), axis=2) > 32
    delta = binary_opening(delta, iterations=1)
    labels, count = label(delta)
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    inside = np.bincount(labels[np.asarray(support, bool)].ravel(), minlength=count+1)
    candidates = [i for i in range(1, count+1) if inside[i] >= 48 and sizes[i] >= 96
                  and (sizes[i]-inside[i]) / sizes[i] >= .25]
    total_inside = int(np.count_nonzero(delta & support))
    dominant = [i for i in candidates if inside[i] >= 256
                and inside[i] >= .5 * max(1, total_inside)]
    return {'version': VERSION, 'cross_boundary_components': len(candidates),
            'max_discarded_fraction': max((float(1-inside[i]/sizes[i]) for i in candidates), default=0.),
            'dominant_change_clipped': bool(dominant), 'warning_only': not bool(dominant)}
