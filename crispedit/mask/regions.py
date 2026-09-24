"""Local segmentation geometry and topology-aware noise cleanup."""

import cv2
import numpy as np
import re


def context_crop(box, shape, margin=0.25):
    height, width = shape
    x1, y1, x2, y2 = box
    dx, dy = max(8, (x2-x1)*margin), max(8, (y2-y1)*margin)
    return (max(0, int(np.floor(x1-dx))), max(0, int(np.floor(y1-dy))),
            min(width, int(np.ceil(x2+dx))), min(height, int(np.ceil(y2+dy))))


def clean_mask(mask, sparse=False):
    """Keep multiple real parts; remove only scale-relative isolated noise.

    Sparse groups must not be reduced to their largest component. Dense
    objects may have legitimate disconnected portions due to occlusion.
    """
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary, {"components_before": 0, "components_after": 0, "removed_pixels": 0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = 2 if sparse else max(3, int(areas.max()*0.005))
    keep = np.r_[False, areas >= threshold]
    result = keep[labels].astype(np.uint8)
    # Only small enclosed holes in dense objects, never gaps between parts.
    if not sparse and result.any():
        holes, hole_labels, hole_stats, _ = cv2.connectedComponentsWithStats(1-result, connectivity=8)
        limit = max(2, int(result.sum()*0.001))
        for index in range(1, holes):
            x, y, width, height, area = hole_stats[index]
            if x > 0 and y > 0 and x+width < result.shape[1] and y+height < result.shape[0] and area <= limit:
                result[hole_labels == index] = 1
    return result, {"components_before": count-1,
                    "components_after": int(cv2.connectedComponents(result, connectivity=8)[0]-1),
                    "removed_pixels": int(((binary > 0) & (result == 0)).sum())}


def sparse_degenerate(mask, box):
    """Reject a sparse detail prediction that has become a solid enclosing object."""
    x1, y1, x2, y2 = box
    box_area = max(1.0, (x2-x1)*(y2-y1))
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return True
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(mask.sum())/box_area > 0.35 or largest/box_area > 0.20


def fine_particle_degenerate(mask, box, ref):
    """Particle safety must also hold when the MLLM splits a global cloud into boxes."""
    if not re.search(r"\b(?:confetti|glitter|sparkles?|snowflakes?)\b", str(ref), re.I):
        return False
    x1, y1, x2, y2 = box
    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return count > 1 and stats[1:, cv2.CC_STAT_AREA].max()/max(1, (x2-x1)*(y2-y1)) > 0.04
