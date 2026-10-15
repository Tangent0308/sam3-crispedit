"""Geometry and candidate-pair gates for grouped SA-1B data."""
from __future__ import annotations

from itertools import combinations
import math
import os
import re
from typing import Any

import numpy as np

from .geometry import load_mask, mask_geometry


BANDED_CATEGORIES = {
    "stone", "stones", "leaf", "leaves", "brick", "bricks", "tile", "tiles",
    "grass", "grasses", "pebble", "pebbles", "sand", "gravel", "hair", "fur",
    "boulder", "boulders", "rock", "rocks",
}
BACKGROUND_CATEGORIES = {
    "sky", "ground", "floor", "wall", "road", "water", "background", "snow",
    "walls", "embankment", "embankments", "riverbank", "riverbanks", "terrain", "pavement",
}


def precise_point(mask: np.ndarray) -> list[int]:
    """Return the foreground pixel furthest from the mask boundary.

    The benchmark point is an interior selector, not a bbox-centre heuristic.
    ``distance_transform_edt`` is exact on the pixel grid and deterministic;
    SciPy is required: choosing distance from the image border is not a valid
    substitute for distance from the selector boundary.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("point requires a non-empty 2D mask")
    try:
        from scipy.ndimage import distance_transform_edt
        distances = distance_transform_edt(np.pad(mask, 1, constant_values=False))[1:-1, 1:-1]
        best = np.flatnonzero(distances == distances.max())
        index = int(best[0])
        y, x = np.unravel_index(index, mask.shape)
        return [int(x), int(y)]
    except ImportError as exc:
        raise RuntimeError("precise_point requires scipy; install it in the planning runtime") from exc



def support_mask(mask: np.ndarray, operation: str, policy: str | None = None) -> np.ndarray:
    """Construct the same bounded writable region used by grouped generation."""
    from .editor import region_support_mask
    return region_support_mask(mask, operation, policy)


def gap_pixels(a: np.ndarray, b: np.ndarray) -> float:
    """Minimum Euclidean foreground-to-foreground gap in pixels."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape or not a.any() or not b.any():
        raise ValueError("aligned non-empty masks required")
    # A few SA-1B masks span most of a 4K canvas.  A full distance transform
    # for every candidate pair is needlessly expensive during selection.  Max
    # pooling preserves every foreground component and gives a deterministic
    # conservative proximity estimate; small crops remain pixel-exact.
    max_pixels = max(262_144, int(os.environ.get("SA1B_GAP_MAX_PIXELS", "1500000")))
    if a.size > max_pixels:
        step = int(math.ceil(math.sqrt(a.size / max_pixels)))
        height, width = a.shape
        pooled_height = int(math.ceil(height / step))
        pooled_width = int(math.ceil(width / step))
        pad_height = pooled_height * step - height
        pad_width = pooled_width * step - width
        if pad_height or pad_width:
            a = np.pad(a, ((0, pad_height), (0, pad_width)), constant_values=False)
            b = np.pad(b, ((0, pad_height), (0, pad_width)), constant_values=False)
        a = a.reshape(pooled_height, step, pooled_width, step).any(axis=(1, 3))
        b = b.reshape(pooled_height, step, pooled_width, step).any(axis=(1, 3))
        if not a.any() or not b.any():
            raise ValueError("non-empty masks disappeared during pooling")
        return gap_pixels(a, b) * step
    try:
        from scipy.ndimage import distance_transform_edt
        return float(distance_transform_edt(~a)[b].min())
    except ImportError:
        ay, ax = np.where(a)
        by, bx = np.where(b)
        # Chunk to keep the fallback bounded for SA-1B-sized masks.
        best = float("inf")
        for start in range(0, len(by), 2048):
            dy = by[start:start + 2048, None] - ay[None, :]
            dx = bx[start:start + 2048, None] - ax[None, :]
            best = min(best, float(np.sqrt((dx * dx + dy * dy).min(axis=1)).min()))
        return best


def bbox_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Cheap lower bound used before loading expensive pair masks."""
    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def normalized_category(value: str) -> str:
    return " ".join(str(value or "object").lower().replace("_", " ").split())


def category_allowed(value: str) -> tuple[bool, str | None]:
    category = normalized_category(value)
    # Dense labels commonly add modifiers: "artificial autumn leaves" and
    # "brick wall" must not evade the explicit twin exclusions. Inspect noun
    # endings rather than every token so "stone fountain" remains eligible.
    heads = {alias.strip().split()[-1] for alias in re.split(r"[,;/]", category) if alias.strip()}
    if heads & BANDED_CATEGORIES:
        return False, "texture_or_crowd_category"
    if heads & BACKGROUND_CATEGORIES:
        return False, "background_category"
    return True, None


def object_features(obj: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    mask = load_mask(obj, image_size)
    geometry = mask_geometry(mask)
    geometry["point"] = precise_point(mask)
    geometry["category"] = obj.get("category", "object")
    geometry["object_id"] = str(obj.get("object_id", obj.get("id", "")))
    geometry["mask"] = mask
    return geometry


def candidate_category(obj):
    """Coarse label hint, avoiding alias-list and decorative-adjective splits."""
    first = re.split(r'[,;/]', normalized_category(obj.get('category', '')))[0].strip()
    head = first.split()[-1] if first else 'unknown'
    return {'armchair':'chair', 'couch':'sofa'}.get(head,head)


def candidate_granularity(obj):
    """Cheap candidate hint only; planning verifies actual mask granularity."""
    meta = obj.get('metadata') or {}
    if isinstance(meta.get('is_part'), bool):
        return 'part' if meta['is_part'] else 'object'
    if obj.get('parent_id') or meta.get('parent_id'):
        return 'part'
    words = set(normalized_category(obj.get('category', '')).split())
    # Standalone lamps, planters and signs can be complete objects. Do not
    # classify them as parts merely because they may attach to another object.
    parts = {'sleeve', 'handle', 'wheel', 'tire', 'strap', 'frame', 'hinge',
             'latch', 'armrest', 'headlight', 'taillight', 'bumper', 'fender',
             'pedal', 'spoke', 'handlebar', 'drawer', 'knob', 'cuff', 'roof',
             'arm', 'hand', 'leg', 'foot', 'finger', 'head', 'face', 'torso'}
    return 'part' if words & parts else 'object'


def twin_pair_gate(a: dict[str, Any], b: dict[str, Any], *, image_size: tuple[int, int],
                   min_gap: float = 16.0, allow_same_host: bool = True) -> dict[str, Any]:
    """Validate structural twin constraints without making semantic claims."""
    reasons: list[str] = []
    if str(a.get("object_id")) == str(b.get("object_id")):
        reasons.append("same_object")
    ca, cb = normalized_category(a.get("category", "")), normalized_category(b.get("category", ""))
    if candidate_granularity(a) != candidate_granularity(b):
        reasons.append("granularity_mismatch")
    for obj in (a, b):
        ok, reason = category_allowed(obj.get("category", ""))
        if not ok:
            reasons.append(reason or "category_not_allowed")
    ga, gb = object_features(a, image_size), object_features(b, image_size)
    if ga["area_px"] / max(1, gb["area_px"]) > 3 or gb["area_px"] / max(1, ga["area_px"]) > 3:
        reasons.append("area_ratio_over_3")
    ma, mb = ga.pop("mask"), gb.pop("mask")
    sa = support_mask(ma, "attribute", "target_only")
    sb = support_mask(mb, "attribute", "target_only")
    gap = gap_pixels(sa, sb)
    threshold = max(float(min_gap), 0.1 * min(
        min(ga["bbox"][2] - ga["bbox"][0], ga["bbox"][3] - ga["bbox"][1]),
        min(gb["bbox"][2] - gb["bbox"][0], gb["bbox"][3] - gb["bbox"][1]),
    ))
    if gap < threshold:
        reasons.append("support_gap_too_small")
    parent_a = a.get("parent_id") or (a.get("metadata") or {}).get("parent_id")
    parent_b = b.get("parent_id") or (b.get("metadata") or {}).get("parent_id")
    same_host = bool(parent_a and parent_b and str(parent_a) == str(parent_b))
    if same_host and not allow_same_host:
        reasons.append("same_host_disabled")
    return {
        "eligible": not reasons,
        "reasons": reasons,
        "categories": [ca, cb],
        "category_relation": "same" if ca == cb else "cross",
        "gap_pixels": gap,
        "required_gap_pixels": threshold,
        "same_host": same_host,
        "area_ratio": max(ga["area_px"], gb["area_px"]) / max(1, min(ga["area_px"], gb["area_px"])),
        "features": {"a": ga, "b": gb},
    }


def adjacent_pair_gate(a: dict[str, Any], b: dict[str, Any], *, image_size: tuple[int, int],
                       max_ratio: float = 0.1) -> dict[str, Any]:
    """Return adjacency evidence for a joint-edit candidate pair."""
    ga, gb = object_features(a, image_size), object_features(b, image_size)
    ma, mb = ga.pop("mask"), gb.pop("mask")
    sa = support_mask(ma, "attribute", "target_only")
    sb = support_mask(mb, "attribute", "target_only")
    gap = gap_pixels(sa, sb)
    short = min(
        min(ga["bbox"][2] - ga["bbox"][0], ga["bbox"][3] - ga["bbox"][1]),
        min(gb["bbox"][2] - gb["bbox"][0], gb["bbox"][3] - gb["bbox"][1]),
    )
    overlap = float((sa & mb).sum()) / max(1, int(mb.sum()))
    eligible = gap <= max_ratio * max(1, short) and overlap <= 0.10
    return {"eligible": eligible, "gap_pixels": gap, "threshold_pixels": max_ratio * max(1, short),
            "support_a_over_source_b": overlap, "features": {"a": ga, "b": gb}}


def twin_pairs(objects: list[dict[str, Any]], image_size: tuple[int, int], *, max_pairs: int | None = None):
    for a, b in combinations(objects, 2):
        evidence = twin_pair_gate(a, b, image_size=image_size)
        if evidence["eligible"]:
            yield a, b, evidence
            if max_pairs is not None and max_pairs <= 1:
                return


def adjacent_pairs(objects: list[dict[str, Any]], image_size: tuple[int, int]):
    for a, b in combinations(objects, 2):
        evidence = adjacent_pair_gate(a, b, image_size=image_size)
        if evidence["eligible"]:
            yield a, b, evidence
