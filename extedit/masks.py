"""Masks and regions: COCO RLE, edited region, geometry between S and the change, add anchors.

Ported without changes of rule or parameter from the code that produced the D3 / D4 sets:
samtok_edit branch qwen-image-2.1-v2, src/samtok_edit21/preparation/protocol_adapt.py ("protocol-adapt-4",
commit 2109371) for the RLE helpers, edited_region ("lab-diff-hysteresis-2"), geometry, support_probe,
localize_anchor, main_pieces, sam3_flags and geometry_flags; and branch qwen-image-2.1-v3,
src/samtok_edit21/preparation/protocol_v2_adapt.py ("protocol-v2-blind-1") for SITE_PARAMS, site_spot and main_item.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageFilter

from . import instruction_protocol as ip

# ---------------------------------------------------------------- RLE and boxes


def decode(rle):
    from pycocotools import mask as mask_utils
    return mask_utils.decode({"size": list(rle["rle_size"]), "counts": rle["rle_counts"].encode()}).astype(bool)


def box_1000(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    h, w = mask.shape
    return [int(xs.min() * 1000 // w), int(ys.min() * 1000 // h),
            int(-(-(xs.max() + 1) * 1000 // w)), int(-(-(ys.max() + 1) * 1000 // h))]


def encode(mask):
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"rle_size": [int(v) for v in rle["size"]], "rle_counts": rle["counts"].decode(), "box_1000": box_1000(mask)}


def box_mask(box, shape, grow_frac=0.0):
    """A 0-1000 box as a pixel mask, grown by `grow_frac` of its size on every side."""
    h, w = shape
    mx, my = grow_frac * (box[2] - box[0]), grow_frac * (box[3] - box[1])
    out = np.zeros((h, w), bool)
    out[max(0, int((box[1] - my) * h / 1000)):int(np.ceil((box[3] + my) * h / 1000)),
        max(0, int((box[0] - mx) * w / 1000)):int(np.ceil((box[2] + mx) * w / 1000))] = True
    return out


def box_iou(a, b):
    if not a or not b:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area else 0.0


def dilate(mask, radius):
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(image.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0


def grow(mask, radius):
    """Pixels within `radius` of the mask (Euclidean), in time independent of the radius."""
    import cv2
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius


def side(mask):
    """Mean side of the mask's box in pixels (the "side length" the support rules scale with)."""
    ys, xs = np.nonzero(mask)
    return float(np.sqrt((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)))

# ---------------------------------------------------------------- edited region

EDITED_METHOD = "lab-diff-hysteresis-2"  # 2: noise measured outside S, not over the edit, when S fills the image
# Calibrated on round 7: backgrounds of pasted and most linked pairs are identical (Delta E 0); resized pairs carry
# noise with a median of about 2 and a 99th percentile of about 6; real edits sit at 10 to 40 and more. The noise
# level is 3x the median far from S, which matches that 99th percentile but ignores real changes elsewhere.
EDITED_PARAMS = {"blur_sigma": 1.0, "t_low": 6.0, "t_high": 12.0, "low_k": 1.5, "high_k": 2.5,
                 "noise_far_frac": 0.08, "zone_frac": 0.06, "close_frac": 0.004, "hole_frac": 0.005,
                 "min_component_frac": 0.0002, "keep_in_zone": 0.3, "noise_min_frac": 0.05}


def color_difference(source, target, sigma=EDITED_PARAMS["blur_sigma"]):
    """CIE76 Delta E per pixel after a light blur (JPEG and resampling noise)."""
    import cv2
    labs = []
    for image in (source, target):
        array = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else image
        lab = cv2.cvtColor(cv2.GaussianBlur(array, (0, 0), sigma), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] *= 100 / 255
        lab[..., 1:] -= 128
        labs.append(lab)
    return np.sqrt(((labs[0] - labs[1]) ** 2).sum(-1))


def hysteresis(d, t_low, t_high, diag, p=EDITED_PARAMS):
    """Weak changes connected to a strong one, closed, with small holes filled (the core of edited_region)."""
    import cv2
    h, w = d.shape
    low = d > t_low
    count, labels = cv2.connectedComponents(low.astype(np.uint8), connectivity=8)
    seeds = np.zeros(count, bool)
    seeds[np.unique(labels[d > t_high])] = True
    seeds[0] = False
    mask = seeds[labels]
    radius = max(2, round(p["close_frac"] * diag))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    # Fill small holes (unchanged pixels inside a changed object); keep large ones, which can be real.
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    small = stats[:, cv2.CC_STAT_AREA] < p["hole_frac"] * h * w
    small[border] = False
    return mask | small[labels]


def edited_region(source, target, s_mask, exec_mask=None, params=EDITED_PARAMS, d=None):
    """Mask of the pixels that really changed between BEFORE and AFTER, and how it was found.

    Hysteresis on Delta E: weak changes join a region only when connected to a strong change, so faint
    regeneration noise is ignored while the full extent of an edit (soft shadows, anti-aliased edges) is kept.
    Thresholds rise with this pair's own noise, measured far from S and the execution region. Components mostly
    outside the neighborhood of S and the execution region are dropped and counted, never silently kept.
    """
    import cv2
    p = params
    d = color_difference(source, target, p["blur_sigma"]) if d is None else d
    h, w = d.shape
    diag = (h * h + w * w) ** 0.5
    anchor = s_mask | exec_mask if exec_mask is not None else s_mask
    # Noise is measured where nothing was edited: far from S and the execution region; when they fill the image,
    # outside them; when even that is too small, the calibrated floor.
    enough = p["noise_min_frac"] * h * w
    noise_from = "far"
    pixels = ~grow(anchor, p["noise_far_frac"] * diag)
    if pixels.sum() < enough:
        noise_from, pixels = "outside", ~anchor
    if pixels.sum() < enough:
        noise_from, pixels = "floor", None
    noise = 3 * float(np.median(d[pixels])) if pixels is not None else 0.0
    t_low, t_high = max(p["t_low"], p["low_k"] * noise), max(p["t_high"], p["high_k"] * noise)
    mask = hysteresis(d, t_low, t_high, diag, p)
    zone = grow(anchor, p["zone_frac"] * diag)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    in_zone = np.bincount(labels[zone], minlength=count)
    on_anchor = np.bincount(labels[anchor], minlength=count)
    # Size: a small piece counts when it touches S or the execution region (a sticker, a removed letter).
    big = (area >= max(30, p["min_component_frac"] * h * w)) | ((area >= 12) & (on_anchor > 0))
    # Place: mostly near S, or covering much of S and the execution region (a large replacement spills far).
    near = (in_zone >= p["keep_in_zone"] * area) | (on_anchor >= p["keep_in_zone"] * anchor.sum())
    keep = big & near
    keep[0] = False
    region = keep[labels]
    total = float(h * w)
    stats_out = {"noise": round(noise, 2), "noise_from": noise_from, "t_low": round(t_low, 2),
                 "t_high": round(t_high, 2), "area_frac": round(region.sum() / total, 5), "components": int(keep.sum()),
                 "dropped_far_frac": round(float(area[1:][big[1:] & ~near[1:]].sum()) / total, 5),
                 "dropped_small_frac": round(float(area[1:][~big[1:]].sum()) / total, 5),
                 "s_covered": round(float((region & s_mask).sum()) / max(int(s_mask.sum()), 1), 3),
                 "in_anchor": round(float((region & anchor).sum()) / max(int(region.sum()), 1), 3)}
    return (region if region.any() else None), stats_out

# ---------------------------------------------------------------- geometry between S and the change


def mask_gap(a, b):
    """Shortest distance in pixels between two non-empty masks; 0 when they touch or overlap."""
    import cv2
    return float(cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)[a].min())


def geometry_core(s_mask, exec_mask):
    """Overlap, contact, offset and distances between S and the execution region."""
    h, w = s_mask.shape
    exec_area, s_area = exec_mask.sum(), s_mask.sum()
    near = dilate(s_mask, max(3, round(0.01 * max(h, w))))
    sy, sx = np.nonzero(s_mask)
    ey, ex = np.nonzero(exec_mask)
    sb, eb = box_1000(s_mask), box_1000(exec_mask)
    gap = max(0, max(sb[0], eb[0]) - min(sb[2], eb[2])) + max(0, max(sb[1], eb[1]) - min(sb[3], eb[3]))
    return {"exec_in_s": round(float((exec_mask & s_mask).sum() / exec_area), 3),
            "s_in_exec": round(float((exec_mask & s_mask).sum() / s_area), 3),
            "exec_touch_s": round(float((exec_mask & near).sum() / exec_area), 3),
            "dx": round(float(ex.mean() - sx.mean()) / w, 3), "dy": round(float(ey.mean() - sy.mean()) / h, 3),
            "box_gap_1000": int(gap), "gap_frac": round(mask_gap(exec_mask, s_mask) / (h * h + w * w) ** 0.5, 4)}


# An item that rests on S has S under its center; one that hangs from S has S above it. The probe is the middle 40 %
# of the item's width: from its center down to its bottom and 3 % of the diagonal beyond for resting, and from its
# center up to the top of the image for hanging. Items held, contained, attached, sewn, wrapped or flying meet S
# anywhere ("in the hand" is not "on").
BASE_RELATIONS = {"on", "resting on"}
TOP_RELATIONS = {"hanging from", "hooked over"}
PROBE_PARAMS = {"column_frac": 0.4, "beyond_frac": 0.03}


def support_probe(footprint, relation="", host_scope="", params=PROBE_PARAMS):
    """The region where S must be for the relation (see BASE_RELATIONS), or None when it has no direction."""
    if relation not in BASE_RELATIONS | TOP_RELATIONS or re.search(r"\bsky\b", host_scope or ""):
        return None
    h, w = footprint.shape
    ys, xs = np.nonzero(footprint)
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    half = max(1, round(params["column_frac"] * (xs.max() + 1 - xs.min()) / 2))
    beyond = round(params["beyond_frac"] * (h * h + w * w) ** 0.5)
    probe = np.zeros((h, w), bool)
    if relation in TOP_RELATIONS:
        probe[:cy + 1, max(0, cx - half):cx + half + 1] = True
    else:
        probe[cy:ys.max() + 1 + beyond, max(0, cx - half):cx + half + 1] = True
    return probe


def geometry(spec, s_mask, exec_mask, host_scope=""):
    """How the execution region (for add: the new item) sits relative to S; for an add resting on or hanging
    from S also whether S is under (above) the item's center (support_probe)."""
    if exec_mask is None or not exec_mask.any() or not s_mask.any():
        return {}
    out = geometry_core(s_mask, exec_mask)
    if spec["operation"] == "add":
        probe = support_probe(exec_mask, spec["placement"].get("relation", "to"), host_scope)
        if probe is not None:
            out["s_in_probe"] = bool((probe & s_mask).any())
    return out

# ---------------------------------------------------------------- add anchors

ANCHOR_METHOD = "anchor-window-1"
# An anchor already says where the item goes, and stays whole, when it reaches no further than 4x the item's extent
# (box diagonal) and is either at most 3x the item's area and 5 % of the image, or at most 1 % of the image. Otherwise
# (a road, a cake, a long roof edge) the spot is S inside an ellipse around the new item's box (its pieces of at least
# a quarter of the largest one), grown on every side by 0.35x the item's mean side (at least 1.5 % of the image
# diagonal), and widened 1.5x at a time until it holds half the item's area (at least 0.2 % of the image). An anchor
# the window holds almost whole (85 %) stays whole; of a cut, pieces under a quarter of the main one are dropped.
# Reference anchors must lie within 1.5 % of the diagonal of the item; a support with contact more than 3 % away is
# not one.
ANCHOR_PARAMS = {"local_extent_k": 4.0, "local_k": 3.0, "local_frac": 0.05, "small_frac": 0.01, "piece_frac": 0.25,
                 "margin_k": 0.35, "margin_min_frac": 0.015, "grow": 1.5, "steps": 6, "min_area_k": 0.5,
                 "min_area_frac": 0.002, "keep_whole": 0.85, "piece_keep": 0.25, "adjacent_frac": 0.015,
                 "apart_frac": 0.03}
CONTACT_RELATIONS = {"to", "on", "in", "inside", "at", "around", "across", "along", "over", "attached to",
                     "clipped to", "pinned to", "resting on", "mounted on", "sewn onto", "tied around", "hooked over",
                     "strapped over"}
OVER_OVERLAP = 0.2  # "over" is draped over or covering S: a fifth of the item on S, or of S under the item
# Protocol 2.0 (protocol_v2_adapt): the site leaves the sentence, so the mask carries it. The window above is reused
# with two changes: an anchor stays whole only when it is at most 3x the new item's area (no exemption for anchors
# under 1 % of the image), and the window stops growing once it holds half the item's area or 0.02 % of the image.
SITE_PARAMS = {**ANCHOR_PARAMS, "small_frac": 0.0, "min_area_frac": 0.0002}
MIN_SPOT_FRAC, MIN_SPOT_PX = 0.0001, 64


def box_extent(mask):
    """Diagonal of the mask's box in pixels."""
    ys, xs = np.nonzero(mask)
    return float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def main_pieces(mask, keep_frac=ANCHOR_PARAMS["piece_keep"]):
    """One compact region: the main piece, and pieces of at least `keep_frac` of it."""
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    area = stats[:, cv2.CC_STAT_AREA]
    keep = area >= keep_frac * area[1:].max()
    keep[0] = False
    return keep[labels]


def localize_anchor(s_mask, footprint, params=ANCHOR_PARAMS):
    """The part of the anchor S around the new item's footprint, so that S says where on it the item goes.

    Returns (mask, info). The mask is S itself when S is already local or the window holds nearly all of it, and
    None when no part of S is near the item at all.
    """
    import cv2
    p = params
    h, w = s_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(footprint.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    main = area >= p["piece_frac"] * area[1:].max()
    main[0] = False
    footprint = main[labels]
    ys, xs = np.nonzero(footprint)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    a0, b0 = (xs.max() + 1 - xs.min()) / 2, (ys.max() + 1 - ys.min()) / 2
    margin = max(p["margin_k"] * 2 * (a0 * b0) ** 0.5, p["margin_min_frac"] * (h * h + w * w) ** 0.5)
    s_area = int(s_mask.sum())
    target = min(max(p["min_area_k"] * footprint.sum(), p["min_area_frac"] * h * w), p["keep_whole"] * s_area)
    yy, xx = np.ogrid[:h, :w]
    for step in range(p["steps"]):
        local = s_mask & ((((xx - cx) / (a0 + margin)) ** 2 + ((yy - cy) / (b0 + margin)) ** 2) <= 1)
        if local.sum() >= target or step == p["steps"] - 1:
            break
        margin *= p["grow"]
    info = {"method": ANCHOR_METHOD, "margin_px": int(round(margin)), "full_area_frac": round(s_area / (h * w), 5)}
    if not local.any():
        return None, {**info, "cut": False}
    already_local = box_extent(s_mask) <= p["local_extent_k"] * box_extent(footprint) and (
        (s_area <= p["local_k"] * footprint.sum() and s_area <= p["local_frac"] * h * w)
        or s_area <= p["small_frac"] * h * w)
    if already_local or local.sum() >= p["keep_whole"] * s_area:
        return s_mask, {**info, "cut": False}
    local = main_pieces(local, p["piece_keep"])
    return local, {**info, "cut": True, "area_frac": round(float(local.sum()) / (h * w), 5)}


def main_item(changed, keep=0.5):
    """The new item among the changed pixels: the largest connected piece and pieces at least half its size."""
    import cv2
    if changed is None or not changed.any():
        return None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    big = area >= keep * area[1:].max()
    big[0] = False
    return big[labels]


def site_spot(s_mask, footprint):
    """S cut to where the new item sits, or (None, reason).  The protocol-1.0 anchor window with SITE_PARAMS."""
    if footprint is None or not footprint.any():
        return None, "no_footprint"
    spot, info = localize_anchor(s_mask, footprint, SITE_PARAMS)
    if spot is None:
        return None, "item_not_on_anchor"
    if spot.sum() < max(MIN_SPOT_PX, MIN_SPOT_FRAC * spot.size):
        return None, "spot_too_small"
    return spot, info

# ---------------------------------------------------------------- flags


def sam3_flags(mask, box):
    """A SAM3 mask of S should fill its grounding box and stay inside it (box grown by 10 % per side)."""
    if not box or not mask.any():
        return []
    h, w = mask.shape
    x1, y1, x2, y2 = box[0] * w / 1000, box[1] * h / 1000, box[2] * w / 1000, box[3] * h / 1000
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    inside = np.zeros_like(mask)
    inside[max(0, int(y1 - my)):int(np.ceil(y2 + my)), max(0, int(x1 - mx)):int(np.ceil(x2 + mx))] = True
    flags = []
    if (mask & ~inside).sum() / mask.sum() > 0.15:
        flags.append("sam3_spill")
    if mask.sum() / max((x2 - x1) * (y2 - y1), 1) < 0.15:
        flags.append("sam3_sparse")
    return flags


def is_text_change(spec):
    return spec["operation"] == "attribute" and spec["attributes"][0]["property"] == "text"


def geometry_flags(spec, geo):
    """(flags, hard reasons) from where the execution region (add: the new item) lies relative to S."""
    flags, hard = [], []
    if ip.needs_local_execution(spec):
        if not geo:
            hard.append("execution_region_missing")
        elif spec["remove_mode"] == "contents":
            if geo["s_in_exec"] < 0.9:  # the container and what it holds are both removed
                hard.append("execution_region_misses_container")
        elif geo["s_in_exec"] >= 0.9:  # a part, the contents only or a cut: never the whole host
            hard.append("execution_region_covers_host")
    if spec["operation"] == "add":
        mode = spec["placement"].get("anchor_mode", "support")
        relation = spec["placement"].get("relation", "to")
        if not geo:
            hard.append("add_without_item")
        elif mode == "support":
            # The anchor is what the item touches: one with open space between them is a reference in disguise.
            if relation in CONTACT_RELATIONS and geo["gap_frac"] > ANCHOR_PARAMS["apart_frac"]:
                hard.append("support_not_touching_item")
            elif geo.get("s_in_probe") is False:
                hard.append("support_not_under_item")
            elif relation == "over" and max(geo["exec_in_s"], geo["s_in_exec"]) < OVER_OVERLAP:
                hard.append("over_without_overlap")
            elif geo["gap_frac"] > ANCHOR_PARAMS["adjacent_frac"]:
                flags.append("item_not_touching_anchor")
        else:
            if geo["gap_frac"] > ANCHOR_PARAMS["adjacent_frac"]:
                hard.append("reference_anchor_not_adjacent")
            if geo["exec_in_s"] > 0.5:
                (flags if relation == "near" else hard).append("reference_item_inside_anchor")
            if (relation in ("under", "below") and geo["dy"] < 0) or (relation == "above" and geo["dy"] > 0):
                hard.append("reference_direction")
            site = spec["placement"].get("site", "")
            if ("left" in site and geo["dx"] > 0) or ("right" in site and geo["dx"] < 0):
                hard.append("reference_side")
    if is_text_change(spec) and geo and geo["exec_in_s"] < 0.7:
        flags.append("text_outside_carrier")
    return flags, hard
