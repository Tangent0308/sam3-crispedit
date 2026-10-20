"""Pixel steps without masks: size, registration, change map (precheck), paste-back and its checks (gate).

register, stats, paste, ring_metrics and the RULES / GATE values are ported from samtok_edit
scripts/data_cleaning/clean_pairs.py, and coherent_outside / alignment from scripts/data_cleaning/checks.py (the
pixel cleaning that produced the cleaned pairs behind D3 / D4).  The change map is edited_region's hysteresis
(masks.hysteresis) without an S: the precheck has no mask yet.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import masks as mk

WORK = 512            # long side for the pixel statistics
REG_SIDE = 768        # long side for feature matching
ALIGN_SIDE = 256
PASTE_DILATE = 0.02   # paste-back region: support dilated by 2% of the short side
# A registered target need not cover the whole frame (a zoomed generator output does not): the precheck asks for 85 %,
# and the gate for 98 % of the paste region (clean_pairs.RULES["register_cover"] was measured on the region too).
RULES = {"align_tiles": 5, "align_min": 0.4, "align_good": 0.8, "register_gain": 0.1, "register_cover": 0.85,
         "paste_cover": 0.98}
GATE = {"far_out_final": 0.02, "fill_final": 0.10, "ring_cut": 0.30, "ring_mad": 25.0, "area_kept": 0.60,
        "blob_max": 0.01}
# Precheck on the change map (doc 07 section 3.2): no visible change, a change over most of the image (style,
# relighting, a regenerated scene), or a pair whose structures stay misaligned after registration.
# A noise level (3x the median Delta E) above 30 means the whole image was re-rendered or stays misregistered.
PRECHECK = {"min_change": 0.0005, "max_change": 0.40, "max_noise": 15.0, "drop_noise": 30.0, "area_merge_frac": 0.01, "max_areas": 8,
            "min_area_frac": 0.0002}


def small(x, size, nearest=False):
    img = Image.fromarray(x.astype(np.uint8) if x.dtype != bool else x.astype(np.uint8) * 255)
    out = np.asarray(img.resize(size, Image.NEAREST if nearest else Image.BOX))
    return out > 0 if x.dtype == bool else out.astype(np.float32)


def work_size(shape, side=WORK):
    h, w = shape
    s = side / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def far_from(z, size, margin, valid=None):
    """Pixels (at `size`) farther than `margin` of the long side from the region z, inside `valid`."""
    zs = small(z, size, nearest=True)
    vs = small(valid, size, nearest=True) if valid is not None else np.ones(zs.shape, bool)
    if not zs.any():
        return vs
    return (ndimage.distance_transform_edt(~zs) > margin * max(size)) & vs


def stats(src, tgt, z, valid=None):
    """far_out: share of clearly changed pixels beyond 4% of the long side from the region;
    in_share: share of all changed pixels that lie within that distance; fill: changed share of the region."""
    size = work_size(z.shape)
    a, b, zs = small(src, size), small(tgt, size), small(z, size, nearest=True)
    vs = small(valid, size, nearest=True) if valid is not None else np.ones(zs.shape, bool)
    changed = (np.abs(b - a).mean(axis=2) > 25) & vs
    near = ndimage.distance_transform_edt(~zs) <= 0.04 * max(size) if zs.any() else np.zeros(zs.shape, bool)
    far = ~near & vs
    return {"far_out": float(changed[far].mean()) if far.any() else 0.0,
            "in_share": float((changed & near).sum() / changed.sum()) if changed.any() else 1.0,
            "fill": float(changed[zs].mean()) if zs.any() else 0.0}


def register(src, tgt, z=None):
    """Affine from target pixels to source pixels, estimated outside the region; (None, reason) if it fails."""
    sa = REG_SIDE / max(src.shape[:2])
    ta = REG_SIDE / max(tgt.shape[:2])
    a = cv2.resize(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
    t = cv2.resize(cv2.cvtColor(tgt, cv2.COLOR_RGB2GRAY), None, fx=ta, fy=ta, interpolation=cv2.INTER_AREA)
    keep = None
    if z is not None and z.any():
        zs = cv2.resize(z.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        keep = (ndimage.distance_transform_edt(~zs) > 0.06 * REG_SIDE).astype(np.uint8) * 255
    sift = cv2.SIFT_create(4000)
    ka, da = sift.detectAndCompute(a, keep)
    kb, db = sift.detectAndCompute(t, None)
    if da is None or db is None or len(ka) < 20 or len(kb) < 20:
        return None, "few_features"
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None, "few_matches"
    pa = np.float32([ka[m.queryIdx].pt for m in good]) / sa
    pb = np.float32([kb[m.trainIdx].pt for m in good]) / ta
    M, inl = cv2.estimateAffine2D(pb, pa, method=cv2.RANSAC, ransacReprojThreshold=2.0 / sa, maxIters=4000)
    if M is None or inl is None or inl.sum() < 12:
        return None, "no_transform"
    return M, f"ok:{int(inl.sum())}/{len(good)}"


def alignment(src, tgt, z=None, margin=0.06, valid=None, tile=32, search=8, min_std=6.0, min_ncc=0.7):
    """(share of confident tiles whose best shift is <= 1 px, number of confident tiles) at ALIGN_SIDE px.
    Textured tiles far outside the region are matched by normalised cross-correlation within +-`search` px in
    the target.  Re-rendered but aligned targets keep their structures in place; zoomed or shifted targets do not.
    Tiles inside an edit rarely reach the correlation threshold, so a mask is optional."""
    h, w = src.shape[:2]
    size = work_size((h, w), ALIGN_SIDE)
    a = np.asarray(Image.fromarray(src).resize(size, Image.BOX))
    b = np.asarray(Image.fromarray(tgt.astype(np.uint8)).resize(size, Image.BOX))
    far = far_from(np.zeros((h, w), bool) if z is None else z, size, margin, valid)
    a = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    b = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    H, W = a.shape
    good = conf = 0
    for y in range(search, H - tile - search + 1, tile // 2):
        for x in range(search, W - tile - search + 1, tile // 2):
            if far[y:y + tile, x:x + tile].mean() < 0.9:
                continue
            t = a[y:y + tile, x:x + tile]
            if t.std() < min_std:
                continue
            res = cv2.matchTemplate(b[y - search:y + tile + search, x - search:x + tile + search], t, cv2.TM_CCOEFF_NORMED)
            _, peak, _, loc = cv2.minMaxLoc(res)
            if peak < min_ncc:
                continue
            conf += 1
            good += max(abs(loc[0] - search), abs(loc[1] - search)) <= 1
    return (good / conf if conf else float("nan")), conf


def coherent_outside(src, tgt, z, thr=30.0, open_radius=3, margin=0.06, valid=None):
    """Largest solid block of change far outside the region, as a fraction of the image (at WORK px).
    The target is first colour-matched to the source (per-channel affine fitted on pixels beyond `margin` of the
    long side from the region), so global tone and lighting shifts do not count; strong differences are then opened
    with a disk so thin edge and texture noise from re-rendering disappears."""
    size = work_size(z.shape)
    a = np.asarray(Image.fromarray(src).resize(size, Image.BOX)).astype(np.float32)
    b = np.asarray(Image.fromarray(tgt.astype(np.uint8)).resize(size, Image.BOX)).astype(np.float32)
    far = far_from(z, size, margin, valid)
    if far.sum() < 100:
        return 0.0
    bn = b.copy()
    for c in range(3):
        x, y = b[..., c][far], a[..., c][far]
        var = x.var()
        gain = ((x - x.mean()) * (y - y.mean())).mean() / var if var > 1e-3 else 1.0
        gain = float(np.clip(gain, 0.5, 2.0))
        bn[..., c] = gain * (b[..., c] - x.mean()) + y.mean()
    strong = np.abs(bn - a).mean(axis=2) > thr
    yy, xx = np.mgrid[-open_radius:open_radius + 1, -open_radius:open_radius + 1]
    blobs = ndimage.binary_opening(strong, structure=(xx ** 2 + yy ** 2) <= open_radius ** 2) & far
    lab, n = ndimage.label(blobs)
    if n == 0:
        return 0.0
    areas = ndimage.sum(np.ones(lab.shape), lab, index=np.arange(1, n + 1))
    return float(areas.max() / lab.size)


def paste(src, tgt, z, band=None):
    """Target inside the region grown by `band` pixels (default 2% of the short side) with a soft (Gaussian) edge,
    exactly the source beyond it."""
    r = max(2, int(PASTE_DILATE * min(z.shape))) if band is None else band
    grown = mk.grow(z, r) if r else z.copy()
    alpha = ndimage.gaussian_filter(grown.astype(np.float32), sigma=2)[..., None]
    alpha[~mk.grow(grown, 6)] = 0.0  # exactly the source beyond the soft edge
    out = tgt.astype(np.float32) * alpha + src.astype(np.float32) * (1 - alpha)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), grown


TONE = {"gain_min": 0.85, "gain_max": 1.18, "min_gain_levels": 1.0, "sample": 4}


def tone_match(src, tgt, away, valid=None, params=TONE):
    """The target with a per-channel gain and offset that map its tone onto the source's on the pixels away from the
    edit, applied only when that lowers their mean difference by at least `min_gain_levels`; and what was done."""
    m = away if valid is None else away & valid
    step = params["sample"]
    a = src[::step, ::step][m[::step, ::step]].astype(np.float32)
    b = tgt[::step, ::step][m[::step, ::step]].astype(np.float32)
    if len(a) < 1000:
        return tgt, {"applied": False, "reason": "few_pixels"}
    gains, offsets = [], []
    for c in range(3):
        x, y = b[:, c], a[:, c]
        var = x.var()
        gain = ((x - x.mean()) * (y - y.mean())).mean() / var if var > 1e-3 else 1.0
        gain = float(np.clip(gain, params["gain_min"], params["gain_max"]))
        gains.append(gain)
        offsets.append(float(y.mean() - gain * x.mean()))
    gains, offsets = np.array(gains, np.float32), np.array(offsets, np.float32)
    before = float(np.abs(b - a).mean())
    after = float(np.abs(np.clip(b * gains + offsets, 0, 255) - a).mean())
    info = {"gain": [round(float(g), 4) for g in gains], "offset": [round(float(o), 2) for o in offsets],
            "before": round(before, 2), "after": round(after, 2)}
    if before - after < params["min_gain_levels"]:
        return tgt, {"applied": False, **info}
    out = np.clip(np.rint(tgt.astype(np.float32) * gains + offsets), 0, 255).astype(np.uint8)
    return out, {"applied": True, **info}


def ring_metrics(src, tgt, grown):
    """Change in the band just outside the paste region, where a paste would cut the edit or leave a seam."""
    size = work_size(grown.shape)
    g = small(grown, size, nearest=True)
    band = ndimage.binary_dilation(g, iterations=max(2, int(0.02 * max(size)))) & ~g
    if not band.any():
        return 0.0, 0.0
    d = np.abs(small(tgt, size) - small(src, size)).mean(axis=2)
    return float((d[band] > 25).mean()), float(d[band].mean())

# ---------------------------------------------------------------- precheck


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def resize_to(tgt, shape):
    h, w = shape
    if tgt.shape[:2] == (h, w):
        return tgt
    return np.asarray(Image.fromarray(tgt).resize((w, h), Image.BICUBIC))


def apply_alignment(tgt_orig, shape, align):
    """The target in the source frame: as stored, resized, or warped by the stored affine."""
    h, w = shape
    if align["kind"] == "register":
        M = np.asarray(align["matrix"], np.float64)
        return cv2.warpAffine(tgt_orig, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return resize_to(tgt_orig, shape)


def valid_mask(tgt_shape, shape, align):
    """Source pixels the target covers (all of them unless registered)."""
    h, w = shape
    if align["kind"] != "register":
        return np.ones((h, w), bool)
    M = np.asarray(align["matrix"], np.float64)
    return cv2.warpAffine(np.full(tgt_shape[:2], 255, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0


def choose_alignment(src, tgt_orig):
    """Plain (resized) target when its structures sit in place, else the registered one when that is clearly better.

    Returns (align, aligned target, valid mask, reason or None); reason "misaligned" means neither is in place."""
    shape = src.shape[:2]
    base = resize_to(tgt_orig, shape)
    kind = "same" if tgt_orig.shape[:2] == shape else "resize"
    share, tiles = alignment(src, base)
    align = {"kind": kind, "share": None if tiles == 0 else round(share, 3), "tiles": tiles}
    if tiles >= RULES["align_tiles"] and share >= RULES["align_good"]:
        return align, base, np.ones(shape, bool), None
    M, status = register(src, tgt_orig)
    align["register"] = status
    if M is not None:
        warped = cv2.warpAffine(tgt_orig, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpAffine(np.full(tgt_orig.shape[:2], 255, np.uint8), M, (shape[1], shape[0]),
                               flags=cv2.INTER_NEAREST) > 0
        r_share, r_tiles = alignment(src, warped, valid=valid)
        cover = float(valid.mean())
        align.update(reg_share=None if r_tiles == 0 else round(r_share, 3), reg_tiles=r_tiles, cover=round(cover, 4),
                     scale=round(float(np.sqrt(abs(np.linalg.det(M[:, :2])))), 4),
                     shift=round(float(np.hypot(M[0, 2], M[1, 2])), 1))
        before = share if tiles else 0.0
        if (r_tiles >= RULES["align_tiles"] and r_share >= max(RULES["align_min"], before + RULES["register_gain"])
                and cover >= RULES["register_cover"]):
            align.update(kind="register", matrix=[[float(v) for v in row] for row in M])
            return align, warped, valid, None
    if tiles >= RULES["align_tiles"] and share < RULES["align_min"]:
        return align, base, np.ones(shape, bool), "misaligned"
    return align, base, np.ones(shape, bool), None


def change_map(src, tgt, valid=None, params=mk.EDITED_PARAMS):
    """All pixels that changed (edited_region's hysteresis on Delta E, noise from the whole image), and its stats."""
    import cv2 as _cv2
    p = params
    d = mk.color_difference(src, tgt, p["blur_sigma"])
    if valid is not None:
        d = np.where(valid, d, 0.0)
    h, w = d.shape
    diag = (h * h + w * w) ** 0.5
    noise = 3 * float(np.median(d if valid is None else d[valid]))
    t_low, t_high = max(p["t_low"], p["low_k"] * noise), max(p["t_high"], p["high_k"] * noise)
    mask = mk.hysteresis(d, t_low, t_high, diag, p)
    count, labels, st, _ = _cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = st[:, _cv2.CC_STAT_AREA] >= max(30, PRECHECK["min_area_frac"] * h * w)
    keep[0] = False
    mask = keep[labels]
    return mask, {"noise": round(noise, 2), "t_low": round(t_low, 2), "t_high": round(t_high, 2),
                  "area_frac": round(float(mask.mean()), 5), "components": int(keep.sum())}, d


def change_areas(mask):
    """Numbered change areas for the plan views: pieces within 1 % of the diagonal joined, largest first; at most
    PRECHECK["max_areas"], the rest reported as a count."""
    h, w = mask.shape
    if not mask.any():
        return [], 0
    radius = max(2, round(PRECHECK["area_merge_frac"] * (h * h + w * w) ** 0.5))
    joined = mk.grow(mask, radius)
    count, labels = cv2.connectedComponents(joined.astype(np.uint8), connectivity=8)
    labels = np.where(mask, labels, 0)
    areas = np.bincount(labels.ravel(), minlength=count)
    order = [k for k in np.argsort(-areas) if k != 0 and areas[k] > 0]
    out = []
    for n, k in enumerate(order[:PRECHECK["max_areas"]], 1):
        piece = labels == k
        out.append({"n": n, "area_frac": round(float(areas[k]) / (h * w), 5), **mk.encode(piece)})
    return out, max(0, len(order) - PRECHECK["max_areas"])
