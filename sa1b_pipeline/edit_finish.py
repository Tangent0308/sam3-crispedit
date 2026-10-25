"""Opt-in finishing for generated crops: grain, tone and stray-change cleanup.

All functions work on aligned crop arrays (uint8 HxWx3) and return
``(new_crop, info)``; ``info['applied']`` is False when nothing changed. None of
them changes masks, instructions or the writable support. (A shade-lock for
colour edits in shade was tried and removed: it fired once in a 115-pair A/B
and hid the requested colour; such edits are now filtered at planning by the
evidence-gate lighting check.)

* ``regrain``      add photo-matched sensor grain where the generated pixels are
                   smoother than the untouched surroundings (no flat/CG patch).
* ``tone_match``   removal only, diagnostic (off in the --edit-finish preset): the
                   collar between the target and the support edge must equal the
                   source; a smooth low-amplitude field removes the generator's
                   tone offset there and inside the fill. On smooth sky it
                   extrapolated the collar offset into a visible ghost (A/B C056).
* ``add_cleanup``  one-item additions: revert scattered changes far from the
                   added item (re-rendered host text, drifting texture).
* ``defocus_match`` additions/replacements: when the photo is out of focus at the
                   anchor (edge blur estimated by the re-blur gradient ratio), blur
                   the new item to the same depth of field; a crisp item on a soft
                   background looks pasted.
* ``detail_restore`` additions, diagnostic (off in the preset): put back the source where the
                   generator only resampled the photo (no change at a coarse scale). In a
                   24-pair blind A/B on full-run additions it was preferred 10:9 (5 same): fine
                   detail came back, but a soft ring around the new item read as a pasted
                   patch. A soft fine-detail transfer (``detail_transfer``) was 4:4 (16 same).
                   Not for removals/replacements: it revived thin remnants (P036 twigs).
* ``focus_match``  replaces defocus_match (which fired on 4 of 103 replayed results):
                   the focus at the new content's place is the smallest blur of at least
                   two references (anchor band, ring of surroundings, and for a
                   replacement the old object's own pixels). It does not touch exposure:
                   a brightness cap for shaded/backlit sites was tried and dropped (it
                   missed the over-saturated items it was meant for).
"""
from __future__ import annotations

import re

import numpy as np

VERSION = 'edit-finish-4-focus-match'

LIMITS = {
    'grain_ratio': .75,          # regrain when generated grain < this x surrounding grain
    'grain_max': 3.5,            # cap added grain sigma (8-bit levels)
    'grain_correlation': .8,     # luminance-correlated share of the added grain
    'tone_min': 2.0,             # minimal collar offset (levels) worth correcting
    'tone_max': 12.0,            # cap of the correction field (levels)
    'tone_change': 20.0,         # collar differences above this are real changes, ignored
    'cleanup_delta': 20,         # 8-bit max-channel difference (after low-pass) of a visible change
    'cleanup_sigma': 1.5,        # low-pass sigma for the change map
    'cleanup_reach': .35,        # keep changes within this x the main item's extent
    'cleanup_min_main': 48,
    'blur_sigma0': 1.0,          # re-blur sigma of the blur estimator (px)
    'blur_min_edges': 40,        # edge pixels needed for a blur estimate
    'blur_min_reference': 1.2,   # only act when the photo itself is soft at the anchor (px)
    'blur_min_gap': .6,          # ... and the new item is at least this much sharper (px)
    'blur_max': 5.0,             # cap of the added blur (px)
    # focus_match. In a blind replay, blur below ~0.8 px was not reliably visible (the reviewer
    # picked the softer side at chance), so it acts only when the change would be visible.
    'focus_min_reference': .8,   # act when every focus reference is at least this soft (px)
    'focus_min_gap': .3,         # ... and the new content is at least this much sharper (px)
    'focus_ring': .5,            # surroundings ring radius, x the new content's extent (8..120 px)
    # detail_restore: resampling blur barely moves a sigma-4 low-pass (a 1.5 px blur of a
    # 150-level step edge: < 4 levels), new content and its shadow do.
    'restore_sigma': 4.0,
    'restore_delta': 5.0,        # coarse change (8-bit levels) that counts as content
    'restore_grow': 3,           # keep this many px around coarse changes
    'restore_min_pixels': 200,   # act only when at least this many changed pixels come back
    # detail_transfer: fine detail (scale < transfer_sigma) of the source replaces the generated
    # detail where the latter is a damped copy of the former (local correlation), e.g. resampling.
    'transfer_sigma': 2.0,
    'transfer_window': 9,
    'transfer_ncc': (.45, .75),  # correlation ramp: below keep the edit, above take source detail
    'transfer_min_std': 2.0,     # source detail (8-bit levels) worth restoring
    'transfer_min_pixels': 400,
}

PLURAL = re.compile(r'\b(rows?|pairs?|sets?|groups?|clusters?|bunch(?:es)?|few|several|some|many|multiple|two|three|'
                    r'four|five|line|string|series|collection|pile)\b', re.I)

# Opt-in only (editor finish option 'material_rule'): no measurable gain in the paired A/B.
MATERIAL_RULE = (' Render the new item as a real photographed object at this photo\'s scale, focus, noise and '
                 'lighting, with the natural surface texture and subtle shading variation of its material; '
                 'avoid flat, uniform, plastic-looking or computer-rendered surfaces.')
REMOVE_RULE = (' Fill the revealed area with the surrounding surface\'s own texture, grain, sharpness and lighting, '
               'so no smooth, blurred, brighter or darker patch marks where the object was.')
# Opt-in (editor finish option 'lighting_rule'): added/replacing content at a site the planning lighting
# check called shade or silhouette is lit like its surroundings. Light sources keep their own glow.
LIGHTING_RULES = {
    'shade': (' The edited spot lies in shade: light the new item only with the same soft, dim shade light as its '
              'surroundings, with no direct sunlight and no bright highlights, and keep its colors as muted as the '
              'nearby shaded surfaces.'),
    'silhouette': (' The edited spot is backlit, facing away from the light: render the new item as dark and '
                   'low-contrast as the nearby backlit surfaces, with muted colors and no light falling on its front.'),
}
EMISSIVE_PAYLOAD = re.compile(r'\b(lights?(?![ -](?:gray|grey|blue|brown|green|pink|beige|yellow|red|purple|orange|tan|'
                              r'wood|colou?red|weight))|lamps?|lanterns?|bulbs?|leds?|neon|candles?|fire|flames?|glow\w*|'
                              r'screens?|displays?|monitors?|torch(?:es)?|sparklers?|fireworks?|signal)\b', re.I)


def lighting_rule(unit: dict) -> str:
    """Prompt sentence for an add/replace whose site is in shade or backlit ('' otherwise)."""
    spec = unit.get('instruction_spec') or {}
    if unit.get('operation') not in ('add', 'replace') or unit.get('site_lighting') not in LIGHTING_RULES:
        return ''
    if EMISSIVE_PAYLOAD.search(str(spec.get('payload') or unit.get('payload') or '')):
        return ''
    return LIGHTING_RULES[unit['site_lighting']]


EFFECT_RULE = (' Its own cast shadow and reflection next to it are part of the removal: they must disappear too, '
               'restoring the ground, wall or water there with the same light and texture as the surrounding surface.')


def _disk(radius):
    import cv2
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask, radius):
    import cv2
    mask = np.asarray(mask, bool)
    if radius <= 0:
        return mask
    if radius > 8:
        # Euclidean distance instead of a huge structuring element (same disk, O(N)).
        if not mask.any():
            return mask
        return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) <= radius
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0


def _smooth(array, sigma):
    """Gaussian blur; very wide kernels run on a downsampled copy (the result is smooth anyway)."""
    import cv2
    array = np.asarray(array, np.float32)
    if sigma <= 8:
        return cv2.GaussianBlur(array, (0, 0), sigma)
    scale = 8. / sigma
    h, w = array.shape[:2]
    small = cv2.resize(array, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(cv2.GaussianBlur(small, (0, 0), 8.), (w, h), interpolation=cv2.INTER_LINEAR)


def _erode(mask, radius):
    import cv2
    return cv2.erode(np.asarray(mask, np.uint8), _disk(radius)) > 0 if radius > 0 else np.asarray(mask, bool)


def _extent(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 1, 1
    h, w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    return min(h, w), max(h, w)


def grain_sigma(channel: np.ndarray, region: np.ndarray) -> float | None:
    """Robust high-pass noise sigma on the flattest 40% of ``region`` (None if too small)."""
    import cv2
    if region.sum() < 200:
        return None
    channel = channel.astype(np.float32)
    residual = channel - cv2.GaussianBlur(channel, (0, 0), 1.0)
    smooth = cv2.GaussianBlur(channel, (0, 0), 2.0)
    magnitude = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))[region]
    flat = magnitude <= np.quantile(magnitude, .4)
    values = residual[region][flat]
    if values.size < 100:
        return None
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _unit_noise(shape, rng):
    """Slightly correlated Gaussian field scaled so grain_sigma() measures ~1."""
    import cv2
    field = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), .6)
    patch = field[:256, :256]   # stationary field: a patch measures the same sigma
    measured = grain_sigma(patch, np.ones(patch.shape, bool)) or float(field.std())
    return field / max(measured, 1e-6)


def changed_weight(source: np.ndarray, edited: np.ndarray, alpha: np.ndarray, threshold: float = 6.) -> np.ndarray:
    """Soft weight of pixels the generator actually changed (inside the paste alpha)."""
    import cv2
    changed = np.abs(edited.astype(np.int16) - source.astype(np.int16)).max(axis=2) > threshold
    soft = cv2.GaussianBlur(_dilate(changed, 2).astype(np.float32), (0, 0), 1.5)
    return np.clip(np.maximum(soft, changed), 0, 1) * np.clip(np.asarray(alpha, np.float32), 0, 1)


def regrain(source: np.ndarray, edited: np.ndarray, weight: np.ndarray, *, seed: int = 0,
            untouched: np.ndarray | None = None, limits=LIMITS):
    """Match per-channel grain of generated pixels (weight > .9) to the untouched surroundings."""
    weight = np.clip(np.asarray(weight, np.float32), 0, 1)
    generated = weight > .9
    untouched = weight <= 0 if untouched is None else np.asarray(untouched, bool)
    info = {'applied': False}
    sigmas = []
    for c in range(3):
        reference, current = grain_sigma(source[..., c], untouched), grain_sigma(edited[..., c], generated)
        if reference is None or current is None:
            info['reason'] = 'insufficient_reference'
            return edited, info
        # The photo's own grain at the same place bounds the reference: textured
        # surroundings must not be mistaken for sensor noise.
        here = grain_sigma(source[..., c], generated)
        sigmas.append((reference if here is None else min(reference, here), current))
    info['reference_sigma'] = [round(r, 3) for r, _ in sigmas]
    info['generated_sigma'] = [round(g, 3) for _, g in sigmas]
    add = [min(limits['grain_max'], float(np.sqrt(max(0., r * r - g * g)))) if g < limits['grain_ratio'] * r else 0.
           for r, g in sigmas]
    if not any(add):
        info['reason'] = 'grain_matches'
        return edited, info
    rng = np.random.default_rng(seed)
    shared = _unit_noise(edited.shape[:2], rng)
    rho = limits['grain_correlation']
    out = edited.astype(np.float32)
    for c in range(3):
        if add[c]:
            noise = rho * shared + np.sqrt(1 - rho * rho) * _unit_noise(edited.shape[:2], rng)
            out[..., c] += add[c] * noise * weight
    info.update(applied=True, added_sigma=[round(a, 3) for a in add])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def tone_match(source: np.ndarray, edited: np.ndarray, target: np.ndarray, support: np.ndarray,
               weight: np.ndarray, effect: np.ndarray | None = None, limits=LIMITS):
    """Removal: cancel a smooth tone offset measured on the unchanged collar."""
    import cv2
    target, support = np.asarray(target, bool), np.asarray(support, bool)
    keep_out = _dilate(target, 3)
    if effect is not None and effect.any():
        keep_out |= _dilate(effect, 3)
    collar = support & ~keep_out & (np.asarray(weight) > .5)
    delta = edited.astype(np.float32) - source.astype(np.float32)
    small = collar & (np.abs(delta).max(axis=2) < limits['tone_change'])
    info = {'applied': False, 'collar_pixels': int(small.sum())}
    if small.sum() < 200:
        info['reason'] = 'small_collar'
        return edited, info
    median = np.median(delta[small], axis=0)
    info['collar_offset'] = [round(float(v), 2) for v in median]
    if np.abs(median).max() < limits['tone_min']:
        info['reason'] = 'no_offset'
        return edited, info
    _, ext_max = _extent(target)
    sigma = max(8., .25 * ext_max)
    w = small.astype(np.float32)
    denominator = _smooth(w, sigma) + 1e-6
    field = np.stack([_smooth(np.clip(delta[..., c], -20, 20) * w, sigma) / denominator
                      for c in range(3)], axis=2)
    # Far from any collar evidence fall back to the global median offset.
    confidence = np.clip(denominator * 4, 0, 1)
    field = field * confidence[..., None] + median[None, None, :] * (1 - confidence[..., None])
    field = np.clip(field, -limits['tone_max'], limits['tone_max'])
    out = edited.astype(np.float32) - field * (np.asarray(weight, np.float32) * support)[..., None]
    info.update(applied=True, max_field=round(float(np.abs(field[support]).max()), 2))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def add_cleanup(source: np.ndarray, edited: np.ndarray, weight: np.ndarray, limits=LIMITS):
    """One-item addition: keep the dominant change and its neighbourhood, revert scattered drift."""
    import cv2
    from scipy.ndimage import binary_opening, label
    region = np.asarray(weight) > 0
    # Low-pass difference: a re-rendered texture (grass, stone grain) keeps its local
    # mean and is not an item; an added object changes it.
    smooth = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), limits['cleanup_sigma'])
    delta = np.abs(smooth(edited) - smooth(source)).max(axis=2) > limits['cleanup_delta']
    delta = binary_opening(delta & region, iterations=1)
    labels, count = label(delta)
    info = {'applied': False, 'components': int(count)}
    if not count:
        info['reason'] = 'no_change'
        return edited, info
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    sizes[0] = 0
    main = int(sizes.argmax())
    if sizes[main] < limits['cleanup_min_main']:
        info['reason'] = 'no_dominant_item'
        return edited, info
    main_mask = labels == main
    _, extent = _extent(main_mask)
    reach = max(12, round(limits['cleanup_reach'] * extent))
    # Single linkage: an extended item (vine, garland) is a chain of nearby parts;
    # only changes out of reach of every kept part are drift.
    kept = main_mask
    member = np.zeros(count + 1, bool)
    for _ in range(12):
        member[np.unique(labels[_dilate(kept, reach) & delta])] = True
        member[0] = False
        grown = member[labels]
        if grown.sum() == kept.sum():
            break
        kept = grown
    kept_ids = np.unique(labels[kept])
    stray = delta & ~kept
    info.update(main_pixels=int(sizes[main]), kept_components=int((kept_ids > 0).sum()),
                stray_pixels=int(stray.sum()))
    if stray.sum() < 24:
        info['reason'] = 'no_stray_change'
        return edited, info
    keep = cv2.GaussianBlur(_dilate(kept, max(8, round(.25 * extent))).astype(np.float32), (0, 0), 3.0)
    keep = np.maximum(keep, kept.astype(np.float32))
    out = source.astype(np.float32) + (edited.astype(np.float32) - source.astype(np.float32)) * keep[..., None]
    info['applied'] = True
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def estimate_blur(gray: np.ndarray, region: np.ndarray, limits=LIMITS):
    """Median edge blur sigma (px) in ``region`` by the re-blur gradient ratio (None if too few edges).

    For a step edge blurred by s, the gradient ratio between the image and its copy
    re-blurred by s0 is R = sqrt(s^2 + d^2 + s0^2) / sqrt(s^2 + d^2) (d: derivative
    pre-smoothing), so s = sqrt(s0^2 / (R^2 - 1) - d^2).
    """
    import cv2
    region = np.asarray(region, bool)
    g = np.asarray(gray, np.float32)
    edges = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8), 40, 100) > 0
    edges &= region
    if edges.sum() < limits['blur_min_edges']:
        return None
    d, s0 = .5, limits['blur_sigma0']

    def grad(sigma):
        smooth = cv2.GaussianBlur(g, (0, 0), sigma)
        return np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))

    ratio = grad(d)[edges] / np.maximum(grad(float(np.hypot(d, s0)))[edges], 1e-6)
    ratio = ratio[ratio > 1.01]
    if ratio.size < limits['blur_min_edges'] // 2:
        return None
    sigma2 = s0 * s0 / (ratio * ratio - 1) - d * d
    return float(np.sqrt(max(0., float(np.median(sigma2)))))


def defocus_match(source: np.ndarray, edited: np.ndarray, weight: np.ndarray, reference: np.ndarray,
                  limits=LIMITS):
    """Blur new content (changed pixels inside ``weight``) to the source's edge blur at ``reference``."""
    import cv2
    region = np.asarray(weight) > 0
    smooth = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), limits['cleanup_sigma'])
    changed = (np.abs(smooth(edited) - smooth(source)).max(axis=2) > limits['cleanup_delta']) & region
    info = {'applied': False}
    if changed.sum() < limits['cleanup_min_main']:
        info['reason'] = 'no_new_content'
        return edited, info
    gray = lambda a: a.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    reference_sigma = estimate_blur(gray(source), _dilate(reference, 4) & ~_dilate(changed, 2), limits)
    if reference_sigma is None:
        # The item covers the anchor: read the focus from the source pixels it replaced.
        reference_sigma = estimate_blur(gray(source), _dilate(reference, 4), limits)
    new_sigma = estimate_blur(gray(edited), _dilate(changed, 2), limits)
    info.update(reference_sigma=None if reference_sigma is None else round(reference_sigma, 2),
                new_sigma=None if new_sigma is None else round(new_sigma, 2), changed_pixels=int(changed.sum()))
    if reference_sigma is None or new_sigma is None:
        info['reason'] = 'no_edges'
        return edited, info
    if reference_sigma < limits['blur_min_reference'] or reference_sigma - new_sigma < limits['blur_min_gap']:
        info['reason'] = 'focus_matches'
        return edited, info
    add = float(min(limits['blur_max'], np.sqrt(reference_sigma ** 2 - new_sigma ** 2)))
    blurred = cv2.GaussianBlur(edited.astype(np.float32), (0, 0), add)
    mix = cv2.GaussianBlur(_dilate(changed, max(2, int(round(2 * add)))).astype(np.float32), (0, 0), max(1., add))
    mix = (np.clip(mix * 1.5, 0, 1) * np.clip(np.asarray(weight, np.float32), 0, 1))[..., None]
    out = edited.astype(np.float32) * (1 - mix) + blurred * mix
    info.update(applied=True, added_sigma=round(add, 2))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def focus_match(source: np.ndarray, edited: np.ndarray, weight: np.ndarray, reference: np.ndarray,
                operation: str = 'add', limits=LIMITS):
    """Blur new content to the focus at its place; at least two independent references must agree.

    References, each an edge-blur estimate on the SOURCE: the anchor band just outside the
    new content, a ring of surroundings at the new content's own scale and, for a
    replacement, the old object's pixels (the same depth as the new one). defocus_match read
    the anchor only, which fails on smooth objects (the white car of F047 read as 1.8 px, so
    the replacing van came out blurrier than the sharp street); a ring alone reads flat
    fabric or the bokeh behind a subject as defocus (a CAFE print on a sharp awning was
    smeared). The smallest of the available estimates is matched.
    """
    import cv2
    region = np.asarray(weight) > 0
    smooth = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), limits['cleanup_sigma'])
    changed = (np.abs(smooth(edited) - smooth(source)).max(axis=2) > limits['cleanup_delta']) & region
    info = {'applied': False}
    if changed.sum() < limits['cleanup_min_main']:
        info['reason'] = 'no_new_content'
        return edited, info
    gray = lambda a: a.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    source_gray = gray(source)
    reference = np.asarray(reference, bool)
    anchor = estimate_blur(source_gray, _dilate(reference, 4) & ~_dilate(changed, 2), limits)
    old = None
    if operation == 'replace':
        old = estimate_blur(source_gray, _erode(reference, 2), limits)
    elif anchor is None:
        # The item covers its anchor: read the focus from the anchor pixels it covered.
        anchor = estimate_blur(source_gray, _dilate(reference, 4), limits)
    ys, xs = np.nonzero(changed)
    extent = max(int(np.ptp(xs)), int(np.ptp(ys))) + 1
    radius = int(np.clip(limits['focus_ring'] * extent, 8, 120))
    ring = estimate_blur(source_gray, _dilate(changed, radius) & ~_dilate(changed, 3), limits)
    new_sigma = estimate_blur(gray(edited), _dilate(changed, 2), limits)
    rounded = lambda v: None if v is None else round(v, 2)
    info.update(anchor_sigma=rounded(anchor), ring_sigma=rounded(ring), old_sigma=rounded(old),
                new_sigma=rounded(new_sigma), changed_pixels=int(changed.sum()))
    references = [v for v in (anchor, ring, old) if v is not None]
    if len(references) < 2 or new_sigma is None:
        info['reason'] = 'no_edges'
        return edited, info
    reference_sigma = min(references)
    info['reference_sigma'] = round(reference_sigma, 2)
    if reference_sigma < limits['focus_min_reference'] or reference_sigma - new_sigma < limits['focus_min_gap']:
        info['reason'] = 'focus_matches'
        return edited, info
    add = float(min(limits['blur_max'], np.sqrt(reference_sigma ** 2 - new_sigma ** 2)))
    blurred = cv2.GaussianBlur(edited.astype(np.float32), (0, 0), add)
    mix = cv2.GaussianBlur(_dilate(changed, max(2, int(round(2 * add)))).astype(np.float32), (0, 0), max(1., add))
    mix = (np.clip(mix * 1.5, 0, 1) * np.clip(np.asarray(weight, np.float32), 0, 1))[..., None]
    out = edited.astype(np.float32) * (1 - mix) + blurred * mix
    info.update(applied=True, added_sigma=round(add, 2))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def detail_restore(source: np.ndarray, edited: np.ndarray, alpha: np.ndarray, protect=None, limits=LIMITS):
    """Put back source pixels that the generator only resampled.

    Generation runs at about 1 MP on the crop and pastes the whole writable support, so
    everything inside it is re-rendered; neighbours that were not edited come back softer
    (full run 4 X091: a woman beside an added plaque lost ~3/4 of her Laplacian detail). At a
    coarse scale resampling leaves the image unchanged, while new or removed content, its
    shadow or a recolour does not. Pixels away from every coarse change take the source.
    ``protect`` pixels always keep the edit. The editor uses this for additions only: around a
    removal or replacement, thin remnants of the old object outside its mask (twigs, wires) that
    the generator cleaned change the coarse image too little and would come back.
    """
    import cv2
    region = np.asarray(alpha) > 0
    info = {'applied': False}
    smooth = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), limits['restore_sigma'])
    delta = np.abs(smooth(edited) - smooth(source)).max(axis=2)
    changed = (delta > limits['restore_delta']) & region
    keep = cv2.GaussianBlur(_dilate(changed, limits['restore_grow']).astype(np.float32), (0, 0), 2.0)
    keep = np.maximum(keep, changed.astype(np.float32))
    if protect is not None:
        # Protected pixels always keep the edit (a soft edge just outside them).
        guard = _dilate(np.asarray(protect, bool), 3)
        keep = np.maximum(keep, np.maximum(cv2.GaussianBlur(guard.astype(np.float32), (0, 0), 1.5), guard))
    restored = region & (keep < .5)
    differs = np.abs(edited.astype(np.int16) - source.astype(np.int16)).max(axis=2) > 6
    info.update(coarse_changed_pixels=int(changed.sum()), restored_pixels=int(restored.sum()),
                restored_differing_pixels=int((restored & differs).sum()))
    if info['restored_differing_pixels'] < limits['restore_min_pixels']:
        info['reason'] = 'nothing_to_restore'
        return edited, info
    out = source.astype(np.float32) + (edited.astype(np.float32) - source.astype(np.float32)) * keep[..., None]
    info['applied'] = True
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), info


def detail_transfer(source: np.ndarray, edited: np.ndarray, alpha: np.ndarray, limits=LIMITS):
    """Give back the photo's fine detail where the generator only damped it.

    The generated pixels are split into coarse and fine parts (Gaussian, transfer_sigma). Where
    the fine part is a damped copy of the source's fine part (local normalised correlation over a
    small window), it is replaced by the source's fine part, scaled by the local brightness ratio
    so that a new contact shadow keeps its texture. New content has detail unrelated to the
    source (or none), so it keeps the generated pixels. Unlike detail_restore there is no hard
    change mask, hence no soft ring around the new item.
    """
    import cv2
    region = np.asarray(alpha, np.float32) > 0
    info = {'applied': False}
    src, out = source.astype(np.float32), edited.astype(np.float32)
    blur = lambda a, s: cv2.GaussianBlur(a, (0, 0), s)
    gray = lambda a: a @ np.array([.299, .587, .114], np.float32)
    sigma, win = limits['transfer_sigma'], limits['transfer_window']
    gs, ge = gray(src), gray(out)
    hs, he = gs - blur(gs, sigma), ge - blur(ge, sigma)
    box = lambda a: cv2.boxFilter(a, -1, (win, win))
    vs, ve, cov = box(hs * hs), box(he * he), box(hs * he)
    ncc = cov / np.sqrt(vs * ve + 1e-3)
    lo, hi = limits['transfer_ncc']
    weight = np.clip((ncc - lo) / (hi - lo), 0, 1)
    weight *= (vs > limits['transfer_min_std'] ** 2) & region
    weight = blur(weight.astype(np.float32), 1.0) * region
    ratio = np.clip(blur(ge, 4.0) / (blur(gs, 4.0) + 1.0), .3, 1.5)
    detail_s = src - blur(src, sigma)
    detail_e = out - blur(out, sigma)
    fixed = out + weight[..., None] * (ratio[..., None] * detail_s - detail_e)
    fixed = np.clip(np.rint(fixed), 0, 255).astype(np.uint8)
    moved = np.abs(fixed.astype(np.int16) - edited.astype(np.int16)).max(axis=2) > 4
    info.update(transferred_pixels=int((weight > .5).sum()), changed_pixels=int(moved.sum()))
    if info['changed_pixels'] < limits['transfer_min_pixels']:
        info['reason'] = 'nothing_to_transfer'
        return edited, info
    info['applied'] = True
    return fixed, info

