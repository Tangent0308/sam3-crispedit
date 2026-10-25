"""Opt-in: find a removed object's own cast shadow / reflection so removal can erase them too.

SAM3 text prompts ("shadow", "reflection") propose regions on a context crop;
plain geometric and photometric rules then decide which proposals belong to the
target:

* shadow:     touches the target, lies mostly outside it and not above it, is
              darker than its surroundings and is not scene-scale (area bounded
              by the target's); only the part within reach of the target is kept.
* reflection: the part of a reflection proposal that covers the target's
              vertically mirrored footprint below its waterline, and only when
              that footprint lies on a SAM3 "water" region (a bush or pavement
              below an object is not a mirror).

The source mask and the instruction never change. The effect region only widens
the writable support of a whole-object removal (and adds one generator
sentence), so a removal no longer leaves its shadow or reflection behind.
"""
from __future__ import annotations

import time

import numpy as np

VERSION = 'removal-effects-2-water-mirror'

LIMITS = {
    'score': .35,              # SAM3 instance score for "shadow"
    'reflection_score': .4,    # SAM3 instance score for "reflection"
    'context': 1.0,            # crop margin, x target max extent
    'contact': .04,            # touching radius, x target min extent ...
    'contact_min': 4,          # ... at least this many pixels
    'outside_min': .6,         # a shadow proposal must lie mostly outside the target
    'below': 0.0,              # shadow centroid row >= target centroid row - this x target height
    'darker': .88,             # shadow mean luminance <= this x surrounding median
    'reach': 1.0,              # keep shadow pixels within this x max extent of the target
    'max_area': 2.5,           # attached shadow area <= this x target area
    'mirror_dilate': .06,      # mirrored footprint tolerance, x max extent
    'mirror_cover': .25,       # reflection must cover this fraction of the mirrored footprint
    'reflection_max_area': 1.5,
    'water_score': .35,        # SAM3 instance score for "water"
    'water_cover': .5,         # mirrored footprint inside water (fraction)
    'penumbra': 2,             # final dilation in pixels (soft shadow edge)
    'min_pixels': 24,
}


def _disk(radius: int):
    import cv2
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2
    if radius <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0


def _extents(target: np.ndarray):
    ys, xs = np.nonzero(target)
    if not len(xs):
        raise ValueError('empty removal target')
    h, w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    return ys, xs, min(h, w), max(h, w)


def _attached(piece: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Connected components of ``piece`` that touch ``near``."""
    from scipy.ndimage import label
    labels, count = label(piece)
    if not count:
        return piece
    keep = np.unique(labels[near & piece])
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def _mirror(target: np.ndarray, line: float) -> np.ndarray:
    """Vertical mirror of the target pixels above ``line`` (image rows grow downward)."""
    out = np.zeros_like(target)
    ys, xs = np.nonzero(target)
    keep = ys <= line
    ny = np.rint(2 * line - ys[keep]).astype(int)
    ok = (ny >= 0) & (ny < target.shape[0])
    out[ny[ok], xs[keep][ok]] = True
    return out


def associate(target: np.ndarray, shadows, reflections, gray: np.ndarray, limits=LIMITS,
              water: np.ndarray | None = None):
    """Pure association on one crop.

    target: bool HxW; shadows / reflections: iterables of (bool HxW, score);
    gray: float HxW luminance; water: bool HxW (None = unknown, reflections rejected).
    Returns (effect bool HxW, per-proposal records).
    """
    from scipy.ndimage import distance_transform_edt
    target = np.asarray(target, dtype=bool)
    ys, xs, ext_min, ext_max = _extents(target)
    area = int(target.sum())
    # A dark groove or seam above the object is structure, not its cast shadow.
    lowest_centroid = ys.mean() - limits['below'] * (ys.max() - ys.min() + 1)
    near = _dilate(target, max(limits['contact_min'], round(limits['contact'] * ext_min)))
    reach = distance_transform_edt(~target) <= limits['reach'] * ext_max
    all_shadow = np.zeros_like(target)
    for mask, _ in shadows:
        all_shadow |= np.asarray(mask, dtype=bool)
    effect = np.zeros_like(target)
    records = []
    for index, (mask, score) in enumerate(shadows):
        mask = np.asarray(mask, dtype=bool)
        rec = {'kind': 'shadow', 'index': index, 'score': round(float(score), 3), 'pixels': int(mask.sum())}
        out = mask & ~target
        if score < limits['score']:
            rec['reason'] = 'low_score'
        elif out.sum() < limits['min_pixels']:
            rec['reason'] = 'tiny'
        elif out.sum() < limits['outside_min'] * mask.sum():
            rec['reason'] = 'mostly_inside_target'
        elif not (out & near).any():
            rec['reason'] = 'not_touching'
        else:
            piece = _attached(out & reach, near)
            ring = _dilate(piece, 6) & ~piece & ~target & ~all_shadow
            rec['attached_pixels'] = int(piece.sum())
            if piece.sum() < limits['min_pixels']:
                rec['reason'] = 'tiny'
            elif piece.sum() > limits['max_area'] * area:
                rec['reason'] = 'scene_scale'
            elif np.nonzero(piece)[0].mean() < lowest_centroid:
                rec['reason'] = 'above_target'
            elif ring.sum() < limits['min_pixels']:
                rec['reason'] = 'no_lit_surround'
            else:
                ratio = float(gray[piece].mean() / max(1e-6, np.median(gray[ring])))
                rec['luminance_ratio'] = round(ratio, 3)
                if ratio > limits['darker']:
                    rec['reason'] = 'not_darker'
                else:
                    rec['reason'] = 'accepted'
                    effect |= piece
        records.append(rec)
    x0, x1, yb = int(xs.min()), int(xs.max()), int(ys.max())
    for index, (mask, score) in enumerate(reflections):
        mask = np.asarray(mask, dtype=bool)
        rec = {'kind': 'reflection', 'index': index, 'score': round(float(score), 3), 'pixels': int(mask.sum())}
        out = mask & ~target
        below = out[:, x0:x1 + 1].any(axis=1)
        rows = np.nonzero(below[int(ys.mean()):])[0]
        if score < limits['reflection_score']:
            rec['reason'] = 'low_score'
        elif out.sum() < limits['min_pixels'] or not len(rows):
            rec['reason'] = 'not_below'
        else:
            top = int(ys.mean()) + int(rows[0])
            line = (yb + max(yb, top)) / 2
            mirror = _mirror(target, line) & ~target
            piece = out & _dilate(mirror, max(3, round(limits['mirror_dilate'] * ext_max)))
            cover = float((piece & mirror).sum() / max(1, mirror.sum()))
            on_water = float((mirror & water).sum() / max(1, mirror.sum())) if water is not None else 0.
            rec.update(waterline=round(line, 1), mirror_pixels=int(mirror.sum()),
                       cover=round(cover, 3), attached_pixels=int(piece.sum()), on_water=round(on_water, 3))
            if mirror.sum() < limits['min_pixels']:
                rec['reason'] = 'mirror_outside_frame'
            elif cover < limits['mirror_cover']:
                rec['reason'] = 'mirror_not_covered'
            elif on_water < limits['water_cover']:
                rec['reason'] = 'not_on_water'
            elif piece.sum() > limits['reflection_max_area'] * area:
                rec['reason'] = 'scene_scale'
            else:
                rec['reason'] = 'accepted'
                effect |= piece
        records.append(rec)
    if effect.any():
        effect = _dilate(effect, limits['penumbra']) & ~target
    return effect, records


class EffectFinder:
    """SAM3 text-prompt proposals + :func:`associate`; loaded once per editor worker."""

    PROMPTS = ('shadow', 'reflection')
    WATER = 'water'

    def __init__(self, model_path: str, device: str = 'cuda', limits=None):
        import torch
        from transformers import Sam3Model, Sam3Processor
        from .model_cache import resolve_model
        started = time.perf_counter()
        local = resolve_model(model_path)
        self.model = Sam3Model.from_pretrained(local, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
        self.processor = Sam3Processor.from_pretrained(local, local_files_only=True)
        self.device, self.model_path = device, str(model_path)
        self.limits = dict(LIMITS, **(limits or {}))
        self.load_seconds = time.perf_counter() - started

    def config(self) -> dict:
        return {'version': VERSION, 'model': self.model_path, 'limits': self.limits}

    def _segment(self, crop, prompt: str):
        import torch
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.processor(images=crop, text=prompt, return_tensors='pt').to(self.device)
            outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs, threshold=min(self.limits['score'], self.limits['reflection_score'], self.limits['water_score']),
                mask_threshold=.5, target_sizes=[crop.size[::-1]])[0]
        return [(m.cpu().numpy().astype(bool), float(s)) for m, s in zip(result['masks'], result['scores'])]

    def __call__(self, image, target: np.ndarray, protected: np.ndarray | None = None):
        started = time.perf_counter()
        target = np.asarray(target, dtype=bool)
        ys, xs, _, ext_max = _extents(target)
        margin = max(32, round(self.limits['context'] * ext_max))
        box = (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
               min(image.width, int(xs.max()) + margin + 1), min(image.height, int(ys.max()) + margin + 1))
        crop = image.convert('RGB').crop(box)
        rgb = np.asarray(crop, dtype=np.float32)
        gray = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
        proposals = {p: self._segment(crop, p) for p in self.PROMPTS}
        water = None
        if any(score >= self.limits['reflection_score'] for _, score in proposals['reflection']):
            # Only a reflection candidate needs the (cheap) water check.
            water = np.zeros(gray.shape, bool)
            for mask, score in self._segment(crop, self.WATER):
                if score >= self.limits['water_score']:
                    water |= mask
        local, records = associate(target[box[1]:box[3], box[0]:box[2]], proposals['shadow'],
                                   proposals['reflection'], gray, self.limits, water=water)
        effect = np.zeros_like(target)
        effect[box[1]:box[3], box[0]:box[2]] = local
        if protected is not None:
            effect &= ~np.asarray(protected, dtype=bool)
        info = {'version': VERSION, 'crop_box': list(box), 'pixels': int(effect.sum()),
                'target_pixels': int(target.sum()),
                'accepted': sorted({r['kind'] for r in records if r['reason'] == 'accepted'}),
                'proposals': records, 'seconds': round(time.perf_counter() - started, 3)}
        return effect, info


def effect_support(effect: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Writable support for an effect region: the region plus a small blending collar."""
    _, _, _, ext_max = _extents(np.asarray(target, dtype=bool))
    return _dilate(np.asarray(effect, dtype=bool), int(np.clip(round(.05 * ext_max), 4, 16)))
