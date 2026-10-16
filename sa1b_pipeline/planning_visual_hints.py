"""Experimental visual callouts; no target selection, mask edits, or model calls."""
import numpy as np
from .fonts import load_font
from PIL import Image, ImageDraw, ImageFont

from .geometry import load_mask
from .per_mask import _outer_mask_band


def annotate_full_scene(source_path, source_mask_path, destination, max_edge=1280):
    """Show the exact target in scene context without recolouring its pixels."""
    with Image.open(source_path) as source:
        scene = source.convert('RGB')
    scene.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    with Image.open(source_mask_path) as source_mask:
        mask = source_mask.convert('L').resize(scene.size, Image.Resampling.NEAREST)
    outer = _outer_mask_band(mask, 9)
    ring = _outer_mask_band(mask, 5)
    scene.paste('white', mask=Image.fromarray(outer.astype('uint8') * 255))
    scene.paste((255, 0, 180), mask=Image.fromarray(ring.astype('uint8') * 255))
    scene.save(destination, compress_level=1)


def _tags(obj):
    return {x.strip().lower() for x in str(obj.get('category', '')).split(',')} - {'', 'object', 'part'}


def nearest_peer(target, objects, crop_box):
    """Find an optional comparison region, never remove a planning target.

    Source categories and boxes are only fallible retrieval hints. A retrieved
    region is not a verified semantic twin or evidence of a complete group.
    """
    box = target.get('bbox')
    if not box or len(box) != 4:
        return None
    b = np.array(box, dtype=float)
    extent = b[2:] - b[:2]
    if np.any(extent <= 0):
        return None
    center = (b[:2] + b[2:]) / 2
    area = float(np.prod(extent))
    candidates = []
    for obj in objects:
        if str(obj['object_id']) == str(target['object_id']) or not (_tags(target) & _tags(obj)):
            continue
        if not obj.get('bbox') or len(obj['bbox']) != 4:
            continue
        p = np.array(obj['bbox'], dtype=float)
        dims = p[2:] - p[:2]
        if np.any(dims <= 0):
            continue
        other_area = float(np.prod(dims))
        overlap = np.prod(np.maximum(0, np.minimum(b[2:], p[2:]) - np.maximum(b[:2], p[:2])))
        visible = np.prod(np.maximum(0, np.minimum(crop_box[2:], p[2:]) - np.maximum(crop_box[:2], p[:2])))
        if not .2 <= other_area / area <= 5 or overlap / min(area, other_area) > .35 or visible / other_area < .5:
            continue
        distance = float(np.linalg.norm((p[:2] + p[2:]) / 2 - center))
        candidates.append((distance, str(obj['object_id']), obj))
    return min(candidates, key=lambda x: x[:2])[2] if candidates else None


def annotate_crop(path, source_mask_path, source_size, crop_box, target, objects, mode):
    """Add callouts outside selected pixels; preserve both source files and A RGB."""
    if mode not in {'arrow', 'peer'}:
        raise ValueError('unknown visual hint mode')
    panel = Image.open(path).convert('RGB')
    size = (panel.width, panel.height - 38)
    target_full = np.asarray(Image.open(source_mask_path).convert('L')) > 0
    target_local = np.asarray(Image.fromarray(target_full).crop(crop_box).resize(size, Image.Resampling.NEAREST))
    peer = nearest_peer(target, objects, crop_box) if mode == 'peer' else None
    peer_local = np.zeros(target_local.shape, dtype=bool)
    evidence = {'mode': mode, 'target_label': 'A', 'peer': None}
    if peer is not None:
        peer_full = load_mask(peer, source_size)
        # B must be separate from A in the actual masks, not just their boxes.
        intersection = int((peer_full & target_full).sum())
        if peer_full.any() and intersection / max(1, min(int(peer_full.sum()), int(target_full.sum()))) <= .1:
            peer_local = np.asarray(Image.fromarray(peer_full).crop(crop_box).resize(size, Image.Resampling.NEAREST))
            ay, ax = np.nonzero(target_full)
            by, bx = np.nonzero(peer_full)
            evidence['peer'] = {
                'label': 'B', 'category_hint': peer.get('category', ''),
                'target_center_xy_fraction': [round(float(ax.mean()) / source_size[0], 4), round(float(ay.mean()) / source_size[1], 4)],
                'peer_center_xy_fraction': [round(float(bx.mean()) / source_size[0], 4), round(float(by.mean()) / source_size[1], 4)],
                'target_minus_peer_center_xy_pixels': [round(float(ax.mean() - bx.mean()), 1), round(float(ay.mean() - by.mean()), 1)],
                'retrieval_basis': 'nearby nonoverlapping dataset region with an overlapping category tag; not a verified semantic identity or complete instance inventory',
            }
    body = panel.crop((0, 38, panel.width, panel.height))
    original = np.array(body)
    overlay = Image.new('RGB', size)
    coverage = Image.new('L', size)
    draw, alpha = ImageDraw.Draw(overlay), ImageDraw.Draw(coverage)
    font = load_font(24, bold=True)

    def callout(mask, label, color):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            return
        # Use an occupied column nearest the centroid, then its first selected
        # pixel. The callout approaches the actual mask, not its bounding box.
        x = int(xx[np.argmin(abs(xx - xx.mean()))])
        y = int(np.flatnonzero(mask[:, x])[0])
        bottom = max(1, y - 7)
        top = max(1, bottom - 62)
        for d, c in [(draw, color), (alpha, 255)]:
            d.line([(x, top + 26), (x, bottom)], fill=c, width=4)
            d.line([(x - 9, bottom - 12), (x, bottom), (x + 9, bottom - 12)], fill=c, width=4)
            d.text((x - 10, top), label, fill=c, font=font)

    if peer_local.any():
        ring = _outer_mask_band(Image.fromarray(peer_local.astype('uint8') * 255), 5)
        overlay.paste((90, 255, 60), mask=Image.fromarray(ring.astype('uint8') * 255))
        coverage.paste(255, mask=Image.fromarray(ring.astype('uint8') * 255))
        callout(peer_local, 'B', (90, 255, 60))
    callout(target_local, 'A', (255, 0, 180))
    coverage_arr = np.array(coverage)
    coverage_arr[target_local | peer_local] = 0
    body.paste(overlay, mask=Image.fromarray(coverage_arr))
    assert np.array_equal(np.asarray(body)[target_local], original[target_local])
    panel.paste(body, (0, 38))
    head = ImageDraw.Draw(panel)
    head.rectangle((0, 0, panel.width, 37), fill='white')
    text = 'A MAGENTA = TARGET; B GREEN = COMPARISON ONLY' if peer_local.any() else 'A MAGENTA = TARGET TO EDIT'
    head.text((7, 8), text, fill='black', font=ImageFont.load_default(size=14))
    panel.save(path, compress_level=1)
    return evidence


def visual_hint_prompt(prompt, evidence):
    import json
    note = ('VISUAL CALLOUTS: In Image 2, A and the MAGENTA boundary identify the only edit target. '
            'The arrow ends just outside its real pixels. If B appears, its thin GREEN boundary identifies '
            'a nearby comparison region, NOT another editing target. Name what is inside A, then use B '
            'only to check whether your reference excludes a similar neighbour. B is retrieved from fallible '
            'dataset tags, so it may be a different kind of object. Computed center differences can corroborate '
            'image-left/right or above/below; they do not establish an ordinal or a complete group. '
            'The final reference must work on the unmarked original: never mention A, B, arrows or annotation colours. '
            'If useful, describe the actual visible neighbour and its relation, not its artificial label.\n')
    return note + prompt + '\nVisual callout metadata: ' + json.dumps(evidence, ensure_ascii=False)
