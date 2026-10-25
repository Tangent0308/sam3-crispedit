"""Untouched scene plus two independently labelled, untinted mask crops."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .fonts import load_font
from .full_contour_views import COLORS

VERSION = 'clean-twin-context-crops-1'
INTRO = ('Image 1: the original full photo without annotations. Image 2: a nearby-context '
         'crop of Mask U1, outlined in pink. Image 3: a nearby-context crop of Mask U2, '
         'outlined in cyan. Each crop outlines ONLY its own mask; holes are excluded and '
         'disconnected outlined pieces belong to that mask. Crop orientation is unchanged. '
         'Match each crop to Image 1; ref positions and ordinals must refer to the FULL photo, '
         'not crop order.\n')


def _interval(low, high, limit, pad):
    span = min(limit, max(160, high - low + 2 * pad))
    start = min(max(0, (low + high - span) // 2), limit - span)
    return int(start), int(start + span)


def mask_context_views(source, masks, unit_ids=None):
    source = source.convert('RGB')
    unit_ids = unit_ids or [f'U{i+1}' for i in range(len(masks))]
    if not masks or len(unit_ids) != len(masks) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError('mask crop IDs must be nonempty, unique and aligned')
    views, metadata = [source.copy()], []
    for index, (raw_mask, color) in enumerate(zip(masks, COLORS), 1):
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (source.height, source.width) or not mask.any():
            raise ValueError('planning masks must be nonempty and source-aligned')
        ys, xs = np.nonzero(mask)
        pad = max(32, round(.6 * max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)))
        left, right = _interval(int(xs.min()), int(xs.max())+1, source.width, pad)
        top, bottom = _interval(int(ys.min()), int(ys.max())+1, source.height, pad)
        box = (left, top, right, bottom)
        crop = source.crop(box)
        scale = min(4., 768 / max(crop.size))
        size = tuple(max(1, round(v * scale)) for v in crop.size)
        crop = crop.resize(size, Image.Resampling.LANCZOS)
        binary = Image.fromarray(mask[top:bottom, left:right].astype('uint8')*255)
        binary = binary.resize(size, Image.Resampling.NEAREST)
        frame = 6
        padded = Image.new('L', (size[0]+2*frame, size[1]+2*frame))
        padded.paste(binary, (frame, frame))
        binary = padded
        selected = np.asarray(binary) > 0
        outside = np.asarray(binary.filter(ImageFilter.MaxFilter(7))) > 0
        halo = np.asarray(binary.filter(ImageFilter.MaxFilter(11))) > 0
        pixels = np.full((size[1]+2*frame, size[0]+2*frame, 3), (12, 15, 20), dtype='uint8')
        pixels[frame:frame+size[1], frame:frame+size[0]] = np.array(crop)
        pixels[halo & ~outside] = (10, 10, 10)
        pixels[outside & ~selected] = color
        # A separate title band keeps labels off the target and its context.
        header = 48
        canvas = Image.new('RGB', (size[0]+2*frame, size[1]+2*frame+header), (12, 15, 20))
        canvas.paste(Image.fromarray(pixels), (0, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 6), f'Mask {unit_ids[index-1]}', font=load_font(26, bold=True), fill=color)
        views.append(canvas)
        metadata.append(dict(unit_id=unit_ids[index-1], color=list(color), crop_box_xyxy=list(box),
                             resized_size=list(size), header_height=header, body_offset=[frame, header+frame],
                             mask_bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1],
                             source_mask_pixels=int(mask.sum()), selected_pixels_under_label=0))
    return views, metadata


def twin_crop_views(source, masks):
    if len(masks) != 2:
        raise ValueError('three-image twin crops require exactly two masks')
    return mask_context_views(source, masks)
