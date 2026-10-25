"""Pixel-aligned crop, outside contour, and mask without tinting target colors."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .fonts import load_font

VERSION = 'aligned-original-outline-binary-1'


def crop_triplet(source, mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty planning mask')
    extent = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    pad = max(32, round(extent * .5))
    box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
           min(source.width, int(xs.max()) + pad + 1), min(source.height, int(ys.max()) + pad + 1))
    original = source.convert('RGB').crop(box)
    binary = Image.fromarray(mask[box[1]:box[3], box[0]:box[2]].astype('uint8') * 255)
    scale = min(512 / original.width, 400 / original.height)
    size = (max(1, round(original.width * scale)), max(1, round(original.height * scale)))
    original = original.resize(size, Image.Resampling.LANCZOS)
    binary = binary.resize(size, Image.Resampling.NEAREST)
    selected = np.asarray(binary) > 0
    outside = (np.asarray(binary.filter(ImageFilter.MaxFilter(5))) > 0) & ~selected
    outlined = np.array(original)
    outlined[outside] = [0, 255, 100]
    return original, Image.fromarray(outlined), binary.convert('RGB')


def aligned_sheet(source, masks):
    sheet = Image.new('RGB', (1536, 436 * len(masks)), (30, 35, 42))
    draw = ImageDraw.Draw(sheet)
    for i, mask in enumerate(masks):
        for c, (tile, label) in enumerate(zip(crop_triplet(source, mask),
                ('ORIGINAL CROP', 'OUTSIDE CONTOUR', 'WHITE = SELECTED'))):
            y = i * 436
            draw.text((c * 512 + 10, y + 6), f'U{i+1} | {label}', font=load_font(19), fill='white')
            sheet.paste(tile, (c * 512 + (512 - tile.width) // 2, y + 36 + (400 - tile.height) // 2))
    return sheet
