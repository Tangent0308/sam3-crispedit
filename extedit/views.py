"""Images shown to the 27B model: resized full views, outlined areas, BEFORE / AFTER close-ups, review sheets.

Ported from samtok_edit v2 src/samtok_edit21/preparation/edit_check.py (resized, crop_box, labeled, data_url) and
protocol_adapt.py (outlined, close_up), with the same sizes, colors and JPEG quality.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

FULL_PIXELS = 640 * 640          # full BEFORE / AFTER views
BOX_PIXELS = 1024 * 1024         # the BEFORE view the model draws boxes on
CLOSE_LONG_SIDE = 1400           # side-by-side close-up of the edit
CLOSE_MAX_SCALE = 3.0
JPEG_QUALITY = 92
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
PALETTE = (("red", (235, 50, 45)), ("blue", (40, 110, 240)), ("yellow", (250, 210, 0)),
           ("magenta", (220, 40, 220)), ("cyan", (0, 200, 220)), ("orange", (255, 140, 0)),
           ("green", (40, 190, 60)), ("white", (250, 250, 250)))
S_COLOR, EXEC_COLOR = (40, 220, 60), (255, 150, 0)


def font(size):
    path = os.environ.get("EXTEDIT_FONT") or FONT
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def resized(image, max_pixels):
    image = image.convert("RGB")
    scale = min(1.0, (max_pixels / (image.width * image.height)) ** 0.5)
    if scale >= 1:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def crop_box(box_1000, size, pad=0.25, min_frac=0.2):
    """Box grown by `pad` of its size on each side, at least `min_frac` of the long image side, inside the image."""
    W, H = size
    x1, y1, x2, y2 = box_1000
    x1, x2, y1, y2 = x1 / 1000 * W, x2 / 1000 * W, y1 / 1000 * H, y2 / 1000 * H
    w, h = x2 - x1, y2 - y1
    side = max(max(w, h) * (1 + 2 * pad), min_frac * max(W, H))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = min(max(side, w * (1 + 2 * pad)), W), min(max(side, h * (1 + 2 * pad)), H)
    left = min(max(0, cx - bw / 2), W - bw)
    top = min(max(0, cy - bh / 2), H - bh)
    return round(left), round(top), round(left + bw), round(top + bh)


def labeled(image, text, size=22):
    out = Image.new("RGB", (image.width, image.height + size + 12), "white")
    out.paste(image, (0, size + 12))
    ImageDraw.Draw(out).text((8, 6), text, font=font(size), fill=(0, 0, 0))
    return out


def data_url(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def outlined(image, masks, max_pixels, numbers=True):
    """Each (mask, color) as a colored contour with a black halo, numbered from 1 in list order."""
    image = resized(image, max_pixels)
    width = max(2, round(max(image.size) / 400))
    labels = []
    for number, (mask, color) in enumerate(masks, 1):
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
        outer, inner = m.filter(ImageFilter.MaxFilter(2 * (width + 1) + 1)), m.filter(ImageFilter.MaxFilter(2 * width + 1))
        image.paste((0, 0, 0), mask=ImageChops.subtract(outer, m))
        image.paste(color, mask=ImageChops.subtract(inner, m))
        box = m.getbbox()
        if numbers and box:
            labels.append((number, color, box))
    draw, face = ImageDraw.Draw(image), font(max(14, round(max(image.size) / 40)))
    for number, color, (x, y, _, _) in labels:
        text = str(number)
        w, h = draw.textbbox((0, 0), text, font=face)[2:]
        x, y = min(max(0, x), image.width - w - 6), max(0, y - h - 6)
        draw.rectangle((x, y, x + w + 6, y + h + 6), fill=color, outline=(0, 0, 0))
        draw.text((x + 3, y + 1), text, font=face, fill=(0, 0, 0))
    return image


def crop_masks(masks, box):
    x1, y1, x2, y2 = box
    return [(mask[y1:y2, x1:x2], color) for mask, color in masks]


def close_up(source, target, masks, box_1000, max_long=CLOSE_LONG_SIDE, left="BEFORE (close-up)",
             right="AFTER (close-up)", after_masks=None, numbers=False):
    box = crop_box(box_1000, source.size)
    before = outlined(source.crop(box), crop_masks(masks, box), 10 ** 9, numbers=numbers)
    after = target.convert("RGB").crop(box)
    if after_masks:
        after = outlined(after, crop_masks(after_masks, box), 10 ** 9, numbers=False)
    scale = min((max_long / 2 - 10) / max(before.width, before.height), CLOSE_MAX_SCALE)
    size = (max(1, round(before.width * scale)), max(1, round(before.height * scale)))
    before = labeled(before.resize(size, Image.Resampling.LANCZOS), left)
    after = labeled(after.resize(size, Image.Resampling.LANCZOS), right)
    canvas = Image.new("RGB", (before.width * 2 + 20, before.height), "white")
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width + 20, 0))
    return canvas


def union_box(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def chat_parts(views):
    """[(label, image)] as OpenAI chat content parts."""
    parts = []
    for label, image in views:
        parts += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": data_url(image)}}]
    return parts


def wrap(draw, text, face, width):
    lines, line = [], ""
    for word in str(text).split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=face) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + ([line] if line else [])


def sheet(panels, lines, panel_height=420):
    """Review sheet: image panels side by side with captions, then text lines."""
    panels = [(caption, image.convert("RGB")) for caption, image in panels]
    scaled = []
    for caption, image in panels:
        s = panel_height / image.height
        scaled.append((caption, image.resize((max(1, round(image.width * s)), panel_height), Image.Resampling.LANCZOS)))
    width = sum(i.width for _, i in scaled) + 10 * (len(scaled) + 1)
    face, small = font(18), font(16)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    text = [l for line in lines for l in wrap(probe, line, small, width - 20)]
    out = Image.new("RGB", (width, panel_height + 40 + 22 * len(text) + 10), "white")
    draw, x = ImageDraw.Draw(out), 10
    for caption, image in scaled:
        draw.text((x, 8), caption, font=face, fill=(0, 0, 0))
        out.paste(image, (x, 34))
        x += image.width + 10
    y = panel_height + 44
    for line in text:
        draw.text((10, y), line, font=small, fill=(0, 0, 0))
        y += 22
    return out
