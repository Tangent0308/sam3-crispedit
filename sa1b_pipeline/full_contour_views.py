"""Two full-frame planning inputs: untouched RGB source and labelled contours.

Never fill a mask or use its bounding box as its contour. Holes and disconnected
pieces are retained. Labels avoid selected pixels where space permits; leaders
terminate on an actual mask boundary, not on a bounding-box centre or a hole.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .fonts import load_font

VERSION = 'clean-full-contours-1'
COLORS = ((255, 60, 110), (0, 220, 255), (255, 210, 0),
          (110, 255, 70), (180, 105, 255), (255, 135, 30))
INTRO = ('Image 1: the original photo without annotations. Image 2: the SAME full photo '
         'with differently colored mask contours and matching Mask U labels. Each label\'s '
         'leader ends on its mask boundary. Holes are excluded; disconnected contours with '
         'the same color belong to the same mask. IDs are not scene order.\n')


def contour_pair(source, masks, *, unit_ids=None):
    source = source.convert('RGB')
    if not masks or len(masks) > len(COLORS):
        raise ValueError('full contours require 1 to 6 masks with distinct colors')
    unit_ids = list(unit_ids) if unit_ids is not None else [f'U{i+1}' for i in range(len(masks))]
    if len(unit_ids) != len(masks) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError('mask IDs must be unique and aligned')
    masks = [np.asarray(m, dtype=bool) for m in masks]
    if any(m.shape != (source.height, source.width) or not m.any() for m in masks):
        raise ValueError('planning masks must be nonempty and pixel-aligned with source')
    scale = max(source.size) / 1536
    width = max(2, round(3 * scale))
    font = load_font(max(14, round(24 * scale)), bold=True)
    gap = max(4, round(8 * scale))
    out = np.array(source)
    bounds = []
    for mask, color in zip(masks, COLORS):
        binary = Image.fromarray(mask.astype('uint8') * 255)
        outside = np.asarray(binary.filter(ImageFilter.MaxFilter(width * 2 + 1))) > 0
        halo = np.asarray(binary.filter(ImageFilter.MaxFilter((width + 1) * 2 + 1))) > 0
        # Outside-only ink preserves selected colors, including very small masks.
        out[halo & ~outside] = (10, 10, 10)
        out[outside & ~mask] = color
        inside_edge = mask & ~(np.asarray(binary.filter(ImageFilter.MinFilter(3))) > 0)
        # Border-touching/full-image masks also need boundary points for leaders.
        inside_edge[0] |= mask[0]; inside_edge[-1] |= mask[-1]
        inside_edge[:, 0] |= mask[:, 0]; inside_edge[:, -1] |= mask[:, -1]
        ys, xs = np.nonzero(inside_edge)
        bounds.append(np.column_stack((xs, ys)))
    marked = Image.fromarray(out)
    draw = ImageDraw.Draw(marked)
    union = np.logical_or.reduce(masks)
    integral = np.pad(union.astype('int64'), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    occupied, labels = [], []
    for i, (mask, boundary, color) in enumerate(zip(masks, bounds, COLORS)):
        text = f'Mask {unit_ids[i]}'
        box = draw.textbbox((0, 0), text, font=font)
        bw, bh = box[2] - box[0] + 2 * gap, box[3] - box[1] + 2 * gap
        ys, xs = np.nonzero(mask)
        left, right, top, bottom = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (left + right) / 2, (top + bottom) / 2
        candidates = [(left - bw - gap, cy - bh/2), (right + gap, cy - bh/2),
                      (cx - bw/2, top - bh - gap), (cx - bw/2, bottom + gap)]
        # Extra candidates handle dense adjacent masks and edge-touching targets.
        for px, py in boundary[::max(1, len(boundary)//32)]:
            candidates.extend(((px-bw-gap, py-bh/2), (px+gap, py-bh/2),
                               (px-bw/2, py-bh-gap), (px-bw/2, py+gap)))
        best = None
        for x, y in candidates:
            x = int(np.clip(round(x), 0, max(0, source.width-bw-1)))
            y = int(np.clip(round(y), 0, max(0, source.height-bh-1)))
            x2, y2 = min(source.width, x+bw), min(source.height, y+bh)
            overlap = integral[y2, x2]-integral[y, x2]-integral[y2, x]+integral[y, x]
            label_overlap = sum(max(0, min(x2,b[2])-max(x,b[0])) *
                                max(0, min(y2,b[3])-max(y,b[1])) for b in occupied)
            dx = np.maximum(np.maximum(x-boundary[:,0], boundary[:,0]-x2), 0)
            dy = np.maximum(np.maximum(y-boundary[:,1], boundary[:,1]-y2), 0)
            distances = dx*dx + dy*dy
            j = int(np.argmin(distances))
            score = 1000*label_overlap + 100*overlap + float(np.sqrt(distances[j]))
            if best is None or score < best[0]:
                best = (score, (x,y,x2,y2), tuple(map(int,boundary[j])))
        _, rect, endpoint = best
        occupied.append(rect)
        start = (int(np.clip(endpoint[0], rect[0], rect[2])),
                 int(np.clip(endpoint[1], rect[1], rect[3])))
        labels.append(dict(unit_id=unit_ids[i], color=list(color), box=list(rect),
                           leader_start=list(start), leader_end=list(endpoint),
                           selected_pixels_under_label=int(union[rect[1]:rect[3],rect[0]:rect[2]].sum())))
    # Leaders first, then opaque labels: line crossings cannot obscure text.
    for label in labels:
        draw.line([tuple(label['leader_start']),tuple(label['leader_end'])], fill=(10,10,10), width=width+2)
        draw.line([tuple(label['leader_start']),tuple(label['leader_end'])], fill=tuple(label['color']), width=width)
    for label in labels:
        x,y,x2,y2 = label['box']; color = tuple(label['color'])
        draw.rounded_rectangle((x,y,x2,y2), radius=gap, fill=(12,15,20), outline=color, width=width)
        text = f"Mask {label['unit_id']}"
        box = draw.textbbox((0,0),text,font=font)
        draw.text((x+gap,y+gap-box[1]), text, font=font, fill=color)
    return source.copy(), marked, labels
