#!/usr/bin/env python3
"""Validate every selected row and export a small, curated ScaleEdit gallery."""
import argparse
import base64
from collections import Counter
import html
import io
import json
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from tqdm import tqdm

from scaleedit.runner import decode, load_selection


GROUPS = (
    ('quality_keep', 'Quality prefilter — KEEP', 'quality', 'PASS'),
    ('quality_drop', 'Quality prefilter — DROP', 'quality', 'DROP'),
    ('scene_keep', 'Fine-grained prefilter — KEEP', 'scene', 'PASS'),
    ('scene_drop', 'Fine-grained prefilter — DROP', 'scene', 'DROP'),
    ('mask', 'Mask labeling', 'mask', 'OK'),
)


def image_uri(image):
    image = image.copy()
    image.thumbnail((800, 640))
    buffer = io.BytesIO()
    mime = 'png' if image.mode == 'L' else 'jpeg'
    if mime == 'png':
        image.save(buffer, 'PNG')
    else:
        image.convert('RGB').save(buffer, 'JPEG', quality=88)
    return f'data:image/{mime};base64,' + base64.b64encode(buffer.getvalue()).decode()


def audit_rows(root, shard):
    path = root / shard
    return {row['row_idx']: row for row in pq.read_table(path).to_pylist()} if path.exists() else {}


def case_key(shard, row_idx):
    return f'{shard}:{row_idx}'


def annotated_images(source, target, payload):
    source_boxes, target_boxes = source.copy(), target.copy()
    for side, image, color in (
        ('source', source_boxes, 'cyan'), ('target', target_boxes, 'orange')
    ):
        draw = ImageDraw.Draw(image)
        for box in payload.get('boxes', {}).get(side, []):
            x1, y1, x2, y2 = box['bbox_2d']
            xy = (
                x1 * image.width / 1000, y1 * image.height / 1000,
                x2 * image.width / 1000, y2 * image.height / 1000,
            )
            draw.rectangle(xy, outline=color, width=3)
            if box.get('polygon_2d'):
                points = [(x * image.width / 1000, y * image.height / 1000)
                          for x, y in box['polygon_2d']]
                draw.line(points + [points[0]], fill='lime', width=3)
            draw.text(xy[:2], str(box['change_id']), fill=color,
                      stroke_width=1, stroke_fill='black')
    if payload.get('boxes', {}).get('source'):
        return source_boxes, 'SOURCE grounding (cyan box / green polygon)'
    return target_boxes, 'TARGET grounding (orange box / green polygon)'


def wrapped(draw, xy, value, width, fill='black', spacing=3):
    draw.multiline_text(xy, '\n'.join(textwrap.wrap(value, width=width)),
                        fill=fill, spacing=spacing)


def pair_tile(case, stage, verdict):
    tile = Image.new('RGB', (1040, 355), 'white')
    draw = ImageDraw.Draw(tile)
    color = '#087f23' if verdict == 'PASS' else '#b00020'
    draw.text((6, 5), f'{stage.upper()} {verdict} | {case["key"]} | {case["row"]["final_task"]}',
              fill=color)
    wrapped(draw, (6, 22), case['row']['final_instruction'], 150)
    reason = case['quality']['reason'] if stage == 'quality' else case['scene']['reason']
    wrapped(draw, (6, 52), reason, 150, fill='#333')
    if stage == 'quality':
        for index, image in enumerate((case['source'], case['target'])):
            image = image.copy().convert('RGB'); image.thumbnail((510, 250))
            tile.paste(image, (index * 520, 102))
        draw.text((6, 87), 'SOURCE', fill='#555'); draw.text((526, 87), 'TARGET', fill='#555')
    else:
        image = case['source'].copy().convert('RGB'); image.thumbnail((620, 250))
        tile.paste(image, (6, 102))
        draw.text((6, 87), 'SOURCE (the only image shown to the fine-grained filter)', fill='#555')
        wrapped(draw, (645, 115), case['note'], 55, fill='#333')
    return tile


def mask_tile(case):
    tile = Image.new('RGB', (1300, 285), 'white')
    draw = ImageDraw.Draw(tile)
    draw.text((6, 5), f'MASK | {case["key"]} | {case["row"]["final_task"]} | '
              f'{case["mask"]["qc_flag"]}', fill='#087f23')
    wrapped(draw, (6, 22), case['row']['final_instruction'], 190)
    labels = ('SOURCE', 'TARGET', case['grounding_label'], 'MASK OVERLAY', 'BINARY MASK')
    images = (case['source'], case['target'], case['grounding'], case['overlay'], case['binary'])
    for index, (label, image) in enumerate(zip(labels, images)):
        draw.text((index * 260 + 5, 57), label, fill='#555')
        image = image.copy().convert('RGB'); image.thumbnail((255, 205))
        tile.paste(image, (index * 260, 76))
    return tile


def save_stack(tiles, path):
    width = max(tile.width for tile in tiles)
    height = sum(tile.height for tile in tiles)
    sheet = Image.new('RGB', (width, height), 'white')
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (0, offset)); offset += tile.height
    sheet.save(path, quality=92)


def html_case(case, stage, verdict):
    if stage == 'quality':
        reason = case['quality']['reason']
        images = (('SOURCE', case['source']), ('TARGET', case['target']))
    elif stage == 'scene':
        reason = case['scene']['reason']
        images = (('SOURCE — only image used by this filter', case['source']),)
    else:
        reason = ''
        images = (
            ('SOURCE', case['source']), ('TARGET', case['target']),
            (case['grounding_label'], case['grounding']),
            ('MASK OVERLAY', case['overlay']), ('BINARY MASK', case['binary']),
        )
    image_html = ''.join(
        f'<figure><img src="{image_uri(image)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        for label, image in images
    )
    reason_html = f'<p><b>Decision:</b> {html.escape(reason)}</p>' if reason else ''
    return (
        f'<article><h3>{html.escape(case["key"])} · {html.escape(case["row"]["final_task"])}'
        f' · {html.escape(verdict)}</h3><p>{html.escape(case["row"]["final_instruction"])}</p>'
        f'{reason_html}<p class="note">Why selected: {html.escape(case["note"])}</p>'
        f'<div class="images">{image_html}</div></article>'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--selection-file', type=Path, required=True)
    parser.add_argument('--filter-run-dir', type=Path,
                        help='Reuse quality/scene outputs from another validated run')
    parser.add_argument('--gallery-file', type=Path,
                        default=Path(__file__).with_name('scaleedit_review_cases.json'))
    parser.add_argument('--output-dir', type=Path, help='Defaults to RUN_DIR/review')
    args = parser.parse_args()

    gallery = json.loads(args.gallery_file.read_text())
    wanted = {case_key(item['shard'], item['row_idx'])
              for name, *_ in GROUPS for item in gallery[name]}
    selected = load_selection(args.selection_file)
    output = args.output_dir or args.run_dir / 'review'
    output.mkdir(parents=True, exist_ok=True)
    filters = args.filter_run_dir or args.run_dir
    errors, counts, cases = [], Counter(), {}

    for shard, indices in tqdm(selected.items(), desc='Validate/review', unit='shard'):
        source_rows = pq.read_table(args.input_dir / shard).to_pylist()
        quality = audit_rows(filters / 'quality/audit', shard)
        scene = audit_rows(filters / 'scene/audit', shard)
        ground = audit_rows(args.run_dir / 'grounding', shard)
        masks = audit_rows(args.run_dir / 'mask', shard)
        expected_scene = {index for index in indices if quality.get(index, {}).get('keep')}
        expected_mask = {index for index in expected_scene if scene.get(index, {}).get('keep')}
        for label, actual, expected in (
            ('quality', set(quality), set(indices)), ('scene', set(scene), expected_scene),
            ('grounding', set(ground), expected_mask), ('mask', set(masks), expected_mask),
        ):
            if actual != expected:
                errors.append(f'{shard}: {label} coverage mismatch {sorted(actual ^ expected)}')

        for row_idx in indices:
            row = source_rows[row_idx]
            quality_row, scene_row = quality.get(row_idx), scene.get(row_idx)
            ground_row, mask_row = ground.get(row_idx), masks.get(row_idx)
            counts['selected'] += 1
            for label, item in (
                ('quality', quality_row), ('scene', scene_row),
                ('grounding', ground_row), ('mask', mask_row),
            ):
                if item:
                    if (item['sample_id'] != row['sample_id'] or
                            item['final_instruction'] != row['final_instruction'].strip()):
                        errors.append(f'{shard}:{row_idx} {label} identity mismatch')
                    result = item.get('verdict', item.get('qc_flag', 'unknown'))
                    counts[f'{label}_{result}'] += 1
                    counts[f'{label}_errors'] += bool(item.get('error'))

            decoded = []
            for image_key in ('source_image', 'edited_image'):
                try:
                    decoded.append(decode(row[image_key]))
                except Exception as exc:
                    counts['input_decode_errors'] += 1
                    placeholder = Image.new('RGB', (640, 480), '#ddd')
                    ImageDraw.Draw(placeholder).text(
                        (20, 30), f'{image_key}: unreadable image\n{type(exc).__name__}', fill='black')
                    decoded.append(placeholder)
                    if (quality_row is None or quality_row['keep'] or
                            quality_row['reason'] != 'invalid_input'):
                        errors.append(f'{shard}:{row_idx} corrupt image was not safely dropped')
            source_image, target_image = decoded
            payload = json.loads(ground_row['ground_json']) if ground_row else {}
            overlay = binary_image = grounding_image = grounding_label = None

            if mask_row:
                binary = np.asarray(Image.open(io.BytesIO(mask_row['mask_png'])))
                shape = (source_image.height, source_image.width)
                if binary.shape != shape or not set(np.unique(binary)) <= {0, 255}:
                    errors.append(f'{shard}:{row_idx} PNG dimensions/values')
                union = np.zeros(shape, dtype=bool)
                for instance in mask_row['instance_masks']:
                    mask = mask_utils.decode(dict(
                        size=instance['rle_size'], counts=instance['rle_counts'].encode()
                    )).astype(bool)
                    if mask.shape != shape or int(mask.sum()) != instance['area']:
                        errors.append(f'{shard}:{row_idx} instance dimensions/area')
                    union |= mask
                if (not np.array_equal(union, binary > 0) or
                        int(union.sum()) != mask_row['mask_sum']):
                    errors.append(f'{shard}:{row_idx} RLE/PNG/area mismatch')
                overlay_array = np.asarray(source_image).copy()
                overlay_array[union] = (
                    overlay_array[union] * 0.55 + np.array([255, 30, 70]) * 0.45
                ).astype(np.uint8)
                overlay, binary_image = Image.fromarray(overlay_array), Image.fromarray(binary)
                grounding_image, grounding_label = annotated_images(source_image, target_image, payload)

            key = case_key(shard, row_idx)
            if key in wanted:
                cases[key] = dict(
                    key=key, row=row, quality=quality_row, scene=scene_row,
                    ground=ground_row, mask=mask_row, source=source_image,
                    target=target_image, grounding=grounding_image,
                    grounding_label=grounding_label, overlay=overlay, binary=binary_image,
                )

    html_groups, gallery_summary = [], {}
    quality_tiles, scene_tiles, mask_tiles = [], [], []
    for group_name, title, stage, expected in GROUPS:
        group_html, group_summary = [], []
        for item in gallery[group_name]:
            key = case_key(item['shard'], item['row_idx'])
            case = cases.get(key)
            if case is None:
                errors.append(f'Gallery case missing from selection: {key}')
                continue
            case['note'] = item['note']
            if stage == 'quality':
                actual = case['quality']['verdict'] if case['quality'] else 'MISSING'
                quality_tiles.append(pair_tile(case, stage, actual))
            elif stage == 'scene':
                actual = case['scene']['verdict'] if case['scene'] else 'MISSING'
                scene_tiles.append(pair_tile(case, stage, actual))
            else:
                actual = case['mask']['qc_flag'] if case['mask'] else 'MISSING'
                if actual != 'MISSING':
                    mask_tiles.append(mask_tile(case))
            if actual != expected:
                errors.append(f'{key}: gallery expected {stage}={expected}, got {actual}')
            group_html.append(html_case(case, stage, actual))
            group_summary.append(dict(key=key, verdict=actual, note=item['note']))
        html_groups.append(f'<section><h2>{html.escape(title)}</h2>{"".join(group_html)}</section>')
        gallery_summary[group_name] = group_summary

    save_stack(quality_tiles, output / 'quality_prefilter.jpg')
    save_stack(scene_tiles, output / 'fine_grained_prefilter.jpg')
    save_stack(mask_tiles, output / 'mask_examples.jpg')

    style = '''<style>
body{font:15px/1.45 sans-serif;margin:24px;max-width:1500px}section{margin:32px 0}
article{border:1px solid #bbb;border-radius:8px;padding:14px;margin:18px 0}h3{margin:0 0 8px}
.images{display:flex;gap:10px;align-items:flex-start;flex-wrap:wrap}.images figure{margin:0;max-width:280px}
.images img{max-width:280px;max-height:260px}.images figcaption{font-size:12px;color:#555}.note{color:#555}
code{overflow-wrap:anywhere}pre{background:#f6f6f6;padding:12px;overflow:auto}
</style>'''
    landing = (
        '<!doctype html><meta charset="utf-8"><title>ScaleEdit representative review</title>'
        + style + '<h1>ScaleEdit representative stage review</h1>'
        + '<p>This compact gallery is intentionally curated. Structural validation still covers all '
          'selected rows and all their double-PASS masks. Quality sees SOURCE + TARGET; the '
          'fine-grained filter sees SOURCE only; mask panels show SOURCE / TARGET / grounding / overlay / binary.</p>'
        + '<pre>' + html.escape(json.dumps(dict(counts), indent=2)) + '</pre>'
        + ''.join(html_groups)
    )
    (output / 'index.html').write_text(landing)
    summary = dict(
        counts=counts, errors=errors, gallery=gallery_summary,
        review=str(output / 'index.html'),
        contact_sheets=[str(output / name) for name in (
            'quality_prefilter.jpg', 'fine_grained_prefilter.jpg', 'mask_examples.jpg')],
    )
    (output / 'validation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
