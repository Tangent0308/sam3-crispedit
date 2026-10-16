#!/usr/bin/env python3
"""Compare frozen multi-target planning runs without generating target images."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from sa1b_pipeline.group_planning import group_views, normalize_multi_grounding
from sa1b_pipeline.instruction_protocol import make_spec, render
from scripts.run_group_pipeline import _group_id


def record(root, gid, objects):
    for folder in ('planned', 'rejected'):
        path = root / folder / f'{gid}.json'
        if path.exists():
            row = json.loads(path.read_text())
            raw = row.get('planning', {}).get('raw', row.get('raw', {}))
            try:
                raw = normalize_multi_grounding(raw, objects)
            except (KeyError, TypeError, ValueError):
                pass  # Malformed responses remain visible, never presented as accepted.
            return row, raw
    return {}, {}


def sentence(target):
    if not target:
        return 'No target returned'
    if target.get('edit_feasible') is False:
        return 'INFEASIBLE: ' + target.get('reason', '')
    try:
        spec = make_spec(target['operation'], target['ref_phrase'], target['scope'],
                         target.get('payload', ''), attributes=target.get('attributes'),
                         placement=target.get('placement'), target_part=target.get('target_part', ''),
                         remove_mode=target.get('remove_mode', 'whole'), qualifier=target.get('qualifier', ''))
        return render(spec)
    except (KeyError, TypeError, ValueError):
        return 'Invalid/incomplete edit: ' + json.dumps(target, ensure_ascii=False)[:250]


def target_index(raw):
    rows = raw.get('targets', [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get('object_id')): row for row in rows if isinstance(row, dict)}


def wrapped(draw, text, xy, width, font, fill='#263b51', spacing=8):
    x, y = xy
    # Measure actual glyphs so Chinese and English both fit the page.
    for paragraph in text.split('\n'):
        line = ''
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_'-]*|[^\S\n]+|.", paragraph):
            if line and draw.textlength(line + token, font=font) > width:
                draw.text((x, y), line, font=font, fill=fill)
                y += font.size + spacing
                line = ''
            line += token if line else token.lstrip()
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + spacing
    return y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--font', default='/tmp/NotoSansCJKsc-Regular.otf')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.root/'inputs/selected.jsonl').read_text().splitlines() if line.strip()]
    review_file = args.root/'visual_review.json'
    reviews = json.loads(review_file.read_text()) if review_file.exists() else {}
    per_mask = (args.root/'mask_responses').exists()
    output = args.root/'inspection'; output.mkdir(exist_ok=True)
    font = lambda size: ImageFont.truetype(args.font, size)
    pages = []
    for index, row in enumerate(rows, 1):
        gid = _group_id('multi', row)
        old, raw_old = record(args.baseline, gid, row['objects'])
        new, raw_new = record(args.root, gid, row['objects'])
        old_targets, new_targets = target_index(raw_old), target_index(raw_new)
        views = new.get('planning', {}).get('views') or new.get('planning_views')
        if not views:
            views, _, _ = group_views(row['candidate'], row['objects'], output/'views', selected_only=True,
                                     context_cutout=per_mask, locate_in_scene=True)
        canvas = Image.new('RGB', (2400, 3200), '#f3f6fa'); draw = ImageDraw.Draw(canvas)
        draw.text((60, 35), f'{index:02d}/{len(rows):02d}   {row["candidate"]["source_id"]}', font=font(46), fill='#163452')
        draw.text((60, 110), '规划对照｜左：全图定位　右：局部证据（抠图灰色为排除区域）', font=font(33), fill='#263b51')
        subtitle = ('实际每个 mask 单独输入，本页合并展示；未生成目标图。' if per_mask else
                    '本轮未生成目标图；自动规划通过不等于视觉合格。')
        draw.text((60, 160), subtitle, font=font(29), fill='#a04327')
        for image_path, box, origin in ((views[0], (860, 1150), (60, 235)), (views[1], (1320, 1150), (990, 235))):
            image = ImageOps.contain(Image.open(image_path).convert('RGB'), box)
            canvas.paste(image, origin)
        y = 1430
        for i, obj in enumerate(row['objects'], 1):
            oid = str(obj['object_id']); before, after = old_targets.get(oid, {}), new_targets.get(oid, {})
            draw.rounded_rectangle((45, y, 2355, y+225), radius=14, fill='white')
            draw.text((65, y+8), f'U{i}  {row["operations"][i-1]}', font=font(29), fill='#163452')
            scope = after.get('scope', next((r.get('entity', '') for r in raw_new.get('regions', []) if str(r.get('object_id')) == oid), ''))
            wrapped(draw, '旧：' + sentence(before), (65, y+52), 1090, font(26))
            wrapped(draw, '新：' + sentence(after), (1210, y+52), 1080, font(26))
            wrapped(draw, '识别：' + scope, (1210, y+165), 1080, font(24))
            y += 240
        status = '结构检查通过' if new.get('multi_id') else new.get('reason', '未完成')
        y = wrapped(draw, '自动结果：' + status, (60, y+12), 2270, font(27), '#a04327')
        note = reviews.get(gid, {}).get('note_zh', '尚无视觉复核记录。')
        wrapped(draw, '视觉复核：' + note, (60, y+20), 2270, font(29))
        path = output/f'planning_{index:02d}.jpg'; canvas.save(path, quality=92)
        pages.append(path)
    images = [Image.open(path).convert('RGB') for path in pages]
    pdf = output/'multi_grounding_comparison.pdf'
    images[0].save(pdf, save_all=True, append_images=images[1:], resolution=160)
    print(pdf)


if __name__ == '__main__':
    main()
