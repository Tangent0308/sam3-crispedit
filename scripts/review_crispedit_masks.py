#!/usr/bin/env python3
"""Render raw pairs and source-coordinate masks for manual quality review."""

import argparse
import base64
import html
import json
import re
import textwrap
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--mask-dir', type=Path)
    parser.add_argument('--pairs-only', action='store_true', help='Inspect raw edits before seeing predictions')
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--embed-existing', action='store_true',
                        help='Inline local image references in an existing gallery index.html')
    args = parser.parse_args()
    if args.embed_existing:
        index_path = args.output_dir / 'index.html'
        page = index_path.read_text()

        def inline_image(match):
            filename = html.unescape(match.group(1))
            image_path = args.output_dir / filename
            if not image_path.is_file():
                raise FileNotFoundError(f'Gallery image not found: {image_path}')
            image_data = base64.b64encode(image_path.read_bytes()).decode('ascii')
            return f'src="data:image/jpeg;base64,{image_data}"'

        page = re.sub(r'src="([^\"]+\.jpe?g)"', inline_image, page, flags=re.IGNORECASE)
        index_path.write_text(page)
        print(json.dumps({'embedded': True, 'output': str(index_path)}))
        return
    if not args.input_dir or not args.selection_file:
        parser.error('--input-dir and --selection-file are required unless --embed-existing is used')
    import pyarrow.parquet as pq
    from PIL import Image, ImageDraw

    from build_category_previews import _read_rows, _decode_image, _decode_mask, _font, _fit, _overlay

    if not args.pairs_only and not args.mask_dir:
        parser.error('--mask-dir is required unless --pairs-only is used')
    cases = json.loads(args.selection_file.read_text())['cases']
    cases = [c for c in cases if c.get('expected_selection','SELECTED') == 'SELECTED']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cached, boards, rows, links = {}, [], [], []
    width,height,header = (630 if args.pairs_only else 420),320,108
    columns = 2 if args.pairs_only else 4
    for n,case in enumerate(cases):
        name,idx = case['shard'],case['row_idx']
        if name not in cached:
            indices = [c['row_idx'] for c in cases if c['shard']==name]
            cached[name] = (_read_rows(args.input_dir/name, indices), {} if args.pairs_only else
                {r['row_idx']:r for r in pq.read_table(args.mask_dir/name).to_pylist()})
        raw, masks = cached[name]
        record = raw[idx]
        result = {'instruction':record['instruction'],'qc_flag':'RAW PAIR'} if args.pairs_only else masks[idx]
        source,target = _decode_image(record['input_img']),_decode_image(record['output_img'])
        panels = [('Source',source),('Target',target)]
        if not args.pairs_only:
            mask = _decode_mask(result['mask_png'],source.size)
            panels += [('Source mask',_overlay(source,mask)),('Binary mask',Image.fromarray(mask*255))]
        sheet = Image.new('RGB',(width*columns,height+header),'white')
        draw = ImageDraw.Draw(sheet)
        title = f"{n+1}. {name}:{idx} | {result['qc_flag']}"
        draw.text((8,4),title,fill='black',font=_font(22,True))
        for j,line in enumerate(textwrap.wrap(result['instruction'],width=155)[:2]):
            draw.text((8,31+j*23),line,fill='black',font=_font(18))
        for j,(label,panel) in enumerate(panels):
            draw.text((j*width+8,82),label,fill='black',font=_font(19,True))
            sheet.paste(_fit(panel,(width,height)),(j*width,header))
        filename = f'{Path(name).stem.replace(" ","_")}_{idx}.jpg'
        sheet.save(args.output_dir/filename,quality=92)
        rows.append(sheet)
        if len(rows)==4 or n==len(cases)-1:
            board = Image.new('RGB',(width*columns,(height+header)*len(rows)),'white')
            for j,row in enumerate(rows):board.paste(row,(0,j*(height+header)))
            board_name = f'page_{len(boards)+1:03d}.jpg'
            board.save(args.output_dir/board_name,quality=92)
            boards.append(board_name)
            rows = []
        image_data = base64.b64encode((args.output_dir / filename).read_bytes()).decode('ascii')
        links.append(
            f'<section><h3>{html.escape(title)}</h3>'
            f'<img width="100%" src="data:image/jpeg;base64,{image_data}" '
            f'alt="{html.escape(title)}"></section>'
        )
    (args.output_dir/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Mask quality review</title>'
        '<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}'
        'section{margin:28px 0}img{display:block;height:auto;border:1px solid #bbb}</style>'
        '<h1>Raw image pairs and masks</h1>'
        '<p>QC flags are automatic diagnostics, NOT semantic accuracy.</p>'+''.join(links))
    print(json.dumps({'cases':len(cases),'pages':boards,'output':str(args.output_dir)}))


if __name__=='__main__':
    main()
