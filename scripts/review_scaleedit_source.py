#!/usr/bin/env python3
"""Self-contained source/target download audit; this does not run a filter."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
from scaleedit.download import LOCAL_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pattern', default='*.parquet')
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    chosen = {}
    for path in sorted(args.input_dir.glob(args.pattern)):
        pf = pq.ParquetFile(path)
        tasks = pf.read(columns=['final_task'])['final_task'].to_pylist()
        wanted = {i for i, task in enumerate(tasks) if task in LOCAL_TASKS and task not in chosen}
        if not wanted:
            continue
        offset = 0
        for batch in pf.iter_batches(batch_size=16):
            for i, row in enumerate(batch.to_pylist()):
                if offset + i not in wanted or row['final_task'] in chosen:
                    continue
                chosen[row['final_task']] = (path.name, offset + i, row)
                if len(chosen) >= args.limit:
                    break
            offset += batch.num_rows
            if len(chosen) >= args.limit:
                break
        if len(chosen) >= args.limit:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = ['<!doctype html><meta charset="utf-8"><title>ScaleEdit source pairs</title>',
              '<style>body{font:16px sans-serif;margin:24px}section{margin-bottom:32px}img{max-width:48%;height:auto}code{overflow-wrap:anywhere}</style>',
              '<h1>Downloaded filtered ScaleEdit pairs</h1><p>Source (left) / edited (right). Not two-stage prefilter results.</p>']
    manifest = []
    thumbnails = []
    for task, (shard, row_idx, row) in chosen.items():
        chunks.append(f'<section><h2>{html.escape(task)}</h2><code>{html.escape(shard)}:{row_idx}</code><p>{html.escape(row["final_instruction"])}</p>')
        panels = []
        for key in ('source_image', 'edited_image'):
            im = Image.open(io.BytesIO(row[key])).convert('RGB')
            im.thumbnail((640, 480))
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=88)
            chunks.append('<img src="data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode() + '">')
            im.thumbnail((320, 240))
            panels.append(im.copy())
        chunks.append('</section>')
        tile = Image.new('RGB', (660, 285), 'white')
        draw = ImageDraw.Draw(tile)
        draw.text((5, 5), f'{task} | {shard}:{row_idx}', fill='black')
        for i, im in enumerate(panels):
            tile.paste(im, (i * 330, 40))
        thumbnails.append(tile)
        manifest.append({'shard':shard, 'row_idx':row_idx, 'sample_id':row['sample_id'],
                         'final_task':task, 'instruction':row['final_instruction']})
    (args.output_dir / 'index.html').write_text('\n'.join(chunks))
    (args.output_dir / 'selection.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if thumbnails:
        sheet = Image.new('RGB', (1320, 285 * ((len(thumbnails) + 1) // 2)), 'white')
        for i, tile in enumerate(thumbnails):
            sheet.paste(tile, ((i % 2) * 660, (i // 2) * 285))
        sheet.save(args.output_dir / 'pairs.jpg', quality=92)
    print('Source-pair audit:', len(chosen), args.output_dir / 'index.html')


if __name__ == '__main__':
    main()
