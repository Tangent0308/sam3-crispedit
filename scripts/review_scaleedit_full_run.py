#!/usr/bin/env python3
"""Build a self-contained, stratified gallery from a completed ScaleEdit run."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import io
import json
from pathlib import Path
import random
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from scaleedit.runner import decode


RUN = Path('/mnt/bn/strategy-mllm-train/user/tanyue/experiments/ScaleEdit/labeling_4node_scaleedit_300k_20260926')
SOURCE = Path('/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-source')
QUOTAS = {
    'object_addition': 10,
    'object_removal': 10,
    'object_replacement': 10,
    'color_change': 8,
    'action_editing': 8,
    'material_change': 8,
    'building_surface_text_editing': 4,
    'compositional_editing': 6,
    'object_surface_text_editing': 4,
    'gui_interface_text_editing': 4,
    'symbolic_reasoning': 3,
    'scientific_reasoning': 3,
    'size_change': 3,
    'movie_poster_text_editing': 3,
    'count_change': 2,
    'perceptual_reasoning': 2,
    'social_reasoning': 2,
}
KNOWN_GOOD = [
    ('part-00307.parquet', 435),
    ('expand-20260924-00071.parquet', 84),
    ('expand-20260924-00427.parquet', 215),
    ('expand-20260924-00511.parquet', 144),
    ('expand-20260924-00106.parquet', 32),
    ('part-00178.parquet', 243),
    ('part-00206.parquet', 63),
    ('expand-20260924-00640.parquet', 124),
]
HIGHLIGHTS = [
    ('added_vase', ('expand-20260924-00511.parquet', 144)),
    ('removed_person', ('expand-20260924-00427.parquet', 215)),
    ('selected_caps', ('part-00307.parquet', 435)),
    ('localized_text', ('expand-20260924-00071.parquet', 84)),
]
# These two were manually identified as semantic failures despite automatic OK.
KNOWN_BAD = {
    ('expand-20260924-00519.parquet', 27),
    ('expand-20260924-00526.parquet', 31),
}
SHEETS = [
    ('add_remove', 'Addition and removal', {'object_addition': 4, 'object_removal': 4}),
    ('replace_color', 'Replacement and color', {'object_replacement': 4, 'color_change': 4}),
    ('action_material', 'Action and material', {'action_editing': 4, 'material_change': 4}),
    ('text', 'Localized text', {
        'building_surface_text_editing': 2, 'object_surface_text_editing': 2,
        'gui_interface_text_editing': 2, 'movie_poster_text_editing': 2,
    }),
    ('composition_reasoning', 'Composition and reasoning', {
        'compositional_editing': 4, 'symbolic_reasoning': 2, 'scientific_reasoning': 2,
    }),
    ('other_local', 'Other local edits', {
        'size_change': 2, 'count_change': 2,
        'perceptual_reasoning': 2, 'social_reasoning': 2,
    }),
]


def key(item: dict) -> tuple[str, int]:
    return item['shard'], int(item['row_idx'])


def scan_mask_shard(path: Path) -> list[dict]:
    columns = ['row_idx', 'final_task', 'qc_flag', 'mask_sum', 'area_frac',
               'instance_masks', 'error']
    rows = pq.read_table(path, columns=columns).to_pylist()
    return [
        dict(shard=path.name, row_idx=int(row['row_idx']), task=row['final_task'],
             area_frac=float(row['area_frac']), instances=len(row['instance_masks'] or []),
             known_good=(path.name, int(row['row_idx'])) in KNOWN_GOOD)
        for row in rows
        if row['qc_flag'] == 'OK' and row['mask_sum'] > 0 and not row['error']
        and 0 < row['area_frac'] < 0.60
        and (path.name, int(row['row_idx'])) not in KNOWN_BAD
    ]


def select_cases(mask_root: Path, workers: int, seed: int) -> list[dict]:
    paths = sorted(mask_root.glob('*.parquet'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(tqdm(pool.map(scan_mask_shard, paths), total=len(paths),
                          desc='Scan OK mask metadata', unit='shard'))
    candidates = [row for part in parts for row in part]
    by_task = defaultdict(list)
    by_key = {}
    for item in candidates:
        by_task[item['task']].append(item)
        by_key[key(item)] = item
    rng = random.Random(seed)
    chosen, seen_keys, seen_shards = [], set(), set()

    def add(item: dict, allow_existing_shard: bool = False) -> bool:
        if key(item) in seen_keys or (item['shard'] in seen_shards and not allow_existing_shard):
            return False
        chosen.append(item)
        seen_keys.add(key(item))
        seen_shards.add(item['shard'])
        return True

    for known in KNOWN_GOOD:
        if known not in by_key:
            raise ValueError(f'known reviewed OK case absent from final run: {known}')
        add(by_key[known])

    for task, quota in QUOTAS.items():
        pool = by_task[task][:]
        rng.shuffle(pool)
        target_multi = max(1, quota // 3)
        while sum(item['task'] == task and item['instances'] > 1 for item in chosen) < target_multi:
            if not next((add(item) for item in pool if item['instances'] > 1 and
                         item['shard'] not in seen_shards), False):
                break
        for item in pool:
            if sum(existing['task'] == task for existing in chosen) >= quota:
                break
            add(item)
        for item in pool:
            if sum(existing['task'] == task for existing in chosen) >= quota:
                break
            add(item, allow_existing_shard=True)
        if sum(item['task'] == task for item in chosen) != quota:
            raise ValueError(f'not enough distinct-shard OK cases for {task}')
    return sorted(chosen, key=lambda item: (list(QUOTAS).index(item['task']),
                                            not item['known_good'], item['shard'], item['row_idx']))


def read_case_group(run: Path, source: Path, shard: str, selected: list[dict]) -> list[dict]:
    source_path = source / shard
    mask_path = run / 'mask' / shard
    if not source_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f'missing source or mask shard for {shard}')
    source_table = pq.ParquetFile(source_path).read(columns=[
        'sample_id', 'final_task', 'final_instruction', 'source_image', 'edited_image'])
    mask_rows = {int(row['row_idx']): row for row in pq.ParquetFile(mask_path).read().to_pylist()}
    cases = []
    for item in selected:
        row_idx = int(item['row_idx'])
        source_row = source_table.slice(row_idx, 1).to_pylist()[0]
        mask_row = mask_rows[row_idx]
        if source_row['sample_id'] != mask_row['sample_id'] or mask_row['qc_flag'] != 'OK':
            raise ValueError(f'identity/QC mismatch in {shard}:{row_idx}')
        before = decode(source_row['source_image']).convert('RGB')
        after = decode(source_row['edited_image']).convert('RGB')
        binary = Image.open(io.BytesIO(mask_row['mask_png'])).convert('L')
        if binary.size != before.size or int(np.count_nonzero(np.asarray(binary))) != mask_row['mask_sum']:
            raise ValueError(f'mask shape/area mismatch in {shard}:{row_idx}')
        mask_alpha = binary.point(lambda value: 105 if value else 0)
        red = Image.new('RGB', before.size, (255, 30, 75))
        overlay = Image.composite(red, before, mask_alpha)
        payload = json.loads(mask_row['ground_json'])
        marked = {'source': before.copy(), 'target': after.copy()}
        for side, color in (('source', '#00d7ff'), ('target', '#ffae00')):
            draw = ImageDraw.Draw(marked[side])
            for box in payload.get('boxes', {}).get(side, []):
                coords = box.get('bbox_2d') or []
                if len(coords) != 4:
                    continue
                xy = [coords[i] * (marked[side].width if i % 2 == 0 else marked[side].height) / 1000
                      for i in range(4)]
                draw.rectangle(xy, outline=color, width=max(2, before.width // 300))
                polygon = box.get('polygon_2d') or []
                if len(polygon) >= 3:
                    points = [(x * marked[side].width / 1000, y * marked[side].height / 1000)
                              for x, y in polygon]
                    draw.line(points + [points[0]], fill='#2aff36', width=3)
        sides = [side for side in ('source', 'target') if payload.get('boxes', {}).get(side)]
        grounding_side = 'source' if 'source' in sides else 'target'
        cases.append(dict(**item, instruction=source_row['final_instruction'],
                          sample_id=source_row['sample_id'], source=before, target=after,
                          grounding=marked[grounding_side], grounding_side=grounding_side,
                          overlay=overlay, binary=binary, mask_source=mask_row['mask_source'],
                          instance_refs=[instance['ref'] for instance in mask_row['instance_masks']]))
    return cases


def image_uri(image: Image.Image, binary: bool = False) -> str:
    image = image.copy()
    image.thumbnail((430, 330), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    if binary:
        image.save(output, format='PNG', optimize=True)
        mime = 'png'
    else:
        image.convert('RGB').save(output, format='JPEG', quality=80, optimize=True)
        mime = 'jpeg'
    return f'data:image/{mime};base64,' + base64.b64encode(output.getvalue()).decode('ascii')


def render_html(cases: list[dict], manifest: dict, path: Path) -> None:
    stages = manifest['stages']
    counts = stages['mask']['counts']
    cards = []
    for item in cases:
        labels = [
            ('Source', item['source'], False), ('Target', item['target'], False),
            (f"Grounding on {item['grounding_side']}", item['grounding'], False),
            ('Mask overlay on source canvas', item['overlay'], False),
            ('Binary mask', item['binary'], True),
        ]
        figures = ''.join(
            f'<figure><img loading="lazy" src="{image_uri(image, binary)}" alt="{html.escape(label)}">'
            f'<figcaption>{html.escape(label)}</figcaption></figure>'
            for label, image, binary in labels
        )
        refs = html.escape('; '.join(item['instance_refs']))
        title = f"{item['shard']}:{item['row_idx']}"
        cards.append(
            f'<article class="case" data-task="{html.escape(item["task"])}" '
            f'data-search="{html.escape((title + " " + item["task"] + " " + item["instruction"]).lower())}">'
            f'<h3>{html.escape(title)} <span class="tag">{html.escape(item["task"])}</span></h3>'
            f'<p class="instruction">{html.escape(item["instruction"])}</p>'
            f'<p class="meta">automatic QC: OK · area {item["area_frac"]:.2%} · '
            f'{item["instances"]} instance(s) · {html.escape(item["mask_source"])}</p>'
            f'<div class="images">{figures}</div>'
            f'<details><summary>Instance references and sample ID</summary><p>{refs}</p>'
            f'<code>{html.escape(item["sample_id"])}</code></details></article>')
    options = ''.join(f'<option value="{html.escape(task)}">{html.escape(task)} '
                      f'({sum(item["task"] == task for item in cases)})</option>' for task in QUOTAS)
    title = 'ScaleEdit 300k full run: representative OK masks'
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
body{{font:15px/1.5 system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1b2733}}
header{{background:#162638;color:white;padding:24px max(22px,calc((100vw - 1480px)/2))}}
main{{max-width:1480px;margin:auto;padding:18px}}h1{{margin:0 0 8px}}h2{{margin:30px 0 12px}}
.stats{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.stats div{{background:#fff;color:#142331;padding:10px 16px;border-radius:8px;min-width:130px}}
.stats b{{display:block;font-size:22px}}.controls{{position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:2}}
select,input{{font:inherit;padding:8px;border:1px solid #aaa;border-radius:5px;max-width:100%}}
.case{{background:white;border:1px solid #d6dde4;border-radius:10px;margin:14px 0;padding:16px}}
.case h3{{margin:0 0 7px;font-size:17px}}.tag{{font-size:12px;background:#d9eaf7;padding:3px 7px;border-radius:4px}}
.instruction{{font-size:16px;margin:6px 0}}.meta{{color:#51606e;margin:6px 0 13px}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}}
figure{{margin:0;background:#f2f4f5;padding:5px;text-align:center;border-radius:5px}}
figure img{{max-width:100%;height:240px;object-fit:contain}}figcaption{{font-size:12px;color:#52606c}}
details{{margin-top:10px;color:#51606e}}code{{overflow-wrap:anywhere}}
</style></head><body><header><h1>{title}</h1><p>Full four-node run completed {html.escape(manifest['completed_utc'])}. {len(cases)} diverse examples sampled from {counts['OK']:,} automatic OK masks. Images are embedded in this file.</p>
<div class="stats"><div><b>{stages['quality']['counts']['rows']:,}</b>source pairs</div>
<div><b>{stages['quality']['counts']['PASS']:,}</b>quality PASS</div>
<div><b>{stages['scene']['counts']['PASS']:,}</b>scene PASS</div>
<div><b>{counts['OK']:,}</b>mask OK</div><div><b>{counts['MASK_REVIEW']:,}</b>mask review</div>
<div><b>{counts['GROUND_FAIL']:,}</b>ground fail</div></div></header>
<main><p>These are automatic QC examples, not a random estimate of semantic accuracy. The source and target show the edit; the red overlay and binary mask are on the source-sized output canvas. Target-side additions are mapped to that canvas.</p>
<div class="controls"><label>Task <select id="task"><option value="">All tasks ({len(cases)})</option>{options}</select></label>
<label> Search <input id="search" type="search" placeholder="instruction or shard"></label>
<span id="shown"></span></div>{''.join(cards)}</main>
<script>const cards=[...document.querySelectorAll('.case')];const task=document.querySelector('#task');
const search=document.querySelector('#search');function update(){{let n=0;for(const card of cards){{
const show=(!task.value||card.dataset.task===task.value)&&card.dataset.search.includes(search.value.toLowerCase());
card.hidden=!show;if(show)n++;}}document.querySelector('#shown').textContent=`${{n}} shown`;}}
task.onchange=update;search.oninput=update;update();</script></body></html>'''
    path.write_text(page)


def tile(item: dict) -> Image.Image:
    canvas = Image.new('RGB', (900, 252), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f"{item['task']} | {item['shard']}:{item['row_idx']} | OK | "
              f"{item['instances']} instance(s)", fill='#087f23')
    instruction = ' '.join(textwrap.wrap(item['instruction'], width=110)[:2])
    draw.text((8, 24), instruction[:150], fill='#222')
    for i, (label, image) in enumerate((('SOURCE', item['source']), ('TARGET', item['target']),
                                         ('MASK OVERLAY', item['overlay']))):
        left = i * 300 + 8
        draw.text((left, 48), label, fill='#555')
        thumb = ImageOps.contain(image.convert('RGB'), (285, 183))
        canvas.paste(thumb, (left, 65))
    return canvas


def contact_sheet(items: list[dict], path: Path) -> None:
    sheet = Image.new('RGB', (1800, 1008), '#e8ecef')
    for i, item in enumerate(items):
        sheet.paste(tile(item), ((i % 2) * 900, (i // 2) * 252))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=89, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--source-dir', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--docs-assets-dir', type=Path)
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20260927)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    source = args.source_dir.resolve()
    output = (args.output_dir or run / 'review_full_300k').resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run / 'reports/run_manifest.json').read_text())
    if args.selection_file:
        selected = json.loads(args.selection_file.read_text())
    else:
        selected = select_cases(run / 'mask', args.workers, args.seed)
    expected = sum(QUOTAS.values())
    if len(selected) != expected or len({key(item) for item in selected}) != expected:
        raise ValueError(f'selection must contain {expected} unique cases')
    (output / 'selected_cases.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2) + '\n')
    groups = defaultdict(list)
    for item in selected:
        groups[item['shard']].append(item)
    cases = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(read_case_group, run, source, shard, items): shard
                   for shard, items in groups.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Decode gallery cases', unit='shard'):
            cases.extend(future.result())
    by_key = {key(item): item for item in cases}
    cases = [by_key[key(item)] for item in selected]
    render_html(cases, manifest, output / 'index.html')

    assets = args.docs_assets_dir or output / 'contact_sheets'
    sheet_info = []
    for slug, label, tasks in SHEETS:
        items = []
        for task, count in tasks.items():
            items.extend(item for item in cases if item['task'] == task and len(
                [other for other in items if other['task'] == task]) < count)
        if len(items) != 8:
            raise ValueError(f'{slug} has {len(items)} instead of 8 cases')
        path = assets / f'{slug}.jpg'
        contact_sheet(items, path)
        sheet_info.append(dict(name=slug, title=label, path=str(path),
                               cases=[f"{item['shard']}:{item['row_idx']}" for item in items]))
    highlights = []
    for slug, case_key in HIGHLIGHTS:
        path = assets / f'highlight_{slug}.jpg'
        tile(by_key[case_key]).save(path, quality=92, subsampling=0)
        highlights.append(dict(name=slug, path=str(path),
                               case=f'{case_key[0]}:{case_key[1]}'))
    summary = dict(run=str(run), source=str(source), html=str(output / 'index.html'),
                   selected=len(cases), automatic_qc='OK',
                   by_task=dict(sorted(Counter(item['task'] for item in cases).items())),
                   known_reviewed=sum(item['known_good'] for item in cases),
                   sheets=sheet_info, highlights=highlights)
    (output / 'review_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
