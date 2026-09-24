#!/usr/bin/env python3
"""Compare fixed mask cases without rerunning inference or altering results."""

import argparse
import base64
import io
import json
import html
import os
import textwrap
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from build_category_previews import _read_rows, _decode_image, _decode_mask, _font, _fit, _overlay, _draw_boxes


def metrics(row):
    if not row["mask_png"]:
        return {"qc_flag":row["qc_flag"], "area_frac":0.0, "components":0}
    mask = (np.asarray(Image.open(io.BytesIO(row["mask_png"]))) > 0).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)
    return {"qc_flag":row["qc_flag"], "area_frac":float(mask.mean()), "components":count-1,
            "largest_components":areas[:5],
            "target_instances":sum(bool(item["mapped_from_target"]) for item in row["instance_masks"]),
            "refs":[item["ref"] for item in row["instance_masks"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--box-preview-dir", type=Path,
                        help="Also compare the grounding boxes on their actual source/target canvas")
    parser.add_argument("--detail-preview-dir", type=Path,
                        help="Also render a shared context crop around the before/after masks")
    parser.add_argument("--detail-cases", nargs='*',
                        help="Optional detail-only filename stems, e.g. remove_01360_184")
    args = parser.parse_args()
    selection = json.loads(args.selection_file.read_text())
    cases = selection.get("cases", selection) if isinstance(selection, dict) else selection
    caches = {}
    rows = []
    raw_cache, previews = {}, []
    if args.preview_dir:
        if not args.input_dir:
            parser.error('--preview-dir requires --input-dir')
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    if args.box_preview_dir:
        if not args.preview_dir:
            parser.error('--box-preview-dir requires --preview-dir')
        args.box_preview_dir.mkdir(parents=True, exist_ok=True)
    if args.detail_preview_dir:
        if not args.preview_dir:
            parser.error('--detail-preview-dir requires --preview-dir')
        args.detail_preview_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        name, index = case["shard"], case["row_idx"]
        record = {"shard":name, "row_idx":index}
        for label, directory in [("baseline",args.baseline), ("candidate",args.candidate)]:
            path = directory/name
            if path not in caches:
                caches[path] = {row["row_idx"]:row for row in pq.read_table(path).to_pylist()}
            if index not in caches[path]:
                raise ValueError(f"missing case: {path}:{index}")
            row = caches[path][index]
            record["instruction"] = row["instruction"]
            record[label] = metrics(row)
        rows.append(record)
        if args.preview_dir:
            if name not in raw_cache:
                indices = [c['row_idx'] for c in cases if c['shard'] == name]
                raw_cache[name] = _read_rows(args.input_dir/name, indices)
            raw = raw_cache[name][index]
            source, target = _decode_image(raw['input_img']), _decode_image(raw['output_img'])
            before, after = [_decode_mask(caches[d/name][index]['mask_png'], source.size)
                             for d in (args.baseline, args.candidate)]
            record['mask_iou'] = float(np.logical_and(before,after).sum()/max(np.logical_or(before,after).sum(),1))
            width, height = 420, 420
            sheet = Image.new('RGB', (6*width, height+110), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 5), f"{name} row={index} | {record['baseline']['qc_flag']} -> {record['candidate']['qc_flag']}",
                      font=_font(23, True), fill='black')
            for line_i,line in enumerate(textwrap.wrap(record['instruction'], width=180)[:2]):
                draw.text((8, 34+line_i*23), line, font=_font(20), fill='black')
            panels = [('Source', source), ('Target', target), ('Before', _overlay(source,before)),
                      ('After', _overlay(source,after)), ('Before binary', Image.fromarray(before*255)),
                      ('After binary', Image.fromarray(after*255))]
            for col,(label,panel) in enumerate(panels):
                draw.text((col*width+8, 82), label, font=_font(21, True), fill='black')
                sheet.paste(_fit(panel, (width,height)), (col*width,110))
            filename = f'{Path(name).stem.replace(" ", "_")}_{index}.jpg'
            sheet.save(args.preview_dir/filename, quality=93)
            if args.detail_preview_dir and (not args.detail_cases or Path(filename).stem in args.detail_cases):
                ys,xs = np.where(np.logical_or(before, after))
                if len(xs):
                    margin = max(24, round(max(xs.max()-xs.min(), ys.max()-ys.min())*.15))
                    crop = [max(0,int(xs.min())-margin), max(0,int(ys.min())-margin),
                            min(source.width,int(xs.max())+margin+1), min(source.height,int(ys.max())+margin+1)]
                    for col,(_,panel) in enumerate(panels):
                        panel_crop = [round(crop[0]*panel.width/source.width),round(crop[1]*panel.height/source.height),
                                      round(crop[2]*panel.width/source.width),round(crop[3]*panel.height/source.height)]
                        sheet.paste(_fit(panel.crop(panel_crop),(width,height)),(col*width,110))
                sheet.save(args.detail_preview_dir/filename,quality=93)
            previews.append((filename, f'{name}:{index}', record['mask_iou']))
            if args.box_preview_dir:
                side = 'target' if raw['type'] == 'add' else 'source'
                canvas = target if side == 'target' else source
                box_sets = [json.loads(caches[d/name][index]['ground_json']).get('boxes', {}).get(side, [])
                            for d in (args.baseline, args.candidate)]
                record['grounding_before'], record['grounding_after'] = box_sets
                box_sheet = Image.new('RGB', (4*width, height+250), 'white')
                box_draw = ImageDraw.Draw(box_sheet)
                box_draw.text((8, 8), f'{name}:{index} | Grounding canvas: {side}', font=_font(23, True), fill='black')
                for col,(label,panel) in enumerate([('Source',source),('Target',target),
                    ('Before boxes',_draw_boxes(canvas,box_sets[0],'#00cc00')),
                    ('After boxes',_draw_boxes(canvas,box_sets[1],'#00cc00'))]):
                    box_draw.text((col*width+8, 40), label, font=_font(21,True), fill='black')
                    box_sheet.paste(_fit(panel,(width,height)), (col*width,70))
                for row,(label,boxes) in enumerate(zip(['Before','After'],box_sets)):
                    caption = label+': '+ '; '.join(f"{i}: {b['ref']}" for i,b in enumerate(boxes))
                    for line_i,line in enumerate(textwrap.wrap(caption,width=150)[:3]):
                        box_draw.text((8,height+80+row*75+line_i*23),line,font=_font(18),fill='black')
                box_sheet.save(args.box_preview_dir/filename,quality=93)
    output = {"baseline":str(args.baseline), "candidate":str(args.candidate), "cases":rows,
              "flags":{label:dict(Counter(row[label]["qc_flag"] for row in rows)) for label in ["baseline","candidate"]},
              "note":"Connected-component count is diagnostic, not pixel-level accuracy; inspect both misses and excess."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    if args.preview_dir:
        links = ''
        for filename,label,iou in previews:
            links += f'<h3>{html.escape(label)} | mask IoU {iou:.3f}</h3>'
            for title,directory in [('Detail',args.detail_preview_dir),('Grounding boxes',args.box_preview_dir)]:
                if directory and (directory/filename).exists():
                    link = html.escape(os.path.relpath(directory/filename,args.preview_dir),quote=True)
                    links += f'<a href="{link}">{title}</a> &nbsp; '
            data = base64.b64encode((args.preview_dir / filename).read_bytes()).decode('ascii')
            links += f'<img width="100%" src="data:image/jpeg;base64,{data}" alt="{html.escape(label)}">'
        (args.preview_dir/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Mask pipeline comparison</title>'
            '<h1>Source / Target / Before / After / Before binary / After binary</h1>'
            '<p>Mask IoU is before-vs-after overlap, NOT ground-truth accuracy.</p>'+links)
    print(json.dumps(output["flags"],ensure_ascii=False))


if __name__ == "__main__":
    main()
