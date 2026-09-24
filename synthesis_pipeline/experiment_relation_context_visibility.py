"""Freeze a saved relation run and ablate KEEP descriptions anchored off-crop.

Diagnostic experiment only: a relation's point outside the crop does not prove
its entire object is invisible. Do not use this heuristic as a production filter.
Training instructions, execution masks and reconstruction remain unchanged.
"""
import argparse
import json
from pathlib import Path

from synthesis_pipeline.prepare_samtok_data import load_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--ids', required=True)
    args = parser.parse_args()
    wanted = {int(value) for value in args.ids.split(',')}
    rows = [row for row in load_jsonl(args.regions/'annotations.jsonl')
            if int(row['image'].split('_')[0]) in wanted]
    if len(rows) != len(wanted):
        raise ValueError('Missing requested cases')
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root/'sources').symlink_to((args.regions/'sources').resolve())
    records = []
    for row in rows:
        request_path = args.baseline/'diagnostics'/Path(row['image']).stem/'generation_request.json'
        request = json.loads(request_path.read_text())
        left, top, right, bottom = request['crop_bbox']
        width, height = request['source_size']
        kept, omitted = [], []
        for relation in row['relation_plan']['relations']:
            if relation['action'] != 'keep':
                continue
            x, y = relation['point']
            (kept if left <= x*width/1000 < right and top <= y*height/1000 < bottom
             else omitted).append(relation['description'])
        original = row['relation_execution_context']
        row['relation_execution_context'] = (
            ('Keep '+ '; '.join(kept)+'. ' if kept else '')
            + row['relation_plan']['reconstruction'])
        records.append(dict(image=row['image'],crop_bbox=request['crop_bbox'],
            omitted_keep_descriptions=omitted,original_context=original,
            candidate_context=row['relation_execution_context'],
            diagnostic_only=True))
    write_jsonl(args.out_root/'annotations.jsonl', rows)
    write_jsonl(args.out_root/'ablation.jsonl', records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
