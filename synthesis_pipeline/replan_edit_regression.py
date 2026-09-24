"""Replay planning on fixed masks without modifying original annotations."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.generate_samtok_plan import (
    instruction_messages, mask_geometry_hint, normalize_generated,
    parse_json_object, validation_feedback, write_jsonl, PLANNING_VISUAL_INPUT_VERSION,
)
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
from synthesis_pipeline.reference_binding import bind_reference
import utils.vlm_utils as vlm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--ids', required=True)
    p.add_argument('--model-id', default=None)
    p.add_argument('--vlm',choices=['qwen8b-vllm','qwen38-vllm'],default='qwen8b-vllm')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--reference-jsonl', type=Path, help='Original candidate_source_rows.jsonl for referring context')
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root / 'sources').symlink_to((args.data_root / 'sources').resolve())
    (args.out_root / 'inputs').mkdir()
    ids = {int(x) for x in args.ids.split(',')}
    rows = [json.loads(x) for x in (args.data_root / 'annotations.jsonl').read_text().splitlines()]
    rows = [r for r in rows if int(r['image'].split('_')[0]) in ids]
    if len(rows) != len(ids):
        raise ValueError('Some requested IDs are missing')
    jobs = []
    reference_rows = {}
    if args.reference_jsonl:
        reference_rows = {int(r['parquet_row_index']): r for r in
            (json.loads(x) for x in args.reference_jsonl.read_text().splitlines())}
    for row in rows:
        source = Image.open(args.data_root / 'sources' / row['source_image']).convert('RGB')
        mask = mask_array(source.size, row['mask'])
        crop = instruction_target_crop(source, mask)
        crop.save(args.out_root / 'inputs' / row['image'])
        reference = reference_rows.get(int(row['parquet_row_index']), {})
        row={**row,**{k:reference[k] for k in ('problem','answer') if k in reference}}
        if 'mask_index' in row and 'num_masks' in row:
            row['reference_binding']=bind_reference(row.get('answer',''),row['mask_index'],row['num_masks'])
        jobs.append(dict(row={**row, 'mask_geometry_hint': mask_geometry_hint(mask),
                             **{k: reference[k] for k in ('problem', 'answer') if k in reference}},
                         source=source, crop=crop, previous=None, feedback=None))
    started = time.perf_counter()
    vlm.configure_backend(args.vlm, model_id=args.model_id, device='cuda:0', dtype='bf16')
    backend = vlm.get_backend()
    load_seconds = time.perf_counter() - started
    responses, accepted, calls, seconds = [], [], 0, 0.
    try:
        pending = jobs
        for attempt in range(3):
            again = []
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset+args.batch_size]
                messages = [instruction_messages(j['source'], j['crop'], j['row'],
                            j['row']['task_type'], j['previous'], j['feedback']) for j in batch]
                t = time.perf_counter()
                outputs = backend.chat_batch(messages, max_new_tokens=512)
                seconds += time.perf_counter() - t
                calls += len(batch)
                for job, raw, message in zip(batch, outputs, messages):
                    row = job['row']
                    parsed = parse_json_object(raw)
                    value = normalize_generated(parsed, row['task_type'])
                    responses.append(dict(image=row['image'], attempt=attempt+1,
                        raw_response=raw, parsed=value, prompt=message[0]['content'][-1]['text']))
                    if value is None:
                        job.update(previous=raw, feedback=validation_feedback(parsed, row['task_type']))
                        again.append(job)
                    elif value['mask_compatibility'] == 'compatible':
                        accepted.append({**row, **value, 'refer_object': [value['refer_object']],
                            'planning_visual_input': PLANNING_VISUAL_INPUT_VERSION,
                            'planning_revision': {'original_instruction': row['editing_instruction'],
                                                  'verification': 'model_planned_not_manually_verified'}})
                    print(json.dumps(dict(image=row['image'], attempt=attempt+1, result=value)), flush=True)
                write_jsonl(args.out_root / 'responses.jsonl', responses)
            pending = again
            if not pending:
                break
    finally:
        vlm.shutdown_backend()
    accepted.sort(key=lambda r: r['image'])
    write_jsonl(args.out_root / 'annotations.jsonl', accepted)
    # Retain every sibling mask, including ones not selected for this experiment.
    all_rows = [json.loads(x) for x in (args.data_root / 'annotations.jsonl').read_text().splitlines()]
    sources = {r['source_image'] for r in rows}
    write_jsonl(args.out_root / 'input_annotations.jsonl', [r for r in all_rows if r['source_image'] in sources])
    summary = dict(input_cases=len(rows), accepted=len(accepted), calls=calls,
                   load_seconds=load_seconds, inference_seconds=seconds,
                   wall_seconds=time.perf_counter()-started)
    (args.out_root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
