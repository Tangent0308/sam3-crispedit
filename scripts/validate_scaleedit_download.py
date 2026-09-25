#!/usr/bin/env python3
"""Verify appended IDs, pair identity, category scope, schema and sample images."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq
from tqdm import tqdm
from scaleedit.download import LOCAL_TASKS, SCHEMA, atomic_json, verified_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--image-check-shards', type=int, default=64)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    if not state.get('complete'):
        raise SystemExit('Download is not complete; do not certify a moving dataset')
    root = Path(state['output_dir'])
    new_files = set(root.glob(f'expand-{state["run_id"]}-*.parquet'))
    image_files = set(random.Random(20260924).sample(sorted(new_files), min(len(new_files), args.image_check_shards)))
    old_ids, old_pairs, new_ids, new_pairs = set(), set(), set(), set()
    counts = Counter()
    errors = Counter()
    checked_images = 0
    for path in tqdm(sorted(root.glob('*.parquet')), desc='Verify ScaleEdit download', unit='shard'):
        new = path in new_files
        pf = pq.ParquetFile(path)
        if new and not pf.schema_arrow.equals(SCHEMA):
            errors['schema_mismatch'] += 1
        ids, pairs = (new_ids, new_pairs) if new else (old_ids, old_pairs)
        rows = pf.read(columns=['sample_id', 'source_relative_path', 'original_instruction', 'final_task']).to_pylist()
        for row in rows:
            key = (row['source_relative_path'], row['original_instruction'])
            if row['sample_id'] in ids:
                errors['duplicate_new_id' if new else 'duplicate_old_id'] += 1
            if new and key in pairs:
                errors['duplicate_new_pair'] += 1
            ids.add(row['sample_id'])
            pairs.add(key)
            if new:
                counts[row['final_task']] += 1
                if row['final_task'] not in LOCAL_TASKS:
                    errors['excluded_task'] += 1
        if path in image_files:
            # All rows were decoded before export; independently sample persisted shards.
            row = next(pf.iter_batches(batch_size=1)).to_pylist()[0]
            try:
                _, _, _, sizes = verified_images(row, False)
                if sizes != [(row['source_image_width'], row['source_image_height']),
                             (row['edited_image_width'], row['edited_image_height'])]:
                    errors['dimensions_mismatch'] += 1
                checked_images += 2
            except Exception:
                errors['undecodable_sample_image'] += 1
    if len(old_ids) != state['baseline_rows']:
        errors['baseline_row_count_changed'] += 1
    if len(new_ids) != state['new_rows']:
        errors['new_row_count_mismatch'] += 1
    tolerance = state.get('completion_tolerance_percent', 0)
    if not state['target_new_rows'] * (1 - tolerance / 100) <= len(new_ids) <= state['target_new_rows']:
        errors['outside_target_tolerance'] += 1
    if old_ids & new_ids:
        errors['overlap_with_old_ids'] += len(old_ids & new_ids)
    if old_pairs & new_pairs:
        errors['overlap_with_old_pairs'] += len(old_pairs & new_pairs)
    if Counter(state['new_counts']) != counts:
        errors['state_count_mismatch'] += 1
    report = dict(old_rows=len(old_ids), new_rows=len(new_ids), total_rows=len(old_ids) + len(new_ids),
                  new_shards=len(new_files), new_counts=dict(counts), checked_images=checked_images,
                  requested_new_rows=state['target_new_rows'], shortfall_rows=state['target_new_rows'] - len(new_ids),
                  errors=dict(errors), source_revision=state['source_revision'], filtered_revision=state['filtered_revision'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
