"""Check stage coverage, native row identity and binary/RLE mask consistency."""
from collections import Counter
import io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as mask_utils


def validate_shard(source, output, expected, stage):
    columns = ['sample_id', 'final_instruction', 'final_task']
    if stage == 'mask':
        columns.append('source_image')
    originals = pq.read_table(source, columns=columns).to_pylist()
    rows = pq.read_table(output).to_pylist()
    if [r['row_idx'] for r in rows] != sorted(expected):
        raise ValueError(f'{stage} row coverage mismatch: {output}')
    counts = Counter()
    for row in rows:
        original = originals[row['row_idx']]
        if row['sample_id'] != str(original['sample_id']) or row['final_instruction'] != original['final_instruction'].strip():
            raise ValueError(f'{stage} identity mismatch: {output}:{row["row_idx"]}')
        counts['rows'] += 1
        counts['errors'] += bool(row.get('error'))
        label = row.get('verdict', row.get('qc_flag'))
        counts[label] += 1
        if stage in ('quality', 'scene'):
            if label not in ('PASS', 'DROP') or row['keep'] != (label == 'PASS'):
                raise ValueError(f'Invalid binary verdict: {output}')
        if stage == 'mask':
            from scaleedit.runner import decode
            source_image = decode(original['source_image'])
            shape = (source_image.height, source_image.width)
            mask = np.asarray(Image.open(io.BytesIO(row['mask_png'])))
            if mask.shape != shape or not set(np.unique(mask)) <= {0, 255}:
                raise ValueError(f'Invalid mask PNG: {output}')
            union = np.zeros(shape, dtype=bool)
            for instance in row['instance_masks']:
                decoded = mask_utils.decode(dict(size=instance['rle_size'], counts=instance['rle_counts'].encode())).astype(bool)
                if decoded.shape != shape or int(decoded.sum()) != instance['area']:
                    raise ValueError(f'Invalid instance mask: {output}')
                union |= decoded
            if not np.array_equal(union, mask > 0) or int(union.sum()) != row['mask_sum']:
                raise ValueError(f'Mask PNG/RLE union mismatch: {output}')
            if (row['mask_height'], row['mask_width']) != shape:
                raise ValueError(f'Incorrect mask dimensions: {output}')
            counts['nonempty_masks'] += bool(union.any())
            counts['instances'] += len(row['instance_masks'])
    return counts
