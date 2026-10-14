"""Unranked SA-1B random-access input reservoir, benchmark exclusion, all masks."""
from __future__ import annotations

import gc
import hashlib
import io
import json
import pickle
import random
from pathlib import Path

from .production_state import atomic_json, atomic_text, file_hash, write_rows
from .source_pool import benchmark_keys

IMAGE_INDEX = 'sa1b_tfd_1k/sa_000000_unified_tot1000.index'
DENSE_INDEX = 'sa1b_dense_label/annotation/all1000.index'


def prepare_reservoir(root, source_root, benchmark, capacity=16000, seed=20261014):
    root, source_root = Path(root), Path(source_root)
    manifest = root/'inputs/reservoir.jsonl'
    if (root/'inputs/ready.json').exists():
        return
    excluded, hashes = benchmark_keys(benchmark)
    # These are the dataset owner's trusted pickle indexes, not user input.
    with (source_root/IMAGE_INDEX).open('rb') as f:
        index = pickle.load(f)
    keys = random.Random(seed).sample(list(index), min(len(index), capacity+len(excluded)))
    keys = [k for k in keys if k not in excluded][:capacity]
    entries = {k: {'source_id': k, 'image': index[k]} for k in keys}
    del index
    gc.collect()
    with (source_root/DENSE_INDEX).open('rb') as f:
        index = pickle.load(f)
    for k in keys:
        entries[k]['annotation'] = index[k]
    del index
    gc.collect()
    rows = [{**entries[k], 'rank': i % 4} for i, k in enumerate(keys)]
    write_rows(manifest, rows)
    for rank in range(4):
        write_rows(root/f'inputs/node{rank}.jsonl', [r for r in rows if r['rank'] == rank])
    atomic_json(root/'inputs/exclusion.json', {'source_ids': sorted(excluded), 'image_hashes': sorted(hashes),
                                             'benchmark_sha256': file_hash(benchmark), 'seed': seed,
                                             'reservoir_size': len(rows), 'selection': 'random sources, no mask quality ranking'})
    atomic_json(root/'inputs/ready.json', {'sources': len(rows), 'reservoir_sha256': file_hash(manifest)})


def read_record(source_root, info, subdir):
    from .production_source_cache import decode_frame, read_frame
    path = Path(source_root)/subdir/Path(info['file']).name
    with path.open('rb') as f:
        return decode_frame(read_frame(f, info['offset'], path))


def materialize(entry, root, source_root, exclusion):
    from PIL import Image

    from .geometry import load_mask
    root = Path(root)
    sid = entry['source_id']
    directory = root/'sources'/sid
    marker = directory/'complete.json'
    if marker.exists():
        return json.loads(marker.read_text())
    if (root/'source_cache/ready.json').exists():
        from .production_source_cache import cached_record
        image = cached_record(root/'source_cache', entry['image'], 'sa1b_tfd_1k')
        dense = cached_record(root/'source_cache', entry['annotation'], 'sa1b_dense_label/annotation')
    else:
        image = read_record(source_root, entry['image'], 'sa1b_tfd_1k')
        dense = read_record(source_root, entry['annotation'], 'sa1b_dense_label/annotation')
    if image['key'].decode() != sid or dense['key'].decode() != sid:
        raise ValueError('SA1B image/annotation/index identity mismatch')
    raw = image['jpg']
    hashed = hashlib.sha256(raw).hexdigest()
    if sid in exclusion['source_ids'] or hashed in exclusion['image_hashes']:
        result = {'source_id': sid, 'status': 'excluded_benchmark'}
        atomic_json(marker, result)
        return result
    # Shared atomic directory claim deduplicates identical image bytes across ranks.
    claim = root/'control/image_hashes'/hashed
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        claim.mkdir()
        atomic_text(claim/'owner', sid)
    except FileExistsError:
        if not (claim/'owner').exists():
            raise RuntimeError(f'incomplete image ownership claim; inspect before resume: {claim}')
        if (claim/'owner').read_text() != sid:
            result = {'source_id': sid, 'status': 'duplicate_image_hash'}
            atomic_json(marker, result)
            return result
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        width, height = im.size
    directory.mkdir(parents=True, exist_ok=True)
    temp = directory/'image.jpg.tmp'
    temp.write_bytes(raw)
    temp.replace(directory/'image.jpg')
    objects, rejected, ids = [], [], set()
    for obj in json.loads(dense.get('objects_json', b'[]')):
        oid = str(obj['object_id'])
        if oid in ids:
            raise ValueError(f'duplicate source mask: {sid}/{oid}')
        ids.add(oid)
        rle = obj.get('segmentation')
        try:
            mask = load_mask({'rle': rle}, (width, height))
            if mask.shape != (height, width) or not mask.any():
                raise ValueError('empty mask or dimensions differ from image')
        except (ValueError, TypeError, KeyError, AttributeError, AssertionError, OverflowError) as exc:
            rejected.append({'object_id': oid, 'reason': 'undecodable_or_empty_mask', 'detail': str(exc)})
            continue
        objects.append({'object_id': oid, 'category': obj.get('category', 'object'),
                        'description': obj.get('gar_caption', ''), 'rle': rle, 'bbox': obj.get('bbox'),
                        'metadata': {'score': obj.get('score')}})
    atomic_json(directory/'dense.json', {k: v.decode('utf-8') for k, v in dense.items()})
    result = {'source_id': sid, 'status': 'ready' if objects else 'no_decodable_masks',
              'invalid_masks': rejected, 'candidate': {
                  'source_id': sid, 'image_path': str(directory/'image.jpg'), 'image_sha256': hashed,
                  'caption': dense.get('caption', b'').decode(),
                  'tags': dense.get('merged_tags', b'').decode().split(','), 'objects': objects,
                  'provenance': {'dataset': 'SA-1B companion dense labels', 'index_entry': entry,
                                 'dense_record': str(directory/'dense.json'),
                                 'benchmark_exclusion_sha256': exclusion['benchmark_sha256']}}}
    atomic_json(marker, result)
    return result
