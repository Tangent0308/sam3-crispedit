"""Copy the frozen reservoir's records once; generation workers need no HDFS mount."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sa1b_pipeline.production_source_cache import (
    decode_frame,
    read_frame,
    record_key,
    validate_cache,
)
from sa1b_pipeline.production_state import atomic_json, digest, file_hash, read_rows


def stage_pack(key, items, source_root, cache):
    pack_id = digest(key)
    relative = f'packs/{pack_id}.tfrecord'
    pack, marker = cache/relative, cache/'parts'/f'{pack_id}.json'
    contract = digest(items)
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['contract'] != contract or file_hash(pack) != saved['pack_sha256']:
            raise ValueError('existing source pack changed')
        return saved
    pack.parent.mkdir(parents=True, exist_ok=True)
    temp = pack.with_suffix('.tmp')
    records = {}
    subdir, filename = key
    with (source_root/subdir/filename).open('rb', buffering=0) as source, temp.open('wb') as target:
        for sid, info in sorted(items, key=lambda pair: pair[1]['offset']):
            frame = read_frame(source, info['offset'], filename, info['length'])
            if decode_frame(frame)['key'].decode() != sid:
                raise ValueError(f'source/index identity mismatch: {sid}')
            records[record_key(info, subdir)] = {'pack': relative, 'offset': target.tell(),
                'length': len(frame), 'sha256': hashlib.sha256(frame).hexdigest(), 'source_id': sid}
            target.write(frame)
        target.flush()
        os.fsync(target.fileno())
    temp.replace(pack)
    result = {'contract': contract, 'records': records, 'pack': relative,
              'bytes': pack.stat().st_size, 'pack_sha256': file_hash(pack)}
    atomic_json(marker, result)
    return result


def stage(root, source_root, workers=24):
    cache = root/'source_cache'
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/'stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = root/'inputs/reservoir.jsonl'
        if (cache/'ready.json').exists():
            ready = validate_cache(cache, manifest)
            print(json.dumps({'status': 'already_complete', 'records': ready['records']}), flush=True)
            return
        rows = read_rows(manifest)
        groups = defaultdict(list)
        for row in rows:
            for field, subdir in [('image', 'sa1b_tfd_1k'), ('annotation', 'sa1b_dense_label/annotation')]:
                info = row[field]
                groups[subdir, Path(info['file']).name].append((row['source_id'], info))
        combined, packs = {}, []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(stage_pack, key, items, source_root, cache) for key, items in groups.items()]
            for future in as_completed(futures):
                saved = future.result()
                if set(combined).intersection(saved['records']):
                    raise ValueError('duplicate record in source cache')
                combined.update(saved['records'])
                packs.append({k: v for k, v in saved.items() if k not in ('records', 'contract')})
                if len(packs) % 20 == 0 or len(packs) == len(groups):
                    report = {'status': 'staging', 'packs_done': len(packs), 'packs_total': len(groups),
                              'records_done': len(combined), 'records_total': len(rows)*2,
                              'elapsed_seconds': time.monotonic()-started, 'updated_unix': time.time()}
                    atomic_json(cache/'progress.json', report)
                    print(json.dumps(report), flush=True)
        if len(combined) != len(rows)*2:
            raise ValueError('source cache does not cover all frozen image/annotation records')
        atomic_json(cache/'index.json', combined)
        ready = {'status': 'complete', 'sources': len(rows), 'records': len(combined),
                 'reservoir_sha256': file_hash(manifest), 'index_sha256': file_hash(cache/'index.json'),
                 'bytes': sum(p['bytes'] for p in packs), 'packs': packs,
                 'elapsed_seconds': time.monotonic()-started}
        atomic_json(cache/'ready.json', ready)
        atomic_json(cache/'progress.json', {k: v for k, v in ready.items() if k != 'packs'})
        print(json.dumps({k: v for k, v in ready.items() if k != 'packs'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    stage(args.root.resolve(), args.source_root, args.workers)


if __name__ == '__main__':
    main()
