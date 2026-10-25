"""Materialize whole SA-1B shard pairs into the standard source layout (production feeder).

On the HDFS FUSE mount, random record reads ran at ~15 records/s (64 threads), while contiguous reads
of whole shards reached ~240 MB/s with 32 parallel streams (2026-10-25 measurement). Annotation shard
i and image shard i hold exactly the same keys, so a shard pair is read as contiguous byte ranges in
parallel and every record is converted exactly as ``production_sources.materialize`` does
(``materialize_records``: image-hash exclusion, image-hash claims, mask validation, and
``sources/<sid>/{image.jpg, dense.json, complete.json}``).

Per shard the feeder also writes ``categories/<shard>.json``: the categories (with one
description) of objects that pass the selection prefilters (score, area). Only these need an
affordance label, so labelling stays proportional to what selection can actually use. The shard
marker ``shards/<shard>.json`` is written last.

Usage: python -m sa1b_pipeline.shard_feeder --root POOL_ROOT --entries-dir DIR --exclusion EXCLUSION.json
           --shards 17,203 [--claims-root DIR] [--streams 32] [--workers 32] [--max-records N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from .production_state import atomic_json

SG2 = Path('/mnt/hdfs/sg_byte_ttlive_strategy_llm/user/haobo.yuan/datasets')
VERSION = 'shard-feeder-1'
CHUNK = 64 << 20


def image_file(shard: int) -> Path:
    return SG2 / 'sa1b_tfd_1k' / f'sa_{shard:06d}.tfrecord'


def dense_file(shard: int) -> Path:
    return SG2 / 'sa1b_dense_label/annotation' / f'{shard:05d}_of_00999.tfrecord'


def shard_tables(image_index: dict, dense_index: dict, shards) -> dict[int, list]:
    """{shard: [[key, image_offset, image_length, dense_offset, dense_length], ...]} from the SG2 pickles."""
    wanted = {int(s) for s in shards}
    out = {s: [] for s in wanted}
    for key, info in dense_index.items():
        shard = int(Path(info['file']).name.split('_')[0])
        if shard in wanted:
            image = image_index[key]
            if int(Path(image['file']).name.split('_')[1].split('.')[0]) != shard:
                raise ValueError(f'image and annotation shards differ for {key}')
            out[shard].append([key, image['offset'], image['length'], info['offset'], info['length']])
    for rows in out.values():
        rows.sort(key=lambda r: r[1])
    return out


def runs(spans: list, n: int) -> list:
    """Split (offset, length, key) spans, sorted by offset, into at most n contiguous runs."""
    spans = sorted(spans)
    if not spans:
        return []
    size = max(1, -(-len(spans) // max(1, n)))
    return [spans[i:i + size] for i in range(0, len(spans), size)]


def read_run(path: Path, run: list, emit) -> None:
    """Read one run of frames with one handle, sequentially, and emit (key, frame bytes)."""
    from .production_source_cache import read_frame
    start = run[0][0]
    end = run[-1][0] + run[-1][1] + 16
    buffer, base = b'', start
    with open(path, 'rb', buffering=0) as f:
        f.seek(start)
        position = start
        pending = list(run)
        while pending:
            need = pending[0][0] + pending[0][1] + 16
            while base + len(buffer) < need:
                block = f.read(min(CHUNK, end - position))
                if not block:
                    raise OSError(f'truncated shard {path} at {position}')
                position += len(block)
                buffer += block
            offset, length, key = pending.pop(0)
            frame = buffer[offset - base: offset - base + length + 16]
            emit(key, read_frame(io.BytesIO(frame), 0, f'{path}@{offset}'))
            # Drop consumed bytes so a run holds at most one chunk plus one frame in memory.
            cut = offset + length + 16 - base
            buffer, base = buffer[cut:], base + cut


def read_frames(path: Path, spans: list, streams: int, emit) -> None:
    """Read all spans of one shard file with up to ``streams`` parallel contiguous runs."""
    errors = []

    def work(run):
        try:
            read_run(path, run, emit)
        except BaseException as exc:  # surfaced after join
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(r,), daemon=True) for r in runs(spans, streams)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def prefilter_categories(candidate: dict, *, min_score=.6, min_area=.0015, max_area=.3) -> dict:
    """{category: description} of objects that can pass the selection score/area prefilters."""
    from pycocotools import mask as coco
    out = {}
    for obj in candidate['objects']:
        rle = obj['rle']
        height, width = rle['size']
        area = float(coco.area(rle if isinstance(rle['counts'], bytes) else {**rle, 'counts': rle['counts'].encode()}))
        score = float((obj.get('metadata') or {}).get('score') or 0)
        if score >= min_score and min_area <= area / float(height * width) <= max_area:
            out.setdefault(str(obj.get('category')), obj.get('description', ''))
    return out


def _materialize(task):
    from .production_source_cache import decode_frame
    from .production_sources import materialize_records
    entry, image_frame, dense_frame, root, exclusion, claims_root = task
    result = materialize_records(entry, decode_frame(image_frame), decode_frame(dense_frame), root, exclusion,
                                 claims_root)
    status = result.get('status')
    return status, (prefilter_categories(result['candidate']) if status == 'ready' else {})


def materialize_shard(shard: int, rows: list, root, exclusion: dict, *, streams: int = 32, workers: int = 32,
                      max_records: int | None = None, claims_root=None) -> dict:
    """Materialize one shard pair; resumable (finished sources are kept, the marker is written last)."""
    root = Path(root)
    marker = root / 'shards' / f'{shard:05d}.json'
    if marker.exists():
        info = json.loads(marker.read_text())
        # A partial (--max-records) marker does not cover a larger request; finished sources are kept.
        if info.get('max_records') is None or (max_records is not None and info['max_records'] >= max_records):
            return info
    rows = rows[:max_records] if max_records else rows
    started = time.time()
    entries = {r[0]: {'source_id': r[0],
                      'image': {'file': str(image_file(shard)), 'offset': r[1], 'length': r[2]},
                      'annotation': {'file': str(dense_file(shard)), 'offset': r[3], 'length': r[4]}} for r in rows}
    # One listing instead of a stat per record (a missing path costs ~14 ms on the shared mount).
    present = set(os.listdir(root / 'sources')) if (root / 'sources').exists() else set()
    todo = {k for k in entries if k not in present or not (root / 'sources' / k / 'complete.json').exists()}
    dense = {}
    lock = threading.Lock()

    def keep_dense(key, frame):
        with lock:
            dense[key] = frame
    read_frames(dense_file(shard), [(r[3], r[4], r[0]) for r in rows if r[0] in todo], streams, keep_dense)
    read_seconds = time.time() - started
    statuses, categories = {}, {}
    tasks = queue.Queue(maxsize=4 * workers)
    done = object()

    def produce():
        try:
            read_frames(image_file(shard), [(r[1], r[2], r[0]) for r in rows if r[0] in todo], streams,
                        lambda key, frame: tasks.put((entries[key], frame, dense.pop(key), str(root), exclusion,
                                                      claims_root and str(claims_root))))
        finally:
            tasks.put(done)
    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    def collect(future):
        status, cats = future.result()
        statuses[status] = statuses.get(status, 0) + 1
        for c, d in cats.items():
            categories.setdefault(c, d)
    with ProcessPoolExecutor(workers) as pool:
        # The executor queues without limit; bound in-flight records (each holds ~1.5 MB of frames).
        pending = set()
        while True:
            task = tasks.get()
            if task is done:
                break
            pending.add(pool.submit(_materialize, task))
            if len(pending) >= 4 * workers:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    collect(future)
        for future in pending:
            collect(future)
    producer.join()
    # Sources finished by an earlier attempt still contribute their categories.
    for key in set(entries) - todo:
        rec = json.loads((root / 'sources' / key / 'complete.json').read_text())
        statuses[rec.get('status')] = statuses.get(rec.get('status'), 0) + 1
        if rec.get('status') == 'ready':
            for c, d in prefilter_categories(rec['candidate']).items():
                categories.setdefault(c, d)
    atomic_json(root / 'categories' / f'{shard:05d}.json', categories)
    info = {'version': VERSION, 'shard': shard, 'records': len(rows), 'statuses': statuses,
            'categories': len(categories), 'seconds': round(time.time() - started, 1),
            'annotation_read_seconds': round(read_seconds, 1), 'max_records': max_records}
    atomic_json(marker, info)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--entries-dir', type=Path, required=True, help='<shard>.json tables from shard_tables')
    p.add_argument('--exclusion', type=Path, required=True)
    p.add_argument('--claims-root', type=Path, help='holds control/image_hashes (default: --root)')
    p.add_argument('--shards', required=True)
    p.add_argument('--streams', type=int, default=32)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--max-records', type=int)
    args = p.parse_args(argv)
    exclusion = json.loads(args.exclusion.read_text())
    exclusion = {'source_ids': set(exclusion['source_ids']), 'image_hashes': set(exclusion['image_hashes']),
                 'benchmark_sha256': exclusion['benchmark_sha256']}
    for shard in [int(s) for s in args.shards.split(',') if s]:
        rows = json.loads((args.entries_dir / f'{shard:05d}.json').read_text())
        info = materialize_shard(shard, rows, args.root, exclusion, streams=args.streams, workers=args.workers,
                                 max_records=args.max_records, claims_root=args.claims_root)
        print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
