"""CPU-only recovery/finalization of completed production node exports.

Does not load models, alter frozen run configuration, or regenerate samples.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time

from sa1b_pipeline.operation_balance import automatic_pass
from sa1b_pipeline.production_state import (
    MEDIA_KEYS, MEDIA_SUFFIXES, OPERATIONS, atomic_json, file_hash, mask_key,
    read_rows, write_rows,
)


def emit(**event):
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}), flush=True)


@contextmanager
def directory_lock(path):
    """Atomic mkdir works on the shared mount where flock returns ENOSYS.

    Never steal a lock automatically: a different host may still be publishing.
    """
    try:
        path.mkdir()
    except FileExistsError:
        raise RuntimeError(f'Finalizer lock already exists: {path}; verify its owner has stopped before recovery') from None
    try:
        atomic_json(path/'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'started_unix': time.time()})
        yield
    finally:
        (path/'owner.json').unlink(missing_ok=True)
        path.rmdir()


def media_references(value, key=''):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from media_references(v, k)
    elif isinstance(value, list):
        for v in value:
            yield from media_references(v, key)
    elif isinstance(value, str) and value and (
        key in MEDIA_KEYS or (Path(value).suffix.lower() in MEDIA_SUFFIXES
                              and ('/' in value or key.endswith('_path')))):
        yield value


def validate_media(root, value):
    relative = Path(value)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix() != value:
        raise ValueError(f'Noncanonical media path: {value}')
    path = (root/relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing, empty, or external media: {value}')
    if path.relative_to(root).as_posix() != value:
        raise ValueError(f'Media path redirects: {value}')


def validate_rows(rows, target):
    counts = Counter(r['operation'] for r in rows)
    if (len(rows) != target or len({r['job_id'] for r in rows}) != target
            or len({mask_key(r) for r in rows}) != target
            or counts != Counter({op: target//4 for op in OPERATIONS})
            or not all(automatic_pass(r, require_operation=True) for r in rows)):
        raise ValueError('Final count, type, uniqueness, or audit validation failed')
    return counts


def finalize_exports(root, workers=16):
    root = Path(root).resolve()
    config = json.loads((root/'config.json').read_text())
    repo = Path(__file__).resolve().parents[1]
    # Only the unchanged data contract and audit predicate are needed for recovery.
    for name in ('production_state', 'operation_balance', 'shared_operation_quota'):
        relative = f'sa1b_pipeline/{name}.py'
        if file_hash(repo/relative) != config['code_hashes'][relative]:
            raise ValueError(f'Frozen export/audit contract differs: {relative}')
    final = root/'final'
    final.mkdir(exist_ok=True)
    with directory_lock(final/'finalize.lock.d'):
        started = time.time()
        def progress(stage, **extra):
            event = {'status': 'running', 'stage': stage, 'updated_unix': time.time(), **extra}
            atomic_json(final/'progress.json', event)
            emit(**event)
        progress('read_node_exports')
        rows, receipts = [], []
        for rank in range(config['nodes']):
            node = root/f'nodes/node{rank}'
            done = json.loads((node/'done.json').read_text())
            if done['status'] != 'complete' or done['rank'] != rank:
                raise ValueError(f'Node {rank} is incomplete')
            part = read_rows(node/'auto_pass.jsonl')
            counts = validate_rows(part, config['target_total']//config['nodes'])
            if dict(counts) != done['automatically_passed']:
                raise ValueError(f'Node {rank} completion counts differ')
            receipts.append({'rank': rank, 'count': len(part), 'sha256': file_hash(node/'auto_pass.jsonl')})
            rows.extend(part)
            progress('read_node_exports', nodes_read=rank+1, rows_read=len(rows))
        counts = validate_rows(rows, config['target_total'])
        paths = sorted({p for row in rows for p in media_references(row)})
        progress('validate_media', checked=0, total=len(paths))
        last = time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(validate_media, root, path) for path in paths]
            for checked, future in enumerate(as_completed(futures), 1):
                future.result()
                if time.monotonic()-last > 20 or checked == len(paths):
                    progress('validate_media', checked=checked, total=len(paths))
                    last = time.monotonic()
        # Refuse publication if node manifests changed during validation.
        for receipt in receipts:
            if file_hash(root/f"nodes/node{receipt['rank']}/auto_pass.jsonl") != receipt['sha256']:
                raise ValueError('Node export changed during finalization')
        progress('write_final_manifest', count=len(rows))
        write_rows(final/'auto_pass.jsonl', rows)
        summary = {'status': 'complete', 'count': len(rows), 'operations': dict(counts),
                   'unique_source_masks': len({mask_key(r) for r in rows}),
                   'unique_source_images': len({str(r['source_id']) for r in rows}),
                   'manual_review': 'not_performed', 'all_media_inside_dataset_root': True,
                   'media_files_checked': len(paths), 'media_validation': 'path containment, existence, nonzero size; no full image decode',
                   'manifest_sha256': file_hash(final/'auto_pass.jsonl'), 'node_exports': receipts,
                   'finalizer_sha256': file_hash(Path(__file__)),
                   'completed_unix': time.time(), 'finalization_seconds': time.time()-started}
        atomic_json(final/'summary.json', summary)
        # Keep failure.json as history; record successful recovery explicitly.
        for rank in range(config['nodes']):
            node = root/f'nodes/node{rank}'
            previous = json.loads((node/'progress.json').read_text())
            if previous.get('status') == 'failed' and previous.get('error') == '[Errno 38] Function not implemented':
                done = json.loads((node/'done.json').read_text())
                atomic_json(node/'progress.json', {**done, 'stage': 'finalized', 'updated_unix': time.time(),
                    'recovered_finalization_error': previous['error']})
        atomic_json(final/'progress.json', {**summary, 'stage': 'complete', 'updated_unix': time.time()})
        emit(**summary)
        return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    finalize_exports(args.root, args.workers)
