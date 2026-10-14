"""Verified node-local model copies; canonical model IDs stay in data contracts."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from .production_state import atomic_json, file_hash


def stage_model(model, cache_root, progress=lambda _: None):
    source = Path(model).resolve()
    cache_root = Path(cache_root)
    if cache_root.resolve().is_relative_to(Path('/mnt')):
        raise ValueError('model cache must be on node-local storage')
    files = [p for p in sorted(source.rglob('*')) if p.is_file()
             and not any(part.startswith('.') for part in p.relative_to(source).parts)]
    if not files:
        raise ValueError(f'empty model directory: {source}')
    inventory = {str(p.relative_to(source)): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                 for p in files}
    identity = hashlib.sha256(json.dumps([str(source), inventory], sort_keys=True).encode()).hexdigest()
    root = cache_root/identity
    root.mkdir(parents=True, exist_ok=True)
    with (root/'copy.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = root/'files.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        target = root/'model'
        target.mkdir(exist_ok=True)
        total = sum(r['size'] for r in inventory.values())
        done = 0
        for name, meta in inventory.items():
            src, dst = source/name, target/name
            saved = state.get(name, {})
            if (dst.is_file() and dst.stat().st_size == meta['size']
                    and saved.get('source') == meta and file_hash(dst) == saved.get('sha256')):
                done += meta['size']
                progress({'stage': 'model_cache', 'model': str(source), 'bytes_done': done, 'bytes_total': total})
                continue
            if shutil.disk_usage(root).free < meta['size'] + 2*1024**3:
                raise OSError(f'Insufficient local disk for model cache: {root}')
            dst.parent.mkdir(parents=True, exist_ok=True)
            temporary = dst.with_name(dst.name+'.partial')
            hashed = hashlib.sha256()
            copied, last_log = 0, 0
            with src.open('rb') as reader, temporary.open('wb') as writer:
                while chunk := reader.read(8*1024**2):
                    writer.write(chunk)
                    hashed.update(chunk)
                    copied += len(chunk)
                    if time.monotonic()-last_log >= 10:
                        progress({'stage': 'model_cache', 'model': str(source), 'file': name,
                                  'bytes_done': done+copied, 'bytes_total': total})
                        last_log = time.monotonic()
                writer.flush()
                os.fsync(writer.fileno())
            if (copied != meta['size'] or src.stat().st_mtime_ns != meta['mtime_ns']
                    or file_hash(temporary) != hashed.hexdigest()):
                raise ValueError(f'model changed or cache copy damaged: {src}')
            temporary.replace(dst)
            state[name] = {'source': meta, 'sha256': hashed.hexdigest()}
            atomic_json(state_path, state)
            done += copied
        atomic_json(root/'ready.json', {'source': str(source), 'identity': identity,
                                       'bytes': total, 'files_sha256': file_hash(state_path)})
        return str(target)


def resolve_model(model):
    """Only loading uses the copy; fingerprints/provenance keep the original ID."""
    mapping = json.loads(os.environ.get('SA1B_LOCAL_MODEL_MAP', '{}'))
    target = mapping.get(str(model))
    if target is None:
        return model
    path = Path(target)
    ready = json.loads((path.parent/'ready.json').read_text())
    if ready['source'] != str(Path(model).resolve()) or not path.is_dir():
        raise ValueError('local model mapping does not match canonical model')
    if file_hash(path.parent/'files.json') != ready['files_sha256']:
        raise ValueError('local model cache manifest changed')
    return target
