"""Atomic shared checkpoints; SQLite queues remain node-local and disposable."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

OPERATIONS = ('add', 'replace', 'remove', 'attribute')
MEDIA_KEYS = {'image', 'image_path', 'source_image', 'edited_image', 'mask', 'mask_path',
              'support_mask', 'execution_mask_path', 'change_mask', 'new_object_mask',
              'protected_mask_path', 'target_panel', 'images', 'image_paths'}
MEDIA_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    atomic_text(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'checkpoint/config changed: {path}')
    else:
        atomic_json(path, value)


def portable(value, root, key='', base=None, cache=None):
    """Recursively materialize every image reference, including nested provenance.

    Files already under root stay in place. External files are copied by content
    hash, never symlinked. Relative paths in final manifests are root-relative.
    """
    root = Path(root).resolve()
    cache = {} if cache is None else cache
    if isinstance(value, dict):
        return {k: portable(v, root, k, base, cache) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v, root, key, base, cache) for v in value]
    if not isinstance(value, str):
        return value
    is_media = key in MEDIA_KEYS or (Path(value).suffix.lower() in MEDIA_SUFFIXES
                                    and ('/' in value or key.endswith('_path')))
    if not is_media or not value:
        return value
    if value in cache:
        return cache[value]
    src = Path(value)
    if not src.is_absolute():
        src = Path(base or root) / src
    src = src.resolve(strict=True)
    if not src.is_file():
        raise ValueError(f'media is not a file: {src}')
    if not src.is_relative_to(root):
        hashed = file_hash(src)
        dest = root / 'media' / hashed[:2] / (hashed + src.suffix.lower())
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temp = dest.with_name(dest.name + '.' + uuid.uuid4().hex + '.tmp')
            shutil.copyfile(src, temp)
            if file_hash(temp) != hashed:
                raise ValueError(f'media changed during copy: {src}')
            os.replace(temp, dest)
        src = dest
    result = src.relative_to(root).as_posix()
    cache[value] = result
    return result


def mask_key(row):
    return str(row['source_id']), str(row['object_id'])


class PlanningJournal:
    """Never resubmit an ambiguous in-flight MLLM call after a process crash."""
    def __init__(self, root, contract, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=resume)
        immutable_json(self.root / 'contract.json', contract)

    def path(self, key):
        return self.root / (digest(key) + '.json')

    def submit(self, keys):
        # Write intent before GPU submission. On a crash, conservatively skip
        # every uncertain entry; this can undercount actual calls, never retry it.
        for key in keys:
            path = self.path(key)
            if path.exists():
                raise ValueError('mask already submitted')
            atomic_json(path, {'key': key, 'status': 'submitted'})

    def finish(self, key, audit, terminal, case):
        path = self.path(key)
        if json.loads(path.read_text())['status'] != 'submitted':
            raise ValueError('mask completion has no matching submission')
        atomic_json(path, {'key': key, 'status': 'complete', 'audit': audit,
                           'terminal': terminal, 'case': case})

    def entries(self):
        return [json.loads(p.read_text()) for p in sorted(self.root.glob('*.json'))
                if p.name != 'contract.json']
