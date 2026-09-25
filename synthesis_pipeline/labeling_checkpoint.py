"""Atomic, dependency-checked checkpoints for interrupted labeling jobs.

Completion is published last. A file's existence alone never proves that an
edit or audit finished. No model settings or quality decisions are changed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def recover_rows(path):
    """Salvage whole records from old JSONL; malformed records must be retried."""
    path = Path(path)
    if not path.exists():
        return []
    records = {}
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        try:
            record = json.loads(text)
            records[record['image']] = record
        except (ValueError, KeyError, TypeError):
            print(f'RESUME: ignoring incomplete record {path}:{line}', flush=True)
    return list(records.values())


def bind_settings(root, settings, resume):
    """Reject a different invocation before reusing its completed cases."""
    path = Path(root) / 'checkpoint_settings.json'
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        if json.loads(path.read_text()) != settings:
            raise ValueError(f'Resume settings/input mismatch: {path}')
    else:
        atomic_json(path, settings)


def ensure_sources(root, source):
    link = Path(root) / 'sources'
    target = Path(source).resolve()
    if link.is_symlink() or link.exists():
        if link.resolve() != target:
            raise ValueError(f'Resume source directory mismatch: {link}')
    else:
        link.symlink_to(target)


class CaseCheckpoints:
    def __init__(self, root, settings):
        self.root = Path(root) / 'checkpoints'
        self.settings = settings

    def signature(self, row, dependencies=None):
        return fingerprint(dict(row=row, settings=self.settings, dependencies=dependencies))

    def load(self, row, dependencies=None):
        path = self.root / (row['image'] + '.json')
        try:
            value = json.loads(path.read_text())
            if value['version'] != 1 or fingerprint(value['record']) != value['record_sha256']:
                return None
            if value['signature'] != self.signature(row, dependencies):
                return None
            for artifact in value['artifacts']:
                if file_digest(artifact['path']) != artifact['sha256']:
                    return None
            return value['record']
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, row, record, artifacts=(), dependencies=None):
        files = [dict(path=str(Path(p).resolve()), sha256=file_digest(p)) for p in artifacts]
        atomic_json(self.root / (row['image'] + '.json'), dict(
            version=1,signature=self.signature(row, dependencies), record=record,
            record_sha256=fingerprint(record),artifacts=files))


def wait_workers(jobs, label):
    """Notice a dead worker immediately, even when earlier workers are busy."""
    import time
    try:
        while any(proc.poll() is None for _, proc, _ in jobs):
            failures = [(i, proc.returncode) for i, proc, _ in jobs
                        if proc.poll() not in (None, 0)]
            if failures:
                raise RuntimeError(f'{label} worker failures: {failures}')
            time.sleep(.5)
        failures = [(i, proc.returncode) for i, proc, _ in jobs if proc.returncode]
        if failures:
            raise RuntimeError(f'{label} worker failures: {failures}')
    finally:
        for _, proc, _ in jobs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, log in jobs:
            try:
                proc.wait(timeout=20)
            except __import__('subprocess').TimeoutExpired:
                proc.kill(); proc.wait()
            log.close()
