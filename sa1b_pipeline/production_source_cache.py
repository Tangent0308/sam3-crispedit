"""Stage only indexed SA-1B records into portable shared TFRecord packs."""
from __future__ import annotations

import hashlib
import io
import json
import struct
from functools import lru_cache
from pathlib import Path

from .production_state import digest, file_hash


def record_key(info, subdir):
    return digest([subdir, Path(info['file']).name, info['offset']])


def _feeder_import(name):
    """Import a feeder dependency; the locked production runtimes do not carry crc32c / tfd_utils, nodes
    install them next to the environments (scripts/evidence_production/setup_extras.sh, SA1B_FEEDER_LIB)."""
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError:
        import os
        import sys
        lib = os.environ.get('SA1B_FEEDER_LIB')
        if not lib:
            raise
        if lib not in sys.path:
            sys.path.append(lib)
        return importlib.import_module(name)


def read_frame(stream, offset, description, payload_length=None):
    crc32c = _feeder_import('crc32c')
    stream.seek(offset)
    if payload_length is not None:
        if not 0 < payload_length < 512*1024*1024:
            raise ValueError('invalid indexed TFRecord length')
        # One remote range read, rather than separate header/payload/trailer IO.
        frame = stream.read(payload_length+16)
        if len(frame) != payload_length+16:
            raise OSError(f'truncated indexed record: {description}')
        validated = read_frame(io.BytesIO(frame), 0, description)
        if len(validated) != len(frame):
            raise ValueError('indexed length disagrees with TFRecord header')
        return validated
    head = stream.read(12)
    if len(head) != 12:
        raise OSError(f'truncated record header: {description}')
    size = struct.unpack('<Q', head[:8])[0]
    if not 0 < size < 512*1024*1024:
        raise ValueError('invalid TFRecord length')
    payload, trailer = stream.read(size), stream.read(4)
    if len(payload) != size or len(trailer) != 4:
        raise OSError(f'truncated record: {description}')
    def masked_crc(data):
        crc = crc32c.crc32c(data)
        return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff
    if struct.unpack('<I', head[8:])[0] != masked_crc(head[:8]) or struct.unpack('<I', trailer)[0] != masked_crc(payload):
        raise ValueError(f'TFRecord checksum mismatch: {description}')
    return head+payload+trailer


def decode_frame(frame):
    Example = _feeder_import('tfd_utils.pb2').Example
    return {k: v.bytes_list.value[0] for k, v in Example.FromString(frame[12:-4]).features.feature.items()
            if v.bytes_list.value}


def validate_cache(cache_root, reservoir, expected_ready_sha256=None):
    """CPU-only bootstrap check, usable before inference dependencies install."""
    cache_root = Path(cache_root).resolve()
    if not (cache_root/'ready.json').is_file():
        raise RuntimeError('Shared SA-1B source cache is not ready; see source_cache/progress.json and logs/stage_source_cache.log')
    if expected_ready_sha256 and file_hash(cache_root/'ready.json') != expected_ready_sha256:
        raise ValueError('source cache completion manifest changed')
    ready = json.loads((cache_root/'ready.json').read_text())
    if ready['reservoir_sha256'] != file_hash(reservoir):
        raise ValueError('source cache belongs to another input reservoir')
    if ready['index_sha256'] != file_hash(cache_root/'index.json'):
        raise ValueError('shared source cache index checksum mismatch')
    for item in ready['packs']:
        pack = (cache_root/item['pack']).resolve()
        if not pack.is_relative_to(cache_root) or pack.stat().st_size != item['bytes']:
            raise ValueError(f'source pack missing, truncated or outside cache: {pack}')
    return ready


@lru_cache(maxsize=4)
def cache_index(cache_root):
    cache_root = Path(cache_root)
    ready = json.loads((cache_root/'ready.json').read_text())
    index_path = cache_root/'index.json'
    if file_hash(index_path) != ready['index_sha256']:
        raise ValueError('shared source cache index checksum mismatch')
    return json.loads(index_path.read_text())


def cached_record(cache_root, info, subdir):
    cache_root = Path(cache_root).resolve()
    index = cache_index(str(cache_root))
    key = record_key(info, subdir)
    if key not in index:
        raise ValueError(f'shared source cache missing frozen record: {key}')
    record = index[key]
    path = (cache_root/record['pack']).resolve()
    if not path.is_relative_to(cache_root):
        raise ValueError('cache pack escapes source cache directory')
    with path.open('rb') as stream:
        frame = read_frame(stream, record['offset'], path, record['length']-16)
    if len(frame) != record['length'] or hashlib.sha256(frame).hexdigest() != record['sha256']:
        raise ValueError('shared source cache record checksum mismatch')
    return decode_frame(frame)
