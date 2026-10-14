import io
import json
import struct
from pathlib import Path

import crc32c
import numpy as np
import pytest
from PIL import Image
from pycocotools import mask as coco
from tfd_utils.pb2 import Example

from sa1b_pipeline.production_source_cache import (
    cache_index,
    cached_record,
    read_frame,
    validate_cache,
)
from sa1b_pipeline.production_sources import materialize, read_record
from sa1b_pipeline.production_state import file_hash, write_rows
from scripts.production.stage_source_cache import stage


def frame(fields):
    example = Example()
    for name, value in fields.items():
        example.features.feature[name].bytes_list.value.append(value)
    payload = example.SerializeToString()
    length = struct.pack('<Q', len(payload))
    def crc(data):
        value = crc32c.crc32c(data)
        return struct.pack('<I', (((value >> 15) | (value << 17))+0xa282ead8) & 0xffffffff)
    return length+crc(length)+payload+crc(payload), len(payload)


@pytest.fixture
def reservoir(tmp_path):
    source, root = tmp_path/'original', tmp_path/'dataset'
    rows, originals = [], {}
    for i in range(4):
        sid = f'sa_{i}'
        image = io.BytesIO()
        Image.new('RGB', (12, 10), (i*50, 30, 40)).save(image, format='JPEG')
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        rle = coco.encode(np.asfortranarray(mask))
        rle['counts'] = rle['counts'].decode()
        obj = {'object_id': 0, 'segmentation': rle, 'category': 'square'}
        fields = {'image': {'key': sid.encode(), 'jpg': image.getvalue()},
                  'annotation': {'key': sid.encode(), 'objects_json': json.dumps([obj]).encode()}}
        row = {'source_id': sid, 'rank': i}
        originals[sid] = fields
        for kind, subdir in [('image', 'sa1b_tfd_1k'), ('annotation', 'sa1b_dense_label/annotation')]:
            path = source/subdir/f'{i%2}.tfrecord'
            path.parent.mkdir(parents=True, exist_ok=True)
            raw, length = frame(fields[kind])
            with path.open('ab') as f:
                offset = f.tell()
                f.write(raw)
            row[kind] = {'file': str(path), 'offset': offset, 'length': length}
        rows.append(row)
    write_rows(root/'inputs/reservoir.jsonl', rows)
    return root, source, rows, originals


def test_all_cached_inputs_survive_missing_original_mount(reservoir):
    root, source, rows, originals = reservoir
    stage(root, source, workers=2)
    ready = validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl')
    assert ready['sources'] == 4 and ready['records'] == 8
    for row in rows:
        for kind, subdir in [('image', 'sa1b_tfd_1k'), ('annotation', 'sa1b_dense_label/annotation')]:
            assert cached_record(root/'source_cache', row[kind], subdir) == read_record(source, row[kind], subdir)
    source.rename(source.with_name('hidden-mount'))
    for row in rows:
        out = materialize(row, root, source, {'source_ids': [], 'image_hashes': [], 'benchmark_sha256': 'frozen'})
        assert out['status'] == 'ready'
        assert Path(out['candidate']['image_path']).read_bytes() == originals[row['source_id']]['image']['jpg']
        assert len(out['candidate']['objects']) == 1
    # Re-staging complete packs reuses checksummed copies without original mount.
    frozen = file_hash(root/'source_cache/ready.json')
    stage(root, source, workers=2)
    assert file_hash(root/'source_cache/ready.json') == frozen
    assert validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl', frozen)['records'] == 8
    with pytest.raises(ValueError, match='completion manifest changed'):
        validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl', 'wrong-expected-sha256')


def test_cache_identity_corruption_and_incomplete_pack_are_blocked(reservoir):
    root, source, rows, _ = reservoir
    with pytest.raises(RuntimeError, match='not ready'):
        validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl')
    stage(root, source, workers=2)
    write_rows(root/'other.jsonl', rows[:1])
    with pytest.raises(ValueError, match='another input reservoir'):
        validate_cache(root/'source_cache', root/'other.jsonl')
    cache_index.cache_clear()
    index = json.loads((root/'source_cache/index.json').read_text())
    record = next(iter(index.values()))
    pack = root/'source_cache'/record['pack']
    data = bytearray(pack.read_bytes())
    data[record['offset']+20] ^= 1
    pack.write_bytes(data)
    matching = next(row for row in rows if row['source_id'] == record['source_id'])
    failures = 0
    for key, subdir in [('image', 'sa1b_tfd_1k'), ('annotation', 'sa1b_dense_label/annotation')]:
        try:
            cached_record(root/'source_cache', matching[key], subdir)
        except ValueError as exc:
            assert 'checksum' in str(exc)
            failures += 1
    assert failures == 1
    pack.write_bytes(data[:-1])
    with pytest.raises(ValueError, match='truncated'):
        validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl')


def test_index_mutation_is_rejected(reservoir):
    root, source, rows, _ = reservoir
    stage(root, source, workers=1)
    cache_index.cache_clear()
    (root/'source_cache/index.json').write_text('{}')
    with pytest.raises(ValueError, match='index checksum'):
        cached_record(root/'source_cache', rows[0]['image'], 'sa1b_tfd_1k')


@pytest.mark.parametrize('length_change', [-1, 1])
def test_indexed_length_mismatch_is_rejected(length_change):
    raw, size = frame({'key': b'sa_1'})
    with pytest.raises((ValueError, OSError)):
        read_frame(io.BytesIO(raw+b'extra'), 0, 'fixture', size+length_change)


def test_actual_bootstrap_preflight_works_without_hdfs_or_site_packages(reservoir):
    import subprocess
    import sys
    root, source, _, _ = reservoir
    stage(root, source, workers=2)
    source.rename(source.with_name('mount-unavailable'))
    model = root/'model'
    model.mkdir()
    (model/'config.json').write_text('{}')
    (model/'model_index.json').write_text('{}')
    (root/'config.json').write_text(json.dumps({'source_backend': 'shared_record_cache_v1',
        'source_root': str(source), 'planner_model': str(model), 'editor_model': str(model),
        'audit_model': str(model)}))
    repo = Path(__file__).resolve().parents[1]
    script = (repo/'scripts/production/bootstrap_arnold_4node.sh').read_text()
    preflight = script.split("<<'PY'\n", 1)[1].split('\nPY', 1)[0]
    result = subprocess.run([sys.executable, '-S', '-', str(root)], input=preflight,
                            capture_output=True, text=True, cwd=repo, check=True)
    assert '4 images, 8 records; HDFS mount not required' in result.stdout
