import json
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import select_evidence_groups as sel  # noqa: E402
from sa1b_pipeline import evidence_selection as es  # noqa: E402
from sa1b_pipeline import production_pool as pp  # noqa: E402
from sa1b_pipeline import shard_feeder as sf  # noqa: E402

NATURAL = {'remove': 'natural', 'replace': 'natural', 'attribute': 'natural', 'add_anchor': 'natural'}
AFFORDANCE = {'cup': NATURAL, 'box': NATURAL, 'table': dict(NATURAL, remove='unnatural', replace='unnatural'),
              'sign': dict(NATURAL, add_anchor='unnatural')}
SETTINGS = es.selection_settings(twin_members=3, twin_members_add=4, worn_check=True, covering_check=True,
                                 no_surface_recolor=True, skip_grayscale=True)


def _rle(mask):
    from pycocotools import mask as coco
    r = coco.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {'size': [int(x) for x in r['size']], 'counts': r['counts'].decode()}


def _candidate(tmp_path, sid, boxes, size=(640, 480)):
    w, h = size
    image = tmp_path / f'{sid}.jpg'
    rng = np.random.RandomState(abs(hash(sid)) % 2**31)
    Image.fromarray((rng.rand(h, w, 3) * 255).astype(np.uint8)).save(image)
    objects = []
    for i, (category, x0, y0, x1, y1) in enumerate(boxes):
        m = np.zeros((h, w), bool)
        m[y0:y1, x0:x1] = True
        objects.append({'object_id': str(i), 'category': category, 'description': f'a {category}', 'rle': _rle(m),
                        'bbox': [x0, y0, x1 - x0, y1 - y0], 'metadata': {'score': .9}})
    return {'source_id': sid, 'image_path': str(image), 'objects': objects}


def _scenes(tmp_path):
    """Images with repeated separate objects (twin material) and mixed categories (multi material)."""
    scenes = []
    for k in range(10):
        boxes = [('cup', 40 + 120 * j, 60 + 7 * k, 90 + 120 * j, 110 + 7 * k) for j in range(2 + k % 3)]
        boxes += [('table', 20, 300, 400, 460), ('sign', 450, 250 + 5 * k, 520, 320 + 5 * k)]
        if k % 2:
            boxes += [('box', 520, 40, 580, 100)]
        scenes.append(_candidate(tmp_path, f'sa_{1000 + k}', boxes))
    return scenes


def test_twin_menu_predicts_the_operation_select_twin_takes(tmp_path):
    rng = np.random.RandomState(1)
    menus = [(c, es.image_menu(c, AFFORDANCE, SETTINGS)) for c in _scenes(tmp_path)]
    assert any(m['twin'][op] >= 2 for _, m in menus for op in es.OPERATIONS)
    for _ in range(40):
        deficit = Counter({op: int(v) for op, v in zip(es.OPERATIONS, rng.randint(-2, 6, size=4))})
        for candidate, menu in menus:
            row = es.select_twin(candidate, AFFORDANCE, Counter(deficit), members=SETTINGS['members'],
                                 limits=SETTINGS['limits'], support_check=True, saturated_ok=False,
                                 members_by_operation=SETTINGS['members_by_operation'])
            ranked = sorted(es.OPERATIONS, key=lambda op: (-deficit[op], op))
            expected = [op for op in ranked if deficit[op] > 0 and menu['twin'][op] >= 2][:1]
            assert ([row['operation']] if row else []) == expected
            if row:
                assert len(row['objects']) == menu['twin'][row['operation']]


def _records(tmp_path, scenes):
    menus = {}
    for c in scenes:
        d = tmp_path / 'sources' / c['source_id']
        d.mkdir(parents=True)
        (d / 'complete.json').write_text(json.dumps({'status': 'ready', 'candidate': c}))
        menus[c['source_id']] = {'sid': c['source_id'], 'record': str(d / 'complete.json'),
                                 **es.image_menu(c, AFFORDANCE, SETTINGS)}
    return menus


def test_menu_selection_is_the_same_for_any_worker_count(tmp_path):
    menus = _records(tmp_path, _scenes(tmp_path))
    cache = tmp_path / 'affordance.json'
    cache.write_text(json.dumps(AFFORDANCE))
    order = sorted(menus)
    results = []
    for workers in (1, 3):
        deficits = {'twins': Counter({'add': 4, 'remove': 2, 'replace': 2, 'attribute': 4}),
                    'multi': Counter({'add': 3, 'remove': 3, 'replace': 0, 'attribute': 3})}
        args = SimpleNamespace(twins=4, multi=4, workers=workers, affordance_cache=cache)
        rows, _, info = sel.select_from_menus(args, order, menus, deficits, SETTINGS, Counter())
        results.append(([(r['candidate']['source_id'], r['operation']) for r in rows['twins']],
                        [(r['candidate']['source_id'], tuple(r['operations'])) for r in rows['multi']]))
        used = [s for s, _ in results[-1][0]] + [s for s, _ in results[-1][1]]
        assert len(used) == len(set(used))                      # one group per image
        assert all(set(r['operations']) & {'add', 'remove', 'attribute'} for r in rows['multi'])  # a needed op each
        assert info['twin_allocation'] == es.twin_allocation(4, Counter({'add': 4, 'remove': 2, 'replace': 2,
                                                                          'attribute': 4}))
    assert results[0] == results[1]
    assert results[0][0] and results[0][1]


def test_multi_prediction_needs_a_needed_operation_and_two_targets():
    menu = {'multi': ['add', 'remove', 'attribute']}
    assert sel.predict_multi(menu, Counter({'replace': 5})) == []
    assert sel.predict_multi(menu, Counter({'remove': 2, 'add': 1})) == ['remove', 'add']
    assert sel.predict_multi(menu, Counter({'attribute': 1})) == ['attribute', 'add']
    assert sel.predict_multi({'multi': ['add']}, Counter({'add': 3})) == []


def test_twin_allocation_splits_the_wave_by_remaining_need():
    assert es.twin_allocation(10, Counter({'add': 4, 'remove': 2})) == {'add': 2, 'remove': 1, 'replace': 0,
                                                                         'attribute': 0}
    out = es.twin_allocation(10, Counter({'add': 60, 'remove': 20, 'replace': 20, 'attribute': 0}))
    assert sum(out.values()) == 10 and out['add'] == 6 and out['attribute'] == 0


def _frame(payload):
    import crc32c

    def masked(data):
        crc = crc32c.crc32c(data)
        return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff
    head = struct.pack('<Q', len(payload))
    return head + struct.pack('<I', masked(head)) + payload + struct.pack('<I', masked(payload))


def test_shard_reader_emits_every_frame_once_with_parallel_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sf, 'CHUNK', 1000)        # several reads per run
    path, spans, offset = tmp_path / 'shard.tfrecord', [], 0
    with path.open('wb') as f:
        for i in range(57):
            payload = bytes([i % 251]) * (100 + 37 * i)
            f.write(_frame(payload))
            spans.append((offset, len(payload), f'k{i}'))
            offset += len(payload) + 16
    got = {}
    sf.read_frames(path, list(reversed(spans)), 5, lambda key, frame: got.setdefault(key, frame))
    assert sorted(got) == sorted(k for _, _, k in spans)
    for start, length, key in spans:
        assert got[key][12:12 + length] == bytes([int(key[1:]) % 251]) * length
    assert len(sf.runs(spans, 5)) == 5 and sum(len(r) for r in sf.runs(spans, 5)) == 57
    with pytest.raises(OSError):
        sf.read_frames(path, [(offset - 10, 100, 'bad')], 1, lambda *a: None)


def test_pool_asks_for_shards_from_menu_availability(tmp_path):
    rank_dir = tmp_path / 'nodes/node0/pool/menus'
    rank_dir.mkdir(parents=True)
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs/shard_order.json').write_text(json.dumps({'0': [5, 9]}))
    rows = [{'sid': f's{i}', 'twin': {'add': 2 if i < 10 else 0, 'remove': 2, 'replace': 0, 'attribute': 3},
             'multi': ['add', 'remove'] if i % 2 else []} for i in range(100)]
    (rank_dir / '00005.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (rank_dir / '00005.json').write_text(json.dumps({'images': 100, 'twin_available': {'add': 10, 'remove': 100,
                                                     'replace': 0, 'attribute': 100}, 'multi_available': 50}))
    config = {'selection': {'twin_members': 3, 'twin_members_add': 4, 'worn_check': True, 'covering_check': True,
                            'no_surface_recolor': True, 'skip_grayscale': True},
              'pool': {'availability_factor': 1.5}}
    pool = pp.Pool(tmp_path, 0, config)
    plan = {'plan': {'twins': {'groups': 8, 'deficits': {'add': 8, 'remove': 4, 'replace': 0, 'attribute': 0}},
                     'multi': {'groups': 10, 'deficits': {'add': 5, 'remove': 5, 'replace': 0, 'attribute': 0}}}}
    assert pool.shortage(plan, set())[0] == 0
    # Only 10% of images can serve an add twin: 12 add twins need ~120 images, i.e. one more shard.
    plan['plan']['twins'] = {'groups': 30, 'deficits': {'add': 24, 'remove': 0, 'replace': 0, 'attribute': 0}}
    used = {f's{i}' for i in range(5)}
    missing, info = pool.shortage(plan, used)
    assert missing == 1 and info['available']['twin']['add'] == 5


def test_parallel_planning_views_match_the_serial_path(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts import run_group_pipeline as rgp
    from sa1b_pipeline import per_mask_planning as pmp
    scenes = _scenes(tmp_path)[:3]
    serial = []
    for c in scenes:
        per_mask, masks, digest = pmp.views(c, c['objects'][:2], tmp_path / 'serial')
        serial.append((digest, rgp._geometry(masks), [Path(v).name for pair in per_mask for v in pair]))
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        parallel = list(pool.map(pmp.views_with_geometry, [(c, c['objects'][:2], str(tmp_path / 'parallel')) for c in scenes]))
    for (digest, geometry, names), (per_mask, masks, digest2, geometry2) in zip(serial, parallel):
        assert digest == digest2 and geometry == geometry2
        assert names == [Path(v).name for pair in per_mask for v in pair]
        for name in ('original.png', 'U1_crop.png', 'U2_crop.png'):
            assert (tmp_path / 'serial' / digest / name).read_bytes() == (tmp_path / 'parallel' / digest / name).read_bytes()
