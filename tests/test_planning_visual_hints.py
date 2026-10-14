"""Comparison cues must never silently change the target or selected pixels."""
import hashlib

import numpy as np
from PIL import Image

from sa1b_pipeline.planning_visual_hints import annotate_crop, nearest_peer, visual_hint_prompt


def test_peer_retrieval_excludes_self_overlap_wrong_category_and_invisible_region():
    target = dict(object_id='a', category='petal', bbox=[60, 60, 80, 100])
    peer = dict(object_id='b', category='petal, flower', bbox=[85, 60, 105, 100])
    objects = [target, dict(target, object_id='nested', bbox=[61, 60, 79, 100]),
               dict(peer, object_id='wrong', category='leaf'),
               dict(peer, object_id='outside', bbox=[0, 0, 20, 40]), peer]
    assert nearest_peer(target, objects, [40, 40, 120, 120]) == peer
    assert nearest_peer(target, objects[:-1], [40, 40, 120, 120]) is None


def test_annotation_preserves_target_rgb_source_files_and_correct_direction(tmp_path):
    size = (200, 160)
    rgb = np.random.default_rng(17).integers(0, 256, (160, 200, 3), dtype=np.uint8)
    a = np.zeros((160, 200), dtype=np.uint8)
    b = a.copy()
    a[90:130, 70:90] = 255
    b[90:130, 110:130] = 255
    scene, mask, other, crop = [tmp_path / name for name in ['scene.png', 'a.png', 'b.png', 'crop.png']]
    Image.fromarray(rgb).save(scene)
    Image.fromarray(a).save(mask)
    Image.fromarray(b).save(other)
    panel = Image.new('RGB', (200, 198), 'white')
    panel.paste(Image.fromarray(rgb), (0, 38))
    panel.save(crop)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [scene, mask, other]}
    target = dict(object_id='a', category='petal', bbox=[70, 90, 90, 130], mask_path=str(mask))
    peer = dict(object_id='b', category='petal', bbox=[110, 90, 130, 130], mask_path=str(other))
    info = annotate_crop(crop, mask, size, (0, 0, 200, 160), target, [target, peer], 'peer')
    result = np.asarray(Image.open(crop))[38:]
    assert np.array_equal(result[a > 0], rgb[a > 0])
    assert np.array_equal(result[b > 0], rgb[b > 0])
    assert not np.array_equal(result, rgb)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert info['peer']['target_minus_peer_center_xy_pixels'] == [-40.0, 0.0]
    prompt = visual_hint_prompt('Original task', info)
    assert 'NOT another editing target' in prompt and 'Original task' in prompt


def test_arrow_mode_has_no_peer_and_handles_frame_touching_target(tmp_path):
    a = np.ones((80, 100), dtype=np.uint8) * 255
    mask, crop = tmp_path / 'mask.png', tmp_path / 'crop.png'
    Image.fromarray(a).save(mask)
    Image.new('RGB', (100, 118), (70, 90, 120)).save(crop)
    target = dict(object_id='a', category='surface', bbox=[0, 0, 100, 80])
    info = annotate_crop(crop, mask, (100, 80), (0, 0, 100, 80), target, [target], 'arrow')
    assert info['peer'] is None
    assert (np.asarray(Image.open(crop))[38:] == [70, 90, 120]).all()


def test_exact_scene_outline_preserves_selected_pixels_and_source(tmp_path):
    from sa1b_pipeline.planning_visual_hints import annotate_full_scene
    from PIL import ImageFilter
    rng=np.random.default_rng(19)
    rgb=rng.integers(0,256,(160,200,3),dtype=np.uint8)
    mask=np.zeros((160,200),dtype=np.uint8)
    mask[20:140,30:150]=255
    mask[50:110,70:100]=0  # excluded foreground hole inside a background target
    src=tmp_path/'source.png';mp=tmp_path/'mask.png';out=tmp_path/'scene.png'
    Image.fromarray(rgb).save(src);Image.fromarray(mask).save(mp)
    hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,mp]}
    annotate_full_scene(src,mp,out,200)
    result=np.asarray(Image.open(out))
    assert np.array_equal(result[mask>0],rgb[mask>0])
    dilated=np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(9)))>0
    assert np.array_equal(result[~dilated],rgb[~dilated])
    assert not np.array_equal(result[mask==0],rgb[mask==0])
    assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}
    assert np.array_equal(result[70,85],rgb[70,85])  # hole interior remains original
