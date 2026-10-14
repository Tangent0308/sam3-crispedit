import numpy as np
from PIL import Image

from sa1b_pipeline.image_cache import SourceImageCache


def test_cache_keeps_pixels_isolated_and_invalidates_changed_source(tmp_path):
    path = tmp_path/'source.png'
    Image.new('RGB', (20, 30), 'red').save(path)
    cache = SourceImageCache(1800)
    first = cache.get(path)
    first.paste('blue', (0, 0, 20, 30))
    second = cache.get(path)
    assert second.getpixel((0, 0)) == (255, 0, 0)
    assert cache.stats()['hits'] == 1
    Image.new('RGB', (30, 20), 'green').save(path)
    third = cache.get(path)
    assert third.size == (30, 20)
    assert third.getpixel((0, 0)) == (0, 128, 0)
    assert cache.bytes <= 1800


def test_cached_planning_views_match_original_pixels_and_binding(tmp_path):
    from types import SimpleNamespace

    from sa1b_pipeline.per_mask import single_call_views
    source = tmp_path/'image.png'
    mask_path = tmp_path/'mask.png'
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)).save(source)
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:45, 30:60] = 255
    mask[25:35, 40:50] = 0
    Image.fromarray(mask).save(mask_path)
    c = SimpleNamespace(image_path=str(source))
    o = SimpleNamespace(mask_path=str(mask_path))
    uncached = single_call_views(c, o, tmp_path/'a')
    cached = single_call_views(c, o, tmp_path/'b', source_cache=SourceImageCache())
    assert uncached[3:] == cached[3:]
    for p, q in zip(uncached[0], cached[0]):
        assert Image.open(p).tobytes() == Image.open(q).tobytes()
