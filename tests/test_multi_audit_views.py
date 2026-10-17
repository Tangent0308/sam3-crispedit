import numpy as np
from PIL import Image
from sa1b_pipeline.validate import _target_panel


def test_add_detail_centers_changed_item_even_in_anchor_hole(tmp_path):
    source = np.full((600, 1200, 3), 70, dtype=np.uint8)
    source[250:310, 570:630] = 240  # An old sign inside a fence-mask hole.
    result = source.copy(); result[250:310, 570:630] = (0, 0, 220)
    mask = np.zeros((600, 1200), dtype=np.uint8)
    mask[100:500, 100:1100] = 255; mask[240:320, 560:640] = 0
    for name, array in [('source', source), ('result', result), ('mask', mask)]:
        Image.fromarray(array).save(tmp_path / (name + '.png'))
    support = np.full_like(mask, 255); Image.fromarray(support).save(tmp_path/'support.png')
    row = dict(source_image=str(tmp_path/'source.png'), edited_image=str(tmp_path/'result.png'),
               mask=str(tmp_path/'mask.png'), support_mask=str(tmp_path/'support.png'), operation='add')
    wide = _target_panel(row,tmp_path/'panels',binary_locator=True)
    focused = _target_panel(row,tmp_path/'panels',binary_locator=True,tight_detail=True,focus_change=True)
    def bright_fraction(path):
        tile=np.asarray(Image.open(path))[48:, :512]
        return np.mean(np.all(tile>220,axis=2))
    assert focused != wide
    assert bright_fraction(focused) > 10 * bright_fraction(wide)
    # Changing allowed support must invalidate the detail view cache; no
    # accepted difference then falls back to the anchor view.
    support[:] = 0; Image.fromarray(support).save(tmp_path/'support.png')
    fallback = _target_panel(row,tmp_path/'panels',binary_locator=True,tight_detail=True,focus_change=True)
    assert fallback != focused
    assert bright_fraction(fallback) < bright_fraction(focused)
