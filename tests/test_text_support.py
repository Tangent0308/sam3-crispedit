import numpy as np
import pytest

from sa1b_pipeline import text_edits as te
from sa1b_pipeline import support_graph as sg


def test_payload_text_contract():
    assert te.payload_with_text('a small wooden sign', 'OPEN') == 'a small wooden sign reading "OPEN"'
    assert te.payload_with_text('a red star sticker', '') == 'a red star sticker'
    with pytest.raises(ValueError):
        te.payload_with_text('a small wooden sign', '')  # signs need wording
    with pytest.raises(ValueError):
        te.payload_with_text('a potted plant', 'HELLO')  # wording needs a text-bearing item
    with pytest.raises(ValueError):
        te.payload_with_text('a banner', 'THIS IS FAR TOO LONG TO RENDER')
    with pytest.raises(ValueError):
        te.payload_with_text('a sign reading "X"', 'OPEN')


def test_requirement_from_spec():
    add = {'operation': 'add', 'payload': 'a small wooden sign reading "OPEN"', 'attributes': []}
    assert te.requirement(add) == {'mode': 'exact', 'text': 'OPEN', 'replaces_source': False}
    assert te.requirement({'operation': 'add', 'payload': 'a vase', 'attributes': []}) == {'mode': 'none'}
    attr = {'operation': 'attribute', 'payload': '', 'attributes': [{'property': 'text', 'value': 'FRESH COFFEE'}]}
    assert te.requirement(attr)['mode'] == 'exact' and te.requirement(attr)['replaces_source']
    color = {'operation': 'attribute', 'payload': '', 'attributes': [{'property': 'color', 'value': 'red'}]}
    assert te.requirement(color) == {'mode': 'preserve'}
    assert te.requirement({'operation': 'remove', 'payload': '', 'attributes': []}) == {'mode': 'erase'}


def _tok(text, x0, y0, x1, y1, conf=.95):
    return {'text': text, 'conf': conf, 'box': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def test_compare_exact_and_garble():
    shape = (100, 200)
    changed = np.zeros(shape, bool); changed[20:60, 20:180] = True
    target = changed.copy()
    req = {'mode': 'exact', 'text': 'Open Now'}
    assert te.compare([], [_tok('OPEN', 30, 25, 90, 55), _tok('NOW', 100, 25, 170, 55)], changed, target, req)['decision'] == 'pass'
    assert te.compare([], [_tok('0PEN NOW', 30, 25, 170, 55)], changed, target, req)['decision'] == 'pass'  # O/0 OCR confusion
    bad = te.compare([], [_tok('OPFN', 30, 25, 90, 55), _tok('NOW', 100, 25, 170, 55)], changed, target, req)
    assert bad['decision'] == 'fail' and 'requested_text_not_found' in bad['reasons']
    extra = te.compare([], [_tok('OPEN NOW', 30, 25, 150, 55), _tok('VERT', 152, 25, 178, 55)], changed, target, req)
    assert 'unrequested_text' in extra['reasons'] and extra['unrequested_text'] == ['VERT']
    none = te.compare([], [_tok('NEWUA', 30, 25, 90, 55)], changed, target, {'mode': 'none'})
    assert none['decision'] == 'fail' and none['unrequested_text'] == ['NEWUA']
    # Unchanged pre-existing text outside the edit is not "new".
    untouched = te.compare([_tok('EXIT', 0, 70, 40, 95)], [_tok('EXIT', 0, 70, 40, 95)], changed, target, {'mode': 'none'})
    assert untouched['decision'] == 'pass'


def test_compare_replaced_text_must_be_gone():
    shape = (100, 200)
    target = np.zeros(shape, bool); target[10:90, 10:190] = True
    changed = target.copy()
    req = {'mode': 'exact', 'text': 'CAFE', 'replaces_source': True}
    src = [_tok('BAKERY', 30, 30, 170, 60)]
    ok = te.compare(src, [_tok('CAFE', 60, 30, 140, 60)], changed, target, req)
    assert ok['decision'] == 'pass'
    left = te.compare(src, [_tok('CAFE', 60, 62, 140, 85), _tok('BAKERY', 30, 30, 170, 60)], changed, target, req)
    assert 'old_text_remains' in left['reasons']
    # A smaller secondary source line may stay; only the headline must change.
    src2 = src + [_tok('since 1990', 60, 70, 140, 80)]
    kept_small = te.compare(src2, [_tok('CAFE', 60, 30, 140, 60), _tok('since 1990', 60, 70, 140, 80)], changed, target, req)
    assert kept_small['decision'] == 'pass'
    same = te.compare([_tok('CAFE', 60, 30, 140, 60)], [_tok('CAFE', 60, 30, 140, 60)], changed, target, req)
    assert 'no_text_change' in same['reasons']


def test_render_glyphs_draws_inside_quad():
    from PIL import Image, ImageDraw, ImageFont
    sign = Image.new('RGB', (240, 120), (230, 230, 230))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
    ImageDraw.Draw(sign).text((60, 44), 'OLD', fill=(30, 30, 30), font=font)
    img = np.asarray(sign)
    quad = np.array([[50, 40], [190, 40], [190, 82], [50, 82]], np.float32)
    out, evidence = te.render_glyphs(img, quad, 'SALE')
    assert out.shape == img.shape and evidence['lines'] == ['SALE']
    inside = out[40:82, 50:190].astype(int).mean(axis=2)
    assert .05 < (inside < 128).mean() < .6      # glyph strokes, not a filled block
    assert np.abs(out[40:82, 50:190].astype(int) - img[40:82, 50:190].astype(int)).max() > 100  # wording changed
    assert (out[:30] == img[:30]).all()          # untouched away from the quad


def _info(masks, cats):
    info = {}
    for oid, m in masks.items():
        ys, xs = np.nonzero(m)
        info[oid] = {'obj': {'category': cats[oid]}, 'area': int(m.sum()),
                     'bbox': (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                     'heads': {cats[oid].split(',')[0]}}
    return info


def test_support_graph_flags_objects_on_a_table():
    H, W = 80, 120
    table = np.zeros((H, W), bool); table[50:56, 20:100] = True; table[56:78, 22:26] = True; table[56:78, 94:98] = True
    cup = np.zeros((H, W), bool); cup[38:50, 50:62] = True
    floor = np.zeros((H, W), bool); floor[78:, :] = True
    chair = np.zeros((H, W), bool); chair[30:78, 104:116] = True  # stands on the floor beside the table
    masks = {'t': table, 'c': cup, 'f': floor, 'h': chair}
    info = _info(masks, {'t': 'table', 'c': 'cup', 'f': 'floor', 'h': 'chair'})
    deps = sg.dependents(info, masks, 't')
    assert [d['object_id'] for d in deps] == ['c'] and deps[0]['relation'] == 'rests_on'
    assert sg.dependents(info, masks, 'c') == []


def test_compare_erase_mode():
    shape = (100, 200)
    target = np.zeros(shape, bool); target[20:80, 20:180] = True
    changed = target.copy()
    src = [_tok('SCANIA', 40, 30, 160, 60)]
    assert te.compare(src, [], changed, target, {'mode': 'erase'})['decision'] == 'pass'
    ghost = te.compare(src, [_tok('NEWVA', 40, 30, 160, 60)], changed, target, {'mode': 'erase'})
    assert ghost['decision'] == 'fail' and 'unrequested_text' in ghost['reasons']
    left = te.compare(src, [_tok('SCANIA', 40, 30, 160, 60)], changed, target, {'mode': 'erase'})
    assert 'removed_text_remains' in left['reasons']


def test_compare_preserve_flags_new_glyphs():
    shape = (100, 200)
    target = np.zeros(shape, bool); target[20:80, 20:180] = True
    changed = target.copy()
    assert te.compare([], [], changed, target, {'mode': 'preserve'})['decision'] == 'pass'
    bad = te.compare([], [_tok('NEWVA', 40, 30, 120, 50)], changed, target, {'mode': 'preserve'})
    assert bad['decision'] == 'fail' and 'unrequested_text' in bad['reasons']
    # Existing lettering that survives a recolor is fine.
    kept = te.compare([_tok('EXIT', 40, 30, 120, 50)], [_tok('EXIT', 40, 30, 120, 50)], changed, target, {'mode': 'preserve'})
    assert kept['decision'] == 'pass'
