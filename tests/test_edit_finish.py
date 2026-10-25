import json

import numpy as np
from PIL import Image

from sa1b_pipeline import edit_finish as ef
from sa1b_pipeline import evidence_gates as eg
from sa1b_pipeline import removal_effects as re_
from sa1b_pipeline.editor import _grow_span


def _scene(h=200, w=200, value=150.):
    rng = np.random.default_rng(0)
    return np.clip(value + rng.normal(0, 3, (h, w)), 0, 255).astype(np.float32)


def test_attached_shadow_accepted_and_structure_above_rejected():
    gray = _scene()
    target = np.zeros((200, 200), bool)
    target[60:100, 80:120] = True
    shadow = np.zeros_like(target)
    shadow[100:120, 70:140] = True          # touching, below, darker
    gray[shadow] = 60
    groove = np.zeros_like(target)
    groove[50:60, 70:130] = True            # touching, dark, but above the object
    gray[groove] = 40
    effect, records = re_.associate(target, [(shadow, .8), (groove, .8)], [], gray)
    reasons = [r['reason'] for r in records]
    assert reasons == ['accepted', 'above_target']
    assert effect[110, 100] and not effect[55, 100] and not (effect & target).any()


def test_scene_scale_and_bright_shadows_rejected():
    gray = _scene()
    target = np.zeros((200, 200), bool)
    target[90:110, 90:110] = True
    big = np.zeros_like(target)
    big[100:200, 0:200] = True              # building-sized shade around a small object
    gray[big & ~target] = 70
    lit = np.zeros_like(target)
    lit[110:118, 92:108] = True             # "shadow" proposal that is not darker
    effect, records = re_.associate(target, [(big, .9)], [], gray)
    assert records[0]['reason'] == 'scene_scale' and not effect.any()
    gray2 = _scene()
    effect, records = re_.associate(target, [(lit, .9)], [], gray2)
    assert records[0]['reason'] == 'not_darker' and not effect.any()


def test_reflection_keeps_only_mirrored_footprint():
    gray = _scene()
    target = np.zeros((200, 200), bool)
    target[40:80, 90:110] = True            # object standing on the waterline at row 80
    water = np.zeros_like(target)
    water[80:200, :] = True                 # one large "reflection" proposal (whole water body)
    effect, records = re_.associate(target, [], [(water, .8)], gray, water=water)
    assert records[0]['reason'] == 'accepted'
    assert effect[100, 100] and not effect[100, 20] and not effect[150, 100]
    # The same geometry on a bush or pavement (no water) is not a reflection.
    effect, records = re_.associate(target, [], [(water, .8)], gray, water=np.zeros_like(target))
    assert records[0]['reason'] == 'not_on_water' and not effect.any()


def test_regrain_matches_surrounding_grain():
    rng = np.random.default_rng(1)
    source = np.clip(120 + rng.normal(0, 4, (160, 160, 3)), 0, 255).astype(np.uint8)
    edited = source.copy()
    weight = np.zeros((160, 160), np.float32)
    weight[40:120, 40:120] = 1
    edited[40:120, 40:120] = 120            # perfectly smooth generated patch
    out, info = ef.regrain(source, edited, weight, seed=3)
    assert info['applied']
    region = weight > .9
    measured = ef.grain_sigma(out[..., 1].astype(np.float32), region)
    reference = ef.grain_sigma(source[..., 1].astype(np.float32), weight <= 0)
    assert abs(measured - reference) < .35 * reference
    assert np.array_equal(out[weight <= 0], source[weight <= 0])


def test_tone_match_removes_collar_offset():
    rng = np.random.default_rng(2)
    source = np.clip(100 + rng.normal(0, 2, (160, 160, 3)), 0, 255).astype(np.uint8)
    target = np.zeros((160, 160), bool)
    target[60:100, 60:100] = True
    support = np.zeros_like(target)
    support[44:116, 44:116] = True
    edited = source.astype(np.int16)
    edited[support] += 9                     # generator brightened the whole writable patch
    edited = np.clip(edited, 0, 255).astype(np.uint8)
    out, info = ef.tone_match(source, edited, target, support, support.astype(np.float32))
    assert info['applied']
    assert abs(float(out[support].astype(np.float32).mean() - source[support].astype(np.float32).mean())) < 2.5


def test_add_cleanup_reverts_far_stray_changes():
    source = np.full((200, 200, 3), 90, np.uint8)
    edited = source.copy()
    edited[90:120, 90:120] = 220                     # the added item
    edited[10:16, 10:30] = 200                       # re-rendered host text far away
    weight = np.ones((200, 200), np.float32)
    out, info = ef.add_cleanup(source, edited, weight)
    assert info['applied'] and info['stray_pixels'] > 0
    assert np.array_equal(out[10:16, 10:30], source[10:16, 10:30])
    assert (out[100:110, 100:110] == 220).all()


def test_reader_site_containment_for_add_anchor():
    anchor = [100, 100, 400, 900]
    assert eg.reader_verdict([[120, 120, 380, 300]], anchor) == ('wrong', eg.iou([120, 120, 380, 300], anchor))
    status, _ = eg.reader_verdict([[120, 120, 380, 300]], anchor, site_containment=.8)
    assert status == 'pass_site'
    status, _ = eg.reader_verdict([[120, 120, 380, 300], [120, 500, 380, 700]], anchor, site_containment=.8)
    assert status == 'ambiguous'
    status, _ = eg.reader_verdict([[600, 120, 800, 300]], anchor, site_containment=.8)
    assert status == 'wrong'


def test_lighting_gate_passes_only_lit_parts():
    gates = eg.EvidenceGates(lighting=True)
    assert gates.judge({'gate': 'lighting', 'operation': 'attribute'}, {'lighting': 'lit'})['pass']
    assert not gates.judge({'gate': 'lighting', 'operation': 'attribute'}, {'lighting': 'shade'})['pass']
    assert not gates.judge({'gate': 'lighting', 'operation': 'attribute'}, {'lighting': 'silhouette'})['pass']
    assert eg.color_edit({'operation': 'attribute', 'attributes': [{'property': 'color', 'value': 'red'}]})
    assert not eg.color_edit({'operation': 'attribute', 'attributes': [{'property': 'text', 'value': 'OPEN'}]})


def test_medium_gate_only_for_additions_and_passes_only_real(tmp_path):
    gates = eg.EvidenceGates(medium=True)
    assert gates.judge({'gate': 'medium', 'operation': 'add'}, {'medium': 'real'})['pass']
    assert not gates.judge({'gate': 'medium', 'operation': 'add'}, {'medium': 'depicted'})['pass']
    assert not gates.judge({'gate': 'medium', 'operation': 'add'}, {'medium': 'sculpted'})['pass']
    assert not gates.judge({'gate': 'medium', 'operation': 'add'}, {})['pass']
    assert gates.config()['medium'] is True and 'medium' not in eg.EvidenceGates().config()
    Image.fromarray(np.full((40, 40), 255, np.uint8)).save(tmp_path / 'mask_1.png')
    (tmp_path / 'views.json').write_text(json.dumps({'targets': [{'unit_id': 'U1', 'mask_path': str(tmp_path / 'mask_1.png')},
                                                                 {'unit_id': 'U2', 'mask_path': str(tmp_path / 'mask_1.png')}]}))
    item = {'per_mask_views': [[str(tmp_path / 'original.png'), str(tmp_path / 'U1_crop.png')]] * 2, 'objects': []}
    plan = {'targets': [{'unit_id': 'U1', 'operation': 'add', 'feasible': True, 'ref_phrase': 'the sign'},
                        {'unit_id': 'U2', 'operation': 'replace', 'feasible': True, 'ref_phrase': 'the logo', 'payload': 'a leaf'}]}
    asked = [(r['unit_id'], r['gate']) for r in eg.EvidenceGates(reader=False, identity=False, medium=True).requests(item, plan, 'multi')]
    assert ('U1', 'medium') in asked and ('U2', 'medium') not in asked


def test_site_box_centre_must_lie_on_the_anchor_pixels(tmp_path):
    # A thin diagonal anchor: its bounding box also contains unrelated pixels.
    m = np.zeros((1000, 1000), np.uint8)
    for i in range(100, 900):
        m[i, i - 5:i + 5] = 255
    Image.fromarray(m).save(tmp_path / 'stick.png')
    anchor = eg.mask_box_1000(str(tmp_path / 'stick.png'))
    on = eg.centre_on_mask(str(tmp_path / 'stick.png'))
    on_stick = [400, 400, 500, 500]          # centre (450, 450) lies on the diagonal
    off_stick = [600, 150, 700, 250]         # inside the bbox, centre far from the stick
    assert eg.reader_verdict([on_stick], anchor, site_containment=.8, on_target=on)[0] == 'pass_site'
    assert eg.reader_verdict([off_stick], anchor, site_containment=.8, on_target=on)[0] == 'wrong'
    assert eg.reader_verdict([off_stick], anchor, site_containment=.8)[0] == 'pass_site'   # bbox-only rule


def test_identity_part_passes_only_for_add_with_add_site():
    gates = eg.EvidenceGates(add_site=True)
    assert gates.judge({'gate': 'identity', 'operation': 'add'}, {'relation': 'part'})['pass']
    assert not gates.judge({'gate': 'identity', 'operation': 'remove'}, {'relation': 'part'})['pass']
    assert not eg.EvidenceGates().judge({'gate': 'identity', 'operation': 'add'}, {'relation': 'part'})['pass']


def test_add_twins_keep_most_similar_anchor_pair(tmp_path):
    sizes = {'U1': 40, 'U2': 10, 'U3': 38, 'U4': 12}
    targets = []
    for uid, side in sizes.items():
        m = np.zeros((100, 100), np.uint8)
        m[:side, :side] = 255
        Image.fromarray(m).save(tmp_path / f'{uid}.png')
        targets.append({'unit_id': uid, 'mask_path': str(tmp_path / f'{uid}.png')})
    (tmp_path / 'views.json').write_text(json.dumps({'targets': targets}))
    item = {'per_mask_views': [[str(tmp_path / 'U1_crop.png')]]}
    members = [{'unit_id': u, 'object_id': str(i), 'operation': 'add', 'feasible': True} for i, u in enumerate(sizes)]
    plan = {'members': members}
    eg.EvidenceGates(add_site=True)._keep_twins(item, plan)
    assert plan['feasible'] and [m['unit_id'] for m in plan['members']] == ['U1', 'U3']


def test_grow_span_contains_original_and_respects_frame():
    for a, b, limit in [(100, 150, 2250), (0, 30, 2250), (2200, 2250, 2250), (10, 40, 300), (5, 900, 1500)]:
        s, e = _grow_span(a, b, limit, 512)
        assert 0 <= s <= a and b <= e <= limit
        assert e - s == max(b - a, min(512, limit))


def test_worn_body_wear_is_not_removed_but_carried_items_are():
    from sa1b_pipeline import support_graph as sg
    person = np.zeros((100, 100), bool); person[20:95, 30:70] = True
    shoe = np.zeros((100, 100), bool); shoe[88:95, 35:50] = True          # inside the person mask
    pack = np.zeros((100, 100), bool); pack[40:60, 32:50] = True          # backpack, also inside
    lone = np.zeros((100, 100), bool); lone[5:10, 80:95] = True           # a shoe lying away from anyone
    info = {'p': {'heads': {'woman'}}, 's': {'heads': {'sneaker'}}, 'b': {'heads': {'backpack', 'bag'}},
            'l': {'heads': {'shoe'}}}
    masks = {'p': person, 's': shoe, 'b': pack, 'l': lone}
    assert sg.worn_on_body(info, masks, 's') == ['p']
    assert sg.worn_on_body(info, masks, 'b') == []
    assert sg.worn_on_body(info, masks, 'l') == []


def test_replace_retry_replans_only_appearance_only_targets(tmp_path, monkeypatch):
    from sa1b_pipeline import per_mask_planning as pm
    sent = []

    def fake_requests(planner, reqs):
        for r in reqs:
            sent.append(r)
            r['raw'] = {'ref': 'the flag', 'edit': {'replacement': 'a red lantern'}}

    def fake_combine(raws, kind, assignments, objects, *, options=None):
        return {'targets': [{'unit_id': u, 'payload': (raw.get('edit') or {}).get('replacement', ''), 'feasible': True}
                            for u, raw in zip(assignments, raws)], '_combined_from': raws}

    class Gates:
        calls = 0

        def apply(self, planner, batch, plans, kind):
            self.calls += 1
            for plan in plans:
                for t in plan['targets']:
                    t['evidence_gates'] = {'pass': True}

    monkeypatch.setattr(pm, '_requests', fake_requests)
    monkeypatch.setattr(pm, 'combine', fake_combine)
    item = {'_mask_assignments': {'U1': 'replace', 'U2': 'attribute'}, 'objects': [{}, {}]}
    requests = [{'item': item, 'unit_id': 'U1', 'path': tmp_path / 'U1.json', 'prompt': 'P1', 'raw': {'edit': {'replacement': 'a white flag'}}},
                {'item': item, 'unit_id': 'U2', 'path': tmp_path / 'U2.json', 'prompt': 'P2', 'raw': {'edit': {}}}]
    plan = {'targets': [{'unit_id': 'U1', 'payload': 'a white flag', 'feasible': False, 'reason': pm.APPEARANCE_ONLY},
                        {'unit_id': 'U2', 'feasible': False, 'reason': 'evidence_gate: reader=wrong'}],
            '_planning_calls': 2, '_planning_attempts': []}
    results = [plan]
    gates = Gates()
    pm._retry_appearance_only(type('P', (), {'batch_size': 4})(), [item], results, requests,
                              kind='multi', options={'replace_retry': True}, gates=gates)
    assert [r['unit_id'] for r in sent] == ['U1']                      # reader failure is never retried
    assert sent[0]['prompt'].startswith('P1') and 'a white flag' in sent[0]['prompt']
    assert sent[0]['path'].name == 'U1_replace_retry.json'
    new = results[0]
    assert new['_planning_calls'] == 3 and new['_replace_retry']['units'] == ['U1']
    assert new['_combined_from'][0]['edit']['replacement'] == 'a red lantern'
    assert new['_combined_from'][1] is requests[1]['raw']              # untouched answer kept
    assert gates.calls == 1                                              # recombined plan is gated again
    # Twins with enough passing members are never retried.
    sent.clear()
    twin = {'members': [{'unit_id': 'U1', 'reason': pm.APPEARANCE_ONLY}], 'feasible': True}
    pm._retry_appearance_only(type('P', (), {'batch_size': 4})(), [item], [twin], requests,
                              kind='twins', options={'replace_retry': True}, gates=gates)
    assert sent == []


def test_covering_on_object_and_scene_surface_rules():
    from sa1b_pipeline import support_graph as sg
    lantern = np.zeros((100, 100), bool); lantern[50:90, 40:60] = True
    snow_cap = np.zeros((100, 100), bool); snow_cap[42:50, 36:64] = True     # lies on the lantern top
    ground = np.zeros((100, 100), bool); ground[90:, :] = True
    snow_patch = np.zeros((100, 100), bool); snow_patch[84:90, 5:30] = True  # lies on the ground
    info = {'l': {'heads': {'lantern'}}, 'c': {'heads': {'snow'}}, 'g': {'heads': {'ground'}}, 'p': {'heads': {'snow'}}}
    masks = {'l': lantern, 'c': snow_cap, 'g': ground, 'p': snow_patch}
    assert sg.covered_objects(info, masks, 'c') == ['l']
    assert sg.covered_objects(info, masks, 'p') == []
    assert sg.covered_objects(info, masks, 'l') == []
    # The covered object is often not segmented: a cap that rests on nothing known is still excluded.
    floating_cap = {'c': {'heads': {'snow'}}, 'g': {'heads': {'ground'}}}
    assert sg.covered_objects(floating_cap, {'c': snow_cap, 'g': ground}, 'c') == ['unsegmented']
    assert sg.scene_surface({'heads': {'sky'}}) and sg.scene_surface({'heads': {'sidewalk', 'concrete'}})
    assert not sg.scene_surface({'heads': {'tree'}}) and not sg.scene_surface({'heads': {'ceiling'}})


def test_add_twins_pair_by_area_without_add_site(tmp_path):
    sizes = {'U1': 40, 'U2': 10, 'U3': 38}
    targets = []
    for uid, side in sizes.items():
        m = np.zeros((100, 100), np.uint8); m[:side, :side] = 255
        Image.fromarray(m).save(tmp_path / f'{uid}.png')
        targets.append({'unit_id': uid, 'mask_path': str(tmp_path / f'{uid}.png')})
    (tmp_path / 'views.json').write_text(json.dumps({'targets': targets}))
    item = {'per_mask_views': [[str(tmp_path / 'U1_crop.png')]]}
    plan = {'members': [{'unit_id': u, 'object_id': str(i), 'operation': 'add', 'feasible': True} for i, u in enumerate(sizes)]}
    eg.EvidenceGates()._keep_twins(item, plan)
    assert [m['unit_id'] for m in plan['members']] == ['U1', 'U3']


def test_blur_estimate_and_defocus_match():
    import cv2
    rng = np.random.default_rng(4)
    base = np.zeros((200, 200), np.float32)
    for x in range(0, 200, 20):
        base[:, x:x + 10] = 200                      # stripes: many step edges
    soft = cv2.GaussianBlur(base, (0, 0), 3.0)
    region = np.ones((200, 200), bool)
    est = ef.estimate_blur(soft + rng.normal(0, 1, soft.shape).astype(np.float32), region)
    assert est is not None and 2.2 < est < 3.8
    assert ef.estimate_blur(base, region) < 1.0
    # A crisp new square on a soft background gets blurred towards the background's focus.
    src = np.stack([soft] * 3, axis=2).astype(np.uint8)
    edited = src.copy()
    edited[80:120, 80:120] = 255
    edited[95:105, 80:120] = 0
    weight = np.zeros((200, 200), np.float32); weight[60:140, 60:140] = 1
    anchor = np.zeros((200, 200), bool); anchor[60:140, 60:140] = True
    out, info = ef.defocus_match(src, edited, weight, anchor)
    assert info['applied'] and info['reference_sigma'] > 2 and info['new_sigma'] < 1
    assert np.array_equal(out[weight == 0], edited[weight == 0])
    after = ef.estimate_blur(out.astype(np.float32) @ np.array([.299, .587, .114], np.float32), anchor)
    assert after > 1.5
    # In-focus photo: untouched.
    sharp_src = np.stack([base] * 3, axis=2).astype(np.uint8)
    sharp_edit = sharp_src.copy(); sharp_edit[80:120, 80:120] = 255; sharp_edit[95:105, 80:120] = 0
    out, info = ef.defocus_match(sharp_src, sharp_edit, weight, anchor)
    assert not info['applied'] and np.array_equal(out, sharp_edit)


def _stripes(blur=0.):
    import cv2
    base = np.zeros((240, 240), np.float32)
    for x in range(0, 240, 20):
        base[:, x:x + 10] = 200
    return cv2.GaussianBlur(base, (0, 0), blur) if blur else base


def _crisp_item(src):
    edited = src.copy()
    edited[90:130, 90:130] = 255
    edited[105:115, 90:130] = 0
    return edited


def test_focus_match_needs_two_soft_references():
    weight = np.zeros((240, 240), np.float32); weight[70:150, 70:150] = 1
    anchor = np.zeros((240, 240), bool); anchor[80:140, 80:140] = True
    soft = np.stack([_stripes(3.)] * 3, axis=2).astype(np.uint8)
    out, info = ef.focus_match(soft, _crisp_item(soft), weight, anchor, 'add')
    assert info['applied'] and info['reference_sigma'] > 2 and info['new_sigma'] < 1
    assert np.array_equal(out[weight == 0], _crisp_item(soft)[weight == 0])
    # Replacement of a sharp old object in a soft scene (a hat on an in-focus person): untouched.
    sharp = np.stack([_stripes()] * 3, axis=2).astype(np.uint8)
    old = soft.copy(); old[80:140, 80:140] = sharp[80:140, 80:140]
    _, info = ef.focus_match(old, _crisp_item(old), weight, anchor, 'replace')
    assert info['old_sigma'] < 1 and not info['applied']
    # Only the outer ring has (soft) edges; the anchor is a flat surface: one reference is not enough.
    flat = np.full((240, 240, 3), 120, np.uint8)
    outer = np.ones((240, 240), bool); outer[74:166, 74:166] = False
    flat[outer] = soft[outer]
    _, info = ef.focus_match(flat, _crisp_item(flat), weight, anchor, 'add')
    assert info['anchor_sigma'] is None and info['ring_sigma'] is not None
    assert not info['applied'] and info['reason'] == 'no_edges'


def test_focus_and_defocus_match_are_exclusive():
    import pytest
    from sa1b_pipeline.editor import DiffusersQwenImageEditor
    ed = DiffusersQwenImageEditor.__new__(DiffusersQwenImageEditor)
    ed.config = {}
    with pytest.raises(ValueError, match='enable one'):
        ed.configure_finish({'defocus_match': True, 'focus_match': True})
    ed.configure_finish({'focus_match': True, 'regrain': True})
    assert ed.config['finish']['focus_match'] is True


def test_finish_rules_material_sentence_is_separate_opt_in():
    from sa1b_pipeline.editor import DiffusersQwenImageEditor
    ed = DiffusersQwenImageEditor.__new__(DiffusersQwenImageEditor)
    ed.finish = {'prompt_rules': True}
    assert ed._finish_rules({'operation': 'add'}, False) == ''
    assert ed._finish_rules({'operation': 'remove'}, True) == ef.REMOVE_RULE + ef.EFFECT_RULE
    assert ed._finish_rules({'operation': 'remove'}, False) == ef.REMOVE_RULE
    ed.finish = {'prompt_rules': True, 'material_rule': True}
    assert ed._finish_rules({'operation': 'replace'}, False) == ef.MATERIAL_RULE
    ed.finish = {}
    assert ed._finish_rules({'operation': 'remove'}, True) == ''


def test_dial_items_allow_numerals_but_not_words():
    from sa1b_pipeline import text_edits as te
    spec = {'operation': 'add', 'payload': 'a small round wall clock'}
    req = te.requirement(spec)
    assert req == {'mode': 'numerals'}
    assert 'numerals' in te.generator_rule(req, 'add')
    assert te.requirement({'operation': 'add', 'payload': 'a small red bow'}) == {'mode': 'none'}
    assert te.requirement({'operation': 'add', 'payload': 'a sign reading "OPEN"'})['mode'] == 'exact'
    assert te.dial_item('two silver wristwatches') and not te.dial_item('a watchtower')
    shape = (100, 100)
    changed = np.zeros(shape, bool); changed[20:80, 20:80] = True
    target = changed.copy()
    digits = [{'text': '12', 'conf': .99, 'box': [[30, 30], [40, 30], [40, 40], [30, 40]]}]
    words = [{'text': 'ROLEX', 'conf': .99, 'box': [[30, 50], [60, 50], [60, 60], [30, 60]]}]
    assert te.compare([], digits, changed, target, req)['decision'] == 'pass'
    assert te.compare([], digits + words, changed, target, req)['decision'] == 'fail'
    assert te.compare([], digits, changed, target, {'mode': 'none'})['decision'] == 'fail'


def test_detail_restore_brings_back_resampled_detail_but_keeps_new_content():
    from sa1b_pipeline import edit_finish as ef
    rng = np.random.default_rng(0)
    source = (128 + rng.integers(-40, 41, (120, 160, 1)) * np.ones((1, 1, 3))).astype(np.uint8)  # fine texture
    import cv2
    edited = cv2.GaussianBlur(source, (0, 0), 1.5)          # the generator resampled the whole support...
    edited[50:70, 70:90] = (200, 30, 30)                     # ...and drew a new red item
    alpha = np.ones((120, 160), np.float32)
    out, info = ef.detail_restore(source, edited, alpha)
    assert info['applied']
    far = (slice(0, 30), slice(0, 40))
    assert np.array_equal(out[far], source[far])           # texture away from the item is the source again
    assert np.array_equal(out[55:65, 75:85], edited[55:65, 75:85])   # the new item is untouched
    # A protected region keeps the edit even where it is only blurred (pixels outside the writable
    # support are discarded by the paste and not checked).
    protect = np.zeros((120, 160), bool); protect[:20, 140:] = True
    out2, _ = ef.detail_restore(source, edited, alpha, protect=protect)
    assert np.array_equal(out2[:20, 140:], edited[:20, 140:])
    assert np.array_equal(out2[far], source[far])
    unchanged, info = ef.detail_restore(source, source.copy(), alpha)
    assert not info['applied'] and np.array_equal(unchanged, source)


def test_reader_retry_replans_reader_failures_with_the_readers_finding(tmp_path, monkeypatch):
    from sa1b_pipeline import per_mask_planning as pm
    sent = []

    def fake_requests(planner, reqs):
        for r in reqs:
            sent.append(r)
            r['raw'] = {'ref': 'the second tree from the left', 'edit': {'property': 'color', 'value': 'red'}}

    monkeypatch.setattr(pm, '_requests', fake_requests)
    monkeypatch.setattr(pm, 'combine', lambda raws, kind, a, o, *, options=None: {
        'targets': [{'unit_id': u, 'feasible': True} for u in a], '_combined_from': raws})

    class Gates:
        calls = 0

        def apply(self, planner, batch, plans, kind):
            self.calls += 1

    item = {'_mask_assignments': {'U1': 'attribute', 'U2': 'attribute', 'U3': 'replace'}, 'objects': [{}, {}, {}]}
    requests = [{'item': item, 'unit_id': u, 'path': tmp_path / f'{u}.json', 'prompt': f'P{u}', 'raw': {}}
                for u in ('U1', 'U2', 'U3')]
    reader = {'gate': 'reader', 'status': 'wrong', 'boxes': [[100, 200, 300, 400]], 'pass': False}
    plan = {'targets': [{'unit_id': 'U1', 'feasible': False, 'reason': 'evidence_gate: reader=wrong',
                         'ref_phrase': 'the tree on the left', 'evidence_gates': {'verdicts': [reader]}},
                        {'unit_id': 'U2', 'feasible': False, 'reason': 'evidence_gate: reader=wrong, identity=part'},
                        {'unit_id': 'U3', 'feasible': False, 'reason': pm.APPEARANCE_ONLY, 'payload': 'a blue flag'}],
            '_planning_calls': 3, '_planning_attempts': []}
    results = [plan]
    pm._retry_gate_failures(type('P', (), {'batch_size': 8})(), [item], results, requests, kind='multi',
                            options={'reader_retry': True, 'replace_retry': True}, gates=Gates())
    assert sorted(r['unit_id'] for r in sent) == ['U1', 'U3']            # one shared round; U2 also failed identity
    first = next(r for r in sent if r['unit_id'] == 'U1')
    assert first['path'].name == 'U1_reader_retry.json' and 'the tree on the left' in first['prompt']
    assert '[10, 20, 30, 40]' in first['prompt']                         # reader box in percent
    new = results[0]
    assert new['_reader_retry']['units'] == ['U1'] and new['_replace_retry']['units'] == ['U3']
    assert {a['attempt'] for a in new['_planning_attempts']} == {'reader_retry', 'replace_retry'}
    none_note = pm.reader_note({'ref_phrase': 'the lamp', 'evidence_gates': {'verdicts': [{'gate': 'reader', 'boxes': []}]}})
    assert 'could not find any matching region' in none_note


def test_viewpoint_gate_drops_only_aerial_additions():
    from sa1b_pipeline.evidence_gates import EvidenceGates
    gates = EvidenceGates(viewpoint=True)
    assert gates.config()['viewpoint'] is True and 'viewpoint' not in EvidenceGates().config()
    judge = lambda view: gates.judge({'gate': 'viewpoint', 'operation': 'add'}, {'view': view})['pass']
    assert not judge('aerial') and judge('elevated') and judge('ground') and not judge(None)
