from types import SimpleNamespace

import numpy as np
from PIL import Image

from sa1b_pipeline.editor import (DiffusersQwenImageEditor, _target_guide, _target_hint,
                                  _support_radius, edit_prompt, region_support_mask)
from sa1b_pipeline.geometry import mask_geometry, operation_quality_gate
from sa1b_pipeline.validate import _target_panel, deterministic_check


def test_operation_quality_gate_is_operation_specific():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:70, 30:80] = True
    feature = mask_geometry(mask)
    assert operation_quality_gate(feature, "attribute")["eligible"]
    assert operation_quality_gate(feature, "remove")["eligible"]

    tiny = np.zeros((100, 100), dtype=bool)
    tiny[10:12, 10:12] = True
    tiny_feature = mask_geometry(tiny)
    assert not operation_quality_gate(tiny_feature, "remove")["eligible"]
    assert "mask_too_small" in operation_quality_gate(tiny_feature, "remove")["reasons"]


def test_target_guide_is_aligned_and_prompt_contains_coordinates():
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:12, 8:17] = True
    guide = _target_guide(Image.new("RGB", (30, 20), "gray"), mask)
    assert guide.size == (30, 20)
    assert _target_hint(mask) == "x=27%..57%, y=25%..60%"


def test_operation_support_and_add_prompt_use_anchor_contract():
    mask = np.zeros((100, 120), dtype=bool)
    mask[30:60, 40:70] = True
    add_support = region_support_mask(mask, "add", "anchor_plus_attachment")
    attr_support = region_support_mask(mask, "attribute", "target_only")
    assert add_support.sum() > mask.sum()
    np.testing.assert_array_equal(attr_support, mask)
    assert _support_radius(mask, "remove", "target_plus_collar") > _support_radius(mask, "attribute", "target_only")
    prompt = edit_prompt({"operation": "add", "instruction_ref": "Add a tag to the selected bag."})
    assert "host object" in prompt
    assert "not the new item's final segmentation" in prompt
    assert "Replace only the selected target" in edit_prompt(
        {"operation": "replace", "instruction_ref": "Replace the selected bag with a box."})
    assert "Change only the requested visible" in edit_prompt(
        {"operation": "attribute", "instruction_ref": "Change the color of the selected bag to blue."})


def test_editor_uses_aligned_guide_and_regional_weights(tmp_path):
    source = Image.new("RGB", (80, 60), "gray")
    source_path = tmp_path / "source.png"
    source.save(source_path)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)

    captured = {}

    class FakeGenerator:
        def manual_seed(self, seed):
            captured["seed"] = seed
            return self

    class FakeTorch:
        @staticmethod
        def Generator(**kwargs):
            return FakeGenerator()

    class FakePipe:
        def region_callback_kwargs(self, crop, local_mask, operation, protected, policy, editable_mask=None):
            captured["regional_mask"] = local_mask.copy()
            captured["editable_mask"] = editable_mask.copy()
            captured["policy"] = policy
            return {"height": 64, "width": 96, "regional_weights": np.ones((4, 6)),
                    "output_resolution": 1024, "use_kv_cache": True}, None

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (96, 64), "blue")])

    editor = object.__new__(DiffusersQwenImageEditor)
    editor.torch = FakeTorch()
    editor.pipe = FakePipe()
    editor.steps = 4
    unit = {
        "operation": "attribute",
        "instruction_ref": "Change the color of the selected square to blue.",
        "mask_path": str(mask_path),
    }
    output = editor.generate(str(source_path), unit, seed=17)
    assert output.size == source.size
    assert captured["image"].size == (captured["regional_mask"].shape[1], captured["regional_mask"].shape[0])
    assert captured["regional_weights"].shape == (4, 6)
    assert captured["editable_mask"].shape == captured["regional_mask"].shape
    assert captured["editable_mask"].sum() >= captured["regional_mask"].sum()
    assert captured["policy"] == "guard-any-v1"
    assert "x=" in captured["prompt"] and "y=" in captured["prompt"]


def test_validation_vetoes_low_change_attribute_and_builds_target_panel(tmp_path):
    source = Image.new("RGB", (80, 60), "gray")
    edited = source.copy()
    source_path, edited_path = tmp_path / "source.png", tmp_path / "edited.png"
    source.save(source_path)
    edited.save(edited_path)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    mask_path = tmp_path / "mask.png"
    Image.fromarray(mask).save(mask_path)
    row = {"source_image": str(source_path), "edited_image": str(edited_path),
           "mask": str(mask_path), "operation": "attribute", "instruction": "Make it blue."}
    check = deterministic_check(row)
    assert check["decision"] == "fail"
    assert "no_visible_target_change" in check["failure_reasons"]
    panel = Image.open(_target_panel(row, tmp_path / "audit"))
    assert panel.size == (1536, 512)


def test_validation_reports_support_leakage(tmp_path):
    source = Image.new("RGB", (80, 60), "gray")
    edited = source.copy()
    pixels = edited.load()
    # Valid target change.
    for y in range(20, 40):
        for x in range(30, 50):
            pixels[x, y] = (40, 120, 220)
    # Large unrelated change outside the explicit support region.
    for y in range(0, 20):
        for x in range(0, 30):
            pixels[x, y] = (220, 40, 40)
    source_path, edited_path = tmp_path / "source.png", tmp_path / "edited.png"
    source.save(source_path)
    edited.save(edited_path)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[20:40, 30:50] = 255
    mask_path, support_path = tmp_path / "mask.png", tmp_path / "support.png"
    Image.fromarray(mask).save(mask_path)
    Image.fromarray(mask).save(support_path)
    row = {"source_image": str(source_path), "edited_image": str(edited_path),
           "mask": str(mask_path), "support_mask": str(support_path),
           "operation": "attribute", "instruction": "Make it blue."}
    check = deterministic_check(row)
    assert check["decision"] == "fail"
    assert "support_leakage" in check["failure_reasons"]
    assert check["deterministic"]["outside_support_changed_fraction"] > 0.02


def test_target_panel_cache_distinguishes_edit_backends_and_overwrites(tmp_path):
    src=tmp_path/'source.png';Image.new('RGB',(80,60),'gray').save(src)
    mask=tmp_path/'mask.png';Image.new('L',(80,60),255).save(mask)
    edited=tmp_path/'edit.png';Image.new('RGB',(80,60),'red').save(edited)
    row={'source_image':str(src),'edited_image':str(edited),'mask':str(mask),'operation':'attribute'}
    red=_target_panel(row,tmp_path/'panels')
    Image.new('RGB',(80,60),'blue').save(edited)
    blue=_target_panel(row,tmp_path/'panels')
    assert red!=blue
    assert Image.open(red).getpixel((768,256))[0]>200
    assert Image.open(blue).getpixel((768,256))[2]>200


def test_planner_can_skip_unnatural_edits_without_relaxing_target_minimum(tmp_path):
    from sa1b_pipeline.io import candidate_from_dict, read_jsonl
    from sa1b_pipeline.select import select_and_plan

    source = tmp_path / 'source.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    objects = []
    for index, x in enumerate((10, 60)):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, x:x + 25] = 255
        path = tmp_path / f'mask{index}.png'
        Image.fromarray(mask).save(path)
        objects.append({'object_id': str(index), 'category': 'cup', 'mask_path': str(path)})
    candidate = candidate_from_dict({'source_id': 'scene', 'image_path': str(source), 'objects': objects})
    plan = {'units': [{'object_id': '0', 'operation': 'attribute',
                       'ref_phrase': 'the left cup', 'payload': 'blue'}],
            'skipped_objects': [{'object_id': '1', 'reason': 'No natural local edit fits this target.'}]}
    audit = tmp_path / 'audit.jsonl'
    cases = select_and_plan(
        [candidate], None, audit_path=audit,
        selection_decisions={'scene': {'accept': True, 'selected_object_ids': ['0', '1']}},
        planning_decisions={'scene': plan},
    )
    assert cases == []
    rows = read_jsonl(audit)
    assert any(row.get('error') == 'no_natural_local_edit' and row['object_id'] == '1' for row in rows)
    assert any(row.get('error') == 'fewer_than_min_quality_units' for row in rows)


def test_feather_is_strongest_inside_and_zero_outside():
    from sa1b_pipeline.editor import _feather
    mask = np.zeros((11, 11), dtype=bool)
    mask[1:10, 1:10] = True
    alpha = _feather(mask, 2)
    assert alpha[5, 5] == 1
    assert 0 < alpha[1, 5] < alpha[2, 5] < alpha[3, 5]
    assert np.all(alpha[~mask] == 0)


def test_feather_does_not_restore_deleted_object_at_image_edge():
    from sa1b_pipeline.editor import _feather
    mask = np.zeros((12, 15), dtype=bool)
    mask[:9, 3:12] = True
    alpha = _feather(mask, 2)
    assert alpha[0, 7] == 1  # No surviving source colour at the clipped edge.
    assert alpha[0, 3] == alpha[4, 3]  # Real side boundary still feathers.
    assert np.all(alpha[~mask] == 0)
    assert np.all(_feather(np.ones((5, 7), dtype=bool), 2) == 1)


def test_specialist_rejects_host_as_implicit_add_placement(tmp_path):
    import pytest
    from sa1b_pipeline.specialist_editors import writable_mask
    mask = np.zeros((30, 30), dtype=np.uint8); mask[5:25, 5:25] = 255
    path = tmp_path/'host.png'; Image.fromarray(mask).save(path)
    unit = {'operation':'add','mask_path':str(path)}
    with pytest.raises(ValueError, match='explicit execution_mask'):
        writable_mask(unit, (30,30))
    placement = np.zeros_like(mask); placement[:3,:3] = 255
    p = tmp_path/'placement.png'; Image.fromarray(placement).save(p)
    unit['execution_mask_path']=str(p)
    with pytest.raises(ValueError, match='contact'):
        writable_mask(unit, (30,30))
    placement[4:8,4:8]=255; Image.fromarray(placement).save(p)
    support=writable_mask(unit,(30,30))
    assert support[5,5] and not support[20,20]


def test_recolor_preserves_shading_and_unselected_pixels(tmp_path):
    from sa1b_pipeline.specialist_editors import RecolorEditor, to_linear
    rgb=np.full((32,32,3),100,dtype=np.uint8)
    for y in range(8,24): rgb[y,8:24]=80+(y-8)*8
    source=tmp_path/'source.png';Image.fromarray(rgb).save(source)
    mask=np.zeros((32,32),dtype=np.uint8);mask[8:24,8:24]=255
    path=tmp_path/'mask.png';Image.fromarray(mask).save(path)
    unit={'operation':'attribute','attribute_kind':'color','surface_type':'diffuse_uniform',
          'mask_path':str(path),'payload':'navy blue','instruction_ref':'Change the color of the selected fabric to navy blue.'}
    output=np.asarray(RecolorEditor().generate(str(source),unit))
    np.testing.assert_array_equal(output[mask==0],rgb[mask==0])
    assert np.all(np.diff(output[11:21,16,2].astype(float))>0)
    assert output[16,16,2]>output[16,16,0]

    # Protocol 1.0 stores attribute values in the spec and leaves payload empty.
    # This must produce the same edit as the legacy representation.
    unit['instruction_spec'] = {'operation': 'attribute', 'attributes': [
        {'property': 'color', 'value': 'navy blue'}]}
    unit['payload'] = ''
    np.testing.assert_array_equal(np.asarray(RecolorEditor().generate(str(source), unit)), output)


def test_recolor_does_not_silently_drop_other_requested_attributes():
    import pytest
    from sa1b_pipeline.specialist_editors import target_color
    unit = {'target_rgb': [10, 20, 30], 'instruction_spec': {
        'operation': 'attribute', 'attributes': [
            {'property': 'color', 'value': 'blue'},
            {'property': 'texture', 'value': 'woven'}]}}
    with pytest.raises(ValueError, match='exactly one color attribute'):
        target_color(unit)


def test_black_surface_recolor_retains_visible_lighting(tmp_path):
    from sa1b_pipeline.specialist_editors import RecolorEditor, target_color
    rgb = np.repeat(np.linspace(100, 230, 32).astype('uint8')[:, None, None], 32, axis=1)
    rgb = np.repeat(rgb, 3, axis=2)
    source, mask = tmp_path/'source.png', tmp_path/'mask.png'
    Image.fromarray(rgb).save(source)
    Image.new('L', (32, 32), 255).save(mask)
    unit = dict(operation='attribute', surface_type='diffuse_uniform', mask_path=str(mask), payload='black')
    result = np.asarray(RecolorEditor().generate(str(source), unit))
    assert 0 < int(result[0, 16, 0]) < int(result[-1, 16, 0]) < 70
    np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
    # An explicitly supplied numerical RGB value remains authoritative.
    np.testing.assert_array_equal(target_color(dict(unit, target_rgb=[0, 0, 0])), [0, 0, 0])


def test_generation_checkpoint_rejects_changed_instruction(tmp_path):
    import pytest
    from sa1b_pipeline.editor import generate_cases
    from sa1b_pipeline.io import write_jsonl
    from sa1b_pipeline.specialist_editors import RecolorEditor
    source=tmp_path/'source.png';Image.new('RGB',(30,30),'gray').save(source)
    path=tmp_path/'mask.png';Image.new('L',(30,30),255).save(path)
    unit={'unit_id':'U1','operation':'attribute','attribute_kind':'color','surface_type':'diffuse_uniform',
          'mask_path':str(path),'payload':'blue','instruction_ref':'Change the color of the fabric to blue.'}
    case={'id':'x','source':{'image_path':str(source)},'units':[unit]}
    manifest=tmp_path/'input.jsonl';write_jsonl(manifest,[case])
    first=generate_cases(manifest,tmp_path/'out',RecolorEditor())
    second=generate_cases(manifest,tmp_path/'out',RecolorEditor())
    assert first[0]['generation_fingerprint']==second[0]['generation_fingerprint']
    assert first[0]['editor_config']['backend']=='recolor-luminance'
    unit['payload']='green';write_jsonl(manifest,[case])
    with pytest.raises(ValueError,match='different or untracked'):
        generate_cases(manifest,tmp_path/'out',RecolorEditor())


def test_planning_view_keeps_appearance_pixels_clean(tmp_path):
    from sa1b_pipeline.select import planning_view
    from sa1b_pipeline.schema import Candidate,CandidateObject
    source=tmp_path/'source.png'; Image.new('RGB',(40,30),'white').save(source)
    path=tmp_path/'mask.png';Image.new('L',(40,30),255).save(path)
    candidate=Candidate('x',str(source),[CandidateObject('1','car',mask_path=str(path))])
    panel=Image.open(planning_view(candidate,['1']))
    assert panel.size==(80,30)
    np.testing.assert_array_equal(np.asarray(panel)[:,:40],np.asarray(Image.open(source)))


def test_attribute_rejects_person_and_finish_as_material():
    import pytest
    from sa1b_pipeline.instructions import build_unit
    geo={'bbox':[0,0,10,10],'point':[5,5]}
    with pytest.raises(ValueError,match='whole-person'):
        build_unit({'operation':'attribute','payload':'blue'}, {'object_id':'1','category':'person'}, 'mask.png',geo,[])
    with pytest.raises(ValueError,match='color/finish'):
        build_unit({'operation':'attribute','attribute_kind':'material','payload':'matte black'},
                   {'object_id':'1','category':'pedestal'}, 'mask.png',geo,[])
    with pytest.raises(ValueError,match='separately selected'):
        build_unit({'operation':'attribute','payload':'blue','edit_surface':'window frame'},
                   {'object_id':'1','category':'window'}, 'mask.png',geo,[])


def test_planning_prompt_drops_overlay_appearance_metadata():
    from sa1b_pipeline.mllm import planning_prompt
    prompt = planning_prompt({'objects':[{'object_id':'1','category':'car',
                              'description':'imaginary pink car'}]}, ['1'],
                             {'scene_summary':'surreal pink road','evidence':'purple trees',
                              'target_features':[{'object_id':'1','area_px':2000}]})
    assert 'imaginary pink car' not in prompt
    assert 'surreal pink road' not in prompt and 'purple trees' not in prompt
    assert '2000' in prompt


def test_intrinsic_recolor_preserves_source_detail_and_outside_pixels(tmp_path):
    from sa1b_pipeline.specialist_editors import IntrinsicRecolorEditor
    rgb=np.full((32,32,3),150,dtype=np.uint8)
    rgb[15,15]=110
    source=tmp_path/'source.png';Image.fromarray(rgb).save(source)
    mask=np.zeros((32,32),dtype=np.uint8);mask[8:24,8:24]=255
    path=tmp_path/'mask.png';Image.fromarray(mask).save(path)
    editor=object.__new__(IntrinsicRecolorEditor)
    editor.models={};editor.device='cpu';editor.max_side=1024
    def predict(models,image,**kwargs):
        return {'hr_alb':np.full_like(image,.4), 'dif_shd':np.ones_like(image),
                'residual':np.zeros_like(image)}
    editor.run_pipeline=predict
    unit={'operation':'attribute','attribute_kind':'color','surface_type':'diffuse_uniform',
          'mask_path':str(path),'payload':'navy blue'}
    out=np.asarray(editor.generate(str(source),unit))
    np.testing.assert_array_equal(out[mask==0],rgb[mask==0])
    assert out[15,15,2]<out[14,15,2]
    assert out[14,15,2]>out[14,15,0]


def test_flux_fill_adapter_passes_white_edit_mask_and_composites_outside(tmp_path):
    from sa1b_pipeline.specialist_editors import FluxFillEditor
    import torch
    source=tmp_path/'source.png';Image.new('RGB',(96,80),'gray').save(source)
    target=np.zeros((80,96),dtype=np.uint8);target[20:40,30:50]=255
    path=tmp_path/'mask.png';Image.fromarray(target).save(path)
    captured={}
    class Pipe:
        def __call__(self,**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new('RGB',(kwargs['width'],kwargs['height']),'red')])
    editor=object.__new__(FluxFillEditor)
    editor.torch=torch;editor.device='cpu';editor.pipe=Pipe()
    editor.steps=2;editor.guidance=30.;editor.max_side=1024
    unit={'operation':'replace','mask_path':str(path),'payload':'a red cup',
          'instruction_ref':'Replace the cup with a red cup.'}
    out=np.asarray(editor.generate(str(source),unit))
    support=editor.support_mask(Image.open(source),unit)
    assert np.all(out[~support]==128)
    assert out[30,40,0]==255 and out[30,40,1]==0
    supplied=np.asarray(captured['mask_image'])
    assert supplied.max()==255 and supplied.min()==0
    assert captured['mask_image'].size==captured['image'].size
    assert captured['width']%32==0 and captured['height']%32==0


def test_crop_instruction_keeps_local_placement_but_drops_full_scene_count():
    unit = {'operation': 'add',
            'instruction_ref': 'Add a small tattoo on the center of the forearm below the third sleeve in the full scene.',
            'instruction_noref': 'Add a small tattoo on the center of the selected forearm.'}
    before = dict(unit)
    prompt = edit_prompt(unit, crop_instruction=True, target_hint='x=20%..80%, y=20%..80%')
    assert prompt.startswith(unit['instruction_noref'])
    assert 'third sleeve' not in prompt
    assert 'on the center' in prompt
    assert 'x=20%' in prompt
    assert unit == before
    assert edit_prompt(unit).startswith(unit['instruction_ref'])


def test_binary_audit_panel_preserves_shape_and_separates_locator(tmp_path):
    src=tmp_path/'wide.png'; edited=tmp_path/'result.png'; mask=tmp_path/'mask.png'
    a=np.full((60,240,3),(150,50,30),dtype=np.uint8)
    a[20:40,110:130]=(10,220,30)
    Image.fromarray(a).save(src)
    b=a.copy();b[20:40,110:130]=(20,30,240);Image.fromarray(b).save(edited)
    m=np.zeros((60,240),dtype=np.uint8);m[20:40,110:130]=255;Image.fromarray(m).save(mask)
    row={'source_image':str(src),'edited_image':str(edited),'mask':str(mask),'operation':'attribute'}
    old=_target_panel(row,tmp_path/'panels')
    new=_target_panel(row,tmp_path/'panels',binary_locator=True)
    assert new!=old
    p=np.asarray(Image.open(new))
    # The actual square target remains square despite a nonsquare source crop.
    target=(p[48:,:512,1]>160)&(p[48:,:512,0]<60)
    yy,xx=np.where(target)
    assert abs((xx.max()-xx.min())-(yy.max()-yy.min()))<=2
    assert p[280,768,2]>180  # clean result centre is blue
    locator=p[48:,1024:].astype(int)
    assert np.max(np.abs(locator[:,:,0]-locator[:,:,1]))<=2
    assert np.max(np.abs(locator[:,:,1]-locator[:,:,2]))<=2


def test_binary_audit_prompt_matches_locator_roles():
    from sa1b_pipeline.validate import AUDIT_PROMPT, audit_prompt
    instruction='Tint the window gray.'
    assert audit_prompt(instruction)==AUDIT_PROMPT.replace('{instruction}',instruction)
    prompt=audit_prompt(instruction,binary_locator=True)
    assert 'MASK ONLY' in prompt and 'White in MASK ONLY identifies' in prompt
    assert 'cyan' not in prompt and instruction in prompt
