from copy import deepcopy
import json
import numpy as np
from PIL import Image
from sa1b_pipeline import scene_planning as scene
from sa1b_pipeline import minimal_planning as minimal
from sa1b_pipeline.aligned_planning_views import crop_triplet, aligned_sheet


def test_selected_colors_and_holes_are_preserved():
    pixels = np.zeros((80, 120, 3), dtype='uint8')
    pixels[:] = [231, 190, 89]
    pixels[30:50, 40:70] = [45, 80, 122]
    mask = np.zeros((80, 120), dtype=bool)
    mask[15:65, 20:95] = True
    mask[30:50, 40:70] = False
    original, outlined, binary = crop_triplet(Image.fromarray(pixels), mask)
    a, b, m = np.array(original), np.array(outlined), np.array(binary)[:, :, 0] > 0
    assert np.array_equal(a[m], b[m])
    assert (m == False).any() and m.any()
    assert not m[original.height // 2, original.width // 2]
    assert aligned_sheet(Image.fromarray(pixels), [mask, mask]).size == (1536, 872)


def test_scene_prompt_schema_and_blind_protocol():
    import jsonschema
    a = {'U1': 'remove', 'U2': 'remove'}
    valid = dict(members=[dict(unit_id='U1', ref='the lamp'), dict(unit_id='U2', ref='the chair')], instruction='Remove {ref}.')
    jsonschema.validate(valid, scene.schema('twins', a))
    invalid = {**valid, 'instruction': 'Remove {ref} and fill the background.'}
    try:
        jsonschema.validate(invalid, scene.schema('twins', a))
    except jsonschema.ValidationError:
        pass
    else:
        raise AssertionError('schema allowed an extra remove clause')
    assert scene.decode(valid, 'twins', a) == minimal.decode(valid, 'twins', a)
    for kind in ['twins', 'multi']:
        p = scene.prompt(kind, a)
        assert 'OUTSIDE contour' in p and 'unchanged' in p
        assert 'ALL lookalikes' in p and 'Assigned operations:' in p


def test_schema_constrains_operation_prefix_without_extra_output_fields():
    import jsonschema
    examples = {
        'add': 'Add a cushion on {ref}.',
        'replace': 'Replace {ref} with a folding stool.',
        'attribute': 'Change the color of {ref} to blue.',
        'remove': 'Remove {ref}.',
    }
    for op, instruction in examples.items():
        assignments = {'U1': op, 'U2': op}
        twins = dict(members=[dict(unit_id='U1', ref='the left chair'), dict(unit_id='U2', ref='the right chair')], instruction=instruction)
        multi = dict(targets=[dict(unit_id=m['unit_id'], ref=m['ref'], instruction=instruction.replace('{ref}',m['ref'])) for m in twins['members']])
        for kind, raw in [('twins', twins), ('multi', multi)]:
            jsonschema.validate(raw, scene.schema(kind, assignments))
            invalid = deepcopy(raw)
            row = invalid if kind == 'twins' else invalid['targets'][0]
            row['instruction'] = op + ': ' + row['instruction']
            try:
                jsonschema.validate(invalid, scene.schema(kind, assignments))
            except jsonschema.ValidationError:
                pass
            else:
                raise AssertionError('operation label leaked into public instruction')


def test_retry_keeps_scene_contract_and_original_good_target(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    first = {'targets': [dict(unit_id='U1', ref='the left chair', instruction='Remove the left chair.'), dict(unit_id='U2', reject='would remove anatomy')]}
    second = {'targets': [dict(unit_id='U2', ref='the right sleeve', instruction='Change the color of the right sleeve to blue.')]}
    class Planner:
        calls = 0
        def generate_multimage(self, jobs, schemas=None):
            self.calls += 1
            assert 'OUTSIDE contour' in jobs[0][1]
            return [deepcopy(first if self.calls == 1 else second)]
    planner = Planner()
    item = dict(gid='scene', objects=[dict(object_id='a'), dict(object_id='b')], views=[],
                selection={'operations': ['remove', 'remove']},
                prompt=scene.prompt('multi', {'U1': 'remove', 'U2': 'remove'}))
    opts = dict(kind='multi', root=tmp_path, retry=1, minimal=True, scene=True, accepted_counts={'add': 20, 'replace': 20, 'attribute': 0})
    result = plan_batch(planner, [deepcopy(item)], **opts)[0]
    assert result['planning_contract'] == scene.VERSION
    assert result['targets'][0]['_minimal_instruction'] == first['targets'][0]['instruction']
    assert result['targets'][1]['operation'] == 'attribute'
    assert plan_batch(planner, [deepcopy(item)], **opts)[0] == result and planner.calls == 2


def test_view_cache_and_masks_match(tmp_path):
    from sa1b_pipeline.group_planning import group_views
    source = tmp_path/'source.png';Image.new('RGB', (100, 100), 'orange').save(source)
    mask = tmp_path/'mask.png';a=np.zeros((100,100), dtype='uint8');a[20:50,30:60]=255;Image.fromarray(a).save(mask)
    obj = {'object_id':'1','mask_path':str(mask)}
    # Explicit decoded mask avoids dataset-specific aliases.
    import unittest.mock
    with unittest.mock.patch('sa1b_pipeline.group_planning.load_mask', return_value=a>0):
        legacy=group_views({'image_path':str(source)},[obj],tmp_path/'views')
        new=group_views({'image_path':str(source)},[obj],tmp_path/'views',aligned_outline=True)
        again=group_views({'image_path':str(source)},[obj],tmp_path/'views',aligned_outline=True)
    assert new==again and new[2]!=legacy[2]
    assert np.array_equal(np.array(Image.open(new[1][0])),a)
    assert Image.open(new[0][1]).size==(1536,436)
