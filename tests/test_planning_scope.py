from synthesis_pipeline.ground_planning_scope import apply_scope_result, scope_prompt


def fixture():
    row = dict(task_type='attribute', editing_instruction='Change the pink steps on the right to silver.',
               masked_content='Pink metal steps', edit_unit_status='complete_part',
               outside_dependencies='none', protected_objects=['railing'])
    result = dict(decision='revise', reason='Only the uppermost narrow metal riser is selected.',
                  refer_object='uppermost pink metal riser on the right',
                  editing_instruction='Change the uppermost pink metal riser on the right to silver.',
                  segmentation_target='metal riser', mask_refinement='surface')
    return row, result


def test_scope_revision_preserves_type_mask_and_original_instruction():
    row, result = fixture()
    row['mask'] = {'counts': 'opaque'}
    updated, status = apply_scope_result(row, result)
    assert status == 'accepted'
    assert updated['mask'] == row['mask']
    assert updated['task_type'] == row['task_type']
    assert updated['scope_preflight']['original_instruction'] == row['editing_instruction']
    assert updated['new_instruction'] == updated['editing_instruction']


def test_scope_reject_and_malformed_are_not_accepted():
    row, result = fixture()
    assert apply_scope_result(row, {**result, 'decision':'reject'})[0] is None
    assert apply_scope_result(row, {**result, 'decision':'accept'})[1] == 'accept_changed_instruction'
    assert apply_scope_result(row, {**result, 'mask_refinement':'complete'})[0] is None
    assert apply_scope_result(row, {**result, 'reason':''})[0] is None
    assert apply_scope_result(row, {**result, 'editing_instruction':'Remove the pink steps on the right.'})[0] is None


def test_scope_prompt_has_only_current_task_and_does_not_trust_proposal():
    row, _ = fixture()
    prompt = scope_prompt(row)
    assert 'No edited image exists yet' in prompt
    assert 'The proposal below may have misread the outline' in prompt
    assert 'narrow selected strip' in prompt
    assert 'proposed attachment point' not in prompt


def test_scope_observation_must_reach_the_public_instruction():
    row, result = fixture()
    inconsistent = {**result, 'editing_instruction': row['editing_instruction']}
    assert apply_scope_result(row, inconsistent)[1] == 'scope_word_missing_riser'


def test_pointer_adds_external_gutter_without_changing_selected_pixels():
    import numpy as np
    from PIL import Image
    from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
    source = Image.new('RGB', (100, 100), (27, 81, 123))
    mask = np.ones((100, 100), dtype=bool)
    ordinary = np.asarray(instruction_target_crop(source, mask, tile_size=128))
    pointed = np.asarray(instruction_target_crop(source, mask, tile_size=128, target_pointer=True))
    assert pointed.shape[1] == ordinary.shape[1] + 112
    selected = np.all(ordinary == (27, 81, 123), axis=-1)
    changed = np.any(pointed[:, 112:][selected] != ordinary[selected], axis=-1)
    assert 1 <= int(changed.sum()) <= 121


def test_pointer_labels_meaningful_enclosed_hole_without_recoloring_target():
    import numpy as np
    from PIL import Image
    from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
    source_array = np.full((100, 100, 3), (27, 81, 123), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=bool)
    mask[35:65, 40:60] = False
    source_array[~mask] = (190, 30, 40)
    source = Image.fromarray(source_array)
    ordinary = np.asarray(instruction_target_crop(source, mask, tile_size=128))
    pointed = np.asarray(instruction_target_crop(source, mask, tile_size=128, target_pointer=True))
    assert pointed.shape[1] == ordinary.shape[1] + 112 + 128
    selected = np.all(ordinary == (27, 81, 123), axis=-1)
    changed = np.any(
        pointed[:, 112:112+ordinary.shape[1]][selected] != ordinary[selected], axis=-1
    )
    assert 1 <= int(changed.sum()) <= 121


def test_pointer_marks_distant_thin_extension_without_mask_fill():
    import numpy as np
    from PIL import Image
    from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
    source = Image.new('RGB', (120, 120), (27, 81, 123))
    mask = np.zeros((120, 120), dtype=bool)
    mask[45:105, 35:85] = True
    mask[8:52, 78:83] = True
    ordinary = np.asarray(instruction_target_crop(source, mask, tile_size=160))
    pointed = np.asarray(
        instruction_target_crop(source, mask, tile_size=160, target_pointer=True)
    )
    selected = np.all(ordinary == (27, 81, 123), axis=-1)
    changed = np.any(pointed[:, 112:][selected] != ordinary[selected], axis=-1)
    # Main interior plus at least one distant extension landmark, while still
    # changing only the tiny dots rather than painting the mask.
    assert 121 < int(changed.sum()) <= 363
