import pytest
from sa1b_pipeline.instruction_diversity import content_preferences, preference_prompt
from sa1b_pipeline.group_planning import bind_multi_target_ids


def test_scene_content_does_not_choose_payloads_from_ids_or_category_guesses():
    a = [{'object_id': '5', 'bbox': [0, 0, 20, 30]}]
    b = [{'object_id': '999', 'bbox': [40, 0, 70, 50], 'category': 'untrusted'}]
    assert content_preferences('add', a) == content_preferences('add', b)
    hint = preference_prompt(['add', 'add'], a)
    assert hint.count(content_preferences('add', a)) == 1
    assert 'SCENE-GROUNDED CONTENT CHOICE' in hint
    assert 'first idea failed' in hint
    assert content_preferences('remove', a)
    assert not any(word in hint.lower() for word in ('sticker', 'badge', 'decal', 'food garnish', 'e.g.'))


def test_ids_bound_only_by_explicit_unique_uid_and_matching_operation():
    objects = [{'object_id': '12'}, {'object_id': '30'}]
    assignments = {'12': 'add', '30': 'attribute'}
    plan = {'targets': [{'unit_id': 'U2', 'object_id': 'ID', 'operation': 'attribute'},
                        {'unit_id': 'U1', 'object_id': '', 'operation': 'add'}]}
    bound = bind_multi_target_ids(plan, objects, assignments)
    assert [x['object_id'] for x in bound['targets']] == ['30', '12']
    assert plan['targets'][0]['object_id'] == 'ID'
    for field, value in [('object_id','12'),('unit_id','U1'),('unit_id','U3'),('operation','remove')]:
        invalid = {'targets': [dict(plan['targets'][0]), dict(plan['targets'][1])]}
        invalid['targets'][0][field] = value
        with pytest.raises(ValueError):
            bind_multi_target_ids(invalid, objects, assignments)


def test_decoder_does_not_ask_model_to_predict_bookkeeping_ids():
    from sa1b_pipeline.multi_plan_schema import multi_plan_schema
    from sa1b_pipeline.group_planning import multi_prompt
    fields = multi_plan_schema()['properties']['targets']['items']['properties']
    assert fields['object_id'] == {'const': ''}
    prompt = multi_prompt({'999': 'add'}, [{'object_id': '999'}], ['U1'])
    assert 'unit_id=U1, operation=add' in prompt
    assert 'object_id=999' not in prompt
    assert 'Set every object_id to the empty string' in prompt


def test_current_twin_and_compact_multi_paths_receive_scene_rules_without_examples():
    from sa1b_pipeline.group_planning import multi_prompt, twin_prompt
    from sa1b_pipeline.multi_compact_target import group_prompt
    objects = [{'object_id': '4', 'bbox': [0, 0, 10, 10]}]
    prompts = [twin_prompt('add', objects, style='concise-27b-v8'),
               group_prompt({'4': 'add'}, objects, ['U1'])]
    for prompt in prompts:
        assert prompt.count('SCENE-GROUNDED CONTENT CHOICE') == 1
        assert 'the selected region' in prompt  # blind instruction protocol survives
        assert not any(word in prompt.lower() for word in
                       ('sticker', 'badge', 'decal', 'e.g.', 'for example', 'ranked content search'))
    assert 'SCENE-GROUNDED CONTENT CHOICE' in multi_prompt(
        {'4': 'add'}, objects, ['U1'], content_policy='scene')
    from sa1b_pipeline.twin_planning import output_schema
    from sa1b_pipeline.multi_compact_target import group_schema
    twin_fields = output_schema()['properties']['members']['items']['required']
    multi_fields = group_schema()['properties']['targets']['items']['properties']['grounding']['required']
    assert {'scene_role', 'edit_purpose'} <= set(twin_fields)
    assert {'scene_role', 'edit_purpose'} <= set(multi_fields)
    with pytest.raises(ValueError):
        multi_prompt({'4': 'add'}, objects, ['U1'], content_policy='unrecorded')
