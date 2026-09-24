import numpy as np
from synthesis_pipeline.refine_samtok_regions import select_target, neighbor_union, attach_structural_part, declared_carried_queries, carried_outside, external_accessory_risk
from synthesis_pipeline.generate_samtok_plan import unsafe_surface_attribute


def test_refinement_selects_matching_instance_and_rejects_scene_mask():
    anchor=np.zeros((100,100),bool);anchor[20:50,20:50]=True
    whole=anchor.copy();whole[50:80,33:37]=True
    other=np.zeros_like(anchor);other[20:50,60:90]=True
    result,record=select_target([(other,.99),(np.ones_like(anchor),.99),(whole,.9)],anchor,'complete')
    assert np.array_equal(result,whole)
    assert record['candidate_index']==2


def test_surface_refinement_cannot_move_to_neighbor():
    anchor=np.zeros((100,100),bool);anchor[20:70,20:60]=True
    neighbor=np.zeros_like(anchor);neighbor[20:70,70:95]=True
    result,record=select_target([(neighbor,.99)],anchor,'surface')
    assert result is None and record['status']=='unresolved'


def test_strict_surface_does_not_clip_a_broader_requested_object():
    anchor=np.zeros((100,100),bool);anchor[20:70,20:60]=True
    bigger=anchor.copy();bigger[70:75,20:60]=True
    assert select_target([(bigger,.99)],anchor,'surface')[0] is not None
    assert select_target([(bigger,.99)],anchor,'surface',.98)[0] is None


def test_declared_carried_item_requires_pixel_coverage_not_just_owner_match():
    row={'task_type':'remove','masked_content':'Player holding a baseball bat in both hands beside another player.'}
    assert declared_carried_queries(row)==['baseball bat']
    assert declared_carried_queries({**row,'masked_content':'Batter in a jersey, grey pants, and a bat held in both hands.'})==['bat']
    owner=np.zeros((100,100),bool);owner[20:70,40:70]=True
    bat=np.zeros_like(owner);bat[40:45,15:45]=True
    assert carried_outside([(bat,.9)],owner) is not None
    assert carried_outside([(bat,.9)],owner|bat) is None
    assert declared_carried_queries({**row,'task_type':'attribute'})==[]


def test_external_bag_risk_does_not_require_an_exhaustive_vlm_inventory():
    person=np.zeros((100,100),bool);person[10:90,40:80]=True
    bag=np.zeros_like(person);bag[40:60,25:42]=True
    row={'task_type':'replace','masked_content':'Person wearing yellow shirt and backpack.'}
    risk=external_accessory_risk(row,lambda q:[(bag,.9)] if q=='bag' else [],person)
    assert risk and risk['query']=='bag'
    assert external_accessory_risk({**row,'task_type':'attribute'},lambda q:[(bag,.9)],person) is None


def test_neighbor_guard_excludes_target_and_includes_other_instance():
    target=np.zeros((100,100),bool);target[20:60,20:50]=True
    neighbor=np.zeros_like(target);neighbor[20:60,55:85]=True
    assert np.array_equal(neighbor_union([(target,.99),(neighbor,.9)],target),neighbor)


def test_broad_flat_fill_is_rejected_but_material_surface_is_allowed():
    assert unsafe_surface_attribute('Make the head and shoulder on the right bright red.')
    assert unsafe_surface_attribute('Make the left mirror opaque and non-reflective.')
    assert not unsafe_surface_attribute('Make the hair of the person on the right white.')
    assert not unsafe_surface_attribute('Make the glass surface of the left mirror frosted.')
    assert not unsafe_surface_attribute('Make the bag carried by the man on the right red.')
    assert unsafe_surface_attribute('Make the head and shoulder with brown hair glossy.')


def test_ambiguous_touching_structural_parts_are_not_silently_unioned():
    anchor=np.zeros((100,100),bool);anchor[20:40,30:60]=True
    one=np.zeros_like(anchor);one[40:70,32:35]=True
    two=np.zeros_like(anchor);two[40:70,53:56]=True
    assert attach_structural_part([(one,.9),(two,.9)],anchor) is None


def test_same_category_alias_with_between_locator():
    from synthesis_pipeline.generate_samtok_plan import same_replacement_category
    assert same_replacement_category('slender stick between person and foliage','thin branch')


def test_holding_phrase_does_not_replace_head_category():
    from synthesis_pipeline.generate_samtok_plan import same_replacement_category
    assert not same_replacement_category('man holding umbrella','woman holding umbrella')
    assert same_replacement_category('chair under table','wooden chair')


def test_addition_composition_preserves_remote_neighbor_not_old_support_through_item():
    from PIL import Image
    from synthesis_pipeline.compose_segmented_replacements import compose_addition,addition_phrase
    assert addition_phrase('Add a sprig of parsley to the left toast.')=='a sprig of parsley'
    source=Image.new('RGB',(100,100),'white');raw=Image.new('RGB',(100,100),'red')
    target=np.zeros((100,100),bool);target[30:60,30:60]=True
    added=np.zeros_like(target);added[55:65,55:65]=True
    result,alpha=compose_addition(source,raw,added,target,(0,0,100,100),np.ones_like(target))
    assert result.getpixel((62,62))==(255,0,0)
    assert result.getpixel((10,10))==(255,255,255)


def test_same_category_is_detected_in_instruction_despite_generic_reference():
    from synthesis_pipeline.generate_samtok_plan import normalize_generated
    value=dict(masked_content='slender stick',edit_unit_status='complete_object',
        outside_dependencies='none',refer_object='slender stick located between the person and foliage',
        mask_compatibility='compatible',compatibility_reason='',
        editing_instruction='Replace the slender stick between the person and foliage with a thin branch.',
        new_instruction='Replace the slender stick with a thin branch.')
    assert normalize_generated(value,'replace') is None


def test_structural_completion_is_a_provisional_plan_not_an_arbitrary_fragment():
    from synthesis_pipeline.generate_samtok_plan import normalize_generated
    value=dict(masked_content='chocolate cake pop',edit_unit_status='complete_part',
        outside_dependencies='none',refer_object='front-right chocolate cake pop',
        mask_compatibility='compatible',compatibility_reason='',
        editing_instruction='Remove the front-right chocolate cake pop and its wooden stick.',
        new_instruction='Remove this cake pop and its wooden stick.',segmentation_target='cake pop',
        mask_refinement='complete',structural_parts=['wooden stick','wooden stick'],protected_objects=[])
    result=normalize_generated(value,'remove')
    assert result is not None and result['structural_parts']==['wooden stick']
    value['edit_unit_status']='incomplete'
    assert normalize_generated(value,'remove') is None


def test_provisional_part_plan_cannot_bypass_refinement_with_legacy_editor():
    import pytest
    from utils.context_edit import validate_refinement_execution
    row=dict(task_type='remove',edit_unit_status='complete_part',structural_parts=['stick'])
    with pytest.raises(ValueError,match='legacy crop masks'):
        validate_refinement_execution(row,'mirage')
    with pytest.raises(ValueError,match='refinement before'):
        validate_refinement_execution(row,'context_grounded_v3')
    row['region_contract']={'status':'segmented_candidate'}
    validate_refinement_execution(row,'context_grounded_v3')
    row.pop('region_contract');row['task_type']='replace'
    with pytest.raises(ValueError,match='refinement before'):
        validate_refinement_execution(row,'context_grounded_v4')


def test_replacement_can_request_verified_structural_completion():
    from synthesis_pipeline.generate_samtok_plan import normalize_generated,build_prompt
    value=dict(masked_content='dry yellow leaf with thin stem',edit_unit_status='complete_part',
        outside_dependencies='none',refer_object='dry yellow leaf at the bottom right',
        mask_compatibility='compatible',compatibility_reason='',
        editing_instruction='Replace the dry yellow leaf at the bottom right with a brown pinecone.',
        segmentation_target='dry yellow leaf',mask_refinement='complete',
        structural_parts=['thin stem of the bottom-right leaf'],protected_objects=['left leaf'])
    result=normalize_generated(value,'replace')
    assert result is not None and result['structural_parts']==value['structural_parts']
    assert 'structural_parts' in build_prompt({},'replace')
    assert 'structural_parts' not in build_prompt({},'add')
    value['outside_dependencies']='another independent object supported by the leaf'
    assert normalize_generated(value,'replace') is None



def test_reference_binding_tracks_ver_mentions_and_preserves_group_ambiguity():
    import json
    from synthesis_pipeline.reference_binding import bind_reference,binding_prompt
    a='<|mt_start|><|mt_0023|><|mt_0311|><|mt_end|>'
    b='<|mt_start|><|mt_0160|><|mt_0258|><|mt_end|>'
    answer=f'The slender wire ({a}) and the slender stick ({b}) are near the person.'
    assert bind_reference(answer,0,2)['label']=='The slender wire'
    assert bind_reference(answer,1,2)['label']=='the slender stick'
    assert 'slender stick' not in binding_prompt(bind_reference(answer,0,2))
    assert bind_reference(answer,0,1)['status']=='unresolved'
    group=json.dumps([dict(mask_2d=a,label='one of two cups'),dict(mask_2d=b,label='one of two cups')])
    assert bind_reference(group,1,2)['status']=='bound_shared_group'
    assert 'GROUP' in binding_prompt(bind_reference(group,1,2))
    assert bind_reference(answer.replace(b,a),0,2)['status']=='unresolved'
    group=f'The two animals ({a} and {b}) are near a tent.'
    assert bind_reference(group,0,2)['status']=='bound_shared_group'
    assert bind_reference(group,1,2)['label']=='The two animals'


def test_regional_denoising_anchors_context_without_reinjecting_target():
    import torch
    from utils.region_denoise import editable_token_weights,anchor_step
    mask=np.zeros((256,256),bool);mask[120:135,120:135]=True
    weights=editable_token_weights(mask,(32,32),'remove')
    assert weights[16,16]==1 and weights[0,0]==0
    z=torch.full((1,1024,4),3.);reference=torch.ones_like(z);noise=torch.zeros_like(z)
    w=torch.from_numpy(weights.reshape(1,-1,1))
    out=anchor_step(z,reference,noise,w,.25)
    assert out[0,16*32+16,0]==3 and out[0,0,0]==.75


def test_physical_face_mask_is_not_annotation_and_one_instruction_suffices():
    from synthesis_pipeline.generate_samtok_plan import normalize_generated,has_annotation_language
    assert not has_annotation_language('man wearing a face mask on the right')
    assert has_annotation_language('the object in the mask outline')
    val=dict(masked_content='man wearing a face mask and beige shirt',edit_unit_status='complete_object',
        outside_dependencies='none',refer_object='man on the right wearing a face mask',
        mask_compatibility='compatible',compatibility_reason='',
        editing_instruction='Remove the man on the right wearing a face mask.',
        segmentation_target='man',mask_refinement='complete',protected_objects=['person'],structural_parts=[])
    result=normalize_generated(val,'remove')
    assert result and result['new_instruction']==val['editing_instruction']
