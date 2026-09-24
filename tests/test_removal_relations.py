from copy import deepcopy
import numpy as np
import pytest
from PIL import Image
from types import SimpleNamespace
from synthesis_pipeline.resolve_removal_relations import choose_auxiliary,relation_queries
from synthesis_pipeline.plan_removal_relations import validate_relation_plan
from synthesis_pipeline.prepare_samtok_data import encode_rle
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
from utils.context_edit import edit_context_crop


def test_visual_binding_does_not_copy_shared_source_expression():
    from utils.removal_relations import relation_input_evidence
    mask=np.zeros((20,20),bool);mask[5:12,5:12]=True
    row=dict(problem='both named objects',answer='',mask_index=0,num_masks=2,
             reference_binding=dict(status='bound_shared_group',label='both named objects'))
    text=relation_input_evidence(row,mask,(0,0,20,20),visual_binding=True)
    assert 'both named objects' not in text
    assert 'outlined photograph' in text


def test_v11_instruction_is_derived_from_selected_target_only():
    import json
    from synthesis_pipeline.plan_removal_relations import parse_relation_response
    value=parse_relation_response(json.dumps(dict(target='the person on the right',
        instruction='Remove everything.',relations=[])), 'relations-v11')
    assert value['instruction']=='Remove the person on the right.'


def test_v14_normalizes_query_list_and_rejects_verbose_grounding_query():
    import json
    from synthesis_pipeline.plan_removal_relations import parse_relation_response
    record=dict(decision='accept',target='the person on the right',target_point=[450,450],
        support_check=dict(target_supports_other_living=False,other_living_still_supported=True),
        reconstruction='Continue the wall.',relations=[dict(description='wooden tray',action='remove_together',
        reason='held only by the selected person',point=[500,500],bbox=[400,450,600,550],
        segmentation_queries=['wooden tray','small tray'])])
    value=parse_relation_response(json.dumps(record),'relations-v14')
    assert value['relations'][0]['segmentation_query']=='wooden tray'
    assert validate_relation_plan(value,'relations-v14')=='accepted'
    value['relations'][0]['segmentation_queries'][1]='an excessively detailed grounding phrase with too many words'
    assert validate_relation_plan(value,'relations-v14')=='invalid_segmentation_queries'


def test_action_prompt_does_not_repeat_public_accessories():
    from utils.removal_relations import action_removal_prompt
    mask=np.zeros((20,20),bool);mask[5:12,5:12]=True
    plan=dict(target='the selected person',instruction='Remove the person and item.',
              reconstruction='Continue the wall.')
    prompt=action_removal_prompt(plan,mask,(0,0,20,20),dict(resolved_co_removals=['held item','held item']))
    assert prompt.count('held item')==1 and 'Continue the wall.' in prompt


def test_auxiliary_is_point_bound_and_cannot_delete_known_neighbor():
    target=np.zeros((100,100),bool);target[30:60,30:60]=True
    auxiliary=np.zeros_like(target);auxiliary[58:70,45:50]=True
    guard=np.zeros_like(target)
    found,_=choose_auxiliary([(auxiliary,.9)],[470,640],target,guard)
    assert np.array_equal(found,auxiliary)
    assert choose_auxiliary([(auxiliary,.9)],[900,900],target,guard)[0] is None
    assert choose_auxiliary([(auxiliary,.9)],[470,640],target,auxiliary)[0] is None


def test_v14_query_aliases_are_bounded_and_backwards_compatible():
    relation=dict(segmentation_query=' Wooden tray ',
        segmentation_queries=['wooden tray','small tray','basket','fourth alias'])
    assert relation_queries(relation)==['Wooden tray','small tray','basket']


def test_planner_bbox_can_resolve_point_bound_shape_ambiguity():
    target=np.zeros((100,100),bool);target[60:75,45:55]=True
    vertical=np.zeros_like(target);vertical[40:62,47:53]=True
    horizontal=np.zeros_like(target);horizontal[49:55,30:70]=True
    guard=np.zeros_like(target);candidates=[(horizontal,.9),(vertical,.85)]
    assert choose_auxiliary(candidates,[500,520],target,guard)[0] is None
    found,checks=choose_auxiliary(candidates,[500,520],target,guard,bbox=[450,380,550,650],
        min_score=.6,min_bbox_iou=.15)
    assert np.array_equal(found,vertical)
    assert checks[1]['bbox_iou']>checks[0]['bbox_iou']


def test_whole_owner_is_not_accepted_as_auxiliary():
    target=np.zeros((100,100),bool);target[30:60,30:60]=True
    assert choose_auxiliary([(target,.99)],[450,450],target,np.zeros_like(target))[0] is None
    target[:]=False;target[40:45,40:45]=True
    assert choose_auxiliary([(target,.99)],[420,420],target,np.zeros_like(target))[0] is None


def test_small_accessory_already_covered_by_target_is_not_rejected():
    target=np.zeros((100,100),bool);target[30:60,30:60]=True
    part=np.zeros_like(target);part[40:45,40:45]=True
    found,_=choose_auxiliary([(part,.99)],[420,420],target,np.zeros_like(target))
    assert np.array_equal(found,part)


def test_covered_part_confidence_does_not_reaudit_trusted_source_mask():
    target=np.zeros((100,100),bool);target[30:60,30:60]=True
    part=np.zeros_like(target);part[40:45,40:45]=True
    found,checks=choose_auxiliary([(part,.31)],[420,420],target,np.zeros_like(target))
    assert np.array_equal(found,part) and checks[0]['already_covered']
    # Confidence gates remain mandatory for any new pixel and wrong locators.
    part[29,42]=True
    assert choose_auxiliary([(part,.31)],[420,420],target,np.zeros_like(target))[0] is None
    part[29,42]=False
    assert choose_auxiliary([(part,.31)],[900,900],target,np.zeros_like(target))[0] is None


@pytest.mark.parametrize('policy',['relations-v9','relations-v10'])
def test_support_conflict_rejects_even_if_model_decision_says_accept(policy):
    mask=np.zeros((100,100),bool);mask[30:60,30:60]=True
    value=dict(decision='accept',target='selected object',target_point=[450,450],
        instruction='Remove the selected object.',reconstruction='Continue the background.',relations=[],
        support_check=dict(target_supports_other_living=True,other_living_still_supported=False))
    assert validate_relation_plan(value,policy,mask)=='defer_support_conflict'
    value['support_check']['other_living_still_supported']=True
    assert validate_relation_plan(value,policy,mask)=='accepted'
    value['support_check']['other_living_still_supported']='yes'
    assert validate_relation_plan(value,policy,mask)=='invalid_support_check'


def test_plan_reasoning_must_finish_before_json_is_accepted():
    from synthesis_pipeline.plan_removal_relations import parse_relation_response
    assert parse_relation_response('<think>{"decision":"accept"}') is None
    assert parse_relation_response('<think>{"decision":"accept"}</think>{"decision":"defer"}')=={'decision':'defer'}


def test_execution_region_is_separate_and_does_not_mutate_original_label():
    target=np.zeros((96,96),bool);target[32:64,32:64]=True
    execution=target.copy();execution[60:72,45:50]=True
    row={'task_type':'remove','editing_instruction':'Remove the selected item and its attachment.',
        'mask':encode_rle(target),'region_contract':{'status':'original','source_size':[96,96]},
        'execution_region':{'status':'resolved_auxiliary','source_size':[96,96],'mask':encode_rle(execution)},
        'relation_execution_context':'Preserve the independent neighboring instance.'}
    before=deepcopy(row);captured={}
    class QwenImage21Pipeline:
        def __call__(self,**kwargs):
            captured.update(kwargs);return SimpleNamespace(images=[kwargs['image']])
    edit_context_crop(QwenImage21Pipeline(),Image.new('RGB',(96,96)),row,target,None,grounded_composition=True)
    assert row==before and 'independent neighboring' in captured['prompt']
    assert 'fixed anchors' not in captured['prompt']
    edit_context_crop(QwenImage21Pipeline(),Image.new('RGB',(96,96)),row,target,None,
        grounded_composition=True,relation_geometry_policy='visible-v1')
    assert row==before and 'fixed anchors' in captured['prompt']
    row['execution_region']['mask']=encode_rle(np.zeros_like(target))
    with pytest.raises(ValueError,match='shrink'):
        edit_context_crop(QwenImage21Pipeline(),Image.new('RGB',(96,96)),row,target,None,grounded_composition=True)


def test_relation_actions_are_not_inferred_from_object_words():
    value=dict(decision='accept',target='selected object',instruction='Remove the selected object.',reconstruction='Continue the background.',
        relations=[dict(description='another entity',reason='independent',action='keep',point=[200,400],segmentation_query='entity')])
    assert validate_relation_plan(value)=='accepted'
    value['relations'][0]['action']='automatic'
    assert validate_relation_plan(value)=='invalid_action'


def test_outside_pointers_use_disjoint_original_pixels():
    target=np.zeros((100,100),bool);target[30:70,30:60]=True
    excluded=target.copy();excluded[40:60,65:75]=True
    source=Image.new('RGB',(100,100),(80,90,100))
    before=excluded.copy()
    image=instruction_target_crop(source,target,target_pointer=True,excluded_mask=excluded)
    assert image.width>768 and np.array_equal(excluded,before)


def test_v4_skinny_crop_at_edge_preserves_mask_and_adds_context():
    from utils.removal_relations import relation_context_bbox
    mask=np.zeros((832,1248),bool);mask[117:781,1090:]=True
    box=relation_context_bbox(mask)
    assert box[2:]==(1248,832) and box[0]<700 and box[1]==0
    assert mask[box[1]:box[3],box[0]:box[2]].sum()==mask.sum()


def test_v4_editor_filters_by_extent_not_just_point_and_keeps_coremove():
    from utils.removal_relations import compile_relation_context
    plan=dict(reconstruction='Continue the wall.',relations=[
        dict(description='outside person',action='keep',bbox=[700,0,900,900]),
        dict(description='partially visible person',action='keep',bbox=[450,0,900,900]),
        dict(description='exclusive accessory',action='remove_together',bbox=[700,0,900,900])])
    before=deepcopy(plan)
    prompt,evidence=compile_relation_context(plan,(0,0,500,1000),(1000,1000))
    assert 'outside person' not in prompt and 'partially visible person' in prompt
    assert 'exclusive accessory' in prompt and plan==before
    assert evidence['omitted_off_crop_keep']==['outside person']


@pytest.mark.parametrize('policy',['relations-v4','relations-v5','relations-v6','relations-v7'])
def test_v4_target_point_must_match_original_mask_not_neighbor(policy):
    mask=np.zeros((100,100),bool);mask[30:60,30:60]=True
    value=dict(decision='accept',target='selected object',target_point=[450,450],
        instruction='Remove the selected object.',reconstruction='Continue the background.',relations=[])
    assert validate_relation_plan(value,policy,mask)=='accepted'
    value['target_point']=[800,800]
    assert validate_relation_plan(value,policy,mask)=='defer_target_point_outside_mask'


@pytest.mark.parametrize('policy',['relations-v4','relations-v5','relations-v6','relations-v7'])
def test_v4_instruction_separate_from_actual_execution_prompt(policy):
    target=np.zeros((96,96),bool);target[32:64,32:64]=True
    row={'task_type':'remove','editing_instruction':'Remove the selected person.',
        'mask':encode_rle(target),'region_contract':{'status':'original','source_size':[96,96]},
        'execution_region':{'status':'resolved_auxiliary','source_size':[96,96],'mask':encode_rle(target)},
        'relation_policy':policy,'relation_plan':{'relations':[
            {'action':'remove_together','description':'exclusive equipment','bbox':[200,300,600,700]}],
            'reconstruction':'Continue the pavement.'}}
    captured={};before=deepcopy(row)
    class QwenImage21Pipeline:
        def __call__(self,**kwargs):
            captured.update(kwargs);return SimpleNamespace(images=[kwargs['image']])
    edit_context_crop(QwenImage21Pipeline(),Image.new('RGB',(96,96)),row,target,None,grounded_composition=True)
    assert row==before and 'exclusive equipment' in captured['prompt']
    assert 'equipment' not in row['editing_instruction']
    edit_context_crop(QwenImage21Pipeline(),Image.new('RGB',(96,96)),row,target,None,
        grounded_composition=True,qwen21_prompt_policy='relation-compact-v1')
    assert 'together with exclusive equipment' in captured['prompt']
    assert row==before and captured['prompt'].count('Remove ')==1


def test_v4_point_repair_anchors_are_inside_actual_mask():
    import json
    from synthesis_pipeline.plan_removal_relations import point_repair_feedback
    mask=np.zeros((100,100),bool);mask[20:70,40:55]=True
    feedback=point_repair_feedback(mask,{'target':'untrusted'})
    points=json.loads(feedback.split('mask fragments: ')[1].split('. Identify')[0])
    for x,y in points:assert mask[round(y/10),round(x/10)]
    assert 'untrusted' in feedback


def test_editor_locator_uses_crop_coordinates_not_full_photo():
    from utils.removal_relations import crop_relative_target_hint
    mask=np.zeros((100,100),bool);mask[20:40,75:85]=True
    assert 'upper-right' in crop_relative_target_hint(mask,(0,0,100,100))
    assert 'middle-center' in crop_relative_target_hint(mask,(60,0,100,60))


def test_located_prompt_uses_only_resolved_external_attachments():
    from utils.removal_relations import located_removal_prompt
    mask=np.zeros((100,100),bool);mask[20:40,75:85]=True
    plan={'relations':[{'action':'remove_together','description':'covered clothing'},
                       {'action':'remove_together','description':'external equipment'}]}
    prompt=located_removal_prompt('Remove the rightmost person.',plan,mask,(0,0,100,100),
                                  {'resolved_co_removals':['external equipment']})
    assert prompt.startswith('Remove the rightmost person, together with external equipment.')
    assert 'Also remove' not in prompt
    assert 'covered clothing' not in prompt and 'upper-right' in prompt


def test_qwen21_target_guide_survives_prompt_rebuild_and_keeps_clean_image_first(tmp_path):
    mask=np.zeros((96,96),bool);mask[32:64,32:64]=True
    row={'task_type':'remove','editing_instruction':'Remove the middle object.','mask':encode_rle(mask),
         'relation_policy':'relations-v9','relation_plan':{'relations':[],'reconstruction':'Continue the surface.'},
         'region_contract':{'status':'original','source_size':[96,96]},
         'execution_region':{'status':'resolved_auxiliary','source_size':[96,96],'mask':encode_rle(mask)}}
    captured={};source=Image.new('RGB',(96,96),(40,60,80))
    class QwenImage21Pipeline:
        def __call__(self,**kwargs):
            captured.update(kwargs);return SimpleNamespace(images=[kwargs['image'][0]])
    edit_context_crop(QwenImage21Pipeline(),source,row,mask,None,grounded_composition=True,
        target_guide=True,qwen21_prompt_policy='typed-v1',diagnostics_dir=tmp_path)
    assert len(captured['image'])==2 and 'Image 2 marks the target' in captured['prompt']
    assert np.array_equal(np.asarray(captured['image'][0]),np.asarray(source))
    assert not np.array_equal(np.asarray(captured['image'][1]),np.asarray(source))
    assert (tmp_path/'native_model_output.png').exists()
