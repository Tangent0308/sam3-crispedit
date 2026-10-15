import copy
import pytest
from sa1b_pipeline.instruction_protocol import make_spec, render, validate_pair, length_advice


@pytest.mark.parametrize('op,payload,kwargs,expected', [
    ('add','two red tags',{'placement':{'relation':'attached to','site':'handle'}},'Add two red tags attached to the selected region.'),
    ('replace','a rigid arched handle',{'target_part':'handle'},'Replace the selected region with a rigid arched handle.'),
    ('remove','',{'remove_mode':'part','target_part':'logo'},'Remove the selected region.'),
    ('remove','',{'remove_mode':'contents'},'Remove the selected region, including its contents.'),
    ('remove','',{'remove_mode':'contents_only'},'Remove the selected region.'),
    ('remove','',{'remove_mode':'lowerlegs'},'Remove the selected region to make knee-length shorts.'),
    ('remove','',{'remove_mode':'hem'},'Remove the selected region to make it knee-length.'),
    ('attribute','amber',{'attribute_kind':'tint'},'Tint the selected region amber.'),
    ('attribute','OPEN!',{'attribute_kind':'text'},'Change the text on the selected region to "OPEN!".'),
    ('attribute','translucent',{'attribute_kind':'opacity'},'Make the selected region translucent.'),
    ('attribute','',{'attributes':[{'property':'color','value':'blue'},{'property':'finish','value':'matte'}]},'Change the color of the selected region to blue and its finish to matte.'),
])
def test_semantic_slots_survive_ref_removal(op,payload,kwargs,expected):
    # Synthetic grammar fixtures only, not source-grounded editing proposals.
    s=make_spec(op,'the red suitcase on the left','suitcase',payload,**kwargs)
    assert render(s,False)==expected
    assert 'the red suitcase on the left' in render(s)
    validate_pair(s,render(s),expected)
    with pytest.raises(ValueError,match='semantic|blind'):
        validate_pair(s,render(s),expected.replace('selected region','selected bag'))


def test_no_minimum_or_hard_truncation_and_invalid_property_rejected():
    s=make_spec('remove','it','vase')
    assert render(s)=='Remove it.'
    s['ref']='the '+('very '*50)+'large vase'
    assert length_advice(render(s),render(s,False))
    with pytest.raises(ValueError,match='property'):
        make_spec('attribute','the vase','vase','blue',attribute_kind='invented')


def test_relative_add_must_be_explicit_and_text_is_exact():
    with pytest.raises(ValueError,match='anchor_mode'):
        make_spec('add','the desk','desk','a bin',placement={'relation':'beside'})
    s=make_spec('add','the desk','desk','a bin',placement={'relation':'beside','anchor_mode':'reference'})
    assert render(s,False)=='Add a bin beside the selected region.'
    s=make_spec('attribute','the sign','sign','A "quote"!',attribute_kind='text')
    assert '"A \\"quote\\"!"' in render(s)

from pathlib import Path
import numpy as np
from PIL import Image
from sa1b_pipeline.instructions import build_unit
from sa1b_pipeline.specialist_editors import writable_mask
from sa1b_pipeline.training import build_training_rows
from sa1b_pipeline.editor import edit_prompt


def test_verified_scope_wins_and_samtok_binds_before_tint_value():
    unit=build_unit(dict(operation='attribute',ref_phrase='the clear bottle',scope='bottle',
                        payload='amber',attribute_kind='tint'),
                    dict(object_id='1',category='container,box'), 'mask.png',dict(bbox=[0,0,8,8],point=[4,4]),[])
    assert unit.scope=='bottle'
    row={**unit.to_dict(),'instruction':unit.instruction_ref,'source_image':'s.png','edited_image':'e.png'}
    code='<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>'
    rows=build_training_rows(row,code)
    assert rows[2]['prompt']==f'Tint the selected region {code} amber.'


def test_subpart_cannot_silently_use_the_whole_host(tmp_path):
    host=np.zeros((20,20),dtype='uint8');host[2:18,2:18]=255
    part=np.zeros_like(host);part[4:7,4:7]=255
    hp,ep=tmp_path/'host.png',tmp_path/'part.png'
    Image.fromarray(host).save(hp);Image.fromarray(part).save(ep)
    raw=dict(operation='remove',ref_phrase='the red shirt',scope='shirt',target_part='logo',remove_mode='part')
    obj=dict(object_id='1',category='shirt')
    geometry=dict(bbox=[2,2,18,18],point=[10,10])
    with pytest.raises(ValueError,match='execution_mask'):
        build_unit(raw,obj,str(hp),geometry,[])
    obj['metadata']={'execution_mask_path':str(ep)}
    unit=build_unit(raw,obj,str(hp),geometry,[])
    assert unit.instruction_noref=='Remove the selected region.'
    assert writable_mask(unit.to_dict(),(20,20)).sum()==9
    assert 'retain the remaining host' in edit_prompt(unit.to_dict())
    broken=unit.to_dict();broken['execution_mask_path']=None
    with pytest.raises(ValueError,match='execution_mask'):
        writable_mask(broken,(20,20))


def test_extended_attribute_prompt_does_not_forbid_requested_text():
    unit=build_unit(dict(operation='attribute',ref_phrase='the blue sign',scope='sign',
                        attribute_kind='text',payload='OPEN!'),dict(object_id='1',category='sign'),
                    'mask.png',dict(bbox=[0,0,8,8],point=[4,4]),[])
    assert '"OPEN!"' in unit.instruction_ref
    prompt=edit_prompt(unit.to_dict())
    assert 'unrequested text' in prompt and 'requested content change' in prompt
    assert 'pose, edges, text,' not in prompt


def test_candidate_roundtrip_keeps_resolved_geometry(tmp_path):
    from sa1b_pipeline.io import candidate_from_dict
    raw={'source_id':'s','image_path':'source.png','objects':[{'object_id':'1','category':'shirt','metadata':{'execution_mask_path':'part.png'}}]}
    candidate=candidate_from_dict(raw,tmp_path)
    expected=str(tmp_path/'part.png')
    assert candidate.objects[0].metadata['execution_mask_path']==expected
    assert candidate_from_dict(candidate.to_dict()).objects[0].metadata['execution_mask_path']==expected


def test_small_subpart_edit_not_vetoed_by_unchanged_host(tmp_path):
    from sa1b_pipeline.validate import deterministic_check
    host=np.zeros((30,30),dtype='uint8');host[2:28,2:28]=255
    part=np.zeros_like(host);part[10:13,10:13]=255
    a=np.zeros((30,30,3),dtype='uint8');b=a.copy();b[part>0]=255
    for name,arr in [('source',a),('edited',b),('host',host),('part',part)]:Image.fromarray(arr).save(tmp_path/f'{name}.png')
    spec=make_spec('remove','the shirt','shirt',target_part='logo',remove_mode='part')
    row={'source_image':str(tmp_path/'source.png'),'edited_image':str(tmp_path/'edited.png'),'mask':str(tmp_path/'host.png'),'execution_mask_path':str(tmp_path/'part.png'),'operation':'remove','instruction_spec':spec}
    result=deterministic_check(row)
    assert 'remove_mask_change_too_weak' not in result['failure_reasons']
    row['edited_image']=row['source_image']
    assert 'no_visible_target_change' in deterministic_check(row)['failure_reasons']


def test_text_literal_equal_to_locator_does_not_capture_mask_token():
    unit=build_unit(dict(operation='attribute',ref_phrase='the selected sign',scope='sign',
                        attribute_kind='text',payload='the selected sign'),dict(object_id='1',category='sign'),
                    'mask.png',dict(bbox=[0,0,8,8],point=[4,4]),[])
    pair={**unit.to_dict(),'instruction':unit.instruction_ref,'source_image':'s.png','edited_image':'e.png'}
    code='<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>'
    rows=build_training_rows(pair,code)
    assert rows[2]['prompt']==f'Change the text on the selected region {code} to "the selected sign".'
