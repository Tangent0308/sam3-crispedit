from copy import deepcopy
import pytest
from sa1b_pipeline.lean_planning import (schema, prompt, normalize, validate_response,
    validate_edit, merge_retry, choose_retry)

def target(uid='U1', feasible=True):
    return dict(unit_id=uid, entity='door',ref_phrase='the left door',current_value='white',
                feasible=feasible,reason='' if feasible else 'required support',payload='',
                attributes=[{'property':'color','value':'blue'}],placement=None,contact_xy=[])

def test_no_synthetic_evidence_and_public_render():
    from sa1b_pipeline.instruction_protocol import make_spec, render
    raw={'targets':[target(),target('U2')], 'edit_dependencies':[]}
    result=normalize(raw,'multi',{'U1':'attribute','U2':'attribute'},[{'object_id':'a'},{'object_id':'b'}])
    assert result['grounding_version']==6
    assert not {'reason','mask_description','kind','distinction','scene_role'} & set(result['targets'][0]['grounding'])-{'reason'}
    validate_edit(result['targets'][0])
    spec=make_spec(operation='attribute',scope='door',ref='the left door',attributes=[{'property':'color','value':'blue'}])
    assert 'the selected region' in render(spec, has_ref=False)

def test_retry_preserves_accepted_targets_and_validates_dependencies():
    a,b=target(),target('U2',False)
    first={'targets':[a,b],'edit_dependencies':[]}
    second={'targets':[target('U2')],'edit_dependencies':[]}
    merged=merge_retry(first,second,['U2'],{'U1':'attribute','U2':'replace'})
    assert merged['targets'][0]==a and first['targets'][1]==b
    bad=deepcopy(second);bad['targets'].append(target())
    with pytest.raises(ValueError):merge_retry(first,bad,['U2'],{'U1':'attribute','U2':'replace'})
    bad=deepcopy(second);bad['edit_dependencies']=[{'before':'U9','after':'U2','relation':'uses_edited_host','evidence':'shared host'}]
    with pytest.raises(ValueError):merge_retry(first,bad,['U2'],{'U1':'attribute','U2':'replace'})

def test_noop_and_add_contact_rejected():
    t=target();t.update(scope='door',operation='attribute',grounding={'current_value':'blue','contact_xy':[]})
    with pytest.raises(ValueError):validate_edit(t)
    t.update(operation='add',payload='a handle',attributes=[],placement={'relation':'on','site':'','anchor_mode':'support'})
    with pytest.raises(ValueError):validate_edit(t)

def test_deficit_retry_does_not_repeat_original_operation():
    new=choose_retry({'U1':'remove','U2':'remove'},['U1','U2'],{'add':0,'replace':0,'attribute':100})
    assert set(new.values())=={'add','replace'}

def test_schema_and_prompt_are_lean():
    assert len(schema('twins')['properties']['members']['items']['properties'])==6
    assert 'scene_role' not in str(schema('multi'))
    for kind in ('twins','multi'):
        p=prompt(kind,{'U1':'add','U2':'add'})
        assert len(p.split())<800
        assert 'sticker' not in p and 'badge' not in p

def test_both_group_builders_accept_lean_contract(tmp_path):
    import numpy as np
    from PIL import Image
    from sa1b_pipeline.group_planning import build_twin_group, build_multi_group
    image=tmp_path/'source.png';Image.new('RGB',(100,60),'white').save(image)
    objects=[];paths=[];geoms=[]
    for i,x in enumerate((10,60)):
        a=np.zeros((60,100),dtype=np.uint8);a[10:30,x:x+20]=255
        p=tmp_path/f'm{i}.png';Image.fromarray(a).save(p);paths.append(str(p))
        objects.append({'object_id':str(i),'category':'door'})
        geoms.append({'bbox':[x,10,x+20,30],'point':[x+10,20],'area_fraction':400/6000})
    candidate={'source_id':'test','image_path':str(image)}
    assignments={'U1':'attribute','U2':'attribute'}
    raw={'targets':[target(),{**target('U2'),'ref_phrase':'the right door'}],'edit_dependencies':[]}
    plan=normalize(raw,'multi',assignments,objects)
    g=build_multi_group(candidate,objects,plan,paths,split='train',group_id='multi',geometry=geoms,assignments={'0':'attribute','1':'attribute'},require_grounding=True)
    assert len(g['units'])==2
    members=[{k:t[k] for k in ('unit_id','entity','ref_phrase','current_value','contact_xy')} | {'placement_site':''} for t in raw['targets']]
    raw={'members':members,'feasible':True,'reason':'','payload':'','placement':None,'attributes':[{'property':'color','value':'blue'}]}
    plan=normalize(raw,'twins',assignments,objects)
    g=build_twin_group(candidate,objects,plan,paths,split='train',scope_level='object',group_id='twins',geometry=geoms,assigned_operation='attribute',require_grounding=True)
    assert g['members'][0]['instruction_noref']==g['members'][1]['instruction_noref']


def test_bounded_retry_checkpoint_and_accepted_target_lock(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    a,b=target(),target('U2',False)
    first={'targets':[a,b],'edit_dependencies':[]}
    second={'targets':[{**target('U2'),'ref_phrase':'the right door'}],'edit_dependencies':[]}
    class Planner:
        def __init__(self):self.calls=0
        def generate_multimage(self,requests,*,schemas=None):
            self.calls+=1
            assert len(requests)==1
            return [deepcopy(first if self.calls==1 else second)]
    def item():
        return dict(gid='multi-test',views=['a','b'],objects=[{'object_id':'x'},{'object_id':'y'}],
                    selection={'operations':['attribute','remove']},
                    prompt=prompt('multi',{'U1':'attribute','U2':'remove'},locations=[{'unit_id':'U1','bbox':[0,0,30,30]}]))
    planner=Planner();raw=plan_batch(planner,[item()],kind='multi',root=tmp_path,retry=1,accepted_counts={'attribute':0,'add':20,'replace':20})[0]
    assert planner.calls==2 and raw['_planning_calls']==2
    assert raw['targets'][0]['ref_phrase']==a['ref_phrase']
    assert raw['targets'][1]['operation']=='attribute'
    again=plan_batch(planner,[item()],kind='multi',root=tmp_path,retry=1,accepted_counts={'attribute':100})[0]
    assert planner.calls==2 and again==raw


def test_truncation_does_not_trigger_operation_retry(tmp_path):
    from sa1b_pipeline.lean_retry import plan_batch
    class Planner:
        def __init__(self):self.calls=0
        def generate_multimage(self,requests,*,schemas=None):
            self.calls+=1
            return [{'_generation_metadata':{'finish_reason':'length'}}]
    planner=Planner()
    item=dict(gid='bad',views=[],objects=[{'object_id':'a'},{'object_id':'b'}],
              selection={'operation':'add'},prompt=prompt('twins',{'U1':'add','U2':'add'}))
    raw=plan_batch(planner,[item],kind='twins',root=tmp_path,retry=1)[0]
    assert planner.calls==1 and raw['_planning_calls']==1 and not raw['feasible']

def test_operation_specific_schema_removes_dummy_fields():
    s=schema('twins',{'U1':'remove','U2':'remove'})
    assert set(s['properties'])=={'members','feasible','reason'}
    for row in s['properties']['members']['items']['anyOf']:
        assert set(row['properties'])=={'unit_id','entity','ref_phrase'}
    s=schema('multi',{'U1':'remove','U2':'add'},['U2'])
    row=s['properties']['targets']['items']['anyOf'][0]['properties']
    assert row['unit_id']=={'const':'U2'} and 'contact_xy' in row and 'attributes' not in row


def test_omitted_irrelevant_fields_are_not_visual_evidence():
    raw={'members':[dict(unit_id='U1',entity='chair',ref_phrase='the left chair'),
                    dict(unit_id='U2',entity='lamp',ref_phrase='the right lamp')], 'feasible':True,'reason':''}
    out=normalize(raw,'twins',{'U1':'remove','U2':'remove'},[{'object_id':'a'},{'object_id':'b'}])
    assert out['placement'] is None and out['members'][0]['grounding']['contact_xy']==[]
    assert not set(out['members'][0]['grounding']) & {'edit_feasible','site_visible','ref_unique','kind'}
