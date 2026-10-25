from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest
from test_multi_transactional import plan, Auditor
from sa1b_pipeline.multi_execution_contract import (scope_contradiction, validate_target_evidence,
    order_dependencies, infer_reveal_dependencies, schedule_content_dependencies, retain_feasible_plan)
from sa1b_pipeline.multi_parallel import generate_parallel_multi_group
from sa1b_pipeline.group_planning import build_multi_group
from sa1b_pipeline.group_generation import multi_support_preflight
from sa1b_pipeline.geometry import mask_geometry


def test_explicit_extent_mismatch_not_source_occlusion():
    assert scope_contradiction('Hull of a long-tail boat in water','boat')
    assert scope_contradiction('Collar of the coat','coat')
    assert not scope_contradiction('Hull of the boat','boat hull')
    assert not scope_contradiction('Occluded boat behind the pier','boat')
    assert not scope_contradiction('Front of a boat','boat')
    with pytest.raises(ValueError,match='unsupported'):
        validate_target_evidence({'operation':'remove','grounding':{'unsupported_after':['flowers']}})


def test_topology_repairs_preferred_order_but_never_swallows_cycles():
    assert order_dependencies(['A','B','C'],['B','C','A'],[['A','B']])[0]==['C','A','B']
    for edges in [[['A','A']],[['X','B']],[['A','B'],['B','A']]]:
        with pytest.raises(ValueError):order_dependencies(['A','B'],['A','B'],edges)


def version4_inputs(plan):
    objects=[];targets=[];paths=[];geoms=[]
    for u in plan['units']:
        paths.append(u['mask']);mask=np.array(Image.open(u['mask']))>0;g=mask_geometry(mask);g['point']=u['point'];geoms.append(g)
        objects.append(dict(object_id=u['object_id'],category='cup',metadata={}))
        target=deepcopy(u['instruction_spec']);target.update(unit_id=u['unit_id'],object_id=u['object_id'],ref_phrase=u['ref'],evidence='visible')
        target['grounding']=dict(entity='cup',mask_description='selected cup',visible_before='a gray cup',
            current_value='gray' if u['operation']=='attribute' else '',edit_feasible=True,reason='visible fit',
            extent_match=True,ref_unique=True,unsupported_after=[],preserve=['background'],required_change='requested visible change',
            contact_xy=[u['point'][0]/127,u['point'][1]/127] if u['operation']=='add' else [])
        targets.append(target)
    return objects,paths,geoms,dict(grounding_version=4,feasible=True,targets=targets,
        generation_order=plan['generation_order'],edit_dependencies=[])


def test_partial_plan_preserves_uids_geometry_and_dropped_selector(plan,tmp_path):
    objects,paths,geoms,raw=version4_inputs(plan)
    raw['targets'][1]['grounding']['edit_feasible']=False
    built=build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,
        split='train',group_id='p0-test',geometry=geoms,assignments={u['object_id']:u['operation'] for u in plan['units']},require_grounding=True)
    assert [u['unit_id'] for u in built['units']]==['U1','U3','U4']
    assert built['units'][1]['bbox']==plan['units'][2]['bbox']
    assert built['protected_source_masks']==[paths[1]]
    assert built['planning_retention']['dropped'][0]['unit_id']=='U2'
    multi_support_preflight(built,tmp_path/'preflight')
    rejected=np.array(Image.open(paths[1]))>0
    for u in built['units']:
        protected=np.array(Image.open(u['unit']['protected_mask_path']))>0
        assert protected[rejected].all()


def test_v4_dependency_is_carried_to_executable_group(plan):
    objects,paths,geoms,raw=version4_inputs(plan)
    raw['edit_dependencies']=[dict(before='U3',after='U1',relation='reveals_edited_surface',evidence='cup covers edited surface')]
    built=build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,
        split='train',group_id='p0-test',geometry=geoms,assignments={u['object_id']:u['operation'] for u in plan['units']},require_grounding=True)
    assert built['edit_dependencies']==[['U3','U1']]
    assert built['generation_order'].index('U3')<built['generation_order'].index('U1')


def test_reveal_uses_recolored_input_without_extra_generation(plan,tmp_path,monkeypatch):
    # Disjoint selectors: a held object occupies a hole in a large fabric mask.
    g=deepcopy(plan);g['units']=[g['units'][2],g['units'][0]]
    host,rem=g['units'];g.update(n_targets=2,operations=['attribute','remove'],generation_order=[rem['unit_id'],host['unit_id']])
    tool=np.zeros((128,128),bool);tool[50:70,50:70]=True
    fabric=np.zeros_like(tool);fabric[15:110,15:110]=True;fabric &= ~tool
    for u,m in [(host,fabric),(rem,tool)]:
        p=tmp_path/(u['unit_id']+'.png');Image.fromarray(m.astype('uint8')*255).save(p);u['mask']=str(p);u['unit']['mask_path']=str(p)
    original=np.zeros((128,128,3),np.uint8);original[fabric]=(0,0,200);original[tool]=120;Image.fromarray(original).save(g['source_image'])
    assert infer_reveal_dependencies(g)[0]['before']==host['unit_id']
    inputs=[]
    class Editor:
        config={'model':'test'};last_generation_context={}
        def generate(self,path,unit,seed=0):
            a=np.array(Image.open(path));inputs.append((unit['unit_id'],a.copy()))
            m=np.array(Image.open(unit['mask_path']))>0
            a[m]=(200,0,0) if unit['operation']=='attribute' else a[45,55]
            return Image.fromarray(a)
    class Audit(Auditor):
        def call_batch(self,jobs):
            result=super().call_batch(jobs)
            for j,r in zip(jobs,result):
                if 'observed_operation' in r['semantic_audit']:
                    r['semantic_audit']['observed_operation']='remove' if j['job_id'].endswith('::'+rem['unit_id']) else 'attribute'
            return result
    result=generate_parallel_multi_group(g,tmp_path/'run',Editor,Audit())
    assert len(inputs)==2 and inputs[0][0]==host['unit_id']
    assert tuple(inputs[1][1][45,55])==(200,0,0)
    assert tuple(np.array(Image.open(result['joint_target']))[55,55])==(200,0,0)
    # Input plan stays immutable and no expensive inference is needed on resume.
    assert g['generation_order'][0]==rem['unit_id']
    assert generate_parallel_multi_group(g,tmp_path/'run',Editor,Audit())==result
    assert len(inputs)==2


def test_plan_subset_does_not_require_post_generation_site_resolution(plan):
    from sa1b_pipeline.group_schema import validate_multi_record
    objects,paths,geoms,raw=version4_inputs(plan)
    built=build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,
        split='train',group_id='p0-pregen',geometry=geoms,
        assignments={u['object_id']:u['operation'] for u in plan['units']},require_grounding=True)
    pairs=[{'object_ids':[objects[0]['object_id'],objects[1]['object_id']],'eligible':True}]
    retained=retain_feasible_plan(built,pairs)
    assert len(retained['units'])==4
    assert all(u.get('blind_eligible') is not True for u in retained['units'])
    for u in retained['units']:u['support_mask']=u['mask']
    validate_multi_record(retained,require_resolved_regions=False)
    with pytest.raises(ValueError,match='unresolved'):
        validate_multi_record(retained)


def test_observation_expansion_is_explicit_not_the_default():
    from sa1b_pipeline.multi_plan_schema import multi_plan_schema
    from sa1b_pipeline.group_planning import multi_prompt
    assert multi_plan_schema()['properties']['grounding_version']=={'const':3}
    assert multi_plan_schema(execution_contract=True)['properties']['grounding_version']=={'const':4}
    args=({'x':'remove'},[{'object_id':'x'}],['U1'])
    assert 'EXECUTION CONTRACT' not in multi_prompt(*args)
    assert 'EXECUTION CONTRACT' in multi_prompt(*args,execution_contract=True)


def test_v5_partial_planning_keeps_source_binding_without_self_certification(plan):
    from sa1b_pipeline.multi_compact_target import group_schema as output_schema
    objects,paths,geoms,raw=version4_inputs(plan)
    raw['grounding_version']=5
    for t in raw['targets']:
        t['grounding'].pop('extent_match');t['grounding'].pop('ref_unique')
        t['grounding'].update(excluded='table',nearest_rival='neighbor cup',distinction=t['ref_phrase'])
    raw['targets'][1]['grounding']['edit_feasible']=False
    built=build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,
        split='train',group_id='v5-test',geometry=geoms,assignments={u['object_id']:u['operation'] for u in plan['units']},require_grounding=True)
    assert [u['unit_id'] for u in built['units']]==['U1','U3','U4']
    assert built['units'][1]['mask']==paths[2]
    assert built['protected_source_masks']==[paths[1]]
    g=output_schema()['properties']['targets']['items']['properties']['grounding']
    assert 'ref_unique' not in g['properties']
    assert list(g['properties']).index('excluded')<list(g['properties']).index('edit_feasible')


def test_tight_mask_view_keeps_excluded_pixels_gray_and_enlarges_tiny_target():
    from sa1b_pipeline.multi_grounded_views import grounded_sheet
    source=np.zeros((180,240,3),dtype='uint8');source[:]=(200,0,0)
    mask=np.zeros((180,240),bool);mask[85:95,115:125]=True;source[mask]=(0,180,0)
    sheet=np.asarray(grounded_sheet(Image.fromarray(source),[{'object_id':'a'}],[mask]))
    context=sheet[32:384,:512];zoom=sheet[32:384,512:]
    assert ((zoom[:,:,1]>120)&(zoom[:,:,0]<50)).sum()>((context[:,:,1]>120)&(context[:,:,0]<50)).sum()*3
    # Red excluded source pixels never leak through the enlarged selector.
    assert not ((zoom[:,:,0]>100)&(zoom[:,:,1]<50)).any()


def test_compact_target_preserves_edit_content_and_derives_fixed_fields():
    from sa1b_pipeline.multi_compact_target import normalize, schema, prompt
    raw={'grounding':{'entity':'chair'},'ref_phrase':'the chair beside the table',
         'payload':'a square cushion','placement':{'relation':'on','site':'seat','anchor_mode':'support'},'attributes':[]}
    built=normalize(raw,'add')
    assert built['payload']==raw['payload'] and built['placement']==raw['placement']
    assert built['operation']=='add' and built['object_id']=='' and built['target_part']==''
    assert 'operation' not in schema()['properties']
    assert schema()['properties']['grounding']['properties']['reason']['maxLength']==160
    assert 'fallible' in prompt('remove',[0,0,1,1],'bottle cap')


def test_v5_ref_selection_marker_is_dropped_not_published(plan):
    objects,paths,geoms,raw=version4_inputs(plan);raw['grounding_version']=5
    raw['targets'][1]['ref_phrase']='the selected cup'
    built=build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,
        split='train',group_id='v5-ref',geometry=geoms,assignments={u['object_id']:u['operation'] for u in plan['units']},require_grounding=True)
    assert [u['unit_id'] for u in built['units']]==['U1','U3','U4']
    assert 'selection markers' in built['planning_retention']['dropped'][0]['reason']


def test_thinking_planner_exposes_only_final_json():
    from types import SimpleNamespace
    from sa1b_pipeline.mllm import VLLMSelector
    planner=VLLMSelector.__new__(VLLMSelector)
    planner.batch_size=8;planner._inherited_backend=False;planner.enable_thinking=True;planner.params=None
    planner._multi_messages=lambda paths,prompt: ('prompt',[])
    planner.llm=SimpleNamespace(generate=lambda *a,**k: [SimpleNamespace(prompt_token_ids=[1,2],outputs=[
        SimpleNamespace(text='internal analysis </think> {"feasible":true}',token_ids=[1,2,3],finish_reason='stop')])])
    result=planner.generate_multimage([([], 'test')])[0]
    assert result['feasible'] is True
    assert result['_generation_metadata']['raw_output']=='{"feasible":true}'
    assert result['_generation_metadata']['thinking_enabled'] is True


def test_binary_planning_column_preserves_exact_selector_shape():
    from sa1b_pipeline.multi_grounded_views import grounded_sheet
    mask=np.zeros((180,240),bool);mask[50:130,90:150]=True;mask[70:90,110:130]=False
    source=Image.new('RGB',(240,180),(110,110,110))
    im=np.asarray(grounded_sheet(source,[{'object_id':'a'}],[mask],binary_column=True))
    assert im.shape[:2]==(384,1536)
    binary=im[32:,1024:]
    assert (binary==255).all(axis=2).any() and (binary==0).all(axis=2).any()


def test_partial_order_can_omit_only_a_rejected_target(plan):
    objects,paths,geoms,raw=version4_inputs(plan)
    raw['targets'][1]['grounding']['edit_feasible']=False
    raw['generation_order']=[u for u in raw['generation_order'] if u!='U2']
    kwargs=dict(split='train',group_id='order',geometry=geoms,assignments={u['object_id']:u['operation'] for u in plan['units']},require_grounding=True)
    g=build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,**kwargs)
    assert set(g['generation_order'])=={'U1','U3','U4'}
    raw['generation_order'].remove('U3')
    with pytest.raises(ValueError,match='invalid generation order'):
        build_multi_group({'source_id':'s','image_path':plan['source_image']},objects,raw,paths,**kwargs)


def test_explicit_attachment_guard_leaves_removable_caps_and_neighbors_valid():
    from sa1b_pipeline.multi_execution_contract import excluded_support_contradiction
    assert excluded_support_contradiction({'entity':'boat hull','excluded':'the blue canopy above the hull'})
    assert not excluded_support_contradiction({'entity':'bottle cap','excluded':'bottle neck'})
    assert not excluded_support_contradiction({'entity':'boat hull','excluded':'a canopy on another boat'})
    assert not excluded_support_contradiction({'entity':'boat hull','excluded':'a canopy on the pier'})
    assert not excluded_support_contradiction({'entity':'boat','excluded':'water and distant boats'})


def test_multiscale_reveal_crosses_annotation_gap_but_not_far_surface(plan,tmp_path):
    g=deepcopy(plan);g['units']=[g['units'][2],g['units'][0]]
    host,rem=g['units'];g.update(generation_order=[rem['unit_id'],host['unit_id']])
    target=np.zeros((128,128),bool);target[50:70,40:90]=True
    fabric=np.zeros_like(target);fabric[15:115,15:115]=True;fabric[46:74,36:94]=False
    for u,m in ((host,fabric),(rem,target)):
        path=tmp_path/(u['unit_id']+'.png');Image.fromarray(m.astype('uint8')*255).save(path)
        u['mask']=str(path);u['unit']['mask_path']=str(path)
    host['grounding']={'entity':'woven fabric'}
    rem['grounding']={'excluded':'woven fabric around the badge','reason':'removing badge reveals fabric'}
    edges=infer_reveal_dependencies(g)
    assert len(edges)==1 and edges[0]['gap_pixels']==5
    assert edges[0]['ring_coverage']>.2
    scheduled=schedule_content_dependencies(deepcopy(g))
    assert scheduled['generation_order']==[host['unit_id'],rem['unit_id']]
    # Text alone must not join a large but distant surface to the removal.
    fabric[25:105,25:105]=False
    Image.fromarray(fabric.astype('uint8')*255).save(host['mask'])
    assert infer_reveal_dependencies(g)==[]


def test_dependency_conditioning_keeps_public_instruction_unchanged(plan):
    from sa1b_pipeline.multi_execution_contract import dependency_context,dependency_prompt
    from sa1b_pipeline.editor import edit_prompt
    from sa1b_pipeline.group_generation import _unit_dict
    g=deepcopy(plan)
    g['dependency_schedule']={'evidence':[dict(before='U3',after='U1',relation='reveals_edited_surface')]}
    ctx=dependency_context(g,'U1',{'U3'})
    assert len(ctx)==1 and dependency_context(g,'U1',set())==[]
    u=_unit_dict(g['units'][0]);ref=u['instruction_ref'];blind=u['instruction_noref']
    u['dependency_context']=ctx
    assert 'CURRENT adjacent' in edit_prompt(u)
    assert ref==u['instruction_ref'] and blind==u['instruction_noref']
    assert dependency_prompt([])==''


def test_shared_context_relation_is_validated_and_ordered():
    from sa1b_pipeline.multi_execution_contract import bind_dependencies
    raw={'edit_dependencies':[{'before':'U2','after':'U1','relation':'changes_shared_context',
                              'evidence':'the removed item casts a shadow on the edited ground'}]}
    order,pairs,evidence=bind_dependencies(raw,['U1','U2'],['U1','U2'],['U1','U2'])
    assert order==['U2','U1'] and pairs==[['U2','U1']]
    raw['edit_dependencies'][0]['relation']='unknown'
    with pytest.raises(ValueError):bind_dependencies(raw,['U1','U2'],['U1','U2'],['U1','U2'])


def test_dependency_seam_permission_is_local_and_protects_unrelated_targets(plan,tmp_path):
    from sa1b_pipeline.multi_execution_contract import prepare_dependency_interfaces
    from sa1b_pipeline.geometry import load_mask
    from sa1b_pipeline.multi_parallel import conflict_components
    g=deepcopy(plan);host=g['units'][2];rem=g['units'][0];third=g['units'][1]
    target=np.zeros((128,128),bool);target[50:70,40:90]=True
    fabric=np.zeros_like(target);fabric[15:115,15:115]=True;fabric[46:74,36:94]=False
    other=np.zeros_like(target);other[43:46,55:65]=True;fabric &= ~other
    g['units']=[host,rem,third];g['generation_order']=[host['unit_id'],rem['unit_id'],third['unit_id']]
    for u,m in ((host,fabric),(rem,target),(third,other)):
        p=tmp_path/(u['unit_id']+'.png');Image.fromarray(m.astype('uint8')*255).save(p)
        u['mask']=str(p);u['unit']['mask_path']=str(p)
        # Fixture units carry no explicit write supports; use source-bound support.
        u['unit'].pop('execution_mask_path',None)
    g['dependency_schedule']={'evidence':[dict(before=host['unit_id'],after=rem['unit_id'],relation='reveals_edited_surface')]}
    g['edit_dependencies']=[[host['unit_id'],rem['unit_id']]]
    multi_support_preflight(g,tmp_path/'protect')
    records=prepare_dependency_interfaces(g,tmp_path/'interfaces')
    permission=load_mask({'mask_path':rem['unit']['dependency_overlap_mask_path']},(128,128))
    assert records[0]['permission_pixels']>0
    assert not (permission & ~fabric).any() and not (permission & other).any()
    assert not permission[15:25].any()
    protected=load_mask({'mask_path':rem['unit']['protected_mask_path']},(128,128))
    assert protected[other].all()
    support=load_mask({'mask_path':rem['unit']['execution_mask_path']},(128,128))
    assert support[target].all() and not (support & other).any()
    assert load_mask({'mask_path':rem['mask']},(128,128)).sum()==target.sum()
    components,_=conflict_components(g)
    assert any(host['unit_id'] in c and rem['unit_id'] in c for c in components)


def test_plan_retention_does_not_require_adjacency_metadata(plan):
    plan.pop('adjacent_pairs', None)
    retained = retain_feasible_plan(plan)
    assert retained['n_targets'] == 4
    assert retained['adjacent_pairs'] == []


def test_exact_selection_recheck_accepts_distant_masks_but_rejects_overlap(plan):
    from scripts.recheck_multi_selection import _exact_geometry
    objects = [dict(object_id=u['object_id'], mask_path=u['mask']) for u in plan['units']]
    row = dict(objects=[objects[0], objects[-1]], image_size=[128,128])
    assert _exact_geometry(row)['distance_required'] is False
    row['objects'][1] = dict(object_id='another', mask_path=objects[0]['mask_path'])
    assert _exact_geometry(row) is None
    row['objects'] = objects[:1]
    assert _exact_geometry(row) is None


def test_large_background_conditions_remove_and_replace_without_extending_masks(plan,tmp_path):
    from sa1b_pipeline.multi_execution_contract import infer_context_dependencies, schedule_content_dependencies
    g=deepcopy(plan);g['units']=g['units'][:3]
    host,remove,replacement=g['units']
    host['operation']='attribute';remove['operation']='remove';replacement['operation']='replace'
    # Background excludes an entire foreground silhouette; its upper part is
    # well separated from the visible background, despite sharing local context.
    bg=np.zeros((128,128),bool);bg[40:,:]=True;bg[45:95,30:98]=False
    a=np.zeros_like(bg);a[48:58,45:72]=True
    b=np.zeros_like(bg);b[82:92,35:95]=True
    for u,m in zip(g['units'],[bg,a,b]):
        p=tmp_path/(u['unit_id']+'context.png');Image.fromarray(m.astype('uint8')*255).save(p)
        u['mask']=str(p);u['unit']['mask_path']=str(p)
    g['generation_order']=[replacement['unit_id'],remove['unit_id'],host['unit_id']]
    g['edit_dependencies']=[]
    before=[Path(u['mask']).read_bytes() for u in g['units']]
    inferred=infer_context_dependencies(g)
    assert [host['unit_id'],replacement['unit_id']] in [[e['before'],e['after']] for e in inferred]
    schedule_content_dependencies(g)
    assert g['generation_order'].index(host['unit_id']) < g['generation_order'].index(replacement['unit_id'])
    assert [Path(u['mask']).read_bytes() for u in g['units']]==before
    # A distant surface does not create dependence or reject local targets.
    bg[:]=False;bg[:15,:]=True;Image.fromarray(bg.astype('uint8')*255).save(host['mask'])
    assert infer_context_dependencies(g)==[]
