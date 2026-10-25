import json
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from test_multi_transactional import plan, Editor
from sa1b_pipeline.multi_parallel import generate_parallel_multi_group, select_valid_component_subset
from sa1b_pipeline.multi_component_audit import CHECKS, normalize_result, verify_component_fusion


def test_generation_failure_can_skip_only_when_later_input_is_last_committed_image(tmp_path):
    from sa1b_pipeline.multi_component_audit import generation_prefix
    def image(name,value):
        p=tmp_path/name;Image.new('RGB',(16,16),(value,0,0)).save(p);return str(p)
    source=image('source.png',0);good=image('good.png',20)
    bad=image('failed.png',99);later=image('later.png',40)
    def step(uid,ok,src,tar):
        return {'accepted':ok,'member':{'unit_id':uid,'step_input_image':src,'step_output_image':tar}}
    rows=[step('U1',True,source,good),step('U2',False,good,bad),step('U3',True,good,later)]
    assert [d['member']['unit_id'] for d in generation_prefix(rows)]==['U1','U3']
    rows[2]['member']['step_input_image']=bad
    assert len(generation_prefix(rows))==1
    rows[2]['member']['step_input_image']=good
    assert len(generation_prefix(rows,{'edit_dependencies':[['U2','U3']]}))==1


def test_failed_write_does_not_discard_later_valid_steps_or_enter_final(plan,tmp_path,independent,monkeypatch):
    import sa1b_pipeline.multi_parallel as mod
    original=mod.run_committed_step
    def reject_second(*args,**kwargs):
        result=original(*args,**kwargs)
        if result['member']['unit_id']=='U2':
            result['accepted']=False
            result['reasons']=['test_deterministic_failure']
        return result
    monkeypatch.setattr(mod,'run_committed_step',reject_second)
    plan['edit_dependencies']=[['U1','U2'],['U1','U3'],['U1','U4']]
    auditor=ComponentAuditor()
    group=generate_parallel_multi_group(plan,tmp_path/'skip_failed',lambda:Editor(),
                                       auditor,audit_policy='packed-components-v4')
    assert {u['unit_id'] for u in group['units']}=={'U1','U3','U4'}
    assert len(auditor.calls)==1 and verify_component_fusion(group)['pass']
    assert tuple(np.asarray(Image.open(group['joint_target']))[20,50])==(100,100,100)


class ComponentAuditor:
    config={'model':'fake-component'}
    def __init__(self, fail=(), crash=False): self.fail=set(fail);self.calls=[];self.crash=crash
    def call_batch(self,jobs):
        self.calls.extend(deepcopy(jobs))
        if self.crash:
            self.crash=False
            raise RuntimeError('transient audit failure')
        result=[]
        for job in jobs:
            assert '::component::' in job['job_id']
            assert len(job['images']) == 3
            ids=job['output_schema']['properties']['units']['items']['properties']['unit_id']['enum']
            rows=[dict(unit_id=uid, source_state='original surface', result_state='changed surface',
                       reference_count='one',closest_alternative='none',observed_operation={'U1':'remove','U2':'replace','U3':'attribute','U4':'add'}[uid],
                       **{k:uid not in self.fail for k in CHECKS}, serious_defects=[], warnings=[],issue='') for uid in ids]
            raw={'units':rows}
            if 'interfaces' in job['output_schema']['properties']:
                raw['interfaces']=[dict(before='U3',after='U1',exposed_appearance='blue surface',
                    adjacent_appearance='blue surface',consistent=True,issue='none')]
            result.append(dict(job_id=job['job_id'],semantic_audit=raw))
        return result


@pytest.fixture
def independent(monkeypatch):
    import sa1b_pipeline.multi_parallel as mod
    real=mod.multi_support_preflight
    def preflight(g,root):
        result=real(g,root)
        for u in g['units']:u['unit']['execution_mask_path']=u['mask']
        return result
    monkeypatch.setattr(mod,'multi_support_preflight',preflight)


def test_single_audit_per_final_independent_region_no_joint_and_resume(plan,tmp_path,independent):
    a=ComponentAuditor(fail=['U2']);e=Editor();root=tmp_path/'run'
    g=generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')
    assert g['n_targets']==3 and g['joint_audit']['pass']
    assert len(a.calls)==4 and g['joint_audit']['group_mllm_calls']==0
    final=np.asarray(Image.open(g['joint_target']))
    assert tuple(final[20,50])==(100,100,100)
    assert tuple(final[50,20])==(0,0,240)
    assert verify_component_fusion(g)['pass']
    assert generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')==g
    assert len(a.calls)==4 and e.calls==4
    final=final.copy();final[20,20]=[2,3,4];Image.fromarray(final).save(g['joint_target'])
    with pytest.raises(ValueError,match='differs from audited'):
        verify_component_fusion(g)


def test_dependent_final_audited_once_and_atomic_rollback(plan,tmp_path,independent):
    plan['edit_dependencies']=[['U2','U3']]
    a=ComponentAuditor(fail=['U2']);e=Editor()
    g=generate_parallel_multi_group(plan,tmp_path/'run',lambda:e,a,audit_policy='component-final-v2')
    assert len(a.calls)==3
    assert set(g['retention']['dropped_unit_ids'])=={'U2','U3'}
    job=next(j for j in a.calls if 'U2+U3' in j['job_id'])
    audited=np.asarray(Image.open(job['images'][2]))
    assert tuple(audited[20,50])==(250,0,0) and tuple(audited[50,20])==(0,0,240)
    final=np.asarray(Image.open(g['joint_target']))
    assert tuple(final[20,50])==(100,100,100) and tuple(final[50,20])==(100,100,100)


def test_all_connected_targets_one_call_no_reaudit_after_transient(plan,tmp_path):
    plan['edit_dependencies']=[['U2','U3']]
    a=ComponentAuditor(crash=True);e=Editor();root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='transient'):
        generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')
    assert e.calls==4
    g=generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')
    assert e.calls==4 and len(a.calls)==2 and g['joint_audit']['pass']
    assert g['parallel']['component_semantic_calls']==1


def test_malformed_units_and_missing_checks_fail_closed():
    members=[{'unit_id':'U1'},{'unit_id':'U2'}]
    for raw in ({},{'units':[{'unit_id':'U1'},{'unit_id':'U1'}]},{'units':[{'unit_id':'U1'},{'unit_id':'U2'}]}):
        assert all(a['decision']=='fail' for a in normalize_result(raw,members).values())
    raw={'decision':'fail','failure_reasons':['truncated_audit'], 'units':[
        dict(unit_id=m['unit_id'], source_state='old', result_state='new', **{k:True for k in CHECKS}) for m in members]}
    assert all(a['decision']=='fail' for a in normalize_result(raw,members).values())


def test_atomic_audit_cannot_keep_unaudited_prefix(plan,tmp_path,monkeypatch):
    # Contract rejection of the full chain must not silently pick its old prefix.
    import sa1b_pipeline.multi_parallel as mod
    def only_two(g):
        if len(g['units'])!=2:raise ValueError('must be two')
    monkeypatch.setattr(mod,'validate_multi_record',only_two)
    r=[{'audit_atomic':True,'kept':plan['units'],'current':'final'}]
    result,dropped=select_valid_component_subset(plan,r)
    assert result==r and dropped==[]


def test_observed_type_and_reference_override_optimistic_booleans():
    member={'unit_id':'U1','operation':'replace'}
    row=dict(unit_id='U1',source_state='floral cloth',result_state='plain blue cloth',
             **{k:True for k in CHECKS},reference_count='one',observed_operation='attribute',
             serious_defects=[],warnings=['minor texture difference'])
    assert 'observed_operation_mismatch' in normalize_result({'units':[row]},[member])['U1']['failure_reasons']
    row['observed_operation']='replace';row['reference_count']='multiple'
    assert 'reference_not_unique' in normalize_result({'units':[row]},[member])['U1']['failure_reasons']
    row['reference_count']='one'
    assert normalize_result({'units':[row]},[member])['U1']['decision']=='pass'


def test_nonoverlapping_semantic_dependency_uses_prior_pixels_and_single_audit(plan,tmp_path,independent):
    plan['edit_dependencies']=[['U3','U1']]
    plan['generation_order']=['U3','U1','U2','U4']
    plan['dependency_evidence']=[{'before':'U3','after':'U1','relation':'reveals_edited_surface',
                                'evidence':'removal reveals modified surface'}]
    seen=[]
    class RecordingEditor(Editor):
        def generate(self,path,unit,seed=0):
            seen.append((unit['unit_id'],np.array(Image.open(path)),deepcopy(unit)))
            return super().generate(path,unit,seed=seed)
    editor=RecordingEditor();audit=ComponentAuditor()
    result=generate_parallel_multi_group(plan,tmp_path/'run',lambda:editor,audit,audit_policy='component-final-v2')
    remove=next(x for x in seen if x[0]=='U1')
    assert tuple(remove[1][50,20])==(0,0,240)
    assert remove[2]['dependency_context'][0]['before']=='U3'
    assert len(audit.calls)==3 and result['joint_audit']['group_mllm_calls']==0
    job=next(j for j in audit.calls if 'U3+U1' in j['job_id'])
    assert 'DEPENDENT EDITS' in job['prompt'] and 'old-color patch' in job['prompt']
    assert len(seen)==4 and verify_component_fusion(result)['pass']


def test_interface_contradiction_overrides_optimistic_member_checks():
    deps=[dict(before='U3',after='U1',relation='reveals_edited_surface')]
    members=[dict(unit_id='U3',operation='attribute'),dict(unit_id='U1',operation='remove')]
    rows=[dict(unit_id=m['unit_id'],source_state='before',result_state='after',reference_count='one',
               observed_operation=m['operation'],**{k:True for k in CHECKS}) for m in members]
    raw={'units':rows,'interfaces':[dict(before='U3',after='U1',exposed_appearance='gray patch',
              adjacent_appearance='blue water',consistent=False,issue='old color hole')]}
    result=normalize_result(raw,members,deps)
    assert result['U3']['decision']=='pass' and result['U1']['decision']=='fail'
    raw.pop('interfaces')
    assert normalize_result(raw,members,deps)['U1']['decision']=='fail'


@pytest.mark.parametrize('failed', [False, True])
def test_source_overlap_serial_pixels_joint_audit_and_independent_fusion(plan,tmp_path,independent,failed):
    # U1/U2 share two columns; the other two writes remain independent.
    mask_path=plan['units'][1]['mask']
    m=np.asarray(Image.open(mask_path)).copy();m[10:40,38:41]=255
    Image.fromarray(m).save(mask_path)
    from sa1b_pipeline.geometry import mask_geometry
    geometry=mask_geometry(m>0)
    plan['units'][1].update(geometry)
    plan['units'][1]['unit'].update({k:geometry[k] for k in ('bbox','point')})
    seen={}
    class RecordingEditor(Editor):
        def generate(self,path,unit,seed=0):
            seen[unit['unit_id']]=np.array(Image.open(path))
            return super().generate(path,unit,seed=seed)
    e=RecordingEditor();a=ComponentAuditor(fail=['U2'] if failed else [])
    root=tmp_path/'overlapping'
    result=generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')
    assert result['parallel']['components']==[['U1','U2'],['U3'],['U4']]
    assert result['parallel']['edges'][0]['reason']=='source_mask_overlap'
    assert tuple(seen['U2'][20,39])==(240,240,240) # Sees U1's committed pixels.
    assert np.all(seen['U3']==100) and np.all(seen['U4']==100)
    assert len(a.calls)==(4 if failed else 3) # One bounded prefix audit on failure.
    joint=next(j for j in a.calls if j['job_id'].endswith('U1+U2'))
    assert tuple(np.asarray(Image.open(joint['images'][2]))[20,39])==(250,0,0)
    final=np.asarray(Image.open(result['joint_target']))
    assert tuple(final[20,39])==((240,240,240) if failed else (250,0,0))
    assert tuple(final[50,20])==(0,0,240) and tuple(final[50,50])==(0,240,0)
    assert result['n_targets']==(3 if failed else 4)
    assert verify_component_fusion(result)['pass']
    before=(e.calls,len(a.calls))
    assert generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')==result
    assert (e.calls,len(a.calls))==before


def compact_row(member, **changes):
    row=dict(unit_id=member['unit_id'], image_ok=True, instruction_ok=True,
             reference_unique=True, observed_operation=member['operation'], ref='',
             optimized_instruction='', reason='Requested edit is visible on the uniquely identified source object.')
    row.update(changes)
    return row


def test_compact_correction_updates_export_and_uses_repaired_uniqueness(plan):
    from sa1b_pipeline.instruction_protocol import render
    member=deepcopy(plan['units'][0]);old_noref=member['instruction_noref']
    spec=deepcopy(member['instruction_spec']);spec['ref']='the topmost cup on the left'
    row=compact_row(member,ref=spec['ref'],optimized_instruction=render(spec),
                    reason='Topmost and left distinguish this cup from the lower cup.')
    audit=normalize_result({'units':[row]},[member])[member['unit_id']]
    assert audit['decision']=='pass'
    assert member['instruction']==render(spec) and member['instruction_noref']==old_noref
    assert member['instruction_revision']['applied']
    assert member['unit']['instruction_ref']==render(spec)


@pytest.mark.parametrize('failure',['image','payload','ambiguous','malformed','instruction'])
def test_compact_correction_cannot_hide_failed_image_or_change_goal(plan,failure):
    from sa1b_pipeline.instruction_protocol import render
    member=deepcopy(plan['units'][1]);before=deepcopy(member)
    spec=deepcopy(member['instruction_spec']);spec['ref']='the topmost cup on the right'
    row=compact_row(member,ref=spec['ref'],optimized_instruction=render(spec))
    if failure=='image':row['image_ok']=False
    elif failure=='payload':row['optimized_instruction']='Replace the topmost cup on the right with a chair.'
    elif failure=='ambiguous':row['reference_unique']=False
    elif failure=='instruction':row['instruction_ok']=False
    else:row.pop('reason')
    audit=normalize_result({'units':[row]},[member])[member['unit_id']]
    assert audit['decision']=='fail'
    assert member==before


def test_compact_three_image_input_is_clean_marked_final(plan,tmp_path,independent):
    a=ComponentAuditor();e=Editor()
    g=generate_parallel_multi_group(plan,tmp_path/'run',lambda:e,a,audit_policy='component-final-v2')
    for job in a.calls:
        clean,marked,final=map(Image.open,job['images'])
        assert clean.size==marked.size==final.size
        assert job['images'][0]==plan['source_image']
        assert job['images'][1].endswith('source_masks.png')
        assert not np.array_equal(np.array(clean),np.array(marked))
        labels=json.loads((__import__('pathlib').Path(job['images'][1]).parent/'mask_labels.json').read_text())
        assert [x['unit_id'] for x in labels]==job['job_id'].split('::')[-1].split('+')
        fields=job['output_schema']['properties']['units']['items']['properties']
        assert len(fields)==8 and 'optimized_instruction' in fields


def test_compact_response_drives_real_retention_and_resume(plan,tmp_path,independent):
    class CompactAuditor(ComponentAuditor):
        def call_batch(self,jobs):
            self.calls.extend(deepcopy(jobs))
            by_id={m['unit_id']:m for m in plan['units']}
            return [dict(job_id=j['job_id'],semantic_audit={'units':[
                compact_row(by_id[uid],image_ok=uid!='U2',
                            reason='Replacement missing.' if uid=='U2' else 'Requested change complete; source reference unique.')
                for uid in j['output_schema']['properties']['units']['items']['properties']['unit_id']['enum']]})
                for j in jobs]
    a=CompactAuditor();e=Editor();root=tmp_path/'compact'
    g=generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')
    assert g['n_targets']==3 and g['retention']['dropped_unit_ids']==['U2']
    assert verify_component_fusion(g)['pass']
    assert len(a.calls)==4
    assert generate_parallel_multi_group(plan,root,lambda:e,a,audit_policy='component-final-v2')==g
    assert len(a.calls)==4


@pytest.mark.parametrize('full_anchor',[False,True])
def test_empty_rewrite_with_exact_original_reference_is_noop(plan,full_anchor):
    from sa1b_pipeline.instruction_protocol import noun,render
    member=deepcopy(plan['units'][3])
    member['instruction_spec']['placement']['site']='front surface'
    member['instruction']=render(member['instruction_spec'])
    member['instruction_noref']=render(member['instruction_spec'],False)
    member['ref']=member['instruction_spec']['ref']
    ref=member['ref']
    if full_anchor:ref=noun('front surface')+' of '+ref
    row=compact_row(member,ref=ref)
    result=normalize_result({'units':[row]},[member])[member['unit_id']]
    assert result['decision']=='pass' and result['reference_echo_ignored']
    assert member['ref']==member['instruction_spec']['ref']


def test_empty_rewrite_with_changed_reference_still_rejected(plan):
    member=deepcopy(plan['units'][3]);before=deepcopy(member)
    row=compact_row(member,ref='the upper left cup')
    result=normalize_result({'units':[row]},[member])[member['unit_id']]
    assert result['decision']=='fail' and member==before


def test_prefix_reuse_requires_identical_displayed_context(tmp_path):
    from sa1b_pipeline.multi_component_audit import _prefix_pixels_unchanged
    from sa1b_pipeline.production_state import file_hash
    old=tmp_path/'old.png';new=tmp_path/'new.png'
    pixels=np.zeros((100,100,3),dtype='uint8');pixels[70:80,70:80]=255
    Image.fromarray(pixels).save(old);pixels[70:80,70:80]=0;Image.fromarray(pixels).save(new)
    report={'audited_final':str(old),'image_hashes':{str(old):file_hash(old)},
            'aligned_crop_boxes':[{'unit_id':'U1','box':[0,0,50,50],'detail_box':[0,0,50,50]}]}
    assert _prefix_pixels_unchanged(report,[{'unit_id':'U1'}],str(new))
    pixels[45,45]=255;Image.fromarray(pixels).save(new)
    assert not _prefix_pixels_unchanged(report,[{'unit_id':'U1'}],str(new))
    assert not _prefix_pixels_unchanged({},[{'unit_id':'U1'}],str(new))


def test_prefix_reaudit_failure_is_not_retried_again(plan,tmp_path,independent):
    plan['edit_dependencies']=[['U1','U2']]
    class RejectPrefix(ComponentAuditor):
        def call_batch(self,jobs):
            result=super().call_batch(jobs)
            for job,r in zip(jobs,result):
                if job['job_id'].endswith('::U1'):
                    for row in r['semantic_audit']['units']:
                        row.update({k:False for k in CHECKS})
            return result
    a=RejectPrefix(fail=['U2']);e=Editor()
    result=generate_parallel_multi_group(plan,tmp_path/'run',lambda:e,a,audit_policy='component-final-v2')
    assert set(result['retention']['dropped_unit_ids'])=={'U1','U2'}
    assert len(a.calls)==4 and e.calls==4 and verify_component_fusion(result)['pass']


def test_dropped_enclosing_mask_does_not_revoke_retained_selector_and_preflight_is_repeatable(plan,tmp_path):
    from sa1b_pipeline.group_generation import multi_support_preflight,_unit_dict
    from sa1b_pipeline.specialist_editors import writable_mask,read_binary_mask
    size=Image.open(plan['source_image']).size
    parent=tmp_path/'dropped_parent.png';Image.new('L',size,255).save(parent)
    plan['protected_source_masks']=[str(parent)]
    first=multi_support_preflight(plan,tmp_path/'first')
    before=[]
    for member in plan['units']:
        support=writable_mask(_unit_dict(member),size)
        selector=np.asarray(Image.open(member['mask']))>0
        assert not (support&~selector).any()
        if member['operation'] in {'remove','replace'}:assert not (selector&~support).any()
        before.append(support.copy())
    multi_support_preflight(plan,tmp_path/'second')
    for expected,member in zip(before,plan['units']):
        assert np.array_equal(writable_mask(_unit_dict(member),size),expected)
    empty=tmp_path/'empty_protection.png';Image.new('L',size,0).save(empty)
    assert not read_binary_mask(str(empty),size,allow_empty=True).any()
    with pytest.raises(ValueError,match='empty mask'):read_binary_mask(str(empty),size)


def test_local_replacement_receives_current_background_without_more_generations(plan,tmp_path,independent):
    group=deepcopy(plan);host=group['units'][2];local=group['units'][1]
    group.update(units=[host,local],n_targets=2,operations=['attribute','replace'],
                 generation_order=['U2','U3'],edit_dependencies=[])
    bg=np.zeros((128,128),bool);bg[40:,:]=True;bg[45:95,30:98]=False
    target=np.zeros_like(bg);target[82:92,35:95]=True
    for member,m in [(host,bg),(local,target)]:
        p=tmp_path/(member['unit_id']+'_context.png');Image.fromarray(m.astype('uint8')*255).save(p)
        member['mask']=str(p);member['unit']['mask_path']=str(p)
    original=np.full((128,128,3),100,np.uint8);original[bg]=(20,40,170)
    Image.fromarray(original).save(group['source_image'])
    inputs=[]
    class ContextEditor:
        config={'model':'current-context'};last_generation_context={}
        def generate(self,path,unit,seed=0):
            a=np.asarray(Image.open(path)).copy();inputs.append((unit['unit_id'],a.copy()))
            m=np.asarray(Image.open(unit['mask_path']))>0
            a[m]=(20,160,120) if unit['unit_id']=='U3' else a[110,110]
            return Image.fromarray(a)
    class ContextAuditor(ComponentAuditor):
        def call_batch(self,jobs):
            responses=super().call_batch(jobs)
            for response in responses:
                raw=response['semantic_audit']
                if 'interfaces' in raw:
                    raw['interfaces']=[dict(before='U3',after='U2',exposed_appearance='green',
                        adjacent_appearance='green',consistent=True,issue='')]
            return responses
    auditor=ContextAuditor()
    result=generate_parallel_multi_group(group,tmp_path/'generation',ContextEditor,auditor,
                                        audit_policy='packed-components-v4')
    assert [uid for uid,_ in inputs]==['U3','U2']
    assert tuple(inputs[1][1][110,110])==(20,160,120)
    assert result['joint_audit']['pass'] and len(auditor.calls)==1
    assert result['edit_dependencies']==[]  # Conditioning is not an eligibility constraint.
    assert verify_component_fusion(result)['pass']
