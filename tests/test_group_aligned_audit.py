from copy import deepcopy
import numpy as np
from PIL import Image
import pytest
from test_multi_transactional import plan, Editor
from test_multi_component_audit import ComponentAuditor, independent
from sa1b_pipeline.multi_parallel import generate_parallel_multi_group
from sa1b_pipeline.multi_component_audit import normalize_result, verify_component_fusion
from sa1b_pipeline.instruction_protocol import render


def test_grouped_audit_single_call_keeps_only_valid_independent_pixels(plan,tmp_path,independent):
    auditor=ComponentAuditor(fail=['U2']);editor=Editor();root=tmp_path/'run'
    result=generate_parallel_multi_group(plan,root,lambda:editor,auditor,audit_policy='group-final-v3')
    assert result['joint_audit']['pass'] and result['n_targets']==3
    assert len(auditor.calls)==2 and result['parallel']['component_semantic_calls']==2
    assert verify_component_fusion(result)['pass']
    assert tuple(np.asarray(Image.open(result['joint_target']))[20,50])==(100,100,100)
    assert generate_parallel_multi_group(plan,root,lambda:editor,auditor,audit_policy='group-final-v3')==result
    assert len(auditor.calls)==2 and editor.calls==4


def test_grouped_dependent_failure_discards_entire_component(plan,tmp_path,independent):
    plan['edit_dependencies']=[['U2','U3']]
    auditor=ComponentAuditor(fail=['U2']);editor=Editor()
    result=generate_parallel_multi_group(plan,tmp_path,lambda:editor,auditor,audit_policy='group-final-v3')
    assert result['joint_audit']['pass'] and result['n_targets']==2
    assert result['retention']['dropped_unit_ids']==['U2','U3']
    assert len(auditor.calls)==2 and verify_component_fusion(result)['pass']


def test_reference_only_revision_is_canonical_and_preserves_blind_goal(plan):
    member=deepcopy(plan['units'][0]);before=member['instruction_noref']
    row=dict(unit_id=member['unit_id'],reason='Upper left is distinct from the lower cup; the cup is removed.',
        image_ok=True,operation_ok=True,instruction_ok=True,reference_unique=True,ref='the topmost cup on the left')
    result=normalize_result({'units':[row]},[member])[member['unit_id']]
    assert result['decision']=='pass' and member['instruction_noref']==before
    assert member['instruction']==render(member['instruction_spec'])
    assert member['instruction_revision']['applied']
    member=deepcopy(plan['units'][0]);row['operation_ok']=False
    assert normalize_result({'units':[row]},[member])[member['unit_id']]['decision']=='fail'
    assert not member.get('instruction_revision',{}).get('applied')


def test_repeated_types_are_valid_group_members(plan):
    from sa1b_pipeline.group_schema import validate_multi_record
    for u in plan['units']:
        spec=deepcopy(plan['units'][0]['instruction_spec']);spec['ref']=u['ref']
        u.update(operation='remove',instruction_spec=spec,instruction=render(spec),instruction_noref=render(spec,False))
        u['support_mask']=u['mask']
    plan['operations']=['remove']*len(plan['units'])
    validate_multi_record(plan, require_resolved_regions=False)


def test_low_contrast_remove_needs_semantic_confirmation_not_automatic_acceptance(plan,tmp_path):
    from sa1b_pipeline.validate import deterministic_check
    member=plan['units'][0];source=Image.open(plan['source_image']).convert('RGB')
    pixels=np.asarray(source).copy();pixels[10:18,10:40]=240
    result=tmp_path/'removed.png';Image.fromarray(pixels).save(result)
    row=dict(source_image=plan['source_image'],edited_image=str(result),mask=member['mask'],operation='remove')
    assert deterministic_check(row)['decision']=='fail'
    assert deterministic_check(row,semantic_low_change=True)['decision']=='pending_semantic_audit'
    row['edited_image']=plan['source_image']
    assert deterministic_check(row,semantic_low_change=True)['decision']=='fail'


def test_global_recolor_of_tiny_fragment_is_not_a_completed_color_change(plan,tmp_path):
    from sa1b_pipeline.validate import deterministic_check
    member=plan['units'][0];pixels=np.asarray(Image.open(plan['source_image'])).copy()
    pixels[10:12,10:40]=240
    result=tmp_path/'partial.png';Image.fromarray(pixels).save(result)
    row=dict(source_image=plan['source_image'],edited_image=str(result),mask=member['mask'],operation='attribute',
             instruction_spec={'operation':'attribute','attributes':[{'property':'color','value':'white'}]})
    assert deterministic_check(row)['decision']=='pending_semantic_audit'
    check=deterministic_check(row,semantic_low_change=True)
    assert check['failure_reasons']==['global_color_change_confined_to_small_fragment']
    row['instruction_spec']['attributes']=[{'property':'text','value':'new label'}]
    assert deterministic_check(row,semantic_low_change=True)['decision']=='pending_semantic_audit'


def test_explicit_scope_failure_cannot_be_overridden_by_image_or_ref_rewrite(plan):
    member=deepcopy(plan['units'][0])
    row=dict(unit_id=member['unit_id'],source_state='two selected items',result_state='only one removed',
             image_ok=True,scope_ok=False,operation_ok=True,instruction_ok=True,reference_unique=True,
             ref='the topmost cup on the left')
    assert normalize_result({'units':[row]},[member])[member['unit_id']]['decision']=='fail'
    assert not member.get('instruction_revision',{}).get('applied')


def test_audit_evidence_uses_source_geometry_and_actual_final_pixels(plan):
    import json
    from sa1b_pipeline.aligned_audit import requests
    member=deepcopy(plan['units'][0])
    member['source_annotation_hint']={'category':'cup','description':'A fallible source description.'}
    original=deepcopy(member)
    evidence=json.loads(requests([member],plan['source_image'],[plan['source_image']]))[0]
    assert evidence['selected_pixels_changed_percent']==0
    assert len(evidence['source_mask_box_percent'])==4
    assert evidence['fallible_source_annotation']==member['source_annotation_hint']
    assert member==original


def test_source_annotation_hints_remain_bounded_and_mask_bound():
    from sa1b_pipeline.content_planning import annotation_hints
    objects=[{'object_id':7,'category':'bag','description':'x'*1000},
             {'object_id':3,'category':'unknown','description':'A narrow metal rim.'}]
    hints=annotation_hints(objects)
    assert hints['U1']=={'category':'bag','description':'x'*240}
    assert hints['U2']=={'description':'A narrow metal rim.'}


def test_final_reference_verdict_repairs_text_without_changing_goal(plan):
    member=deepcopy(plan['units'][0]);before=deepcopy(member['instruction_spec'])
    row=dict(unit_id=member['unit_id'],source_state='Selected cup on the upper left.',
             result_state='That cup is gone; nearby cup remains.',
             image_ok=True,scope_ok=True,operation_ok=True,reference_unique=True,
             final_ref='the topmost cup on the left')
    result=normalize_result({'units':[row]},[member])[member['unit_id']]
    assert result['decision']=='pass'
    assert member['instruction_spec']==dict(before,ref=row['final_ref'])
    assert member['instruction_noref']==render(before,False)
    member=deepcopy(plan['units'][0]);row['image_ok']=False
    assert normalize_result({'units':[row]},[member])[member['unit_id']]['decision']=='fail'
    assert not member.get('instruction_revision',{}).get('applied')


def test_add_audit_site_echo_renders_once_without_changing_blind_goal(plan):
    from sa1b_pipeline.multi_component_audit import expand_compact_result
    from sa1b_pipeline.instruction_protocol import render
    member=deepcopy(plan['units'][0])
    spec=deepcopy(member['instruction_spec']);spec.update(operation='add',ref='the green suitcase',payload='a luggage tag',attributes=[],placement={'anchor_mode':'support','relation':'hanging from','site':'handle'})
    member.update(operation='add',instruction_spec=spec,instruction=render(spec),instruction_noref=render(spec,False))
    raw={'units':[dict(unit_id=member['unit_id'],source_state='suitcase handle',result_state='tag attached',image_ok=True,scope_ok=True,operation_ok=True,reference_unique=True,final_ref='the handle of the green suitcase')]}
    expanded=expand_compact_result(raw,[member])['units'][0]
    assert expanded['ref']=='the green suitcase'
    assert 'handle of the handle' not in expanded['optimized_instruction']
    before=member['instruction_noref'];assert normalize_result(raw,[member])[member['unit_id']]['decision']=='pass'
    assert member['instruction_noref']==before
    raw['units'][0]['final_ref']='the strap of the green suitcase'
    assert expand_compact_result(raw,[member])['units'][0]['ref']=='the strap of the green suitcase'


def test_unverified_ref_repair_does_not_turn_good_pixels_into_image_damage(plan):
    member=deepcopy(plan['units'][0])
    before=deepcopy(member)
    raw={'units':[dict(unit_id=member['unit_id'],source_state='source cup',result_state='cup removed',
        image_ok=True,scope_status='matched',observed_operation='remove',reference_unique=False,
        ref_issue='Another cup still fits the proposed phrase.',final_ref='the small cup on the left')]}
    result=normalize_result(raw,[member])[member['unit_id']]
    assert result['decision']=='fail' and 'reference_not_unique' in result['failure_reasons']
    assert result['image_ok'] and not result['serious_defects']
    assert member==before


def test_twin_add_refresh_preserves_prior_clipping_reason(monkeypatch):
    from sa1b_pipeline import group_audit as audit
    monkeypatch.setattr(audit, "_member_audit_row", lambda *a, **kw: {})
    monkeypatch.setattr(audit, "deterministic_check", lambda row: {
        "decision": "pending_semantic_audit", "failure_reasons": []})
    member = {"unit_id": "U1", "deterministic": {
        "decision": "fail", "failure_reasons": ["dominant_addition_clipped_by_support"]}}
    group = {"operation": "add", "members": [member],
             "group_checks": {"deterministic": {"pass": False}}}
    assert audit.refresh_twin_add_checks(group) is False
    assert member["deterministic"]["failure_reasons"] == ["dominant_addition_clipped_by_support"]
    assert member["deterministic"]["decision"] == "fail"
