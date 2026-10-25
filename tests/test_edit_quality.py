import numpy as np
import pytest
from PIL import Image
from copy import deepcopy
from sa1b_pipeline.edit_quality import contact_metadata, contact_support, paste_diagnostics
from sa1b_pipeline.group_audit import _enforce_multi_integrity
from sa1b_pipeline.operation_evidence import normalize_audit, CHECKS
from sa1b_pipeline.multi_parallel import select_valid_component_subset, fuse_components
from test_multi_transactional import plan


def test_contact_is_on_host_not_neighbor():
    m=np.zeros((200,300),bool);m[50:100,40:100]=1
    c=contact_metadata([.2,.35],m)
    assert m[c['point'][1],c['point'][0]]
    with pytest.raises(ValueError,match='outside'):contact_metadata([.8,.8],m)
    assert contact_metadata(None,m) is None
    for bad in [[float('nan'),.4],[True,.4],[1.2,.4]]:
        with pytest.raises(ValueError):contact_metadata(bad,m)


def test_attachment_envelope_has_room_outside_thin_host():
    m=np.zeros((200,200),bool);m[20:180,98:102]=1
    c=contact_metadata([.5,.5],m);s=contact_support(m,c)
    assert s[100,80] and s[100,120] and not s[10,10]
    assert np.count_nonzero(s & ~m)>0


def test_dominant_addition_clipping_is_rejected_but_minor_drift_is_not():
    src=np.zeros((100,100,3),np.uint8);raw=src.copy();raw[30:70,30:70]=200
    s=np.zeros((100,100),bool);s[:50,:]=1
    d=paste_diagnostics(src,raw,s)
    assert d['cross_boundary_components']==1 and d['max_discarded_fraction']>.4
    assert d['dominant_change_clipped'] and not d['warning_only']
    raw[10:30,2:25]=200
    s[:,:25]=True
    small=np.zeros_like(src);small[40:55,60:70]=200;small[10:40,2:35]=200
    assert not paste_diagnostics(src,small,s)['dominant_change_clipped']
    assert paste_diagnostics(src,raw,np.ones_like(s))['cross_boundary_components']==0


def test_minor_warnings_do_not_override_core_pass_but_serious_defects_do():
    audit=dict(decision='pass',host_integrity=True,natural=True,ref_matches_mask=True,
               location_correct=True,addition_is_new=True,operation_correct=True,
               warnings=['slight edge softness'],serious_defects=[])
    assert _enforce_multi_integrity(audit)['decision']=='pass'
    audit['serious_defects']=['tag lower half is cut off']
    assert _enforce_multi_integrity(audit)['decision']=='fail'
    observed=dict(observed_operation='attribute',source_state='two sticks',result_state='four rods',
                  changed_structure='extra rods',observed_location='selected sticks',issue='quantity changed',
                  serious_defects=['two sticks became four rods'],**{k:True for k in CHECKS})
    assert normalize_audit(observed,'attribute')['decision']=='fail'


def test_subset_keeps_valid_repeated_types_without_extra_render(plan,tmp_path):
    # Repeated operation types are valid: retain every otherwise usable target.
    g=deepcopy(plan)
    for u in g['units']:
        u['operation']='attribute' if u['unit_id']=='U2' else u['operation']
        u['instruction_spec']['operation']=u['operation']
    # Keep metadata realistic by rendering its canonical pair.
    from sa1b_pipeline.instruction_protocol import render
    u=g['units'][1];u['instruction_spec']['payload']='';u['instruction_spec']['attributes']=[{'property':'color','value':'blue'}]
    u['instruction'],u['instruction_noref']=(render(u['instruction_spec'],has_ref=True),render(u['instruction_spec'],has_ref=False))
    results=[]
    for i,u in enumerate(g['units']):
        a=np.array(Image.open(g['source_image']));m=np.array(Image.open(u['mask']))>0;a[m]=(i+1)*40
        p=tmp_path/f'{i}.png';Image.fromarray(a).save(p);u.update(step_output_image=str(p),support_mask=u['mask'],blind_eligible=True)
        results.append(dict(kept=[u],current=str(p),decisions=[]))
    selected,dropped=select_valid_component_subset(g,results)
    assert dropped == []
    kept=[u for r in selected for u in r['kept']]
    assert len(kept)==4 and len({u['operation'] for u in kept})==3
    output,_=fuse_components(g['source_image'],selected,tmp_path/'fuse')
    pixels=np.array(Image.open(output))
    for i,u in enumerate(g['units']):
        m=np.array(Image.open(u['mask']))>0
        assert np.all(pixels[m]==(i+1)*40)
    # A dependent serial output cannot be retained after deleting an earlier edit.
    serial=[dict(kept=g['units'],current=g['units'][-1]['step_output_image'],decisions=[])]
    chosen,dropped=select_valid_component_subset(g,serial)
    ids=[u['unit_id'] for r in chosen for u in r['kept']]
    assert ids==g['generation_order'][:len(ids)]


def test_batch_twins_entry_uses_full_contact_schema(monkeypatch):
    import sys,types
    from types import SimpleNamespace
    from scripts.run_group_pipeline import _planner
    import sa1b_pipeline.mllm as module
    monkeypatch.setattr(module,'VLLMSelector',lambda *a,**k:SimpleNamespace(params=SimpleNamespace()))
    fake=types.ModuleType('vllm.sampling_params');fake.StructuredOutputsParams=lambda **kw:kw
    monkeypatch.setitem(sys.modules,'vllm.sampling_params',fake)
    args=SimpleNamespace(planner_model='test',planner_backend='vllm',planner_tp=1,max_new_tokens=100,
                         planner_batch_size=2,structured_planning_json=True,kind='twins',twin_planning_style='concise-27b-v8')
    schema=_planner(args).params.structured_outputs['json']
    assert 'contact_xy' in schema['properties']['members']['items']['required']


def test_explicit_attribute_count_mismatch_cannot_be_downgraded_to_warning():
    audit=dict(observed_operation='attribute',source_state='two rods',result_state='three rods',
               changed_structure='metallic',observed_location='same place',issue='minor',
               source_count=2,result_count=3,warnings=['extra rod'],serious_defects=[],**{k:True for k in CHECKS})
    normalized=normalize_audit(audit,'attribute')
    assert normalized['decision']=='fail' and 'attribute_instance_count_changed' in normalized['failure_reasons']
