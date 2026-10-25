from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from test_multi_transactional import plan
from sa1b_pipeline.instruction_protocol import make_spec,render
from sa1b_pipeline.group_audit import audit_twin_groups_joint
from sa1b_pipeline.twin_compact_audit import VERSION,PROMPT


@pytest.fixture
def twins(plan,tmp_path):
    members=deepcopy(plan['units'][:2])
    for i,m in enumerate(members):
        spec=make_spec('remove',m['ref'],'cup')
        out=tmp_path/f'edited{i}.png';Image.new('RGB',(128,128),'red' if i==0 else 'blue').save(out)
        m.update(operation='remove',instruction_spec=spec,instruction=render(spec),
                 instruction_noref=render(spec,False),edited_image=str(out))
    return dict(twin_id='compact-test',source_id='test',source_image=plan['source_image'],
        twin_contract_version='same-operation-adapted-3',operation='remove',members=members,
        group_checks={k:{'pass':True} for k in ('deterministic','cross_target_preservation','support_gap','blind_protocol')})


class Auditor:
    batch_size=8
    def __init__(self,group,fail=None,revise=False):self.group=group;self.fail=fail;self.revise=revise;self.jobs=[]
    def call_batch(self,jobs):
        self.jobs+=jobs;responses=[]
        for job in jobs:
            rows=[]
            for m in self.group['members']:
                row=dict(unit_id=m['unit_id'],image_ok=m['unit_id']!=self.fail,instruction_ok=True,
                    reference_unique=True,observed_operation=m['operation'],ref='',optimized_instruction='',
                    reason='The selected source item is removed; other objects are preserved.')
                if self.revise and m['unit_id']=='U1':
                    spec=deepcopy(m['instruction_spec']);spec['ref']='the topmost cup on the left'
                    row.update(ref=spec['ref'],optimized_instruction=render(spec),reason='Topmost distinguishes the upper cup from the lower cup.')
                rows.append(row)
            responses.append(dict(job_id=job['job_id'],semantic_audit={'units':rows},finish_reason='stop'))
        return responses


def test_compact_default_three_images_one_call_and_full_separate_results(twins,tmp_path):
    a=Auditor(twins);g=audit_twin_groups_joint([twins],a,tmp_path/'audit',exact_selector=True)[0]
    assert g['group_checks']['group_pass'] and g['group_checks']['audit_mode']==VERSION
    assert len(a.jobs)==1 and len(a.jobs[0]['images'])==3
    source,marked,result=map(Image.open,a.jobs[0]['images'])
    assert source.size==(128,128)
    assert marked.size==result.size==(1152,720)
    assert np.asarray(result)[200,900,0]>240 and np.asarray(result)[550,900,2]>240
    assert 'SOURCE / RESULT / BINARY MASK' not in a.jobs[0]['prompt']
    # Keep the tested baseline when shorter prompts admit severe failed edits.
    assert len(PROMPT.split())<600
    assert len(a.jobs[0]['output_schema']['properties']['units']['items']['properties'])==9


def test_one_failed_member_rejects_whole_group(twins,tmp_path):
    a=Auditor(twins,fail='U2');g=audit_twin_groups_joint([twins],a,tmp_path)[0]
    assert not g['group_checks']['group_pass']
    assert g['members'][0]['semantic_audit']['decision']=='pass'
    assert g['members'][1]['semantic_audit']['decision']=='fail'


def test_revision_preserves_member_goal_and_does_not_hide_image_failure(twins,tmp_path):
    before=twins['members'][0]['instruction_noref'];a=Auditor(twins,revise=True)
    g=audit_twin_groups_joint([twins],a,tmp_path)[0]
    assert g['group_checks']['group_pass']
    assert g['members'][0]['ref']=='the topmost cup on the left'
    assert g['members'][0]['instruction_noref']==before
    assert g['members'][0]['instruction_revision']['applied']


def test_generation_failure_does_not_trigger_semantic_call(twins,tmp_path):
    twins['group_checks']['deterministic']['pass']=False
    a=Auditor(twins);g=audit_twin_groups_joint([twins],a,tmp_path)[0]
    assert not a.jobs and not g['group_checks']['group_pass']


def test_group_program_gate_cannot_be_overridden_by_model(twins,tmp_path):
    twins['group_checks']['blind_protocol']['pass']=False
    a=Auditor(twins);g=audit_twin_groups_joint([twins],a,tmp_path)[0]
    assert not g['group_checks']['group_pass']


def test_result_crops_cover_the_full_write_support_and_preserve_source(twins,tmp_path):
    from sa1b_pipeline.twin_compact_audit import views
    from sa1b_pipeline.production_state import file_hash
    hashes={p:file_hash(p) for p in [twins['source_image']]+[m['edited_image'] for m in twins['members']]}
    images=views(twins,tmp_path/'views')
    boxes=json.loads((tmp_path/'views/result_crop_boxes.json').read_text())
    for m,b in zip(twins['members'],boxes):
        assert b['unit_id']==m['unit_id'] and b['image']==m['edited_image']
        mask=np.array(Image.open(m.get('support_mask') or m['mask']))>0
        ys,xs=np.where(mask)
        x0,y0,x1,y1=b['box']
        assert x0<=xs.min() and x1>xs.max() and y0<=ys.min() and y1>ys.max()
    assert {p:file_hash(p) for p in hashes}==hashes
    assert len(images)==3
