from copy import deepcopy
import threading

import numpy as np
from PIL import Image
import pytest

from test_multi_transactional import plan, Editor, Auditor
from sa1b_pipeline.multi_parallel import conflict_components, fuse_components, generate_parallel_multi_group
from sa1b_pipeline.group_generation import multi_support_preflight
from sa1b_pipeline.production_state import file_hash


def test_support_components_not_crop_boxes(plan, tmp_path):
    g = deepcopy(plan)
    multi_support_preflight(g, tmp_path/'preflight')
    # Disjoint writes despite overlapping read-only crop/context boxes.
    for u in g['units']: u['unit']['execution_mask_path'] = u['mask']
    assert conflict_components(g)[0] == [['U1'],['U2'],['U3'],['U4']]
    g['edit_dependencies'] = [['U1','U3']]
    assert conflict_components(g)[0] == [['U1','U3'],['U2'],['U4']]
    g['edit_dependencies'] = [['U4','U1']]
    with pytest.raises(ValueError,match='contradicts'):conflict_components(g)


def test_fusion_preserves_other_edits_and_never_blends_twice(plan, tmp_path):
    source = plan['source_image'];a=np.array(Image.open(source)); b=a.copy()
    a[10:40,10:40] = (110,120,130);b[10:40,41:71] = (200,0,0)
    paths=[tmp_path/'a.png',tmp_path/'b.png']
    for arr,p in zip([a,b],paths):Image.fromarray(arr).save(p)
    results=[{'current':str(p),'kept':[dict(plan['units'][i],support_mask=plan['units'][i]['mask'])]} for i,p in enumerate(paths)]
    out,proof=fuse_components(source,results,tmp_path/'fused')
    final=np.array(Image.open(out))
    assert tuple(final[20,20])==(110,120,130) # already-feathered pixel copied exactly once
    assert tuple(final[20,50])==(200,0,0)
    assert tuple(final[100,100])==(100,100,100)
    reverse,_=fuse_components(source,list(reversed(results)),tmp_path/'reverse')
    assert file_hash(out)==file_hash(reverse)
    with pytest.raises(ValueError,match='overlapping'):
        fuse_components(source,results+[results[0]],tmp_path/'overlap')
    a[100,100]=0;Image.fromarray(a).save(paths[0])
    with pytest.raises(ValueError,match='outside'):
        fuse_components(source,results,tmp_path/'leak')


def test_independent_inputs_failure_discard_and_zero_call_resume(plan,tmp_path,monkeypatch):
    # Keep full preflight except use source-only supports, so all four may run together.
    import sa1b_pipeline.multi_parallel as mod
    real=mod.multi_support_preflight
    def independent(g,root):
        result=real(g,root)
        for u in g['units']:u['unit']['execution_mask_path']=u['mask']
        return result
    monkeypatch.setattr(mod,'multi_support_preflight',independent)
    barrier=threading.Barrier(4);editors=[]
    class ConcurrentEditor(Editor):
        def generate(self,*args,**kwargs):
            barrier.wait(timeout=10)
            return super().generate(*args,**kwargs)
    def factory():
        e=ConcurrentEditor();editors.append(e);return e
    auditor=Auditor(fail=['U2']);root=tmp_path/'run'
    g=generate_parallel_multi_group(plan,root,factory,auditor)
    assert g['n_targets']==3 and g['joint_audit']['pass']
    assert g['parallel']['components']==[['U1'],['U2'],['U3'],['U4']]
    assert sum(e.calls for e in editors)==4 and len(auditor.calls)==5
    assert all(np.all(a==100) for e in editors for a in e.inputs)
    final=np.array(Image.open(g['joint_target']))
    assert tuple(final[20,50])==(100,100,100) # rejected red edit absent
    assert tuple(final[20,20])==(240,240,240)
    assert tuple(final[50,20])==(0,0,240)
    assert generate_parallel_multi_group(plan,root,factory,auditor)==g
    assert sum(e.calls for e in editors)==4 and len(auditor.calls)==5
    Image.new('RGB',(128,128),'black').save(g['joint_target'])
    with pytest.raises(ValueError,match='checkpoint image changed'):
        generate_parallel_multi_group(plan,root,factory,auditor)


def test_conflicting_targets_keep_serial_rollback(plan,tmp_path):
    plan['edit_dependencies'] = [['U2','U3']]
    auditor=Auditor(fail=['U2']);editors=[]
    def factory():
        e=Editor();editors.append(e);return e
    g=generate_parallel_multi_group(plan,tmp_path/'run',factory,auditor)
    assert g['parallel']['components']==[['U1','U2','U3','U4']]
    assert g['joint_audit']['pass']
    e=next(e for e in editors if e.calls)
    assert tuple(e.inputs[2][20,50])==(100,100,100)
    assert tuple(e.inputs[2][20,20])==(240,240,240)


def test_interrupted_component_resumes_without_regenerating(plan,tmp_path):
    plan['edit_dependencies'] = [['U2','U3']]
    e=Editor();a=Auditor(crash='U3');root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='transient'):
        generate_parallel_multi_group(plan,root,lambda:e,a)
    assert e.calls==3
    g=generate_parallel_multi_group(plan,root,lambda:e,a)
    assert e.calls==4 and g['joint_audit']['pass']
    assert a.calls.count('multi-test::U1')==1
    assert a.calls.count('multi-test::U2')==1
