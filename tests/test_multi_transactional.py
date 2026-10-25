from copy import deepcopy
import re

import numpy as np
from PIL import Image
import pytest

from sa1b_pipeline.group_planning import build_multi_group
from sa1b_pipeline.multi_transactional import generate_committed_multi_group
from sa1b_pipeline.group_schema import validate_multi_record
from sa1b_pipeline.group_production import accept, cell
from sa1b_pipeline.geometry import mask_geometry


class Editor:
    config = {'model': 'test-editor', 'steps': 40}
    last_generation_context = {}
    def __init__(self): self.inputs = []; self.calls = 0
    def generate(self, source, unit, seed=0):
        self.calls += 1
        self.inputs.append(np.array(Image.open(source)))
        result = self.inputs[-1].copy()
        mask = np.asarray(Image.open(unit['mask_path'])) > 0
        result[mask] = {'remove': (240, 240, 240), 'replace': (250, 0, 0),
                        'attribute': (0, 0, 240), 'add': (0, 240, 0)}[unit['operation']]
        return Image.fromarray(result)


class Auditor:
    config = {'model': 'test-auditor', 'contract': 'v5'}
    batch_size = 4
    def __init__(self, fail=(), crash=None, joint_fail=False):
        self.fail = set(fail); self.crash = crash; self.calls = []; self.joint_fail = joint_fail
    def call_batch(self, jobs):
        result = []
        for j in jobs:
            self.calls.append(j['job_id'])
            uid = j['job_id'].split('::')[-1]
            if uid == self.crash:
                self.crash = None
                raise RuntimeError('transient service error')
            ok = uid not in self.fail
            audit = dict(decision='pass' if ok else 'fail', operation_correct=ok,
                         target_done=ok, non_target_preserved=True, natural=True,
                         host_integrity=True, observed_change='visible changed surface',
                         failure_reasons=[] if ok else ['wrong edit'], evidence='visible')
            if 'observed_operation' in (j.get('output_schema') or {}).get('properties', {}):
                audit.update(observed_operation='replace' if uid == 'U2' else 'attribute',
                             source_state='old cup', result_state='new cup', changed_structure='visible change',
                             observed_location='selected cup', issue='none', ref_matches_mask=True,
                             location_correct=True)
            if '::' not in j['job_id']:
                audit.update(decision='fail' if self.joint_fail else 'pass', unique_refs=True,
                             operations_correct=True, all_targets_done=not self.joint_fail,
                             no_crosstalk=True, outside_preserved=True,
                             units=[dict(unit_id=u, source_entity='cup', ref_matches_mask=True,
                                         step_done=True, survives=True, natural=True, host_integrity=True,
                                         observed_change='visible change', issue='')
                                    for u in re.findall(r'(U\d+): object_id=', j['prompt'])])
            result.append({'job_id': j['job_id'], 'semantic_audit': audit})
        return result


@pytest.fixture
def plan(tmp_path):
    source = tmp_path/'source.png'; Image.new('RGB', (128,128), (100,100,100)).save(source)
    ops = ['remove', 'replace', 'attribute', 'add']
    objects, paths, geoms, targets = [], [], [], []
    for i, (x, y) in enumerate([(10,10),(41,10),(10,41),(41,41)]):
        m = np.zeros((128,128), np.uint8); m[y:y+30,x:x+30] = 255
        path = tmp_path/f'mask{i}.png'; Image.fromarray(m).save(path)
        objects.append({'object_id':str(i),'category':'cup','mask_path':str(path), 'metadata':{}})
        paths.append(str(path)); g = mask_geometry(m>0); g['point']=[x+15,y+15]; geoms.append(g)
        op = ops[i]
        targets.append(dict(unit_id=f'U{i+1}', object_id=str(i), operation=op, scope='cup',
                            ref_phrase=['the upper left cup','the upper right cup','the lower left cup','the lower right cup'][i],
                            payload='a square cup' if op=='replace' else 'a star decal' if op=='add' else '',
                            attributes=[{'property':'color','value':'blue'}] if op=='attribute' else [],
                            placement={'relation':'on','site':'','anchor_mode':'support'} if op=='add' else None,
                            target_part='',remove_mode='whole',qualifier='',evidence='visible'))
    raw = {'feasible':True,'targets':targets,'generation_order':['U1','U2','U3','U4']}
    return build_multi_group({'source_id':'source','image_path':str(source)},objects,raw,paths,
                             split='train',group_id='multi-test',geometry=geoms,
                             assignments={str(i):o for i,o in enumerate(ops)})


def test_failed_middle_edit_is_absent_from_later_input_and_final(plan, tmp_path):
    original = deepcopy(plan); editor = Editor(); auditor = Auditor(fail=['U2'])
    group = generate_committed_multi_group(plan,tmp_path/'run',editor,auditor)
    assert plan == original
    assert group['n_targets']==3 and group['generation_order']==['U1','U3','U4']
    assert group['operations']==['remove','attribute','add']
    assert group['joint_audit']['pass'] is True
    # Failed red U2 is never in C's input or the final image; A survives.
    assert tuple(editor.inputs[2][20,50])==(100,100,100)
    assert tuple(editor.inputs[2][20,20])==(240,240,240)
    final=np.array(Image.open(group['joint_target']))
    assert tuple(final[20,50])==(100,100,100)
    assert tuple(final[50,20])==(0,0,240)
    assert len(auditor.calls)==5 and editor.calls==4
    assert group['adjacent_pairs']==[]
    assert all('1' not in p['object_ids'] for p in group['adjacent_pairs'])
    validate_multi_record(group)
    assert cell(group,'multi')=='3:add+attribute+remove'
    assert accept([group],{'3:add+attribute+remove':1},'multi')==[group]
    assert accept([group],{'4:add+attribute+remove+replace':1},'multi')==[]


def test_resume_never_retries_a_rejected_edit_or_reaudits_committed_steps(plan,tmp_path):
    editor=Editor(); auditor=Auditor(fail=['U2'],crash='U3');root=tmp_path/'run'
    with pytest.raises(RuntimeError,match='transient'):
        generate_committed_multi_group(plan,root,editor,auditor)
    assert editor.calls==3
    group=generate_committed_multi_group(plan,root,editor,auditor)
    assert group['joint_audit']['pass']
    assert editor.calls==4  # Pending U3 render was already saved before the audit crash.
    assert auditor.calls.count('multi-test::U1')==1
    assert auditor.calls.count('multi-test::U2')==1
    before=(editor.calls,len(auditor.calls))
    assert generate_committed_multi_group(plan,root,editor,auditor)==group
    assert (editor.calls,len(auditor.calls))==before
    Image.new('RGB',(128,128),'black').save(group['joint_target'])
    with pytest.raises(ValueError,match='checkpoint image changed'):
        generate_committed_multi_group(plan,root,editor,auditor)


def test_one_or_zero_survivors_never_becomes_multi(plan,tmp_path):
    for failed in [('U1','U2','U4'),('U1','U2','U3','U4')]:
        auditor=Auditor(fail=failed)
        g=generate_committed_multi_group(plan,tmp_path/str(len(failed)),Editor(),auditor)
        assert g['n_targets']<2 and g['joint_audit']['pass'] is False
        assert 'multi-test' not in auditor.calls  # No pointless joint call.


def test_joint_failure_is_not_overridden_by_step_passes(plan,tmp_path):
    g=generate_committed_multi_group(plan,tmp_path/'run',Editor(),Auditor(joint_fail=True))
    assert g['n_targets']==4 and not g['joint_audit']['pass']


def test_distant_survivors_remain_valid_after_failed_edits(plan,tmp_path):
    # Move U4 far from U1; two quality-passing targets still form a valid group.
    mask=np.zeros((128,128),np.uint8);mask[95:120,95:120]=255
    Image.fromarray(mask).save(plan['units'][3]['mask'])
    auditor=Auditor(fail=['U2','U3'])
    g=generate_committed_multi_group(plan,tmp_path/'run',Editor(),auditor)
    assert g['n_targets']==2 and not g['adjacent_pairs'] and g['joint_audit']['pass']
    assert 'multi-test' in auditor.calls


def test_changed_plan_refuses_resume(plan,tmp_path):
    root=tmp_path/'run';generate_committed_multi_group(plan,root,Editor(),Auditor())
    plan['generation_order']=list(reversed(plan['generation_order']))
    with pytest.raises(ValueError,match='checkpoint/config changed'):
        generate_committed_multi_group(plan,root,Editor(),Auditor())


def test_retained_quota_ignores_stale_plan_metadata_but_export_rejects_it(plan,tmp_path):
    g=generate_committed_multi_group(plan,tmp_path/'run',Editor(),Auditor(fail=['U2']))
    g['operations']=plan['operations'][:3]
    assert cell(g,'multi')=='3:add+attribute+remove'
    with pytest.raises(ValueError,match='operations do not match'):
        validate_multi_record(g)


def test_killed_coordinator_cleans_its_service_process_group(tmp_path):
    import json
    import os
    import signal
    import subprocess
    import sys
    import time

    pidfile=tmp_path/'pids.json'
    child_code='''
import os,json,subprocess,sys
from pathlib import Path
from scripts.group_production.inference_worker import watch_parent
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path(sys.argv[2]).write_text(json.dumps([os.getpid(),child.pid]))
watch_parent(int(sys.argv[1]))
'''
    owner_code='''
import os,subprocess,sys,time
subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]],start_new_session=True)
time.sleep(60)
'''
    owner=subprocess.Popen([sys.executable,'-c',owner_code,child_code,str(pidfile)])
    service_pid=None
    def stopped(pid):
        from pathlib import Path
        path=Path(f'/proc/{pid}/stat')
        return not path.exists() or path.read_text().split()[2]=='Z'
    try:
        deadline=time.monotonic()+10
        while not pidfile.exists() and time.monotonic()<deadline:time.sleep(.05)
        pids=json.loads(pidfile.read_text());service_pid=pids[0]
        owner.kill();owner.wait()
        deadline=time.monotonic()+5
        while not all(stopped(p) for p in pids) and time.monotonic()<deadline:time.sleep(.05)
        assert all(stopped(p) for p in pids)
    finally:
        if owner.poll() is None:owner.kill();owner.wait()
        if service_pid:
            try:os.killpg(service_pid,signal.SIGKILL)
            except ProcessLookupError:pass


def test_add_site_refinement_does_not_redefine_source_adjacency(plan,tmp_path):
    from sa1b_pipeline.multi_transactional import _retained_adjacency
    g=generate_committed_multi_group(plan,tmp_path/'run',Editor(),Auditor())
    g['units']=[g['units'][1],g['units'][3]]
    original=g['units'][1]['mask_v1']
    tiny=np.zeros((128,128),np.uint8);tiny[56:58,56:58]=255
    path=tmp_path/'tiny_site.png';Image.fromarray(tiny).save(path)
    g['units'][1]['mask']=str(path)
    assert g['units'][1]['mask_v1']==original
    assert len(_retained_adjacency(g))==1


def test_zero_survivor_preview_shows_original_and_dropped_attempts(plan,tmp_path):
    from scripts.preview_group_cases import page
    from scripts.review_multi_batch import append_retention_evidence
    g=generate_committed_multi_group(plan,tmp_path/'run',Editor(),Auditor(fail=['U1','U2','U3','U4']))
    canvas=page(g,'multi',1,1,tmp_path)
    expanded=append_retention_evidence(canvas,g,'/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    assert expanded.height>canvas.height


def test_completed_cli_wave_does_not_start_gpu_services(tmp_path, monkeypatch):
    from scripts import run_multi_transactional as cli
    import sa1b_pipeline.multi_parallel as parallel
    from sa1b_pipeline.production_state import write_rows, atomic_json
    root=tmp_path/'run';plans=tmp_path/'plans.jsonl'
    write_rows(plans,[{'multi_id':'done','units':[]}])
    atomic_json(root/'generated/done/committed.json',{'sentinel':'validated by group loader'})
    calls=[]
    class Pool:
        def __init__(self,**kw):self.model=kw['model'];self.batch_size=kw.get('batch_size',4)
        def start(self):raise AssertionError('completed wave must not load GPU models')
        def close(self):pass
    monkeypatch.setattr(cli,'ServicePool',Pool)
    monkeypatch.setattr(cli,'RemoteAuditor',lambda p:object())
    def replay(g,*args,**kw):
        calls.append(g['multi_id'])
        return {**g,'n_targets':0,'operations':[],'joint_audit':{'pass':False,'status':'retained_contract_rejected'}}
    monkeypatch.setattr(parallel,'generate_parallel_multi_group',replay)
    cli.main(['--planned-manifest',str(plans),'--output-root',str(root),'--editor-model','editor',
              '--audit-model','audit','--editor-python','python','--audit-python','python'])
    assert calls==['done']  # Replay is still delegated to the validating group loader.
