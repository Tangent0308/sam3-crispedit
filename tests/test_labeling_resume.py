import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
from PIL import Image
import pytest

from synthesis_pipeline.labeling_checkpoint import (
    CaseCheckpoints, atomic_json, file_digest, recover_rows, wait_workers,
)
from synthesis_pipeline.prepare_samtok_data import encode_rle, load_jsonl, write_jsonl


def inputs(tmp_path, count=3, empty=None):
    data=tmp_path/'data';(data/'sources').mkdir(parents=True)
    rows=[]
    for i in range(count):
        name=f'{i:06d}_gres_r{i}_m0_remove.png'
        mask=np.zeros((64,64),bool)
        if i!=empty:mask[20:40,20:40]=True
        Image.new('RGB',(64,64),(90+i,120,130)).save(data/'sources'/name)
        rows.append(dict(image=name,source_image=name,mask=encode_rle(mask),
                         task_type='remove',editing_instruction='',answer='object',mask_index=0,num_masks=1))
    write_jsonl(data/'annotations.jsonl',rows)
    return data,rows


def plan():
    return dict(decision='accept',reason='isolated object',support_check='coherent',
                target='central object',target_point=[460,460],relations=[],
                reconstruction='Continue the wall behind the object.')


def install_backend(monkeypatch,module,responses):
    calls=[]
    class Backend:
        def chat_batch(self,messages,**kwargs):
            calls.append(len(messages))
            response=next(responses)
            if isinstance(response,Exception):raise response
            return [response for _ in messages]
    backend=Backend()
    monkeypatch.setattr(module.vlm,'configure_backend',lambda *a,**k:None)
    monkeypatch.setattr(module.vlm,'get_backend',lambda:backend)
    monkeypatch.setattr(module.vlm,'shutdown_backend',lambda:None)
    return calls


def test_planner_empty_mask_and_partial_failure_resume(monkeypatch,tmp_path):
    from synthesis_pipeline import plan_removal_relations as module
    data,rows=inputs(tmp_path,4,empty=1);out=tmp_path/'plans'
    args=['planner','--data-root',str(data),'--out-root',str(out),'--policy','relations-v16','--batch-size','1']
    monkeypatch.setattr(sys,'argv',args)
    install_backend(monkeypatch,module,iter([json.dumps(plan()),'broken JSON',RuntimeError('injected outage')]))
    with pytest.raises(RuntimeError,match='injected outage'):module.main()
    before={r['image']:r for r in load_jsonl(out/'annotations.jsonl')}
    assert len(before)==3
    assert before[rows[1]['image']]['relation_status']=='invalid_input_empty_mask'
    assert before[rows[2]['image']]['relation_status']=='invalid_decision'
    calls=install_backend(monkeypatch,module,iter([json.dumps(plan()),json.dumps(plan())]))
    monkeypatch.setattr(sys,'argv',[*args,'--resume']);module.main()
    after={r['image']:r for r in load_jsonl(out/'annotations.jsonl')}
    assert len(after)==4 and calls==[1,1]
    assert after[rows[0]['image']]==before[rows[0]['image']]
    assert after[rows[2]['image']]['relation_status']=='accepted'
    # Fully completed resume makes zero calls, including the empty-mask input.
    monkeypatch.setattr(module.vlm,'configure_backend',lambda *a,**k:pytest.fail('unnecessary model load'))
    module.main()
    assert json.loads((out/'summary.json').read_text())['calls']==0
    # Same filename/annotation with changed source pixels invalidates only that plan.
    Image.new('RGB',(64,64),(10,20,30)).save(data/'sources'/rows[0]['source_image'])
    calls=install_backend(monkeypatch,module,iter([json.dumps(plan())]));module.main()
    assert calls==[1]


def test_checkpoint_rejects_changed_inputs_corruption_and_incomplete_write(tmp_path):
    row={'image':'case.png','instruction':'remove target'};image=tmp_path/'case.png'
    Image.new('RGB',(16,16)).save(image)
    checkpoint=CaseCheckpoints(tmp_path,{'steps':40})
    checkpoint.save(row,{'complete':True},[image],dependencies={'source':'a'})
    assert checkpoint.load(row,{'source':'a'})=={'complete':True}
    assert checkpoint.load(row,{'source':'b'}) is None
    assert checkpoint.load({**row,'instruction':'remove another target'},{'source':'a'}) is None
    assert CaseCheckpoints(tmp_path,{'steps':20}).load(row,{'source':'a'}) is None
    image.write_bytes(b'partial PNG')
    assert checkpoint.load(row,{'source':'a'}) is None
    (tmp_path/'checkpoints/case.png.json').write_text('{"signature":')
    assert checkpoint.load(row,{'source':'a'}) is None


def test_old_jsonl_preserves_valid_records_and_retries_partial_line(tmp_path):
    file=tmp_path/'annotations.jsonl'
    file.write_text('{"image":"a.png","value":1}\n{"image":"broken\n{"image":"b.png","value":2}\n')
    assert [r['image'] for r in recover_rows(file)]==['a.png','b.png']


def test_resolution_resume_reuses_complete_case_and_repairs_corrupt_support(monkeypatch,tmp_path):
    from synthesis_pipeline import resolve_removal_relations as module
    data,rows=inputs(tmp_path,2)
    rows=[{**r,'relation_plan':{**plan(),'instruction':'Remove the central object.'},
           'relation_status':'accepted'} for r in rows]
    write_jsonl(data/'annotations.jsonl',rows)
    out=tmp_path/'regions'
    args=['resolver','--data-root',str(data),'--out-root',str(out)]
    monkeypatch.setattr(sys,'argv',args);module.main()
    original=load_jsonl(out/'annotations.jsonl')
    monkeypatch.setattr(sys,'argv',[*args,'--resume']);module.main()
    assert json.loads((out/'summary.json').read_text())['reused']==2
    (out/'support'/rows[0]['image']).write_bytes(b'partial')
    module.main()
    assert json.loads((out/'summary.json').read_text())['reused']==1
    assert load_jsonl(out/'annotations.jsonl')==original


def test_editor_resume_checks_completion_and_repairs_only_corrupt_output(monkeypatch,tmp_path):
    from synthesis_pipeline import experiment_edit_quality as module
    from utils import qwen_pipeline_loader
    data,rows=inputs(tmp_path,2);out=tmp_path/'edit'
    monkeypatch.setitem(sys.modules,'inference_mydemo_qwen2511',types.SimpleNamespace(
        collect_crop_inputs=None,load_crop_records=None))
    monkeypatch.setitem(sys.modules,'utils.runner_qwen2511',types.SimpleNamespace(run_qwen_multi_branch=None))
    monkeypatch.setattr(qwen_pipeline_loader,'load_qwen21_omni_pipeline',lambda *a:object())
    monkeypatch.setattr(module.torch,'Generator',lambda **k:types.SimpleNamespace(manual_seed=lambda s:None))
    edited=[]
    def edit(pipe,source,row,*args,**kwargs):
        edited.append(row['image'])
        if len(edited)==2:raise RuntimeError('injected edit failure')
        return Image.new('RGB',source.size,(20,30,40))
    monkeypatch.setattr(module,'edit_context_crop',edit)
    args=['editor','--data-root',str(data),'--out-root',str(out),
          '--variant','context_grounded_v4_qwen21','--ids','0,1','--qwen21-backend','vllm-omni']
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(RuntimeError,match='injected edit failure'):module.main()
    final=out/'context_grounded_v4_qwen21/edited'
    first_sha=file_digest(final/rows[0]['image'])
    monkeypatch.setattr(sys,'argv',[*args,'--resume']);module.main()
    assert edited==[rows[0]['image'],rows[1]['image'],rows[1]['image']]
    assert file_digest(final/rows[0]['image'])==first_sha
    (final/rows[0]['image']).write_bytes(b'partial PNG')
    module.main()
    assert edited[-1]==rows[0]['image'] and len(edited)==4
    module.main();assert len(edited)==4


def test_audit_resume_keeps_valid_fail_retries_unparsed_and_changed_image(monkeypatch,tmp_path):
    from synthesis_pipeline import audit_removal_concise as module
    data,rows=inputs(tmp_path,2);out=tmp_path/'audit';edited=tmp_path/'edited';edited.mkdir()
    for r in rows:Image.new('RGB',(64,64),(20,30,40)).save(edited/r['image'])
    args=['audit','--worker','--data-root',str(data),'--edited-dir',str(edited),'--out-root',str(out),
          '--batch-size','1','--policy','completion-v5']
    verdict=json.dumps(dict(target_removed='fail',quality='pass',instruction_match='fail',reason='The target remains.'))
    monkeypatch.setattr(sys,'argv',args)
    calls=install_backend(monkeypatch,module,iter([verdict,'not JSON']));module.main()
    first=load_jsonl(out/'audit.jsonl')[0]
    calls=install_backend(monkeypatch,module,iter([verdict]))
    monkeypatch.setattr(sys,'argv',[*args,'--resume']);module.main()
    assert calls==[1] and load_jsonl(out/'audit.jsonl')[0]==first
    Image.new('RGB',(64,64),(200,100,50)).save(edited/rows[0]['image'])
    calls=install_backend(monkeypatch,module,iter([verdict]));module.main()
    assert calls==[1]
    monkeypatch.setattr(module.vlm,'configure_backend',lambda *a,**k:pytest.fail('unnecessary model load'))
    module.main()


def test_failed_later_worker_is_detected_without_waiting_for_busy_first(tmp_path):
    from contextlib import ExitStack
    with ExitStack() as stack:
        jobs=[]
        for i,script in enumerate(['import time;time.sleep(30)','raise SystemExit(3)']):
            log=stack.enter_context((tmp_path/f'{i}.log').open('w'))
            jobs.append((i,subprocess.Popen([sys.executable,'-c',script]),log))
        start=time.monotonic()
        with pytest.raises(RuntimeError,match='worker failures'):wait_workers(jobs,'test')
        assert time.monotonic()-start<8
        assert all(proc.poll() is not None for _,proc,_ in jobs)


def test_four_rank_resume_ignores_old_failure_markers_and_preserves_shards(tmp_path):
    data,rows=inputs(tmp_path,7);root=tmp_path/'run'
    common=[sys.executable,'-m','synthesis_pipeline.run_multinode_labeling',
        '--data-root',str(data),'--run-root',str(root),'--run-id','resume-test',
        '--local-test','--coordination-only','--gpus','0','--join-timeout','20']
    def launch(extra):
        jobs=[subprocess.Popen([*common,*extra,'--rank',str(i)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) for i in range(4)]
        for job in jobs:
            output=job.communicate(timeout=35)[0]
            assert job.returncode==0,output
    launch([])
    hashes=[file_digest(root/f'inputs/node{i}/annotations.jsonl') for i in range(4)]
    atomic_json(root/'control/node1.failed.json',{'error':'previous attempt failed'})
    launch(['--resume','--attempt-id','retry1'])
    assert (root/'control/node1.failed.json').is_file()
    assert (root/'attempts/retry1/control/finalize.ok.json').is_file()
    assert hashes==[file_digest(root/f'inputs/node{i}/annotations.jsonl') for i in range(4)]
    duplicate=subprocess.run([*common,'--resume','--attempt-id','retry1','--rank','0'],capture_output=True)
    assert duplicate.returncode!=0


def test_resume_refuses_changed_source_manifest(tmp_path):
    from synthesis_pipeline.run_multinode_labeling import prepare
    data,rows=inputs(tmp_path,2);root=tmp_path/'run'
    prepare(data,root,4)
    rows[0]['answer']='a different target';write_jsonl(data/'annotations.jsonl',rows)
    with pytest.raises(ValueError,match='differs'):prepare(data,root,4,resume=True)


def test_model_staging_resume_repairs_corruption_without_copying_valid_file(tmp_path):
    from synthesis_pipeline.stage_labeling_model import stage
    source=tmp_path/'model';source.mkdir()
    (source/'model_index.json').write_text('{}');(source/'weights').write_bytes(b'valid')
    cache=tmp_path/'cache';stage(source,cache)
    mtime=(cache/'model_index.json').stat().st_mtime_ns
    (cache/'weights').write_bytes(b'bad')
    stage(source,cache,resume=True)
    assert (cache/'weights').read_bytes()==b'valid'
    assert (cache/'model_index.json').stat().st_mtime_ns==mtime


def test_bootstrap_resume_environment_gate_ignores_old_failures(tmp_path):
    import os
    script=(Path(__file__).resolve().parents[1]/'scripts/labeling/bootstrap_arnold_4node.sh').read_text()
    check=script.split('check_peers() {',1)[1].split('\n}',1)[0]
    gate=script.split('# Do not stage tens of GB',1)[1].split('export DIFFUSION_ATTENTION_BACKEND',1)[0].split('\n',1)[1]
    attempt=tmp_path/'attempts/retry1'
    atomic_json(tmp_path/'control/node0.failed.json',{'error':'old crash'})
    for rank in (1,2,3):atomic_json(attempt/f'control/environment.node{rank}.ok.json',{'ready':True})
    result=subprocess.run(['bash','-c','set -euo pipefail\ncheck_peers() {'+check+'\n}\n'+gate],
        env={**os.environ,'SAMTOK_RUN_ROOT':str(tmp_path),'SAMTOK_CONTROL_ROOT':str(attempt),'ARNOLD_ID':'0'},
        capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert (attempt/'control/environment.node0.ok.json').is_file()
    assert not (tmp_path/'control/environment.node0.ok.json').exists()
