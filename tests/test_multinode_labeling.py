import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from synthesis_pipeline.run_multinode_labeling import (
    PROFILE, atomic, merge, pipeline_command, prepare, read_rows, split_sources,
    wait_for, write_rows, run_stage,
)


def row(i, source=None):
    return dict(image=f'{i:03d}_gres_r{i}_m0_remove.png', source_image=source or f's{i}.png',
                task_type='remove', mask={'size':[2,2], 'counts':'013'}, answer='target')


def test_group_partition_exact_once_and_preserves_all_regions():
    rows=[row(0,'same.png'),row(1),row(2,'same.png'),row(3),row(4)]
    shards=split_sources(rows,4)
    assert sorted(r['image'] for s in shards for r in s)==sorted(r['image'] for r in rows)
    owners=[i for i,s in enumerate(shards) for r in s if r['source_image']=='same.png']
    assert len(set(owners))==1
    assert rows[0]==shards[0][0]


@pytest.mark.parametrize('change',[
    {'task_type':'add'}, {'answer':'No target'}, {'image':'../escape.png'},
    {'source_image':'../source.png'}, {'mask':None},
])
def test_rejects_unprepared_and_unsafe_input(change):
    with pytest.raises(ValueError):split_sources([{**row(0),**change}],4)


def test_duplicate_numeric_ids_rejected_before_existing_id_filters():
    with pytest.raises(ValueError):split_sources([row(0),{**row(1),'image':'000_other.png'}],4)


def test_unchanged_policy_commands():
    cmd=pipeline_command(Path('/data'),Path('/out'),'0,1')
    assert '--thinking' in cmd and '--ground-keeps' in cmd
    assert cmd[cmd.index('--policy')+1]=='relations-v16'
    assert cmd[cmd.index('--remove-composition-policy')+1]=='adaptive-remove-v5'
    assert PROFILE['steps']==40 and PROFILE['seed']==0
    assert '--qwen21-target-guide' not in cmd


def test_failure_marker_wins_over_success(tmp_path):
    atomic(tmp_path/'control/node2.failed.json',{'error':'test'})
    atomic(tmp_path/'ready.json',{'ready':True})
    with pytest.raises(RuntimeError,match='Peer failed'):wait_for([tmp_path/'ready.json'],tmp_path,1)


def test_merge_keeps_no_output_and_fails_missing_audit(tmp_path):
    root=tmp_path/'run';data=tmp_path/'data';(data/'sources').mkdir(parents=True)
    r=row(0);(data/'sources'/r['source_image']).touch();write_rows(data/'annotations.jsonl',[r])
    prepare(data,root,4)
    for rank in range(4):
        out=root/'nodes'/f'node{rank}'
        if rank==0:
            write_rows(out/'pipeline/relations/annotations.jsonl',[r])
            write_rows(out/'pipeline/regions/resolution.jsonl',[{'image':r['image'],'status':'defer_support_conflict'}])
        write_rows(out/'audit/audit.jsonl',[])
    result=merge(root,4)
    assert result['decisions']=={'no_output':1}
    assert not read_rows(root/'results/model_pass.jsonl')
    out=root/'nodes/node0'
    write_rows(out/'pipeline/regions/resolution.jsonl',[{'image':r['image'],'status':'accepted'}])
    write_rows(out/'pipeline/editing/context_grounded_v4_qwen21/annotations.jsonl',[r])
    with pytest.raises(ValueError,match='Incomplete stage coverage'):merge(root,4)


def test_merge_requires_existing_output_and_preserves_evidence(tmp_path):
    root=tmp_path/'run';data=tmp_path/'data';(data/'sources').mkdir(parents=True)
    r=row(0);(data/'sources'/r['source_image']).touch();write_rows(data/'annotations.jsonl',[r])
    prepare(data,root,4)
    out=root/'nodes/node0';final=out/'pipeline/editing/context_grounded_v4_qwen21'
    write_rows(out/'pipeline/relations/annotations.jsonl',[r])
    write_rows(out/'pipeline/regions/resolution.jsonl',[{'image':r['image'],'status':'accepted'}])
    write_rows(final/'annotations.jsonl',[r])
    write_rows(out/'audit/audit.jsonl',[{'image':r['image'],'decision':'pass','raw_response':'complete reason'}])
    for rank in range(1,4):write_rows(root/'nodes'/f'node{rank}'/'audit/audit.jsonl',[])
    with pytest.raises(FileNotFoundError):merge(root,4)
    (final/'edited').mkdir();(final/'edited'/r['image']).touch()
    assert merge(root,4)['decisions']=={'pass':1}
    candidate=read_rows(root/'results/model_pass.jsonl')[0]
    assert candidate['quality_label']=='model_pass_not_human_verified'
    assert candidate['original_mask']==r['mask']
    assert read_rows(root/'results/audit.jsonl')[0]['raw_response']=='complete reason'


def test_peer_failure_terminates_local_stage(tmp_path):
    atomic(tmp_path/'control/node1.failed.json',{'error':'peer failure'})
    start=time.monotonic()
    with pytest.raises(RuntimeError,match='Peer failed'):
        run_stage([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'stage.log',tmp_path,60)
    assert time.monotonic()-start<10


def test_model_staging_byte_identity_and_no_overwrite(tmp_path):
    from synthesis_pipeline.stage_labeling_model import stage
    source=tmp_path/'source';source.mkdir()
    (source/'model_index.json').write_text('{}')
    (source/'weights.safetensors').write_bytes(b'test binary contents')
    dest=tmp_path/'cache';records=stage(source,dest)
    assert len(records)==2
    assert (dest/'weights.safetensors').read_bytes()==(source/'weights.safetensors').read_bytes()
    assert json.loads((dest/'staging_manifest.json').read_text())['byte_verified']
    with pytest.raises(FileExistsError):stage(source,dest)
    with pytest.raises(ValueError):stage(source,source/'nested')


def test_four_rank_coordination_and_duplicate_launch(tmp_path):
    data=tmp_path/'data';(data/'sources').mkdir(parents=True)
    rows=[row(i,'same.png' if i<2 else None) for i in range(7)]
    for r in rows:(data/'sources'/r['source_image']).touch()
    write_rows(data/'annotations.jsonl',rows)
    jobs=[]
    common=[sys.executable,'-m','synthesis_pipeline.run_multinode_labeling',
            '--data-root',str(data),'--run-root',str(tmp_path/'run'),'--run-id','test',
            '--local-test','--coordination-only','--gpus','0','--join-timeout','30']
    for rank in range(4):jobs.append(subprocess.Popen([*common,'--rank',str(rank)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True))
    for job in jobs:
        output=job.communicate(timeout=45)[0]
        assert job.returncode==0,output
    assert json.loads((tmp_path/'run/reports/final.json').read_text())['generated']==0
    assert subprocess.run([*common,'--rank','0'],capture_output=True).returncode!=0
