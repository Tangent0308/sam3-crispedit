import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scaleedit import distributed as dist, runner
from scaleedit.validation import validate_shard


def test_balanced_complete_disjoint():
    weights = {'a': 10, 'b': 9, 'c': 8, 'd': 7, 'e': 3, 'f': 2, 'g': 0}
    groups, loads = dist.balanced(weights, weights, 4)
    assert sorted(n for g in groups for n in g) == sorted(weights)
    assert sum(loads) == 39 and max(loads) - min(loads) <= 2


def test_publish_resume_and_conflict(tmp_path):
    source, target = tmp_path/'source', tmp_path/'merged/file'
    source.write_text('original')
    dist.publish(source, target)
    dist.publish(source, target)
    other = tmp_path/'other'
    other.write_text('different')
    with pytest.raises(FileExistsError):
        dist.publish(other, target)
    assert target.read_text() == 'original'


def test_peer_failure_scoped_to_attempt(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path, control=tmp_path/'control/retry1', attempt='retry1')
    dist.atomic_json(tmp_path/'control/initial/node0.failed', {})
    dist.check_peers(args)
    dist.atomic_json(args.control/'node2.failed', {'error':'simulated OOM'})
    with pytest.raises(RuntimeError, match='simulated OOM'):
        dist.wait_for(args, tmp_path/'absent')


def test_bootstrap_failure_propagates(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path, control=tmp_path/'control/initial', attempt='initial')
    dist.atomic_json(tmp_path/'bootstrap_control/initial/node0.failed', {'error':'installation'})
    with pytest.raises(RuntimeError, match='installation'):
        dist.check_peers(args)


def test_native_coverage_verdict_and_identity(tmp_path):
    src = tmp_path/'source.parquet'
    pq.write_table(pa.Table.from_pylist([dict(sample_id='a',final_instruction='edit',final_task='color_change')]),src)
    out = tmp_path/'output.parquet'
    good = dict(row_idx=0,sample_id='a',final_instruction='edit',verdict='PASS',keep=True,error='')
    pq.write_table(pa.Table.from_pylist([good]),out)
    assert validate_shard(src,out,[0],'quality')['PASS']==1
    for change in ({'sample_id':'b'},{'keep':False},{'row_idx':1},{'verdict':'REVIEW'}):
        pq.write_table(pa.Table.from_pylist([{**good,**change}]),out)
        with pytest.raises(ValueError):
            validate_shard(src,out,[0],'quality')


def test_sparse_double_pass_and_missing_manifest(tmp_path):
    args = SimpleNamespace(run_dir=tmp_path)
    name='part-a.parquet'
    plan={'shards':{name:{'indices':[0,1,2]}}}
    for stage,rows in [('quality',[(0,True),(1,False),(2,True)]),('scene',[(0,False),(2,True)])]:
        path=tmp_path/stage/'manifest'/name
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([{'row_idx':i,'keep':k} for i,k in rows]),path)
    assert dist.eligible(args,plan,'quality',name)==[0,1,2]
    assert dist.eligible(args,plan,'scene',name)==[0,2]
    assert dist.eligible(args,plan,'grounding',name)==[2]
    assert dist.eligible(args,plan,'mask',name)==[2]
    pq.write_table(pa.Table.from_pylist([{'row_idx':0,'keep':False}]),tmp_path/'scene/manifest'/name)
    with pytest.raises(ValueError,match='coverage'):
        dist.eligible(args,plan,'mask',name)


def test_command_uses_current_native_runner_and_stable_upstreams(tmp_path):
    args=SimpleNamespace(source_dir=tmp_path/'source',run_dir=tmp_path/'run',devices='0,1',
                         model_path='Qwen',checkpoint_path='sam.pt',grounding_tp=2,
                         batch_size=4,grounding_batch_size=4)
    cmd=dist.stage_command(args,'mask',tmp_path/'work',tmp_path/'selection.json')
    assert cmd[cmd.index('--tensor-parallel-size')+1]=='1'
    assert cmd[cmd.index('--grounding-dir')+1]==str(tmp_path/'run/grounding')
    assert all('crispedit_' not in Path(v).name for v in cmd)


def test_source_plan_snapshot_and_selection(tmp_path):
    src=tmp_path/'source'; src.mkdir()
    pq.write_table(pa.Table.from_pylist([dict(sample_id='a',final_instruction='edit',final_task='color_change',source_image=b'a',edited_image=b'b')]),src/'part-a.parquet')
    sel=tmp_path/'selection.json'; dist.atomic_json(sel,{'cases':[{'shard':'part-a.parquet','row_idx':0}]})
    args=SimpleNamespace(source_dir=src,selection_file=sel,nodes=4,model_path='q',checkpoint_path='s',
                         batch_size=4,grounding_tp=2,grounding_batch_size=4,local_test=True)
    p=dist.source_plan(args)
    assert p['shards']['part-a.parquet']['rows']==1
    dist.atomic_json(sel,{'cases':[{'shard':'part-a.parquet','row_idx':2}]})
    with pytest.raises(ValueError,match='outside'):
        dist.source_plan(args)


def test_mask_validation_rejects_png_rle_disagreement(tmp_path):
    import io
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils
    source = tmp_path/'source.parquet'
    image=io.BytesIO(); Image.new('RGB',(12,8)).save(image,format='PNG')
    pq.write_table(pa.Table.from_pylist([dict(sample_id='a',final_instruction='edit',
                    final_task='color_change',source_image=image.getvalue())]),source)
    binary=np.zeros((8,12),np.uint8); binary[2:5,3:7]=1
    encoded=mask_utils.encode(np.asfortranarray(binary))
    png=io.BytesIO(); Image.fromarray(binary*255).save(png,format='PNG')
    row=dict(row_idx=0,sample_id='a',final_instruction='edit',qc_flag='OK',error='',
             mask_png=png.getvalue(),mask_sum=12,mask_height=8,mask_width=12,
             instance_masks=[dict(area=12,rle_size=encoded['size'],rle_counts=encoded['counts'].decode())])
    output=tmp_path/'output.parquet'; pq.write_table(pa.Table.from_pylist([row]),output)
    assert validate_shard(source,output,[0],'mask')['nonempty_masks']==1
    row['mask_sum']=11; pq.write_table(pa.Table.from_pylist([row]),output)
    with pytest.raises(ValueError,match='union'):
        validate_shard(source,output,[0],'mask')


def test_cluster_preflight_rejects_same_host_or_different_code():
    from scripts.preflight_scaleedit_env import validate_cluster_reports
    reports=[dict(hostname=f'node{i}',commit='a',code_sha256='b',versions={},vllm_probe=True,
                  gpus=8,base_python='/opt/python',python='/opt/env/bin/python') for i in range(4)]
    validate_cluster_reports(reports)
    reports[1]['hostname']='node0'
    with pytest.raises(ValueError,match='distinct'):
        validate_cluster_reports(reports)
    reports[1]['hostname']='node1'; reports[1]['code_sha256']='c'
    with pytest.raises(ValueError,match='code_sha256'):
        validate_cluster_reports(reports)


def test_running_stage_exits_when_peer_fails(tmp_path):
    import sys
    import time
    args=SimpleNamespace(run_dir=tmp_path,control=tmp_path/'control/initial',attempt='initial',rank=0)
    (tmp_path/'logs').mkdir()
    dist.atomic_json(args.control/'pipeline.node2.failed',{'error':'worker crashed'})
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='worker crashed'):
        dist.execute(args,[sys.executable,'-c','import time; time.sleep(120)'],'quality')
    assert time.monotonic()-started<20
