"""Run a frozen cohort with durable logs; no resampling to hide rejections."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

VLM='/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python'
EDIT='/opt/tiger/tanyue/.venvs/mirage_official/bin/python'
EDIT_QWEN21='/opt/tiger/tanyue/.venvs/qwen_image_21/bin/python'


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--editor-gpus',default='0,1,2,5,6,7')
    p.add_argument('--planner-gpu',default='3');p.add_argument('--sam-gpu',default='4')
    p.add_argument('--reuse-completed-plan',action='store_true',help='Explicitly reuse a complete, separately timed planning stage')
    p.add_argument('--reuse-completed-regions',action='store_true',help='Reuse complete source-mask preparation; never reuse diffusion outputs')
    p.add_argument('--scope-preflight', action='store_true',
                   help='Experimental source-scope preflight plus semantic anchor/material verification before diffusion')
    p.add_argument('--dataset-mask-plan',action='store_true',help='MLLM planning/review on fixed dataset masks; skip all source SAM calls')
    p.add_argument('--editor-backend',choices=['qwen2511','qwen21'],default='qwen2511',
                   help='Keep the cohort pipeline fixed while selecting the image editor backend')
    p.add_argument('--qwen21-prompt-policy',choices=['legacy','typed-v1','remove-evidence-v1'],default='typed-v1',
                   help='Qwen2.1 quality candidate; legacy explicitly reproduces the old prompt')
    p.add_argument('--composition-policy',choices=['legacy','guarded-v1'],default=None,
                   help='Default: guarded-v1 for Qwen2.1, legacy for Qwen2511')
    p.add_argument('--remove-composition-policy',choices=['legacy','adaptive-remove-v1','adaptive-remove-v2'],default=None,
                   help='Default: adaptive-remove-v2 for Qwen2.1, legacy for Qwen2511')
    p.add_argument('--editor-model-id',help='Optional verified node-local checkpoint directory')
    a=p.parse_args()
    if a.composition_policy is None:
        a.composition_policy='guarded-v1' if a.editor_backend=='qwen21' else 'legacy'
    if a.remove_composition_policy is None:
        a.remove_composition_policy='adaptive-remove-v2' if a.editor_backend=='qwen21' else 'legacy'
    logs=a.root/'logs';logs.mkdir(exist_ok=True)
    times={};started=time.time()
    def spawn(name,python,module,options,gpu):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'8'}
        if python==VLM:env['PATH']=str(Path(VLM).parent)+':'+env['PATH']
        command=[python,'-m',module,*map(str,options)]
        handle=(logs/f'{name}.log').open('w')
        print(json.dumps(dict(stage=name,command=command,gpu=gpu,start=time.time())),flush=True)
        return subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT),handle,time.time()
    def finish(name,job):
        proc,handle,t=job;code=proc.wait();handle.close();times[name]=time.time()-t
        (logs/'timing.json').write_text(json.dumps(dict(stages=times,wall_seconds=time.time()-started),indent=2))
        print(json.dumps(dict(stage=name,exit_code=code,seconds=times[name])),flush=True)
        if code:raise RuntimeError(f'{name} failed: {logs/name}.log')
    ids=','.join(str(int(r['image'].split('_')[0])) for r in read(a.root/'annotations.jsonl'))
    if a.dataset_mask_plan:
        if a.scope_preflight or a.reuse_completed_plan or a.reuse_completed_regions:
            p.error('--dataset-mask-plan cannot be combined with legacy planning/refinement flags')
        finish('dataset_planning',spawn('dataset_planning',VLM,'synthesis_pipeline.plan_dataset_regions',[
            '--data-root',a.root,'--out-root',a.root],a.planner_gpu))
    elif a.reuse_completed_plan:
        summary=json.loads((a.root/'plan/summary.json').read_text())
        original=read(a.root/'annotations.jsonl');planned=read(a.root/'plan/annotations.jsonl')
        if summary['input_cases']!=len(original) or summary['accepted']!=len(planned):
            raise ValueError('Cannot reuse an incomplete planning stage')
        wanted={r['image'] for r in original}
        if not {r['image'] for r in planned}.issubset(wanted):raise ValueError('Planning cohort mismatch')
        print(json.dumps(dict(stage='planning',reused=True,original_timing=summary)),flush=True)
    else:
        finish('planning',spawn('planning',VLM,'synthesis_pipeline.replan_edit_regression',[
            '--data-root',a.root,'--out-root',a.root/'plan','--ids',ids,'--vlm','qwen38-vllm',
            '--model-id','/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B'],a.planner_gpu))
    planned_root=a.root/'plan'
    if a.scope_preflight:
        finish('scope_preflight',spawn('scope_preflight',VLM,'synthesis_pipeline.ground_planning_scope',[
            '--data-root',planned_root,'--out-root',a.root/'scope'],a.planner_gpu))
        planned_root=a.root/'scope'
        if not read(planned_root/'annotations.jsonl'):
            raise RuntimeError('No plans passed source scope; cohort and rejections preserved')
    if a.dataset_mask_plan:
        print(json.dumps(dict(stage='source_regions',source_sam_calls=0,policy='original_dataset')),flush=True)
    elif a.reuse_completed_regions:
        planned=read(planned_root/'annotations.jsonl');refined=read(a.root/'regions/annotations.jsonl')
        if sorted(r['image'] for r in planned)!=sorted(r['image'] for r in refined):
            raise ValueError('Cannot reuse incomplete or different source-mask cohort')
        summary=json.loads((a.root/'regions/summary.json').read_text())
        print(json.dumps(dict(stage='source_regions',reused=True,original_timing=summary)),flush=True)
    else:
        finish('source_regions',spawn('source_regions','/usr/bin/python','synthesis_pipeline.refine_samtok_regions',[
            '--data-root',planned_root,'--out-root',a.root/'regions',
            *(['--verify-original-scope'] if a.scope_preflight else [])],a.sam_gpu))
    valid=[r for r in read(a.root/'regions/annotations.jsonl') if r['region_contract']['status']!='unresolved']
    ids=','.join(str(int(r['image'].split('_')[0])) for r in valid)
    if not valid:raise RuntimeError('No executable regions; cohort preserved')
    variant='context_grounded_v4_qwen21' if a.editor_backend=='qwen21' else 'context_grounded_v4'
    editor_python=EDIT_QWEN21 if a.editor_backend=='qwen21' else EDIT
    model_id=(
        '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1'
        if a.editor_backend=='qwen21' else
        '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-Edit-2511')
    model_id=a.editor_model_id or model_id
    options=['--data-root',a.root/'regions','--out-root',a.root/'generation',
             '--variant',variant,'--model-id',model_id,'--ids',ids,
             '--qwen21-prompt-policy',a.qwen21_prompt_policy,
             '--remove-composition-policy',a.remove_composition_policy]
    finish('manifest',spawn('manifest','/usr/bin/python','synthesis_pipeline.experiment_edit_quality',options+['--manifest-only'],''))
    gpus=a.editor_gpus.split(',');jobs=[]
    for shard,gpu in enumerate(gpus):
        name=f'editing_{shard}'
        jobs.append((name,spawn(name,editor_python,'synthesis_pipeline.experiment_edit_quality',
            options+['--shard',shard,'--shards',len(gpus)],gpu)))
    # Join all workers even if one fails, then propagate the error.
    errors=[]
    for name,job in jobs:
        try:finish(name,job)
        except RuntimeError as e:errors.append(str(e))
    if errors:raise RuntimeError('; '.join(errors))
    raw=a.root/'generation'/variant
    finish('composition',spawn('composition','/usr/bin/python','synthesis_pipeline.compose_segmented_replacements',[
        '--data-root',a.root/'regions','--raw-root',raw,'--out-root',a.root/'composed','--include-add','--carry-others',
        '--composition-policy',a.composition_policy],a.sam_gpu))
    finish('snapshot',spawn('snapshot','/usr/bin/python','synthesis_pipeline.assemble_fresh_snapshot',[
        '--raw-root',raw,'--composed-root',a.root/'composed','--out-root',a.root/'final'],''))
    for split in ['dev','holdout']:
        wanted={r['image'] for r in read(a.root/split/'annotations.jsonl')}
        rows=[r for r in read(a.root/'final/annotations.jsonl') if r['image'] in wanted]
        (a.root/'final'/f'{split}_annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(dict(done=True,regions=len(valid),composed=len(read(a.root/'final/annotations.jsonl')))),flush=True)


if __name__=='__main__':main()
