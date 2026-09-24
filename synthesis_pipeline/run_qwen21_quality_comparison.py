"""Frozen 40-step paired experiment; preserve baseline and every rejection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

VLM='/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python'
EDIT='/opt/tiger/tanyue/.venvs/qwen_image_21/bin/python'


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model-id',default='/tmp/tanyue_qwen_image_21')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--reuse-regions',action='store_true')
    a=p.parse_args();logs=a.root/'quality_comparison_logs';logs.mkdir(exist_ok=False)
    times={};start=time.perf_counter()
    def launch(name,python,module,options,gpu=''):
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'8'}
        env['PATH']=str(Path(python).parent)+':'+env['PATH']
        log=(logs/f'{name}.log').open('w')
        command=[python,'-m',module,*map(str,options)]
        print(json.dumps(dict(stage=name,command=command,gpu=gpu)),flush=True)
        return name,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT),log,time.perf_counter()
    def finish(jobs):
        errors=[]
        for name,proc,log,t in jobs:
            code=proc.wait();log.close();times[name]=dict(seconds=time.perf_counter()-t,exit_code=code)
            if code:errors.append(name)
        (logs/'timing.json').write_text(json.dumps(dict(stages=times,wall_seconds=time.perf_counter()-start),indent=2))
        if errors:raise RuntimeError(f'Failed stages: {errors}')
    gpus=a.gpus.split(',')
    if not a.reuse_regions:
        rows=read(a.root/'annotations.jsonl');jobs=[]
        for i,gpu in enumerate(gpus):
            # Keep the two regions of a source together for deterministic batching.
            sources=list(dict.fromkeys(r['source_image'] for r in rows))[i::len(gpus)]
            selected=[r for r in rows if r['source_image'] in sources]
            if not selected:continue
            jobs.append(launch(f'plan_{i}',VLM,'synthesis_pipeline.plan_dataset_regions',[
                '--data-root',a.root,'--out-root',a.root/'planner_shards'/str(i),
                '--ids',','.join(r['image'].split('_')[0] for r in selected)],gpu))
        finish(jobs)
        finish([launch('merge_plan','/usr/bin/python','synthesis_pipeline.merge_dataset_plan_shards',[
            '--data-root',a.root,'--shard-root',a.root/'planner_shards','--out-root',a.root])])
    rows=read(a.root/'regions/annotations.jsonl')
    if not rows:raise RuntimeError('No executable plans; preserve cohort and stop')
    ids=','.join(r['image'].split('_')[0] for r in rows)
    variant='context_grounded_v4_qwen21'
    for label,policy in [('baseline','legacy'),('typed','typed-v1')]:
        options=['--data-root',a.root/'regions','--out-root',a.root/label,
                 '--variant',variant,'--model-id',a.model_id,'--ids',ids,
                 '--steps','40','--qwen21-prompt-policy',policy]
        finish([launch(f'{label}_manifest','/usr/bin/python','synthesis_pipeline.experiment_edit_quality',options+['--manifest-only'])])
        finish([launch(f'{label}_edit_{i}',EDIT,'synthesis_pipeline.experiment_edit_quality',
                       options+['--shard',i,'--shards',len(gpus)],gpu) for i,gpu in enumerate(gpus)])
    # Cross both raw generators with both composition policies to isolate effects.
    for label in ['baseline','typed']:
        jobs=[]
        for i,policy in enumerate(['legacy','guarded-v1']):
            jobs.append(launch(f'{label}_{policy}_compose','/usr/bin/python',
                'synthesis_pipeline.compose_segmented_replacements',[
                    '--data-root',a.root/'regions','--raw-root',a.root/label/variant,
                    '--out-root',a.root/label/f'composed_{policy}',
                    '--include-add','--carry-others','--composition-policy',policy],gpus[i%len(gpus)]))
        finish(jobs)
        for policy in ['legacy','guarded-v1']:
            finish([launch(f'{label}_{policy}_final','/usr/bin/python','synthesis_pipeline.assemble_fresh_snapshot',[
                '--raw-root',a.root/label/variant,'--composed-root',a.root/label/f'composed_{policy}',
                '--out-root',a.root/label/f'final_{policy}'])])
    print(json.dumps(dict(done=True,executable=len(rows),diffusion_calls=2*len(rows),steps=40,
                         wall_seconds=time.perf_counter()-start)),flush=True)


if __name__=='__main__':main()
