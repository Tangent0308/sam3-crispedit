"""Fresh frozen removal cohort: shared planning, fixed 40-step prompt A/B."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

VLM='/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python'
EDIT='/opt/tiger/tanyue/.venvs/qwen_image_21/bin/python'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--model-id',default='/tmp/tanyue_qwen_image_21')
    p.add_argument('--baseline-only',action='store_true',help='Generate typed-v1 once for composition-only paired tests')
    a=p.parse_args();logs=a.root/'remove_experiment_logs';logs.mkdir(exist_ok=False)
    timing={};start=time.perf_counter()
    def launch(name,python,module,options,gpu=''):
        log=(logs/(name+'.log')).open('w')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'8'}
        env['PATH']=str(Path(python).parent)+':'+env['PATH']
        command=[python,'-m',module,*map(str,options)]
        print(json.dumps(dict(stage=name,command=command,gpu=gpu)),flush=True)
        return name,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT),log,time.perf_counter()
    def finish(jobs):
        failed=[]
        for name,proc,log,t in jobs:
            code=proc.wait();log.close();timing[name]={'seconds':time.perf_counter()-t,'exit_code':code}
            if code:failed.append(name)
        (logs/'timing.json').write_text(json.dumps(dict(stages=timing,wall_seconds=time.perf_counter()-start),indent=2))
        if failed:raise RuntimeError(str(failed))
    rows=[json.loads(x) for x in (a.root/'annotations.jsonl').read_text().splitlines()]
    if any(r['task_type']!='remove' for r in rows):raise ValueError('Removal-only cohort required')
    gpus=a.gpus.split(',');sources=list(dict.fromkeys(r['source_image'] for r in rows));jobs=[]
    for i,gpu in enumerate(gpus):
        selected=set(sources[i::len(gpus)]);ids=','.join(r['image'].split('_')[0] for r in rows if r['source_image'] in selected)
        if ids:jobs.append(launch(f'plan_{i}',VLM,'synthesis_pipeline.plan_dataset_regions',[
            '--data-root',a.root,'--out-root',a.root/'planner_shards'/str(i),'--ids',ids],gpu))
    finish(jobs)
    finish([launch('merge_plan','/usr/bin/python','synthesis_pipeline.merge_dataset_plan_shards',[
        '--data-root',a.root,'--shard-root',a.root/'planner_shards','--out-root',a.root])])
    planned=[json.loads(x) for x in (a.root/'regions/annotations.jsonl').read_text().splitlines()]
    if not planned:raise RuntimeError('No executable frozen plans')
    ids=','.join(r['image'].split('_')[0] for r in planned);variant='context_grounded_v4_qwen21'
    variants=[('baseline','typed-v1')] if a.baseline_only else [('baseline','typed-v1'),('shadow','remove-shadow-v2')]
    for label,policy in variants:
        options=['--data-root',a.root/'regions','--out-root',a.root/label,'--variant',variant,
                 '--model-id',a.model_id,'--ids',ids,'--steps','40','--qwen21-prompt-policy',policy]
        finish([launch(label+'_manifest','/usr/bin/python','synthesis_pipeline.experiment_edit_quality',options+['--manifest-only'])])
        finish([launch(f'{label}_edit_{i}',EDIT,'synthesis_pipeline.experiment_edit_quality',
            options+['--shard',i,'--shards',len(gpus)],gpu) for i,gpu in enumerate(gpus)])
        finish([launch(label+'_recompose','/usr/bin/python','synthesis_pipeline.experiment_remove_support',[
            '--data-root',a.root/'regions','--raw-root',a.root/label/variant,
            '--out-root',a.root/label/'adaptive'])])
    print(json.dumps(dict(done=True,planned=len(planned),steps=40,diffusion_calls=len(variants)*len(planned))),flush=True)


if __name__=='__main__':main()
