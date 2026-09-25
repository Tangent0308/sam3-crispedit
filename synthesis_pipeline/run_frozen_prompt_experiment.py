"""Generate one prompt variant on a frozen cohort; preserve all runs and logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import sys
from utils.runtime_paths import runtime_path, editor_model
from synthesis_pipeline.labeling_checkpoint import bind_settings, wait_workers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--policy',choices=['typed-v1','remove-evidence-v1','remove-parts-v1','remove-context-v1','relation-compact-v1','relation-spatial-v1','relation-located-v2','relation-action-v3'],required=True)
    p.add_argument('--remove-composition-policy',choices=['adaptive-remove-v2','adaptive-remove-v3','adaptive-remove-v4','adaptive-remove-v5'],default='adaptive-remove-v2')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--latent-protection-policy',choices=['legacy','guard-any-v1','guard-fraction-v1'],default='legacy')
    p.add_argument('--relation-geometry-policy',choices=['legacy','visible-v1'],default='legacy')
    p.add_argument('--model-id',default=editor_model())
    p.add_argument('--qwen21-backend',choices=['diffusers','vllm-omni'],default='diffusers')
    p.add_argument('--qwen21-target-guide',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=a.resume)
    bind_settings(a.out_root,{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()
        if k not in {'resume','out_root'}},a.resume)
    logs=a.out_root/'logs';logs.mkdir(exist_ok=True);start=time.perf_counter()
    rows=[json.loads(s) for s in (a.data_root/'annotations.jsonl').read_text().splitlines() if s.strip()]
    if not rows:raise ValueError('Empty frozen cohort')
    python=(runtime_path('EDITOR_PYTHON','/opt/tiger/tanyue/.venvs/qwen_omni_21/bin/python')
            if a.qwen21_backend=='vllm-omni' else
            '/opt/tiger/tanyue/.venvs/qwen_image_21/bin/python')
    variant='context_grounded_v4_qwen21'
    options=['--data-root',str(a.data_root),'--out-root',str(a.out_root),'--variant',variant,
        '--ids',','.join(r['image'].split('_')[0] for r in rows),'--steps','40','--model-id',a.model_id,
        '--qwen21-prompt-policy',a.policy,'--remove-composition-policy',a.remove_composition_policy,
        '--latent-protection-policy',a.latent_protection_policy,
        '--relation-geometry-policy',a.relation_geometry_policy,
        '--qwen21-backend',a.qwen21_backend]
    if a.qwen21_target_guide:options.append('--qwen21-target-guide')
    if a.resume:options.append('--resume')
    subprocess.run([runtime_path('SAM_PYTHON',sys.executable),'-m','synthesis_pipeline.experiment_edit_quality',*options,'--manifest-only'],check=True)
    jobs=[];gpus=a.gpus.split(',')
    for index,gpu in enumerate(gpus):
        if index>=len(rows):continue
        handle=(logs/f'editor_{index}.log').open('a' if a.resume else 'w')
        command=[python,'-m','synthesis_pipeline.experiment_edit_quality',*options,'--shard',str(index),'--shards',str(len(gpus))]
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'8'}
        if a.qwen21_backend == 'vllm-omni':
            # Dedicated environment; SDPA is the tested fallback on the
            # current H100 driver (the bundled FA3 wheel requires CUDA 13).
            site=Path(python).parent.parent/'lib/python3.12/site-packages/nvidia'
            env['LD_LIBRARY_PATH']=':'.join([str(site/'cu13/lib'),
                str(site/'cuda_runtime/lib'),env.get('LD_LIBRARY_PATH','')])
            env.setdefault('DIFFUSION_ATTENTION_BACKEND','TORCH_SDPA')
        jobs.append((index,subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT),handle))
    wait_workers(jobs,'Editor')
    results=[dict(shard=index,exit_code=proc.returncode) for index,proc,_ in jobs]
    summary=dict(policy=a.policy,remove_composition_policy=a.remove_composition_policy,qwen21_backend=a.qwen21_backend,qwen21_target_guide=a.qwen21_target_guide,latent_protection_policy=a.latent_protection_policy,relation_geometry_policy=a.relation_geometry_policy,cases=len(rows),steps=40,wall_seconds=time.perf_counter()-start,workers=results)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
    if any(r['exit_code'] for r in results):raise RuntimeError(f'Failed workers: {results}')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
