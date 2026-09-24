"""Isolated multi-GPU relation pilot with retained rejected plans and logs."""
import argparse,json,os,subprocess,time
import sys
from utils.runtime_paths import runtime_path
from pathlib import Path
from synthesis_pipeline.prepare_samtok_data import load_jsonl,write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--ids',default='')
    p.add_argument('--policy',choices=['legacy','relations-v4','relations-v5','relations-v6','relations-v7','relations-v8','relations-v9','relations-v10','relations-v11','relations-v12','relations-v13','relations-v14','relations-v15','relations-v16'],default='legacy')
    p.add_argument('--editor-policy',choices=['typed-v1','relation-spatial-v1','relation-located-v2','relation-action-v3'],default='typed-v1')
    p.add_argument('--remove-composition-policy',choices=['adaptive-remove-v2','adaptive-remove-v3','adaptive-remove-v4','adaptive-remove-v5'],default='adaptive-remove-v2')
    p.add_argument('--latent-protection-policy',choices=['legacy','guard-any-v1','guard-fraction-v1'],default='legacy')
    p.add_argument('--relation-geometry-policy',choices=['legacy','visible-v1'],default='legacy')
    p.add_argument('--qwen21-target-guide',action='store_true')
    p.add_argument('--qwen21-backend',choices=['diffusers','vllm-omni'],default='diffusers')
    p.add_argument('--thinking',action='store_true',help='Use low-effort reasoning in the same planning call')
    p.add_argument('--ground-keeps',action='store_true',help='Resolve planned touching neighbors, never audit the source mask')
    p.add_argument('--auxiliary-policy',choices=['legacy','ownership-v1'],default='legacy')
    p.add_argument('--keep-fallback',choices=['none','box-protect-v1'],default='none')
    p.add_argument('--plan-only',action='store_true',help='Stop after planning and relation grounding for controlled editor ablations')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False);(a.out_root/'logs').mkdir();gpus=a.gpus.split(',')
    rows=load_jsonl(a.data_root/'annotations.jsonl')
    if a.ids:rows=[r for r in rows if int(r['image'].split('_')[0]) in {int(i) for i in a.ids.split(',')}]
    start=time.perf_counter();jobs=[]
    for i,gpu in enumerate(gpus):
        subset=rows[i::len(gpus)]
        if not subset:continue
        log=(a.out_root/'logs'/f'plan_{i}.log').open('w')
        python=runtime_path('MLLM_PYTHON','/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python')
        command=[python,'-m','synthesis_pipeline.plan_removal_relations','--data-root',str(a.data_root),'--out-root',str(a.out_root/'planners'/str(i)),
                 '--ids',','.join(r['image'].split('_')[0] for r in subset),'--outside-pointers','--policy',a.policy]
        if a.thinking:command.append('--thinking')
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PATH':str(Path(python).parent)+':'+os.environ['PATH'],'OMP_NUM_THREADS':'8'}
        jobs.append((i,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    codes=[]
    for i,proc,log in jobs:codes.append(proc.wait());log.close()
    if any(codes):raise RuntimeError(f'Planner failures {codes}')
    merged=a.out_root/'relations';merged.mkdir();(merged/'sources').symlink_to((a.data_root/'sources').resolve())
    plans=sorted([r for i,_,_ in jobs for r in load_jsonl(a.out_root/'planners'/str(i)/'annotations.jsonl')],key=lambda r:r['image'])
    write_jsonl(merged/'annotations.jsonl',plans)
    write_jsonl(merged/'input_annotations.jsonl',load_jsonl(a.data_root/'annotations.jsonl'))
    with (a.out_root/'logs/resolve.log').open('w') as log:
        subprocess.run([runtime_path('SAM_PYTHON','/opt/tiger/tanyue/.venvs/mirage_official/bin/python'),'-m','synthesis_pipeline.resolve_removal_relations',
            '--data-root',str(merged),'--out-root',str(a.out_root/'regions'),'--auxiliary-policy',a.auxiliary_policy,'--keep-fallback',a.keep_fallback,*(['--ground-keeps'] if a.ground_keeps else [])],env={**os.environ,'CUDA_VISIBLE_DEVICES':gpus[0],'OMP_NUM_THREADS':'8'},stdout=log,stderr=subprocess.STDOUT,check=True)
    if not a.plan_only and load_jsonl(a.out_root/'regions/annotations.jsonl'):
        with (a.out_root/'logs/editor.log').open('w') as log:
            subprocess.run([sys.executable,'-m','synthesis_pipeline.run_frozen_prompt_experiment','--data-root',str(a.out_root/'regions'),
                '--out-root',str(a.out_root/'editing'),'--policy',a.editor_policy,'--gpus',a.gpus,
                '--qwen21-backend',a.qwen21_backend,'--remove-composition-policy',a.remove_composition_policy,
                '--latent-protection-policy',a.latent_protection_policy,'--relation-geometry-policy',a.relation_geometry_policy,
                *(['--qwen21-target-guide'] if a.qwen21_target_guide else [])],stdout=log,stderr=subprocess.STDOUT,check=True)
    (a.out_root/'summary.json').write_text(json.dumps(dict(input_cases=len(rows),policy=a.policy,thinking=a.thinking,editor_policy=a.editor_policy,remove_composition_policy=a.remove_composition_policy,qwen21_backend=a.qwen21_backend,ground_keeps=a.ground_keeps,auxiliary_policy=a.auxiliary_policy,keep_fallback=a.keep_fallback,plan_only=a.plan_only,latent_protection_policy=a.latent_protection_policy,relation_geometry_policy=a.relation_geometry_policy,qwen21_target_guide=a.qwen21_target_guide,wall_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
