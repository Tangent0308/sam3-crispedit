"""Parallel local 27B auditors, preserving every stage and cohort denominator."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time

VLM='/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python'


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--annotations-jsonl',type=Path)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    p.add_argument('--quality-from',type=Path)
    p.add_argument('--reconstruction-from',type=Path)
    p.add_argument('--quality-exemplars',type=Path)
    p.add_argument('--quality-only',action='store_true')
    p.add_argument('--pixel-evidence',action='store_true')
    p.add_argument('--contact-guard',action='store_true')
    p.add_argument('--label-policy',choices=['standard','grounded','observed'],default='standard')
    p.add_argument('--quality-policy',choices=['trace','contour-aware','photographic','critical','topology'],default='contour-aware')
    p.add_argument('--quality-scope',choices=['overview','crop','tight','full'],default='tight')
    p.add_argument('--input-layout',choices=['vertical','paired','full'],default='vertical')
    p.add_argument('--verification-policy',choices=['dual','candidate-only','selective','observed'],default='candidate-only')
    a=p.parse_args();started=time.time()
    rows=read(a.annotations_jsonl or a.data_root/'annotations.jsonl')
    if not rows:raise ValueError('Empty cohort')
    a.out_root.mkdir(parents=True,exist_ok=False)
    gpus=a.gpus.split(',');jobs=[]
    for i,gpu in enumerate(gpus):
        # Contiguous shards preserve the batch grouping used during calibration.
        start=len(rows)*i//len(gpus);end=len(rows)*(i+1)//len(gpus)
        selected=rows[start:end]
        if not selected:continue
        manifest=a.out_root/f'input_{i}.jsonl'
        manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
        command=[VLM,'-m','synthesis_pipeline.audit_quality_v4','--data-root',str(a.data_root),
            '--annotations-jsonl',str(manifest),'--out-root',str(a.out_root/f'shard_{i}'),
            '--thinking','--max-tokens','3072','--quality-max-tokens','2048',
            '--quality-scope',a.quality_scope,'--quality-policy',a.quality_policy,
            '--verification-policy',a.verification_policy,'--label-policy',a.label_policy,'--input-layout',a.input_layout]
        if a.quality_only:command.append('--quality-only')
        if a.pixel_evidence:command.append('--pixel-evidence')
        if a.contact_guard:command.append('--contact-guard')
        for flag,path in [('--quality-from',a.quality_from),('--reconstruction-from',a.reconstruction_from),('--quality-exemplars',a.quality_exemplars)]:
            if path:command.extend([flag,str(path)])
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'8',
            'PATH':str(Path(VLM).parent)+':'+os.environ['PATH']}
        log=(a.out_root/f'shard_{i}.log').open('w')
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((i,proc,log));print(json.dumps(dict(shard=i,gpu=gpu,cases=len(selected),pid=proc.pid)),flush=True)
    errors=[]
    for i,proc,log in jobs:
        code=proc.wait();log.close()
        if code:errors.append(dict(shard=i,exit_code=code))
    if errors:raise RuntimeError(json.dumps(errors))
    order={r['image']:i for i,r in enumerate(rows)}
    stages=['quality'] if a.quality_only else ['quality','reconstruction','verification','edit_audit','model_accepted_annotations']
    for stage in stages:
        merged=[]
        for i,_,_ in jobs:
            path=a.out_root/f'shard_{i}'/f'{stage}.jsonl'
            if path.exists():merged.extend(read(path))
        merged.sort(key=lambda r:order[r['image']])
        if len({r['image'] for r in merged})!=len(merged):raise ValueError('Duplicate merged ids')
        if stage in {'quality','edit_audit'} and len(merged)!=len(rows):raise ValueError('Incomplete audit')
        (a.out_root/f'{stage}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in merged))
    summaries=[json.loads((a.out_root/f'shard_{i}/summary.json').read_text()) for i,_,_ in jobs]
    records=[] if a.quality_only else read(a.out_root/'edit_audit.jsonl')
    summary=dict(cases=len(rows),workers=len(jobs),wall_seconds=time.time()-started,
        decisions=dict(Counter(r['decision'] for r in records)),
        inference_gpu_seconds=sum(sum(s['inference_seconds'].values()) for s in summaries),
        calls={key:sum(s['calls'].get(key,0) for s in summaries) for key in ['quality','reconstruction','verification']},
        quality_reused_from=str(a.quality_from) if a.quality_from else None,
        reconstruction_reused_from=str(a.reconstruction_from) if a.reconstruction_from else None,
        quality_exemplars=str(a.quality_exemplars) if a.quality_exemplars else None,
        diagnostic_quality_only=a.quality_only,quality_policy=a.quality_policy,quality_scope=a.quality_scope,
        pixel_evidence=a.pixel_evidence,
        contact_guard=a.contact_guard,label_policy=a.label_policy,
        input_layout=a.input_layout,
        verification_policy=a.verification_policy,shards=summaries)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='shards'}),flush=True)


if __name__=='__main__':main()
