"""One-call, task-aware removal audit; exact inputs retained, fails closed.

An isolated experiment, not a substitute for the production rewrite workflow.
No assistant labels or previous verdicts are provided to the model.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time
from utils.runtime_paths import runtime_path, qwen38_model

from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.audit_quality_v4 import native_full_pair, paired_overview, clean_overview
from synthesis_pipeline.generate_samtok_plan import parse_json_object
from synthesis_pipeline.prepare_samtok_data import load_jsonl, write_jsonl
from synthesis_pipeline.labeling_checkpoint import CaseCheckpoints, bind_settings, file_digest, wait_workers
from utils import vlm_utils as vlm
from utils.edit_quality_guard import removal_change_evidence


PROMPT = '''Judge this removal from BEFORE and AFTER, not from the request alone.
First locate the selected instance in BEFORE, then track its entire visible
body and attachments in AFTER. Check the rest of the scene for accidental
deletions or invented objects. Use the actual pixels: an exposed pre-existing
object is not an insertion, and a retained neighbor is not a target remnant.
Pass visual quality when the result is broadly natural. Fail clear new
fragments, cut-off bodies, floating items, broken boundaries, implausible fills
or damaged neighbors. Do not penalize source defects or normal texture variation.
Separately check whether the named target is fully removed at the intended
location; a no-op, wrong instance, or partly surviving target fails the request.
The request must refer to the outlined target, not a different object or area.
Do not reinterpret an incomplete removal as a successful smaller edit.
Return only JSON with three fields:
{"quality":"pass|fail", "instruction_match":"pass|fail", "reason":"brief
concrete before/after evidence for both decisions"}.
'''

SCOPE_PREFIX = '''Before using the request, identify ALL outlined instances in BEFORE.
Separate fragments can belong to one instance; several complete outlined objects
form a selected group. Fail instruction_match if the request or result addresses
only a subset. Do not let singular wording redefine the selected mask.
'''

VISUAL_PROMPT = '''Compare the photographs at matching positions. The BEFORE
outline selects the original target, not scene content. First identify that whole
object, part or group; then inspect what occupies the same place in AFTER.
Distinguish the target from nearby similar objects and its packaging/background.
A visible main subject fails removal even if an accessory or its texture changed.

Judge quality separately: accept broadly natural completion; reject conspicuous
fragments, stranded attachments, damaged neighbors, implausible fills or a visible
old-silhouette seam. Revealing a partly occluded existing object is legitimate.
Check that the request describes this selected scope and the observed removal;
do not rescue an incomplete removal by silently narrowing the target.

Return JSON only with quality and instruction_match as pass or fail, plus reason.
In reason, state the target's visible identity in BEFORE and the actual visible
content at that position in AFTER, followed by any concrete defect. Do not assume
that the requested edit happened. Unclear completion fails instruction_match.
{"quality":"pass|fail","instruction_match":"pass|fail","reason":"observed evidence"}
'''


COMPLETION_PROMPT = '''Audit an attempted removal. The request is an untrusted draft,
not evidence of what happened. Locate the whole outlined object, part or group in
BEFORE, then identify the actual contents of that location in AFTER.
target_removed passes only if the whole selected subject is absent; a changed
accessory, color or small part is not removal. Count selected instances visually.
quality passes for a broadly natural photograph. Fail obvious surviving fragments,
target-specific residual shadows, implausible fills, broken boundaries or damaged
neighbors. Do not penalize pre-existing defects, revealed background objects or
independent airborne objects merely because the target is gone.
instruction_match checks the draft's target identity, scope and observable details.
A description error can fail this field even when the removal and quality pass.
Use the actual pictures, not the intended result. If completion is unclear, fail.
Return only JSON:
{"target_removed":"pass|fail","quality":"pass|fail",
"instruction_match":"pass|fail","reason":"actual before/after evidence and any defect"}
'''

COMPLETION_V6_PROMPT = COMPLETION_PROMPT.replace(
    'target_removed passes only if the whole selected subject is absent; a changed',
    'Judge target_removed from the outline, NEVER from the draft identity. A wrong\n'
    'draft only fails instruction_match when the outlined subject was removed.\n'
    'target_removed passes only if the whole selected subject is absent; a changed'
).replace(
    'neighbors. Do not penalize pre-existing defects, revealed background objects or',
    'neighbors. Inspect the surrounding contact surface too: a leftover detached\n'
    'cast shadow or a newly cut background edge fails quality even outside the\n'
    'outline. Compare BEFORE to distinguish these from existing scene features.\n'
    'Do not penalize pre-existing defects, revealed background objects or'
)


def parse_audit(raw, policy='legacy'):
    if '<think>' in raw and '</think>' not in raw:
        return None
    value = parse_json_object(raw.rsplit('</think>', 1)[-1])
    if not isinstance(value, dict):
        return None
    if any(value.get(k) not in {'pass', 'fail'} for k in ('quality', 'instruction_match')):
        return None
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        return None
    keys=['quality', 'instruction_match', 'reason']
    if policy in {'completion-v5','completion-v6'}:
        if value.get('target_removed') not in {'pass','fail'}:return None
        keys.insert(0,'target_removed')
    return {k: value[k] for k in keys}


def admitted(parsed):
    return bool(parsed and parsed['quality'] == parsed['instruction_match'] == 'pass'
                and parsed.get('target_removed','pass')=='pass')


def selected_layout(mask, requested):
    """Give small source targets readable detail without duplicating large subjects."""
    return ('stacked' if float(mask.mean()) < .05 else 'full') if requested=='adaptive' else requested


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--edited-dir', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--annotations-jsonl', type=Path)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--input-layout', choices=['full', 'paired', 'stacked', 'adaptive'], default='full')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--policy',choices=['legacy','mask-coverage-v2','pixels-first-v3','visual-evidence-v4','completion-v5','completion-v6'],default='legacy')
    p.add_argument('--pixel-veto',action='store_true',help='Removal-only conservative low-change veto; never yields a pass')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--checkpoint-root',type=Path)
    a = p.parse_args()
    if a.policy in {'pixels-first-v3','completion-v5','completion-v6'} and a.input_layout=='paired':
        raise ValueError('After-first audit requires separate full images')
    rows = load_jsonl(a.annotations_jsonl or a.data_root/'annotations.jsonl')
    if not rows or any(r['task_type'] != 'remove' for r in rows):
        raise ValueError('Nonempty removal cohort required')
    if len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image IDs')
    for row in rows:
        if not (a.edited_dir/row['image']).is_file():
            raise FileNotFoundError(a.edited_dir/row['image'])
    a.out_root.mkdir(parents=True, exist_ok=a.resume)
    settings=dict(policy=a.policy,input_layout=a.input_layout,pixel_veto=a.pixel_veto,
                  batch_size=a.batch_size,model=qwen38_model())
    bind_settings(a.out_root,settings,a.resume)
    started = time.perf_counter()
    if not a.worker:
        jobs=[]
        python=runtime_path('MLLM_PYTHON','/opt/tiger/tanyue/.venvs/qwen38_audit/bin/python')
        gpus=a.gpus.split(',')
        for i,gpu in enumerate(gpus):
            selected=rows[i::len(gpus)]
            if not selected: continue
            manifest=a.out_root/f'input_{i}.jsonl'
            write_jsonl(manifest, selected)
            log=(a.out_root/f'worker_{i}.log').open('a' if a.resume else 'w')
            command=[python,'-m','synthesis_pipeline.audit_removal_concise','--worker',
                '--data-root',str(a.data_root),'--edited-dir',str(a.edited_dir),
                '--annotations-jsonl',str(manifest),'--out-root',str(a.out_root/f'worker_{i}'),
                '--input-layout',a.input_layout,'--batch-size',str(a.batch_size),'--policy',a.policy,
                '--checkpoint-root',str(a.out_root)]
            if a.pixel_veto:command.append('--pixel-veto')
            if a.resume:command.append('--resume')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'8',
                 'PATH':str(Path(python).parent)+os.pathsep+os.environ.get('PATH','')}
            jobs.append((i,subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        wait_workers(jobs,'Audit')
        results=[r for i,_,_ in jobs for r in load_jsonl(a.out_root/f'worker_{i}/audit.jsonl')]
        order={r['image']:i for i,r in enumerate(rows)}
        results.sort(key=lambda r:order[r['image']])
        if len(results)!=len(rows) or len({r['image'] for r in results})!=len(rows):
            raise ValueError('Incomplete audit')
        summaries=[json.loads((a.out_root/f'worker_{i}/summary.json').read_text()) for i,_,_ in jobs]
        calls=sum(r['calls'] for r in summaries)
        inference=sum(r['inference_seconds'] for r in summaries)
        reused=sum(r.get('reused',0) for r in summaries)
    else:
        inputs=a.out_root/'inputs';inputs.mkdir(exist_ok=True)
        checkpoints=CaseCheckpoints(a.checkpoint_root or a.out_root,settings)
        dependencies={r['image']:dict(source_sha256=file_digest(a.data_root/'sources'/r['source_image']),
            edited_sha256=file_digest(a.edited_dir/r['image'])) for r in rows}
        results=[];calls=0;inference=0.
        pending=[]
        for row in rows:
            saved=checkpoints.load(row,dependencies[row['image']]) if a.resume else None
            if saved is not None and saved.get('parsed') is not None:results.append(saved)
            else:pending.append(row)
        reused=len(results)
        print(json.dumps(dict(stage='audit',total=len(rows),reused=reused,pending=len(pending))),flush=True)
        if pending:
            vlm.configure_backend('qwen38-vllm',model_id=qwen38_model(),device='cuda:0',dtype='bf16')
            backend=vlm.get_backend()
            backend.enable_thinking=True
            backend.chat_template_overrides={'reasoning_effort':'low'}
            backend.sampling_overrides={'temperature':1.,'top_p':.95,'top_k':20,'presence_penalty':0.,'repetition_penalty':1.}
        try:
            for offset in range(0,len(pending),a.batch_size):
                batch=pending[offset:offset+a.batch_size];messages=[];evidence=[]
                for row in batch:
                    source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
                    after=Image.open(a.edited_dir/row['image']).convert('RGB')
                    mask=mask_array(source.size,row['mask'])
                    input_layout=selected_layout(mask,a.input_layout)
                    make_pair={'full':native_full_pair,'paired':paired_overview,'stacked':clean_overview}[input_layout]
                    pair=make_pair(source,after,mask)
                    layout=('Image 1: BEFORE. Image 2: AFTER. Only BEFORE has a target outline.\n'
                        if input_layout=='full' else 'Both images: LEFT BEFORE, RIGHT AFTER. Image 1: full scene; image 2: matching detail. Only BEFORE detail has a target outline.\n')
                    if a.policy in {'pixels-first-v3','completion-v5','completion-v6'}:
                        pair=list(reversed(pair))
                        layout=('Image 1: actual AFTER. Image 2: BEFORE with target outline.\n'
                            'Inspect AFTER first, without assuming the requested removal happened. '
                            'Locate the target position using BEFORE, then report what visibly occupies '
                            'that position in AFTER. If the same subject is still visible, fail instruction_match '
                            'even when its accessories changed. In the reason, describe actual AFTER pixels '
                            'before giving a verdict; never describe the requested outcome as an observation.\n')
                    if input_layout=='stacked':
                        layout=('Image 1: AFTER. Image 2: BEFORE. ' if a.policy in {'pixels-first-v3','completion-v5','completion-v6'} else
                                'Image 1: BEFORE. Image 2: AFTER. ')+('Each image shows its full photo on top and the same magnified target context below. '
                                'Only BEFORE detail has a target outline. These are two views of the same scene, not extra instances.\n')
                    accessories=[r['description'] for r in row.get('relation_plan',{}).get('relations',[])
                                 if r['action']=='remove_together']
                    prompt=layout+(COMPLETION_V6_PROMPT if a.policy=='completion-v6' else COMPLETION_PROMPT if a.policy=='completion-v5' else VISUAL_PROMPT if a.policy=='visual-evidence-v4' else
                        (SCOPE_PREFIX if a.policy!='legacy' else '')+PROMPT)+'\nRequest: '+row['editing_instruction']
                    if accessories and a.policy not in {'visual-evidence-v4','completion-v5','completion-v6'}:prompt+='\nPlanned co-removal (verify in pixels): '+json.dumps(accessories)
                    paths=[]
                    for i,im in enumerate(pair):
                        path=inputs/f'{Path(row["image"]).stem}_{i}.png';im.save(path);paths.append(str(path))
                    messages.append([{'role':'user','content':[{'type':'image','image':im} for im in pair]+[{'type':'text','text':prompt}]}])
                    evidence.append((prompt,paths,removal_change_evidence(source,after,mask),input_layout))
                t=time.perf_counter();outputs=backend.chat_batch(messages,max_new_tokens=2048)
                inference+=time.perf_counter()-t
                if len(outputs)!=len(batch):raise ValueError('Incomplete model batch')
                for row,raw,(prompt,paths,pixels,input_layout) in zip(batch,outputs,evidence):
                    parsed=parse_audit(raw,a.policy);calls+=1
                    pixel_rejected=a.pixel_veto and pixels['insufficient_change']
                    results.append(dict(image=row['image'],parsed=parsed,decision='pass' if admitted(parsed) and not pixel_rejected else 'fail',
                        model_decision=('unparsed' if parsed is None else 'pass' if admitted(parsed) else 'fail'),pixel_evidence=pixels,pixel_veto_applied=bool(pixel_rejected),
                        raw_response=raw,prompt=prompt,input_images=paths,input_layout=input_layout,editing_instruction=row['editing_instruction'],
                        edited_path=str((a.edited_dir/row['image']).resolve()),reviewer='Qwen3.8-27B-vLLM; not assistant review'))
                    if parsed is not None:
                        checkpoints.save(row,results[-1],artifacts=paths,dependencies=dependencies[row['image']])
                write_jsonl(a.out_root/'audit.jsonl',results)
                print(f'audited {len(results)}/{len(rows)}',flush=True)
        finally:
            if pending:vlm.shutdown_backend()
    order={r['image']:i for i,r in enumerate(rows)};results.sort(key=lambda r:order[r['image']])
    write_jsonl(a.out_root/'audit.jsonl',results)
    summary=dict(cases=len(rows),calls=calls,reused=reused,input_layout=a.input_layout,policy=a.policy,pixel_veto=a.pixel_veto,
        decisions=dict(Counter(r['decision'] for r in results)),parse_errors=sum(r['parsed'] is None for r in results),
        inference_seconds=inference,wall_seconds=time.perf_counter()-started)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
