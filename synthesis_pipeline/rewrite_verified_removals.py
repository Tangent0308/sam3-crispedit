"""Independent concise label reconstruction after positive removal/quality evidence.

No draft instruction is shown. This cannot rescue a no-op or failed removal.
Outputs are labelled model candidates, not independently verified ground truth.
"""
import argparse
import json
from pathlib import Path
import time
from PIL import Image
from synthesis_pipeline.prepare_samtok_data import load_jsonl, write_jsonl
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.audit_quality_v4 import clean_overview, native_full_pair
from synthesis_pipeline.generate_samtok_plan import parse_json_object
from utils import vlm_utils as vlm

PROMPT='''Describe the observed removal with one short training instruction.
Image 1 is BEFORE; image 2 is AFTER. Each shows the full photo above and matching
target detail below. The BEFORE outline marks the selected object, part or group.
Identify the whole original selected target from the pictures and confirm it is
gone. Name it concisely with enough full-photo position/appearance to distinguish
similar neighbors. Use only reliable visible details; omit incidental attributes.
Do not narrow the target to a changed accessory or subset, describe another
operation, or invent a removal to explain a defective result. Do not inventory
attachments or add background/preservation clauses. If the images do not support
a coherent complete removal, return null. Output only JSON:
{"instruction":"Remove ... ."} or {"instruction":null}. Aim for 5-20 words.
'''

PROMPT_V2=PROMPT.replace(
    'similar neighbors. Use only reliable visible details; omit incidental attributes.',
    'similar neighbors. Always include its full-photo location or the selected panel\n'
    'when the source contains multiple views. Verify that the sentence cannot also\n'
    'refer to an unselected same-category instance. Prefer a plain visible category\n'
    'and location over a speculative subtype or micro-detail. Omit incidental attributes.'
)

PROMPT_V3=PROMPT_V2.replace(
    'refer to an unselected same-category instance. Prefer a plain visible category',
    'refer to an unselected same-category instance. If several similar neighbors\n'
    'share that side, include the target\'s relation to a visible neighbor; a broad\n'
    'side alone is not a unique reference. Prefer a plain visible category'
)


def eligible(record):
    p=record.get('parsed') or {}
    return (p.get('quality')==p.get('target_removed')=='pass'
            and not record.get('pixel_veto_applied',False))


def parse_instruction(raw):
    if '<think>' in raw and '</think>' not in raw:return None
    value=parse_json_object(raw.rsplit('</think>',1)[-1])
    instruction=value.get('instruction') if isinstance(value,dict) else None
    if not isinstance(instruction,str):return None
    instruction=' '.join(instruction.split())
    if not instruction.startswith('Remove ') or not 3<=len(instruction.split())<=25:return None
    return instruction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--annotations-jsonl',type=Path,required=True)
    p.add_argument('--audit-jsonl',type=Path,required=True)
    p.add_argument('--edited-dir',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--policy',choices=['legacy','grounded-v2','grounded-v3'],default='legacy')
    p.add_argument('--ids',default='',help='Optional preselected experiment IDs; no quality-based filtering')
    p.add_argument('--input-layout',choices=['stacked','full'],default='stacked')
    p.add_argument('--rewrite-scope',choices=['mismatches','all'],default='mismatches',
                   help='Avoid replacing already-matching labels; all is an explicit ablation')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False)
    (a.out_root/'inputs').mkdir()
    audits={r['image']:r for r in load_jsonl(a.audit_jsonl)}
    rows=[r for r in load_jsonl(a.annotations_jsonl) if eligible(audits.get(r['image'],{}))]
    if a.ids:
        ids={int(i) for i in a.ids.split(',')}
        rows=[r for r in rows if int(r['image'].split('_')[0]) in ids]
    if a.rewrite_scope=='mismatches':
        rows=[r for r in rows if audits[r['image']]['parsed']['instruction_match']=='fail']
    if any(r['task_type']!='remove' for r in rows):raise ValueError('Removal-only label reconstruction')
    started=time.perf_counter();results=[]
    prompt=PROMPT_V3 if a.policy=='grounded-v3' else PROMPT_V2 if a.policy=='grounded-v2' else PROMPT
    if a.input_layout=='full':
        prompt=prompt.replace('Each shows the full photo above and matching\ntarget detail below.',
                              'Each is the complete photograph at matching coordinates.')
    if rows:
        vlm.configure_backend('qwen38-vllm',model_id='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B',device='cuda:0',dtype='bf16')
        backend=vlm.get_backend();backend.enable_thinking=True
        backend.chat_template_overrides={'reasoning_effort':'low'}
        backend.sampling_overrides={'temperature':1.,'top_p':.95,'top_k':20,'presence_penalty':0.,'repetition_penalty':1.}
        try:
            for offset in range(0,len(rows),a.batch_size):
                batch=rows[offset:offset+a.batch_size];messages=[]
                for row in batch:
                    before=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
                    after=Image.open(a.edited_dir/row['image']).convert('RGB')
                    pair=({'stacked':clean_overview,'full':native_full_pair}[a.input_layout])(
                        before,after,mask_array(before.size,row['mask']))
                    for i,im in enumerate(pair):im.save(a.out_root/'inputs'/f'{Path(row["image"]).stem}_{i}.png')
                    messages.append([{'role':'user','content':[{'type':'image','image':im} for im in pair]+[{'type':'text','text':prompt}]}])
                responses=backend.chat_batch(messages,max_new_tokens=2048)
                if len(responses)!=len(batch):raise ValueError('Incomplete rewrite batch')
                for row,raw in zip(batch,responses):
                    instruction=parse_instruction(raw)
                    results.append(dict(image=row['image'],original_instruction=row['editing_instruction'],
                        instruction=instruction,status='model_candidate' if instruction else 'rejected',raw_response=raw))
                write_jsonl(a.out_root/'rewrites.jsonl',results)
                print(f'rewritten {len(results)}/{len(rows)}',flush=True)
        finally:vlm.shutdown_backend()
    write_jsonl(a.out_root/'rewrites.jsonl',results)
    (a.out_root/'prompt.txt').write_text(prompt)
    summary=dict(policy=a.policy,input_layout=a.input_layout,rewrite_scope=a.rewrite_scope,eligible_cases=len(rows),calls=len(results),candidates=sum(r['instruction'] is not None for r in results),
        verification='model_candidates_require_validation',wall_seconds=time.perf_counter()-started)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)


if __name__=='__main__':main()
