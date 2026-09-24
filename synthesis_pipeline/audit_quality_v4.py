"""Independent quality, reconstruction, and visual label verification.

Saved model candidates are NOT assistant-reviewed training labels. Each stage
retains its exact images, prompt, response, and latency; malformed output fails
closed. Original instructions cannot influence the first two stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from synthesis_pipeline.audit_quality_v3 import (
    read, messages, parse_compact, parse_rewrite, BALANCED_EVIDENCE_PROMPT,
)
from synthesis_pipeline.audit_edit_pairs import mask_array, write_jsonl, parse_json_object
from synthesis_pipeline.visual_prompt_utils import audit_two_image_inputs, padded_mask_bbox
import utils.vlm_utils as vlm
from synthesis_pipeline.audit_pixel_evidence import changed_surface_evidence, evidence_prompt
from synthesis_pipeline.audit_geometry import row_contact, apply_contact_gate
from synthesis_pipeline.audit_policy_photographic import (
    PHOTOGRAPHIC_QUALITY, INSTANCE_RECONSTRUCTION, PHOTOGRAPHIC_VERIFICATION, CRITICAL_QUALITY,
    critical_verification_prompt,
    GROUNDED_RECONSTRUCTION, grounded_verification_prompt,
    selective_verification_prompt,
    observed_verification_prompt,
    observed_reconstruction_prompt,
    TOPOLOGY_CHECKS,
)

LAYOUT = """There are exactly two input images: BEFORE and AFTER. Each panel shows the complete photograph above and a magnified crop of the SAME photograph below, not different scenes. The upper photograph is clean. Only the lower BEFORE crop has a black/white target-mask outline; it is annotation, not a real edge or color. Compare matching positions. Use the full photograph to locate the instance and check other instances; use the detail to inspect boundaries.\n"""

PAIRED_LAYOUT = """Exactly TWO comparison panels are provided. In BOTH panels, LEFT is BEFORE and RIGHT is AFTER. Image 1 is the complete scene pair, entirely clean. Image 2 is the matching magnified target-context pair: only its LEFT/BEFORE detail has a black/white mask outline, which is annotation, not real content. AFTER is always clean. Compare LEFT versus RIGHT within each panel at matching coordinates. The second panel magnifies the first scene; it is not a different edit. Use image 1 to locate the instance and neighbors, image 2 to inspect small defects.\n"""

FULL_LAYOUT = """Exactly TWO full photographs are provided at identical coordinates. IMAGE 1 is BEFORE; IMAGE 2 is AFTER. There are no additional panels, repeated views or magnified crops. Only BEFORE has a black/white outline outside the target mask, labeled in its header. That outline is annotation, not source content. AFTER is entirely clean. Track the same location and all neighboring instances across the two full photographs.\n"""

QUALITY = """Decide whether the actual local edit is visually usable for photographic edit training, without guessing a requested instruction.
First name what was actually present BEFORE and what is different AFTER. Check each possibly new object against BEFORE: an exposed pre-existing background object is not an insertion. A different surface finish is not a new identity. No clear useful change at the target/anchor means fail.
Then trace the entire old target footprint, not just the center of the new object. Compare the top, bottom/contact, narrow parts and neighboring instances at identical coordinates. Look for recognizable old-object fragments, amputated or abruptly cut-off new parts, doubled silhouettes, flat painted strips, conspicuous rectangular fills, broken background continuation, and newly damaged neighbors. Check whether the new object actually contacts the depicted hand/surface/support. Do not imagine an invisible support to excuse a visibly disconnected object.
Distinguish new defects from pre-existing occlusion, blur, compression or perspective by checking BEFORE. Do not require perfection, extra shadows or a visible base when normal occlusion/resolution explains it. A coherent partial recoloring of an intact object can be good quality; a leftover limb after apparent person removal is not a valid partial edit. A clear new seam or surviving target fragment is a failure even if most of the image looks natural.
Return only three fields, evidence before verdict:
{"observed_change":"actual before -> after difference", "reason":"specific findings at the old footprint, boundary/contact and neighboring objects; identify any decisive new defect", "quality":"pass|fail"}.
"""

RECONSTRUCT = """Write a concise instruction describing the actual useful local edit in this BEFORE/AFTER pair. No old request is provided.
Track the same target across the two full photographs. Confirm in BEFORE whether an apparent insertion already existed. Determine the true operation: add introduces a new item while retaining its anchor; remove eliminates a whole named object or a genuinely separable component; replace substitutes a different identity; attribute changes a visible property of the same recognizable instance. Do not turn a revealed background object into a replacement, or a color change into replacement.
Describe precisely the actual changed scope. A coherent body-only recoloring must say body, not the entire statue; a single intact changed instance within a group must identify that instance, not claim the group changed. These are legitimate narrower edits, not permission to reinterpret leftover fragments after destructive removal. Never excuse residual body parts, detached objects, damaged neighbors or hidden additional edits with a clever instruction. If no clean useful edit can be described, return null fields.
One imperative English sentence, at most 25 words. Identify the target from the ORIGINAL FULL photograph alone: include the shortest sufficient position or stable visible relation when similar instances exist. Include only the clearly visible result, without speculative material, pose or style details. No mask/outline references, preservation clauses or reconstruction recipes. No fixed object suggestions.
Return only {"task_type":"add|remove|replace|attribute or null", "instruction":"command or null"}.
"""

CONTOUR_QUALITY = """Inspect AFTER as a photograph FIRST, before trying to explain the edit. AFTER contains NO annotation whatsoever. Any bright cutout, dark sliver, line or halo in AFTER is real output content, not a mask overlay to ignore.
Inspect the target's entire outer contour and interior, especially its top and attachment/contact points. Does each visible form make physical sense? A person's head needs a plausible rounded scalp above the face: background cutting through the forehead or replacing the top of the skull is a defect, not ordinary hair styling. Limbs, narrow attachments and held objects must join coherently. Newly flat painted regions must not erase the object's lighting and photographic texture. Look for visible seams, doubled edges, detached fragments and implausible clipped shapes, rather than just recognizing an object category.
Now compare identical positions in BEFORE. Its black/white outline is the ONLY annotation. Establish what genuinely changed and whether a suspected defect already existed. Pre-existing blur, compression, occlusion and ordinary perspective are acceptable. Do not require invisible details, a stand or a shadow that cannot reasonably be resolved. Inspect neighboring instances and the entire old footprint; a remaining piece of a removed/replaced target is a defect even if the new center looks convincing. An exposed old background object is not an insertion. No clear target-area edit means fail.
Pass if there is a clearly useful localized change and the result is broadly natural. Fail a definite new physical defect, recognizable target remnant, conspicuous paint/fill seam, incorrect instance or damaged unrelated object. Do not reinterpret a defect to invent a successful operation. A coherent change to an intact object's part may pass; a destructive edit leaving fragments may not.
Only JSON, three concise fields: {"observed_change":"actual photographic before -> after difference, at most 30 words", "reason":"specific visible evidence, at most 70 words", "quality":"pass|fail"}.
"""

VERIFY = """Independently verify proposed labels against the actual BEFORE/AFTER pictures. Neither proposal is evidence that an edit happened. A previous stage's verdict is deliberately not provided.
Original label ({original_type}): {original}
Reconstructed label ({candidate_type}): {candidate}
First compare the original target and its complete footprint against AFTER, including narrow remnants, contact, boundaries and neighbors. Reject a no-op, wrong-instance edit, recognizable leftover target piece, newly broken or floating part, conspicuous pasted fill/edge, or unrelated damage. Pre-existing source defects and ordinary texture variation are not failures.
For each label separately, check the action, identity/property, count and location against the pixels. A label must uniquely identify its target in the original full scene WITHOUT using the annotation. Add requires a new item, remove requires disappearance of the whole named target, replace requires an actual different identity instead of the target, attribute retains identity. Do not infer details just because the label says them. Different small wording is acceptable, false appearance/count/operation is not.
For the reconstructed label, is its target/anchor the same complete target marked in BEFORE (same), a coherent intact subset of it (subset), another instance (wrong), or uncertain (unclear)? A surviving fragment of a removed object is NOT a valid subset. Subset labels require separate mask realignment and must not be automatically admitted with the old mask.
Return exactly five concise fields:
{{"reason":"concrete before/after evidence for physical quality and both label decisions", "quality":"pass|fail", "original_match":"pass|fail", "candidate_match":"pass|fail", "candidate_scope":"same|subset|wrong|unclear"}}.
"""

CANDIDATE_VERIFY = """Independently verify ONE proposed instruction from actual BEFORE and AFTER pixels. You are not given an old request or any earlier verdict.
Proposed type: {candidate_type}
Proposed instruction: {candidate}
First track the object at the marked target/anchor across BOTH photographs. Identify the actual source object from BEFORE, not from the proposed wording. Search BEFORE for anything the instruction claims is newly added. Compare neighboring same-category instances. The source target must be uniquely locatable from the original full photograph without an annotation.
Check the operation, identity/property, count and position. Check ALL meaningful actual changes, not merely whether the sentence describes ONE thing that happened. A label that ignores another substantial insertion/deletion is false even if the described part is true. Minor rendering differences and paraphrases are acceptable. A genuinely exposed background object is not a newly inserted replacement; a retained identity with a changed property is attribute, not replace. A null instruction fails.
Separately inspect physical quality without imagining an old request: remnants after an actual destructive removal/replacement, newly cut-off parts, disconnected objects, obvious paint/fill seams or damaged neighbors fail. An unchanged head/handle/color on an otherwise intact partly recolored object is NOT a removal remnant. A natural body-only recoloring can pass physical quality with an accurately scoped body-only label. AFTER has no annotations: bright outlines there are real image defects, not a mask overlay.
Scope is same when the proposed target/anchor corresponds to the marked complete target; subset for a coherent intact part/group subset; wrong for another instance; unclear when uncertain. Do not call a truncated remnant an intentional subset. Judge location in the full photograph, not relative to the crop.
Only four JSON fields: {{"reason":"concrete observations supporting the label, coverage of actual changes, and physical quality; at most 100 words", "quality":"pass|fail", "candidate_match":"pass|fail", "candidate_scope":"same|subset|wrong|unclear"}}.
"""


def clean_overview(source, edited, mask):
    """Clean whole photos preserve pixels obscured by the detail's annotation."""
    if source.size != edited.size:
        raise ValueError('Audit image dimensions differ; do not silently rescale')
    detail = audit_two_image_inputs(source, edited, mask, 1024, 'context_crop')
    panels = []
    for label, whole, crop in zip(('BEFORE', 'AFTER'), (source, edited), detail):
        panel = Image.new('RGB', (1024, 1344), 'white')
        draw = ImageDraw.Draw(panel)
        for y, photo, title in [(0, whole, f'{label}: FULL ORIGINAL COORDINATES'),
                                (672, crop, f'{label}: MAGNIFIED TARGET CONTEXT')]:
            draw.text((8, y + 8), title, fill='black')
            tile = ImageOps.contain(photo, (1024, 640))
            panel.paste(tile, ((1024 - tile.width)//2, y + 32 + (640-tile.height)//2))
        panels.append(panel)
    return tuple(panels)


def paired_overview(source, edited, mask):
    """Rearrange identical pixels, with no resize or extra image, to compare L/R."""
    before,after=clean_overview(source,edited,mask)
    panels=[]
    for y in (0,672):
        panel=Image.new('RGB',(2048,672),'white')
        panel.paste(before.crop((0,y,1024,y+672)),(0,0))
        panel.paste(after.crop((0,y,1024,y+672)),(1024,0))
        panels.append(panel)
    return tuple(panels)


def native_full_pair(source, edited, mask):
    """Two native-size photos, no crop, duplication or hidden output resizing."""
    if source.size != edited.size:
        raise ValueError('Audit image dimensions differ; do not silently rescale')
    return audit_two_image_inputs(source,edited,mask,max(source.size),'full')


def parse_verification(raw):
    if '<think>' in raw and '</think>' not in raw:
        return None
    val = parse_json_object(raw.rsplit('</think>', 1)[-1])
    if not isinstance(val, dict) or not isinstance(val.get('reason'), str) or not val['reason'].strip():
        return None
    if any(val.get(k) not in ('pass', 'fail') for k in ('quality', 'original_match', 'candidate_match')):
        return None
    if val.get('candidate_scope') not in ('same', 'subset', 'wrong', 'unclear'):
        return None
    return val


def parse_candidate_verification(raw):
    # Reuse strict validation without implying that an unseen old label was checked.
    if '<think>' in raw and '</think>' not in raw:return None
    val=parse_json_object(raw.rsplit('</think>',1)[-1])
    if not isinstance(val,dict):return None
    checked=parse_verification(json.dumps({**val,'original_match':'fail'}))
    if checked is not None:checked['original_match']='not_checked'
    return checked


def parse_reconstruction(raw):
    val=parse_rewrite(raw)
    if val and len(val['editing_instruction'].split())>25:return None
    return val


def parse_selective_verification(raw):
    if '<think>' in raw and '</think>' not in raw:return None
    val=parse_json_object(raw.rsplit('</think>',1)[-1])
    if not isinstance(val,dict) or val.get('label_choice') not in ('original','rewritten','none'):
        return None
    choice=val['label_choice']
    return parse_verification(json.dumps({**val,'original_match':'pass' if choice=='original' else 'fail',
        'candidate_match':'pass' if choice=='rewritten' else 'fail','candidate_scope':val.get('chosen_scope')}))


def admission(quality, candidate, verification):
    """Pure final gate; no rewrite can overrule a physical-quality failure."""
    if not quality or quality.get('visual_quality') != 'pass':
        return 'reject_quality'
    if not verification:
        return 'reject_verification_parse'
    if verification['quality'] != 'pass':
        return 'reject_verified_quality'
    # A selected ORIGINAL must obey the same mask contract as a rewrite.
    if verification.get('label_choice')=='original':
        if verification['candidate_scope']=='subset':return 'needs_mask_realignment'
        if verification['candidate_scope']!='same':return 'reject_scope'
    if verification['original_match'] == 'pass':
        return 'keep_original'
    if not candidate or verification['candidate_match'] != 'pass':
        return 'reject_label'
    if verification['candidate_scope'] == 'subset':
        return 'needs_mask_realignment'
    if verification['candidate_scope'] != 'same':
        return 'reject_scope'
    return 'accept_rewrite'


def corrected_annotation(row, candidate, decision):
    """Keep every active label alias consistent; preserve generation provenance."""
    corrected=dict(row)
    if decision=='accept_rewrite':
        corrected.update(candidate)
        corrected['new_instruction']=candidate['editing_instruction']
    corrected['audit_v4']=dict(decision=decision,original_instruction=row['editing_instruction'],
        original_new_instruction=row.get('new_instruction'),original_task_type=row['task_type'],
        verification='model_verified_not_assistant_reviewed')
    return corrected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--annotations-jsonl', type=Path)
    p.add_argument('--edited-dir', type=Path)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=6)
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--max-tokens', type=int, default=3072)
    p.add_argument('--quality-max-tokens',type=int,help='Optional separate thinking budget for quality, e.g. with a reference grid')
    p.add_argument('--compare-baseline', action='store_true')
    p.add_argument('--quality-only', action='store_true', help='Diagnostic ablation; never exports accepted labels')
    p.add_argument('--quality-from',type=Path,help='Reuse immutable quality records for a label-verification ablation')
    p.add_argument('--reconstruction-from',type=Path,help='Reuse exact same-image candidates to isolate verifier changes')
    p.add_argument('--quality-scope',choices=['overview','crop','tight','full'],default='overview')
    p.add_argument('--input-layout',choices=['vertical','paired','full'],default='vertical')
    p.add_argument('--quality-policy',choices=['trace','contour-aware','photographic','critical','topology'],default='trace')
    p.add_argument('--quality-exemplars',type=Path,help='Optional third image: labeled development-only QA reference grid')
    p.add_argument('--pixel-evidence',action='store_true',help='Opt-in luminance measurements, not a hard quality veto')
    p.add_argument('--contact-guard',action='store_true',help='Reject additions clearly disconnected from their source mask host')
    p.add_argument('--label-policy',choices=['standard','grounded','observed'],default='standard')
    p.add_argument('--verification-policy',choices=['dual','candidate-only','selective','observed'],default='dual')
    p.add_argument('--ids',default='',help='Optional explicit diagnostic subset; never used to curate evaluation results')
    p.add_argument('--model-id', default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
    args = p.parse_args()
    if args.verification_policy=='observed' and args.label_policy not in ('grounded','observed'):
        p.error('observed verification requires grounded label policy')
    if args.input_layout in ('paired','full') and args.quality_scope!='overview':
        p.error('paired/full layout requires overview scope so all stages use the selected pair')
    layout={'vertical':LAYOUT,'paired':PAIRED_LAYOUT,'full':FULL_LAYOUT}[args.input_layout]
    wall_start = time.perf_counter()
    edited_dir = args.edited_dir or args.data_root/'edited'
    if args.batch_size < 1 or args.max_tokens < 1:
        p.error('Positive batch/token limits required')
    rows = read(args.annotations_jsonl or args.data_root/'annotations.jsonl')
    if args.ids:
        wanted={int(x) for x in args.ids.split(',')}
        rows=[r for r in rows if int(r['image'].split('_')[0]) in wanted]
        if len(rows)!=len(wanted):raise ValueError('Missing diagnostic ids')
    if not rows or len({r['image'] for r in rows}) != len(rows):
        raise ValueError('Empty cohort or duplicate image ids')
    args.out_root.mkdir(parents=True, exist_ok=False)
    inputs = args.out_root/'inputs'; inputs.mkdir()
    prepared = {}
    for row in rows:
        source = Image.open(args.data_root/'sources'/row['source_image']).convert('RGB')
        edited = Image.open(edited_dir/row['image']).convert('RGB')
        mask = mask_array(source.size, row['mask'])
        pair = {'vertical':clean_overview,'paired':paired_overview,'full':native_full_pair}[args.input_layout](source,edited,mask)
        suffixes=('full_pair','detail_pair') if args.input_layout=='paired' else ('before','after')
        paths = [inputs/f'{Path(row["image"]).stem}_{s}.png' for s in suffixes]
        for im, path in zip(pair, paths): im.save(path)
        prepared[row['image']] = dict(pair=pair, paths=[str(x) for x in paths],
            geometric_contact=row_contact(row,source.size) if args.contact_guard else None,
            pixel_evidence=changed_surface_evidence(source,edited,mask,row['task_type']) if args.pixel_evidence else None)
    preparation_seconds = time.perf_counter()-wall_start
    started = time.perf_counter()
    vlm.configure_backend('qwen38-vllm', model_id=args.model_id, device='cuda:0', dtype='bf16')
    backend = vlm.get_backend(); load_seconds = time.perf_counter()-started
    stages, timings, calls = {}, {}, {}

    def run_stage(name, selected, prompt_fn, parser, baseline=False):
        results = {}; seconds=0
        for offset in tqdm(range(0,len(selected),args.batch_size),desc=name):
            batch=selected[offset:offset+args.batch_size]; prompts=[]; jobs=[]
            for row in batch:
                prompt=prompt_fn(row)
                if name=='verification' and args.quality_policy=='topology':
                    prompt+=TOPOLOGY_CHECKS
                if name in ('quality','verification') and args.pixel_evidence:
                    prompt+=evidence_prompt(prepared[row['image']]['pixel_evidence'])
                prompts.append(prompt)
                pair=prepared[row['image']]['pair']
                if baseline:
                    source=Image.open(args.data_root/'sources'/row['source_image']).convert('RGB')
                    edited=Image.open(edited_dir/row['image']).convert('RGB')
                    pair=audit_two_image_inputs(source,edited,mask_array(source.size,row['mask']),1280,'context_crop')
                    for im,suffix in zip(pair,('before','after')):
                        im.save(inputs/f'{Path(row["image"]).stem}_baseline_{suffix}.png')
                elif name=='quality' and args.quality_scope!='overview':
                    source=Image.open(args.data_root/'sources'/row['source_image']).convert('RGB')
                    edited=Image.open(edited_dir/row['image']).convert('RGB')
                    mask=mask_array(source.size,row['mask'])
                    if args.quality_scope=='tight':
                        box=padded_mask_bbox(mask,padding_fraction=.25,min_padding=32)
                        source=source.crop(box);edited=edited.crop(box)
                        mask=mask[box[1]:box[3],box[0]:box[2]]
                    pair=audit_two_image_inputs(source,edited,mask,1280,'context_crop' if args.quality_scope=='crop' else 'full')
                    for im,suffix in zip(pair,('before','after')):
                        im.save(inputs/f'{Path(row["image"]).stem}_quality_{suffix}.png')
                message=messages(*pair,prompt)
                if name=='quality' and args.quality_exemplars:
                    message[0]['content'].insert(2,{'type':'image','image':Image.open(args.quality_exemplars).convert('RGB')})
                    prompt+='\nIMAGE 3 is a reference grid of FOUR DIFFERENT development examples, NOT this case. Its PASS/FAIL captions illustrate the visual standard. Judge only images 1 and 2; do not copy objects, operations, or verdicts from the examples. Use the references to recognize actual missing geometry/bright contour remnants and to avoid speculative support failures.\n'
                    message[0]['content'][-1]['text']=prompt
                    prompts[-1]=prompt
                jobs.append(message)
            t=time.perf_counter()
            budget=(args.quality_max_tokens or args.max_tokens) if name=='quality' else args.max_tokens
            raw=backend.chat_batch(jobs,max_new_tokens=256 if baseline else (budget if args.thinking else 640))
            seconds+=time.perf_counter()-t
            for row,text,prompt in zip(batch,raw,prompts):
                complete=not (args.thinking and not baseline) or '</think>' in text
                val=parser(text) if complete else None
                result=dict(image=row['image'],task_type=row['task_type'],editing_instruction=row['editing_instruction'],
                    stage=name,prompt=prompt,raw_response=text,parsed=val,reasoning_complete=complete,
                    edited_path=str((edited_dir/row['image']).resolve()),
                    input_images=[str(inputs/f'{Path(row["image"]).stem}_baseline_{s}.png') for s in ('before','after')] if baseline else
                        ([str(inputs/f'{Path(row["image"]).stem}_quality_{s}.png') for s in ('before','after')] if name=='quality' and args.quality_scope!='overview' else prepared[row['image']]['paths']))
                if args.pixel_evidence:
                    result['pixel_evidence']=prepared[row['image']]['pixel_evidence']
                if name in ('quality','baseline'):
                    result.update(audit=val,quality=val['quality'] if val else 'parse_error')
                if name=='quality' and args.quality_exemplars:
                    result['input_images']=result['input_images']+[str(args.quality_exemplars)]
                results[row['image']]=result
                with (args.out_root/f'{name}.jsonl').open('a') as f: f.write(json.dumps(result,ensure_ascii=False)+'\n')
        timings[name]=seconds;calls[name]=len(selected);stages[name]=results
        return results

    try:
        if args.compare_baseline:
            run_stage('baseline',rows,lambda r:BALANCED_EVIDENCE_PROMPT,lambda s:parse_compact(s,True),True)
        if args.thinking:
            backend.enable_thinking=True
            backend.chat_template_overrides={'reasoning_effort':'low'}
            backend.sampling_overrides={'temperature':1.,'top_p':.95,'top_k':20,'presence_penalty':0.,'repetition_penalty':1.}
        quality_layout=layout if args.quality_scope=='overview' else 'Image 1 is BEFORE and image 2 is AFTER at identical coordinates. Only BEFORE has a black/white target-mask outline; ignore this annotation when comparing content. These are '+('full photographs.\n' if args.quality_scope=='full' else 'aligned target-area crops with surrounding context.\n')
        quality_prompt={'trace':QUALITY,'contour-aware':CONTOUR_QUALITY,
            'photographic':PHOTOGRAPHIC_QUALITY,'critical':CRITICAL_QUALITY,
            'topology':TOPOLOGY_CHECKS+CRITICAL_QUALITY}[args.quality_policy]
        if args.quality_from:
            saved={r['image']:r for r in read(args.quality_from)}
            quality={}
            for row in rows:
                record=saved[row['image']]
                if Path(record['edited_path']).resolve()!=(edited_dir/row['image']).resolve():
                    raise ValueError('Cached quality verdict refers to a different output image')
                quality[row['image']]={**record,'reused_from':str(args.quality_from)}
            write_jsonl(args.out_root/'quality.jsonl',list(quality.values()))
            calls['quality']=0;timings['quality']=0
        else:
            quality=run_stage('quality',rows,lambda r:quality_layout+quality_prompt,lambda s:parse_compact(s,True))
        if args.quality_only:
            summary=dict(cases=len(rows),diagnostic_quality_only=True,scope=args.quality_scope,policy=args.quality_policy,
                thinking=args.thinking,input_layout=args.input_layout,pixel_evidence=args.pixel_evidence,calls=calls,inference_seconds=timings,load_seconds=load_seconds,
                input_preparation_seconds=preparation_seconds,wall_seconds=time.perf_counter()-wall_start)
            (args.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)
            return
        approved=[r for r in rows if (quality[r['image']]['parsed'] or {}).get('visual_quality')=='pass']
        reconstruct_prompt=RECONSTRUCT+(INSTANCE_RECONSTRUCTION if args.quality_policy in ('photographic','critical') else '')
        if args.label_policy in ('grounded','observed'):reconstruct_prompt=GROUNDED_RECONSTRUCTION
        def reconstruction_prompt(row):
            if args.label_policy=='observed':
                reading=quality[row['image']]['parsed'] or {}
                return layout+observed_reconstruction_prompt(reading.get('observed_edit') or reading.get('observed_change'))
            return layout+reconstruct_prompt
        if args.reconstruction_from:
            cached={r['image']:r for r in read(args.reconstruction_from)};rewrite={}
            for row in approved:
                record=cached[row['image']]
                if Path(record['edited_path']).resolve()!=(edited_dir/row['image']).resolve():
                    raise ValueError('Cached reconstruction refers to another output image')
                rewrite[row['image']]={**record,'reused_from':str(args.reconstruction_from)}
            write_jsonl(args.out_root/'reconstruction.jsonl',list(rewrite.values()))
            calls['reconstruction']=0;timings['reconstruction']=0
        else:
            rewrite=run_stage('reconstruction',approved,reconstruction_prompt,parse_reconstruction)
        def verify_prompt(row):
            candidate=rewrite[row['image']]['parsed'] or {}
            if args.verification_policy=='observed':
                reading=quality[row['image']]['parsed'] or {}
                return layout+observed_verification_prompt(candidate.get('task_type','null'),
                    candidate.get('editing_instruction','NO VALID RECONSTRUCTION'),
                    reading.get('observed_edit') or reading.get('observed_change'))
            if args.verification_policy=='selective':
                return layout+selective_verification_prompt(row['task_type'],row['editing_instruction'],
                    candidate.get('task_type','null'),candidate.get('editing_instruction','NO VALID RECONSTRUCTION'))
            if args.label_policy in ('grounded','observed') and args.verification_policy=='candidate-only':
                return layout+grounded_verification_prompt(candidate.get('task_type','null'),
                    candidate.get('editing_instruction','NO VALID RECONSTRUCTION'))
            if args.quality_policy=='critical' and args.verification_policy=='candidate-only':
                return layout+critical_verification_prompt(candidate.get('task_type','null'),
                    candidate.get('editing_instruction','NO VALID RECONSTRUCTION'))
            template=VERIFY if args.verification_policy=='dual' else CANDIDATE_VERIFY
            prompt=layout+template.format(original_type=row['task_type'],original=row['editing_instruction'],
                candidate_type=candidate.get('task_type','null'),candidate=candidate.get('editing_instruction','NO VALID RECONSTRUCTION'))
            return prompt+(PHOTOGRAPHIC_VERIFICATION if args.quality_policy in ('photographic','critical') else '')
        verifier_parser={'dual':parse_verification,'candidate-only':parse_candidate_verification,
                         'selective':parse_selective_verification,'observed':parse_candidate_verification}[args.verification_policy]
        verification=run_stage('verification',approved,verify_prompt,verifier_parser)
        final=[];accepted=[]
        for row in rows:
            name=row['image'];q=quality[name]['parsed'];c=rewrite.get(name,{}).get('parsed');v=verification.get(name,{}).get('parsed')
            decision=admission(q,c,v)
            entry=dict(image=name,task_type=row['task_type'],editing_instruction=row['editing_instruction'],
                quality='pass' if decision in ('keep_original','accept_rewrite') else 'fail',
                decision=decision,audit=q,candidate=c,verification=v,
                edited_path=str((edited_dir/name).resolve()))
            if args.contact_guard:entry=apply_contact_gate(entry,prepared[name]['geometric_contact'])
            final.append(entry)
            if entry['quality']=='pass':
                accepted.append(corrected_annotation(row,c,decision))
        write_jsonl(args.out_root/'edit_audit.jsonl',final)
        write_jsonl(args.out_root/'model_accepted_annotations.jsonl',accepted)
        summary=dict(cases=len(rows),decisions=dict(Counter(r['decision'] for r in final)),
            model=args.model_id,thinking=args.thinking,quality_scope=args.quality_scope,quality_policy=args.quality_policy,
            pixel_evidence=args.pixel_evidence,
            contact_guard=args.contact_guard,label_policy=args.label_policy,
            input_layout=args.input_layout,
            verification_policy=args.verification_policy,quality_reused_from=str(args.quality_from) if args.quality_from else None,
            reconstruction_reused_from=str(args.reconstruction_from) if args.reconstruction_from else None,
            calls=calls,inference_seconds=timings,
            input_preparation_seconds=preparation_seconds,
            load_seconds=load_seconds,wall_seconds=time.perf_counter()-wall_start)
        (args.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary),flush=True)
    finally: backend.close()


if __name__=='__main__':main()
