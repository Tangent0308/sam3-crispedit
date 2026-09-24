"""Compose replacements or optional additions using SAM3-segmented silhouettes."""
import argparse
import json
import sys
import time
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.generate_samtok_plan import replacement_text_from_instruction
from synthesis_pipeline.refine_samtok_regions import encode_mask
from utils.context_edit import compose_grounded_crop
from utils.edit_quality_guard import replacement_phrase, raw_locality, composition_retention


def addition_phrase(instruction):
    """Extract the inserted noun phrase, without an extra model call."""
    text = re.sub(r'^(?:add|attach|place|put|insert)\s+', '', instruction.strip(), flags=re.I)
    return re.split(r'\s+(?:to|onto|on|beside|near|at|under|above|into|next to)\s+',
                    text, maxsplit=1, flags=re.I)[0].rstrip(' .')


def compose_addition(source, raw, added, target, bbox, protected):
    # Only the inserted silhouette and a narrow contact collar are writable.
    # Reusing the anchor's entire crop also copies incidental regenerated people.
    import cv2
    from PIL import ImageFilter
    x1,y1,x2,y2=bbox
    local=added[y1:y2,x1:x2].astype(np.uint8)
    support=cv2.dilate(local,np.ones((5,5),np.uint8))
    alpha=np.asarray(Image.fromarray(support*255).filter(ImageFilter.GaussianBlur(1))).copy()
    alpha[local>0]=255
    # A new item may legitimately occlude its support (e.g. sauce on a plate).
    # Do not punch the old plate/person back through the inserted silhouette.
    guard=(protected[y1:y2,x1:x2].astype(bool)&~target[y1:y2,x1:x2].astype(bool)
           &~support.astype(bool))
    alpha[guard]=0
    result=source.copy();a=Image.fromarray(alpha)
    result.paste(Image.composite(raw,source.crop(bbox),a),bbox[:2])
    return result,a


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--ids',default='')
    p.add_argument('--include-add',action='store_true',help='Experimental silhouette-only additive composition')
    p.add_argument('--carry-others',action='store_true',help='Preserve all other generated rows in a complete run manifest')
    p.add_argument('--composition-policy', choices=['legacy', 'guarded-v1'], default='legacy')
    p.add_argument('--sam3-root',default='/opt/tiger/tanyue/sam3-crispedit')
    p.add_argument('--checkpoint',default='/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt')
    args=p.parse_args();sys.path.insert(0,args.sam3_root)
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    args.out_root.mkdir(parents=True,exist_ok=False)
    (args.out_root/'edited').mkdir();(args.out_root/'support').mkdir()
    started=time.perf_counter()
    processor=Sam3Processor(build_sam3_image_model(checkpoint_path=args.checkpoint,load_from_HF=False),confidence_threshold=.3)
    load_seconds=time.perf_counter()-started
    rows=[json.loads(x) for x in (args.data_root/'annotations.jsonl').read_text().splitlines()]
    wanted={int(x) for x in args.ids.split(',')} if args.ids else None
    results=[];failures=[];fallbacks=[]
    def fallback(row, source, raw_full, target, protect, reason, phrase, evidence=None):
        locality, changed_support = raw_locality(source, raw_full, target, protect)
        if not locality['locality_pass']:
            failures.append(dict(image=row['image'], phrase=phrase, reason=reason,
                                 fallback_locality=locality, retention=evidence))
            return
        (args.out_root/'edited'/row['image']).symlink_to((args.raw_root/'edited'/row['image']).resolve())
        Image.fromarray(changed_support.astype(np.uint8)*255).save(args.out_root/'support'/row['image'])
        # Pixel changes are NOT a semantic segmentation of the inserted item.
        # Do not fabricate added_mask/replacement_mask from a difference map.
        results.append({**row, 'edit_support_mask':encode_mask(changed_support),
            'composition_revision':dict(method='localized_raw_fallback', reason=reason,
                phrase=phrase, locality=locality, retention=evidence,
                quality_status='requires_visual_audit',
                semantic_mask_status='unresolved', changed_from_original=False)})
        fallbacks.append(dict(image=row['image'], reason=reason, locality=locality))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for row in rows:
            if wanted is not None and int(row['image'].split('_')[0]) not in wanted:continue
            if not (args.raw_root/'edited'/row['image']).exists():continue
            if row['task_type'] not in ({'replace','add'} if args.include_add else {'replace'}):
                if args.carry_others:
                    (args.out_root/'edited'/row['image']).symlink_to((args.raw_root/'edited'/row['image']).resolve())
                    results.append(row)
                continue
            root=args.raw_root/'diagnostics'/Path(row['image']).stem
            request=json.loads((root/'generation_request.json').read_text());bbox=tuple(request['crop_bbox'])
            x1,y1,x2,y2=bbox
            source=Image.open(args.data_root/'sources'/row['source_image']).convert('RGB')
            raw=Image.open(root/'raw_edited_crop.png').convert('RGB')
            target=mask_array(source.size,row['mask']).astype(bool);local=target[y1:y2,x1:x2]
            protect=mask_array(source.size,row['region_contract']['protected_mask'])
            raw_full=Image.open(args.raw_root/'edited'/row['image']).convert('RGB')
            phrase=(addition_phrase(row['editing_instruction']) if row['task_type']=='add' else
                    replacement_phrase(row) if args.composition_policy=='guarded-v1' else
                    row.get('replacement_target') or replacement_text_from_instruction(row['editing_instruction']))
            state=processor.set_image(raw)
            queries=[phrase]
            if args.composition_policy=='guarded-v1':
                short=re.split(r'\s+(?:wearing|holding|carrying|standing|sitting|hanging|positioned|located)\b',phrase,maxsplit=1,flags=re.I)[0]
                short=re.sub(r'^(?:a|an|the)\s+','',short,flags=re.I).strip()
                if short and short.lower() not in {'a','an','the'} and short.lower()!=re.sub(r'^(?:a|an|the)\s+','',phrase,flags=re.I).lower():
                    queries.append(short)
            candidates=[]
            attempted_queries=[]
            for query in queries:
                attempted_queries.append(query)
                output=processor.set_text_prompt(prompt=query,state=state)
                for mask,score in zip(output.get('masks',[]),output.get('scores',[])):
                    m=mask[0].detach().cpu().numpy().astype(bool);s=float(score)
                    intersection=(m&local).sum();union=(m|local).sum()
                    if row['task_type']=='add':
                        import cv2
                        distance=cv2.distanceTransform((~local).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
                        near=float(distance[m].min()) if m.any() else float('inf')
                        if s>=.35 and m.any() and m.sum()<=local.sum()*1.5 and near<=max(12,np.sqrt(local.sum())*.5):
                            candidates.append((intersection/union-near/max(1,np.sqrt(local.sum())),s,m))
                    elif s>=.35 and intersection/max(1,local.sum())>=.02 and .02<=m.sum()/local.sum()<=12:
                        candidates.append((intersection/union,s,m))
                if candidates:
                    phrase=query
                    break
            if not candidates:
                if args.composition_policy=='guarded-v1':
                    fallback(row,source,raw_full,target,protect,'new silhouette unresolved',phrase)
                else:
                    failures.append(dict(image=row['image'],phrase=phrase,reason='new silhouette unresolved'))
                continue
            overlap,score,best=max(candidates,key=lambda x:(x[0],x[1]))
            replacement=np.zeros_like(target);replacement[y1:y2,x1:x2]=best
            if row['task_type']=='add':
                final,alpha=compose_addition(source,raw,replacement,target,bbox,protect)
            else:
                final,alpha=compose_grounded_crop(source,raw,target,'replace',bbox,protect,replacement)
            # Addition composition intentionally discards host regeneration.
            # Measure preservation on the NEW item's silhouette, not the whole
            # host; a host-wide ratio falsely rejects valid small stickers.
            retention=composition_retention(source,raw_full,final,
                replacement if row['task_type']=='add' else target)
            if args.composition_policy=='guarded-v1' and retention['collapsed']:
                fallback(row,source,raw_full,target,protect,'composition_collapsed_edit',phrase,retention)
                continue
            final.save(args.out_root/'edited'/row['image']);alpha.save(args.out_root/'support'/row['image'])
            writable=np.zeros_like(target);writable[y1:y2,x1:x2]=np.asarray(alpha)>0
            changed=raw_full.tobytes()!=final.tobytes()
            record={**row,('added_mask' if row['task_type']=='add' else 'replacement_mask'):encode_mask(replacement),
                'edit_support_mask':encode_mask(writable),'composition_revision':{
                'method':'semantic_added_silhouette' if row['task_type']=='add' else 'semantic_old_new_union',
                'phrase':phrase,'score':score,'selection_score':float(overlap),
                'raw_request':str(root/'generation_request.json'),'changed_from_original':changed,
                'policy':args.composition_policy,'retention':retention,'attempted_queries':attempted_queries}}
            results.append(record)
            print(json.dumps(dict(image=row['image'],phrase=phrase,score=score,changed=changed)),flush=True)
    (args.out_root/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    (args.out_root/'summary.json').write_text(json.dumps(dict(cases=len(results),failures=failures,
        fallbacks=fallbacks,policy=args.composition_policy,
        load_seconds=load_seconds,wall_seconds=time.perf_counter()-started,diffusion_calls=0),indent=2))


if __name__=='__main__':main()
