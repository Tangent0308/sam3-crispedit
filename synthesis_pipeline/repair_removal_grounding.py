"""One conditional VLM decision over saved auxiliary candidates, no new SAM call.

Candidate selection cannot bypass area, protection or proximity safety checks.
Original mask and concise instruction remain unchanged. Isolated experiment.
"""
import argparse
from copy import deepcopy
import json
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.generate_samtok_plan import parse_json_object
from synthesis_pipeline.prepare_samtok_data import load_jsonl, write_jsonl, encode_rle
from synthesis_pipeline.resolve_removal_relations import choose_auxiliary,relation_queries
from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox
from utils.context_edit import protected_neighbors
import utils.vlm_utils as vlm


def candidate_panel(source, candidates):
    if not candidates:
        raise ValueError('Cannot render an empty candidate set')
    tile=384;cols=min(3,len(candidates));rows=(len(candidates)+cols-1)//cols
    panel=Image.new('RGB',(cols*tile,rows*(tile+48)),'white');draw=ImageDraw.Draw(panel)
    w,h=source.size
    for index,(mask,_) in enumerate(candidates):
        box=padded_mask_bbox(mask,.6,96)
        pixels=np.array(source.crop(box));m=mask[box[1]:box[3],box[0]:box[2]].astype('uint8')
        contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pixels,contours,-1,(0,0,0),4);cv2.drawContours(pixels,contours,-1,(255,255,255),2)
        im=Image.fromarray(pixels);im.thumbnail((tile,tile))
        x=(index%cols)*tile;y=(index//cols)*(tile+48)
        norm=[round(v/s*1000) for v,s in zip(box,(w,h,w,h))]
        draw.text((x+5,y+5),f'CANDIDATE {index}; B/W CONTOUR',fill='black')
        draw.text((x+5,y+22),f'Full-photo crop: {norm}',fill='black')
        panel.paste(im,(x+(tile-im.width)//2,y+48))
    return panel


def safe_candidate_anchor(mask):
    if mask.ndim != 2 or not mask.any():
        raise ValueError('Candidate mask must be nonempty and two-dimensional')
    distance=cv2.distanceTransform(mask.astype('uint8'),cv2.DIST_L2,5)
    y,x=np.unravel_index(distance.argmax(),mask.shape);h,w=mask.shape
    return [round(x*1000/w),round(y*1000/h)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False)
    (a.out_root/'sources').symlink_to((a.root/'relations/sources').resolve())
    (a.out_root/'inputs').mkdir()
    plans={r['image']:r for r in load_jsonl(a.root/'relations/annotations.jsonl')}
    resolutions=load_jsonl(a.root/'regions/resolution.jsonl');outputs=[];records=[];backend=None;calls=0
    started=time.perf_counter()
    try:
        for resolution in resolutions:
            if resolution['status']!='defer_unresolved_auxiliary':continue
            row=deepcopy(plans[resolution['image']]);plan=row['relation_plan']
            source=Image.open(a.out_root/'sources'/row['source_image']).convert('RGB')
            target=mask_array(source.size,row['mask']).astype(bool)
            guard_rle=row.get('region_contract',{}).get('protected_mask')
            protected=(mask_array(source.size,guard_rle).astype(bool) if guard_rle else protected_neighbors(a.root/'relations',row,source.size))&~target
            cache={}
            for f in (a.root/'regions/candidates').glob(Path(row['image']).stem+'_q*.npz'):
                with np.load(f) as pack:
                    query=str(pack['query'])
                    if 'scores' in pack:
                        scores=pack['scores'].tolist()
                        candidates=[(pack[f'mask_{index}'].copy(),float(score)) for index,score in enumerate(scores)]
                    else:
                        relation_record=next(r for r in resolution['relations']
                            if r['relation']['segmentation_query']==query)
                        candidates=[(pack[f'mask_{c["index"]}'].copy(),c['score'])
                            for c in relation_record['candidates']]
                    cache.setdefault(query,[]).extend(candidates)
            extra=np.zeros_like(target);fixed=[];events=[];status='accepted'
            for relation in plan['relations']:
                if relation['action']=='keep':fixed.append(relation);continue
                candidates=[candidate for query in relation_queries(relation)
                    for candidate in cache.get(query,[])]
                selected,checks=choose_auxiliary(candidates,relation['point'],target,protected)
                if selected is not None:
                    extra|=selected;fixed.append(relation);continue
                if not candidates:
                    status='defer_no_candidates';continue
                montage=candidate_panel(source,candidates)
                stem=Path(row['image']).stem+f'_{len(events)}'
                montage.save(a.out_root/'inputs'/(stem+'.png'))
                prompt=('Image 1 is the clean full source photograph. Image 2 shows candidate masks '
                    'as BLACK/WHITE OUTLINES on separate photographic context crops, each with its '
                    'candidate ID and crop coordinates in Image 1 (0-1000). These are annotations, '
                    'not object colors. Resolve ONLY an accessory-ownership ambiguity for this edit. '
                    'Main target: '+plan['target']+'. Concise instruction: '+plan['instruction']
                    +'. Unresolved proposed accessory: '+json.dumps(relation,ensure_ascii=False)
                    +'. The previous point failed geometry checks. Inspect whether each candidate '
                    'actually belongs to the selected target, not a neighboring instance. Do not '
                    'substitute a different accessory type or select owner pixels as an accessory. Select ALL distinct visible '
                    'accessories needed for this relation, but no duplicate masks of the same item. '
                    'If none is visible because fully occluded, say not_visible; if visible but absent '
                    'from the candidates or ownership unclear, say unresolved. Never choose the nearest '
                    'candidate just to avoid rejection. Return ONLY JSON: decision (select, not_visible, '
                    'unresolved), selected (list of {candidate_id: integer, description: singular '
                    'photo-relative accessory description}), reason (one evidence-based sentence).')
                (a.out_root/'inputs'/(stem+'.txt')).write_text(prompt)
                if backend is None:
                    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
                    vlm.configure_backend('qwen38-vllm',model_id='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B',device='cuda:0',dtype='bf16')
                    backend=vlm.get_backend()
                raw=backend.chat_batch([[{'role':'user','content':[{'type':'image','image':source},
                    {'type':'image','image':montage},{'type':'text','text':prompt}]}]],max_new_tokens=512)[0];calls+=1
                decision=parse_json_object(raw);event=dict(relation=relation,response=decision,raw=raw,checks=[]);events.append(event)
                if not isinstance(decision,dict):status='invalid_repair';continue
                if decision.get('decision')=='not_visible' and decision.get('selected')==[]:continue
                choices=decision.get('selected')
                if (decision.get('decision')!='select' or not isinstance(choices,list) or not choices
                    or len(choices)>len(candidates) or any(not isinstance(c,dict) or type(c.get('candidate_id'))!=int
                    or not 0<=c['candidate_id']<len(candidates) or not isinstance(c.get('description'),str) or not c['description'].strip() for c in choices)):
                    status='defer_unresolved_auxiliary';continue
                if len({c['candidate_id'] for c in choices})!=len(choices):status='invalid_duplicate_candidates';continue
                selected_masks=[candidates[c['candidate_id']][0] for c in choices]
                if any((m&n).sum()/max(1,(m|n).sum())>.75 for i,m in enumerate(selected_masks) for n in selected_masks[i+1:]):
                    status='invalid_duplicate_candidate_geometry';continue
                for choice in choices:
                    mask,score=candidates[choice['candidate_id']];point=safe_candidate_anchor(mask)
                    selected,checks=choose_auxiliary([(mask,score)],point,target,protected)
                    event['checks'].append(dict(candidate_id=choice['candidate_id'],point=point,checks=checks))
                    if selected is None:status='defer_candidate_safety';continue
                    ys,xs=np.nonzero(mask);w,h=source.size
                    bbox=[round(xs.min()*1000/w),round(ys.min()*1000/h),round((xs.max()+1)*1000/w),round((ys.max()+1)*1000/h)]
                    fixed.append({**relation,'description':choice['description'],'point':point,'bbox':bbox,
                        'reason':decision.get('reason',''),'grounding_repair':'vlm_candidate_identity'})
                    extra|=selected
            record=dict(image=row['image'],status=status,events=events);records.append(record)
            if status=='accepted':
                row['relation_plan']={**plan,'relations':fixed}
                row.update(editing_instruction=plan['instruction'],new_instruction=plan['instruction'],
                    relation_grounding_repair=record,region_contract={**row.get('region_contract',{}),
                    'status':'original','source_size':list(source.size),'segmentation_target':plan['target'],'protected_mask':encode_rle(protected)},
                    execution_region=dict(status='resolved_auxiliary',source_size=list(source.size),
                        mask=encode_rle(target|extra),auxiliary_mask=encode_rle(extra),source_mask_unchanged=True,
                        provenance='conditional_vlm_saved_candidate_identity'))
                outputs.append(row)
            write_jsonl(a.out_root/'annotations.jsonl',outputs);write_jsonl(a.out_root/'repair.jsonl',records)
            print(json.dumps(record,ensure_ascii=False),flush=True)
    finally:
        if backend is not None:vlm.shutdown_backend()
    write_jsonl(a.out_root/'annotations.jsonl',outputs);write_jsonl(a.out_root/'repair.jsonl',records)
    (a.out_root/'summary.json').write_text(json.dumps(dict(cases=len(records),resolved=len(outputs),vlm_calls=calls,
        additional_sam_calls=0,wall_seconds=time.perf_counter()-started),indent=2))


if __name__=='__main__':main()
