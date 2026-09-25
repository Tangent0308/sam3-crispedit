"""Resolve declared co-removal geometry, never revalidate source target masks.

Experimental auxiliary SAM queries only. Original RLE remains byte-identical;
unresolved co-removal is deferred instead of using a bounding box as a mask.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.prepare_samtok_data import load_jsonl,write_jsonl,encode_rle
from utils.context_edit import protected_neighbors
from synthesis_pipeline.labeling_checkpoint import CaseCheckpoints, bind_settings, ensure_sources, file_digest


def _mask_bbox_iou(mask,bbox):
    """IoU between a candidate mask extent and a normalized planner box."""
    if bbox is None:return 0.0
    height,width=mask.shape
    ys,xs=np.nonzero(mask)
    if not len(xs):return 0.0
    predicted=np.array([xs.min()/width,ys.min()/height,(xs.max()+1)/width,(ys.max()+1)/height])
    planned=np.array(bbox,dtype=float)/1000
    intersection=max(0,min(predicted[2],planned[2])-max(predicted[0],planned[0]))*max(0,min(predicted[3],planned[3])-max(predicted[1],planned[1]))
    predicted_area=(predicted[2]-predicted[0])*(predicted[3]-predicted[1])
    planned_area=(planned[2]-planned[0])*(planned[3]-planned[1])
    return float(intersection/max(1e-9,predicted_area+planned_area-intersection))


def choose_auxiliary(candidates,point,target,protected,bbox=None,min_score=.35,min_bbox_iou=0.0,policy='legacy'):
    if policy not in {'legacy','ownership-v1'}:raise ValueError('Unknown auxiliary policy')
    height,width=target.shape;x=min(width-1,round(point[0]*width/1000));y=min(height-1,round(point[1]*height/1000))
    diag=(width**2+height**2)**.5
    target_distance=cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,5)
    valid=[];checks=[]
    for index,(mask,score) in enumerate(candidates):
        if mask.shape!=target.shape:raise ValueError('Auxiliary shape mismatch')
        area=int(mask.sum())
        if not area:continue
        distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
        bbox_iou=_mask_bbox_iou(mask,bbox)
        evidence=dict(index=index,score=score,area=area,point_distance=float(distance[y,x]),bbox_iou=bbox_iou,
            contact_distance=float(target_distance[mask].min()),protected_fraction=float((mask&protected).sum()/area),
            original_target_fraction=float((mask&target).sum()/area),
            target_covered_fraction=float((mask&target).sum()/max(1,target.sum())))
        # A small, point-bound part entirely inside the trusted source mask adds
        # no pixels. Do not turn its auxiliary confidence into a source-mask audit.
        already_covered=(not np.any(mask&~target) and mask[y,x]
            and evidence['target_covered_fraction']<.8)
        strongly_located=(policy=='ownership-v1' and score>=.8 and bbox_iou>=.6 and mask[y,x]
            and not protected[y,x] and evidence['original_target_fraction']<=.15
            and area<=height*width*.5)
        area_ok=area<=max(256,int(target.sum()*.75)) or strongly_located
        guard_limit=.15 if strongly_located else .05
        accepted=already_covered or (score>=min_score and bbox_iou>=min_bbox_iou and area_ok and distance[y,x]<=diag*.025
            and evidence['contact_distance']<=diag*.05 and evidence['protected_fraction']<=guard_limit
            and evidence['target_covered_fraction']<.8)
        evidence.update(auxiliary_policy=policy,strongly_located=bool(strongly_located),area_allowed=bool(area_ok),protected_fraction_limit=guard_limit)
        evidence['already_covered']=bool(already_covered)
        evidence['accepted']=bool(accepted);checks.append(evidence)
        if accepted:valid.append(dict(point_distance=float(distance[y,x]),bbox_iou=bbox_iou,
            score=score,index=index,mask=mask))
    if not valid:return None,checks
    valid.sort(key=lambda value:(value['point_distance'],-value['bbox_iou'],-value['score'],value['index']));best=valid[0]
    # Two different candidates equally consistent with the single point are
    # ambiguous; do not silently choose another owner's equipment.
    for alt in valid[1:]:
        iou=(best['mask']&alt['mask']).sum()/max(1,(best['mask']|alt['mask']).sum())
        comparable_box=(bbox is None or abs(best['bbox_iou']-alt['bbox_iou'])<=.15)
        if alt['point_distance']<=best['point_distance']+2 and iou<.5 and comparable_box:return None,checks
    return best['mask']&~protected,checks


def relation_queries(relation):
    """Return a bounded, backwards-compatible set of visual noun phrases."""
    raw=relation.get('segmentation_queries')
    if not isinstance(raw,list):raw=[]
    raw=[relation.get('segmentation_query')]+raw
    result=[];seen=set()
    for query in raw:
        if not isinstance(query,str):continue
        query=' '.join(query.split()).strip()
        key=query.casefold()
        if query and key not in seen:
            result.append(query);seen.add(key)
        if len(result)==3:break
    return result


def choose_retained(candidates,point,target,protected,bbox=None,min_score=.35,min_bbox_iou=0.0):
    """Ground a KEEP entity without applying accessory-size/removal rules.

    Never replace its segmentation with a box or shrink the trusted target.
    Reject candidates mostly containing the target, and ambiguous identities.
    """
    h,w=target.shape;x=min(w-1,round(point[0]*w/1000));y=min(h-1,round(point[1]*h/1000))
    diag=(w*w+h*h)**.5;valid=[];checks=[]
    for index,(mask,score) in enumerate(candidates):
        if mask.shape!=target.shape:raise ValueError('Retained shape mismatch')
        area=int(mask.sum())
        if not area:continue
        distance=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
        overlap=int((mask&target).sum());iou=_mask_bbox_iou(mask,bbox)
        accepted=(score>=min_score and iou>=max(.1,min_bbox_iou) and
            distance[y,x]<=diag*.015 and not target[y,x] and
            overlap/area<=.15 and overlap/max(1,int(target.sum()))<=.15)
        checks.append(dict(index=index,score=score,area=area,bbox_iou=iou,
            point_distance=float(distance[y,x]),original_target_fraction=overlap/area,
            accepted=bool(accepted),purpose='retain'))
        if accepted:valid.append((float(distance[y,x]),-iou,-score,index,mask&~target))
    if not valid:return None,checks
    valid.sort(key=lambda x:x[:4]);best=valid[0]
    for alt in valid[1:]:
        iou=(best[4]&alt[4]).sum()/max(1,(best[4]|alt[4]).sum())
        if alt[0]<=best[0]+2 and abs(alt[1]-best[1])<=.15 and iou<.5:return None,checks
    return best[4],checks


def conservative_keep_box(relation,target):
    """A KEEP-only fallback, explicitly not a segmentation or deletion mask.

    Protect unselected pixels in a planner box; never override the trusted target
    or separately grounded co-removals. An inside-target anchor is a scope
    conflict, not permission to apply this fallback.
    """
    box=relation.get('bbox');point=relation.get('point')
    if relation.get('action')!='keep' or _normalized_box_prompt(box) is None:return None
    if not isinstance(point,list) or len(point)!=2:return None
    if any(type(v)!=int or not 0<=v<=1000 for v in box+point):return None
    h,w=target.shape;x=min(w-1,max(0,round(point[0]*w/1000)));y=min(h-1,max(0,round(point[1]*h/1000)))
    if target[y,x]:return None
    x1,y1,x2,y2=[int(round(v*s/1000)) for v,s in zip(box,(w,h,w,h))]
    x1,x2=max(0,x1),min(w,x2);y1,y2=max(0,y1),min(h,y2)
    if not (x1<=x<x2 and y1<=y<y2):return None
    guard=np.zeros_like(target);guard[y1:y2,x1:x2]=True;guard&=~target
    return guard if guard.any() else None


def _sam_candidates(output):
    return [(m[0].detach().cpu().numpy().astype(bool),float(score))
        for m,score in zip(output.get('masks',[]),output.get('scores',[]))]


def _normalized_box_prompt(bbox):
    if not isinstance(bbox,list) or len(bbox)!=4:return None
    left,top,right,bottom=[value/1000 for value in bbox]
    if left>=right or top>=bottom:return None
    return [(left+right)/2,(top+bottom)/2,right-left,bottom-top]


def _record_checks(checks,records,mode):
    enriched=[]
    for check in checks:
        record=records[check['index']]
        enriched.append({**check,'query':record['query'],'queries':record['queries'],'grounding_mode':mode})
    return enriched


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--ground-keeps',action='store_true')
    p.add_argument('--auxiliary-policy',choices=['legacy','ownership-v1'],default='legacy')
    p.add_argument('--keep-fallback',choices=['none','box-protect-v1'],default='none')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=a.resume);ensure_sources(a.out_root,a.data_root/'sources')
    settings=dict(ground_keeps=a.ground_keeps,auxiliary_policy=a.auxiliary_policy,keep_fallback=a.keep_fallback)
    bind_settings(a.out_root,settings,a.resume);checkpoints=CaseCheckpoints(a.out_root,settings)
    torch.set_num_threads(8)
    (a.out_root/'support').mkdir(exist_ok=True);(a.out_root/'candidates').mkdir(exist_ok=True)
    rows=load_jsonl(a.data_root/'annotations.jsonl');results=[];records=[]
    start=time.perf_counter();processor=None;query_count=0;guided_count=0;encoded_sources=0;reused=0
    for row in rows:
        dependencies=dict(source_sha256=file_digest(a.data_root/'sources'/row['source_image']))
        saved=checkpoints.load(row,dependencies) if a.resume else None
        if saved is not None:
            records.append(saved['resolution'])
            if saved['annotation'] is not None:results.append(saved['annotation'])
            reused+=1
            continue
        if row['relation_status']!='accepted':
            record=dict(image=row['image'],status=row['relation_status']);records.append(record)
            checkpoints.save(row,dict(resolution=record,annotation=None),dependencies=dependencies)
            continue
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB');target=mask_array(source.size,row['mask']).astype(bool)
        rle=row.get('region_contract',{}).get('protected_mask');protected=(mask_array(source.size,rle).astype(bool) if rle else protected_neighbors(a.data_root,row,source.size))&~target
        plan=row['relation_plan'];extra=np.zeros_like(target);record=dict(image=row['image'],status='accepted',relations=[]);state=None;co_removals=[];candidate_file_index=0
        box_guard=np.zeros_like(target)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            # Resolve preservation before deletion, independent of model list order.
            for relation in sorted(plan['relations'],key=lambda r:r['action']!='keep'):
                keeping=relation['action']=='keep'
                if keeping and not a.ground_keeps:continue
                from functools import partial
                chooser=choose_retained if keeping else partial(choose_auxiliary,policy=a.auxiliary_policy)
                if processor is None:
                    from utils.runtime_paths import runtime_path
                    sys.path.insert(0,runtime_path('SAM3_SOURCE','/opt/tiger/tanyue/sam3-crispedit'))
                    from sam3.model_builder import build_sam3_image_model
                    from sam3.model.sam3_image_processor import Sam3Processor
                    model=build_sam3_image_model(checkpoint_path=runtime_path('SAM3_CHECKPOINT','/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt'),load_from_HF=False,device='cuda')
                    processor=Sam3Processor(model,confidence_threshold=.3)
                if state is None:
                    state=processor.set_image(source);encoded_sources+=1
                queries=relation_queries(relation)
                if not queries:
                    record['relations'].append(dict(relation=relation,queries=[],attempts=[],candidates=[],resolved=False,added_pixels=None))
                    record['status']='defer_unresolved_keep' if keeping else 'defer_unresolved_auxiliary';continue
                bbox=relation.get('bbox');box_prompt=_normalized_box_prompt(bbox);attempts=[];all_checks=[]
                selected=None;selected_mode=None;selected_query=None
                for query in queries:
                    processor.reset_all_prompts(state);output=processor.set_text_prompt(prompt=query,state=state);query_count+=1
                    candidates=_sam_candidates(output)
                    text_records=[dict(mask=mask,score=score,query=query,queries=[query]) for mask,score in candidates]
                    attempt_index=len(attempts);attempts.append(dict(query=query,mode='text',candidate_count=len(candidates)))
                    np.savez_compressed(a.out_root/'candidates'/f'{Path(row["image"]).stem}_q{candidate_file_index}.npz',
                        query=np.array(query),mode=np.array('text'),scores=np.array([score for _,score in candidates]),
                        **{f'mask_{i}':mask for i,(mask,_) in enumerate(candidates)})
                    candidate_file_index+=1
                    selected,checks=chooser(candidates,relation['point'],target,protected,bbox=bbox)
                    enriched=_record_checks(checks,text_records,'text');all_checks.extend(enriched)
                    if selected is not None:
                        selected_mode='text'
                        accepted=next(check for check in enriched if check['accepted'] and
                            np.array_equal(text_records[check['index']]['mask']&~(target if keeping else protected),selected))
                        selected_query=accepted['query'];break
                    if box_prompt is not None:
                        # The official positive-box prompt reuses this query's text
                        # features and the full-image visual encoding. Try aliases
                        # only while the previous name remains unresolved.
                        output=processor.add_geometric_prompt(box_prompt,True,state=state);guided_count+=1
                        candidates=_sam_candidates(output)
                        guided_records=[dict(mask=mask,score=score,query=query,queries=[query]) for mask,score in candidates]
                        attempt_index=len(attempts);attempts.append(dict(query=query,mode='text+box',candidate_count=len(candidates)))
                        np.savez_compressed(a.out_root/'candidates'/f'{Path(row["image"]).stem}_q{candidate_file_index}.npz',
                            query=np.array(query),mode=np.array('text+box'),scores=np.array([score for _,score in candidates]),
                            **{f'mask_{i}':mask for i,(mask,_) in enumerate(candidates)})
                        candidate_file_index+=1
                        selected,checks=chooser(candidates,relation['point'],target,protected,
                            bbox=bbox,min_score=.6,min_bbox_iou=.15)
                        enriched=_record_checks(checks,guided_records,'text+box');all_checks.extend(enriched)
                        if selected is not None:
                            selected_mode='text+box'
                            accepted=next(check for check in enriched if check['accepted'] and
                                np.array_equal(guided_records[check['index']]['mask']&~(target if keeping else protected),selected))
                            selected_query=accepted['query'];break
                if selected is None and keeping and a.keep_fallback=='box-protect-v1':
                    selected=conservative_keep_box(relation,target)
                    if selected is not None:selected_mode='conservative_box_not_segmentation'
                added_pixels=int((selected&~target).sum()) if selected is not None else None
                record['relations'].append(dict(relation=relation,queries=queries,attempts=attempts,candidates=all_checks,
                    resolved=selected is not None,selected_mode=selected_mode,selected_query=selected_query,added_pixels=added_pixels))
                if selected is None:record['status']='defer_unresolved_keep' if keeping else 'defer_unresolved_auxiliary'
                elif keeping:
                    if selected_mode=='conservative_box_not_segmentation':box_guard|=selected
                    else:protected|=selected
                else:
                    extra|=selected&~target
                    if added_pixels:co_removals.append(relation['description'])
        records.append(record);write_jsonl(a.out_root/'resolution.jsonl',records)
        if record['status']!='accepted':
            checkpoints.save(row,dict(resolution=record,annotation=None),dependencies=dependencies)
            continue
        execution=target|extra
        protected|=box_guard&~execution
        if np.any(execution&protected):raise ValueError('Removal and retained geometry overlap')
        contract={**row.get('region_contract',{}),'source_size':list(source.size),'status':'original','segmentation_target':plan['target'],'protected_mask':encode_rle(protected),
            'conservative_keep_box_mask':encode_rle(box_guard&~execution),'keep_fallback':a.keep_fallback}
        result={**row,'original_editing_instruction':row['editing_instruction'],'editing_instruction':plan['instruction'],'new_instruction':plan['instruction'],
            'region_contract':contract,'execution_region':dict(status='resolved_auxiliary',source_size=list(source.size),mask=encode_rle(execution),
                auxiliary_mask=encode_rle(extra),resolved_co_removals=co_removals,provenance='relation_v2_point_bbox_guided_auxiliary_segmentation',source_mask_unchanged=True),
            'relation_execution_context':'Keep '+ '; '.join(r['description'] for r in plan['relations'] if r['action']=='keep')+'. '+plan['reconstruction']}
        Image.fromarray(extra.astype(np.uint8)*255).save(a.out_root/'support'/row['image'])
        checkpoints.save(row,dict(resolution=record,annotation=result),
            artifacts=[a.out_root/'support'/row['image']],dependencies=dependencies)
        results.append(result);write_jsonl(a.out_root/'annotations.jsonl',results)
        print(json.dumps(dict(image=row['image'],status=record['status'],extra_pixels=int(extra.sum()))),flush=True)
    write_jsonl(a.out_root/'resolution.jsonl',records)
    write_jsonl(a.out_root/'annotations.jsonl',results)
    (a.out_root/'summary.json').write_text(json.dumps(dict(input_cases=len(rows),accepted=len(results),reused=reused,ground_keeps=a.ground_keeps,auxiliary_policy=a.auxiliary_policy,keep_fallback=a.keep_fallback,wall_seconds=time.perf_counter()-start,
        source_mask_audit_calls=0,auxiliary_segmentation_queries=query_count,auxiliary_box_guided_queries=guided_count,
        auxiliary_source_encodings=encoded_sources),indent=2))


if __name__=='__main__':main()
