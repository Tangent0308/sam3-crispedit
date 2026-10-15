"""Independent image + no-ref probe, without masks, refs or generated targets.

The model enumerates plausible scene targets. Matching to the actual members is
performed afterwards, outside the model context. Missing/ambiguous matches fail
closed rather than being interpreted as evidence of ambiguity.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from PIL import Image

PROBE_VERSION = 'image-only-preference-2'
PROMPT = '''This is a COUNTERFACTUAL AMBIGUITY TEST, NOT a request to execute an edit or locate an already marked region. The words 'the selected region' are an intentional placeholder. Imagine separately selecting each visible object/part, then ask whether the stated edit would be natural there. The missing locator is EXPECTED and is NEVER a reason to return an empty list. For example, 'Remove the selected region.' may reasonably refer to a removable lamp, a chair, a bag or a passerby; list the actual plausible alternatives visible in this photograph. A generic recoloring request can apply to several unrelated physical objects with different original colors. You see one unmarked original photograph and one editing request. Do not assume which region was selected. Independently enumerate up to 12 distinct visible objects or parts to which this exact request could naturally apply. Include equally plausible alternatives even if their categories differ. Judge the full payload/property in this scene, not just whether an edit is technically possible: already-present colors/items, incompatible replacement hardware/materials, essential supports and occupied chairs can make a request implausible.
For each plausible target give its short visible description, a tight bounding box [left,top,right,bottom] normalized to 0..1000 in the ORIGINAL image, naturalness from 0 to 1, and preference_weight from 0 to 1 measuring how likely this target is intended based ONLY on the semantic fit of image+request. Do not rank by image centrality, visual salience or object size alone. Give equal weights when there is no semantic reason to prefer one; do not invent differences merely to rank. Include the actual edited part/attachment site when the request specifies one. Weights need not sum to one. Describe one target per box, not a collection or an entire scene.
Return JSON only: {"targets":[{"description":"...","bbox_1000":[0,0,100,100],"naturalness":0.9,"preference_weight":0.5}],"evidence":"brief reason for any strong preference, or no clear preference"}. Return an empty targets list ONLY if no visible object or part could naturally receive this edit even if it were selected; ambiguity itself must produce multiple alternatives, not an empty list. Do not infer hidden masks or ask for them.
Editing request: '''


def probe_request(group):
    # Deliberately exclude IDs, categories, member count, refs, bboxes, masks,
    # marked planning views, targets and the planner's evidence.
    return [group['source_image']], PROMPT + group['instruction_noref']


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(union, 1e-9)


def evaluate_probe(group, raw):
    targets = raw.get('targets')
    result = {'version': PROBE_VERSION, 'pass': False, 'decision': 'uncertain',
              'input_contract': 'clean_source_image_and_noref_only', 'raw': raw,
              'thresholds': {'min_bbox_iou': .30, 'min_match_margin': .10,
                             'min_naturalness': .70, 'max_conditional_preference': 1/len(group['members'])+.15}}
    if (raw.get('_generation_metadata') or {}).get('finish_reason') == 'length':
        return {**result,'reason':'truncated_blind_response'}
    if not isinstance(targets, list) or not targets:
        return {**result, 'reason': 'missing_blind_candidates'}
    if not all(isinstance(t,dict) for t in targets):
        return {**result,'reason':'malformed_blind_response'}
    # Qwen sometimes uses its native bbox_2d key despite the requested name.
    # This is a schema alias only; never infer a missing box or coordinates.
    targets = [{**t,'bbox_1000':t.get('bbox_1000',t.get('bbox_2d'))} for t in targets]
    try:
        for t in targets:
            box = t['bbox_1000']
            if len(box)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1000 for x in box):
                raise ValueError('invalid box')
            if box[0]>=box[2] or box[1]>=box[3] or not str(t['description']).strip():
                raise ValueError('invalid target')
            for key in ('naturalness','preference_weight'):
                if type(t[key]) not in (float,int) or not math.isfinite(t[key]) or not 0<=t[key]<=1:
                    raise ValueError('invalid score')
    except (KeyError,TypeError,ValueError):
        return {**result,'reason':'malformed_blind_response'}
    with Image.open(group['source_image']) as image:
        width,height=image.size
    matches=[]
    for member in group['members']:
        box=[1000*v/(width if i%2==0 else height) for i,v in enumerate(member['bbox'])]
        ranking=sorted([(_iou(box,t['bbox_1000']),i) for i,t in enumerate(targets)],reverse=True)
        overlap,index=ranking[0]
        if overlap<.30 or len(ranking)>1 and overlap-ranking[1][0]<.10:
            return {**result,'reason':'blind_target_mapping_uncertain','matches':matches}
        t=targets[index]
        matches.append({'unit_id':member['unit_id'],'candidate_index':index,'bbox_iou':overlap,
                        'naturalness':t['naturalness'],'preference_weight':t['preference_weight']})
    if len({m['candidate_index'] for m in matches})!=len(matches):
        return {**result,'reason':'blind_target_mapping_not_unique','matches':matches}
    weights=[m['preference_weight'] for m in matches];total=sum(weights)
    if not total:
        return {**result,'reason':'zero_blind_preference','matches':matches}
    shares=[w/total for w in weights]
    natural=all(m['naturalness']>=.70 for m in matches)
    neutral=max(shares)<=1/len(matches)+.15 and min(shares)>=.5/len(matches)
    return {**result,'pass':natural and neutral,'decision':'pass' if natural and neutral else 'fail',
            'reason':'no_clear_member_preference' if natural and neutral else 'payload_favors_or_excludes_member',
            'matches':matches,'conditional_preference':shares}


def run_blind_probes(groups, model, output_dir):
    """One separate, stateless request per new group; checkpoint each response."""
    from .production_state import atomic_json
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();pending=[];inference_seconds=0.0
    for group in groups:
        images,prompt=probe_request(group)
        digest=hashlib.sha256(Path(images[0]).read_bytes()).hexdigest()
        fingerprint=hashlib.sha256(json.dumps({'image':digest,'prompt':prompt,
            'members':[(m['unit_id'],m['bbox']) for m in group['members']],
            'model':getattr(model,'model_id','unknown'),'version':PROBE_VERSION},sort_keys=True).encode()).hexdigest()
        path=directory/(group['twin_id']+'.json')
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get('fingerprint')!=fingerprint:raise ValueError('blind probe inputs changed; use a new run root')
            group['group_checks']['blind_disambiguation']=saved['result']
        else:pending.append((group,images,prompt,path,fingerprint))
    size=max(1,int(getattr(model,'batch_size',1)))
    for start in range(0,len(pending),size):
        batch=pending[start:start+size]
        batch_started=time.perf_counter()
        outputs=model.generate_multimage([(item[1],item[2]) for item in batch])
        inference_seconds += time.perf_counter()-batch_started
        if len(outputs)!=len(batch):raise RuntimeError('incomplete blind probe batch')
        for (group,images,prompt,path,fingerprint),raw in zip(batch,outputs):
            result=evaluate_probe(group,raw)
            result['model']=getattr(model,'model_id','unknown')
            result['calls']=1
            atomic_json(path,{'fingerprint':fingerprint,'images':images,'prompt':prompt,'result':result})
            group['group_checks']['blind_disambiguation']=result
    atomic_json(directory/'summary.json',{'groups':len(groups),'new_calls':len(pending),
        'pass':sum(g['group_checks']['blind_disambiguation']['pass'] for g in groups),
        'inference_seconds':inference_seconds,'seconds':time.perf_counter()-started,
        'version':PROBE_VERSION})
    return groups
