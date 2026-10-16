"""Independent image + no-ref probe, without masks, refs or generated targets.

The current probe checks visible alternatives and explicit semantic exclusions
without exhaustive box enumeration. Member scopes are compared afterwards,
outside the model context; omitted example categories are diagnostic only.
Localized shared-fit auditing remains mandatory. Legacy records stay readable.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from PIL import Image

PROBE_VERSION = 'image-only-semantic-4'
PROMPT = '''This is a BLIND INTENT test, not an instruction to execute an edit. You see only the original image and an editing sentence, with no locator. "the selected region" intentionally hides the target. Do NOT ask for a mask. Do NOT rank by centrality, size or visual salience.
Does the wording/payload naturally allow multiple visible targets, or strongly imply one specific target? Consider normal photo editing, physical function and support. Multiple instances of the same class count as alternatives. Give two short visibly plausible examples if ambiguous; no bounding boxes or exhaustive instance list.
Report semantic_scope="broad" for edits naturally applicable across many visible classes; otherwise use "restricted". eligible_categories contains EXAMPLES, not an exhaustive list. Include parts when relevant; omission does not imply incompatibility. Also list excluded_categories only for VISIBLE categories explicitly incompatible with the FULL request. Do not infer exclusions from omission or visual salience. Think about normal use: a drink container naturally belongs on a table, not an occupied seat; a ribbon can attach to a sign as well as a tree. A dashed road marking is not a replacement for a window. Report "specific" if wording clearly favors one visible instance; "uncertain" if core evidence is insufficient.
Return compact JSON only:
{"blind_schema":4,"decision":"ambiguous|specific|uncertain","semantic_scope":"broad|restricted","eligible_categories":[],"excluded_categories":[],"examples":["visible alternative 1","visible alternative 2"],"evidence":"one short reason based on semantic fit, not salience"}.
This check tests blind ambiguity; separate localized checks must still verify EACH intended member's naturalness. Do not claim all hidden members are valid.
Editing request: '''


def _category_covered(scope, categories):
    """Conservative lexical coverage; never infer eligibility from a missing class."""
    import re
    aliases = {'lights':'lamp', 'light':'lamp', 'lantern':'lamp', 'bags':'bag',
               'backpack':'bag', 'sack':'bag', 'shutters':'shutter', 'chairs':'chair',
               'stool':'chair', 'armchair':'chair', 'bottles':'bottle', 'screens':'screen'}
    def words(value):
        return {aliases.get(w, w.rstrip('s') if w.endswith('s') and not w.endswith('ss') else w)
                for w in re.findall(r'[a-z]+',str(value).lower())
                if w not in {'the','a','an','small','large','object','part','surface','item'}}
    target = words(scope)
    # Qualifiers may differ while the physical object class is identical:
    # "canvas bag" and "military bag" must not fail solely on adjectives.
    # Keep host qualifiers for generic components (door/window handle, etc.).
    generic_parts = {'head', 'top', 'side', 'leg', 'arm', 'back', 'front',
                     'handle', 'strap', 'rim', 'edge', 'cover', 'base', 'frame', 'body'}
    def head(value):
        tokens = re.findall(r'[a-z]+', str(value).lower())
        return next(iter(words(tokens[-1])), '') if tokens else ''
    target_head = head(scope)
    # Match a category phrase or a more general head, but never via qualifiers
    # alone (e.g. metal window != metal chair).
    return any(target and (target <= words(c) or words(c) <= target or
                          (target_head and target_head not in generic_parts and target_head == head(c))) and words(c)
               for c in categories)


def evaluate_semantic_probe(group, raw):
    result = {'version':PROBE_VERSION,'pass':False,'decision':'uncertain',
              'input_contract':'clean_source_image_and_noref_only','raw':raw,
              'requires_localized_shared_edit_audit':True}
    if (raw.get('_generation_metadata') or {}).get('finish_reason') == 'length':
        return {**result,'reason':'truncated_blind_response'}
    if raw.get('decision') == 'specific':
        return {**result,'decision':'fail','reason':'blind_instruction_implies_specific_target'}
    examples = raw.get('examples')
    if (raw.get('decision') != 'ambiguous' or not isinstance(examples,list)
            or len({x.strip().casefold() for x in examples if isinstance(x,str) and x.strip()}) < 2
            or not str(raw.get('evidence','')).strip()):
        return {**result,'reason':'insufficient_blind_ambiguity_evidence'}
    if raw.get('blind_schema') == 4:
        categories, excluded = raw.get('eligible_categories'), raw.get('excluded_categories')
        if (raw.get('semantic_scope') not in {'broad', 'restricted'}
                or not isinstance(categories, list) or not isinstance(excluded, list)
                or not all(isinstance(c, str) and c.strip() for c in categories + excluded)):
            return {**result, 'reason':'missing_blind_semantic_scope'}
        covered = [_category_covered(m.get('scope',''), categories) for m in group['members']]
        incompatible = [_category_covered(m.get('scope',''), excluded) for m in group['members']]
        if any(incompatible):
            return {**result, 'decision':'fail', 'reason':'member_explicitly_incompatible_with_payload',
                    'coverage':covered, 'excluded_members':incompatible}
        # A short unlocalized example list cannot establish exhaustive scope.
        # This is only the blind-stage gate; final localized shared_edit_valid
        # and member naturalness must also pass before group acceptance.
        return {**result, 'pass':True, 'decision':'pass',
                'reason':'blind_ambiguous_pending_localized_shared_fit', 'coverage':covered,
                'category_omission_is_not_exclusion':True}
    broad_allowed = group.get('operation') == 'remove' or (
        group.get('operation') == 'attribute' and
        [a.get('property') for a in group.get('instruction_spec',{}).get('attributes',[])] == ['color'])
    if raw.get('semantic_scope') == 'broad' and broad_allowed:
        return {**result,'pass':True,'decision':'pass','reason':'broad_request_with_visible_alternatives'}
    categories = raw.get('eligible_categories')
    if raw.get('semantic_scope') != 'restricted' or not isinstance(categories,list) or not all(isinstance(c,str) for c in categories):
        return {**result,'reason':'missing_blind_semantic_scope'}
    covered = [_category_covered(m.get('scope',''), categories) for m in group['members']]
    if not all(covered):
        return {**result,'reason':'member_category_not_covered_by_blind_evidence','coverage':covered}
    return {**result,'pass':True,'decision':'pass','reason':'shared_semantic_scope_with_visible_alternatives','coverage':covered}


def probe_request(group):
    # Deliberately exclude IDs, categories, member count, refs, bboxes, masks,
    # marked planning views, targets and the planner's evidence.
    return [group['source_image']], PROMPT + group['instruction_noref']


def _iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/max(union, 1e-9)


def evaluate_probe(group, raw):
    if raw.get('blind_schema') in {3, 4}:
        return evaluate_semantic_probe(group, raw)
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
            'members':[(m['unit_id'],m['bbox'],m.get('scope')) for m in group['members']],
            'operation':group.get('operation'),'instruction_spec':group.get('instruction_spec'),
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
