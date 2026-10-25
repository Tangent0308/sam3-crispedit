"""One concise, three-image audit for object-adapted twin groups."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .fonts import load_font
from .full_contour_views import contour_pair
from .multi_component_audit import audit_schema, normalize_result
from .production_state import atomic_json, file_hash

VERSION = 'twins-aligned-three-images-17-source-advisory-baseline-prompt'
from .aligned_audit import PROMPT as ALIGNED_PROMPT, requests as audit_requests
PROMPT = ALIGNED_PROMPT + '\nTwin rows are independent results from the same source, not a merged image. Members share only operation type; payloads/values may differ. All members must pass.'




def views(group, directory):
    from .aligned_audit import views as aligned_views
    return aligned_views(group['source_image'], group['members'],
                         [m['edited_image'] for m in group['members']], directory)


def evidence_schema(ids):
    from .aligned_audit import schema
    return schema(ids)


def audit(groups, model, directory):
    from .group_audit import refresh_twin_add_checks, _call_batched, _apply_joint_twin_audit
    from .twin_contract import adapted
    directory=Path(directory); jobs=[]; pending=[]
    # Optional verifier: its member jobs go out in the same batch as the group's audit.
    verifier=getattr(model,'verifier',None)
    # Optional single-target salvage: a group that fails only group-level checks (e.g. the
    # member size ratio) is still audited so members that pass on their own can be kept.
    keep_singles=getattr(model,'keep_singles',False)
    from .single_salvage import twin_member_eligible, twin_singles
    for group in groups:
        if not adapted(group):
            raise ValueError('concise twin audit requires the same-operation adapted contract')
        checks=group.setdefault('group_checks',{})
        if checks.get('generation_skipped') or not refresh_twin_add_checks(group):
            checks['group_pass']=False
            if not (keep_singles and not checks.get('generation_skipped')
                    and any(twin_member_eligible(group,m) for m in group['members'])):
                continue
        out=directory/group['twin_id']
        if getattr(model, 'grounding', None):
            for member in group['members']:
                member['grounding_evidence'] = model.grounding.inspect(
                    group['source_image'], member, member['edited_image'])
        images=views(group,out)
        size=Image.open(group['source_image']).size
        boxes=json.loads((out/'result_crop_boxes.json').read_text())
        coordinates={r['unit_id']:[round(100*v/size[i%2],1) for i,v in enumerate(r['box'])] for r in boxes}
        jobs.append(dict(job_id=group['twin_id'],images=images,
            prompt=PROMPT+'\nImages 2/3 row order, TOP to BOTTOM: '+', '.join(m['unit_id'] for m in group['members'])+'. Result crop boxes in source percent [left,top,right,bottom]: '+json.dumps(coordinates)+'.\nRequests:\n'+audit_requests(group['members'], group['source_image'], [m['edited_image'] for m in group['members']]),
            output_schema=evidence_schema([m['unit_id'] for m in group['members']]),
            max_tokens=256+360*len(group['members']),image_max_edge=1800,
            enable_thinking=True,thinking_token_budget=256))
        extra=[]
        if verifier is not None:
            from .edit_verifier import member_jobs
            extra=member_jobs(group['members'],group['source_image'],[m['edited_image'] for m in group['members']],
                              out,verifier,prefix=group['twin_id'])
            jobs.extend(extra)
        pending.append((group,len(jobs)-1-len(extra),len(extra)))
    results=_call_batched(model,jobs)
    for group,index,count in pending:
        result,job=results[index],jobs[index]
        raw=deepcopy(result.get('semantic_audit') or {})
        if result.get('finish_reason')=='length':raw['_parse_error']='truncated response'
        reconcile_records,reconcile_calls=[],0
        if getattr(model,'ref_reconcile',False):
            from .audit_reconcile import reconcile
            raw,reconcile_records,reconcile_calls=reconcile(raw,group['members'],group,model,prefix=group['twin_id'])
        rows=normalize_result(raw,group['members'])
        if verifier is not None:
            from .edit_verifier import record
            record(group['members'],rows,results[index+1:index+1+count])
        verifier_calls=count
        members=[]
        for m in group['members']:
            row=rows[m['unit_id']];ok=row['decision']=='pass'
            members.append({**row, 'reference_correct': row.get('ref_matches_mask') is True,
                'single_instance':ok, 'scope_correct':ok, 'operation_correct':ok,
                'target_done':ok, 'non_target_preserved':ok, 'natural':ok})
        adapted_result={**result,'semantic_audit':dict(members=members,
            member_edits_valid=all(r['decision']=='pass' for r in members),
            **({'_parse_error':raw['_parse_error']} if raw.get('_parse_error') else {}))}
        _apply_joint_twin_audit(group,adapted_result)
        checks=group['group_checks'];checks['audit_mode']=VERSION
        if checks.get('blind_protocol',{}).get('pass') is False:
            checks['group_pass']=False
        if keep_singles:
            group['singles']=twin_singles(group,rows)
        out=directory/group['twin_id']
        evidence=dict(version=VERSION,raw=raw,normalized=rows,
            group_pass=checks['group_pass'],images=job['images'],
            image_hashes={p:file_hash(p) for p in job['images']},
            inference={k:result.get(k) for k in ('input_tokens','output_tokens','finish_reason')})
        if verifier is not None:
            from .edit_verifier import contract
            evidence.update(verifier=contract(verifier),verifier_calls=verifier_calls)
        if getattr(model,'ref_reconcile',False):
            from .audit_reconcile import VERSION as RECONCILE_VERSION
            evidence.update(ref_reconcile_version=RECONCILE_VERSION,ref_reconcile=reconcile_records,
                            ref_reconcile_calls=reconcile_calls)
        atomic_json(out/'audit.json',evidence)
        checks['compact_audit_path']=str(out/'audit.json')
    return groups
