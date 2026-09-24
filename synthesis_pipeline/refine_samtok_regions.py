"""Refine source edit units and protect visible neighboring SAM3 instances.

This stage changes effective masks explicitly; original SAMTok masks are retained.
Unresolved segmentation is recorded, never silently treated as confirmed geometry.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask

from synthesis_pipeline.audit_edit_pairs import mask_array
from utils.context_edit import protected_neighbors


def encode_mask(mask):
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {**rle, 'counts': rle['counts'].decode('ascii')}


def select_target(candidates, anchor, policy, min_containment=.8):
    """Reject unrelated/oversized segments instead of moving the target."""
    matches = []
    for index, (candidate, score) in enumerate(candidates):
        intersection = int((candidate & anchor).sum())
        if not intersection or not candidate.any():
            continue
        coverage = intersection / anchor.sum()
        containment = intersection / candidate.sum()
        ratio = candidate.sum() / anchor.sum()
        if policy == 'complete' and not (coverage >= .85 and .85 <= ratio <= 2.5):
            continue
        if policy == 'surface' and not (containment >= min_containment and .03 <= ratio <= 1.2):
            continue
        if score < .35:
            continue
        iou = intersection / (candidate | anchor).sum()
        matches.append((iou, float(score), index, candidate))
    if not matches:
        return None, {'status': 'unresolved', 'policy': policy, 'candidate_count': len(candidates)}
    _, score, index, candidate = max(matches, key=lambda x: (x[0], x[1]))
    effective = (candidate | anchor) if policy == 'complete' else (candidate & anchor)
    return effective, {'status': 'segmented_candidate', 'policy': policy, 'candidate_index': index,
                       'score': score, 'area_ratio': float(effective.sum() / anchor.sum()),
                       'candidate_containment': float((candidate & anchor).sum()/candidate.sum())}


def declared_carried_queries(row):
    """Inspect items the planner itself attributes to the selected owner.

    No ownership is inferred from mere visual proximity or arbitrary scene text.
    """
    if row['task_type'] not in {'remove', 'replace'}:
        return []
    text = str(row.get('masked_content', ''))
    queries = []
    for match in re.finditer(r'\b(?:holding|carrying|supporting)\s+([^.;]+)', text, re.I):
        phrase = re.split(r'\b(?:in|with|beside|near|behind|while|who|next to|in front of)\b', match[1], maxsplit=1, flags=re.I)[0]
        for part in re.split(r'\band\b|,', phrase):
            part = re.sub(r'^(?:a|an|the)\s+', '', part.strip(), flags=re.I)
            if 1 <= len(part.split()) <= 6:
                queries.append(part)
    for match in re.finditer(r'(?:,|\band\b)\s*(?:a |an |the )?([a-z][a-z -]{0,60}?)\s+(?:held|carried)\b', text, re.I):
        phrase = re.sub(r'^and\s+', '', match[1].strip(), flags=re.I)
        phrase = re.sub(r'^(?:a|an|the)\s+', '', phrase, flags=re.I)
        tail = text[match.end():]
        if not re.match(r'\s+by\s+(?:another|a different)\b', tail, re.I) and 1 <= len(phrase.split()) <= 6:
            queries.append(phrase)
    return list(dict.fromkeys(queries))


def carried_outside(candidates, anchor):
    import cv2
    distance = cv2.distanceTransform((~anchor).astype(np.uint8),cv2.DIST_L2,5)
    for candidate, score in candidates:
        if score < .5 or not candidate.any():
            continue
        contained = float((candidate & anchor).sum()/candidate.sum())
        if float(distance[candidate].min()) <= 6 and contained < .9 and (candidate & ~anchor).sum() >= 32:
            return {'score': float(score), 'contained_fraction': contained}
    return None


def external_accessory_risk(row, segment, anchor):
    """Conservatively quarantine external contacts missed by the text inventory.

    This is a risk signal, NOT an assertion of ownership from proximity. The
    fixed terms are segmentation checks, never suggested editing additions.
    """
    if row['task_type'] not in {'remove', 'replace'} or not re.search(
        r'\b(?:person|man|woman|boy|girl|child|player|batter|pedestrian)\b',
        str(row.get('masked_content','')), re.I):
        return None
    for query in ('bag', 'umbrella', 'handheld object'):
        small = [(m,s) for m,s in segment(query) if m.sum() <= anchor.sum()*.75]
        risk = carried_outside(small, anchor)
        if risk:
            return {'query':query, **risk}
    return None


def neighbor_union(candidates, target):
    result = np.zeros_like(target)
    for mask, score in candidates:
        if score < .4 or not mask.any():
            continue
        # A full object that contains an attribute surface belongs to the same
        # instance; preserving it would cut out the requested surface edit.
        overlap = int((mask & target).sum())
        if overlap / target.sum() > .25 or overlap / mask.sum() > .25:
            continue
        result |= mask
    return result & ~target


def attach_structural_part(candidates, anchor):
    """Select a small segmented component touching the selected object only.

    This is a semantic component query, not a union of arbitrary changed pixels.
    Competing nearby components remain unresolved.
    """
    import cv2
    distance = cv2.distanceTransform((~anchor).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    eligible = []
    for candidate, score in candidates:
        if score < .35 or not candidate.any() or candidate.sum() > anchor.sum()*1.5:
            continue
        gap = float(distance[candidate].min())
        if gap <= 6 and (candidate & ~anchor).sum() > candidate.sum()*.5:
            eligible.append((gap,-float(score),candidate))
    eligible.sort(key=lambda x:(x[0],x[1]))
    if not eligible:
        return None
    best=eligible[0][2]
    for gap,score,other in eligible[1:]:
        if gap <= eligible[0][0]+2 and (best&other).sum()/max(1,(best|other).sum()) < .5:
            return None
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--out-root', type=Path, required=True)
    p.add_argument('--overrides', type=Path, help='Explicit, provenance-labelled regression proposals')
    p.add_argument('--ids', default='')
    p.add_argument('--verify-original-scope', action='store_true',
                   help='Require a source semantic match for add anchors and refine every attribute to its requested source surface')
    p.add_argument('--sam3-root', type=Path, default=Path('/opt/tiger/tanyue/sam3-crispedit'))
    p.add_argument('--checkpoint', default='/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt')
    args = p.parse_args()
    sys.path.insert(0, str(args.sam3_root))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    started = time.perf_counter()
    rows = [json.loads(x) for x in (args.data_root / 'annotations.jsonl').read_text().splitlines()]
    all_rows = rows
    if args.ids:
        wanted = {int(x) for x in args.ids.split(',')}
        rows = [r for r in rows if int(r['image'].split('_')[0]) in wanted]
    overrides = {int(r['id']): r for r in map(json.loads, args.overrides.read_text().splitlines())} if args.overrides else {}
    args.out_root.mkdir(parents=True, exist_ok=False)
    (args.out_root / 'sources').symlink_to((args.data_root / 'sources').resolve())
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False, device='cuda')
    processor = Sam3Processor(model, confidence_threshold=.3)
    load_seconds = time.perf_counter() - started
    results, evidence = [], []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for original in rows:
            case_start = time.perf_counter()
            row = dict(original)
            proposal = overrides.get(int(row['image'].split('_')[0]), {})
            row.update({k: v for k, v in proposal.items() if k not in {'id', 'policy', 'provenance'}})
            source = Image.open(args.data_root / 'sources' / row['source_image']).convert('RGB')
            anchor = mask_array(source.size, original['mask']).astype(bool)
            state = processor.set_image(source)
            cache = {}
            crop_state = None
            def segment(text):
                if text not in cache:
                    processor.reset_all_prompts(state)
                    output = processor.set_text_prompt(prompt=text, state=state)
                    masks = output.get('masks', [])
                    scores = output.get('scores', [])
                    cache[text] = [(m[0].detach().cpu().numpy().astype(bool), float(s)) for m,s in zip(masks,scores)]
                return cache[text]
            def local_segment(text):
                nonlocal crop_state
                from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox
                key = 'local::'+text
                bbox = padded_mask_bbox(anchor, padding_fraction=.4, min_padding=48)
                crop = source.crop(bbox)
                scale = min(4., 1024/max(crop.size))
                size = tuple(max(1,round(x*scale)) for x in crop.size)
                if crop_state is None:
                    crop_state = processor.set_image(crop.resize(size, Image.Resampling.LANCZOS))
                if key not in cache:
                    processor.reset_all_prompts(crop_state)
                    output = processor.set_text_prompt(prompt=text, state=crop_state)
                    mapped = []
                    for m,s in zip(output.get('masks',[]),output.get('scores',[])):
                        pixels = m[0].detach().cpu().numpy().astype(np.uint8)*255
                        local = np.asarray(Image.fromarray(pixels).resize(crop.size,Image.Resampling.NEAREST))>0
                        full = np.zeros_like(anchor)
                        full[bbox[1]:bbox[3],bbox[0]:bbox[2]] = local
                        mapped.append((full,float(s)))
                    cache[key] = mapped
                return cache[key]
            policy = proposal.get('policy', row.get('mask_refinement', 'original'))
            declared_policy = policy
            if args.verify_original_scope and row['task_type'] == 'attribute':
                policy = 'surface'
            phrase = row.get('segmentation_target', '')
            if policy not in {'original', 'complete', 'surface'}:
                raise ValueError(f'Unknown mask refinement {policy}')
            target = anchor
            status = {'status': 'original', 'policy': policy}
            if policy != 'original':
                if not phrase:
                    raise ValueError('A refinement needs a segmentation_target phrase')
                containment = .98 if args.verify_original_scope and policy == 'surface' else .8
                target, status = select_target(segment(phrase), anchor, policy, containment)
                if target is None and args.verify_original_scope:
                    target, status = select_target(local_segment(phrase), anchor, policy, containment)
                    status['query_view'] = 'local_context'
                if policy == 'complete' and row.get('structural_parts'):
                    completed = anchor.copy() if target is None else target.copy()
                    parts = [attach_structural_part(segment(part),anchor) for part in row['structural_parts']]
                    if all(part is not None for part in parts):
                        for part in parts:
                            completed |= part
                        target = completed
                        status = {'status':'segmented_candidate','policy':policy,
                                  'method':'anchor_with_semantically_segmented_attached_parts',
                                  'structural_parts':row['structural_parts'],
                                  'area_ratio':float(target.sum()/anchor.sum())}
                    else:
                        target=None;status={'status':'unresolved','policy':policy,'reason':'missing_or_ambiguous_structural_part'}
            elif args.verify_original_scope and row['task_type'] == 'add':
                matched, match_evidence = select_target(segment(phrase), anchor, 'surface') if phrase else (None, {})
                if matched is None and phrase:
                    matched, match_evidence = select_target(local_segment(phrase),anchor,'surface')
                parent_phrase = row.get('scope_preflight',{}).get('original_segmentation_target')
                if matched is None and parent_phrase and parent_phrase != phrase:
                    matched, match_evidence = select_target(segment(parent_phrase),anchor,'surface')
                    match_evidence['fallback_host_query'] = parent_phrase
                if matched is None:
                    target = None
                    status = {'status':'unresolved', 'policy':'original',
                              'reason':'placement_anchor_query_does_not_match_source_mask'}
                else:
                    status = {'status':'original', 'policy':'original',
                              'semantic_anchor_match': {k:v for k,v in match_evidence.items() if k!='status'}}
            if args.verify_original_scope:
                status['declared_policy'] = declared_policy
                status['scope_verification'] = 'source_semantic_match_v1'
                for carried in declared_carried_queries(row):
                    conflict = carried_outside(segment(carried), anchor)
                    if conflict:
                        target = None
                        status = {**status, 'status':'unresolved',
                                  'reason':'declared_carried_item_outside_edit_unit',
                                  'carried_query':carried, 'carried_evidence':conflict}
                        break
                if target is not None:
                    risk = external_accessory_risk(row, segment, anchor)
                    if risk:
                        target = None
                        status = {**status, 'status':'unresolved',
                                  'reason':'ambiguous_external_accessory_contact',
                                  'accessory_contact':risk}
            directory = args.out_root / 'region_evidence' / Path(row['image']).stem
            directory.mkdir(parents=True)
            if target is None:
                target = anchor  # Saved for inspection only; contract blocks generation.
            protect = protected_neighbors(args.data_root, original, source.size) & ~target
            queries = list(dict.fromkeys(([phrase] if phrase else []) + row.get('protected_objects', [])))
            for query in queries:
                protect |= neighbor_union(segment(query), target)
            for name, mask in [('original_mask',anchor), ('effective_mask',target), ('protected_mask',protect)]:
                Image.fromarray(mask.astype(np.uint8)*255).save(directory / f'{name}.png')
            for qi, (query, candidates) in enumerate(cache.items()):
                for j, (mask, score) in enumerate(candidates):
                    Image.fromarray(mask.astype(np.uint8)*255).save(directory / f'query{qi}_instance{j}_{score:.2f}.png')
            row['sam_target_mask'] = original.get('sam_target_mask', original['mask'])
            row['mask'] = encode_mask(target)
            row['region_contract'] = {**status, 'version': 1, 'protected_mask': encode_mask(protect),
                'segmentation_target': phrase, 'source_size': list(source.size),
                'provenance': proposal.get('provenance', 'planner_and_sam3_candidate'),
                'original_instruction': original['editing_instruction'], 'original_task_type': original['task_type']}
            record = dict(image=row['image'], contract={k:v for k,v in row['region_contract'].items() if k!='protected_mask'},
                          queries=[{'text':q,'instances':len(v),'scores':[s for _,s in v]} for q,v in cache.items()],
                          seconds=time.perf_counter()-case_start)
            (directory/'evidence.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
            evidence.append(record);results.append(row)
            print(json.dumps(record,ensure_ascii=False),flush=True)
    (args.out_root/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
    frozen = args.data_root/'input_annotations.jsonl'
    (args.out_root/'input_annotations.jsonl').write_text(frozen.read_text() if frozen.exists() else ''.join(json.dumps(r)+'\n' for r in all_rows))
    (args.out_root/'summary.json').write_text(json.dumps(dict(cases=len(results),load_seconds=load_seconds,
        wall_seconds=time.perf_counter()-started,unresolved=sum(r['region_contract']['status']=='unresolved' for r in results)),indent=2))


if __name__ == '__main__':
    main()
