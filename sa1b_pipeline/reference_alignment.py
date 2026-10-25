"""Opt-in full-scene reference evidence and one bounded planning correction.

Segmentation never rejects a plan or changes its mask. Only flagged requests
receive one extra planning call, with the original two images and operation.
Unflagged evidence is not proof of correctness. Not enabled by default.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .geometry import load_mask
from .production_state import atomic_json, digest, file_hash

VERSION = 'global-reference-advisory-3-candidate-evidence'


def candidate_advisories(candidates):
    """Score-first evidence avoids picking a candidate solely to fit the mask."""
    top = max(candidates, key=lambda c: c['score'], default=None)
    flags = []
    if top and top['score'] >= .65:
        if top['iou'] < .05:
            flags.append('high_score_candidate_disjoint')
        if top['selector_coverage'] >= .85 and top['candidate_precision'] < .3:
            flags.append('candidate_much_broader_than_selector')
        if top['candidate_precision'] >= .85 and top['selector_coverage'] < .3:
            flags.append('candidate_much_narrower_than_selector')
    return flags


def assess_candidates(candidates):
    """A disagreement trigger, never a grounding verdict or a mask selector.

    A text segmenter often ranks the wrong instance first. Any returned
    candidate that substantially matches both extents vetoes automatic repair;
    its existence does not establish that the referring phrase is unique.
    Missing/weak detections are unknown, not evidence for rejection.
    """
    import math
    fields = ('score', 'iou', 'selector_coverage', 'candidate_precision')
    for c in candidates:
        if any(not isinstance(c.get(k), (int, float)) or
               not math.isfinite(c[k]) or not 0 <= c[k] <= 1 for k in fields):
            return {'status': 'invalid_evidence', 'advisory': [], 'repair': False}
    aligned = [c for c in candidates
               if c['selector_coverage'] >= .65 and c['candidate_precision'] >= .65]
    if aligned:
        return {'status': 'compatible_candidate_exists', 'advisory': [],
                'repair': False, 'compatible_indices': [c.get('index') for c in aligned]}
    flags = candidate_advisories(candidates)
    return {'status': 'unopposed_disagreement' if flags else 'unverified',
            'advisory': flags, 'repair': bool(flags)}


def bounded_repair_requests(planner, requests, inspector, *, max_extra_calls):
    """Opt-in experiment: one extra call per target within a shared hard cap.

    Keep unresolved drafts explicitly unresolved. A successful parse or a SAM
    candidate is not semantic approval. No output here authorizes publishing.
    Callers pass the remaining budget after any other recovery attempts.
    """
    from .lean_retry import _requests
    from .per_mask_planning import combine
    if not isinstance(max_extra_calls, int) or max_extra_calls < 0:
        raise ValueError('max_extra_calls must be a nonnegative integer')
    repairs = []
    for req in requests:
        draft = req['raw']
        if any(k in req for k in ('first_raw', 'pre_reference_raw', 'bounded_repair')):
            continue  # Never stack recovery calls or recursively retry.
        op = req['item']['_mask_assignments'][req['unit_id']]
        check = combine([draft], 'multi', {'U1': op}, [{'object_id': 'check'}])
        if not check['targets'][0]['feasible']:
            continue
        metadata = json.loads((Path(req['views'][0]).parent/'views.json').read_text())
        matches = [t for t in metadata['targets'] if t['unit_id'] == req['unit_id']]
        if len(matches) != 1:
            raise ValueError('reference evidence mask binding is ambiguous')
        target = matches[0]
        ev = inspector.inspect_reference(metadata['source'], target['mask_path'], draft['ref'])
        decision = assess_candidates(ev['candidates'])
        req['reference_evidence'] = ev
        req['reference_decision'] = decision
        if not decision['repair']:
            continue
        req['reference_repair_status'] = 'budget_exhausted'
        if len(repairs) >= max_extra_calls:
            continue
        repair = {**req, 'path': req['path'].with_name(req['path'].stem+'_bounded_ref_v1.json'),
                  'views': repair_views(req, metadata, target),
                  'prompt': correction_prompt(req['prompt'], draft, decision),
                  'cache_context': {**req.get('cache_context', {}),
                      'bounded_reference': '1', 'evidence_digest': digest(ev)},
                  'original_request': req, 'source': metadata['source'],
                  'selector': target['mask_path']}
        repair['cache_context']['image_sha256'] = [file_hash(p) for p in repair['views']]
        repair.pop('raw', None)
        repairs.append(repair)
        req['reference_repair_status'] = 'scheduled'
    for start in range(0, len(repairs), max(1, planner.batch_size)):
        _requests(planner, repairs[start:start+max(1, planner.batch_size)])
    for repair in repairs:
        req, answer = repair['original_request'], repair['raw']
        op = req['item']['_mask_assignments'][req['unit_id']]
        check = combine([answer], 'multi', {'U1': op}, [{'object_id': 'check'}])
        target = check['targets'][0]
        malformed = (bool(answer.get('_parse_error')) or bool(target.get('_format_error')) or
                     answer.get('_generation_metadata', {}).get('finish_reason') == 'length')
        repair['applied'] = not malformed
        req['bounded_repair'] = True
        req['pre_reference_raw'] = deepcopy(req['raw'])
        if malformed:
            req['reference_repair_status'] = 'invalid_revision'
            continue
        req['raw'] = answer
        if not target['feasible']:
            req['reference_repair_status'] = 'revised_plan_rejected'
            continue
        ev = inspector.inspect_reference(repair['source'], repair['selector'], answer['ref'])
        decision = assess_candidates(ev['candidates'])
        req['post_reference_evidence'] = ev
        req['post_reference_decision'] = decision
        req['reference_repair_status'] = ('geometrically_compatible_unverified' if
            decision['status'] == 'compatible_candidate_exists' else 'unresolved')
    return repairs


class ReferenceAlignment:
    def __init__(self, resolver, directory):
        self.resolver = resolver
        self.directory = Path(directory)

    def inspect_reference(self, source_path, mask_path, ref):
        import torch
        key = digest(dict(version=VERSION, model=self.resolver.model_path,
                          source=file_hash(source_path), mask=file_hash(mask_path), ref=ref))
        path = self.directory/key/'evidence.json'
        if path.exists():
            return json.loads(path.read_text())
        source = Image.open(source_path).convert('RGB')
        mask = load_mask({'mask_path': mask_path}, source.size)
        if not mask.any():
            raise ValueError('empty reference-alignment selector')
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.resolver.processor(
                images=source, text=ref, return_tensors='pt').to(self.resolver.device)
            outputs = self.resolver.model(**inputs)
            raw = self.resolver.processor.post_process_instance_segmentation(
                outputs, threshold=.3, target_sizes=[source.size[::-1]])[0]
        candidates = []
        path.parent.mkdir(parents=True, exist_ok=True)
        for index, (region, score) in enumerate(zip(raw['masks'], raw['scores'])):
            region = region.cpu().numpy().astype(bool)
            overlap = int((region & mask).sum())
            mask_file = path.parent / f'candidate_{index}.png'
            Image.fromarray(region.astype('uint8') * 255).save(mask_file)
            candidates.append(dict(index=index, score=float(score), mask_path=str(mask_file),
                iou=overlap/max(1, int((region | mask).sum())),
                selector_coverage=overlap/int(mask.sum()),
                candidate_precision=overlap/max(1, int(region.sum()))))
        result = dict(version=VERSION, ref=ref, candidates=candidates,
                      advisory=candidate_advisories(candidates),
                      status='advisory_only', seconds=time.monotonic()-start)
        atomic_json(path, result)
        return result


def correction_prompt(original_prompt, draft, evidence):
    from .per_mask_planning import INTRO
    intro = ('Image 1 is the clean full source. Image 2 shows the SAME source crop twice: '
             'LEFT original context with a green selection outline; RIGHT original selected '
             'pixels with all excluded pixels HATCHED. A recognizable object inside a hatched '
             'hole is NOT selected. Locate the unhatched region in Image 1.\n')
    return (original_prompt.replace(INTRO, intro) +
        '\nRecheck this draft against the SAME selected pixels; keep the assigned operation. '
        'An independent text segmenter disagrees with its location/extent, but may be wrong. '
        'Use the two images to decide; keep a correct draft, otherwise redesign ref/edit for '
        'the actual selected entity. Never move or enlarge the mask to fit the draft. '
        'Return the same ref/edit-or-reject JSON, without explanation fields.\nDraft: ' +
        json.dumps({k:v for k,v in draft.items() if k in ('ref','edit')}, ensure_ascii=False) +
        '\nFallible disagreement: ' + json.dumps(evidence.get('advisory', [])))


def repair_views(req, metadata, target):
    """Keep two input images; make excluded pixels explicit in the second."""
    from PIL import ImageOps, ImageDraw
    from scipy.ndimage import binary_dilation
    from .per_mask_planning import excluded_pixels_view
    source = Image.open(metadata['source']).convert('RGB')
    mask = load_mask({'mask_path':target['mask_path']}, source.size)
    box = target['crop_box_xyxy']
    tile = ImageOps.contain(source.crop(box), (750,750))
    local = np.asarray(Image.fromarray(mask.astype('uint8')*255).crop(box).resize(
        tile.size, Image.Resampling.NEAREST)) > 0
    pixels = np.asarray(tile).copy()
    pixels[binary_dilation(local, iterations=2) & ~local] = (0,255,100)
    pair = Image.new('RGB', (tile.width*2,tile.height+35), '#e9edf3')
    pair.paste(Image.fromarray(pixels),(0,35))
    pair.paste(excluded_pixels_view(tile,local),(tile.width,35))
    draw = ImageDraw.Draw(pair)
    draw.text((10,10),'CONTEXT + CONTOUR',fill='black')
    draw.text((tile.width+10,10),'SELECTED PIXELS | HATCHING EXCLUDED',fill='black')
    path = req['path'].parent/(req['unit_id']+'_reference_repair_v2.png')
    path.parent.mkdir(parents=True,exist_ok=True)
    pair.save(path)
    return [req['views'][0],str(path)]


def repair_requests(planner, requests, inspector):
    """Mutate only successfully parsed flagged drafts, recording every attempt."""
    from .lean_retry import _requests
    from .per_mask_planning import combine
    repairs = []
    for req in requests:
        draft = req['raw']
        if not isinstance(draft.get('ref'), str) or not isinstance(draft.get('edit'), dict):
            continue
        metadata = json.loads((Path(req['views'][0]).parent/'views.json').read_text())
        matches = [t for t in metadata['targets'] if t['unit_id'] == req['unit_id']]
        if len(matches) != 1:
            raise ValueError('reference evidence mask binding is ambiguous')
        evidence = inspector.inspect_reference(metadata['source'], matches[0]['mask_path'], draft['ref'])
        req['reference_evidence'] = evidence
        if not evidence.get('advisory'):
            continue
        repair = {**req,
                  'path': req['path'].with_name(req['path'].stem+'_reference_repair_v2.json'),
                  'views': repair_views(req, metadata, matches[0]),
                  'prompt': correction_prompt(req['prompt'], draft, evidence),
                  'cache_context': {**req.get('cache_context', {}), 'reference_alignment': VERSION},
                  'original_request': req}
        repair['cache_context']['image_sha256'] = [file_hash(p) for p in repair['views']]
        repair['cache_context']['evidence_digest'] = digest({
            k:evidence.get(k) for k in ('version','ref','candidates','advisory')})
        repair.pop('raw', None)
        repairs.append(repair)
    for start in range(0, len(repairs), max(1, planner.batch_size)):
        _requests(planner, repairs[start:start+max(1, planner.batch_size)])
    for repair in repairs:
        req, answer = repair['original_request'], repair['raw']
        op = req['item']['_mask_assignments'][req['unit_id']]
        # This isolated syntax check has one object. Keep the public U binding
        # in req, but use the one-object normalizer's local U1 index here.
        check = combine([answer], 'multi', {'U1':op}, [{'object_id':'check'}])
        malformed = (bool(answer.get('_parse_error')) or
                     answer.get('_generation_metadata', {}).get('finish_reason') == 'length' or
                     any(t.get('_format_error') for t in check['targets']))
        req['pre_reference_raw'] = deepcopy(req['raw'])
        if not malformed:
            req['raw'] = answer
        repair['applied'] = not malformed
    return repairs
