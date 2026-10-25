"""Optional, cached SAM3 evidence; missing detections NEVER mean a bad edit.

Source segmentation is advisory: a text segmenter may select a whole host for
a correct part Ref. Only positive post-edit evidence of unchanged old content
can veto; missing detections never establish success or failure.
This does not change source masks, goals, or published references.
"""
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image

from .geometry import load_mask
from .production_state import atomic_json, digest, file_hash

VERSION = 'sam3-grounding-evidence-2-source-advisory'


def hard_failures(evidence):
    """Read old evidence safely; source extent alone is not a verified mismatch."""
    return [f for f in (evidence or {}).get('failure_reasons', [])
            if f != 'grounded_source_extent_exceeds_selector']


def source_advisories(source):
    if source and (source['score'] >= .8 and source['selection_recall'] >= .85
                   and source['selection_precision'] <= .3 and not source['border']):
        return ['segmented_source_exceeds_selector_verify_visible_extent']
    return []


def preflight_groups(groups, kind, inspector):
    """Reject verified source-scope conflicts before spending editor inference."""
    from copy import deepcopy
    retained, rejected = [], []
    for original in groups:
        group = deepcopy(original)
        members = group['members' if kind == 'twins' else 'units']
        bad = set()
        for member in members:
            member['source_grounding_evidence'] = inspector.inspect(group['source_image'], member)
            if hard_failures(member['source_grounding_evidence']):
                bad.add(member['unit_id'])
        # Explicit dependencies cannot survive removal of their prerequisite.
        if kind == 'multi':
            while True:
                expanded = bad | {b for a,b in group.get('edit_dependencies', []) if a in bad}
                if expanded == bad:
                    break
                bad = expanded
        if not bad:
            retained.append(group)
            continue
        if kind == 'twins' or len(members)-len(bad) < 2:
            rejected.append(dict(group=group, stage='source_grounding',
                reason='verified source-mask scope conflict', dropped_unit_ids=sorted(bad)))
            continue
        kept = [m for m in members if m['unit_id'] not in bad]
        ids = {m['unit_id'] for m in kept}
        group.update(units=kept, n_targets=len(kept), operations=[m['operation'] for m in kept],
                     generation_order=[uid for uid in group['generation_order'] if uid in ids],
                     edit_dependencies=[pair for pair in group.get('edit_dependencies',[]) if set(pair)<=ids],
                     adjacent_pairs=[])
        group['dependency_evidence'] = [e for e in group.get('dependency_evidence',[])
                                       if e.get('before') in ids and e.get('after') in ids]
        for member in members:
            if member['unit_id'] in bad:
                group.setdefault('protected_source_masks', []).append(member['mask'])
                group.setdefault('planning_retention', {}).setdefault('dropped', []).append(
                    dict(unit_id=member['unit_id'], mask=member['mask'],
                         reason='verified source-mask scope conflict or dependent edit',
                         stage='source_grounding', evidence=member.get('source_grounding_evidence')))
        from .multi_execution_contract import schedule_content_dependencies
        from .group_schema import validate_multi_record
        schedule_content_dependencies(group)
        # Writable supports are resolved by generation. Match the existing
        # planning validator without turning a selector into a write support.
        trial = {**group, 'units':[dict(m, support_mask=m.get('support_mask') or m['mask'])
                                   for m in group['units']]}
        validate_multi_record(trial, require_resolved_regions=False)
        retained.append(group)
    return retained, rejected


def query_phrase(ref):
    """Drop location postmodifiers for local segmentation, preserving 'of' parts."""
    return re.split(
        r'\b(?:on|in|at|behind|holding|held|carried|worn|with|near|that|lying|walking|stretching|beside)\b',
        ref, maxsplit=1, flags=re.I)[0].strip() or ref


def contradictions(source, target, operation):
    """Require positive, spatially corroborated evidence; abstain otherwise."""
    flags = []
    if operation in ('remove', 'replace') and source and target:
        if (source['score'] >= .6 and source['iou'] >= .5
                and target['score'] >= .6 and target['iou'] >= .5
                and target.get('old_region_retained_fraction', 0) >= .65
                and target.get('unchanged_old_pixels_fraction', 0) >= .85):
            flags.append('grounded_old_content_remains_unchanged')
    return flags


class GroundingEvidence:
    def __init__(self, model_path, directory, device='cuda'):
        from .site_segmentation import Sam3SiteResolver
        self.resolver = Sam3SiteResolver(model_path, device=device)
        self.directory = Path(directory)

    def inspect(self, source_path, member, result_path=None):
        """One source segmentation, plus one result segmentation for remove/replace."""
        import json
        import torch
        spec = member['instruction_spec']
        operation = spec['operation']
        if operation == 'add':
            return dict(version=VERSION, status='not_applicable_add_site',
                        failure_reasons=[], segmentation_calls=0)
        key = digest(dict(version=VERSION, model=self.resolver.model_path,
                          source=file_hash(source_path), mask=file_hash(member['mask']),
                          ref=spec['ref'], operation=operation,
                          result=file_hash(result_path) if result_path else None))
        root = self.directory / key
        path = root/'evidence.json'
        if path.exists():
            return json.loads(path.read_text())
        root.mkdir(parents=True, exist_ok=True)
        source = Image.open(source_path).convert('RGB')
        selected = load_mask({'mask_path': member['mask']}, source.size)
        ys, xs = np.where(selected)
        if not len(xs):
            raise ValueError('empty grounding selector')
        pad = max(64, round(max(np.ptp(xs)+1, np.ptp(ys)+1)*.8))
        box = (max(0, int(xs.min())-pad), max(0, int(ys.min())-pad),
               min(source.width, int(xs.max())+pad+1), min(source.height, int(ys.max())+pad+1))
        local = selected[box[1]:box[3], box[0]:box[2]]
        query = query_phrase(spec['ref'])
        started = time.monotonic()

        def segment(image, label):
            crop = image.crop(box)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                inputs = self.resolver.processor(images=crop, text=query, return_tensors='pt').to(self.resolver.device)
                output = self.resolver.model(**inputs)
                raw = self.resolver.processor.post_process_instance_segmentation(
                    output, threshold=.3, target_sizes=[crop.size[::-1]])[0]
            candidates = []
            for index, (mask, score) in enumerate(zip(raw['masks'], raw['scores'])):
                a = mask.cpu().numpy().astype(bool)
                overlap = int((a & local).sum())
                candidates.append(dict(index=index, score=float(score),
                    iou=overlap/max(1, int((a | local).sum())),
                    selection_recall=overlap/max(1, int(local.sum())),
                    selection_precision=overlap/max(1, int(a.sum())),
                    border=bool(a[0].any() or a[-1].any() or a[:, 0].any() or a[:, -1].any()),
                    pixels=int(a.sum()), _mask=a))
            best = max(candidates, key=lambda c:c['iou'], default=None)
            if best:
                Image.fromarray(best['_mask'].astype('uint8')*255).save(root/(label+'_segment.png'))
            return best, candidates

        # The pre-generation check and post-generation audit share the same
        # source pixels/query. Reuse that independent inference, not its verdict.
        source_key = digest(dict(version=VERSION, model=self.resolver.model_path,
            source=file_hash(source_path), mask=file_hash(member['mask']), query=query, box=list(box)))
        source_root = self.directory/'source_cache'/source_key
        source_meta = source_root/'source.json'
        if source_meta.exists():
            cached = json.loads(source_meta.read_text())
            before, source_candidates = cached['best'], cached['candidates']
            if before:
                before['_mask'] = np.asarray(Image.open(source_root/'mask.png')) > 0
            source_calls = 0
        else:
            before, source_candidates = segment(source, 'source')
            source_root.mkdir(parents=True, exist_ok=True)
            if before:
                Image.fromarray(before['_mask'].astype('uint8')*255).save(source_root/'mask.png')
            strip = lambda c: {k:v for k,v in c.items() if k!='_mask'} if c else None
            atomic_json(source_meta, dict(best=strip(before), candidates=[strip(c) for c in source_candidates]))
            source_calls = 1
        after, target_candidates, calls = None, [], source_calls
        if result_path and operation in ('remove', 'replace') and before and before['iou'] >= .1:
            result = Image.open(result_path).convert('RGB')
            if result.size != source.size:
                raise ValueError('grounding result dimensions changed')
            after, target_candidates = segment(result, 'result')
            calls += 1
            if after:
                old = before['_mask'] & local
                shared = old & after['_mask']
                after['old_region_retained_fraction'] = float(shared.sum()/max(1, old.sum()))
                change = np.max(abs(np.asarray(source.crop(box), dtype=np.int16) -
                                    np.asarray(result.crop(box), dtype=np.int16)), axis=2)
                # Photometric evidence is conservative: natural shade changes
                # can remove the veto, never prove that deletion succeeded.
                after['unchanged_old_pixels_fraction'] = float((change[shared] <= 12).mean()) if shared.any() else 0.
        flags = contradictions(before, after, operation)
        def clean(c):
            return {k:v for k,v in c.items() if k != '_mask'} if c else None
        evidence = dict(version=VERSION, status='conflict' if flags else 'no_verified_conflict',
            query=query, ref=spec['ref'], box=list(box), source=clean(before), result=clean(after),
            source_candidates=[clean(c) for c in source_candidates],
            result_candidates=[clean(c) for c in target_candidates],
            failure_reasons=flags, advisory_reasons=source_advisories(before), segmentation_calls=calls,
            seconds=time.monotonic()-started, directory=str(root),
            source_sha256=file_hash(source_path),
            result_sha256=file_hash(result_path) if result_path else None,
            mask_sha256=file_hash(member['mask']))
        atomic_json(path, evidence)
        return evidence

    def close(self):
        del self.resolver
