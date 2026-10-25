"""Independent post-planning evidence gates (opt-in; never changes masks or wording).

Three short, schema-constrained checks run WITHOUT reasoning on the resident
planner service after the normal program checks:

* reader:   only the clean full photo + the Ref; every box the reader returns
            must land on the target mask (IoU >= threshold). Catches wrong
            instance, non-unique Ref and whole/part scope mismatch.
* identity: text-only; the Ref's head entity must be the same kind of thing as
            the independent SA-1B companion category (synonyms allowed).
* payload:  text-only, replace only; a pure color/material/pattern change of the
            same object is not a replacement.
* medium:   opt-in, additions only; the clean photo + outlined crop judged real /
            depicted (inside a printed, painted or displayed picture) / sculpted. Only
            "real" passes: inside a picture or on a carving the generator draws a
            photographic object that clashes with the medium. On 304 labelled add/replace
            targets, the 11 additions judged depicted/sculpted were good 2, minor 5, bad 3,
            text-fix 1 (other additions: 65% good, 8% bad); replacements inside pictures
            were mostly fine (good 4 of 7, no bad), so they are not gated.
* lighting: opt-in, colour edits only; the clean photo + outlined crop judged
            lit / shade / silhouette. Only "lit" passes: the generator renders a
            requested colour as if lit, so a part in shade or backlit comes out
            separately brightened. Offline on 77 reviewed colour edits: 4 of the 5
            "brightened in shade" defects were judged shade/silhouette; the cost is
            23 of 77 edits dropped, 13 of them judged good.

When more add twin candidates pass than are kept, the pair with the most similar
anchor areas is kept (the blind protocol compares resolved site areas).

Opt-in ``add_site`` (not recommended): an add Ref may name the attachment site on
the anchor (protocol 2.0 resolves the final add mask as difference x host). A single
reader box contained in the anchor box whose centre lies on the anchor's own pixels
then passes, and identity relation "part" passes; two such boxes remain ambiguous.
In the 2026-10-23 run it changed one of eight add twin groups; both delivered
targets it admitted were flawed (one bad).

A failed gate marks that target infeasible with an explicit reason; multi
plans then keep their feasible subset, twins keep two passing members when the
selection over-generated a third candidate. Gates only remove plans; they are
never treated as semantic approval of the image result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

VERSION = 'evidence-gates-1-reader-identity-payload'

READER = ('Find every distinct object or region in this photo that the following phrase could refer to:\n'
          '"{ref}"\n'
          'A reader sees only this photo and the phrase. If the phrase fits more than one visible instance, '
          'return a box for EACH fitting instance. If it fits exactly one, return one box. If nothing fits, return [].\n'
          'Boxes use coordinates normalized to 0-1000: [x1, y1, x2, y2], x from left, y from top.\n'
          'Return JSON only: {{"boxes": [[x1, y1, x2, y2], ...]}}')
READER_SCHEMA = {'type': 'object', 'properties': {'boxes': {'type': 'array', 'maxItems': 12, 'items': {
    'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': 1000}, 'minItems': 4, 'maxItems': 4}}},
    'required': ['boxes'], 'additionalProperties': False}

IDENTITY = '''Two independent sources name the same selected image region.
Source A (dense annotation; aliases separated by commas, may be imprecise): "{category}"
Source B (reference phrase written by another model): "{ref}"
Compare ONLY the head entity that Source B points to (ignore its locating context such as "on the table", "worn by the man").
Return one relation:
- "same": the same kind of thing (synonym, broader or narrower term, or a compatible name for the same entity)
- "part": B names only a part/component/surface of what A names
- "whole": B names a larger whole/host that contains what A names
- "different": clearly different kinds of things
Return JSON only: {{"relation": "same|part|whole|different"}}'''
IDENTITY_SCHEMA = {'type': 'object', 'properties': {'relation': {'type': 'string', 'enum': ['same', 'part', 'whole', 'different']}},
                   'required': ['relation'], 'additionalProperties': False}

PAYLOAD = '''A photo-editing request replaces an existing thing with new content.
Existing thing: "{ref}" (annotated as: "{category}")
New content: "{replacement}"
Classify the change:
- "different_object": the new content is a different kind of object or entity
- "new_design": the same kind of object, but with a clearly different shape, structure or style that a viewer would see as a different design (not just its color, material, finish, texture or pattern)
- "appearance_only": essentially the same object whose color, material, finish, texture, pattern or printed decoration changes
Return JSON only: {{"change": "different_object|new_design|appearance_only"}}'''
PAYLOAD_SCHEMA = {'type': 'object', 'properties': {'change': {'type': 'string', 'enum': ['different_object', 'new_design', 'appearance_only']}},
                  'required': ['change'], 'additionalProperties': False}

LIGHTING = '''Image 1 is a photo. Image 2 shows one region of it outlined (outside darkened).
Judge only how the outlined region is LIT compared with the scene's main light, not its own colour:
- "lit": it receives the scene's normal or direct light (a black object in sunlight is "lit")
- "shade": it lies in shadow, shade or clearly dimmer light than the scene's lit parts
- "silhouette": it is a nearly black backlit shape whose own colour cannot be seen
Return JSON only: {"lighting": "lit|shade|silhouette"}'''
LIGHTING_SCHEMA = {'type': 'object', 'properties': {'lighting': {'type': 'string', 'enum': ['lit', 'shade', 'silhouette']}},
                   'required': ['lighting'], 'additionalProperties': False}

MEDIUM = '''Image 1 is a photo. Image 2 shows one region of it outlined (outside darkened).
What is the outlined region?
- "real": a real object, person or surface in the scene (a whole poster, sign, screen or book is real)
- "depicted": something shown INSIDE a flat picture: printed, painted or displayed, such as a figure or object on a
  stamp, poster, cover, label, painting, mural or screen
- "sculpted": part of a statue, carving or relief, such as a carved figure, its head, hair or clothing
Return JSON only: {"medium": "real|depicted|sculpted"}'''
MEDIUM_SCHEMA = {'type': 'object', 'properties': {'medium': {'type': 'string', 'enum': ['real', 'depicted', 'sculpted']}},
                 'required': ['medium'], 'additionalProperties': False}

VIEWPOINT = '''Image 1 is a photo. Image 2 shows one region of it outlined (outside darkened).
Where is the camera?
- "aerial": high above the ground looking steeply down on it (drone, plane, tall building or cliff), so the
  outlined region is seen nearly from above
- "elevated": above the scene but looking across it at a shallow angle (a hill, balcony or upper floor)
- "ground": at about human height
Return JSON only: {"view": "aerial|elevated|ground"}'''
VIEWPOINT_SCHEMA = {'type': 'object', 'properties': {'view': {'type': 'string', 'enum': ['aerial', 'elevated', 'ground']}},
                    'required': ['view'], 'additionalProperties': False}

TOKEN_LIMITS = {'reader': 200, 'identity': 40, 'payload': 40, 'lighting': 30, 'medium': 30, 'viewpoint': 30}


def color_edit(target: dict) -> bool:
    attrs = target.get('attributes') or []
    return target.get('operation') == 'attribute' and any(
        isinstance(a, dict) and a.get('property') in ('color', 'tint') for a in attrs)


def mask_box_1000(mask_path: str) -> list[float]:
    """Target box on the 0-1000 grid the reader uses (aspect-preserving thumbnails share it)."""
    mask = np.asarray(Image.open(mask_path).convert('L')) > 127
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('empty gate mask')
    h, w = mask.shape
    return [xs.min() / w * 1000, ys.min() / h * 1000, (xs.max() + 1) / w * 1000, (ys.max() + 1) / h * 1000]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def containment(box, target) -> float:
    """Fraction of ``box`` inside ``target``."""
    ix = max(0.0, min(box[2], target[2]) - max(box[0], target[0]))
    iy = max(0.0, min(box[3], target[3]) - max(box[1], target[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy / area if area > 0 else 0.0


def reader_verdict(boxes, target, threshold: float = .5, *, site_containment: float | None = None,
                   on_target=None):
    """'pass' only if the reader returns >=1 box and EVERY box lands on the target.

    With ``site_containment`` (add anchors) a box may also land as a sub-region of
    the target (>= that fraction inside its box and, when ``on_target`` is given,
    its centre on the target's pixels); at most one such site box is allowed.
    """
    if not isinstance(boxes, list):
        return 'invalid', 0.0
    boxes = [b for b in boxes if isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]]
    if not boxes:
        return 'none', 0.0
    scores = [iou(b, target) for b in boxes]
    if site_containment is not None:
        sites = [s < threshold and containment(b, target) >= site_containment
                 and (on_target is None or on_target(b)) for b, s in zip(boxes, scores)]
        hits = [s >= threshold or site for s, site in zip(scores, sites)]
        if not any(hits):
            return 'wrong', max(scores)
        if not all(hits) or sum(sites) > 1:
            return 'ambiguous', max(scores)
        return ('pass' if not any(sites) else 'pass_site'), max(scores)
    hits = [s >= threshold for s in scores]
    if not any(hits):
        return 'wrong', max(scores)
    if not all(hits):
        return 'ambiguous', max(scores)
    return 'pass', max(scores)


def centre_on_mask(mask_path: str, tolerance: float = .02):
    """Predicate: does a 0-1000 box's centre lie on the mask (dilated by tolerance x short side)?"""
    import cv2
    mask = np.asarray(Image.open(mask_path).convert('L')) > 127
    h, w = mask.shape
    radius = max(1, int(round(tolerance * min(h, w))))
    grown = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0

    def check(box) -> bool:
        x = min(w - 1, max(0, int((box[0] + box[2]) / 2 / 1000 * w)))
        y = min(h - 1, max(0, int((box[1] + box[3]) / 2 / 1000 * h)))
        return bool(grown[y, x])
    return check


def _members(plan: dict, kind: str) -> list[dict]:
    return plan.get('members' if kind == 'twins' else 'targets') or []


def _mask_paths(item: dict) -> dict[str, str]:
    """unit_id -> exact mask written beside the planning views (same pixels the planner saw)."""
    views = item.get('per_mask_views') or []
    if not views:
        raise ValueError('evidence gates require per-mask planning views')
    meta = json.loads((Path(views[0][0]).parent / 'views.json').read_text())
    return {t['unit_id']: t['mask_path'] for t in meta['targets']}


class EvidenceGates:
    def __init__(self, *, reader: bool = True, identity: bool = True, payload: bool = True,
                 iou_threshold: float = .5, twin_keep: int = 2, add_site: bool = False,
                 site_containment: float = .8, lighting: bool = False, medium: bool = False,
                 viewpoint: bool = False):
        if not (reader or identity or payload):
            raise ValueError('at least one evidence gate is required')
        self.reader, self.identity, self.payload = reader, identity, payload
        self.iou_threshold, self.twin_keep = float(iou_threshold), int(twin_keep)
        self.add_site, self.site_containment = bool(add_site), float(site_containment)
        self.lighting = bool(lighting)
        self.medium = bool(medium)
        # Additions seen from high above (aerial): the editor draws a side-view item at crop scale
        # (giant scarecrows on a field); on 216 labelled additions 4 were "aerial": 3 bad, 1 minor.
        self.viewpoint = bool(viewpoint)
        self.calls = 0

    def config(self) -> dict:
        config = {'version': VERSION, 'reader': self.reader, 'identity': self.identity, 'payload': self.payload,
                  'iou_threshold': self.iou_threshold, 'twin_keep': self.twin_keep}
        if self.add_site:
            config.update(add_site=True, site_containment=self.site_containment)
        if self.lighting:
            config['lighting'] = True
        if self.medium:
            config['medium'] = True
        if self.viewpoint:
            config['viewpoint'] = True
        return config

    def requests(self, item: dict, plan: dict, kind: str) -> list[dict]:
        masks = _mask_paths(item)
        out = []
        for index, target in enumerate(_members(plan, kind)):
            if not target.get('feasible'):
                continue
            uid = target.get('unit_id') or f'U{index + 1}'
            ref = str(target.get('ref_phrase') or '').strip()
            if not ref:
                continue
            obj = next((o for o in item['objects'] if str(o['object_id']) == str(target.get('object_id'))), None)
            category = str((obj or {}).get('category') or '').strip()
            base = {'unit_id': uid, 'operation': target.get('operation')}
            if self.reader:
                out.append({**base, 'gate': 'reader', 'images': [item['per_mask_views'][index][0]],
                            'prompt': READER.format(ref=ref), 'schema': READER_SCHEMA,
                            'target_box': mask_box_1000(masks[uid]), 'mask_path': masks[uid]})
            if self.identity and category:
                out.append({**base, 'gate': 'identity', 'images': [],
                            'prompt': IDENTITY.format(category=category, ref=ref), 'schema': IDENTITY_SCHEMA})
            if self.lighting and color_edit(target):
                out.append({**base, 'gate': 'lighting', 'images': list(item['per_mask_views'][index][:2]),
                            'prompt': LIGHTING, 'schema': LIGHTING_SCHEMA})
            if self.medium and target.get('operation') == 'add':
                out.append({**base, 'gate': 'medium', 'images': list(item['per_mask_views'][index][:2]),
                            'prompt': MEDIUM, 'schema': MEDIUM_SCHEMA})
            if self.viewpoint and target.get('operation') == 'add':
                out.append({**base, 'gate': 'viewpoint', 'images': list(item['per_mask_views'][index][:2]),
                            'prompt': VIEWPOINT, 'schema': VIEWPOINT_SCHEMA})
            replacement = str(target.get('payload') or '').strip()
            if self.payload and target.get('operation') == 'replace' and replacement:
                out.append({**base, 'gate': 'payload', 'images': [],
                            'prompt': PAYLOAD.format(category=category, ref=ref, replacement=replacement),
                            'schema': PAYLOAD_SCHEMA})
        return out

    def judge(self, request: dict, response: dict) -> dict:
        gate = request['gate']
        site = self.add_site and request.get('operation') == 'add'
        if gate == 'reader':
            status, best = reader_verdict(response.get('boxes'), request['target_box'], self.iou_threshold,
                                          site_containment=self.site_containment if site else None,
                                          on_target=centre_on_mask(request['mask_path'])
                                          if site and request.get('mask_path') else None)
            return {'gate': gate, 'status': status, 'best_iou': round(best, 3), 'boxes': response.get('boxes'),
                    'pass': status in ('pass', 'pass_site')}
        if gate == 'identity':
            relation = response.get('relation')
            return {'gate': gate, 'relation': relation, 'pass': relation == 'same' or (site and relation == 'part')}
        if gate == 'lighting':
            lighting = response.get('lighting')
            return {'gate': gate, 'lighting': lighting, 'pass': lighting == 'lit'}
        if gate == 'medium':
            medium = response.get('medium')
            return {'gate': gate, 'medium': medium, 'pass': medium == 'real'}
        if gate == 'viewpoint':
            view = response.get('view')
            return {'gate': gate, 'view': view, 'pass': view in ('elevated', 'ground')}
        change = response.get('change')
        return {'gate': gate, 'change': change, 'pass': change in ('different_object', 'new_design')}

    def apply(self, planner, batch: list[dict], plans: list[dict], kind: str) -> None:
        """Mutate plans in place: annotate every target, fail the ones without evidence."""
        requests = []
        for item, plan in zip(batch, plans):
            for request in self.requests(item, plan, kind):
                requests.append((item, plan, request))
        responses = planner.generate_multimage(
            [(r['images'], r['prompt']) for _, _, r in requests],
            schemas=[r['schema'] for _, _, r in requests],
            token_limits=[TOKEN_LIMITS[r['gate']] for _, _, r in requests],
            thinking=[False] * len(requests)) if requests else []
        self.calls += len(requests)
        evidence: dict[int, dict[str, list]] = {}
        for (item, plan, request), response in zip(requests, responses):
            verdict = self.judge(request, response if isinstance(response, dict) else {})
            evidence.setdefault(id(plan), {}).setdefault(request['unit_id'], []).append(verdict)
        for item, plan in zip(batch, plans):
            per_unit = evidence.get(id(plan), {})
            for target in _members(plan, kind):
                verdicts = per_unit.get(target.get('unit_id'))
                if not target.get('feasible') or verdicts is None:
                    continue
                failed = [v for v in verdicts if not v['pass']]
                target['evidence_gates'] = {'version': VERSION, 'pass': not failed, 'verdicts': verdicts}
                if failed:
                    reason = 'evidence_gate: ' + ', '.join(
                        f"{v['gate']}={v.get('status') or v.get('relation') or v.get('change') or v.get('lighting') or v.get('medium') or v.get('view')}"
                        for v in failed)
                    target['feasible'] = False
                    target['reason'] = reason
                    if isinstance(target.get('grounding'), dict):
                        target['grounding']['edit_feasible'] = False
                        target['grounding']['reason'] = reason
            plan['evidence_gate_config'] = self.config()
            if kind == 'twins':
                self._keep_twins(item, plan)

    def _keep_twins(self, item: dict, plan: dict) -> None:
        members = plan.get('members') or []
        passing = [m for m in members if m.get('feasible')]
        if len(members) <= self.twin_keep:
            plan['feasible'] = len(passing) == len(members) and bool(members)
            if not plan['feasible']:
                plan['reason'] = '; '.join(str(m.get('reason')) for m in members if not m.get('feasible'))
            return
        # Over-generated candidates: keep the first passing members in selection
        # priority order; selection already enforced the pairwise area ratio.
        kept = passing[:self.twin_keep]
        if len(passing) > self.twin_keep == 2 and all(m.get('operation') == 'add' for m in passing):
            # Add twins are compared on resolved attachment sites: prefer the most similar anchors.
            masks = _mask_paths(item)
            area = {id(m): int((np.asarray(Image.open(masks[m['unit_id']]).convert('L')) > 127).sum()) for m in passing}
            pairs = [(a, b) for i, a in enumerate(passing) for b in passing[i + 1:]]
            kept = list(min(pairs, key=lambda ab: max(area[id(ab[0])], area[id(ab[1])]) /
                            max(1, min(area[id(ab[0])], area[id(ab[1])]))))
        plan['twin_overgeneration'] = {'candidates': [m.get('object_id') for m in members],
                                       'kept': [m.get('object_id') for m in kept],
                                       'dropped': [{'object_id': m.get('object_id'), 'reason': m.get('reason') or 'not needed',
                                                    'ref_phrase': m.get('ref_phrase'),
                                                    'evidence_gates': m.get('evidence_gates')}
                                                   for m in members if m not in kept]}
        if len(kept) < self.twin_keep:
            plan['feasible'] = False
            plan['reason'] = 'fewer than two over-generated twin members passed: ' + '; '.join(
                str(m.get('reason')) for m in members if not m.get('feasible'))
            return
        plan['members'] = kept
        plan['feasible'] = True
        plan['reason'] = ''
        plan['_twin_kept_object_ids'] = [str(m.get('object_id')) for m in kept]
