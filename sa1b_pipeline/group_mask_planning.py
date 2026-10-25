"""One full original + one full-frame multi-mask contour image, one call/group.

Uses the same target-local content contract as per_mask_planning. Twins share
only the operation type, not the payload. Keys bind replies to marked masks.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from PIL import Image
from . import per_mask_planning as single
from .content_planning import obj
from .full_contour_views import contour_pair, INTRO, VERSION as VIEW_VERSION
from .geometry import load_mask
from .lean_retry import _requests

VERSION = 'group-mask-grounded-content-2'


def schema(assignments):
    return obj({uid: single.schema(op) for uid, op in assignments.items()})


def prompt(assignments, kind, hints=None):
    # Keep the exact grounding/content rules shared between both settings.
    common = single.COMMON.replace('Return only ref/edit or reject JSON.',
        'Return one JSON object keyed by the requested mask IDs; each value contains ref/edit or reject.')
    text = INTRO + common + '\nApply these steps independently to EACH labeled mask.'
    if kind == 'twins':
        text += (' Each member edits a SEPARATE copy of the source. Only the operation type is shared; '
                 'adapt the concrete edit to each object independently.')
    else:
        text += ' The accepted edits will coexist in one result; avoid conflicting edits.'
    for op in dict.fromkeys(assignments.values()):
        rule = single.operation_rule(op)
        text += '\n' + rule
    text += '\nAssigned operations: ' + json.dumps(assignments)
    if hints:
        text += '\nSource category hints by mask (may be wrong): ' + json.dumps(hints, ensure_ascii=False) + '. Trust actual mask extent over these hints.'
    return text


def views(candidate, objects, directory):
    source = Image.open(candidate['image_path']).convert('RGB')
    masks = [load_mask(o, source.size) for o in objects]
    digest = hashlib.sha256(json.dumps([VIEW_VERSION, hashlib.sha256(source.tobytes()).hexdigest(),
        [str(o['object_id']) for o in objects],
        [hashlib.sha256(m.tobytes()).hexdigest() for m in masks]]).encode()).hexdigest()[:20]
    root = Path(directory) / digest
    root.mkdir(parents=True, exist_ok=True)
    original, marked, labels = contour_pair(source, masks)
    pair = [root/'original.png', root/'all_masks.png']
    for p, im in zip(pair, [original, marked]):
        if not p.exists(): im.save(p)
    paths = []
    for o, m, label in zip(objects, masks, labels):
        mp = root/f'mask_{o["object_id"]}.png'
        if not mp.exists(): Image.fromarray(m.astype('uint8')*255).save(mp)
        paths.append(str(mp))
        label.update(object_id=str(o['object_id']), mask_path=str(mp))
    (root/'views.json').write_text(json.dumps(dict(version=VIEW_VERSION, source=candidate['image_path'], targets=labels), indent=2))
    return list(map(str, pair)), paths, digest


def combine(raw, kind, assignments, objects):
    original_raw = deepcopy(raw)
    meta = raw.get('_generation_metadata', {})
    if set(raw) - set(assignments) - {'_generation_metadata'}:
        raw = {}
    responses = []
    for uid in assignments:
        item = deepcopy(raw.get(uid, {'reject': False}))
        if not isinstance(item, dict): item = {'reject': False}
        item['_generation_metadata'] = meta
        responses.append(item)
    result = single.combine(responses, kind, assignments, objects)
    result.update(planning_contract=VERSION, _group_mask_response=original_raw,
                  _generation_metadata=meta, _planning_calls=1)
    return result


def plan_batch(planner, batch, *, kind, root):
    requests = []
    for item in batch:
        ops = [item['selection']['operation']]*len(item['objects']) if kind == 'twins' else item['selection']['operations']
        assignments = {f'U{i+1}': op for i, op in enumerate(ops)}
        item['_mask_assignments'] = assignments
        item['prompt'] = prompt(assignments, kind, item.get('planning_source_hints'))
        requests.append(dict(path=Path(root)/'planning_attempts'/item['gid']/'all_masks.json',
            views=item['views'], prompt=item['prompt'], schema=schema(assignments), item=item,
            cache_context={'contract': VERSION, 'model': getattr(planner, 'model_id', 'unknown')}))
    _requests(planner, requests)
    return [combine(r['raw'], kind, r['item']['_mask_assignments'], r['item']['objects']) for r in requests]
