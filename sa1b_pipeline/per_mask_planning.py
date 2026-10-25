"""One original + one outlined context crop per stateless planning request.

Requests are flattened across groups for GPU batching, never combined in a
conversation. The model emits only ref/edit or reject; IDs bind in code.
"""
from copy import deepcopy
import re
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from . import content_planning as content
from .geometry import load_mask
from .twin_contract import ADAPTED
from .lean_retry import _requests

VERSION = 'per-mask-grounded-content-13.8-bounded-alignment-dependencies'
VIEW_VERSION = 'per-mask-single-focus-5'
INTRO = ('Image 1 is the clean full source. Image 2 is ONE crop at unchanged orientation: '
         'ONLY the bright, original-color pixels inside the green contour are selected. '
         'Darkened pixels outside it are context, NOT the target. Locate the bright selected part in Image 1.\n')
COMMON = '''Design ONE ordinary photo-editing request suited to this object's role and scene. First imagine the assigned operation applied to the WHOLE selected extent: the remaining person/object must stay anatomically intact, usable and supported. If that is not possible, reject this operation rather than disguise a different operation as it. Prefer a useful or aesthetically coherent change, not decoration merely to fill the request. The change must be visibly meaningful at the FULL photo's resolution, not only in the magnified crop.
Identify the selected pixels first, then locate that SAME region in Image 1 using the unmarked full photo and source coordinates. Do not let a recognizable object beside or inside a mask hole become the target. Ref and edit must match the full selected extent: name a selected surface/part as that surface/part, not its whole host. Before naming a whole object, check whether its visible attached structure extends into the excluded pixels. If it does, name the selected part precisely; reject if you cannot identify that part. Do not infer whole-object selection from a familiar silhouette. If multiple inseparable entities are selected, edit all naturally or reject. Crop magnification does not imply real size.
Ref: one complete English noun phrase starting with "the". A reader seeing ONLY the unmarked full photo must locate this exact target. Check all similar instances, including unmarked ones, and exclude the closest rival with the shortest verified distinction. Use the full photo's left/right; ordinals need an explicit visible row/group and a verified count. Omit uncertain colors, functions, landmarks and unnecessary modifiers. No coordinates, crop positions or annotation words in Ref.
Use the assigned operation. Reject only if identification/unique reference remains unresolved or no natural edit of that type is possible; smallness, clipping or occlusion alone is not a reason.
Return only ref/edit or reject JSON. Give content slots, not sentences or explanations; code renders the instruction.'''
ADD = ('add: addition names ONLY new content with an article/count. relation describes direct attachment/support '
       'at a visible site INSIDE the selected anchor, not on a neighboring body part or object. '
       'Use site="" when the selected part/surface itself supports the addition; otherwise name only a visible '
       'subpart of this anchor, without repeating the anchor or a possessive. '
       'The new item must fit and serve a plausible purpose without duplicating an existing part. Preserve existing contents. Distinguish physical additions '
       'from printed/embroidered designs explicitly; never invent an unseen attachment or add an existing item again.')


TEXT_RULE = ('Writing: if the new item would naturally carry writing (a sign, nameplate, label, price tag, banner, '
             'placard, poster, menu or plaque inscription), set text to its exact wording: 1-3 short English words or '
             'digits that suit the scene, such as a shop name, a price or a room number; never gibberish. Otherwise '
             'text is "" and the item stays plain, without letters, numbers or logos. Never put wording inside '
             'addition/replacement.')
ATTRIBUTE_TEXT_RULE = (' For property text, value is the exact new wording: 1-4 English words or digits, different from '
                       'the current wording and short enough to stay legible on the selected surface.')
TWIN_ADD_RULE = ('Add exactly ONE compact item (not a row, pair, group or several items), sized like an ordinary '
                 'accessory of this anchor.')
TWIN_PLURAL = re.compile(r'\b(rows?|pairs?|sets?|groups?|clusters?|bunch(?:es)?|few|several|some|many|multiple|two|three|'
                         r'four|five|line|string|series|collection|pile)\b', re.I)


REPLACE_RETRY_NOTE = ('\nA previous answer for this region proposed the replacement "{old}". An independent check found that it '
                      'keeps the same kind of object and mainly changes its color, material, finish or pattern; that is an '
                      'attribute edit, not a replacement. Propose a clearly different kind of object (or a clearly different '
                      'construction) that would naturally stand in exactly this place and fit the scene, keeping the same '
                      'Ref; otherwise reject.')


def retry_note(old_payload):
    """Feedback appended to the original request for one bounded replace retry."""
    return REPLACE_RETRY_NOTE.format(old=str(old_payload).replace('"', "'")[:120])


READER_FAILURES = ('evidence_gate: reader=wrong', 'evidence_gate: reader=none')
READER_RETRY_NOTE = ('\nA previous answer for this region used the Ref "{old}". An independent reader who saw ONLY the '
                     'unmarked full photo and that Ref {found}, not the selected region. Write a Ref that a reader would '
                     'match to exactly the selected region and nothing else: name what the selected pixels actually show and '
                     'add the visible cue (position in the full photo, colour, nearby landmark) that separates it from that '
                     'other match. Keep an edit that still fits; if no phrase can single out the selected region, reject.')


def reader_note(target):
    """Feedback for one bounded re-plan after the reader check placed the Ref elsewhere (or nowhere)."""
    verdict = next((v for v in (target.get('evidence_gates') or {}).get('verdicts') or [] if v.get('gate') == 'reader'), {})
    boxes = [b for b in verdict.get('boxes') or [] if isinstance(b, list) and len(b) == 4]
    if boxes:
        where = '; '.join('[' + ', '.join(str(round(v / 10)) for v in b) + ']' for b in boxes[:3])
        found = ('picked a different region (box in percent of the photo, left, top, right, bottom: ' + where + ')')
    else:
        found = 'could not find any matching region'
    return READER_RETRY_NOTE.format(old=str(target.get('ref_phrase') or '').replace('"', "'")[:160], found=found)


def options_of(planner):
    """Opt-in planning options; an empty dict keeps the default contract."""
    options = dict(getattr(planner, 'planning_options', None) or {})
    unknown = set(options) - {'explicit_text', 'twin_single_add', 'replace_retry', 'garment_extent', 'reader_retry'}
    if unknown:
        raise ValueError(f'unknown planning options: {sorted(unknown)}')
    return options


def schema(operation, *, explicit_text=False):
    edit = (content.obj({'replacement': {'type':'string','minLength':1,'maxLength':120}})
            if operation == 'replace' else content.edit_schema(operation))
    if operation == 'add':
        edit = deepcopy(edit)
        edit['properties']['addition'] = edit['properties'].pop('object')
        edit['required'] = ['addition' if k == 'object' else k for k in edit['required']]
    if explicit_text and operation in ('add', 'replace'):
        edit = deepcopy(edit)
        edit['properties']['text'] = {'type': 'string', 'maxLength': 40}
        edit['required'] = list(edit['required']) + ['text']
    return {'anyOf': [content.obj({'ref': {'type':'string','minLength':1,'maxLength':240},
                                  'edit': edit}),
                      content.obj({'reject': {'type':'string','minLength':1,'maxLength':120}})]}


def operation_rule(operation):
    return (ADD if operation == 'add' else
            'replace: replacement names ONLY the NEW physical object/design, with an article/count. It must physically substitute for the entire selected entity while preserving its host. Covering the still-present entity with an accessory is add, not replace; retouching its surface is attribute. Reject if no real substitution is natural. Never copy source locating words into replacement. Avoid a with-clause inside Ref that could be mistaken for replacement content.'
            if operation == 'replace' else
            'remove: edit is empty {}. Delete the selected content with plausible background completion. Its absence must leave the remaining host usable, supported and anatomically intact; reject removal of essential mechanical, anatomical or load-bearing parts.'
            if operation == 'remove' else content.RULES[operation] +
            ' For text, value is the desired written wording, never the color of the lettering. Preserve unrelated lettering and motifs when changing a surface color.')


def prompt(operation, hint=None, location=None, *, explicit_text=False, twin_add=False):
    text = INTRO + COMMON + '\n' + operation_rule(operation)
    if explicit_text and operation in ('add', 'replace'):
        text += '\n' + TEXT_RULE
    if explicit_text and operation == 'attribute':
        text += ATTRIBUTE_TEXT_RULE
    if twin_add and operation == 'add':
        text += '\n' + TWIN_ADD_RULE
    text += '\nAssigned operation: ' + operation
    if location:
        text += '\nImage 1 coordinates, percent [left, top, right, bottom], origin top-left: ' + json.dumps(location) + '. Coordinates locate the crop/target only; the contour defines the selected pixels. Never copy coordinates into Ref.'
    if hint:
        text += '\nSource annotation hint (may be wrong or describe a larger host): ' + json.dumps(hint, ensure_ascii=False) + '. Verify it against the selected pixels and their position in Image 1. Do not invent a different object to reconcile a conflict; reject unresolved identity. Trust actual mask extent over this hint.'
    return text


def request_prompt(operation, image_paths, hint=None, *, explicit_text=False, twin_add=False):
    """Bind source geometry to this crop without requesting model output fields."""
    metadata_path = Path(image_paths[0]).parent/'views.json'
    location = None
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        uid = Path(image_paths[1]).stem.removesuffix('_crop')
        matches = [t for t in metadata['targets'] if t['unit_id'] == uid]
        if len(matches) != 1:
            raise ValueError('planning crop geometry is not uniquely bound')
        with Image.open(image_paths[0]) as original:
            width, height = metadata.get('source_size', original.size)
        def percent(box):
            return [round(100*v/limit, 1) for v,limit in zip(box,[width,height,width,height])]
        location = {'mask_box': percent(matches[0]['mask_bbox_xyxy']),
                    'crop_box': percent(matches[0]['crop_box_xyxy'])}
    return prompt(operation, hint, location, explicit_text=explicit_text, twin_add=twin_add)


def clean_remote_speckles(mask):
    """Remove only 1–2 px remote artifacts; preserve meaningful multipart masks.

    The original is retained alongside the exact derived selector. Never prune
    an instance, a thin connected part, a small mask or more than 0.5% of pixels.
    """
    from scipy.ndimage import label
    labels,n=label(mask,structure=np.ones((3,3),dtype=bool))
    if n<2:return mask,0
    counts=np.bincount(labels.ravel());counts[0]=0
    main=int(np.argmax(counts));area=int(mask.sum())
    if counts[main]<128 or counts[main]/area<.995:return mask,0
    ys,xs=np.where(labels==main);distance=max(32,round(max(mask.shape)*.04))
    remove=[]
    for component in np.flatnonzero((counts>0)&(counts<=2)):
        cy,cx=np.where(labels==component)
        if (cx.max()<xs.min()-distance or cx.min()>xs.max()+distance
                or cy.max()<ys.min()-distance or cy.min()>ys.max()+distance):
            remove.append(int(component))
    cleaned=mask&~np.isin(labels,remove)
    return cleaned,int(mask.sum()-cleaned.sum())


def excluded_pixels_view(tile, selected):
    """Make negative space explicit without coloring any selected pixels."""
    from scipy.ndimage import distance_transform_edt
    from .fonts import load_font
    mask = np.asarray(Image.fromarray(selected.astype('uint8')*255).resize(
        tile.size, Image.Resampling.NEAREST)) > 0
    original = np.asarray(tile)
    pixels = original.copy()
    yy, xx = np.indices(mask.shape)
    pixels[~mask] = (72, 72, 72)
    pixels[(~mask) & (((xx+yy)//12) % 2 == 0)] = (110, 110, 110)
    layer = Image.fromarray(pixels)
    distance = distance_transform_edt(np.pad(~mask, 1))[1:-1, 1:-1]
    if distance.max() > 35:
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        draw = ImageDraw.Draw(layer); font = load_font(22, bold=True)
        bounds = draw.textbbox((0, 0), 'EXCLUDED', font=font)
        draw.text((int(x)-(bounds[2]-bounds[0])//2, int(y)-12), 'EXCLUDED',
                  font=font, fill='#ffcccc', stroke_width=1, stroke_fill='#300000')
    pixels = np.asarray(layer).copy()
    pixels[mask] = original[mask]
    return Image.fromarray(pixels)


def render_context_crop(source, mask, box, *, context_style='dark'):
    """Preserve selected RGB pixels and geometry in a controlled input trial.

    Natural context is experimental; this does not change the default planner.
    The contour is drawn outside the selection so tiny targets remain visible.
    """
    if context_style not in {'dark', 'natural', 'soft'}:
        raise ValueError('unknown planning context style')
    from scipy.ndimage import binary_dilation
    tile = ImageOps.contain(source.crop(box), (900, 800), Image.Resampling.LANCZOS)
    selected = np.asarray(Image.fromarray(
        mask[box[1]:box[3], box[0]:box[2]].astype('uint8') * 255).resize(
            tile.size, Image.Resampling.NEAREST)) > 0
    pixels = np.asarray(tile).copy()
    if context_style == 'dark':
        gray = np.asarray(tile.convert('L')) * .22
        pixels[~selected] = np.stack([gray] * 3, -1).astype('uint8')[~selected]
    elif context_style == 'soft':
        pixels[~selected] = (pixels[~selected].astype('float32') * .5).astype('uint8')
    pixels[binary_dilation(selected, iterations=2) & ~selected] = (0, 255, 100)
    return Image.fromarray(pixels)


def views_with_geometry(task):
    """``views`` plus the mask geometry run_group_pipeline computes from the saved masks (``_geometry``);
    a process-pool worker for --planning-view-workers (same files, same results, in parallel)."""
    from .geometry import mask_geometry
    from .group_geometry import precise_point
    candidate, objects, directory = task
    per_mask_views, mask_paths, digest = views(candidate, objects, directory)
    geometries = []
    for path in mask_paths:
        with Image.open(path) as image:
            arr = load_mask({'mask_path': path}, image.size)
        geometry = mask_geometry(arr)
        geometry['point'] = precise_point(arr)
        geometries.append(geometry)
    return per_mask_views, mask_paths, digest, geometries


def views(candidate, objects, directory):
    source = Image.open(candidate['image_path']).convert('RGB')
    original_masks = [load_mask(o, source.size) for o in objects]
    cleaned = [clean_remote_speckles(m) for m in original_masks]
    masks = [m for m,_ in cleaned]
    digest = hashlib.sha256(json.dumps([VIEW_VERSION, hashlib.sha256(source.tobytes()).hexdigest(),
              [str(o['object_id']) for o in objects],
              [hashlib.sha256(m.tobytes()).hexdigest() for m in original_masks]]).encode()).hexdigest()[:20]
    root = Path(directory)/digest;root.mkdir(parents=True, exist_ok=True)
    original = root/'original.png'
    meta_path = root/'views.json'
    if original.exists() and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        targets = cached.get('targets', [])
        if (cached.get('version') == VIEW_VERSION
                and [t.get('object_id') for t in targets] == [str(o['object_id']) for o in objects]
                and all((root/f'U{i}_crop.png').exists() and Path(t['mask_path']).exists()
                        for i,t in enumerate(targets,1))):
            return ([[str(original),str(root/f'U{i}_crop.png')] for i in range(1,len(objects)+1)],
                    [t['mask_path'] for t in targets], digest)
    if not original.exists():
        scene=source.copy();scene.thumbnail((1536,1536),Image.Resampling.LANCZOS);scene.save(original,compress_level=1)
    paths, requests, meta = [], [], []
    for index,(obj,mask) in enumerate(zip(objects,masks),1):
        uid=f'U{index}';mp=root/f'mask_{obj["object_id"]}.png';cp=root/f'{uid}_crop.png'
        if not mp.exists():Image.fromarray(mask.astype('uint8')*255).save(mp,compress_level=1)
        ys,xs=np.where(mask)
        pad=max(20,round(max(np.ptp(xs)+1,np.ptp(ys)+1)*.3))
        box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
             min(source.width,int(xs.max())+pad+1),min(source.height,int(ys.max())+pad+1))
        info=[dict(unit_id=uid,mask_bbox_xyxy=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],
                   crop_box_xyxy=list(box))]
        if not cp.exists():
            render_context_crop(source, mask, box).save(cp, compress_level=1)
        paths.append(str(mp));requests.append([str(original),str(cp)])
        cleanup={'removed_remote_pixels':cleaned[index-1][1], 'original_pixels':int(original_masks[index-1].sum())}
        if cleanup['removed_remote_pixels']:
            raw_path=root/f'mask_{obj["object_id"]}_original.png'
            Image.fromarray(original_masks[index-1].astype('uint8')*255).save(raw_path,compress_level=1)
            cleanup['original_mask_path']=str(raw_path)
        meta.append({**info[0], 'object_id':str(obj['object_id']), 'mask_path':str(mp),'mask_cleanup':cleanup})
    (root/'views.json').write_text(json.dumps({'version':VIEW_VERSION,'source':candidate['image_path'],'source_size':list(source.size),'targets':meta},indent=2))
    return requests,paths,digest


def apply_text_slot(raw, op, *, kind='multi', twin_single_add=False):
    """Explicit-text contract: bind wording into the payload or reject.

    Returns a rejection reason or None. Mutates ``raw['edit']`` in place.
    """
    from . import text_edits
    edit = raw.get('edit')
    if not isinstance(edit, dict):
        return None
    key = 'addition' if op == 'add' else 'replacement' if op == 'replace' else None
    try:
        if key:
            wording = edit.pop('text', '')
            edit[key] = text_edits.payload_with_text(edit.get(key, ''), wording)
        elif op == 'attribute' and edit.get('property') == 'text':
            edit['value'] = text_edits.clean_wording(edit.get('value'))
            if not edit['value']:
                return 'text_edit_without_wording'
    except ValueError as exc:
        return 'text_contract: ' + str(exc)
    if twin_single_add and kind == 'twins' and op == 'add' and TWIN_PLURAL.search(str(edit.get('addition', ''))):
        return 'twin_add_requires_one_compact_item'
    return None


def combine(raws, kind, assignments, objects, *, options=None):
    if len(raws)!=len(assignments):raise ValueError('per-mask response count mismatch')
    options = options or {}
    targets=[]
    for (uid,op),raw in zip(assignments.items(),raws):
        raw = deepcopy(raw)
        incompatibility = None
        if options.get('explicit_text') or options.get('twin_single_add'):
            if options.get('explicit_text'):
                incompatibility = apply_text_slot(raw, op, kind=kind,
                                                  twin_single_add=options.get('twin_single_add', False))
            elif kind == 'twins' and op == 'add' and TWIN_PLURAL.search(str((raw.get('edit') or {}).get('addition', ''))):
                incompatibility = 'twin_add_requires_one_compact_item'
        incompatibility = incompatibility or content.guard.incompatible_edit(raw.get('ref'), op)
        if not incompatibility and op == 'replace' and isinstance(raw.get('edit'), dict):
            incompatibility = content.guard.appearance_only_replacement(
                raw.get('ref'), raw['edit'].get('replacement'))
            if not incompatibility and options.get('garment_extent'):
                incompatibility = content.guard.garment_extent(raw.get('ref'), raw['edit'].get('replacement'))
        if not incompatibility and op == 'attribute':
            incompatibility = (content.guard.unchanged_declared_text(raw.get('ref'), raw.get('edit')) or
                content.guard.incompatible_attribute(raw.get('ref'), raw.get('edit')))
        if incompatibility:
            raw = {'reject': incompatibility,
                   '_generation_metadata': raw.get('_generation_metadata', {})}
        if isinstance(raw.get('edit'), dict):
            if op == 'replace':
                if set(raw['edit']) == {'replacement'}:
                    raw['edit'] = {'object': raw['edit']['replacement']}
                else:
                    raw = {'reject': False}
            elif op == 'add' and isinstance(raw['edit'].get('site'), str):
                if 'addition' in raw['edit'] and 'object' not in raw['edit']:
                    raw['edit']['object'] = raw['edit'].pop('addition')
                # In this single-anchor contract initial 'its' denotes that
                # anchor. The renderer owns the possessive connector. Do not
                # rewrite his/her or add a missing body part.
                site = raw['edit']['site'].strip()
                if site.lower().startswith('its '):
                    raw['edit']['site'] = site[4:]
        # IDs originate in the request, not a guessed model row order.
        if set(raw)-{'ref','edit','reject','_generation_metadata'}:
            raw={'reject':False, '_generation_metadata':raw.get('_generation_metadata',{})}
        wrapped={'targets':[dict(raw,unit_id=uid)],'_generation_metadata':raw.get('_generation_metadata',{})}
        decoded=content.decode(wrapped,'multi',{uid:op})
        if decoded.get('_minimal_error') or not decoded.get('targets'):
            targets.append(dict(unit_id=uid,entity='',ref_phrase='',feasible=False,
                                reason=decoded.get('_minimal_error','invalid response'),_format_error=True,
                                payload='',attributes=[],placement=None))
        else:targets.append(decoded['targets'][0])
    conflicts = []
    if kind == 'multi':
        from .host_conflicts import apply_conflicts
        conflicts = apply_conflicts(targets, assignments, objects)
    normalized=content.normalize(dict(targets=targets,edit_dependencies=[]),'multi',assignments,objects)
    if conflicts:
        normalized['host_conflicts'] = conflicts
    normalized['planning_contract']=VERSION
    if options:
        normalized['planning_options']=dict(options)
    normalized['_per_mask_responses']=deepcopy(raws)
    if kind=='twins':
        if len(set(assignments.values()))!=1:raise ValueError('adapted twins require the same operation')
        normalized['members']=normalized.pop('targets');normalized['operation']=next(iter(assignments.values()))
        normalized['twin_contract_version']=ADAPTED
        normalized['feasible']=all(t['feasible'] for t in normalized['members'])
        normalized['reason']='; '.join(t['reason'] for t in normalized['members'] if not t['feasible'])
    return normalized


def plan_batch(planner, batch, *, kind, root):
    options=options_of(planner)
    explicit_text=bool(options.get('explicit_text'))
    twin_add=bool(options.get('twin_single_add')) and kind=='twins'
    requests=[]
    for item in batch:
        ops=[item['selection']['operation']]*len(item['objects']) if kind=='twins' else item['selection']['operations']
        item['_mask_assignments']={f'U{i+1}':op for i,op in enumerate(ops)}
        for i,(uid,op) in enumerate(item['_mask_assignments'].items()):
            requests.append(dict(path=Path(root)/'planning_attempts'/item['gid']/f'{uid}.json',
                                 views=item['per_mask_views'][i],prompt=request_prompt(op,item['per_mask_views'][i],(item.get('planning_source_hints') or {}).get(uid),
                                                                                         explicit_text=explicit_text,twin_add=twin_add),
                                 schema=schema(op,explicit_text=explicit_text),item=item,unit_id=uid,
                                 cache_context={'contract':VERSION,'model':getattr(planner,'model_id','unknown'),
                                                **({'options':options} if options else {})}))
    # Persist after each bounded GPU batch; restart only unfinished requests.
    for start in range(0,len(requests),max(1,planner.batch_size)):
        _requests(planner,requests[start:start+max(1,planner.batch_size)])
    retries=[]
    for req in requests:
        meta=req['raw'].get('_generation_metadata',{})
        if meta.get('finish_reason')=='length':
            recovery={**req,'path':req['path'].with_name(req['path'].stem+'_truncation_retry.json'),
                      'token_limit':min(4096,max(2048,int(meta.get('output_tokens',1024))+768))}
            recovery.pop('raw',None)
            recovery['original_request']=req
            retries.append(recovery)
    for start in range(0,len(retries),max(1,planner.batch_size)):
        _requests(planner,retries[start:start+max(1,planner.batch_size)])
    for req in retries:
        original=req['original_request']
        original['first_raw']=original['raw']
        original['raw']=req['raw']
    reference_repairs = []
    if getattr(planner, 'reference_inspector', None) is not None:
        from .reference_alignment import bounded_repair_requests
        reference_repairs = bounded_repair_requests(
            planner, requests, planner.reference_inspector,
            max_extra_calls=max(0, len(requests)//4-len(retries)))
    results=[]
    for item in batch:
        own=[r for r in requests if r['item'] is item]
        raw=combine([r['raw'] for r in own],kind,item['_mask_assignments'],item['objects'],options=options)
        own_retries=[r for r in retries if r['item'] is item]
        own_repairs = [r for r in reference_repairs if r['item'] is item]
        raw['_planning_calls']=len(own)+len(own_retries)+len(own_repairs)
        raw['_planning_attempts']=[dict(unit_id=r['unit_id'],raw=r.get('first_raw',r.get('pre_reference_raw',r['raw'])),
                                        attempt='initial') for r in own]
        raw['_planning_attempts'] += [dict(unit_id=r['unit_id'],raw=r['raw'],attempt='truncation_recovery',
                                            token_limit=r['token_limit']) for r in own_retries]
        raw['_planning_attempts'] += [dict(unit_id=r['unit_id'],raw=r['raw'],
            attempt='reference_repair',applied=r['applied'],
            evidence=r['reference_evidence']) for r in own_repairs]
        if getattr(planner, 'reference_inspector', None) is not None:
            raw['_reference_evidence'] = {r['unit_id']:r.get('reference_evidence') for r in own}
            raw['_reference_repair_diagnostics'] = {r['unit_id']:{
                'before':r.get('reference_decision'),
                'after':r.get('post_reference_decision'),
                'status':r.get('reference_repair_status', 'not_retried'),
            } for r in own}
        results.append(raw)
    gates = getattr(planner, 'evidence_gates', None)
    if gates is not None:
        before = gates.calls
        gates.apply(planner, batch, results, kind)
        if options.get('replace_retry') or options.get('reader_retry'):
            _retry_gate_failures(planner, batch, results, requests, kind=kind, options=options, gates=gates)
        for raw in results:
            raw['_evidence_gate_calls_batch'] = gates.calls - before
    elif kind == 'twins' and any(item['selection'].get('twin_keep') for item in batch):
        raise ValueError('over-generated twin candidates require --evidence-gates')
    return results


APPEARANCE_ONLY = 'evidence_gate: payload=appearance_only'


def _retry_gate_failures(planner, batch, results, requests, *, kind, options, gates):
    """Opt-in: one re-plan round for targets whose only gate failure is retryable.

    replace_retry: payload=appearance_only, with ``retry_note`` (the rejected payload and why).
    reader_retry: reader=wrong/none, with ``reader_note`` (the old Ref and where the reader looked).
    Both kinds share one round of planner calls. Retried items are recombined from all their
    answers and gated again from scratch, so a retry never bypasses a gate. Twins are retried
    only when too few members passed.
    """
    retries = []
    for item, raw in zip(batch, results):
        if kind == 'twins' and raw.get('feasible'):
            continue
        for target in raw.get('members' if kind == 'twins' else 'targets') or []:
            if options.get('replace_retry') and target.get('reason') == APPEARANCE_ONLY:
                tag, note = 'replace_retry', retry_note(target.get('payload', ''))
            elif options.get('reader_retry') and target.get('reason') in READER_FAILURES:
                tag, note = 'reader_retry', reader_note(target)
            else:
                continue
            original = next(r for r in requests if r['item'] is item and r['unit_id'] == target.get('unit_id'))
            retry = {k: v for k, v in original.items() if k not in ('raw', 'first_raw', 'fingerprint')}
            retry.update(path=original['path'].with_name(original['path'].stem + f'_{tag}.json'),
                         prompt=original['prompt'] + note, retry_kind=tag, first_reason=target.get('reason'),
                         first_gates=target.get('evidence_gates'))
            retries.append(retry)
    if not retries:
        return
    for start in range(0, len(retries), max(1, planner.batch_size)):
        _requests(planner, retries[start:start + max(1, planner.batch_size)])
    redo = []
    for index, item in enumerate(batch):
        mine = {r['unit_id']: r for r in retries if r['item'] is item}
        if not mine:
            continue
        own = [r for r in requests if r['item'] is item]
        raw = combine([mine[r['unit_id']]['raw'] if r['unit_id'] in mine else r['raw'] for r in own],
                      kind, item['_mask_assignments'], item['objects'], options=options)
        old = results[index]
        for key in ('_reference_evidence', '_reference_repair_diagnostics'):
            if key in old:
                raw[key] = old[key]
        raw['_planning_calls'] = old.get('_planning_calls', 0) + len(mine)
        raw['_planning_attempts'] = list(old.get('_planning_attempts') or []) + [
            dict(unit_id=u, raw=r['raw'], attempt=r['retry_kind']) for u, r in mine.items()]
        for tag, key in (('replace_retry', '_replace_retry'), ('reader_retry', '_reader_retry')):
            units = sorted(u for u, r in mine.items() if r['retry_kind'] == tag)
            if units:
                raw[key] = {'units': units, 'first_reason': {u: mine[u]['first_reason'] for u in units}
                            if tag == 'reader_retry' else APPEARANCE_ONLY,
                            'first_gates': {u: mine[u]['first_gates'] for u in units}}
        results[index] = raw
        redo.append(index)
    gates.apply(planner, [batch[i] for i in redo], [results[i] for i in redo], kind)


_retry_appearance_only = _retry_gate_failures  # earlier name (replace retry only)
