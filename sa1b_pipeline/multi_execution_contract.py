"""Single-call plan evidence, scope contradictions and content dependency scheduling.

No category whitelist, extra model request or source-specific exception. Geometry
only supplies a conservative ordering hint; it does not certify visual semantics.
"""
from __future__ import annotations
import re
from itertools import combinations

VERSION = 'multi-execution-contract-3-shared-context'

RULES = '''EXECUTION CONTRACT (grounding_version=4):
Observe the actual WHITE selected extent before designing each edit. In grounding:
- extent_match: true only when the entity and ref name exactly the selected visible object/part. A hull is not a whole boat, a collar is not the coat, a painted strip is not the whole facade. Occlusion/image-edge clipping alone is not a mismatch.
- unsupported_after: list visible attached loads/parts that removal would strand outside the selector; [] for non-remove or no such remnants. Never erase those excluded parts to hide a bad plan.
- ref_unique: compare with ALL lookalikes in the original image; include the shortest necessary verified ordinal/location. 'The window/building/background person' is insufficient if several match. Use a neutral physical noun if identity is uncertain, not a guessed tool name.
- preserve: at most three short observable invariants, e.g. window grid, glass transparency, lettering, carried bag. Do not promise generic 'everything'.
- required_change: one short visible success criterion; replace must name the new structure, not only its color.
Return each assigned target even when incompatible; set its edit_feasible=false with a concrete reason. Group feasible means at least two individually feasible edits remain. The program may drop invalid targets, never change their assigned operations or source masks. Do not mark the other targets infeasible merely because one fails.
Return edit_dependencies as [{before:'U1',after:'U2',relation:'reveals_edited_surface|uses_edited_host|changes_shared_context',evidence:'short visible relation'}], or []. IDs must exist. If removing target B reveals the material of attribute target A, execute A BEFORE B, so the hole is filled using A's final appearance. If an addition uses a modified host, modify the host before adding. Spatial proximity alone is not a dependency. Host deletion plus a dependent attachment is incompatible, not fixable by ordering. Apply these relations to generation_order; never default to removal-first against a dependency.
Before returning edit_dependencies, compare EVERY pair for actual content dependence even when their source masks do not touch. A removed foreground object can reveal another target's water, fabric, wall or ground; those masks are normally disjoint. Record the modified surface BEFORE the removal as reveals_edited_surface, naming the visible revealed material in evidence. The fixed surface selector excludes the foreground object, so removing first then recoloring only that selector would leave an old-color hole. For a new item on a modified host use uses_edited_host, host edit BEFORE addition. For visible shared shadows, reflections or occlusion use changes_shared_context with the physically required order and concrete evidence. Mere proximity or overlapping context crops is NOT dependence. Never delete an addition's host; mark the conflicting edit infeasible instead. Every relation links existing U IDs; reflect the relations in generation_order. Use [] only after checking these cases.
Keep the published instruction one concise clause. These evidence and dependency fields are internal and do not enter no-ref.
'''


def scope_contradiction(description, entity):
    """Catch explicit selected-part-of-host versus whole-host statements only."""
    text = str(description).lower().strip()
    ent = re.sub(r'^(the|a|an)\s+', '', str(entity).lower().strip())
    # 'hull of a long-tail boat' explicitly describes a part; 'front of a boat'
    # may merely describe viewpoint and is intentionally not included.
    parts = r'hull|roof|roof edge|collar|cuff|sleeve|handle|frame|rim|shaft|stem|base|panel|strip|patch|section|portion'
    match = re.match(r'^(?:(?:the|a|an|visible|selected|upper|lower|left|right|wooden|metal)\s+)*('+parts+r')\s+of\s+(?:the\s+|a\s+|an\s+)?(.+)', text)
    if not match or not ent:
        return ''
    part, host = match.groups()
    if re.search(r'\b'+re.escape(part)+r'\b', ent):
        return ''
    if re.search(r'\b'+re.escape(ent)+r'\b', host):
        return 'selected part-of-host contradicts whole entity: '+part+' versus '+ent
    return ''


def excluded_support_contradiction(grounding):
    """Reject explicit attached-load contradictions, not merely nearby objects.

    This is a narrow consistency guard on observations, not a visual detector.
    A cap/label is deliberately absent: its removal does not strand its host.
    """
    entity=str(grounding.get('entity','')).lower()
    excluded=str(grounding.get('excluded','')).lower()
    if re.search(r'\b(?:another|other|unrelated|background)\b',excluded):
        return ''
    roles={'hull':('canopy|mast|cabin|deck','hull|boat'),
           'column':('roof|beam|arch','column'), 'pillar':('roof|beam|arch','pillar'),
           'trunk':('branches|crown','trunk|tree'), 'stem':('flower|flower head','stem')}
    for part,(loads,host) in roles.items():
        if (re.search(r'\b'+part+r'\b',entity)
                and re.search(r'\b(?:'+loads+r')\b',excluded)
                and re.search(r'\b(?:above|on|over|of|attached to) (?:the |a |this )?(?:'+host+r')\b',excluded)):
            return 'removal contradicts explicit excluded attachment: '+excluded
    return ''


def validate_target_evidence(item, *, required=False):
    g = item.get('grounding') or {}
    reason = scope_contradiction(g.get('mask_description',''), g.get('entity', item.get('scope','')))
    if reason:
        raise ValueError(reason)
    if required:
        if g.get('extent_match') is not True:
            raise ValueError('selected extent and requested entity do not match')
        if g.get('ref_unique') is not True:
            raise ValueError('reference is ambiguous in the original scene')
        if not isinstance(g.get('unsupported_after'), list) or not isinstance(g.get('preserve'), list):
            raise ValueError('missing execution observations')
        if not str(g.get('required_change','')).strip():
            raise ValueError('missing visible completion criterion')
    if item.get('operation') == 'remove' and excluded_support_contradiction(g):
        raise ValueError(excluded_support_contradiction(g))
    if item.get('operation') == 'remove' and g.get('unsupported_after'):
        raise ValueError('removal leaves unsupported source contents: '+', '.join(g['unsupported_after']))


def order_dependencies(ids, preferred, dependencies):
    """Stable topological order; reject bad IDs and cycles instead of dropping edges."""
    if len(set(ids))!=len(ids) or set(preferred)!=set(ids) or len(preferred)!=len(ids):
        raise ValueError('invalid dependency target order')
    edges=[]
    for pair in dependencies:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2 or pair[0] not in ids or pair[1] not in ids or pair[0]==pair[1]:
            raise ValueError('invalid edit dependency')
        if list(pair) not in edges:edges.append(list(pair))
    result=[]
    while len(result)<len(ids):
        ready=next((u for u in preferred if u not in result and all(a in result for a,b in edges if b==u)),None)
        if ready is None:raise ValueError('cyclic edit dependencies')
        result.append(ready)
    return result,edges


def bind_dependencies(raw, all_ids, kept_ids, preferred):
    records=raw.get('edit_dependencies',[])
    if not isinstance(records,list):raise ValueError('invalid dependency records')
    evidence=[]
    for d in records:
        if not isinstance(d,dict) or d.get('relation') not in {'reveals_edited_surface','uses_edited_host','changes_shared_context'} or not d.get('evidence'):
            raise ValueError('invalid dependency evidence')
        pair=[d.get('before'),d.get('after')]
        order_dependencies(all_ids,all_ids,[pair])
        if d["relation"] == "uses_edited_host" and pair[1] in kept_ids and pair[0] not in kept_ids:
            raise ValueError("attachment depends on a rejected host edit")
        if set(pair)<=set(kept_ids):evidence.append(d)
    order,pairs=order_dependencies(kept_ids,preferred,[[d['before'],d['after']] for d in evidence])
    return order,pairs,evidence


def _content_words(text):
    # No object-category whitelist. Shared observation words only corroborate
    # geometry; they are never sufficient to create an edge on their own.
    stop = set('the a an of on in and or to with from by at this that selected visible '
               'object part surface background surrounding nearest original edited '
               'left right upper lower front back small large'.split())
    return {w for w in re.findall(r"[a-z]+", str(text).lower()) if len(w)>2 and w not in stop}


def infer_reveal_dependencies(group):
    """Recover a narrow annotation gap between an occluder and edited surface.

    Multiple scale-relative rings avoid the old single thin ring's false
    negatives. Require enclosure, a short gap, a larger host, and either strong
    surrounding geometry or corroborating source observations. Nearby crop
    boxes, generic nouns and far-away surfaces alone never establish dependency.
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    size = Image.open(group['source_image']).size
    from .geometry import load_mask
    masks={u['unit_id']:load_mask({'mask_path':u.get('mask_v1') or u['mask']}, size)
           for u in group['units']}
    evidence=[]
    for rem in group['units']:
        if rem['operation']!='remove':continue
        m=masks[rem['unit_id']];ys,xs=np.nonzero(m)
        if not len(xs):continue
        extent=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
        radii=sorted({max(3,round(extent*f)) for f in (.08,.16,.32)})
        pad=radii[-1]+1
        box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
             min(size[0],int(xs.max())+pad+1),min(size[1],int(ys.max())+pad+1))
        sl=np.s_[box[1]:box[3],box[0]:box[2]]
        distance=distance_transform_edt(~m[sl])
        observations=rem.get('grounding') or {}
        rem_words=_content_words(' '.join(str(observations.get(k,'')) for k in
            ('excluded','reason','mask_description')))
        candidates=[]
        for host in group['units']:
            if host['operation']!='attribute':continue
            h=masks[host['unit_id']];hy,hx=np.nonzero(h)
            if not len(hx) or h.sum()<2*m.sum():continue
            contained=float(((xs>=hx.min())&(xs<=hx.max())&(ys>=hy.min())&(ys<=hy.max())).mean())
            if contained<.85 or not h[sl].any():continue
            gap=float(distance[h[sl]].min())
            if gap>max(3,extent*.12):continue
            rings=[]
            for radius in radii:
                ring=(distance>0)&(distance<=radius)
                rings.append({'radius':radius,'coverage':float((ring & h[sl]).sum()/max(1,ring.sum()))})
            coverage=max(r['coverage'] for r in rings)
            host_words=_content_words((host.get('grounding') or {}).get('entity') or host.get('scope',''))
            shared=sorted(host_words & rem_words)
            if coverage>=.50 or (coverage>=.20 and shared):
                candidates.append({'before':host['unit_id'],'after':rem['unit_id'],
                    'relation':'reveals_edited_surface','evidence':'multiscale source-mask enclosure with gap check',
                    'bbox_containment':contained,'gap_pixels':gap,'ring_coverage':coverage,
                    'rings':rings,'shared_observation_words':shared})
        # Multiple plausible revealed materials need an explicit planner edge;
        # geometry must not guess which one is behind the removed object.
        if len(candidates)==1:evidence.extend(candidates)
    return evidence


def infer_context_dependencies(group):
    """Condition nearby local edits on a large surface's already edited image.

    This is an execution dependency, never a distance-based eligibility rule.
    It grants no new writable pixels and does not change public selectors.
    Multiple possible surfaces are left to explicit evidence rather than guessed.
    """
    import numpy as np
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from .geometry import load_mask
    size = Image.open(group['source_image']).size
    units = group['units']
    masks = {u['unit_id']: load_mask({'mask_path':u.get('mask_v1') or u['mask']}, size)
             for u in units}
    areas = {uid:int(m.sum()) for uid,m in masks.items()}
    result = []
    for target in units:
        uid = target['unit_id']; m = masks[uid]; ys,xs = np.nonzero(m)
        if not len(xs): continue
        extent = max(int(np.ptp(xs))+1, int(np.ptp(ys))+1)
        radius = max(4, round(extent*.5))
        box = (max(0,int(xs.min())-radius),max(0,int(ys.min())-radius),
               min(size[0],int(xs.max())+radius+1),min(size[1],int(ys.max())+radius+1))
        sl = np.s_[box[1]:box[3],box[0]:box[2]]
        distance = distance_transform_edt(~m[sl]); ring=(distance>0)&(distance<=radius)
        candidates=[]
        for host in units:
            hid = host['unit_id']
            if hid == uid or host['operation'] != 'attribute': continue
            if areas[hid] < max(5*areas[uid], .1*size[0]*size[1]): continue
            h=masks[hid]; hy,hx=np.nonzero(h)
            cx,cy=float(xs.mean()),float(ys.mean())
            if not (hx.min()-radius <= cx <= hx.max()+radius and
                    hy.min()-radius <= cy <= hy.max()+radius): continue
            coverage=float((h[sl]&ring).sum()/max(1,ring.sum()))
            if coverage < .2: continue
            candidates.append(dict(before=hid, after=uid, relation='changes_shared_context',
                evidence='large edited surface occupies the local edit context',
                ring_coverage=coverage, radius=radius, permission='conditioning_only'))
        if len(candidates)==1: result.extend(candidates)
    return result


def schedule_content_dependencies(group):
    ids=[u['unit_id'] for u in group['units']]
    inferred=infer_reveal_dependencies(group)
    evidence=list(group.get('dependency_evidence',[]))
    explicit=list(group.get('edit_dependencies',[]))
    order,pairs=order_dependencies(ids,group['generation_order'],explicit+[[e['before'],e['after']] for e in inferred])
    context=[]; scheduling=list(pairs)
    for e in infer_context_dependencies(group):
        pair=[e['before'],e['after']]
        if pair in scheduling:continue
        try:
            revised,edges=order_dependencies(ids,order,scheduling+[pair])
        except ValueError:
            continue  # A geometric conditioning hint never overrides explicit ordering.
        order,scheduling=revised,edges;context.append(e)
    group.update(generation_order=order,edit_dependencies=pairs)
    group['dependency_schedule']={'version':VERSION,'inferred':inferred,'order':order,
                                  'conditioning_edges':[[e['before'],e['after']] for e in context],
                                  'evidence':evidence+inferred+context}
    return group


def dependency_context(group, uid, completed_ids):
    """Internal conditioning only; published instructions and masks stay fixed."""
    members={u['unit_id']:u for u in group['units']}
    records=group.get('dependency_schedule',{}).get('evidence',[])
    seen=set();context=[]
    for e in records:
        before=e.get('before')
        if e.get('after')!=uid or before not in completed_ids or before in seen:continue
        seen.add(before);host=members[before]
        context.append({'before':before,'relation':e['relation'],
                        'completed_instruction':host['instruction'],
                        'entity':(host.get('grounding') or {}).get('entity') or host.get('scope','')})
    return context


def dependency_prompt(context):
    if not context:return ''
    lines=['This input photograph already contains the following completed edits; preserve their CURRENT appearance.']
    for e in context:
        lines.append(e['completed_instruction'])
        if e['relation']=='reveals_edited_surface':
            lines.append('When removing the target, continue the CURRENT adjacent '+e['entity']+
                ' through the exposed area with matching color, texture, perspective and lighting. '
                'Do not reconstruct its old appearance or leave a differently colored patch.')
        elif e['relation']=='uses_edited_host':
            lines.append('Fit the new item to the already modified host; retain that host modification.')
        else:
            lines.append('Keep shared background, occlusion, lighting and shadows consistent with these completed edits.')
    return ' '.join(lines)


def retain_feasible_plan(group, adjacent_pairs=None):
    """Keep the largest valid pre-generation subset; record every removal."""
    from .group_schema import validate_multi_record
    units=group['units'];ids={u['unit_id'] for u in units}
    for n in range(len(units),1,-1):
        for chosen in combinations(units,n):
            kept={u['unit_id'] for u in chosen};objects={str(u['object_id']) for u in chosen}
            if any(b in kept and a not in kept for a,b in group.get('edit_dependencies',[])):continue
            pairs=[]  # Legacy argument accepted for callers; distance never gates retention.
            trial={**group,'units':[dict(u,support_mask=u['mask']) for u in chosen],
                   'n_targets':n,'operations':[u['operation'] for u in chosen],
                   'generation_order':[u for u in group['generation_order'] if u in kept],
                   'adjacent_pairs':pairs}
            try:validate_multi_record(trial, require_resolved_regions=False)
            except ValueError:continue
            removed=[u for u in units if u['unit_id'] not in kept]
            group.update(units=list(chosen),n_targets=n,operations=trial['operations'],
                         generation_order=trial['generation_order'],adjacent_pairs=pairs)
            group['edit_dependencies']=[p for p in group.get('edit_dependencies',[]) if set(p)<=kept]
            group.setdefault('protected_source_masks',[]).extend(u['mask'] for u in removed)
            group.setdefault('planning_retention',{}).setdefault('dropped',[]).extend(
                {'unit_id':u['unit_id'],'object_id':u['object_id'],'mask':u['mask'],
                 'reason':'retained type/dependency contract','stage':'plan_subset'} for u in removed)
            return group
    raise ValueError('no feasible multi subset satisfies type and dependency contract')


def prepare_dependency_interfaces(group, directory):
    """Allow a bounded removal seam to meet an already edited background.

    Source selectors stay immutable. Only a declared reveal predecessor can
    waive previous-pixel protection, inside this local band. All other source
    targets and dropped selectors remain protected. The connected component is
    audited and committed atomically, so this never permits cross-component writes.
    """
    import numpy as np
    from pathlib import Path
    from PIL import Image
    from scipy.ndimage import distance_transform_edt
    from .geometry import load_mask
    from .group_generation import _write_mask
    from .editor import _support_radius
    size=Image.open(group['source_image']).size
    members={u['unit_id']:u for u in group['units']}
    masks={uid:load_mask({'mask_path':u.get('mask_v1') or u['mask']},size) for uid,u in members.items()}
    records=group.get('dependency_schedule',{}).get('evidence',[])
    evidence=[]
    for uid,member in members.items():
        if member['operation']!='remove':continue
        hosts={e['before'] for e in records if e.get('after')==uid and e.get('relation')=='reveals_edited_surface'
               and e.get('before') in members}
        if not hosts:continue
        target=masks[uid];ys,xs=np.nonzero(target)
        if not len(xs):continue
        extent=max(int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))
        unit=member['unit']
        base=_support_radius(target,'remove',unit.get('support_policy'))
        radius=min(40,max(base,round(extent*.24),8))
        host=np.logical_or.reduce([masks[h] for h in hosts])
        # Work in a bounded crop: no full-resolution distance map per dependency.
        pad=radius*2+2
        box=(max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
             min(size[0],int(xs.max())+pad+1),min(size[1],int(ys.max())+pad+1))
        sl=np.s_[box[1]:box[3],box[0]:box[2]]
        if not host[sl].any():continue
        near_target=distance_transform_edt(~target[sl])<=radius
        near_host=distance_transform_edt(~host[sl])<=radius
        band=np.zeros(target.shape,bool);band[sl]=near_target & near_host
        other=[m for k,m in masks.items() if k not in hosts and k!=uid]
        other += [load_mask({'mask_path':p},size) for p in group.get('protected_source_masks',[])]
        forbidden=np.logical_or.reduce(other) if other else np.zeros(target.shape,bool)
        forbidden &= ~target
        band &= ~forbidden
        old_protected=load_mask({'mask_path':unit['protected_mask_path']},size)
        permission=band & host & ~forbidden
        protected=old_protected & ~permission
        support=load_mask({'mask_path':unit['execution_mask_path']},size)
        support=(support|band)&~protected
        root=Path(directory)/uid
        unit['execution_mask_path']=_write_mask(root/'execution.png',support)
        unit['protected_mask_path']=_write_mask(root/'protected.png',protected)
        unit['dependency_overlap_mask_path']=_write_mask(root/'prior_write_permission.png',permission)
        unit['dependency_interface_mask_path']=_write_mask(root/'interface.png',band)
        evidence.append({'unit_id':uid,'host_ids':sorted(hosts),'radius':radius,
                         'permission_pixels':int(permission.sum()),'support_pixels':int(support.sum())})
    group['dependency_interfaces']={'version':1,'regions':evidence}
    return evidence
