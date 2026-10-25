"""Ref + instruction planning; deterministic bridge to the existing protocol.

No generated contact points, source observations or dependency claims are
invented. The original response is retained beside the internal execution data.
"""
from copy import deepcopy
import json
import re
from .instruction_protocol import make_spec, render, validate_pair, RELATIONS, REFERENCE_RELATIONS
from . import lean_planning

VERSION = 'minimal-ref-instruction-4'
COMMON = '''Design an everyday photo edit grounded in human preferences, visible context and the object's function; no arbitrary decoration or invented story.
Identify the WHITE mask pixels using their context, excluding holes/neighbors. Name the selected content, not its host; do not guess identity from location.
Ref must be unique BEFORE concise. Check the FULL image, including unmarked lookalikes. Keep distinctions needed to exclude them; use unique landmarks or verified scene-wide ordering, never crop order. Use an English noun phrase starting with "the"; no IDs, coordinates or annotation colors.
Follow the assigned operation. Judge the completed edit, preserving plausible anatomy, support and surroundings. Reject unresolved identification/ambiguity or an incompatible edit, not smallness/occlusion alone.
Check internally. Return only required JSON fields. Use one listed sentence ending with a period, without explanatory or protection clauses.'''

GRAMMAR = {
    'remove': 'remove: Remove <ref>. Allow plausible background/surface completion after deletion; preserve essential anatomy/support.',
    'replace': 'replace: Replace <ref> with <new item>. Specify a concrete new form fitting location, scale and function; appearance-only changes are attribute.',
    'attribute': 'attribute: Change the <color|material|finish|pattern|surface texture> of <ref> to <value>. Change the text on <ref> to "<text>". Tint <ref> <color>. Make <ref> <transparent|translucent|opaque>. Change a meaningful property visibly. Structural objects may receive appearance edits without altering function/structure.',
    'add': 'add: Add <new item> <relation> <ref>. Or: Add <new item> <relation> the <site> of <ref>. Use an article/count. Add absent content at a visible, available contact site inside the anchor, with space and no overwriting; no guessed hidden attachment. Relations (use exactly): ' + ' | '.join(sorted(RELATIONS - REFERENCE_RELATIONS - {'to'})) + '.',
}


def schema(kind, assignments=None, expected_ids=None):
    ids = list(expected_ids or assignments or [])
    def variant(uid):
        unit = {'const': uid} if uid else {'type':'string'}
        def obj(props):
            return dict(type='object', properties=props, required=list(props), additionalProperties=False)
        return [obj(dict(unit_id=unit, ref={'type':'string','maxLength':160},
                         instruction={'type':'string','maxLength':320})),
                obj(dict(unit_id=unit, reject={'type':'string','minLength':1,'maxLength':180}))]
    rows={'type':'array','items':{'anyOf':[v for uid in (ids or [None]) for v in variant(uid)]},
          'minItems':len(ids) if ids else 1,'maxItems':len(ids) if ids else 5}
    if kind=='twins':
        members=[]
        for uid in (ids or [None]):
            props=dict(unit_id={'const':uid} if uid else {'type':'string'},ref={'type':'string','maxLength':160})
            members.append(dict(type='object',properties=props,required=list(props),additionalProperties=False))
        rows['items']={'anyOf':members}
        return {'anyOf':[
            dict(type='object',properties=dict(members=rows,instruction={'type':'string','maxLength':320}),required=['members','instruction'],additionalProperties=False),
            dict(type='object',properties={'reject':{'type':'string','minLength':1,'maxLength':180}},required=['reject'],additionalProperties=False)]}
    return dict(type='object',properties={'targets':rows},required=['targets'],additionalProperties=False)


def prompt(kind, assignments, *, locations=None, retry_targets=None, locked=None, previous_reasons=None):
    cols='context / binary mask / selected pixels on gray' if kind=='twins' else 'context / selected pixels on gray / binary mask'
    text=f'Image 1: full scene with outside U locators. Image 2: one row per target, columns {cols}. Rows are not scene order.\n'+COMMON+'\n'+ '\n'.join(GRAMMAR[op] for op in GRAMMAR if op in set(assignments.values()))+'\n'
    if kind=='twins':
        text+='Twins: edit members independently from the SAME source. Specify ONE concrete change natural for all members, including across categories; its content must not depend on which member is edited. Return members with unit_id/ref and one instruction using literal {ref} exactly once in place of <ref>. If incompatible, return only reject with a short reason.\n'
    else:
        text+='Multi-target: edits must be compatible; preserve each other\'s support. Return targets with unit_id/ref/instruction, or unit_id/reject for rejected entries. Each instruction uses its ref verbatim exactly once. Return each requested ID once. The final group needs at least two accepted targets; never force edits to meet this count.\n'
    text+='Assigned operations: '+json.dumps(assignments)+'\n'
    if locations:
        text+='Original-image boxes PERCENT [0,100] (not editable extents): '+json.dumps(locations)+'\n'
    if retry_targets:
        text+='One retry with reassigned operations. Return ONLY '+json.dumps(retry_targets)+'. Previous rejection: '+json.dumps(previous_reasons)+'\n'
    if locked:
        concise=[dict(unit_id=t['unit_id'],ref=t['ref_phrase'],instruction=t['_minimal_instruction']) for t in locked]
        text+='Keep these accepted edits unchanged and compatible: '+json.dumps(concise)+'\n'
    return text


def validate_reference(ref):
    if not isinstance(ref,str) or not re.fullmatch(r'the [^\n]+',ref) or re.search(r'\b(?:selected|U\d+)\b|\bmask\s*\d+\b',ref,re.I):
        raise ValueError('ref must name a unique source target without selection markers')


def parse_instruction(ref, instruction, operation):
    """Parse anchored, canonical forms; never silently reinterpret prose."""
    validate_reference(ref)
    if not isinstance(instruction,str) or not instruction.strip():
        raise ValueError('instruction must be a nonempty sentence')
    ref=' '.join(ref.split());instruction=' '.join(instruction.split()).rstrip('.')+'.';body=instruction[:-1]
    if instruction.count(ref)!=1:
        raise ValueError('instruction must use the exact ref once')
    # Retain the full description internally rather than guessing a new category.
    scope=ref.removeprefix('the ')
    kwargs={}
    if operation=='remove':
        if body!='Remove '+ref:raise ValueError('remove target/grammar mismatch')
    elif operation=='replace':
        prefix='Replace '+ref+' with '
        if not body.startswith(prefix):raise ValueError('replace target/grammar mismatch')
        kwargs['payload']=body[len(prefix):]
    elif operation=='add':
        if not body.startswith('Add ') or not body.endswith(' '+ref):raise ValueError('add anchor/grammar mismatch')
        tail=body[4:-(len(ref)+1)]
        matches=[]
        for relation in sorted(RELATIONS-REFERENCE_RELATIONS,key=len,reverse=True):
            direct=re.fullmatch(r'(.+) '+re.escape(relation),tail)
            site=re.fullmatch(r'(.+) '+re.escape(relation)+r' ((?:the|a|an|its|each|every|both) .+) of',tail)
            if direct:matches.append((direct[1],relation,''))
            if site:matches.append((site[1],relation,site[2].removeprefix('the ')))
        # Prefer a compound relation over its trailing preposition.
        if not matches:raise ValueError('add must use a direct attachment relation')
        payload,relation,site=matches[0]
        if not re.match(r'^(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|[1-9][0-9]*)\s+',payload,re.I):
            raise ValueError('add needs explicit new content/count')
        kwargs.update(payload=payload,placement=dict(relation=relation,site=site,anchor_mode='support'))
    elif operation=='attribute':
        change=re.fullmatch(r'Change the (color|material|finish|pattern|surface texture) of '+re.escape(ref)+r' to (.+)',body)
        writing=re.fullmatch(r'Change the text on '+re.escape(ref)+r' to (".*")',body)
        tint=re.fullmatch(r'Tint '+re.escape(ref)+r' (.+)',body)
        opacity=re.fullmatch(r'Make '+re.escape(ref)+r' (transparent|translucent|opaque)',body)
        if change:prop,value=change[1].replace('surface texture','texture'),change[2]
        elif writing:prop,value='text',json.loads(writing[1])
        elif tint:prop,value='tint',tint[1]
        elif opacity:prop,value='opacity',opacity[1]
        else:raise ValueError('attribute target/grammar mismatch')
        kwargs['attributes']=[dict(property=prop,value=value)]
    else:raise ValueError('unknown assigned operation')
    spec=make_spec(operation,ref,scope,**kwargs)
    if render(spec)!=instruction:raise ValueError('instruction cannot round-trip through protocol 2.0')
    return spec


def decode(raw, kind, assignments, *, content_specs=None):
    """Bind original compact output to program-owned execution slots."""
    result={'_minimal_response':deepcopy(raw),'_generation_metadata':raw.get('_generation_metadata',{})}
    key='members' if kind=='twins' else 'targets'
    try:
        if raw.get('_generation_metadata',{}).get('finish_reason')=='length':raise ValueError('planner output truncated')
        if kind=='twins':
            if set(raw)-{'_generation_metadata'}=={'reject'} and raw['reject'].strip():
                rows=[dict(unit_id=uid,reject=raw['reject']) for uid in assignments]
            else:
                template=raw['instruction']
                if template.count('{ref}')!=1:raise ValueError('shared instruction needs exactly one {ref}')
                rows=[]
                for row in raw['members']:
                    if set(row)!={'unit_id','ref'}:raise ValueError('twin members need only unit_id/ref')
                    rows.append({**row,'instruction':template.replace('{ref}',row['ref'])})
        else:rows=raw[key]
        if len(rows)!=len(assignments) or {r['unit_id'] for r in rows}!=set(assignments):raise ValueError('missing/duplicate/unknown target ID')
        converted=[];specs=[]
        for row in rows:
            uid=row['unit_id']
            if set(row)=={'unit_id','reject'} and row['reject'].strip():
                converted.append(dict(unit_id=uid,entity='',ref_phrase='',feasible=False,reason=row['reject']))
                continue
            try:
                if set(row)!={'unit_id','ref','instruction'}:raise ValueError('unexpected output fields')
                if content_specs is None:
                    spec=parse_instruction(row['ref'],row['instruction'],assignments[uid])
                else:
                    # Structured planners already own the semantic boundaries.
                    # Re-parsing "on ... of ... on ..." can absorb a site into
                    # payload while still round-tripping the visible sentence.
                    if uid not in content_specs:
                        raise ValueError('missing valid structured content for target')
                    spec=deepcopy(content_specs[uid])
                    validate_reference(row['ref'])
                    if spec['ref'] != row['ref'] or spec['operation'] != assignments[uid]:
                        raise ValueError('structured content is bound to a different target/operation')
                    if spec['operation'] == 'add' and not re.match(
                            r'^(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|[1-9][0-9]*)\s+',
                            spec['payload'], re.I):
                        raise ValueError('add needs explicit new content/count')
                    validate_pair(spec,row['instruction'],render(spec,False))
                specs.append(spec)
                converted.append(dict(unit_id=uid,entity=spec['scope'],ref_phrase=spec['ref'],feasible=True,reason='',
                    payload=spec['payload'],attributes=spec['attributes'],placement=spec['placement'] or None,
                    placement_site=spec['placement'].get('site',''),_minimal_instruction=render(spec)))
            except (ValueError,TypeError) as exc:
                converted.append(dict(unit_id=uid,entity='',ref_phrase=row.get('ref',''),feasible=False,
                                      reason=str(exc),_format_error=True))
        result[key]=converted
        if kind=='twins':
            result.update(feasible=all(t['feasible'] for t in converted),reason='; '.join(t['reason'] for t in converted if not t['feasible']))
            if any(t.get('_format_error') for t in converted):result['_minimal_error']=result['reason']
            if result['feasible']:
                if len({render(s,False) for s in specs})!=1:
                    raise ValueError('twins do not share one blind instruction')
                first=specs[0];result.update(payload=first['payload'],attributes=first['attributes'],
                    placement={**first['placement'],'site':''} if first['placement'] else None)
        else:result['edit_dependencies']=[]
    except (ValueError,KeyError,TypeError,AttributeError) as exc:
        result['_minimal_error']=str(exc)
    return result


def validate_response(raw, kind, assignments, expected_ids=None):
    if raw.get('_minimal_error'):raise ValueError(raw['_minimal_error'])
    lean_planning.validate_response(raw,kind,assignments,expected_ids)


def normalize(raw,kind,assignments,objects):
    validate_response(raw,kind,assignments)
    out=lean_planning.normalize(raw,kind,assignments,objects)
    out.update(grounding_version=7,planning_contract=VERSION)
    for t in out['members' if kind=='twins' else 'targets']:
        # These are unavailable observations, not fabricated checks or points.
        t.pop('current_value',None);t.pop('contact_xy',None)
        t['grounding'].pop('current_value',None);t['grounding'].pop('contact_xy',None)
        t['grounding']['source']='parsed_ref_only'
    return out


def validate_edit(item):
    from .multi_semantics import operation_exclusion, _MANUFACTURED_HOSTS
    ref=item['ref_phrase']
    leading=re.split(r'\s+(?:of|on|in|at|behind|beside|near|with|under|above|below|from|worn by|held by|carried by)\s+',ref,1)[0]
    hosts=set(re.findall(r'[a-z]+',ref.lower())) & _MANUFACTURED_HOSTS
    check_scope=(' '.join(sorted(hosts))+' '+leading) if hosts else leading
    mechanical_head=bool(re.search(r'\b(?:lamp|light|signal|screw|bolt|shower) head$',leading))
    exclusion='' if mechanical_head else operation_exclusion(check_scope,item['operation'])
    if exclusion:raise ValueError(exclusion)
    if item.get('grounding',{}).get('edit_feasible') is False:
        raise ValueError(item['grounding'].get('reason') or 'rejected minimal edit')
    spec=make_spec(item['operation'],item['ref_phrase'],item['scope'],item.get('payload',''),
                   attributes=item.get('attributes',[]),placement=item.get('placement'))
    parse_instruction(item['ref_phrase'],render(spec),item['operation'])


def named_site_contact(mask_path, site):
    """Geometric hint for explicit upper/lower edges, never observed evidence.

    Do not guess front/back, anatomical sides, or an unnamed attachment site.
    The post-generation difference/contact gate remains authoritative.
    """
    if not re.fullmatch(r'(?:top|upper|bottom|lower) (?:edge|rim|border)',site.strip().lower()):return None
    import numpy as np
    from PIL import Image
    mask=np.asarray(Image.open(mask_path).convert('L'))>127
    ys,xs=np.nonzero(mask)
    if not len(xs):return None
    bottom=site.strip().lower().startswith(('bottom','lower'))
    target_y=ys.max() if bottom else ys.min()
    band=np.abs(ys-target_y)<=max(1,round((ys.max()-ys.min())*.04))
    bx,by=xs[band],ys[band]
    j=int(np.argmin((bx-np.median(bx))**2+(by-target_y)**2))
    return dict(version='named-site-geometry-1',point=[int(bx[j]),int(by[j])],site=site,
                provenance='deterministic mask-edge hint; not a model-observed contact',verified=False)
