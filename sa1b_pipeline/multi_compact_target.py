"""Compact observations for one-call multi-target planning; public protocol 2.0."""
from .instruction_protocol import BLIND_RULE

VERSION='grounded-thinking-v9-scene-grounded-2'


def schema():
    from .multi_plan_schema import multi_plan_schema
    old=multi_plan_schema()['properties']['targets']['items']['properties']
    def obj(p):return dict(type='object',properties=p,required=list(p),additionalProperties=False)
    def txt(n):return dict(type='string',maxLength=n)
    grounding=obj(dict(entity=txt(70),mask_description=txt(130),excluded=txt(100),distinction=txt(150),
        current_value=txt(70),scene_role=txt(100),edit_purpose=txt(100),unsupported_after=dict(type='array',items=txt(70),maxItems=2),
        edit_feasible=dict(type='boolean'),reason=txt(160),contact_xy=old['grounding']['properties']['contact_xy']))
    return obj(dict(grounding=grounding,ref_phrase=old['ref_phrase'],payload=old['payload'],
                    placement=old['placement'],attributes=old['attributes']))


def prompt(operation,bbox,label):
    import json
    from .instruction_diversity import content_preferences
    rule = content_preferences(operation, [])
    return '''Plan ONE natural image edit of the selected mask. Image 1 is the clean whole scene. Image 2 LEFT is ONLY selected original pixels on gray; RIGHT is local context with a yellow outside contour. These columns have different zoom scales. The LEFT cutout is the target, including ALL its pieces; gray/holes are excluded. Use the right context to identify the host and compare nearby lookalikes. Locate it in the whole image using the supplied box. Never choose a different instance.

Write short observation PHRASES, not explanations:
- entity: the truthful physical object/part noun covering the entire selected extent. No ID, annotation, generic region placeholder or explanatory sentence. Do not rename the larger host or a convenient subpart as the selected entity.
- mask_description: what pixels/parts are actually included. excluded: the nearest missing part or neighboring object. Context does not grant ownership of excluded pixels.
- distinction: compare this instance to its CLOSEST similar neighbor, including unselected neighbors. Write the truthful differentiating fact. Ref uses the shortest noun phrase carrying that distinction. Match the cutout to the exact object in the right context before claiming leftmost/rightmost; if two objects match, add a verified ordinal within a named set or a short local landmark.
- scene_role: the selected entity's actual visible use in this scene, at most 10 words. edit_purpose: what the chosen result achieves here, at most 10 words. Fit to a surface is not a purpose.
- edit_feasible/reason: choose a scene-appropriate request of the assigned operation, or give a concrete incompatibility. Small size, clipping, occlusion or unusual but coherent taste alone are not rejection reasons. A bad first idea is not a bad mask: try another appropriate content choice before rejecting.

Assigned operation: '''+operation+'\n'+rule+'''

Ref is the PUBLIC EXPLICIT reference: start with "the", name the actual object/part, and include only necessary distinguishing words. NEVER write 'selected', 'mask', 'region', U1 or coordinates in ref_phrase. Do not guess a technical identity. No trailing description clause, repeated host phrase, or source locator inside payload. Do not output no-ref; the program derives it. '''+BLIND_RULE+'''

Only add/replace has a concise new-item payload with an article; otherwise payload="". Only attribute has one property/value entry; otherwise attributes=[]. Only add has placement and contact_xy; otherwise null and []. Add site is empty if already the selected part, and must never repeat the host noun. unsupported_after=[] except actual stranded loads from remove. current_value="" for non-attribute. Keep internal phrases brief and instructions specific; prefer a simple fitting edit over forced novelty.

Source category hint (fallible, verify against LEFT pixels; never copy blindly): '''+str(label)+'\nOriginal-image normalized box [left,top,right,bottom]: '+json.dumps(bbox)


def normalize(raw,operation):
    result=dict(raw)
    result.update(unit_id='U1',object_id='',operation=operation,target_part='',remove_mode='whole',qualifier='',evidence='Compact per-mask visual plan.')
    return result


def group_schema():
    from .multi_plan_schema import multi_plan_schema
    s=multi_plan_schema(execution_contract=True)
    s['properties']['grounding_version']={'const':5}
    keys=('grounding_version','targets','generation_order','edit_dependencies','feasible','evidence')
    s['properties']={k:s['properties'][k] for k in keys};s['required']=list(keys)
    t=s['properties']['targets']['items'];base=schema()['properties']
    t['properties']={'unit_id':{'type':'string'},'operation':t['properties']['operation'],**base}
    t['required']=list(t['properties'])
    return s


def group_prompt(assignments,objects,order):
    base=prompt('attribute',[],'')
    prefix=base[:base.index('Assigned operation:')]
    prefix=prefix.replace('Plan ONE natural image edit of the selected mask.',
        'Plan one natural edit for EACH assigned U target in a single group response.')
    prefix=prefix.replace('Image 2 LEFT is ONLY selected original pixels on gray; RIGHT is local context with a yellow outside contour. These columns have different zoom scales. The LEFT cutout is the target, including ALL its pieces; gray/holes are excluded. Use the right context',
        'Image 2 has one row per U tag: LEFT local context with a yellow outside contour, MIDDLE an enlarged selected-pixel cutout on gray, RIGHT its enlarged binary mask. White in the RIGHT binary selects ALL editable pixels; black excludes, including holes. Match ALL white pieces to the MIDDLE photograph. The left context and middle cutout have different scales. Use the left context')
    prefix=prefix.replace('RIGHT cutout','MIDDLE cutout').replace('right context','left context').replace('LEFT cutout','MIDDLE cutout')
    rules=[]
    for op in ('add','remove','replace','attribute'):
        p=prompt(op,[],'');rules.append(p[p.index('Assigned operation:'):p.index('\n\nRef is')])
    suffix=base[base.index('\n\nRef is'):base.index('\n\nSource category hint')]
    from .instruction_diversity import PREFERENCE_RULE
    suffix += '\n' + PREFERENCE_RULE
    return prefix+'\n'.join(rules)+suffix+'''
Return every assigned target once, with its unit_id and assigned operation. Do not invent IDs or change operations. Keep infeasible entries with edit_feasible=false; group feasible means at least two feasible targets. generation_order includes every ID. Before returning edit_dependencies, compare EVERY pair for actual content dependence even when their source masks do not touch. A removed foreground object can reveal another target's surface even when their source masks are disjoint. Record the modified surface BEFORE the removal as reveals_edited_surface, naming the visible revealed material in evidence. The fixed surface selector excludes the foreground object, so removing first then recoloring only that selector would leave an old-color hole. For a new item on a modified host use uses_edited_host, host edit BEFORE addition. For visible shared shadows, reflections or occlusion use changes_shared_context with the physically required order and concrete evidence. Mere proximity or overlapping context crops is NOT dependence. Never delete an addition's host; mark the conflicting edit infeasible instead. Every relation links existing U IDs; reflect the relations in generation_order. Use [] only after checking these cases.
Use the reasoning budget to VERIFY the binary-mask extent and actual counterexample to each reference. Do not repeat the instructions. In particular, check whether attached parts outside the white mask remain coherent after removal. Output only brief observation phrases in the final JSON.
Source-category hints are fallible; pixels override them. Assigned targets:\n'''+'\n'.join(
        f"U{i+1}: operation={assignments[str(o['object_id'])]}, category_hint={o.get('category','')}" for i,o in enumerate(objects))


def normalize_group(raw):
    out=dict(raw);out['targets']=[]
    for t in raw.get('targets',[]):
        n=normalize(t,t.get('operation'));n['unit_id']=t.get('unit_id');out['targets'].append(n)
    return out
