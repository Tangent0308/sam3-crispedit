"""Compact one-call twin planning contract for the resident 27B planner."""
from copy import deepcopy


def output_schema():
    """Require identification even for rejection, preventing a cheap blind exit."""
    fields = {k:{'type':'string'} for k in ('object_id','entity','ref_phrase','kind','host',
              'current_value','mask_extent','excludes','distinction','nearest_rival','placement_site','scene_role','edit_purpose','fit')}
    fields['contact_xy'] = {'type':'array','items':{'type':'number','minimum':0,'maximum':1},'maxItems':2}
    fields.update({k:{'type':'boolean'} for k in ('ref_unique','site_visible','edit_feasible')})
    return {'type':'object','properties':{
        'members':{'type':'array','minItems':2,'maxItems':3,'items':{
            'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}},
        'feasible':{'type':'boolean'},'reason':{'type':'string'},'grounding_version':{'type':'integer','enum':[5]},
        'operation':{'type':'string','enum':['add','remove','replace','attribute']},
        'payload':{'type':'string'},'placement':{'type':['object','null']},
        'attributes':{'type':'array','items':{'type':'object'}},'replacement_form':{'type':'string'}},
        'required':['members','feasible'],'additionalProperties':False}

VERSION = 'concise-27b-scene-grounded-2'
PROMPT = '''Plan INDEPENDENT counterfactual edits: each output image edits ONLY ONE supplied region, starting from the same original photo. The same operation and payload sentence is reused for each output; this is NOT one joint edit of all regions. Different categories, original shapes and mounting orientations are allowed when each independent edit makes sense. Image 1 is the original scene with outside yellow locators. Image 2 has rows of CONTEXT, BINARY MASK, and SELECTED PIXELS on gray. White/selected pixels define the exact source extent; context explains function. Row order is not scene order. Identify every region independently before designing the shared edit. Do not copy category guesses between regions.

Each member must select a distinct editable region of comparable SIZE (mask area ratio at most 3). Mixing a whole object with a part is explicitly allowed. Object/part labels describe the source only and NEVER determine pair eligibility. Describe the exact selected extent, not its larger host. Disconnected visible pixels can belong to one occluded object. Small size, blur or clipping alone is not a reason to reject an otherwise identifiable target.

For each member, identify the EXACT entity and its nearest similar neighbor in the ORIGINAL scene. Write the shortest unique noun phrase using only verified visible distinctions. The ref itself must carry the distinction; recording it only in an internal field is insufficient. If a host contains repeated parts, identify both the host and the selected instance within it. Use original-scene relationships, not crop-relative positions. No extra clause, coordinates, IDs or guide colors. Do not invent a hidden component, location or identity.

Keep the assigned operation and select one ordinary request that is natural for EVERY member in its actual scene. Identify each entity independently before choosing shared content. Different categories or original shapes alone are not incompatibility. Never broaden or shrink the source extent to accommodate a convenient request. Choose one suitable edit within this response; no alternatives or revision requests.
- add: retain each host and its existing contents. Choose the SAME new content with a visible direct attachment or resting site inside each selected region. The contact surface, available space and attachment must physically support the complete addition at a plausible scale. Existing content must not be rewritten or replaced as an addition. Non-horizontal surfaces can support an addition when the attachment is physically plausible; resting requires stable support. placement_site is a short necessary subpart noun without repeating the host, or empty if the selected entity already is the contact site. No neighboring or distant reference anchors.
- remove: remove the exact selected dispensable entity. The remaining content must retain coherent support, attachment and structure. Do not delete essential anatomy, structural dependencies or occupied supports. Different removable categories need no shared identity or function; judge the actual consequences for each scene.
- replace: the SAME new content must fit each original extent, function and attachment. State a visible physical form or construction difference in the payload itself. Color, finish or material alone does not establish replacement. Preserve excluded surrounding parts and required support. Reject a concrete shared-design incompatibility.
- attribute: choose one property and a value that differs visibly from EVERY original and is appropriate for each selected material and scene. Preserve shape, markings and excluded parts except the explicitly requested property. Do not change unrequested optical or physical properties, or rename the selected entity to justify the change.

Before accepting, verify the exact shared payload/value against each member's occupancy, support and unchanged components. Keep this evidence internal and short. The same no-ref must naturally fit every member without implying one target. Payload describes ONE independent edit: a/an plus singular item, or an explicit count; never pluralize because there are several members.

Return JSON only, members FIRST. Do not write rendered instructions. Feasible format:
{"members":[{"object_id":"ID","entity":"precise singular source noun","ref_phrase":"the ...","kind":"object|part","host":"host noun or empty","current_value":"original requested property, else empty","mask_extent":"what is selected, up to 8 words","excludes":"nearest excluded part or object","distinction":"verified unique scene cue","nearest_rival":"closest similar instance and how this ref excludes it, or none","ref_unique":true,"placement_site":"visible subpart or empty","scene_role":"actual use here, at most 10 words","edit_purpose":"specific desired result here, at most 10 words","site_visible":true,"edit_feasible":true,"fit":"how this exact edit serves its purpose and preserves function"}],"feasible":true,"grounding_version":5,"operation":"ASSIGNED","payload":"","placement":null,"attributes":[],"replacement_form":"old structure versus new structure, only for replace"}
Every supplied ID appears once. Only add/replace has payload. Only attribute has attributes=[{"property":"color|material|finish|pattern|texture|text|tint|opacity","value":"..."}]. Only add has placement={"relation":"on|in|attached to|clipped to|sewn onto|tied around|hanging from","site":"","anchor_mode":"support"}; each member supplies its own placement_site. Other operations have placement=null and site_visible=true. No target_part, extent qualifier or extra operation. ALWAYS output the complete members array FIRST, even when rejecting. Then output feasible and the other operation fields. A rejection keeps all identified members and ends with "feasible":false,"reason":"concrete visible conflict". Never skip visual identification by returning only a rejection reason.
'''


from .planning_quality import RULES
from .edit_quality import PLAN_RULES
PROMPT += '\n' + RULES + '\n' + PLAN_RULES


def normalize_compact_twin(plan):
    """Bind the single source noun to legacy slots; do not infer missing evidence."""
    if plan.get('grounding_version') != 5:
        return plan
    result = deepcopy(plan)
    members = result.get('members')
    if not isinstance(members, list) or not members:
        raise ValueError('compact twin members missing')
    fit = []
    for m in members:
        required = ('entity','ref_phrase','kind','mask_extent','excludes','distinction','nearest_rival','fit')
        if any(not isinstance(m.get(k),str) or not m[k].strip() for k in required):
            raise ValueError('compact twin source evidence incomplete')
        if m.get('ref_unique') is not True:
            raise ValueError('compact twin reference is not unique')
        if m.get('scope') and m['scope'] != m['entity']:
            raise ValueError('compact twin overrides identified entity')
        if 'grounding' in m:
            raise ValueError('compact twin duplicates source grounding')
        m['scope'] = m['entity']
        m['grounding'] = {k:m.get(k) for k in
                          ('entity','kind','host','current_value','distinction','site_visible','edit_feasible','contact_xy')}
        m['grounding']['mask_description'] = m['mask_extent']+'; excludes '+m['excludes']
        fit.append({'object_id':m['object_id'],'natural':m.get('edit_feasible') is True,'reason':m['fit']})
    result['shared_fit'] = fit
    result.setdefault('target_part','')
    result.setdefault('remove_mode','whole')
    result.setdefault('qualifier','')
    return result
