"""Experimental relation-aware removal planner; never changes source masks.

One independent planning call per case. It replaces neither the production
planner nor its audit until paired visual validation is complete.
"""
import argparse
import json
import time
from pathlib import Path
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
from synthesis_pipeline.generate_samtok_plan import parse_json_object
from synthesis_pipeline.prepare_samtok_data import write_jsonl, load_jsonl
from utils.context_edit import protected_neighbors
import utils.vlm_utils as vlm
from utils.removal_relations import relation_context_bbox, relation_input_evidence, is_grounded_relation_policy
from synthesis_pipeline.labeling_checkpoint import (
    CaseCheckpoints, bind_settings, ensure_sources, recover_rows, fingerprint, file_digest,
)


PROMPT = '''Plan a localized removal using two images: the clean full photograph,
and its context crop with a black/white TARGET boundary. The dots and labels
are annotations, not source colors. Only pixels inside that boundary belong
to the selected target. Holes and objects on the other side do not.
Identify the selected instance directly from these images, not from a prior
instruction. Check each object's own pixels before naming a group.
An outer contour can surround gaps occupied by other objects; being inside its
bounding rectangle is not membership. A selected component is not permission
to name or remove the entire assembled object. Name precisely the selected
component if the rest of that object lies outside the target. Never enlarge
the main target description to bypass the separate co-removal decision.

Separately reason about each relevant contacting or dependent entity:
KEEP: an independently meaningful entity, another living being, a neighbor,
an occluder, or a stable supporting/background surface. Being held, supported,
or overlapped does not transfer an independent entity's identity to the target.
REMOVE_TOGETHER: a non-living accessory used exclusively by this target that
would otherwise be stranded, floating, or visibly incomplete after removal;
also a target-specific visual trace. Do not remove shared equipment, nearby
items merely by proximity, or anything belonging to another instance.
If a kept entity is partly hidden by the target, reconstruct only the revealed
missing portion and its plausible contact/support; keep its identity and
visible appearance. Do not delete that entity to simplify the edit.
If the target is itself an accessory or local part, retain its owner and all
other equipment; ownership is directional, not permission to delete the owner.
Judge the actual scene rather than naming stock objects or inventing contents.
For each co-removed accessory, give one entry and one point per visibly distinct
physical instance. Do not use one point for a plural set. Do not assume unseen
equipment exists from the activity alone. The point must lie on visible pixels
of that exact item, not on its owner or on where an occluded item might be.
Before answering, check that every entity named in the instruction is either
the selected target itself or separately declared remove_together. If retaining
a surface would strand something on it, reason about that dependent item too.
If the target is a component, do not remove independent objects attached to the
unselected remainder. Keep training instructions free of reconstruction and
preservation clauses; put that information in relations and reconstruction.

Return ONLY JSON:
decision: accept or defer;
target: concise unambiguous photo-relative description of selected pixels;
instruction: concise English removal command, 8-28 words, including any
necessary co-removed accessory but no long preservation/reconstruction clauses;
relations: at most 5 objects with description, action (keep or remove_together),
reason (one short sentence about ownership/physical dependency), point (x,y
integers 0-1000 in the FULL clean photo, strictly inside that entity's visible
pixels, not on the TARGET annotation), and segmentation_query (short factual
visible entity name, not a sentence or an invented replacement);
reconstruction: at most 35 words describing exposed surfaces or missing parts
of kept entities to reconstruct; do not invent unrelated objects.
The original mask is immutable. Co-removal outside it needs a separate precise
execution region; this stage must not pretend those pixels are already selected.
Use defer if the visual ownership or target identity is ambiguous.
'''


PROMPT_V4 = '''Plan one removal from IMAGE 1 (clean full photo) and IMAGE 2
(context crop, exact black/white target boundary, annotation dots). Identify the
selected instance using original reference AND outlined pixels. An annotation
color is never an object's color. A shared reference names several candidates,
not permission to remove all of them. Use photo-relative position/appearance
to distinguish this particular instance. Image borders may truncate its body.
Do not rename the target to a nearby accessory, enlarge a component into its
owner, or include another independent person/animal just because masks overlap.
If the reference, visible object and original mask materially conflict, defer.

Separate the DATASET INSTRUCTION from INTERNAL EXECUTION:
instruction is a short natural removal command naming only the main target,
with enough reference to distinguish it. No accessory inventory, KEEP clause,
background instructions or restoration detail. Common-sense associated removal
belongs ONLY to relations, not to the training instruction.

Internally inspect contact, ownership, occlusion and support. KEEP independent
entities and other living beings; contact/holding is not ownership. Co-remove
only nonliving items exclusive to the target that removal would leave stranded
or incomplete. Shared equipment and items belonging to others remain. If the
target is a component, do not remove its owner. Do not list the primary target
again as its own accessory. Do not invent invisible equipment. List EACH visible
accessory instance separately, each with its own point, never a plural set.
Items already covered by the primary mask need not be redundantly listed unless
their identity needs an explicit co-removal decision.
For a retained entity, check whether it loses its sole support: preserving its
pixels must not leave it floating. Do not silently move independent objects or
co-delete them to hide this conflict; defer an infeasible removal. Reconstruct
only newly revealed portions, keeping the retained entity's existing shape and
identity; do not invent extra limbs or detached parts. Continue visible structural
edges into the exposed area, not just background texture. Do NOT reconstruct
the removed target, its shadow, its reflection or its accessories.

Return ONLY JSON with:
decision: accept or defer;
reason: one short sentence explaining feasibility or the precise conflict;
target: concise selected-instance description in the FULL photo;
target_point: [x,y] on visible original TARGET pixels in IMAGE 1, integers 0-1000;
instruction: one short English removal sentence naming the main target only;
relations: at most 6 relevant entities, each with description, action (keep or
remove_together), reason, point [x,y], bbox [left,top,right,bottom] enclosing that
entity's VISIBLE extent, and segmentation_query (short factual entity name).
All points/boxes are 0-1000 in the FULL photo, not the magnified crop. Points
must lie ON each visible entity, not its owner, a gap, or its hidden continuation.
reconstruction: at most 35 words of internal scene-specific completion guidance.
Original mask is immutable. External co-removal needs separately resolved
geometry; do not pretend those pixels were originally selected. Keep final
instruction simple even when the internal plan needs multiple relations.
'''

# V5 makes the post-removal physical state an explicit planning object.  The
# exported instruction remains short; this policy only changes the internal
# relation graph used to build the writable execution region.
PROMPT_V5 = PROMPT_V4.replace(
    "Internally inspect contact, ownership, occlusion and support. KEEP independent\n",
    "First simulate the scene immediately after the target pixels disappear. Identify the\n"
    "target's exclusive attachments and every object whose sole physical support is\n"
    "the target. For a non-living dependent item that would float, be left held by\n"
    "nothing, or leave an impossible cut-through, mark it remove_together so its\n"
    "visible pixels are added to the separate execution region. If the dependent item\n"
    "is a person, animal, shared object, or an item that can plausibly remain supported\n"
    "by visible structure, keep it. If keeping an independent object would be physically\n"
    "impossible and deleting it would violate its identity, defer rather than invent a\n"
    "new support. Do not treat mere overlap, proximity, or a common table as ownership.\n"
    "Internally inspect contact, ownership, occlusion and support. KEEP independent\n")
PROMPT_V5 += (
    "\nBefore accept, perform a second pass over the proposed execution region: every\n"
    "remove_together entity must have visible pixels and a defensible ownership or\n"
    "sole-support reason; every kept entity must have a plausible post-removal support.\n"
    "Never keep a cup, box, diffuser, board, or held item floating merely because the\n"
    "dataset mask did not cover it. Never remove a neighboring living instance to make\n"
    "the geometry easier. If the mask selects a support surface carrying several\n"
    "independent objects and no coherent selective outcome is possible, defer.\n"
)

PROMPT_V6 = PROMPT_V5 + '''

Additional support-surface rule: if the selected target is a table, shelf,
counter, platform, tray, floor section, or other horizontal support being
removed, inspect every non-living object visibly resting on the portion being
removed. Unless that object has another visible support that remains, mark it
remove_together and give its own point. This includes cups, bottles, boxes,
food containers and their contents when they would otherwise float. Do not
call such an object independent merely because it has its own identity. A
person, animal, or a handheld item visibly supported by a retained person is
not automatically removed; defer if the requested surface removal has no
physically plausible outcome without deleting a living entity. A support
surface may be kept only when it is outside the removed footprint or its
remaining geometry visibly supports the item.
'''

# V7 is intentionally short.  It keeps the relation graph and the physical
# feasibility check, but removes the long lists of object examples and repeated
# preservation clauses used by V5/V6.
PROMPT_V7 = '''Plan one localized removal from two images: IMAGE 1 is the clean full
photograph; IMAGE 2 is its context crop with a black/white target outline. The
outline and labels are annotations, never source colors. Identify the exact
instance from the visible pixels, reference and full-photo position. Do not
rename a component as its owner or include a neighboring instance.

Return a short dataset instruction naming only the selected target. Separately
build an internal relation graph for the execution mask. For each relevant
entity choose one action: keep, or remove_together. Use remove_together only
for a visible nonliving item exclusively attached to the target, or for an item
whose only visible support is the portion being removed and which would
otherwise float. Keep independent objects, shared items, people and animals.
If an independent item would lose its only support and no coherent outcome is
possible, defer instead of inventing support or deleting a living entity.
Check this post-removal physical state before accepting. Do not infer unseen
attachments from the activity. Each visible entity gets its own point and bbox;
never use one point for a plural group. Points lie on its visible pixels in the
full photograph. The original mask is immutable; external remove_together
pixels are resolved later into a separate execution region.

Return ONLY JSON:
{"decision":"accept|defer", "reason":"one sentence", "target":"short
photo-relative target", "target_point":[x,y], "instruction":"one short
English removal sentence naming the main target only", "relations":[{"description":
"...", "action":"keep|remove_together", "reason":"...", "point":[x,y],
"bbox":[l,t,r,b], "segmentation_query":"short visible name"}],
"reconstruction":"short scene-specific completion guidance"}.
All coordinates are 0-1000 in IMAGE 1. Keep the instruction concise and do not
write reconstruction or preservation clauses into it.
'''


PROMPT_V8 = '''Plan removal from the clean full photo and an outlined context crop.
Annotations are not scene content. Match the reference to the outlined instance.
The SELECTED target must disappear, even when it is a person or animal; only
OTHER people and animals must stay. Name a selected component, not its owner.

First trace the target's complete visible body and contact points. Assign each
contacting item to its actual owner before listing background neighbors. Include
visible target fragments outside the original mask and exclusive held/worn items
as remove_together. Keep independently supported items; proximity or overlap is
not ownership. Do not assume a nearby surface supports an item without visible
contact and a plausible stable pose. Do not invent hidden attachments.

Mentally remove the target and these dependents. If another person/animal would
lose its support in its current pose, defer: do not move it, delete it, invent a
support, or regenerate the removed target. Otherwise describe only newly exposed
background or portions of kept objects, never replacement copies of the target.
List each co-removed item separately with a point on its own visible pixels.
The original mask stays unchanged; extra geometry is resolved separately.

Return ONLY JSON:
{"decision":"accept|defer", "reason":"one sentence", "support_check":"coherent|conflict",
"target":"unambiguous full-photo target", "target_point":[x,y],
"instruction":"short removal sentence naming only the main target",
"relations":[{"description":"...", "action":"keep|remove_together", "reason":"ownership/support",
"point":[x,y], "bbox":[l,t,r,b], "segmentation_query":"short visible name"}],
"reconstruction":"short completion guidance"}.
Coordinates are integers 0-1000 in the FULL photo. At most six relations;
prioritize co-removal obligations. No preservation clauses in the instruction.
'''


PROMPT_V9 = PROMPT_V8.replace(
    'First trace the target\'s complete visible body and contact points.',
    'First ask what OTHER living entities the target is supporting, NOT what supports the target itself. '
    'For each such dependent, identify the visible support remaining after deletion; being an independent '
    'person does not supply physical support. Then trace the target\'s complete visible body and contact points.'
).replace(
    'Include\nvisible target fragments outside the original mask and exclusive held/worn items\nas remove_together.',
    'Include visible target fragments and exclusive held/worn items extending OUTSIDE the original mask '
    'as remove_together. Do not inventory parts or clothing already fully covered by the target outline; '
    'they are removed by the main mask.'
).replace(
    '"support_check":"coherent|conflict",',
    '"support_check":{"target_supports_other_living":true|false,"other_living_still_supported":true|false},'
)


PROMPT_V10 = '''Plan removal using the clean full photo and its outlined context crop.
Marks are annotations, not scene content. Identify ALL objects or parts selected
by the outline, including separate fragments. If several instances are selected,
name the whole group, not just one member. A selected part is not its owner.

Trace visible contacts and ownership. Remove exclusive nonliving attachments
that would be stranded after deleting the target. Keep other living beings and
independently supported neighbors. Proximity is not ownership. List only
attachments extending outside the target outline; covered clothes/parts need
no extra entry. Do not invent held objects or hidden attachments.

Imagine the target absent. What OTHER living beings does it currently support?
Would they have visible, stable support in their unchanged pose afterwards?
If not, defer. Independence of identity is not physical support. Do not invent
new supports or silently remove another person/animal to make the edit possible.
For an accepted edit, complete exposed background rather than recreating the target.

Return only JSON:
{"decision":"accept|defer","reason":"short visible evidence",
"support_check":{"target_supports_other_living":true|false,"other_living_still_supported":true|false},
"target":"unambiguous full-photo target or group","target_point":[x,y],
"instruction":"short removal sentence naming the main target or complete selected group",
"relations":[{"description":"...","action":"keep|remove_together","reason":"visible ownership/support",
"point":[x,y],"bbox":[l,t,r,b],"segmentation_query":"short visible name"}],
"reconstruction":"short completion guidance"}.
Coordinates are integers 0-1000 in the FULL photo. Put target_point on one
selected instance. At most six relations; one per auxiliary instance, prioritizing
co-removal. The original mask is immutable; extra geometry is resolved separately.
No preservation or reconstruction clauses in the final instruction.
'''


PROMPT_V11 = '''Plan one removal from the clean full photo and outlined context crop.
Annotations are not scene content. The outline determines the selected object,
part or complete group. Identify every selected instance visually and distinguish
it from similar neighbors using concise full-photo location or appearance.
Source text is only a hint; never copy a multi-region expression into this target.

Describe the main target only in target. Put nonliving attachments extending
outside its outline in relations if they would be stranded by removal. Covered
clothes/parts need no separate entry. Keep independent neighbors and other living
beings. Proximity alone is not ownership. If another living being would lose
physical support in its unchanged pose, defer.

For reconstruction, give one short sentence describing the visible surrounding
background to continue into the removed area. Do not recreate the target.
Return only JSON:
{"decision":"accept|defer","reason":"short visible evidence",
"support_check":{"target_supports_other_living":true|false,"other_living_still_supported":true|false},
"target":"concise uniquely located main object, part or complete selected group",
"target_point":[x,y],
"relations":[{"description":"...","action":"keep|remove_together","reason":"visible ownership/support",
"point":[x,y],"bbox":[l,t,r,b],"segmentation_query":"short visible name"}],
"reconstruction":"one short background completion sentence"}.
Coordinates are integers 0-1000 in the FULL photo; target_point lies on a selected
instance. At most six relations. The public instruction is compiled as Remove
plus target; therefore exclude unselected accessories and preservation clauses
from target. The original mask stays fixed; external attachments are resolved separately.
'''


PROMPT_V12 = PROMPT_V11.replace(
    'beings. Proximity alone is not ownership. If another living being would lose\nphysical support in its unchanged pose, defer.',
    'beings and shared items with another visible support. Trace dependency in the\n'
    'correct direction: what OTHER living being is supported by the selected target?\n'
    'Would that being still have stable support in its unchanged pose afterwards?\n'
    'If not, defer. Do not invent support or delete that being. Proximity is not ownership.'
).replace(
    'background to continue into the removed area. Do not recreate the target.',
    'surfaces to continue at their respective depths across the removed area.\n'
    'Continue structural edges; do not fill ground with distant sky or recreate\n'
    'the selected object under a new appearance. If the selected scope cannot be\n'
    'described coherently without changing unselected structure, defer.'
).replace(
    '"target":"concise uniquely located main object, part or complete selected group",',
    '"target":"main object, part or complete group, ideally 5-16 words; only reliable identifying details",'
)


PROMPT_V13 = PROMPT_V12.replace(
    'outside its outline in relations if they would be stranded by removal. Covered',
    'outside its outline in relations if they would be stranded by removal. Only\n'
    'list clearly visible external pixels, never an attachment inferred from function. Covered'
).replace(
    'If not, defer. Do not invent support or delete that being. Proximity is not ownership.',
    'This conflict concerns unselected beings that must remain, not contents already\n'
    'fully included in the selected region. If a retained being loses support, defer.\n'
    'Do not invent support or enlarge the target. Proximity is not ownership.'
).replace(
    'the selected object under a new appearance.',
    'the selected object or its target-specific visual traces under a new appearance.'
)


# V14 keeps the relation reasoning unchanged and makes the auxiliary SAM
# contract explicit.  Relation prose may describe ownership, while grounding
# queries must remain short visual names rather than a brittle conjunction of
# every guessed attribute.
PROMPT_V14 = PROMPT_V13.replace(
    '"point":[x,y],"bbox":[l,t,r,b],"segmentation_query":"short visible name"}],',
    '"point":[x,y],"bbox":[l,t,r,b],"segmentation_queries":["2-3 short visible noun phrases"]}],'
) + '''
For each relation, segmentation_queries contains two or three alternative
names for the SAME visible entity. Use common visual nouns; include material or
color only when unmistakable. Do not join uncertain categories with a slash or
combine several separate objects into one query. These alternatives are only
for grounding and must not make the public instruction longer.
The target names only the selected main pixels. External entities belong only
in relations and never appear in target, even as relational identifying phrases.
A visually distinct neighbor or occluder is not part of target merely because
the contour touches or wraps around it. If this changes the semantic scope and
the source hint plus visible object continuity cannot resolve it, defer.
'''


PROMPT_V15 = '''Plan a localized removal from IMAGE 1 (clean full photo) and IMAGE 2
(context crop with the exact target outline). Marks are annotations, not objects.
Name every selected instance or part concisely and unambiguously; do not promote
a selected part to its owner. The public instruction names only this main target.

Inspect contacts, occlusions and openings. Explicitly list nearby independent
entities that touch, overlap, or appear through the target as keep. Their visible
parts must survive; newly exposed portions should connect at the correct depth,
not be painted over as background. Do not inventory distant unrelated objects.
List visible exclusive attachments or target fragments as remove_together when
they extend beyond the actual mask, including its excluded holes. Being inside
the outer bounding rectangle does NOT mean being selected. Do not infer unseen
equipment. Keep shared items with remaining support and other living beings.
If a retained living being loses its only support, defer rather than delete or
relocate it. Judge what the target supports, not what supports the target.

Use one short reconstruction sentence describing the newly exposed surfaces
and retained-object continuity. Never recreate the removed target. Each relation
gets its own visible point and tight box in FULL-photo coordinates (0-1000).
Return only JSON:
{"decision":"accept|defer","reason":"short visible evidence",
"support_check":{"target_supports_other_living":true|false,"other_living_still_supported":true|false},
"target":"short uniquely located main target or complete selected group",
"target_point":[x,y],"relations":[{"description":"concise located entity",
"action":"keep|remove_together","reason":"visible ownership or contact",
"point":[x,y],"bbox":[l,t,r,b],"segmentation_query":"short visible noun phrase"}],
"reconstruction":"one short completion sentence"}.
At most six relations. Keep points are on the retained entity outside the mask;
remove_together points identify its visible external part. The source mask is
trusted and immutable; these relations specify additional execution geometry.
'''


PROMPT_V16 = PROMPT_V15.replace(
    '"support_check":{"target_supports_other_living":true|false,"other_living_still_supported":true|false},',
    '"support_check":"coherent|conflict",'
).replace(
    'entities that touch, overlap, or appear through the target as keep.',
    'objects or living beings that touch, overlap, or appear through the target as keep. '
    'Use one entity per relation, not a mixture of people and background; '
    'generic fill surfaces belong in reconstruction, not keep.'
).replace(
    'Keep shared items with remaining support and other living beings.',
    'Keep other living beings and shared items with stable remaining support. '
    'Contact with the ground alone does not prove an attached or held item can '
    'stay in its current pose after its owner disappears.'
).replace(
    'The public instruction names only this main target.',
    'The public instruction names only this main target, without enumerating '
    'equipment or unselected neighboring objects.'
)


def parse_relation_response(raw, policy='legacy'):
    # Never accept a provisional JSON inside an incomplete reasoning block.
    if '<think>' in raw and '</think>' not in raw:
        return None
    value=parse_json_object(raw.rsplit('</think>',1)[-1])
    if policy == 'relations-v14' and isinstance(value,dict):
        for relation in value.get('relations',[]):
            queries=relation.get('segmentation_queries')
            if isinstance(queries,list) and queries and isinstance(queries[0],str):
                # Retain the singular key for older reports and editor helpers.
                relation['segmentation_query']=queries[0]
    if policy in {'relations-v11','relations-v12','relations-v13','relations-v14','relations-v15','relations-v16'} and isinstance(value,dict) and isinstance(value.get('target'),str):
        value['instruction']='Remove '+value['target'].strip().rstrip('.')+'.'
    return value


def validate_relation_plan(value, policy='legacy', mask=None):
    if not isinstance(value,dict) or value.get('decision') not in {'accept','defer'}:
        return 'invalid_decision'
    if value['decision']=='defer': return 'defer'
    if policy in {'relations-v9','relations-v10','relations-v11','relations-v12','relations-v13','relations-v14','relations-v15'}:
        support=value.get('support_check')
        if not isinstance(support,dict) or any(type(support.get(k)) is not bool for k in
                ['target_supports_other_living','other_living_still_supported']):
            return 'invalid_support_check'
        if support['target_supports_other_living'] and not support['other_living_still_supported']:
            return 'defer_support_conflict'
    if policy in {'relations-v8','relations-v16'}:
        if value.get('support_check') not in {'coherent', 'conflict'}:
            return 'invalid_support_check'
        if value['support_check'] == 'conflict':
            return 'defer_support_conflict'
    for key,limit in [('target',32),('instruction',32),('reconstruction',45)]:
        text=value.get(key)
        if not isinstance(text,str) or not 1<=len(text.split())<=limit:return 'invalid_'+key
    relations=value.get('relations')
    if not isinstance(relations,list) or len(relations)>(6 if is_grounded_relation_policy(policy) else 5):return 'invalid_relations'
    for relation in relations:
        if not isinstance(relation,dict):return 'invalid_relation'
        if relation.get('action') not in {'keep','remove_together'}:return 'invalid_action'
        for key in ['description','reason','segmentation_query']:
            if not isinstance(relation.get(key),str) or not relation[key].strip():return 'invalid_'+key
        if policy == 'relations-v14':
            queries=relation.get('segmentation_queries')
            if (not isinstance(queries,list) or not 2<=len(queries)<=3 or
                any(not isinstance(query,str) or not query.strip() or len(query.split())>6
                    for query in queries)):
                return 'invalid_segmentation_queries'
        point=relation.get('point')
        if not isinstance(point,list) or len(point)!=2 or any(type(x)!=int or not 0<=x<=1000 for x in point):return 'invalid_point'
        if is_grounded_relation_policy(policy):
            box=relation.get('bbox')
            if (not isinstance(box,list) or len(box)!=4 or
                any(type(x)!=int or not 0<=x<=1000 for x in box) or
                not box[0]<=point[0]<=box[2] or not box[1]<=point[1]<=box[3] or
                box[0]>=box[2] or box[1]>=box[3]):return 'invalid_relation_bbox'
    if is_grounded_relation_policy(policy):
        point=value.get('target_point')
        if not isinstance(point,list) or len(point)!=2 or any(type(x)!=int or not 0<=x<=1000 for x in point):return 'invalid_target_point'
        if mask is not None:
            import cv2
            h,w=mask.shape;x=min(w-1,round(point[0]*w/1000));y=min(h-1,round(point[1]*h/1000))
            distance=cv2.distanceTransform((~mask.astype(bool)).astype('uint8'),cv2.DIST_L2,5)
            if distance[y,x]>max(3,(w*w+h*h)**.5*.005):return 'defer_target_point_outside_mask'
    return 'accepted'


def point_repair_feedback(mask, previous):
    """Provide real interior anchors, never snap an incorrect entity silently."""
    import cv2
    import numpy as np
    binary=mask.astype('uint8')
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    distance=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    h,w=mask.shape;points=[]
    for index in sorted(range(1,count),key=lambda i:stats[i,cv2.CC_STAT_AREA],reverse=True)[:3]:
        y,x=np.unravel_index(np.where(labels==index,distance,-1).argmax(),mask.shape)
        points.append([round(x*1000/w),round(y*1000/h)])
    return ('\nYour previous draft was not executed because its main target_point is outside '
        'the original mask. This may be a coordinate error OR the wrong instance. '
        'Reinspect the same two images and reference, do not just rubber-stamp the draft. '
        'The following measured FULL-photo points really are inside selected mask fragments: '
        +json.dumps(points)+'. Identify the object containing these points, then return the '
        'entire corrected JSON. If a correct target cannot be established, defer with a reason. '
        'Do not move the point onto the mask while retaining a mismatched object description. '
        'Prior draft (untrusted): '+json.dumps(previous,ensure_ascii=False))


def reusable_plan(saved, row, mask, policy):
    """Adopt validated legacy plans; retry malformed responses, not valid defers."""
    if not saved or any(saved.get(k) != v for k,v in row.items()):return False
    if saved.get('relation_policy') != policy:return False
    try:
        value=parse_relation_response(saved.get('relation_raw_response',''),policy)
        status=validate_relation_plan(value,policy,mask)
        return (value==saved.get('relation_plan') and status==saved.get('relation_status')
                and status in {'accepted','defer','defer_support_conflict'})
    except (TypeError,ValueError,KeyError,IndexError,AttributeError):
        return False


def planner_pending(rows, data_root, out_root, policy, resume, checkpoints, dependencies):
    old={r['image']:r for r in recover_rows(out_root/'annotations.jsonl')} if resume else {}
    results=[];pending=[];empty=0;reused=0
    for row in rows:
        with Image.open(data_root/'sources'/row['source_image']) as source:
            mask=mask_array(source.size,row['mask'])
        if not mask.any():
            empty+=1
            results.append({**row,'relation_plan':None,'relation_status':'invalid_input_empty_mask',
                'relation_raw_response':'','relation_policy':policy,
                'relation_input_error':'Source mask has zero foreground pixels; no model call or edit performed.'})
            print(json.dumps(dict(image=row['image'],status='invalid_input_empty_mask')),flush=True)
        else:
            saved=checkpoints.load(row,dependencies[row['image']]) if resume else None
            # Only records from before checkpoint support may be adopted from JSONL.
            # A mismatching existing checkpoint means the source/dependency changed.
            if resume and not (checkpoints.root/(row['image']+'.json')).exists():
                saved=old.get(row['image'])
            if reusable_plan(saved,row,mask,policy):
                results.append(saved);reused+=1
                checkpoints.save(row,saved,dependencies=dependencies[row['image']])
            else:pending.append(row)
    return results,pending,reused,empty


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--ids',default='');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--outside-pointers',action='store_true')
    p.add_argument('--thinking',action='store_true',help='Opt-in low-effort reasoning ablation; does not add model calls')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--policy',choices=['legacy','relations-v4','relations-v5','relations-v6','relations-v7','relations-v8','relations-v9','relations-v10','relations-v11','relations-v12','relations-v13','relations-v14','relations-v15','relations-v16'],default='legacy')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=a.resume)
    rows=load_jsonl(a.data_root/'annotations.jsonl')
    if a.ids:
        ids={int(x) for x in a.ids.split(',')};rows=[r for r in rows if int(r['image'].split('_')[0]) in ids]
        if len(rows)!=len(ids):raise ValueError('Missing IDs')
    if any(r['task_type']!='remove' for r in rows):raise ValueError('Removal-only pilot')
    settings=dict(rows=fingerprint(rows),policy=a.policy,thinking=a.thinking,
        outside_pointers=a.outside_pointers,batch_size=a.batch_size)
    bind_settings(a.out_root,settings,a.resume);checkpoints=CaseCheckpoints(a.out_root,settings)
    ensure_sources(a.out_root,a.data_root/'sources');(a.out_root/'inputs').mkdir(exist_ok=True)
    hashes={name:file_digest(a.data_root/'sources'/name) for name in {r['source_image'] for r in rows}}
    dependencies={r['image']:dict(source_sha256=hashes[r['source_image']]) for r in rows}
    results,pending,reused,empty=planner_pending(rows,a.data_root,a.out_root,a.policy,a.resume,checkpoints,dependencies)
    write_jsonl(a.out_root/'annotations.jsonl',results)
    print(json.dumps(dict(stage='planning',total=len(rows),reused=reused,empty_masks=empty,pending=len(pending))),flush=True)
    if not pending:
        (a.out_root/'summary.json').write_text(json.dumps(dict(cases=len(results),reused=reused,
            empty_masks=empty,thinking=a.thinking,calls=0,repair_calls=0,load_seconds=0,wall_seconds=0),indent=2))
        return
    from utils.runtime_paths import qwen38_model
    start=time.perf_counter();vlm.configure_backend('qwen38-vllm',model_id=qwen38_model(),device='cuda:0',dtype='bf16')
    backend=vlm.get_backend();loaded=time.perf_counter();repair_calls=0;calls=0
    if a.thinking:
        backend.enable_thinking=True
        backend.chat_template_overrides={'reasoning_effort':'low'}
        backend.sampling_overrides={'temperature':1.,'top_p':.95,'top_k':20,'presence_penalty':0.,'repetition_penalty':1.}
    try:
        for offset in range(0,len(pending),a.batch_size):
            batch=pending[offset:offset+a.batch_size];messages=[];masks=[];prompts=[]
            for row in batch:
                source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
                mask=mask_array(source.size,row['mask'])
                guard=protected_neighbors(a.data_root,row,source.size) if a.outside_pointers else None
                context_bbox=relation_context_bbox(mask) if is_grounded_relation_policy(a.policy) else None
                crop=instruction_target_crop(source,mask,target_pointer=True,excluded_mask=guard,context_bbox=context_bbox)
                crop.save(a.out_root/'inputs'/row['image'])
                prompt=(PROMPT_V14 if a.policy=='relations-v14' else PROMPT_V13 if a.policy=='relations-v13' else PROMPT_V12 if a.policy=='relations-v12' else PROMPT_V11 if a.policy=='relations-v11' else PROMPT_V10 if a.policy=='relations-v10' else PROMPT_V9 if a.policy=='relations-v9' else PROMPT_V8 if a.policy=='relations-v8' else PROMPT_V7 if a.policy=='relations-v7' else PROMPT_V6 if a.policy=='relations-v6' else PROMPT_V5 if a.policy=='relations-v5' else PROMPT_V4 if a.policy=='relations-v4' else PROMPT)+ ('\nOrange OUTSIDE dots mark separately annotated pixels outside the original target. Identify their actual visible owner, never include that owner in the selected target merely because it overlaps or is held. Only separately declared exclusive equipment may be co-removed; an independent living being must be retained.\n' if a.outside_pointers else '')
                if a.policy=='relations-v15':prompt=PROMPT_V15
                if a.policy=='relations-v16':prompt=PROMPT_V16
                if is_grounded_relation_policy(a.policy):prompt+=relation_input_evidence(row,mask,context_bbox,visual_binding=a.policy in {'relations-v11','relations-v12','relations-v13','relations-v14','relations-v15','relations-v16'})
                (a.out_root/'inputs'/Path(row['image']).with_suffix('.txt')).write_text(prompt)
                masks.append(mask);prompts.append(prompt)
                messages.append([{'role':'user','content':[{'type':'image','image':source},{'type':'image','image':crop},{'type':'text','text':prompt}]}])
            outputs=backend.chat_batch(messages,max_new_tokens=4096 if a.thinking else 1280 if a.policy in {'relations-v7','relations-v8','relations-v9'} else 1536 if a.policy in {'relations-v5','relations-v6'} else 1408 if a.policy=='relations-v4' else 768)
            if len(outputs)!=len(batch):raise ValueError('Incomplete VLM batch')
            calls+=len(outputs)
            for row,raw,mask,prompt,message in zip(batch,outputs,masks,prompts,messages):
                value=parse_relation_response(raw,a.policy);status=validate_relation_plan(value,a.policy,mask)
                history=[]
                if is_grounded_relation_policy(a.policy) and status=='defer_target_point_outside_mask':
                    history.append(dict(status=status,raw=raw,prompt=prompt))
                    prompt+=point_repair_feedback(mask,value)
                    message[0]['content'][-1]['text']=prompt
                    raw=backend.chat_batch([message],max_new_tokens=4096 if a.thinking else 1280 if a.policy in {'relations-v7','relations-v8','relations-v9'} else 1536 if a.policy in {'relations-v5','relations-v6'} else 1408)[0];repair_calls+=1
                    value=parse_relation_response(raw,a.policy);status=validate_relation_plan(value,a.policy,mask)
                    (a.out_root/'inputs'/Path(row['image']).with_suffix('.txt')).write_text(prompt)
                results.append({**row,'relation_plan':value,'relation_status':status,'relation_raw_response':raw,'relation_policy':a.policy,'relation_planning_prompt':prompt,'relation_repair_history':history})
                write_jsonl(a.out_root/'annotations.jsonl',results)
                if status in {'accepted','defer','defer_support_conflict'}:
                    checkpoints.save(row,results[-1],dependencies=dependencies[row['image']])
                print(json.dumps(dict(image=row['image'],status=status,plan=value)),flush=True)
            print(f'planned {len(results)}/{len(rows)} (reused={reused}, empty={empty})',flush=True)
    finally:
        vlm.shutdown_backend()
    (a.out_root/'prompt.txt').write_text(prompt)
    order={r['image']:i for i,r in enumerate(rows)};results.sort(key=lambda r:order[r['image']])
    write_jsonl(a.out_root/'annotations.jsonl',results)
    (a.out_root/'summary.json').write_text(json.dumps(dict(cases=len(results),reused=reused,empty_masks=empty,thinking=a.thinking,calls=calls+repair_calls,repair_calls=repair_calls,load_seconds=loaded-start,wall_seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
