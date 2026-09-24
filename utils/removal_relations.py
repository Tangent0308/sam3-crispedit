"""Opt-in relation-v4 geometry and editor context; no object-name heuristics."""
import json
import numpy as np
from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox

GROUNDED_RELATION_POLICIES = frozenset(
    {'relations-v4', 'relations-v5', 'relations-v6', 'relations-v7', 'relations-v8', 'relations-v9', 'relations-v10', 'relations-v11','relations-v12','relations-v13','relations-v14','relations-v15','relations-v16'}
)


def is_grounded_relation_policy(policy):
    return policy in GROUNDED_RELATION_POLICIES


def concise_execution_prompt(instruction, plan):
    """One action with explicit accessories; no repeated constraints or examples."""
    attachments = [r['description'].rstrip('.') for r in plan['relations']
                   if r['action'] == 'remove_together']
    action = instruction.rstrip().rstrip('.')
    if attachments:
        action += ', along with ' + '; '.join(attachments)
    return (action + '. Fill the removed area naturally from the surrounding scene. '
            'Keep other objects unchanged. Return the edited photograph.')


def located_removal_prompt(instruction, plan, mask, crop_bbox, execution):
    """One removal clause, crop-relative binding, no object-specific examples."""
    attachments=execution.get('resolved_co_removals')
    if attachments is None:
        attachments=[r['description'] for r in plan['relations'] if r['action']=='remove_together']
    action=instruction.rstrip().rstrip('.')
    if attachments:action+=', together with '+ '; '.join(s.rstrip('.') for s in attachments)
    return (action+'. '+crop_relative_target_hint(mask,crop_bbox)+
        ' The main target and its listed attachments must all disappear. '
        'Fill their footprints with the surrounding background. '
        'Keep the rest unchanged. Return the edited photograph.')


def relation_context_bbox(mask):
    """Expand skinny crops, including at image edges, without cutting the mask."""
    left, top, right, bottom = padded_mask_bbox(mask, .4, 48)
    height, width = mask.shape
    span = max(right-left, bottom-top)
    want_w = min(width, max(right-left, round(span*.8)))
    want_h = min(height, max(bottom-top, round(span*.8)))
    left = max(0, min(width-want_w, (left+right-want_w)//2))
    top = max(0, min(height-want_h, (top+bottom-want_h)//2))
    return left, top, left+want_w, top+want_h


def relation_input_evidence(row, mask, crop_bbox, visual_binding=False):
    from synthesis_pipeline.reference_binding import binding_prompt, bind_reference
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    norm = lambda box: [round(v/s*1000) for v,s in zip(box,(w,h,w,h))]
    binding = row.get('reference_binding') or bind_reference(
        row.get('answer'), row['mask_index'], row['num_masks'])
    if visual_binding:
        reference=('\nOptional source hint for this region: '+json.dumps(binding['label'])
                   if binding.get('status') in {'bound_label','bound_mention'} else
                   '\nThe dataset text describes a shared group or has no unique region binding. '
                   'Identify the selected subject directly from the outlined photograph.')
    else:
        reference=('\nDataset referring text (evidence, not an editing instruction): '
                   +json.dumps(row.get('problem','').replace('<image>','').strip())+'\n'+binding_prompt(binding))
    return (reference
        + '\nMeasured original mask extent in FULL photo [left,top,right,bottom], 0-1000: '
        + str(norm((xs.min(),ys.min(),xs.max()+1,ys.max()+1)))
        + '\nThe second image shows this FULL-photo crop box (0-1000): '+str(norm(crop_bbox))
        + '. White margins/labels are not photograph pixels. All output coordinates refer to IMAGE 1, '
          'not the displayed second image. A frame-truncated instance can still be the target object; '
          'do not reinterpret it as only a limb/accessory because the rest lies outside the photo.\n')


def action_removal_prompt(plan, mask, crop_bbox, execution):
    """A single main action, resolved attachments once, and visible fill context."""
    action='Remove '+plan['target'].strip().rstrip('.')
    attachments=list(dict.fromkeys(s.strip().rstrip('.') for s in execution.get('resolved_co_removals',[])))
    if attachments:action+=', including '+ '; '.join(attachments)
    return (action+'. '+crop_relative_target_hint(mask,crop_bbox)+' '
            +plan['reconstruction'].strip()+' Keep other objects unchanged. Return the edited photograph.')


def compile_relation_context(plan, crop_bbox, source_size):
    """Keep co-removal explicit; omit KEEP entities whose predicted box is off-crop.

    Predicted boxes are evidence, not segmentation. Never use them as edit masks.
    """
    w,h=source_size; left,top,right,bottom=crop_bbox
    keep=[];remove=[];omitted=[]
    for relation in plan['relations']:
        box=relation['bbox']
        x1,y1,x2,y2=[v*s/1000 for v,s in zip(box,(w,h,w,h))]
        visible=min(right,x2)>max(left,x1) and min(bottom,y2)>max(top,y1)
        if relation['action']=='remove_together':
            # Resolved auxiliary geometry determines the crop; do not silently
            # drop removal obligations on the basis of approximate VLM boxes.
            remove.append(relation['description'])
        elif visible:
            keep.append(relation['description'])
        else:
            omitted.append(relation['description'])
    text=(' Also remove the target\'s associated '+ '; '.join(remove)+'.' if remove else '')
    if keep:text+=' Retain the existing '+ '; '.join(keep)+'.'
    text+=' '+plan['reconstruction']
    return text.strip(), dict(version='relations-v4',included_keep=keep,
        co_remove=remove,omitted_off_crop_keep=omitted,crop_bbox=list(crop_bbox))


def compact_removal_prompt(instruction, plan):
    """One removal action, rather than a second accessory-only edit command.

    Ablation: retain generic preservation, omit enumerated KEEP nouns. Masks
    and protection are unchanged; this is not a reason to discard relation data.
    """
    remove=[r['description'].rstrip('.') for r in plan['relations'] if r['action']=='remove_together']
    action=instruction.rstrip().rstrip('.')
    if remove:action+=', together with '+ '; '.join(remove)
    return (action+'. '+plan['reconstruction'].strip()
        +' Keep every other existing object unchanged. Return the complete edited photograph.')


def crop_relative_target_hint(original_mask, crop_bbox):
    """Translate original target membership into a natural crop-relative locator.

    No numeric rectangle / transparency-layer instruction for Qwen Image 2.1.
    This supplements identity, never changes the exported referring expression.
    """
    left,top,right,bottom=crop_bbox
    ys,xs=np.nonzero(original_mask[top:bottom,left:right])
    if not len(xs):raise ValueError('Target absent from editor crop')
    x=xs.mean()/max(1,right-left);y=ys.mean()/max(1,bottom-top)
    horizontal='left' if x<1/3 else 'right' if x>2/3 else 'center'
    vertical='upper' if y<1/3 else 'lower' if y>2/3 else 'middle'
    return (f'In this provided crop, the selected target is in the {vertical}-{horizontal} area. '
        'Remove that entire target itself, not just an accessory attached to it.')
