"""Three-image audit with identically positioned source/result detail rows.

Image 1 preserves scene-wide reference evidence. Images 2/3 use the SAME crop
boxes, scale and row order; the source outline never appears on the result.
No new inference call and no synthetic target pixels are introduced.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import binary_dilation

from .fonts import load_font
from .geometry import load_mask
from .production_state import atomic_json

VERSION = 'aligned-source-result-15-source-advisory-baseline-prompt'
PROMPT = """Image 1: clean full SOURCE. Image 2, each U row: outlined source context LEFT, selected source pixels with hatched EXCLUDED areas RIGHT. Image 3, matching row: clean BEFORE LEFT and AFTER RIGHT, same scale. Hatched areas and holes are excluded. Twin rows may be separate outputs; compare each row only with its own source.
Identify the selected pixels in the full source BEFORE reading the desired outcome as evidence. Record source_state and result_state as visible facts, each within 20 words. Inspect the whole selected extent and surrounding contact. A plausible-looking result is not enough: check whether the old content actually remains and whether the requested content really appeared.
Report observed_operation independently: remove means the selected content is gone with coherent revealed surface; replace requires new physical form/design, not only color/material; attribute changes the requested property of the same entity; add inserts new content while retaining its host. Use none for no meaningful edit, mixed for unintended edits, unclear when not verifiable. Code compares this to the requested operation.
image_ok requires the ORIGINAL request to be completed naturally, without substantial remnants, broken anatomy/support, wrong contact, or damage to unselected content. Tolerate minor edge/texture differences and source blur. Earlier edits must survive in a serial component's final result. For clothing/body edits, check the underlying body and silhouette: exposed background through a limb or torso fails. New content over a still-present original is add, not replace. Judge new-item scale against its physical host, not the crop size.
scope_status compares the SOURCE mask with the requested selector: for remove/replace/attribute it is draft_ref; for add it must cover the actual attachment/support site, not merely one changed fragment or the entire larger host. With placement.site, draft_ref names the host. A selected part/surface is not its whole host. Inspect visible excluded attached structures before agreeing with a whole-object Ref; a recognizable silhouette alone does not establish scope. Do not rename a wrong entity or change the goal to excuse a failed edit.
Check the FINAL combined image for old-color background islands beside modified surfaces, exposed holes matching the current surroundings, and missing physical supports. Check complete requested lettering, including unintended duplicates. Attribute each defect to its own U: another requested edit is not collateral damage. Original trim, seams and occluders are not removal remnants unless they belong to the requested target.
Test reference_unique using ONLY Image 1 and the complete Ref: could a reader select another visible instance without contradicting any word? A mask outline or U label does not make the WORDING unique. If another instance fits, repair with the shortest verified distinction or set reference_unique=false. Keep a correct draft_ref EXACTLY, with empty ref_issue/final_ref; do not shorten for style. Otherwise ref_issue states the defect and distinguishing cue, and final_ref names the SAME entity and extent. Preserve necessary host/part/count/position words. Judge repaired wording when supplied. A wrong entity or mask extent cannot be repaired by renaming it.
Keep operation, payload, attributes and attachment site fixed. For add, compare the host's distinctive parts BEFORE/AFTER before claiming it vanished or was replaced. With placement.site, Ref names its host without repeating the site. reference_unique judges final_ref when nonempty, otherwise draft_ref; a successfully repaired draft is no longer ambiguous. Code renders sentences; no-ref uses "the selected region". Numerical diagnostics are fallible. Return each U once; decisions must agree with observations."""


def requests(members, source_path=None, results=None):
    import json
    rows=[]
    source = np.asarray(Image.open(source_path).convert('RGB'), dtype=np.int16) if source_path else None
    if results is not None and len(results) != len(members):
        raise ValueError('audit evidence row count mismatch')
    for index, member in enumerate(members):
        spec=member['instruction_spec']
        row={'unit_id':member['unit_id'],'operation':spec['operation'],'draft_ref':spec['ref']}
        for key in ('payload','attributes','placement','target_part','qualifier'):
            if spec.get(key):row[key]=spec[key]
        if member.get('source_annotation_hint'):
            row['fallible_source_annotation'] = member['source_annotation_hint']
        if member.get('grounding_evidence') or member.get('source_grounding_evidence'):
            evidence = member.get('grounding_evidence') or member['source_grounding_evidence']
            from .grounding_evidence import hard_failures
            evidence = dict(evidence)
            source_only_warning = 'grounded_source_extent_exceeds_selector' in evidence.get('failure_reasons', [])
            evidence['failure_reasons'] = hard_failures(evidence)
            if source_only_warning:
                evidence['advisory_reasons'] = list(evidence.get('advisory_reasons') or []) + [
                    'segmented_source_exceeds_selector_verify_visible_extent']
                if not evidence['failure_reasons']:
                    evidence['status'] = 'source_scope_advisory'
            evidence['interpretation'] = 'fallible segmentation, not a visual verdict'
            row['independent_segmentation'] = {
                k:evidence.get(k) for k in ('status','query','source','result','failure_reasons','advisory_reasons','interpretation')}
        if source is not None:
            h,w = source.shape[:2]
            mask = load_mask({'mask_path':member['mask']}, (w,h))
            ys,xs=np.where(mask)
            if not len(xs):raise ValueError('empty audit evidence mask')
            row['source_mask_box_percent'] = [round(100*x/lim,1) for x,lim in zip(
                [xs.min(),ys.min(),xs.max()+1,ys.max()+1],[w,h,w,h])]
            if results is not None:
                result=np.asarray(Image.open(results[index]).convert('RGB'),dtype=np.int16)
                if result.shape!=source.shape:raise ValueError('audit evidence dimensions differ')
                row['selected_pixels_changed_percent']=round(100*float((np.max(abs(result-source),axis=2)[mask]>12).mean()),1)
        rows.append(row)
    return json.dumps(rows,ensure_ascii=False)


def schema(ids, dependencies=()):
    from .multi_component_audit import audit_schema
    result = audit_schema(ids, dependencies)
    fields = dict(unit_id={'type':'string','enum':list(ids)}, source_state={'type':'string'},result_state={'type':'string'},
                  image_ok={'type':'boolean'},
                  scope_status={'enum':['matched','too_broad','too_narrow','wrong_entity','unclear']},
                  observed_operation={'enum':['add','remove','replace','attribute','none','mixed','unclear']},
                  reference_unique={'type':'boolean'},
                  ref_issue={'type':'string','maxLength':180},
                  final_ref={'type':'string','maxLength':240})
    fields['reference_unique']['description'] = (
        'Judge final_ref if supplied, otherwise draft_ref. True for a successful unique repair; '
        'false only if the resulting wording still admits another visible target.')
    result['properties']['units']['items'] = dict(type='object', properties=fields,
        required=list(fields), additionalProperties=False)
    return result


def views(source_path, members, results, directory):
    """One result per U; also supports independent twin outcomes of one source."""
    directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
    source=Image.open(source_path).convert('RGB')
    if not members or len(members)!=len(results): raise ValueError('audit row count mismatch')
    width, row_height, header = 1152, 360, 44
    panels=[Image.new('RGB',(width,row_height*len(members)), '#e9edf3') for _ in range(2)]
    boxes=[]
    for index,(member,result_path) in enumerate(zip(members,results)):
        result=Image.open(result_path).convert('RGB')
        if result.size != source.size: raise ValueError('audit dimensions differ')
        mask=load_mask({'mask_path':member['mask']}, source.size)
        support=load_mask({'mask_path':member.get('support_mask') or member['mask']}, source.size)
        if not mask.any(): raise ValueError('empty audit selector')
        # Context always covers the full writable support (including oversizing).
        # Detail covers the selected pixels and actual changes inside that support.
        contexts=[]
        import re
        ref = member['instruction_spec']['ref']
        # A hand mentioned only to locate a ring does not make this a body edit.
        # Excess context previously shrank tiny accessory changes to a few pixels.
        target_phrase = re.split(r'\b(?:on|of|above|below|beside|behind|worn)\b', ref, maxsplit=1)[0]
        host_context = bool(re.search(r'\b(?:skin|forehead|nose|cheek|shoulder|sleeve|shirt|jacket|coat|person|man|woman|hand|arm|leg|head|torso|hoodie|backpack)\b', target_phrase, re.I)
                            or re.match(r"the (?:patch|area|section) of skin\b",ref,re.I))
        margin = .85 if host_context else .25
        for focus,margin in ((mask|support,margin),(mask|support,margin)):
            ys,xs=np.where(focus); pad=max(12,round(max(np.ptp(xs)+1,np.ptp(ys)+1)*margin))
            contexts.append((max(0,int(xs.min())-pad),max(0,int(ys.min())-pad),
                             min(source.width,int(xs.max())+pad+1),min(source.height,int(ys.max())+pad+1)))
        boxes.append(dict(unit_id=member['unit_id'],box=list(contexts[0]),detail_box=list(contexts[1]),image=str(result_path)))
        for column,box in enumerate(contexts):
            crops=[source.crop(box),result.crop(box)]
            # Draw after scaling: tiny source masks retain a legible contour.
            size=ImageOps.contain(crops[0],(width//2-12,row_height-header-8)).size
            local=np.asarray(Image.fromarray(mask.astype('uint8')*255).crop(box).resize(size,Image.Resampling.NEAREST))>0
            ring=binary_dilation(local,iterations=2)&~local
            for side in (0,1):
                # Image 3 now puts BEFORE and AFTER next to one another;
                # cross-image matching is used only for selection evidence.
                tile=(crops[0] if side==1 and column==0 else crops[side]).resize(size,Image.Resampling.LANCZOS)
                if side==0:
                    pixels=np.asarray(tile).copy()
                    if column==1:
                        from .per_mask_planning import excluded_pixels_view
                        tile=excluded_pixels_view(tile,local)
                    else:
                        pixels[ring]=(255,225,0)
                        tile=Image.fromarray(pixels)
                x=column*(width//2)+(width//2-size[0])//2
                y=index*row_height+header+(row_height-header-size[1])//2
                panels[side].paste(tile,(x,y))
                draw=ImageDraw.Draw(panels[side])
                draw.text((column*(width//2)+12,index*row_height+8),
                    member['unit_id']+(' SOURCE | context' if side==0 and column==0 else ' SOURCE | selected pixels' if side==0 else ' BEFORE' if column==0 else ' AFTER'),
                    font=load_font(22,bold=True),fill='#102030')
    paths=[str(source_path)]
    for panel,name in zip(panels,('source_rows.jpg','result_rows.jpg')):
        path=directory/name;panel.save(path,quality=96,subsampling=0);paths.append(str(path))
    atomic_json(directory/'result_crop_boxes.json',boxes)
    return paths
