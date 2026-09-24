"""Optional, diagnostic-only repair of collapsed shading in color-only edits.

Never overwrites a generated image. The frozen audit cohort is not repaired in
place. This branch needs its own new-source comparison and quality audit before
it can be admitted to a dataset.
"""
import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from synthesis_pipeline.audit_edit_pairs import mask_array, write_jsonl


def restore_color_shading(source, edited, mask, row):
    if source.size!=edited.size:raise ValueError('Mismatched images')
    instruction=row.get('editing_instruction','').lower()
    eligible=row.get('task_type')=='attribute' and re.search(
        r'\bto\s+(?:a\s+)?(?:(?:bright|deep|dark|light|vibrant|navy)\s+)*(?:red|blue|green|yellow|purple|pink|orange|brown|black|white|teal|cyan|turquoise|burgundy|magenta|violet|emerald|olive)\b',instruction)
    # This cannot simulate new material, pattern, text, transparency or lighting.
    excluded=re.search(r'\b(metallic|gold|silver|chrome|transparent|glossy|matte|striped|pattern|variegated|solid|text|glowing|leather|velvet)\b',instruction)
    info={'eligible':bool(eligible and not excluded),'applied':False}
    if not info['eligible']:return edited.copy(),info
    mask=mask.astype(bool)
    inner=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if inner.sum()<64:return edited.copy(),{**info,'skip_reason':'too_few_interior_pixels'}
    a=np.asarray(source).astype(np.float32)/255
    b=np.asarray(edited).astype(np.float32)/255
    lab_a=cv2.cvtColor(a,cv2.COLOR_RGB2LAB);lab_b=cv2.cvtColor(b,cv2.COLOR_RGB2LAB)
    original=lab_a[...,0][inner];generated=lab_b[...,0][inner]
    source_range=float(np.percentile(original,90)-np.percentile(original,10))
    edited_range=float(np.percentile(generated,90)-np.percentile(generated,10))
    info.update(source_luma_p90_p10=source_range,edited_luma_p90_p10=edited_range)
    if source_range<8 or edited_range>=source_range*.25:
        return edited.copy(),{**info,'skip_reason':'no_strong_shading_collapse'}
    repaired=lab_b.copy()
    # Keep the requested color from actual Qwen output, restore source luminance
    # variation and geometric detail rather than borrowing a different texture.
    offset=float(np.median(generated)-np.median(original))
    repaired[...,0]=np.clip(lab_a[...,0]+offset,2,98)
    chroma=np.array([np.median(lab_b[...,1][inner]),np.median(lab_b[...,2][inner])])
    magnitude=float(np.linalg.norm(chroma))
    # Extreme generated chroma clips RGB channels and loses restored shading.
    # Preserve hue but bound chroma for this photographic diagnostic branch.
    chroma*=min(1.,60/max(magnitude,1e-6))
    repaired[...,1]=chroma[0];repaired[...,2]=chroma[1]
    rgb=cv2.cvtColor(repaired,cv2.COLOR_LAB2RGB)
    alpha=np.clip(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3)/1.5,0,1)[...,None]
    blended=np.rint(np.clip(rgb*alpha+b*(1-alpha),0,1)*255).astype(np.uint8)
    result=np.asarray(edited).copy()
    result[mask]=blended[mask]  # Exact output bytes outside the original mask.
    return Image.fromarray(result),{**info,'applied':True,'luma_offset':offset,
        'generated_chroma':magnitude,'used_chroma':float(np.linalg.norm(chroma)),
        'method':'source_luminance_with_generated_median_chroma_v1',
        'quality_status':'requires_independent_audit'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=False)
    (a.out_root/'edited').mkdir();(a.out_root/'sources').symlink_to((a.data_root/'sources').resolve())
    rows=[json.loads(x) for x in (a.data_root/'annotations.jsonl').read_text().splitlines() if x.strip()]
    records=[]
    for row in rows:
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        edited=Image.open(a.data_root/'edited'/row['image']).convert('RGB')
        result,record=restore_color_shading(source,edited,mask_array(source.size,row['mask']),row)
        if record['applied']:result.save(a.out_root/'edited'/row['image'])
        else:(a.out_root/'edited'/row['image']).symlink_to((a.data_root/'edited'/row['image']).resolve())
        records.append({**row,'color_texture_repair':record,'repair_source_path':str((a.data_root/'edited'/row['image']).resolve())})
    write_jsonl(a.out_root/'annotations.jsonl',records)
    summary=dict(cases=len(records),applied=sum(r['color_texture_repair']['applied'] for r in records),
        note='Diagnostic branch, not accepted labels; all originals preserved.')
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
