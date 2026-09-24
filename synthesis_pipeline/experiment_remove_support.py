"""Replay removal composition without diffusion, with portable four-way evidence."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw,ImageOps
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.refine_samtok_regions import encode_mask
from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox
from utils.context_edit import compose_grounded_crop
from utils.remove_support import adaptive_remove_support


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def comparison(source,raw,baseline,candidate,mask,name,labels=None):
    bbox=padded_mask_bbox(mask,padding_fraction=.25,min_padding=32)
    sheet=Image.new('RGB',(1440,700),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,4),name,fill='black')
    labels=labels or ['SOURCE / clean','RAW / before writeback','BASELINE / narrow support','CANDIDATE / adaptive support']
    for col,(label,im) in enumerate(zip(labels,[source,raw,baseline,candidate])):
        draw.text((col*360+8,24),label,fill='black')
        for row,picture in enumerate([im,im.crop(bbox)]):
            thumb=ImageOps.contain(picture,(352,316))
            sheet.paste(thumb,(col*360+(360-thumb.width)//2,46+row*326+(316-thumb.height)//2))
    return sheet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--reviews',type=Path)
    p.add_argument('--policy',choices=['adaptive-remove-v1','adaptive-remove-v2'],default='adaptive-remove-v1')
    p.add_argument('--baseline-root',type=Path,help='Existing final outputs to compare; raw-root by default')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    for sub in ['edited','support','report']:(a.out_root/sub).mkdir(exist_ok=True)
    if not (a.out_root/'sources').exists():(a.out_root/'sources').symlink_to((a.data_root/'sources').resolve())
    reviews={x['image']:x for x in read(a.reviews)} if a.reviews else {}
    rows=read(a.raw_root/'annotations.jsonl');records=[];cards=[];times=[];sheets=[]
    for row in rows:
        if row['task_type']!='remove':continue
        name=row['image'];d=a.raw_root/'diagnostics'/Path(name).stem
        request=json.loads((d/'generation_request.json').read_text());bbox=tuple(request['crop_bbox'])
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        raw=Image.open(d/'raw_edited_crop.png').convert('RGB')
        mask=mask_array(source.size,row['mask']).astype(bool)
        protect=mask_array(source.size,row['region_contract']['protected_mask']).astype(bool)
        start=time.perf_counter()
        final,alpha=compose_grounded_crop(source,raw,mask,'remove',bbox,protect,
            remove_composition_policy=a.policy)
        times.append(time.perf_counter()-start)
        x1,y1,x2,y2=bbox
        _,evidence=adaptive_remove_support(source.crop(bbox),raw,mask[y1:y2,x1:x2],protect[y1:y2,x1:x2],policy=a.policy)
        final.save(a.out_root/'edited'/name);alpha.save(a.out_root/'support'/name)
        support=np.zeros_like(mask);support[y1:y2,x1:x2]=np.asarray(alpha)>0
        guard=protect&~mask
        if not np.array_equal(np.asarray(final)[guard],np.asarray(source)[guard]):
            raise AssertionError('Protected source pixels changed')
        records.append({**row,'edit_support_mask':encode_mask(support),'remove_composition':evidence})
        baseline=Image.open((a.baseline_root or a.raw_root)/'edited'/name).convert('RGB')
        raw_full=source.copy();raw_full.paste(raw,bbox[:2])
        sheet=comparison(source,raw_full,baseline,final,mask,name,
            ['SOURCE / clean','RAW / before writeback','BASELINE / saved final',f'CANDIDATE / {a.policy}']);sheets.append(sheet)
        sheet.save(a.out_root/'report'/(Path(name).stem+'.jpg'),quality=94)
        buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=90)
        details={'instruction':row['editing_instruction'],'support_evidence':evidence,
                 'generation_request':request,'assistant_review':reviews.get(name,'not reviewed')}
        cards.append(f'<article><h2>{html.escape(name)}</h2><p>{html.escape(row["editing_instruction"])}</p>'
            f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'
            f'<pre>{html.escape(json.dumps(details,ensure_ascii=False,indent=2))}</pre></article>')
    for offset in range(0,len(sheets),3):
        group=sheets[offset:offset+3];contact=Image.new('RGB',(1152,560*len(group)),'white')
        for i,sheet in enumerate(group):contact.paste(sheet.resize((1152,560)),(0,i*560))
        contact.save(a.out_root/'report'/f'contact_{offset:03d}.jpg',quality=94)
    (a.out_root/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    summary=dict(cases=len(records),mean_composition_seconds=sum(times)/max(1,len(times)),diffusion_calls=0,
                 source_masks_modified=False,policy=a.policy,quality_status='experimental')
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.out_root/'report/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Removal support comparison</title>'
        '<style>body{max-width:1500px;margin:24px auto;font-family:system-ui}img{width:100%}pre{white-space:pre-wrap}article{border:1px solid #bbb;padding:12px;margin:24px 0}</style>'
        '<h1>删除写回：同一raw的受限修补区域对照</h1><p>从左到右：干净原图、模型raw、原合成、修复合成。上排完整场景，下排同坐标局部。raw列用于诊断，未经过邻居保护，不是可直接入库结果。全部保持原生成结果，不新增扩散。</p>'
        +''.join(cards))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
