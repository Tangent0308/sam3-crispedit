"""Portable, fully embedded fixed-version comparison; keep all rejected rows."""
import argparse,base64,html,io,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.prepare_samtok_data import load_jsonl
from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox


def encoded(image):
    stream=io.BytesIO();image.save(stream,format='JPEG',quality=90)
    return 'data:image/jpeg;base64,'+base64.b64encode(stream.getvalue()).decode()


def outlined(source,mask):
    arr=np.asarray(source).copy();contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr,contours,-1,(0,0,0),4);cv2.drawContours(arr,contours,-1,(255,255,255),2)
    return Image.fromarray(arr)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--variant',action='append',required=True,help='label=output root')
    p.add_argument('--reviews',type=Path)
    p.add_argument('--audit',type=Path,help='Full model audit JSONL; separate from Assistant review')
    p.add_argument('--resolution',type=Path,help='Preserve grounding failures and conservative fallback evidence')
    p.add_argument('--title',default='Ownership / generation / seam ablation')
    p.add_argument('--ids',default='')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    variants=[(v.split('=',1)[0],Path(v.split('=',1)[1])) for v in a.variant]
    reviews=json.loads(a.reviews.read_text()) if a.reviews and a.reviews.exists() else {}
    audits={r['image']:r for r in load_jsonl(a.audit)} if a.audit and a.audit.exists() else {}
    resolution={r['image']:r for r in load_jsonl(a.resolution)} if a.resolution and a.resolution.exists() else {}
    rows=load_jsonl(a.data_root/'annotations.jsonl');cards=[];stats={label:dict(output=0,pass_count=0,fail_count=0,not_reviewed=0) for label,_ in variants}
    if a.ids:
        ids={int(x) for x in a.ids.split(',')};rows=[r for r in rows if int(r['image'].split('_')[0]) in ids]
    escape=lambda v:html.escape(json.dumps(v,ensure_ascii=False,indent=2))
    for row in rows:
        name=row['image'];source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        mask=mask_array(source.size,row['mask']).astype(bool);box=padded_mask_bbox(mask,.65,80)
        originals=outlined(source,mask);columns=[('SOURCE: outline is annotation',originals)]
        details=[]
        for label,root in variants:
            file=root/'edited'/name;review=reviews.get(name,{}).get(label)
            columns.append((label,Image.open(file).convert('RGB') if file.exists() else None))
            if file.exists():
                stats[label]['output']+=1
                status=review.get('status') if isinstance(review,dict) else None
                stats[label][status+'_count' if status in {'pass','fail'} else 'not_reviewed']+=1
            d=root/'diagnostics'/Path(name).stem;request_path=d/'generation_request.json'
            if request_path.exists():
                req=json.loads(request_path.read_text());visible={k:req.get(k) for k in ['prompt','reference_images','crop_bbox','latent_protection_policy','relation_geometry_policy','remove_composition_policy','elapsed_seconds']}
                imgs=''.join('<figcaption>'+html.escape(f)+'</figcaption><img loading="lazy" src="'+encoded(Image.open(d/f))+'">'
                    for f in ['source_crop.png','raw_edited_crop.png','composition_alpha.png','protected_instances.png'] if (d/f).exists())
                details.append('<details><summary>'+html.escape(label)+' : raw / protection / prompt</summary><pre>'+escape(visible)+'</pre>'+imgs+'</details>')
        panel=Image.new('RGB',(480*len(columns),1020),'white');draw=ImageDraw.Draw(panel)
        for c,(label,photo) in enumerate(columns):
            draw.text((c*480+6,6),label,fill='black')
            if photo is None:draw.text((c*480+40,200),'NO OUTPUT (not a pass)',fill='red');continue
            for y,img in [(30,photo),(530,photo.crop(box))]:
                tile=ImageOps.contain(img,(476,476));panel.paste(tile,(c*480+(480-tile.width)//2,y+(476-tile.height)//2))
        panel.save(a.out_root/(Path(name).stem+'.jpg'),quality=94)
        instruction=row.get('relation_plan',{}).get('instruction',row.get('editing_instruction',''))
        cards.append('<article id="case-'+name[:3]+'"><h2>'+html.escape(name)+'</h2><p>'+html.escape(instruction)+'</p>'
            '<img loading="lazy" src="'+encoded(panel)+'"><h3>Assistant review (independent visual inspection)</h3><pre>'+escape(reviews.get(name,'Not yet reviewed'))+'</pre>'
            '<details><summary>Plan and status</summary><pre>'+escape({k:row.get(k) for k in ['relation_status','relation_plan']})+'</pre></details>'
            '<details><summary>Grounding / no-output reason</summary><pre>'+escape(resolution.get(name,'Not provided'))+'</pre></details>'
            '<details><summary>27B model audit: complete reason, raw reply and prompt</summary><pre>'+escape(audits.get(name,'Not run / no output; not an Assistant verdict'))+'</pre></details>'+''.join(details)+'</article>')
    summary=dict(input_cases=len(rows),variants=stats)
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.out_root/'index.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(a.title)+'</title>'
        '<style>body{max-width:1800px;margin:24px auto;font-family:system-ui}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border:1px solid #aaa;padding:12px;margin:25px 0}details img{max-height:750px}</style></head><body><h1>'+html.escape(a.title)+'</h1>'
        '<p>上排完整场景，下排同位置放大。第一列白线仅表示原始mask。各列为固定版本，不逐例择优；NO OUTPUT不计作成功。Assistant review是本助手看图判断，不是产线模型审核。展开查看完整plan、实际prompt及raw。全部图片内嵌。</p><pre>'+escape(summary)+'</pre>'+''.join(cards)+'</body></html>')
    print(summary)


if __name__=='__main__':main()
