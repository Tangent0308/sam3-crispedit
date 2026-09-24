"""Portable paired report including failed compositions and full review evidence."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageOps
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import padded_mask_bbox, audit_two_image_inputs


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def panel(source, baseline, candidate, mask, title, instruction):
    width=420; height=300
    sheet=Image.new('RGB',(width*3,height*2+90),'white');draw=ImageDraw.Draw(sheet)
    draw.text((10,6),title,fill='black')
    for n,line in enumerate(textwrap.wrap(instruction,150)):
        draw.text((10,22+n*13),line,fill='black')
    bbox=padded_mask_bbox(mask,padding_fraction=.28,min_padding=32)
    marked,_=audit_two_image_inputs(source,source,mask,max(source.size),'full')
    marked_source=marked.crop((0,40,source.width,source.height+40))
    for col,(label,im) in enumerate([('SOURCE',source),('BASELINE / 40 steps',baseline),('CANDIDATE / 40 steps',candidate)]):
        draw.text((col*width+10,64),label,fill='black')
        # Never obscure tiny source faces/details in both views with contours.
        local=(marked_source if col==0 else im).crop(bbox)
        for row,img in enumerate([im,local]):
            thumb=ImageOps.contain(img,(width-8,height-8))
            sheet.paste(thumb,(col*width+(width-thumb.width)//2,85+row*height+(height-thumb.height)//2))
    return sheet


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reviews',type=Path)
    a=p.parse_args();out=a.root/'paired_report';out.mkdir(exist_ok=True)
    reviews={r['image']:r for r in read(a.reviews)} if a.reviews else {}
    rows=read(a.root/'regions/annotations.jsonl');cards=[];sheets=[];manifest=[]
    variant='context_grounded_v4_qwen21'
    configs=[('baseline','final_legacy'),('typed','final_guarded-v1')]
    annotations={label:{r['image']:r for r in read(a.root/label/final/'annotations.jsonl')} for label,final in configs}
    audits={label:{r['image']:r for r in read(a.root/label/'audit/quality.jsonl')} for label,_ in configs}
    summary={'planned_cases':len(rows),'assistant_reviewed':len(reviews),'variants':{}}
    for label,final in configs:
        review_key='baseline' if label=='baseline' else 'candidate'
        reviewed=[r[review_key] for r in reviews.values()]
        common=[name for name in annotations[label] if name in reviews and name in audits[label]]
        summary['variants'][label]={
            'assembled':len(annotations[label]),
            'assistant_visual_pass_including_raw':sum(r['visual_quality']=='pass' for r in reviewed),
            'assistant_usable_final_with_original_instruction':sum(r['usable_with_original_instruction'] for r in reviewed),
            'model_quality_pass':sum(r.get('quality')=='pass' for r in audits[label].values()),
            'model_quality_reviewed':len(audits[label]),
            'quality_agreement':sum(reviews[name][review_key]['visual_quality']==audits[label][name].get('quality') for name in common),
            'quality_agreement_denominator':len(common),
            'audit_scope':'quality-only; no instruction rewrite or final admission in this experiment',
        }
    for row in rows:
        name=row['image'];source=Image.open(a.root/'regions/sources'/row['source_image']).convert('RGB')
        mask=mask_array(source.size,row['mask']);images=[];evidence={};statuses=[]
        for label,final in configs:
            path=a.root/label/final/'edited'/name
            status='final' if path.exists() else 'RAW ONLY / composition rejected'
            if not path.exists():path=a.root/label/variant/'edited'/name
            images.append(Image.open(path).convert('RGB'));statuses.append(status)
            request=json.loads((a.root/label/variant/'diagnostics'/Path(name).stem/'generation_request.json').read_text())
            evidence[label]=dict(status=status,displayed_path=str(path),generation_request=request,
                annotation=annotations[label].get(name),model_quality=audits[label].get(name))
        sheet=panel(source,*images,mask,name,row['editing_instruction']);sheets.append(sheet)
        jpg=out/(Path(name).stem+'.jpg');sheet.save(jpg,quality=92)
        buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=88)
        review=reviews.get(name)
        e=lambda val:html.escape(str(val))
        cards.append(f'<article id="case-{name[:3]}"><h2>{e(name)}</h2><p>{e(row["editing_instruction"])}</p>'
            f'<p>左：source；中：baseline ({statuses[0]})；右：candidate ({statuses[1]})。上排无标记全图，下排同坐标局部，原图局部的白色轮廓表示源 mask。</p>'
            f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" loading="lazy">'
            f'<h3>Assistant逐图复核</h3><pre>{e(json.dumps(review,ensure_ascii=False,indent=2) if review else "待复核")}</pre>'
            f'<details><summary>原始指令、实际生图prompt、合成证据及模型判断</summary><pre>{e(json.dumps(evidence,ensure_ascii=False,indent=2))}</pre></details></article>')
        manifest.append(dict(image=name,review=review,statuses=statuses))
    for offset in range(0,len(sheets),3):
        group=sheets[offset:offset+3];thumbs=[s.resize((1008,552)) for s in group]
        contact=Image.new('RGB',(1008,552*len(thumbs)),'white')
        for i,thumb in enumerate(thumbs):contact.paste(thumb,(0,i*552))
        contact.save(out/f'contact_{offset:03d}.jpg',quality=92)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Qwen2.1 40-step quality comparison</title>'
        '<style>body{max-width:1280px;margin:24px auto;font-family:system-ui}img{width:100%}article{border:1px solid #bbb;margin:24px 0;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        f'<h1>Qwen-Image-2.1：固定40步的新源对照（{len(rows)}个case）</h1>'
        '<p>baseline=原提示+原合成；candidate=类型提示+带检查的合成。模型、指令、mask、seed相同。合成失败仍展示raw并明确标注，不从分母删除。Assistant review来自Codex看图，model_quality来自27B。</p>'
        '<p>这是开发对照，不是已通过最终审核的数据集。27B本轮仅作独立画质检查，没有运行指令重写或最终入库审核；raw-only不算可交付final。</p>'
        f'<details open><summary>汇总（Assistant复核不是独立人工金标准）</summary><pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre></details>'
        +''.join(cards))
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(dict(cases=len(rows),reviewed=len(reviews),report=str(out/'index.html'))))


if __name__=='__main__':main()
