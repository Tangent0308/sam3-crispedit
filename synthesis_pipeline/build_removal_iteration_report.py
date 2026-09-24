"""Portable multi-arm removal comparison with explicit manual-review provenance."""
import argparse
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from synthesis_pipeline.build_qwen21_backend_report import image_uri, outlined_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--arm',action='append',required=True,help='label=/path/to/run with edited/diagnostics')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--ids',default='')
    p.add_argument('--notice',default='',help='Visible experiment scope and acceptance caveats')
    p.add_argument('--reviews',type=Path)
    p.add_argument('--audit',action='append',default=[],help='label=/path/to/audit.jsonl')
    p.add_argument('--rewrite',action='append',default=[],help='label=/path/to/rewrites.jsonl; candidates, not automatic approvals')
    p.add_argument('--rewrite-reviews',type=Path,help='Independent Assistant review of rewrite candidates, keyed by image then arm')
    p.add_argument('--arm-plan',action='append',default=[],help='label=/path/to/plan or resolved regions')
    p.add_argument('--arm-resolution',action='append',default=[],help='label=/path/to/resolution.jsonl')
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    arms={k:Path(v) for k,v in (s.split('=',1) for s in a.arm)}
    rows=[json.loads(s) for s in (a.data_root/'annotations.jsonl').read_text().splitlines()]
    if a.ids:rows=[r for r in rows if int(r['image'].split('_')[0]) in set(map(int,a.ids.split(',')))]
    reviews=json.loads(a.reviews.read_text()) if a.reviews else {}
    rewrite_reviews=json.loads(a.rewrite_reviews.read_text()) if a.rewrite_reviews else {}
    audits={k:{r['image']:r for r in map(json.loads,Path(v).read_text().splitlines())}
            for k,v in (s.split('=',1) for s in a.audit)}
    rewrites={k:{r['image']:r for r in map(json.loads,Path(v).read_text().splitlines())}
              for k,v in (s.split('=',1) for s in a.rewrite)}
    plans={k:Path(v) for k,v in (s.split('=',1) for s in a.arm_plan)}
    resolution_files={k:Path(v) for k,v in (s.split('=',1) for s in a.arm_resolution)}
    arm_rows={};resolutions={}
    for label,root in arms.items():
        manifest=root/'annotations.jsonl'
        if label in plans:manifest=plans[label]/'annotations.jsonl'
        arm_rows[label]={r['image']:r for r in map(json.loads,manifest.read_text().splitlines())} if manifest.exists() else {}
        resolution=resolution_files.get(label,plans.get(label,root)/'resolution.jsonl')
        resolutions[label]={r['image']:r for r in map(json.loads,resolution.read_text().splitlines())} if resolution.exists() else {}
    cards=[];counts={k:{} for k in arms};sheets=[]
    for row in rows:
        name=row['image'];stem=Path(name).stem
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        panels=[('Source / target outline',outlined_source(source,row))]
        details=[]
        for label,root in arms.items():
            path=root/'edited'/name
            if not path.exists():
                panels.append((label+' / no output',Image.new('RGB',source.size,'#ddd')))
                status=resolutions[label].get(name,{}).get('status',arm_rows[label].get(name,{}).get('relation_status','no_output'))
                counts[label]['no_output']=counts[label].get('no_output',0)+1
                reason=arm_rows[label].get(name,{}).get('relation_plan',{}).get('reason','')
                details.append(f'<h3>{html.escape(label)}：无输出（不计为成功）</h3><p>{html.escape(status)} — {html.escape(reason)}</p>')
                if name in resolutions[label]:
                    details.append('<details><summary>执行区域解析记录（不是源mask质量审核）</summary><pre>'+
                        html.escape(json.dumps(resolutions[label][name],ensure_ascii=False,indent=2))+'</pre></details>')
                continue
            result=Image.open(path).convert('RGB');panels.append((label,result))
            review=reviews.get(name,{}).get(label,{'verdict':'not_reviewed','reason':'尚未人工判断'})
            verdict=review['verdict'];counts[label][verdict]=counts[label].get(verdict,0)+1
            diagnostic=root/'diagnostics'/stem
            request=json.loads((diagnostic/'generation_request.json').read_text())
            rewrite=rewrites.get(label,{}).get(name)
            rewrite_review=rewrite_reviews.get(name,{}).get(label)
            audit=audits.get(label,{}).get(name)
            details.append(f'<h3>{html.escape(label)} — Assistant: {html.escape(verdict)}</h3>'
                f'<p>{html.escape(review["reason"])}</p>'
                f'<p>本版本 instruction: {html.escape(arm_rows[label].get(name,{}).get("relation_plan",{}).get("instruction",arm_rows[label].get(name,{}).get("editing_instruction",row.get("editing_instruction",""))))}</p>'
                +(f'<p>Pipeline最终判定: {html.escape(audits[label][name]["decision"])} / Qwen3.8-27B原始判定: {html.escape(audits[label][name].get("model_decision",audits[label][name]["decision"]))}</p><pre>{html.escape(json.dumps(audits[label][name].get("parsed"),ensure_ascii=False,indent=2))}</pre>'
                  + (f'<p>像素变化否决：原始目标区域几乎未变（不是MLLM的理由）。</p><pre>{html.escape(json.dumps(audits[label][name]["pixel_evidence"],indent=2))}</pre>' if audits[label][name].get('pixel_veto_applied') else '') if name in audits.get(label,{}) else '')
                +(f'<h4>独立改写候选（不是自动验收）</h4><p>{html.escape(str(rewrite.get("instruction")))}</p>'
                  f'<details><summary>改写模型完整回复</summary><pre>{html.escape(rewrite.get("raw_response",""))}</pre></details>' if rewrite else '')
                +(f'<p>Assistant 对改写的判断：{html.escape(rewrite_review["verdict"])} — {html.escape(rewrite_review["reason"])}</p>' if rewrite_review else '')
                +(('<p>审核回复未能解析：流程保守拒绝，不等于模型明确判定质量失败。</p>' if audit.get('parsed') is None else '')
                  +f'<details><summary>审核模型原始完整回复（同一次调用）</summary><pre>{html.escape(audit.get("raw_response",""))}</pre></details>' if audit else '')
                +'<details><summary>实际输入、raw与融合后crop、prompt</summary>'
                '<div class="row">'+''.join(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{image_uri(im)}"></figure>'
                    for title,im in [('actual condition' if (diagnostic/'condition_image.png').exists() else 'clean input',Image.open(diagnostic/('condition_image.png' if (diagnostic/'condition_image.png').exists() else 'source_crop.png'))),
                                     ('raw before composition',Image.open(diagnostic/'raw_edited_crop.png')),
                                     ('final crop',result.crop(request['crop_bbox']))])+'</div>'
                +(f'<img class="guide" src="{image_uri(Image.open(diagnostic/"target_guide.png"))}">' if (diagnostic/'target_guide.png').exists() else '')
                +f'<pre>{html.escape(request["prompt"])}</pre></details>')
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(row.get("editing_instruction",""))}</p>'
            '<div class="row">'+''.join(f'<figure><figcaption>{html.escape(title)}</figcaption><img src="{image_uri(im)}"></figure>' for title,im in panels)
            +'</div>'+''.join(details)+'</section>')
        width=420*len(panels);panel=Image.new('RGB',(width,405),'#eee');draw=ImageDraw.Draw(panel)
        draw.text((8,6),name,fill='black')
        for i,(title,im) in enumerate(panels):
            draw.text((i*420+8,27),title,fill='black')
            thumb=ImageOps.contain(im,(412,350));panel.paste(thumb,(i*420+(420-thumb.width)//2,48+(350-thumb.height)//2))
        sheets.append(panel)
    for i in range(0,len(sheets),3):
        group=sheets[i:i+3];canvas=Image.new('RGB',(group[0].width,405*len(group)),'white')
        for j,panel in enumerate(group):canvas.paste(panel,(0,405*j))
        canvas.save(a.out_root/f'contact_{i//3:02}.jpg',quality=95)
    (a.out_root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Removal iteration</title>'
        '<style>body{font:16px system-ui;background:#eee;margin:20px}section{background:white;padding:16px;margin:20px 0}'
        '.row{display:flex;gap:8px}figure{flex:1;min-width:0;margin:0}img{width:100%}.guide{max-width:700px}'
        'pre{white-space:pre-wrap}summary{cursor:pointer}</style><h1>Removal：同源对照迭代</h1>'
        +(f'<p>{html.escape(a.notice)}</p>' if a.notice else '')
        +'<p>左侧原图轮廓仅用于展示目标。后续各列是独立实验版本，不是审核模型的多条回复。'
        'Assistant理由来自人工看图，not_reviewed不表示通过。展开可检查实际输入、raw、最终crop和prompt。所有图片内嵌。</p>'
        '<pre>'+html.escape(json.dumps(counts,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    (a.out_root/'summary.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts),flush=True)


if __name__=='__main__':main()
