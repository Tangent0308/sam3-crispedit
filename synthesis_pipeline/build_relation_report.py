"""Portable relation-pilot report retaining ungenerated/deferred cases."""
import argparse,base64,html,json
from pathlib import Path
from PIL import Image,ImageDraw
from synthesis_pipeline.build_qwen21_quality_report import panel,read
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop


def embedded(path):
    return '<img loading="lazy" src="data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()+'">'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path)
    p.add_argument('--candidate',type=Path,help='Frozen-plan editor variant; never copy or overwrite original run')
    p.add_argument('--out-root',type=Path)
    p.add_argument('--reviews',type=Path)
    p.add_argument('--audit-jsonl',type=Path,help='Exact model audit records, separate from assistant reviews')
    a=p.parse_args();out=a.out_root or a.root/'report';out.mkdir(parents=True,exist_ok=True)
    rows=read(a.root/'relations/annotations.jsonl');resolved={r['image']:r for r in read(a.root/'regions/annotations.jsonl')}
    resolution={r['image']:r for r in read(a.root/'regions/resolution.jsonl')}
    oldrows={r['image']:r for r in read(a.baseline/'annotations.jsonl')} if a.baseline and (a.baseline/'annotations.jsonl').exists() else {}
    reviews={r['image']:r for r in read(a.reviews)} if a.reviews else {}
    audits={r['image']:r for r in read(a.audit_jsonl)} if a.audit_jsonl else {}
    if reviews and set(reviews)!={r['image'] for r in rows}:raise ValueError('Incomplete reviews')
    rawroot=a.candidate or a.root/'editing/context_grounded_v4_qwen21';cards=[];sheets=[];generated=0
    for row in rows:
        name=row['image'];stem=Path(name).stem
        source=Image.open(a.root/'relations/sources'/row['source_image']).convert('RGB')
        mask=mask_array(source.size,row['mask']);newrow=resolved.get(name);request=None
        path=rawroot/'edited'/name;exists=path.exists()
        instruction=(newrow or {}).get('editing_instruction',row.get('relation_plan',{}).get('instruction',''))
        if exists:
            generated+=1;final=Image.open(path).convert('RGB');d=rawroot/'diagnostics'/stem
            request=json.loads((d/'generation_request.json').read_text());raw=source.copy();raw.paste(Image.open(d/'raw_edited_crop.png'),request['crop_bbox'][:2])
            middle=Image.open(a.baseline/'edited'/name).convert('RGB') if a.baseline else raw
            sheet=panel(source,middle,final,mask,name,instruction)
            draw=ImageDraw.Draw(sheet);draw.rectangle((420,60,1259,80),fill='white')
            draw.text((430,64),'PREVIOUS FINAL' if a.baseline else 'RAW / 40 steps',fill='black');draw.text((850,64),'RELATION FINAL / 40 steps',fill='black')
        else:
            sheet=panel(source,source,source,mask,name+' / NOT GENERATED',instruction)
            draw=ImageDraw.Draw(sheet);draw.rectangle((420,60,1259,80),fill='white')
            draw.text((430,64),'SOURCE ONLY / DEFERRED',fill='black');draw.text((850,64),'NO EDITED OUTPUT',fill='black')
        jpg=out/(stem+'.jpg');sheet.save(jpg,quality=94);sheets.append(sheet)
        review=reviews.get(name,'待复核');details=dict(relation_plan=row.get('relation_plan'),resolution=resolution.get(name),generation_request=request,
            previous_instruction=oldrows.get(name,{}).get('editing_instruction',row.get('editing_instruction')),assistant_review=review,
            model_audit=audits.get(name))
        card=f'<article id="case-{name[:3]}"><h2>{html.escape(name)}</h2>'+embedded(jpg)+'<pre>'+html.escape(json.dumps(review,ensure_ascii=False,indent=2))+'</pre>'
        if name in audits:
            record=audits[name]
            if Path(record['edited_path']).resolve()!=path.resolve():
                raise ValueError('Audit belongs to a different edited image')
            card+='<h3>Qwen3.8-27B审核（独立于Assistant review）</h3><pre>'+html.escape(json.dumps(record.get('parsed'),ensure_ascii=False,indent=2))+'</pre>'
            card+='<details><summary>完整模型回复、审核prompt及实际输入</summary><pre>'+html.escape(record['raw_response'])+'</pre><pre>'+html.escape(record['prompt'])+'</pre>'
            for input_path in record['input_images']:
                data=Path(input_path).read_bytes()
                card+='<img loading="lazy" src="data:image/png;base64,'+base64.b64encode(data).decode()+'">'
            card+='</details>'
        if a.baseline:
            card+='<p>旧指令：'+html.escape(details['previous_instruction'] or '见旧实验记录')+'<br>本版指令：'+html.escape(instruction)+'</p>'
        card+='<details><summary>实际规划输入、关系判断、定位证据与生图prompt</summary>'
        inputs=list((a.root/'planners').glob('*/inputs/'+name))
        if inputs:
            inputjpg=out/(stem+'_input.jpg');Image.open(inputs[0]).convert('RGB').save(inputjpg,quality=94)
            card+='<p>MLLM输入：干净全图（上方左图）和以下轮廓crop。</p>'+embedded(inputjpg)
            prompt=inputs[0].with_suffix('.txt')
            if not prompt.exists():prompt=inputs[0].parent.parent/'prompt.txt'
            if prompt.exists():card+='<pre>'+html.escape(prompt.read_text())+'</pre>'
        card+='<pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2))+'</pre></details></article>';cards.append(card)
        (out/(stem+'.json')).write_text(json.dumps(details,ensure_ascii=False,indent=2))
    for offset in range(0,len(sheets),3):
        group=sheets[offset:offset+3];contact=Image.new('RGB',(1008,552*len(group)),'white')
        for i,sheet in enumerate(group):contact.paste(sheet.resize((1008,552)),(0,i*552))
        contact.save(out/f'contact_{offset:03d}.jpg',quality=94)
    summary=dict(input_cases=len(rows),generated=generated,deferred=len(rows)-generated,reviewed=len(reviews),
        assistant_pass=sum(r.get('result')=='pass' for r in reviews.values()))
    if reviews and all('baseline_result' in r for r in reviews.values()):
        summary.update(baseline_pass=sum(r['baseline_result']=='pass' for r in reviews.values()),
            improvements=[r['image'] for r in reviews.values() if r['baseline_result']=='fail' and r['result']=='pass'],
            regressions=[r['image'] for r in reviews.values() if r['baseline_result']=='pass' and r['result']=='fail'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Relation-aware removal pilot</title>'
        '<style>body{font-family:system-ui;max-width:1300px;margin:24px auto}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border:1px solid #bbb;margin:24px 0;padding:14px}</style>'
        '<h1>保留独立对象与连带编辑：实验结果</h1><p>左原图，中'+('旧版final' if a.baseline else '本版raw')+'，右本版final；上排全图，下排同坐标局部。白轮廓为原始mask。NOT GENERATED仅重复展示原图，不代表编辑输出。'
        'Assistant review来自Codex逐图判断，不是pipeline审核或独立人工金标准。本试验尚未接入生产成图audit和最终准入。图片全部内嵌。</p>'
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
