"""Same-plan prompt comparison, separating raw generation from final composition."""
import argparse
import base64
import html
import json
from pathlib import Path
from PIL import Image
import numpy as np
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.build_qwen21_quality_report import panel, read


def pixel_preservation(source, output, protected):
    if not protected.any():
        return {'pixels':0,'mae':None,'changed_fraction':None}
    difference=np.abs(np.asarray(source,dtype=np.float32)-np.asarray(output,dtype=np.float32)).mean(2)[protected]
    return dict(pixels=int(protected.sum()),mae=float(difference.mean()),changed_fraction=float((difference>16).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--baseline-raw',type=Path,required=True)
    p.add_argument('--baseline-final',type=Path,required=True)
    p.add_argument('--candidate-raw',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--reviews',type=Path)
    a=p.parse_args();a.out_root.mkdir(parents=True,exist_ok=True)
    rows=read(a.candidate_raw/'annotations.jsonl')
    baselines={r['image']:r for r in read(a.baseline_raw/'annotations.jsonl')}
    reviews={r['image']:r for r in read(a.reviews)} if a.reviews else {}
    if reviews and set(reviews)!={r['image'] for r in rows}:raise ValueError('Incomplete review')
    cards=[];sheets=[]
    policies=None
    for row in rows:
        name=row['image'];old=baselines[name]
        if any(old[key]!=row[key] for key in ['mask','editing_instruction','source_image']):
            raise ValueError('Unpaired source, mask or instruction')
        source=Image.open(a.data_root/'sources'/row['source_image']).convert('RGB')
        mask=mask_array(source.size,row['mask'])
        finals=[Image.open(root/'edited'/name).convert('RGB') for root in [a.baseline_final,a.candidate_raw]]
        raw=[];requests=[]
        for root in [a.baseline_raw,a.candidate_raw]:
            diagnostic=root/'diagnostics'/Path(name).stem
            request=json.loads((diagnostic/'generation_request.json').read_text());requests.append(request)
            im=source.copy();im.paste(Image.open(diagnostic/'raw_edited_crop.png'),request['crop_bbox'][:2]);raw.append(im)
        if requests[0]['crop_bbox']!=requests[1]['crop_bbox']:raise ValueError('Different context crops')
        pair=[request['qwen21_prompt_policy']+' / '+request.get('latent_protection_policy','legacy')+' / geometry:'+request.get('relation_geometry_policy','legacy') for request in requests]
        if policies is not None and pair!=policies:raise ValueError('Mixed policies in comparison')
        policies=pair
        panels=[]
        for suffix,outputs in [('',finals),('_raw',raw)]:
            sheet=panel(source,*outputs,mask,name+(' / RAW' if suffix else ' / FINAL'),row['editing_instruction'])
            jpg=a.out_root/(Path(name).stem+suffix+'.jpg');sheet.save(jpg,quality=94)
            panels.append('<img loading="lazy" src="data:image/jpeg;base64,'+base64.b64encode(jpg.read_bytes()).decode()+'">')
            if not suffix:sheets.append(sheet)
        review=reviews.get(name, '待复核')
        guard_rle=row.get('region_contract',{}).get('protected_mask')
        guard=mask_array(source.size,guard_rle).astype(bool)&~mask.astype(bool) if guard_rle else np.zeros_like(mask,dtype=bool)
        metrics=dict(baseline_raw=pixel_preservation(source,raw[0],guard),candidate_raw=pixel_preservation(source,raw[1],guard),
                     baseline_final=pixel_preservation(source,finals[0],guard),candidate_final=pixel_preservation(source,finals[1],guard))
        details=dict(baseline_request=requests[0],candidate_request=requests[1],assistant_review=review,known_neighbor_pixel_preservation=metrics)
        (a.out_root/(Path(name).stem+'_comparison.json')).write_text(json.dumps(details,ensure_ascii=False,indent=2))
        cards.append(f'<article id="case-{name[:3]}"><h2>{html.escape(name)}</h2>'
            f'<p>{html.escape(row["editing_instruction"])}</p><h3>FINAL：原图 / 旧提示 / 新提示</h3>'
            +panels[0]+f'<pre>{html.escape(json.dumps(review,ensure_ascii=False,indent=2))}</pre>'
            +'<details><summary>RAW 对照与完整实际 prompt</summary>'+panels[1]
            +'<pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2))+'</pre></details></article>')
    for offset in range(0,len(sheets),3):
        group=sheets[offset:offset+3];contact=Image.new('RGB',(1008,552*len(group)),'white')
        for index,sheet in enumerate(group):contact.paste(sheet.resize((1008,552)),(0,index*552))
        contact.save(a.out_root/f'contact_{offset:03d}.jpg',quality=94)
    summary=dict(cases=len(rows),reviewed=len(reviews),baseline_pass=sum(r['baseline']=='pass' for r in reviews.values()),
        candidate_pass=sum(r['candidate']=='pass' for r in reviews.values()),
        regressions=[r['image'] for r in reviews.values() if r['baseline']=='pass' and r['candidate']=='fail'],
        improvements=[r['image'] for r in reviews.values() if r['baseline']=='fail' and r['candidate']=='pass'])
    (a.out_root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (a.out_root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Removal execution comparison</title>'
        '<style>body{max-width:1300px;margin:24px auto;font-family:system-ui}img{width:100%}pre{white-space:pre-wrap}article{border:1px solid #bbb;margin:24px 0;padding:14px}</style>'
        '<h1>删除出图：同计划、同mask、同seed、固定40步的提示对照</h1>'
        f'<p>左：原图；中：{html.escape(policies[0])}；右：{html.escape(policies[1])}。上排全图，下排同坐标局部，原图局部白线是mask。'
        '两组final均用adaptive-remove-v2合成。展开可见raw及实际prompt；所有图片内嵌。'
        'Assistant review是Codex逐图判断，不是pipeline模型回复，也不是独立人工金标准。本实验没有成图审核或指令改写调用。</p>'
        '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
