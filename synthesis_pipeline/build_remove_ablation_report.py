"""One portable entry for fixed-cohort removal quality ablations and reviews."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
from PIL import Image
from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.experiment_remove_support import comparison,read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reviews',type=Path)
    p.add_argument('--adopted-only',action='store_true',help='Present only baseline and the adopted support repair')
    a=p.parse_args();out=a.root/('adopted_report' if a.adopted_only else 'comparison_report');out.mkdir(exist_ok=True)
    reviews={r['image']:r for r in read(a.reviews)} if a.reviews else {}
    variant='context_grounded_v4_qwen21'
    configs=[('baseline',a.root/'baseline'/variant),('adaptive',a.root/'baseline/adaptive'),('shadow_adaptive',a.root/'shadow/adaptive')]
    if a.adopted_only:configs=configs[:2]
    audits={key:{x['image']:x for x in read(root/'audit/quality.jsonl')} if (root/'audit/quality.jsonl').exists() else {} for key,root in configs}
    rows=read(a.root/'regions/annotations.jsonl');cards=[];sheets=[]
    for row in rows:
        name=row['image'];source=Image.open(a.root/'regions/sources'/row['source_image']).convert('RGB')
        outputs=[Image.open(root/'edited'/name).convert('RGB') for _,root in configs]
        mask=mask_array(source.size,row['mask'])
        if a.adopted_only:
            from synthesis_pipeline.build_qwen21_quality_report import panel
            sheet=panel(source,*outputs,mask,name,row['editing_instruction'])
        else:
            sheet=comparison(source,*outputs,mask,name,
                ['SOURCE / clean','BASELINE / typed-v1','ADAPTIVE / same raw','SHADOW + ADAPTIVE / new raw'])
        sheets.append(sheet);sheet.save(out/(Path(name).stem+'.jpg'),quality=94)
        buf=io.BytesIO();sheet.save(buf,format='JPEG',quality=90)
        requests={key:json.loads((a.root/key/variant/'diagnostics'/Path(name).stem/'generation_request.json').read_text()) for key in (['baseline'] if a.adopted_only else ['baseline','shadow'])}
        e=lambda x:html.escape(json.dumps(x,ensure_ascii=False,indent=2))
        cards.append(f'<article id="case-{name[:3]}"><h2>{html.escape(name)}</h2><p>{html.escape(row["editing_instruction"])}</p>'
            f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'
            f'<h3>Assistant逐图复核</h3><pre>{e(reviews.get(name,"not reviewed"))}</pre>'
            f'<details><summary>实际生成prompt与全部模型画质审核回复</summary><pre>{e(dict(requests=requests,model_quality={key:table.get(name) for key,table in audits.items()}))}</pre></details></article>')
    for offset in range(0,len(sheets),3):
        group=sheets[offset:offset+3];contact=Image.new('RGB',(1152,560*len(group)),'white')
        for i,sheet in enumerate(group):contact.paste(sheet.resize((1152,560)),(0,i*560))
        contact.save(out/f'contact_{offset:03d}.jpg',quality=94)
    summary=dict(executable=len(rows),reviewed=len(reviews),steps=40,
        assistant_pass={key:sum(r.get(key)=='pass' for r in reviews.values()) for key,_ in configs},
        model_quality={key:dict(reviewed=len(t),passed=sum(r.get('quality')=='pass' for r in t.values())) for key,t in audits.items()})
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    intro=('<h1>本轮采用版本：typed-v1 + adaptive-remove-v1，固定40步</h1>'
        '<p>左：原图；中：旧写回；右：本轮采用的新写回。两种结果复用完全相同的模型raw，不是多次生成后挑图。上排干净全图，下排同坐标局部；原图局部的白色轮廓是数据集mask。</p>'
        '<p>被否决的阴影提示不在主图中展示。逐图理由仍记录其回归问题，完整消融在同级comparison_report。</p>') if a.adopted_only else (
        '<h1>新源删除实验：固定40步</h1><p>左到右：原图、原版typed-v1、同一raw仅改写回、新阴影提示+新写回。上排完整场景，下排同坐标局部。不是按case挑选最优版本，全部冻结计划均展示。</p>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Removal quality ablation</title>'
        '<style>body{max-width:1500px;margin:24px auto;font-family:system-ui}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{border:1px solid #bbb;padding:12px;margin:24px 0}</style>'
        +intro+'<p>Assistant review由Codex看图；模型回复来自独立27B画质检查。</p>'
        '<p>该批是质量开发/确认实验；模型初审不等于最终入库，不包含指令重写。原始mask未改变，扩展的只是后合成支持区。</p>'
        f'<pre>{html.escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>'+''.join(cards))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
