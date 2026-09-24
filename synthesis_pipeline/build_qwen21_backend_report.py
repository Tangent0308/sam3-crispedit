"""Self-contained matched-backend comparison; manual judgments are separate."""
import argparse
import base64
import html
import io
import json
from pathlib import Path
import statistics

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from synthesis_pipeline.audit_edit_pairs import mask_array

VARIANT = 'context_grounded_v4_qwen21'


def image_uri(image):
    """Embed a preview-sized JPEG, keeping standalone HTML practical to open."""
    out = io.BytesIO()
    image = image.convert('RGB')
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    image.save(out, 'JPEG', quality=82, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()


def outlined_source(source, row):
    mask = mask_array(source.size, row['mask']).astype(np.uint8)
    arr = np.asarray(source).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, contours, -1, (0, 0, 0), 4)
    cv2.drawContours(arr, contours, -1, (255, 255, 0), 2)
    return Image.fromarray(arr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reviews', type=Path)
    a = p.parse_args()
    roots = {k: a.root / k / VARIANT for k in ('diffusers', 'vllm-omni')}
    rows = [json.loads(s) for s in (roots['diffusers'] / 'annotations.jsonl').read_text().splitlines()]
    reviews = json.loads(a.reviews.read_text()) if a.reviews else {}
    timings = {}
    stats = {}
    for key, root in roots.items():
        records = [json.loads(s) for s in (root / 'timing_shard0.jsonl').read_text().splitlines()]
        timings[key] = {r['image']: r for r in records}
        secs = [r['seconds'] for r in records]
        summary = json.loads((root / 'summary_shard0.json').read_text())
        stats[key] = dict(count=len(secs), mean_seconds=statistics.mean(secs),
                          median_seconds=statistics.median(secs),
                          mean_excluding_first=statistics.mean(secs[1:]) if len(secs)>1 else None,
                          sum_edit_seconds=sum(secs), model_load_seconds=summary['model_load_seconds'])
    stats['speedup_mean'] = stats['diffusers']['mean_seconds'] / stats['vllm-omni']['mean_seconds']
    cards, sheets, validations, pixel_comparisons = [], [], [], []
    review_counts = {k: {} for k in roots}
    for row in rows:
        name = row['image']; stem = Path(name).stem
        source = Image.open(roots['diffusers'] / 'sources' / row['source_image']).convert('RGB')
        figures = [outlined_source(source, row)]
        requests = {k: json.loads((r / 'diagnostics' / stem / 'generation_request.json').read_text())
                    for k,r in roots.items()}
        d, o = requests['diffusers'], requests['vllm-omni']
        check = dict(image=name, equal_prompt=d['prompt']==o['prompt'],
                     equal_size=d['model_output_size']==o['model_output_size'],
                     regional_anchor_both=d['regional_denoising'] and o['regional_denoising'])
        for filename in ('source_crop.png', 'editable_tokens.png', 'protected_instances.png'):
            paths = [r/'diagnostics'/stem/filename for r in roots.values()]
            check['equal_'+filename] = (not any(x.exists() for x in paths) or
                (all(x.exists() for x in paths) and np.array_equal(np.asarray(Image.open(paths[0])),np.asarray(Image.open(paths[1])))))
        if not all(v for k,v in check.items() if k!='image'):
            raise ValueError(f'Unmatched comparison: {check}')
        validations.append(check)
        for k, r in roots.items():
            figures.append(Image.open(r/'edited'/name).convert('RGB'))
        full_delta=np.abs(np.asarray(figures[1],dtype=np.float32)-np.asarray(figures[2],dtype=np.float32))
        crops=[np.asarray(Image.open(r/'diagnostics'/stem/'raw_edited_crop.png'),dtype=np.float32) for r in roots.values()]
        pixel_comparisons.append(dict(image=name, final_rgb_mae_255=float(full_delta.mean()),
                                      raw_crop_rgb_mae_255=float(np.abs(crops[0]-crops[1]).mean())))
        labels = ['SOURCE (yellow = target)', 'Diffusers', 'vLLM-Omni']
        visuals = ''.join(f'<figure><figcaption>{label}</figcaption><img src="{image_uri(im)}"></figure>'
                          for label,im in zip(labels,figures))
        judgments = reviews.get(name, {})
        for k in roots:
            v = judgments.get(k, {}).get('verdict', 'not_reviewed')
            review_counts[k][v] = review_counts[k].get(v, 0) + 1
        why = ''.join(f'<p><b>{html.escape(k)}</b>: {html.escape(str(judgments.get(k,"待人工复核")))}</p>' for k in roots)
        cards.append(f'<section><h2>{html.escape(name)}</h2><p>{html.escape(row["editing_instruction"])}</p>'
          f'<div class="images">{visuals}</div><p>Diffusers {timings["diffusers"][name]["seconds"]:.2f}s · '
          f'Omni {timings["vllm-omni"][name]["seconds"]:.2f}s · 输出 {d["model_output_size"]} · 40 steps</p>'
          f'<h3>Assistant 人工看图判断（不是 pipeline 审核模型输出）</h3>{why}'
          f'<details><summary>相同的实际编辑 prompt / 对齐检查</summary><pre>{html.escape(d["prompt"])}</pre>'
          f'<pre>{html.escape(json.dumps(check,ensure_ascii=False,indent=2))}</pre></details></section>')
        panel = Image.new('RGB',(1500,440),'#eee')
        draw = ImageDraw.Draw(panel)
        draw.text((8,5),name+' | '+row['editing_instruction'], fill='black')
        for index, im in enumerate(figures):
            draw.text((index*500+8,25),labels[index],fill='black')
            thumb = ImageOps.contain(im,(492,380))
            panel.paste(thumb,(index*500+(500-thumb.width)//2,50+(380-thumb.height)//2))
        sheets.append(panel)
    stats['assistant_review_counts'] = review_counts
    stats['pixel_comparisons'] = pixel_comparisons
    (a.root/'comparison_summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False))
    (a.root/'matched_inputs.json').write_text(json.dumps(validations,indent=2))
    report = ('<!doctype html><meta charset="utf-8"><title>Qwen 2.1 backend A/B</title>'
      '<style>body{font:16px system-ui;margin:24px;background:#eee}section{background:white;padding:16px;margin:20px 0}'
      '.images{display:flex;gap:8px}figure{width:33.33%;margin:0}img{width:100%}pre{white-space:pre-wrap}'
      'figcaption{font-weight:bold}summary{cursor:pointer}</style><h1>Qwen-Image-2.1：仅替换推理后端</h1>'
      '<p>从左到右：原图（黄色轮廓只是展示mask）、Diffusers最终图、Omni最终图。模型实际输入仍是干净crop。'
      '图片内嵌，无需加载外部文件。浏览器放大可检查边缘。此批均为remove，不代表其余类型。</p>'
      '<p>相同冻结计划、mask、prompt、seed=0、40步、约1MP、逐步区域保护和最终融合。'
      '使用两个独立环境/同型号GPU；不同后端与算子数值可能导致不同图像。耗时为单例编辑阶段，'
      '包含crop/保护/推理/融合/保存，不含规划及审核；启动耗时另列，不以并行墙钟冒充单例速度。</p>'
      '<pre>'+html.escape(json.dumps(stats,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    (a.root/'index.html').write_text(report)
    for i in range(0,len(sheets),4):
        canvas=Image.new('RGB',(1500,440*len(sheets[i:i+4])),'white')
        for j,panel in enumerate(sheets[i:i+4]): canvas.paste(panel,(0,j*440))
        canvas.save(a.root/f'contact_{i//4:02}.jpg',quality=94)
    print(json.dumps(stats,indent=2,ensure_ascii=False))


if __name__ == '__main__': main()
