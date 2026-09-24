"""Render a single run's rejected cases with aligned generation diagnostics."""

import argparse
import base64
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from synthesis_pipeline.audit_edit_pairs import mask_array
from synthesis_pipeline.merge_verified_edits import comparison_sheet, jpeg_bytes


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def uri(picture):
    return 'data:image/jpeg;base64,' + base64.b64encode(jpeg_bytes(picture)).decode('ascii')


def diagnostic_sheet(final, mask, directory, request):
    bbox = tuple(request['crop_bbox'])
    x1, y1, x2, y2 = bbox
    protected = directory / 'protected_instances.png'
    panels = [
        ('1. Source crop (actual model input)', Image.open(directory / 'source_crop.png').convert('RGB')),
        ('2. Raw generation (before composition)', Image.open(directory / 'raw_edited_crop.png').convert('RGB')),
        ('3. Final crop (after composition)', final.crop(bbox)),
        ('4. Target mask (white = annotated target)', Image.fromarray(mask[y1:y2, x1:x2].astype('uint8') * 255).convert('RGB')),
        ('5. Composition alpha (white = use raw)', Image.open(directory / 'composition_alpha.png').convert('RGB')),
        ('6. Protected neighbors (white = protected)', Image.open(protected).convert('RGB').crop(bbox)
         if protected.exists() else Image.new('RGB', (x2-x1, y2-y1))),
    ]
    width, height, bar = 520, 580, 36
    sheet = Image.new('RGB', (width * 3, height * 2), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    for i, (title, panel) in enumerate(panels):
        col, row = i % 3, i // 3
        tile = ImageOps.contain(panel, (width - 12, height - bar - 12))
        x, y = col * width, row * height
        draw.text((x + 8, y + 7), title, fill='black', font=font)
        sheet.paste(tile, (x + (width - tile.width) // 2, y + bar + (height - bar - tile.height) // 2))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--review-jsonl', type=Path, required=True)
    parser.add_argument('--variant', default='replanned')
    parser.add_argument('--analysis-jsonl', type=Path)
    parser.add_argument('--out-root', type=Path, required=True)
    args = parser.parse_args()
    reviews = {r['image']: r for r in read(args.review_jsonl) if r.get('variant') == args.variant}
    analysis = {r['image']: r for r in read(args.analysis_jsonl)} if args.analysis_jsonl else {}
    rows = read(args.run_root / 'annotations.jsonl')
    args.out_root.mkdir(parents=True, exist_ok=True)
    e = lambda value: html.escape(str(value), quote=True)
    cards, records = [], []
    for row in rows:
        review = reviews.get(row['image'])
        if not review or (review.get('visual_quality') == 'pass' and review.get('instruction_match') == 'pass'):
            continue
        source_path = args.data_root / 'sources' / row['source_image']
        edited_path = args.run_root / 'edited' / row['image']
        source = Image.open(source_path).convert('RGB')
        final = Image.open(edited_path).convert('RGB')
        mask = mask_array(source.size, row['mask'])
        directory = args.run_root / 'diagnostics' / Path(row['image']).stem
        request = json.loads((directory / 'generation_request.json').read_text())
        sheet = comparison_sheet(source_path, edited_path, row['mask'])
        diagnostic = diagnostic_sheet(final, mask, directory, request)
        stem = Path(row['image']).stem
        (args.out_root / (stem + '_comparison.jpg')).write_bytes(jpeg_bytes(sheet))
        (args.out_root / (stem + '_diagnostic.jpg')).write_bytes(jpeg_bytes(diagnostic))
        detail = analysis.get(row['image'], {})
        description = ''.join(f'<p><strong>{e(label)}：</strong>{e(detail[key])}</p>'
            for key, label in [('category', '问题类别'), ('observation', '逐图观察'),
                               ('raw_evidence', '中间结果证据'), ('cause', '失败原因'),
                               ('optimization', '优化建议'), ('disposition', '当前样本处理')]
            if key in detail)
        category = detail.get('group', 'pending')
        cards.append(f'<article id="case-{e(stem[:3])}" data-group="{e(category)}">'
            f'<h2>{e(row["image"])}</h2><p class="instruction">{e(row["editing_instruction"])}</p>'
            f'{description}<h3>完整场景与局部对比：左原图，右最终结果</h3>'
            f'<img loading="lazy" alt="原图与最终编辑图" src="{uri(sheet)}">'
            f'<p><a href="{e(stem)}_comparison.jpg">打开独立对比 JPG</a></p>'
            f'<details><summary>查看实际输入、回贴前生成图、回贴后结果及可写范围</summary>'
            f'<p>上排：干净输入 crop / 模型 raw 输出 / 最终 crop。下排：原 target mask / 合成 alpha / 保护的邻居。'
            f'各列使用相同空间范围；alpha 白色采用 raw，黑色采用原图，灰色是混合。</p>'
            f'<img loading="lazy" alt="六格生成诊断" src="{uri(diagnostic)}">'
            f'<a href="{e(stem)}_diagnostic.jpg">打开独立诊断 JPG</a></details>'
            f'<details><summary>本次生成的完整 prompt 与参数</summary><pre>{e(json.dumps(request,ensure_ascii=False,indent=2))}</pre></details>'
            f'<details><summary>上一轮复核记录（本轮分析如有修正，以正文为准）</summary><pre>{e(json.dumps(review,ensure_ascii=False,indent=2))}</pre></details></article>')
        records.append(dict(image=row['image'], instruction=row['editing_instruction'],
            edited_path=str(edited_path), analysis=detail, previous_review=review,
            generation_request=request))
    counts = Counter(record['analysis'].get('group', 'pending') for record in records)
    summary = (f'<p>本轮分类：画面 / 空间缺陷 {counts["visual"]} 张；错位导致目标变化不足 {counts["weak"]} 张；'
               f'类型定义问题 {counts["taxonomy"]} 张。</p>') if analysis else ''
    (args.out_root / 'index.html').write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>当前单一版本：出图 bad case 诊断</title><style>'
        'body{font-family:system-ui,sans-serif;max-width:1420px;margin:24px auto;padding:0 18px;background:#f4f5f7;color:#20242c}'
        'article{background:white;border:1px solid #ddd;border-radius:10px;padding:20px;margin:24px 0}'
        'h2{font-size:1.1em;overflow-wrap:anywhere}h3{font-size:1em}p{line-height:1.7}.instruction{background:#eff3fa;padding:12px}'
        'img{display:block;width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}'
        'summary,button{cursor:pointer}details{margin:14px 0}nav{position:sticky;top:0;background:#f4f5f7;padding:10px;z-index:1}'
        'button{padding:8px;margin:3px}</style><h1>v6 指令规划 + context_guarded_v2：出图 bad case 诊断</h1>'
        f'<p>只检查上一页同一批 14 张中的 {len(records)} 张未通过样本。所有图片内嵌，另附独立 JPG。'
        '逐图分析由 assistant 完成；原因分为可直接确认的事实与待验证的机制推断。</p>'
        '<p>本轮重新区分画面/空间缺陷、编辑变化不足、编辑类型定义问题；后两者不能统称为生成画质失败。'
        '当前页面为诊断结果，优化建议尚未通过新出图实验验证。</p>'
        + summary +
        '<nav><button onclick="showGroup(\'all\')">全部</button>'
        '<button onclick="showGroup(\'visual\')">画面 / 空间缺陷</button>'
        '<button onclick="showGroup(\'weak\')">错位 / 变化不足</button>'
        '<button onclick="showGroup(\'taxonomy\')">类型定义</button></nav>'
        + ''.join(cards)
        + '<script>function showGroup(g){for(const el of document.querySelectorAll("article")){el.hidden=g!=="all"&&el.dataset.group!==g;}}</script></html>',
        encoding='utf-8')
    (args.out_root / 'diagnosis_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    print(json.dumps(dict(cases=len(records), report=str(args.out_root/'index.html'))))


if __name__ == '__main__':
    main()
