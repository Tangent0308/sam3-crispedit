"""Portable, non-curating report: retain every output, prompt and review reason."""

import argparse
import html
import json
from pathlib import Path
from synthesis_pipeline.merge_verified_edits import comparison_sheet, jpeg_bytes, preview_data_uri


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run', action='append', required=True, help='label=directory containing edited/')
    p.add_argument('--audit', action='append', default=[], help='label=edit_audit.jsonl or rewrites.jsonl')
    p.add_argument('--review-jsonl', type=Path)
    p.add_argument('--only-reviewed-passes', action='store_true',
                   help='Only show images manually reviewed as both visually sound and instruction-matched')
    p.add_argument('--display-label', help='Human-readable name for the selected run in the gallery')
    p.add_argument('--curated-release',action='store_true',help='Clearly label a cross-cohort reviewed subset, not an unbiased full run')
    p.add_argument('--cohort-summary',type=Path,help='Show full input denominator, including upstream rejections')
    p.add_argument('--ids',default='',help='Optional comma-separated numeric case IDs for diagnostic subsets')
    p.add_argument('--out-root', type=Path, required=True)
    args = p.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    reviews = read(args.review_jsonl) if args.review_jsonl else []
    audits = [(label, read(Path(path))) for label, path in (s.split('=', 1) for s in args.audit)]
    e = lambda x: html.escape(str(x), quote=True)
    cards, records = [], []
    available = 0
    for label, folder in (s.split('=', 1) for s in args.run):
        root = Path(folder)
        rows = read(root / 'annotations.jsonl') or read(args.data_root / 'annotations.jsonl')
        if args.ids:
            wanted={int(x) for x in args.ids.split(',')}
            rows=[r for r in rows if int(r['image'].split('_')[0]) in wanted]
        for row in rows:
            # An unchanged composition is still a generated case, possibly a
            # no-op. Never silently hide it from a full-cohort quality report.
            image = root / 'edited' / row['image']
            if not image.exists():
                continue
            available += 1
            review = [r for r in reviews if r['image'] == row['image'] and r.get('variant', label) == label]
            if args.only_reviewed_passes and not any(
                r.get('visual_quality') == 'pass' and r.get('instruction_match') == 'pass'
                for r in review
            ):
                continue
            sheet = comparison_sheet(args.data_root / 'sources' / row['source_image'], image, row['mask'])
            name = f'{label}_{Path(row["image"]).stem}.jpg'
            (args.out_root / name).write_bytes(jpeg_bytes(sheet))
            # A case ID can have several generated versions. Never display one
            # version's verdict under another version's photograph.
            responses = {title: [r for r in data if r['image'] == row['image']
                and (not r.get('edited_path') or Path(r['edited_path']).resolve() == image.resolve())]
                for title, data in audits if title.split(':', 1)[0] == label}
            responses = {k: v for k, v in responses.items() if v}
            model_summaries = []
            for title, entries in responses.items():
                for entry in entries:
                    parsed = entry.get('parsed') or entry.get('audit') or {}
                    reason = parsed.get('reason') or entry.get('policy_error') or ''
                    decision = entry.get('decision') or entry.get('quality') or parsed.get('quality') or entry.get('status') or ''
                    instruction = parsed.get('editing_instruction') or (entry.get('candidate') or {}).get('editing_instruction') or ''
                    geometry=entry.get('geometric_contact')
                    model_summaries.append(f'<p><strong>{e(title)} / {e(decision)}</strong>: {e(reason)}'
                        + (f'<br>新增物宿主几何检查：{e(json.dumps(geometry,ensure_ascii=False))}' if geometry else '')
                        + (f'<br>重建指令：{e(instruction)}' if instruction else '') + '</p>')
            details = json.dumps(dict(annotation=row, assistant_review=review, model_responses=responses), ensure_ascii=False, indent=2)
            cards.append(f'<article><h2>{e(args.display_label or label)} / {e(row["image"])}</h2>'
                         f'<p>{e(row["editing_instruction"])}</p>'
                         f'<img loading="lazy" src="{preview_data_uri(sheet)}">'
                         f'<p>逐图复核：{e(json.dumps(review, ensure_ascii=False)) if review else "尚未复核；不能作为通过样本"}</p>'
                         + ''.join(model_summaries) +
                         f'<details><summary>完整模型理由、输入 prompt、原始回复及元数据</summary><pre>{e(details)}</pre></details>'
                         f'<a href="{e(name)}">独立 JPG</a></article>')
            records.append(dict(variant=label, image=row['image'], edited_path=str(image.resolve()), review=review))
    # Put alternative versions of the SAME case next to each other.
    ordered = sorted(zip(records, cards), key=lambda item: item[0]['image'])
    records = [item[0] for item in ordered]
    cards = [item[1] for item in ordered]
    if args.curated_release:
        headline=f'本轮逐图复核后的候选合集：{len(records)} 条'
        description=('这是从多个新源批次中经 assistant 逐图、逐标签复核后筛选的合集，不是自动审核通过率，也不是人工金标准。'
                     '每条顶部句子为最终导出指令；下方保留模型当时的候选及原始回复，可能与导出指令不同。'
                     '逐图复核的顶层 instruction_match 指导出指令；original_review 保留改写前判断。'
                     '左原图、右结果，上方全图、下方局部；图片内嵌，独立 JPG 保留更高分辨率。')
        variants='<p>完整失败分母与各轮未筛选报告保留在原批次目录和统一迭代文档中；这里仅展示明确复核通过的候选。</p>'
    elif args.only_reviewed_passes:
        headline = f'当前候选版本：{e(args.display_label or args.run[0].split("=", 1)[0])}'
        description = (f'本次共生成 {available} 张，逐图复核后仅展示其中画质与原指令均通过的 {len(records)} 张。'
                       '这是开发样本的筛选展示，不代表整条流水线的自动通过率。'
                       '每张对比左侧原图、右侧结果；上方完整场景，下方同坐标局部。图片内嵌，无需外部路径。')
        variants = ''
    else:
        headline = '逐步修复回归：所有输出，不仅展示成功案例'
        description = ('每张对比：左侧原图、右侧结果；上方完整场景，下方同坐标局部。图片内嵌，无需访问外部路径。'
                       '模型 pass 不等于人工验收；这里的逐图复核由 assistant 完成。同一 case 的不同版本相邻排列。')
        variants = ('<p>版本与文件来源见每条元数据。保留所有生成结果，包括无效变化和各阶段拒绝项。'
                    '没有模型核验记录的版本仅做了 assistant 复核。</p>')
    cohort_title='合集来源与筛选统计（不是原始生成分母）' if args.curated_release else '完整输入批次统计（包含未出图的拒绝项）'
    cohort = ('<h2>'+cohort_title+'</h2><pre>' + e(args.cohort_summary.read_text()) + '</pre>') if args.cohort_summary else ''
    (args.out_root / 'index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>逐步修复回归</title>'
        '<style>body{font-family:system-ui;max-width:1100px;margin:24px auto}article{border:1px solid #bbb;padding:16px;margin:24px 0}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
        f'<h1>{headline}</h1><p>{description}</p>{variants}{cohort}'
        + ''.join(cards))
    (args.out_root / 'report_manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(json.dumps(dict(cases=len(records), generated=available, report=str(args.out_root / 'index.html'))))


if __name__ == '__main__':
    main()
