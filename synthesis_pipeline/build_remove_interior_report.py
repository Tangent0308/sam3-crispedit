"""Publish reviewed composition-only comparisons, without inferring verdicts."""
import argparse
import base64
from collections import Counter
import html
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    args = parser.parse_args()
    reviews = read(args.reviews)
    cards = []
    counts = {}
    for cohort in dict.fromkeys(row['cohort'] for row in reviews):
        rows = {row['image']: row for row in read(args.root / cohort / 'annotations.jsonl')}
        selected = [review for review in reviews if review['cohort'] == cohort]
        if set(rows) != {review['image'] for review in selected} or len(rows) != len(selected):
            raise ValueError(f'Incomplete or duplicate review: {cohort}')
        counts[cohort] = dict(cases=len(rows), baseline_pass=sum(r['baseline']=='pass' for r in selected),
            candidate_pass=sum(r['candidate']=='pass' for r in selected),
            transitions=dict(Counter(r['baseline']+' -> '+r['candidate'] for r in selected)))
        for review in selected:
            row = rows[review['image']]
            picture = args.root / cohort / 'report' / (Path(row['image']).stem+'.jpg')
            details = dict(assistant_review=review, instruction=row['editing_instruction'],
                support_evidence=row['remove_composition'])
            encoded = base64.b64encode(picture.read_bytes()).decode()
            cards.append(f'<article><h2>{html.escape(cohort+" / "+row["image"])}</h2>'
                f'<p>{html.escape(row["editing_instruction"])}</p>'
                f'<p>Assistant review: {review["baseline"]} → {review["candidate"]}；'
                f'{html.escape(review["reason"])}</p>'
                f'<img loading="lazy" src="data:image/jpeg;base64,{encoded}">'
                f'<details><summary>完整记录</summary><pre>{html.escape(json.dumps(details,ensure_ascii=False,indent=2))}</pre></details></article>')
    out = args.root / 'reviewed_report'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(counts, ensure_ascii=False, indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Removal composition v2</title>'
        '<style>body{max-width:1500px;margin:24px auto;font-family:system-ui}img{width:100%}pre{white-space:pre-wrap}article{border:1px solid #bbb;padding:14px;margin:24px 0}</style>'
        '<h1>删除写回：类别无关修复的回归与新源验证</h1>'
        '<p>每条从左到右：原图 / 同一次生成的 raw / 当前 v1 合成 / 候选 v2 合成。上排全图，下排同坐标放大。'
        '图像全部内嵌。Assistant review 是 Codex 逐图判断，不是 pipeline 调用模型的回复，也不是独立人工金标准。'
        '本轮没有新增成图审核或指令改写调用；比较双方共享相同原指令、mask、raw 和 40 步出图。'
        '所有 case 一律使用同一策略，没有按 case 挑版本。旧回归与新源数据分开统计。</p>'
        '<pre>'+html.escape(json.dumps(counts,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    print(json.dumps(counts, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
