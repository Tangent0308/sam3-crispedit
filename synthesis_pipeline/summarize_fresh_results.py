"""Count a frozen cohort without dropping planning or segmentation failures."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def summarize(root, split):
    suffix='_27b' if split=='dev' else ''
    plans=read(root/f'{split}_plan{suffix}'/'annotations.jsonl')
    regions=read(root/f'{split}_regions{suffix}'/'annotations.jsonl')
    cohort=read(root/split/'annotations.jsonl')
    final=root/f'{split}_final'
    outputs=read(final/'annotations.jsonl')
    reviews={r['image']:r for r in read(root/f'{split}_final_reviews.jsonl')}
    audits=read(root/f'{split}_final_audit'/'balanced_crop'/'edit_audit.jsonl')
    good=[r for r in outputs if all(reviews.get(r['image'],{}).get(k)=='pass'
                                  for k in ('visual_quality','instruction_match'))]
    for row in good:
        row['verification']={'kind':'assistant_visual_and_original_instruction',
                             'reason':reviews[row['image']]['reason']}
        row['edited_path']=str((final/'edited'/row['image']).resolve())
        row['source_path']=str((root/f'{split}_regions{suffix}'/'sources'/row['source_image']).resolve())
    (final/'assistant_verified_annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in good))
    pairs=[(r,reviews[r['image']]) for r in audits if r['image'] in reviews]
    model=lambda r:(r.get('audit') or {}).get('visual_quality')
    stats=dict(input_regions=len(cohort),planned=len(plans),
               input_types=dict(Counter(r['task_type'] for r in cohort)),
               resolved_regions=sum(r.get('region_contract',{}).get('status')!='unresolved' for r in regions),
               generated=len(list((root/(f'{split}27_generation' if split=='dev' else 'holdout_generation')/'context_grounded_v4'/'edited').glob('*.png'))),
               composed=len(outputs),assistant_reviewed=len(reviews),
               assistant_visual_pass=sum(r['visual_quality']=='pass' for r in reviews.values()),
               assistant_original_instruction_and_quality_pass=len(good),
               verified_types=dict(Counter(r['task_type'] for r in good)),
               audit_compared=len(pairs),audit_visual_agreement=sum(model(a)==r['visual_quality'] for a,r in pairs),
               audit_false_pass=[a['image'] for a,r in pairs if model(a)=='pass' and r['visual_quality']=='fail'],
               audit_false_fail=[a['image'] for a,r in pairs if model(a)=='fail' and r['visual_quality']=='pass'],
               note='Assistant verified original instructions only; rewrite candidates are not automatically adopted.')
    (final/'cohort_summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--splits',default='dev,holdout');a=p.parse_args()
    result={s:summarize(a.root,s) for s in a.splits.split(',')}
    (a.root/'cohort_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    reviews=[r for s in a.splits.split(',') for r in read(a.root/f'{s}_final_reviews.jsonl')]
    (a.root/'all_final_reviews.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviews))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
