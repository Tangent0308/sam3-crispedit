"""Package explicitly assistant-reviewed labels, never automatic model passes.

This is a curated release across cohorts, not an unbiased pipeline evaluation.
The complete cohort reports remain the source of acceptance-rate denominators.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_review(row):
    review=row.get('assistant_review') or {}
    if review.get('visual_quality')!='pass':
        raise ValueError('Missing explicit assistant visual approval')
    status=row.get('verification')
    if status=='assistant_reviewed_original_instruction':
        valid=review.get('instruction_match')=='pass'
    elif status=='assistant_reviewed_rewrite':
        valid=(review.get('rewrite_match')=='pass'
               and review.get('reviewed_instruction')==row.get('editing_instruction'))
    else:
        valid=False
    if not valid:
        raise ValueError('Unverified or stale instruction review')
    if row.get('new_instruction',row['editing_instruction'])!=row['editing_instruction']:
        raise ValueError('Conflicting active instruction aliases')
    return review


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',action='append',required=True,
                   help='evaluation_directory=data_directory=audit_directory')
    p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args()
    # Resolve and validate everything before creating the new release.
    rows=[];reviews=[];files={};stages={k:[] for k in ('quality','reconstruction','verification','edit_audit')}
    seen=set();cohorts=[]
    for spec in a.cohort:
        evaluation,data,audit=map(Path,spec.split('='))
        selected=read(evaluation/'assistant_verified_annotations.jsonl')
        stage_rows={k:{r['image']:r for r in read(audit/(k+'.jsonl'))} for k in stages}
        for row in selected:
            name=row['image'];review=validate_review(row)
            if name in seen:raise ValueError(f'Duplicate region: {name}')
            seen.add(name)
            for folder,filename in [('sources',row['source_image']),('edited',name)]:
                if Path(filename).name!=filename:raise ValueError('Expected a plain artifact filename')
                src=(data/folder/filename).resolve(strict=True);key=(folder,filename)
                if key in files and files[key]!=src:raise ValueError(f'Conflicting image: {key}')
                files[key]=src
            with Image.open(files['sources',row['source_image']]) as src, Image.open(files['edited',name]) as dst:
                if src.size!=dst.size:raise ValueError(f'Unaligned pair: {name}')
            rows.append({**row,'reviewed_cohort':str(data.parent.resolve()),
                         'reviewed_evaluation_root':str(evaluation.resolve()),
                         'reviewed_audit_root':str(audit.resolve())})
            reviews.append(dict(image=name,visual_quality='pass',instruction_match='pass',
                reason=review['reason']+' '+review.get('rewrite_reason',''),
                released_instruction=row['editing_instruction'],
                release_verification=row['verification'],original_review=review))
            for key in stages:
                if name in stage_rows[key]:stages[key].append(stage_rows[key][name])
        cohorts.append(dict(data_root=str(data),evaluation_root=str(evaluation),audit_root=str(audit),selected=len(selected)))
    a.out_root.mkdir(parents=True,exist_ok=False)
    for directory in ('sources','edited'):(a.out_root/directory).mkdir()
    for (folder,name),src in files.items():(a.out_root/folder/name).symlink_to(src)
    def write(name,values):
        (a.out_root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    write('annotations.jsonl',rows);write('release_reviews.jsonl',reviews)
    for key,values in stages.items():write(key+'.jsonl',values)
    summary=dict(cases=len(rows),types=dict(Counter(r['task_type'] for r in rows)),cohorts=cohorts,
        verification='assistant visually reviewed; not a human-annotated gold standard or automatic acceptance benchmark',
        note='Curated subset across cohorts. Full reports retain all failures. Model candidates shown in the gallery may differ from the released instruction.')
    (a.out_root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
