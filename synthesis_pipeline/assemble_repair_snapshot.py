"""Freeze explicit per-case experimental selections; never imply one automatic run."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--experiment-root',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    args=p.parse_args()
    selections=read(args.selection)
    if len({r['id'] for r in selections})!=len(selections):
        raise ValueError('Duplicate selected case')
    prepared=[]
    for choice in selections:
        root=args.experiment_root/choice['run']
        matches=[r for r in read(root/'annotations.jsonl') if int(r['image'].split('_')[0])==choice['id']]
        if len(matches)!=1:raise ValueError(f'Missing/ambiguous selection: {choice}')
        row=matches[0];image=root/'edited'/row['image']
        if not image.is_file():raise FileNotFoundError(image)
        row={**row,'selected_experiment':str(root.resolve()),'assistant_review':choice}
        prepared.append((row,image,choice))
    args.out_root.mkdir(parents=True,exist_ok=False)
    (args.out_root/'sources').symlink_to((args.data_root/'sources').resolve())
    (args.out_root/'edited').mkdir()
    rows=[];reviews=[];verified=[]
    for row,image,choice in prepared:
        (args.out_root/'edited'/row['image']).symlink_to(image.resolve())
        rows.append(row)
        reviews.append({**choice,'image':row['image'],'variant':'selected'})
        if choice['visual_quality']==choice['instruction_match']=='pass':verified.append(row)
    for filename,records in [('annotations.jsonl',rows),('assistant_review.jsonl',reviews),('assistant_verified_annotations.jsonl',verified)]:
        (args.out_root/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    (args.out_root/'summary.json').write_text(json.dumps(dict(cases=len(rows),assistant_verified=len(verified),
        provenance='Manually selected experimental regression snapshot, not automatic pipeline acceptance'),indent=2))
    print(json.dumps(dict(cases=len(rows),assistant_verified=len(verified))))


if __name__=='__main__':main()
