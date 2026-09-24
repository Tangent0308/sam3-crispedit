"""Freeze an explicit regression round, retaining revisions and source provenance."""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--merge-root',type=Path,action='append',default=[])
    p.add_argument('--overrides',type=Path)
    p.add_argument('--ids',required=True)
    p.add_argument('--out-root',type=Path,required=True)
    args=p.parse_args()
    rows={r['image']:r for r in read(args.data_root/'annotations.jsonl')}
    for root in args.merge_root:
        rows.update({r['image']:r for r in read(root/'annotations.jsonl')})
    wanted={int(x) for x in args.ids.split(',')}
    rows=[r for r in rows.values() if int(r['image'].split('_')[0]) in wanted]
    if len(rows)!=len(wanted):raise ValueError('Missing requested cases')
    overrides={int(r['id']):r for r in read(args.overrides)} if args.overrides else {}
    for row in rows:
        override=overrides.get(int(row['image'].split('_')[0]),{})
        if override:
            row['editor_revision']={'old_instruction':row['editing_instruction'],
                'old_regional_instruction':row.get('new_instruction'),
                'provenance':override.get('provenance','explicit_regression_override')}
            row.update({k:v for k,v in override.items() if k not in {'id','provenance'}})
    args.out_root.mkdir(parents=True,exist_ok=False)
    (args.out_root/'sources').symlink_to((args.data_root/'sources').resolve())
    (args.out_root/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(rows,key=lambda r:r['image'])))
    (args.out_root/'input_annotations.jsonl').write_text((args.data_root/'input_annotations.jsonl').read_text())
    print(json.dumps(dict(cases=len(rows),root=str(args.out_root))))


if __name__=='__main__':main()
