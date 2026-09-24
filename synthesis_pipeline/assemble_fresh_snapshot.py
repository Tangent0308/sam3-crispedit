"""Assemble one deterministic snapshot: semantic composition for add/replace.

No per-case version selection; missing segmentation is a recorded failure.
"""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def attach_remove_support(row, raw_root):
    """Keep dataset target separate from the actual late-write support."""
    if row['task_type']!='remove':return row
    diagnostic=raw_root/'diagnostics'/Path(row['image']).stem
    path=diagnostic/'generation_request.json'
    if not path.exists():return row
    request=json.loads(path.read_text())
    policy=request.get('remove_composition_policy','legacy')
    if policy=='legacy':return row
    import numpy as np
    from PIL import Image
    from synthesis_pipeline.refine_samtok_regions import encode_mask
    w,h=request['source_size'];x1,y1,x2,y2=request['crop_bbox']
    alpha=np.asarray(Image.open(diagnostic/'composition_alpha.png'))
    if alpha.shape!=(y2-y1,x2-x1):raise ValueError('Saved removal alpha shape mismatch')
    support=np.zeros((h,w),bool);support[y1:y2,x1:x2]=alpha>0
    return {**row,'edit_support_mask':encode_mask(support),
        'remove_composition':dict(policy=policy,source_mask_modified=False,
            support_semantics='pixel_write_support_not_target_segmentation',
            quality_status='requires_visual_audit')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--composed-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    args=p.parse_args()
    rows=read(args.raw_root/'annotations.jsonl')
    composed={r['image']:r for r in read(args.composed_root/'annotations.jsonl')}
    composition_summary=json.loads((args.composed_root/'summary.json').read_text())
    failure_by_image={r['image']:r for r in composition_summary.get('failures',[])}
    # Prevent partial runs from silently becoming a final snapshot.
    missing=[r['image'] for r in rows if not (args.raw_root/'edited'/r['image']).exists()]
    if missing:raise ValueError(f'Generation incomplete: {missing}')
    args.out_root.mkdir(parents=True,exist_ok=False)
    (args.out_root/'edited').mkdir()
    source_root=args.raw_root/'sources'
    if source_root.exists():
        (args.out_root/'sources').symlink_to(source_root.resolve())
    result=[];failures=[]
    rejected=[]
    for row in rows:
        row=attach_remove_support(row,args.raw_root)
        root=args.raw_root
        if row['task_type'] in {'add','replace'}:
            if row['image'] not in composed:
                failures.append({**failure_by_image.get(row['image'],dict(image=row['image'],reason='missing composed output')),
                                 'stage':'composition'})
                rejected.append(row)
                continue
            row=composed[row['image']];root=args.composed_root
        (args.out_root/'edited'/row['image']).symlink_to((root/'edited'/row['image']).resolve())
        result.append(row)
    (args.out_root/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result))
    summary=dict(input_generated=len(rows),assembled=len(result),failures=failures,
                 policy=composition_summary.get('policy','legacy'),
                 composition_methods={method:sum(r.get('composition_revision',{}).get('method','raw_passthrough')==method for r in result)
                     for method in sorted({r.get('composition_revision',{}).get('method','raw_passthrough') for r in result})})
    (args.out_root/'summary.json').write_text(json.dumps(summary,indent=2))
    # Diagnostic copies reference raw files, never masquerading as final edits.
    diagnostic=args.out_root/'composition_rejected'
    diagnostic.mkdir()
    (diagnostic/'edited').symlink_to((args.raw_root/'edited').resolve())
    if source_root.exists(): (diagnostic/'sources').symlink_to(source_root.resolve())
    (diagnostic/'annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rejected))
    print(json.dumps(summary))


if __name__=='__main__':main()
