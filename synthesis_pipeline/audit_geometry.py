"""Conservative pixel-coordinate checks, separate from subjective image quality."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from synthesis_pipeline.audit_edit_pairs import mask_array, write_jsonl


def addition_contact(target, inserted):
    """Check contact only, not containment: a valid addition can extend off its host.

    Four native pixels (or 0.3% of the diagonal on larger images) tolerate SAM
    contour error. Overlap cannot establish semantic ownership in occluded scenes.
    Missing silhouettes are unmeasured, never invented as successful checks.
    """
    target=np.asarray(target,dtype=bool);inserted=np.asarray(inserted,dtype=bool)
    if target.ndim!=2 or target.shape!=inserted.shape:
        raise ValueError('Contact masks must share the original image coordinates')
    tolerance=max(4.,float(np.hypot(*target.shape))*.003)
    if not target.any() or not inserted.any():
        return dict(status='unmeasured',reason='empty target or insertion silhouette',tolerance_px=tolerance)
    distances=cv2.distanceTransform((~target).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    gap=float(distances[inserted].min())
    return dict(status='pass' if gap<=tolerance else 'fail',min_gap_px=gap,
                tolerance_px=tolerance,overlap_fraction=float((target&inserted).sum()/inserted.sum()),
                reason='Actual inserted silhouette must contact its designated host; overlap alone does not prove correct instance')


def row_contact(row, size):
    if row['task_type']!='add' or not row.get('added_mask'):
        return None
    return addition_contact(mask_array(size,row['mask']),mask_array(size,row['added_mask']))


def apply_contact_gate(record, evidence):
    """Never replace a prior rejection or claim that scope failure is bad pixels."""
    result={**record,'geometric_contact':evidence}
    if evidence and evidence['status']=='fail' and record['quality']=='pass':
        result.update(pre_geometry_decision=record['decision'],decision='reject_geometric_anchor',quality='fail')
    return result


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description='Apply contact guard to immutable completed audits; no extra model calls')
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--audit-root',type=Path,required=True)
    p.add_argument('--out-root',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter();a.out_root.mkdir(parents=True,exist_ok=False)
    rows={r['image']:r for r in read(a.data_root/'annotations.jsonl')}
    records=[]
    for record in read(a.audit_root/'edit_audit.jsonl'):
        row=rows[record['image']]
        with Image.open(a.data_root/'sources'/row['source_image']) as image:
            evidence=row_contact(row,image.size)
        records.append(apply_contact_gate(record,evidence))
    accepted={r['image'] for r in records if r['quality']=='pass'}
    for stage in ['quality','reconstruction','verification']:
        (a.out_root/f'{stage}.jsonl').symlink_to((a.audit_root/f'{stage}.jsonl').resolve())
    write_jsonl(a.out_root/'edit_audit.jsonl',records)
    write_jsonl(a.out_root/'model_accepted_annotations.jsonl',[
        r for r in read(a.audit_root/'model_accepted_annotations.jsonl') if r['image'] in accepted])
    summary=dict(cases=len(records),accepted=len(accepted),geometric_rejections=[
        r['image'] for r in records if r['decision']=='reject_geometric_anchor'],
        model_calls=0,wall_seconds=time.perf_counter()-started,reused_audit=str(a.audit_root.resolve()))
    (a.out_root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
