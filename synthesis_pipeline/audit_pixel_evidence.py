"""Read-only pixel measurements for audit, never a quality verdict by themselves."""
import numpy as np
import cv2


def changed_surface_evidence(source, edited, mask, task_type):
    if source.size != edited.size or np.shape(mask) != (source.height, source.width):
        raise ValueError('Pixel evidence requires aligned source, output and mask')
    if task_type != 'attribute':
        return None
    a=np.asarray(source.convert('RGB')).astype(np.float32)/255
    b=np.asarray(edited.convert('RGB')).astype(np.float32)/255
    changed=mask.astype(bool) & (np.max(np.abs(a-b),axis=-1)>20/255)
    interior=cv2.erode(changed.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    if interior.sum()<64:
        return None
    la=cv2.cvtColor(a,cv2.COLOR_RGB2LAB)[...,0]
    lb=cv2.cvtColor(b,cv2.COLOR_RGB2LAB)[...,0]
    before=float(np.percentile(la[interior],90)-np.percentile(la[interior],10))
    after=float(np.percentile(lb[interior],90)-np.percentile(lb[interior],10))
    return dict(changed_interior_pixels=int(interior.sum()),
        before_luminance_range=round(before,3),after_luminance_range=round(after,3),
        range_ratio=round(after/before,3) if before>1e-6 else None,
        interpretation='measurement_only_not_a_verdict')


def evidence_prompt(evidence):
    if not evidence:
        return ''
    return ('\nRead-only pixel check on the SAME changed interior pixels (excluding mask edges): '
        f"BEFORE luminance p90-p10={evidence['before_luminance_range']:.2f}, "
        f"AFTER={evidence['after_luminance_range']:.2f}, on {evidence['changed_interior_pixels']} pixels. "
        'These numbers are NOT a verdict. A large collapse is a cue to inspect lost folds, shading or surface detail, '
        'not evidence that they were preserved. A naturally flat sign can legitimately have little contrast; '
        'do not reject it from the measurement alone. Judge actual photographic content.\n')
