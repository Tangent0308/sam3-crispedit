"""Per-edit-unit masks on the source canvas, using the current SAM3 core."""
from __future__ import annotations

import json
import numpy as np

from scaleedit import policy
from scaleedit.mask import pipeline as core


def render_mask(processor, row_idx, row, ground):
    # Imported here to keep the orchestration process free of torch/SAM imports.
    from scaleedit.runner import base_row, decode
    source, target = decode(row['source_image']), decode(row['edited_image'])
    shape = (source.height, source.width)
    union = np.zeros(shape, dtype=np.uint8)
    instances, flags, errors = [], [], []
    payload = json.loads(ground['ground_json'])
    units = (payload.get('observation', {}).get('parsed') or {}).get('changes', [])
    if ground['qc_flag'] == 'GROUND_FAIL':
        flags.append('GROUND_FAIL')
        errors.append(ground['error'] or 'grounding_failed')
    else:
        if ground['qc_flag'] != 'OK':
            flags.extend(['MASK_REVIEW', 'UNRESOLVED_EDIT_UNIT'])
        for unit in units:
            side, identity = unit['image_side'], unit['change_id']
            boxes = [b for b in payload['boxes'].get(side, []) if b['change_id'] == identity]
            if not boxes:
                flags.extend(['MASK_REVIEW', 'MISSING_EDIT_UNIT'])
                continue
            try:
                if unit['geometry'] == 'text':
                    # Text erasure/replacement needs a compact editing region, not
                    # disconnected letter strokes or SAM's entire sign/carrier.
                    output_masks, output_instances = [], []
                    image = source if side == 'source' else target
                    for box in boxes:
                        import cv2
                        polygon = box['polygon_2d']
                        points = np.asarray(polygon,dtype=np.float64)*[image.width/1000,image.height/1000]
                        points = np.rint(points).astype(np.int32)
                        mask = np.zeros((image.height,image.width),dtype=np.uint8)
                        cv2.fillPoly(mask,[points],1)
                        if side == 'target':
                            mismatch = core.aspect_ratio_delta(source.size, target.size) > core.CFG.ar_mismatch_threshold
                            mask = core.map_target_mask_to_source(mask, shape, mismatch)
                            if mismatch:
                                flags.append('AR_MISMATCH')
                        rle = core.encode_rle(mask)
                        output_masks.append(mask)
                        output_instances.append(dict(ref=box['ref'], bbox_2d=box['bbox_2d'],
                            bbox_xyxy=core.mask_to_box(mask).tolist(), grounding_image=side,
                            mapped_from_target=side == 'target', mask_source='text_polygon',
                            polygon_2d=polygon, mask_method='tight_text_region', area=int(mask.sum()),
                            rle_size=rle['size'], rle_counts=rle['counts']))
                    result = dict(mask=core._union(output_masks, shape), instances=output_instances, qc_flags=[])
                else:
                    sample = dict(input_img=source, output_img=target, type=policy.mask_kind(row['final_task'], unit))
                    local = {**payload, 'boxes': {side: boxes}}
                    result = core.annotate_grounded_sample(processor, sample,
                        dict(ground_json=json.dumps(local), qc_flag='OK'), 'sam3')
                union |= result['mask']
                flags.extend(f for f in result['qc_flags'] if f != 'OK')
                for instance in result['instances']:
                    instances.append({**instance, 'instance_id':f'unit_{identity}_{len(instances)}',
                        'candidate_id':len(instances),
                        'audit_json':json.dumps(dict(unit=unit, sam=instance), ensure_ascii=False)})
            except Exception as exc:
                flags.extend(['MASK_REVIEW', 'SEGMENTATION_ERROR'])
                errors.append(f'unit {identity}: {type(exc).__name__}: {exc}')
    if not union.any() and 'GROUND_FAIL' not in flags:
        flags.extend(['MASK_REVIEW', 'EMPTY_MASK'])
    flags = sorted(set(flags)) or ['OK']
    qc = next((f for f in ('GROUND_FAIL', 'MASK_REVIEW', 'BOX_FALLBACK', 'AR_MISMATCH') if f in flags), 'OK')
    sources = sorted({item['mask_source'] for item in instances})
    return dict(base_row(row_idx, row), ground_json=ground['ground_json'], mask_png=core.encode_mask_png(union),
                instance_masks=instances, qc_flag=qc, qc_flags_json=json.dumps(flags),
                mask_source='+'.join(sources) or 'none', area_frac=float(union.mean()), mask_sum=int(union.sum()),
                mask_height=shape[0], mask_width=shape[1], error='; '.join(errors), method=policy.METHOD)
