"""Cross-mask reference diagnostics. Flags are not source-quality rejection gates."""
from collections import defaultdict
import re

import numpy as np
from PIL import Image


def reference_collisions(rows, max_iou=.05):
    """Flag identical references assigned to almost-disjoint masks in one scene.

    Same-object partial/occluded masks can be disjoint, so this is deliberately
    a review signal only. Distinct wording does not prove semantic uniqueness.
    """
    groups=defaultdict(list)
    for row in rows:
        for unit in row['units']:
            key=(row['source']['source_id'],re.sub(r'\s+',' ',unit['ref_phrase'].strip().lower()).rstrip('.'))
            groups[key].append(unit)
    records=[]
    for (source_id,phrase),units in groups.items():
        if len(units)<2:continue
        masks={u['object_id']:np.asarray(Image.open(u['mask_path']).convert('L'))>0 for u in units}
        for i,left in enumerate(units):
            for right in units[i+1:]:
                a,b=masks[left['object_id']],masks[right['object_id']]
                if a.shape!=b.shape:raise ValueError('Reference comparison requires aligned source masks')
                union=np.count_nonzero(a|b)
                iou=float(np.count_nonzero(a&b)/union) if union else 0.
                if iou<=max_iou:
                    records.append(dict(stage='reference_collision',source_id=source_id,
                        object_ids=[left['object_id'],right['object_id']],ref_phrase=phrase,
                        mask_iou=iou,needs_reference_review=True,source_mask_rejected=False,
                        reason='identical_reference_for_nearly_disjoint_masks'))
    return records
