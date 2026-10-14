import numpy as np
from PIL import Image
from sa1b_pipeline.reference_diagnostics import reference_collisions


def test_identical_reference_for_distinct_petals_is_flagged_without_rejecting(tmp_path):
    rows=[]
    for i in range(3):
        mask=np.zeros((20,20),dtype=np.uint8);mask[2:6,2+i*5:5+i*5]=255
        path=tmp_path/f'{i}.png';Image.fromarray(mask).save(path)
        rows.append({'source':{'source_id':'flower'},'units':[{'object_id':str(i),'ref_phrase':"the petal at 10 o'clock",'mask_path':str(path)}]})
    flags=reference_collisions(rows)
    assert len(flags)==3
    assert all(f['needs_reference_review'] and not f['source_mask_rejected'] and f['mask_iou']==0 for f in flags)
    # Different source identity prevents a false cross-scene collision.
    rows[2]['source']['source_id']='other flower'
    assert len(reference_collisions(rows))==1


def test_overlapping_annotations_of_one_object_are_not_instance_collisions(tmp_path):
    mask=np.zeros((20,20),dtype=np.uint8);mask[2:12,2:12]=255
    path=tmp_path/'mask.png';Image.fromarray(mask).save(path)
    rows=[{'source':{'source_id':'flower'},'units':[{'object_id':str(i),'ref_phrase':'the bottom petal','mask_path':str(path)}]} for i in range(2)]
    assert reference_collisions(rows)==[]
