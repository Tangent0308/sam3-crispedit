import numpy as np
from PIL import Image
import pytest

from sa1b_pipeline.background_mask_refinement import propose, apply_candidate


def test_consensus_preserves_holes_and_protected_objects():
    mask = np.ones((160,240), bool)
    mask[50:100,50:150] = False
    evidence = mask.copy(); evidence[50:60,50:150] = True
    protected = np.zeros_like(mask); protected[:,90:100] = True
    rows = [[dict(mask=evidence, score=.98, recall=.99, box=[0,0,200,160]),
             dict(mask=evidence, score=.98, recall=.99, box=[0,0,240,160])]]
    new, meta = propose(mask, rows, source=Image.new("RGB",(240,160),"blue"), protected=protected)
    assert meta["status"] == "candidate"
    assert new[mask].all()
    assert not new[70:90,60:140].any()
    assert not (new & ~mask & protected).any()
    rows[0][1]["score"] = .7
    unchanged, meta = propose(mask, rows, source=Image.new("RGB",(240,160),"blue"))
    assert np.array_equal(unchanged,mask)


def test_updates_all_public_geometry_and_invalidates_hidden_support(tmp_path):
    mask=np.ones((100,100),bool);mask[30:40,30:40]=False
    original=tmp_path/"old.png";source=tmp_path/"src.png"
    Image.fromarray(mask.astype("uint8")*255).save(original)
    Image.new("RGB",(100,100),"blue").save(source)
    m=dict(unit_id="U1",operation="attribute",mask=str(original),
           instruction_spec={"attributes":[{"property":"color","value":"teal"}]},
           source_grounding_evidence={"old":True},
           unit=dict(mask_path=str(original),execution_mask_path="stale",protected_mask_path="stale"))
    g=dict(source_image=str(source),units=[m],dependency_schedule={"old":True})
    repaired=apply_candidate(g,"U1",np.ones_like(mask),tmp_path/"new",query="water",evidence={})
    u=repaired["units"][0]
    assert g["units"][0]["mask"]==str(original)
    assert u["mask"]==u["unit"]["mask_path"] and u["mask"]!=str(original)
    assert u["bbox"]==u["unit"]["bbox"] and u["point"]==u["unit"]["point"]
    assert u["unit"]["execution_mask_path"] is None and u["support_mask"] is None
    assert "source_grounding_evidence" not in u and "dependency_schedule" not in repaired
    assert np.asarray(Image.open(original))[35,35]==0
    with pytest.raises(ValueError,match="protected"):
        apply_candidate(dict(g,protected_source_masks=[str(u["mask"])]),"U1",
                        np.ones_like(mask),tmp_path/"bad",query="water",evidence={})
    m["instruction_spec"]["attributes"] = []
    with pytest.raises(ValueError, match="color edits"):
        apply_candidate(g, "U1", np.ones_like(mask), tmp_path/"empty",
                        query="water", evidence={})


def test_different_source_surface_is_not_filled():
    mask=np.ones((160,240),bool);mask[50:80,60:130]=False
    rgb=np.full((160,240,3),(20,70,160),dtype=np.uint8);rgb[50:80,60:130]=(240,230,210)
    all_pixels=np.ones_like(mask)
    rows=[[dict(mask=all_pixels,score=.99,recall=1,box=[0,0,240,160]),
           dict(mask=all_pixels,score=.99,recall=1,box=[1,1,240,160])]]
    new,_=propose(mask,rows,source=Image.fromarray(rgb))
    assert not new[55:75,65:125].any()
