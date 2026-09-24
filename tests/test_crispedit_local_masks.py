import json

import numpy as np
import pytest
from PIL import Image

from crispedit.mask import pipeline
from crispedit.mask.grounding import (build_grounding_prompt)
from crispedit.mask.grounding_runner import Qwen38Grounder
from crispedit.mask.regions import clean_mask, sparse_degenerate


def test_add_attachment_is_semantic_only_and_audited_without_rewriting_grounding(monkeypatch):
    calls = []
    def segment(processor,state,ref,box,shape,**kwargs):
        calls.append((ref,kwargs['semantic_only']))
        mask = np.zeros(shape,dtype=np.uint8)
        mask[20:30,20:30] = 1
        return mask,{'mask_source':'pcs','semantic_mask_source':'pcs'}
    monkeypatch.setattr(pipeline,'segment_grounded_box',segment)
    class Processor:
        def set_image(self,image): return {}
    payload = {'boxes':{'target':[{'ref':'sofa with blanket','change_id':0,'bbox_2d':[0,0,1000,1000]}]},
               'observation':{'parsed':{'changes':[{'change_id':0,'source_ref':'sofa',
                    'target_ref':'sofa with blanket','change':'A blanket was added.'}]}}}
    original = json.dumps(payload)
    sample = {'input_img':Image.new('RGB',(100,100)),
              'output_img':Image.new('RGB',(100,100)),'type':'add'}
    result = pipeline.annotate_grounded_sample(Processor(),sample,{'ground_json':original},'test')
    assert calls == [('blanket',True)]
    audit = json.loads(result['instances'][0]['candidate_audit_json'])
    assert audit['added_attachment_query']['original_ref'] == 'sofa with blanket'
    assert result['instances'][0]['mapped_from_target']
    assert json.dumps(payload) == original


def test_split_part_box_is_containment_not_positive_whole_owner_prompt(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('broad part box must not run PVS')
    def text(processor,state,ref,box,shape,**kwargs):
        assert not kwargs['use_geometric_prompt']
        mask = np.zeros(shape,dtype=np.uint8)
        mask[20:40,20:40] = 1
        return mask,dict(selected_count=1,concept_score=.9)
    monkeypatch.setattr(pipeline,'_pvs_mask',forbidden)
    monkeypatch.setattr(pipeline,'_pcs_mask',text)
    mask,metadata = pipeline.segment_grounded_box(None,{},'face',[0,0,1000,1000],
        (100,100),semantic_only=True)
    assert mask.sum() < 1000 and not metadata['errors']


@pytest.mark.parametrize("kind,side", [("add", "target"), ("replace", "source"),
                                     ("motion", "source"), ("remove", "source"), ("color", "source")])
def test_legacy_grounding_cannot_leak_opposite_mask(monkeypatch, kind, side):
    calls = []
    def segment(processor, state, items, shape, canvas, edit_type, image=None):
        calls.append(canvas)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[20:30, 20:30] = 1
        return [mask], [{"grounding_image": canvas, "mask_source": "pcs"}]
    monkeypatch.setattr(pipeline, "_segment_items", segment)
    payload = {"boxes": {"source": [{"ref": "old"}], "target": [{"ref": "new"}]}}
    sample = {"input_img": Image.new("RGB", (100, 80)), "output_img": Image.new("RGB", (100, 80)), "type": kind}
    result = pipeline.annotate_grounded_sample(None, sample, {"ground_json": json.dumps(payload)}, "test")
    assert calls == [side]
    assert result["instances"][0]["mapped_from_target"] == (side == "target")


def test_dense_cleanup_preserves_disconnected_real_parts():
    mask = np.zeros((200,200), dtype=np.uint8)
    mask[10:90,10:90] = 1
    mask[110:160,110:160] = 1
    mask[180,180] = 1
    cleaned, audit = clean_mask(mask)
    assert cleaned[30,30] and cleaned[130,130] and not cleaned[180,180]
    assert audit["components_before"] == 3 and audit["components_after"] == 2


def test_sparse_cleanup_preserves_both_eyes():
    mask = np.zeros((50,50), dtype=np.uint8)
    mask[10:13,10:13] = 1
    mask[10:12,30:32] = 1
    cleaned, _ = clean_mask(mask, sparse=True)
    assert np.array_equal(cleaned, mask)


def test_sparse_enclosing_mask_is_rejected():
    mask = np.zeros((100,100), dtype=np.uint8)
    mask[10:80,10:80] = 1
    assert sparse_degenerate(mask, [0,0,100,100])


def test_low_score_pvs_is_not_reported_as_success():
    class Model:
        def predict_inst(self, **kwargs):
            return np.ones((1,100,100)), np.array([0.108]), None
    class Processor:
        model = Model()
    mask, audit = pipeline.segment_grounded_box(Processor(), {}, "eye", [0,0,1000,1000], (100,100))
    assert not mask.any()
    assert audit["mask_source"] == "none"


def test_crop_mask_is_restored_to_original_coordinates(monkeypatch):
    class Processor:
        def set_image(self, image):
            assert image.size[0] < 200 and image.size[1] < 100
            return {}
    def segment(processor, state, ref, box, shape, **kwargs):
        mask = pipeline.mask_from_box(pipeline.normalized_box_to_pixels(box, shape), shape)
        return mask, {"mask_source": "pcs", "semantic_mask_source": "pcs"}
    monkeypatch.setattr(pipeline, "segment_grounded_box", segment)
    unit = {"ref":"shirt", "bbox_2d":[300,300,600,700], "change_id":3,
            "edit_id":1, "change":"The left shirt becomes white.",
            "grounding_ref":"black shirt", "location":"left seated person"}
    masks, audit = pipeline._segment_items(Processor(), None, [unit],
                                          (100,200), "source", "color", Image.new("RGB", (200,100)))
    assert masks[0].shape == (100,200)
    np.testing.assert_array_equal(pipeline.mask_to_box(masks[0]), [60,30,120,70])
    assert audit[0]["crop_xyxy"]
    identity = json.loads(audit[0]['candidate_audit_json'])['edit_unit']
    assert identity == {key:unit[key] for key in
                        ('change_id','edit_id','change','grounding_ref','location')}


def test_instruction_skin_word_cannot_override_observed_clothing():
    prompt = build_grounding_prompt("color", "fair skin wearing red jackets", "source",
                                   {"changes":[{"source_ref":"black shirt", "sam_ref":"black shirt"}]})
    assert "Independently ground a same-subject surface" not in prompt
    assert "black shirt" in prompt


def test_global_sparse_region_uses_local_tiles_without_hull(monkeypatch):
    sizes = []
    class Processor:
        def set_image(self, image):
            sizes.append(image.size)
            return {}
    def segment(processor, state, ref, box, shape, **kwargs):
        mask = np.zeros(shape, dtype=np.uint8)
        mask[shape[0]//2:shape[0]//2+3, shape[1]//2:shape[1]//2+3] = 1
        return mask, {"mask_source":"pcs", "semantic_mask_source":"pcs", "selected_count":10}
    monkeypatch.setattr(pipeline, "segment_grounded_box", segment)
    masks, audit = pipeline._segment_items(Processor(), None,
        [{"ref":"confetti", "bbox_2d":[0,0,1000,1000], "region_mode":"aggregate_region", "mask_density":"sparse"}],
        (100,100), "target", "add", Image.new("RGB", (100,100)))
    assert len(masks) == len(sizes) == 4
    assert all(width < 100 and height < 100 for width,height in sizes)
    assert len({item["instance_id"] for item in audit}) == 4
    assert sum(int(mask.sum()) for mask in masks) == 36


def test_high_score_almost_empty_mask_is_rejected():
    import torch
    class Model:
        def predict_inst(self, **kwargs):
            return np.ones((1,100,100)), np.array([0.1]), None
    class Processor:
        model = Model()
        def reset_all_prompts(self, state):
            pass
        def set_text_prompt(self, prompt, state):
            masks = torch.zeros((1,1,100,100), dtype=torch.uint8)
            masks[0,0,50,50] = 1
            return {"masks":masks, "boxes":torch.tensor([[0,0,100,100]]), "scores":torch.tensor([0.95])}
        def add_geometric_prompt(self, *args, **kwargs):
            return self.set_text_prompt(None, None)
    mask, audit = pipeline.segment_grounded_box(Processor(), {}, "eyes", [0,0,1000,1000], (100,100))
    assert not mask.any()
    assert audit["mask_source"] == "none"


def test_format_retry_is_not_identical_to_failed_request():
    conversation = [{"role":"user", "content":[{"type":"text", "text":"original schema"}]}]
    retry = Qwen38Grounder.correction_conversation(conversation, "bad JSON", "missing ID 1")
    assert len(conversation) == 1 and len(retry) == 3
    assert retry[-2]["content"][0]["text"] == "bad JSON"
    assert "missing ID 1" in retry[-1]["content"][0]["text"]


def test_global_particles_cannot_turn_into_enclosing_silhouettes(monkeypatch):
    class Processor:
        def set_image(self, image):
            return {}
    def segment(processor, state, ref, box, shape, **kwargs):
        return np.ones(shape,dtype=np.uint8), {"mask_source":"pcs", "semantic_mask_source":"pcs"}
    monkeypatch.setattr(pipeline, "segment_grounded_box", segment)
    masks, audit = pipeline._segment_items(Processor(), None,
        [{"ref":"confetti", "bbox_2d":[0,0,1000,1000], "region_mode":"aggregate_region", "mask_density":"sparse"}],
        (100,100), "target", "add", Image.new("RGB", (100,100)))
    assert not any(mask.any() for mask in masks)
    assert all(row["mask_source"] == "none" for row in audit)


def test_dense_nearby_group_can_be_supported_by_independent_candidates(monkeypatch):
    mask = np.zeros((100,100),dtype=np.uint8)
    mask[30:70,30:70] = 1
    details = {"predicted_iou":0.95, "selected_count":1, "candidate_count":1,
               "inside_ratio":1.0, "box_iou":0.9}
    monkeypatch.setattr(pipeline, "_pvs_mask", lambda *args, **kwargs: (mask.copy(), dict(details)))
    monkeypatch.setattr(pipeline, "_pcs_mask", lambda *args, **kwargs: (mask.copy(), dict(details)))
    result, audit = pipeline.segment_grounded_box(None, {}, "white birds and ribbon", [200,200,800,800],
                                                (100,100), region_mode="aggregate_region", mask_density="sparse")
    assert result.sum() == mask.sum()
    assert audit["mask_source"] == "pcs"


def test_split_confetti_cannot_bypass_global_particle_safety(monkeypatch):
    mask = np.zeros((100,100),dtype=np.uint8)
    mask[30:60,30:60] = 1
    details = {"predicted_iou":0.95,"selected_count":1,"candidate_count":1}
    monkeypatch.setattr(pipeline, "_pvs_mask", lambda *args, **kwargs: (mask.copy(),dict(details)))
    monkeypatch.setattr(pipeline, "_pcs_mask", lambda *args, **kwargs: (mask.copy(),dict(details)))
    result, audit = pipeline.segment_grounded_box(None,{},"red confetti",[200,200,800,800],(100,100),
                                                 region_mode="object",mask_density="dense")
    assert not result.any() and audit["mask_source"] == "none"
    assert audit["selection_reason"] == "FINE_PARTICLE_ENCLOSING_OBJECT"
