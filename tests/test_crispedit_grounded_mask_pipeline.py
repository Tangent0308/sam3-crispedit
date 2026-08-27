import json

import numpy as np

from crispedit_grounded_mask_pipeline import (
    _sam_text_prompt,
    aspect_ratio_delta,
    box_iou,
    map_target_mask_to_source,
    segment_grounded_box,
)
from crispedit_grounding import (
    build_grounding_requests,
    canonicalize_type,
    grounding_is_complete,
    parse_grounding_output,
)
from scripts.build_mask_bad_case_selection import extract_mask_cases


def test_grounding_routes_and_asymmetric_replace():
    assert canonicalize_type("motion change") == "motion"
    assert [request.grounding_image for request in build_grounding_requests("add", "add a bird")] == ["target"]
    assert [request.grounding_image for request in build_grounding_requests("replace", "replace a with b")] == [
        "source",
        "target",
    ]
    assert grounding_is_complete("replace", {"source": [], "target": [{"ref": "hands"}]})
    assert not grounding_is_complete("motion", {"source": [], "target": [{"ref": "hand"}]})


def test_grounding_parser_accepts_fence_label_and_clips():
    text = """<think>ignored</think>\n```json
[{"label":"right arm","bbox_2d":[-2,10,1004,999]}]
```"""
    assert parse_grounding_output(text) == [
        {"ref": "right arm", "bbox_2d": [0.0, 10.0, 1000.0, 999.0]}
    ]


def test_grounding_parser_drops_abstract_absence_box():
    text = '[{"ref":"absence of hands","bbox_2d":[100,200,500,900]}]'
    assert parse_grounding_output(text) == []


def test_grounding_parser_salvages_truncated_multi_instance_output():
    text = (
        '[{"ref":"flowers", "bbox_2d":[10,20,30,40], '
        '"bbox_2d":[50,60,90,100], "bbox_2d":[120'
    )
    assert parse_grounding_output(text) == [
        {"ref": "flowers", "bbox_2d": [10.0, 20.0, 30.0, 40.0]},
        {"ref": "flowers", "bbox_2d": [50.0, 60.0, 90.0, 100.0]},
    ]


def test_grounding_parser_recovers_duplicate_keys_from_valid_json():
    text = '[{"ref":"mushroom","bbox_2d":[1,2,3,4],"bbox_2d":[5,6,7,8]}]'
    assert parse_grounding_output(text) == [
        {"ref": "mushroom", "bbox_2d": [1.0, 2.0, 3.0, 4.0]},
        {"ref": "mushroom", "bbox_2d": [5.0, 6.0, 7.0, 8.0]},
    ]


def test_grounding_parser_salvages_complete_objects_before_truncation():
    text = (
        '[{"bbox_2d":[1,2,3,4],"label":"cow"},'
        '{"bbox_2d":[5,6,7,8],"label":"house"},{"bbox_2d":[9'
    )
    assert parse_grounding_output(text) == [
        {"ref": "cow", "bbox_2d": [1.0, 2.0, 3.0, 4.0]},
        {"ref": "house", "bbox_2d": [5.0, 6.0, 7.0, 8.0]},
    ]


def test_bad_case_doc_extraction_is_mask_section_only():
    markdown = """## 1. Prefilter
#### `skip_00000.parquet` row `1`
## 2. Mask 打标这一侧 bad case
#### `add_00000.parquet` row `17`
#### `remove_00001.parquet` row `2`
## 3. Reference
#### `reference_00000.parquet` row `3`
"""
    assert extract_mask_cases(markdown) == [
        {"shard": "add_00000.parquet", "row_idx": 17},
        {"shard": "remove_00001.parquet", "row_idx": 2},
    ]


class _PVSModel:
    def predict_inst(self, inference_state, box, multimask_output):
        masks = np.zeros((3, 100, 100), dtype=np.uint8)
        masks[0, 19:81, 19:81] = 1
        masks[1, 40:60, 40:60] = 1
        masks[2, :10, :10] = 1
        return masks, np.asarray([0.92, 0.99, 0.2]), np.zeros((3, 1, 1), dtype=np.float32)


class _PVSProcessor:
    model = _PVSModel()


def test_pvs_prefers_consistent_candidate_over_higher_iou_tiny_candidate():
    mask, metadata = segment_grounded_box(
        _PVSProcessor(), {}, "object", [200, 200, 800, 800], (100, 100)
    )
    assert metadata["mask_source"] == "pvs"
    assert metadata["predicted_iou"] == 0.92
    assert mask[50, 50] == 1
    assert mask[5, 5] == 0


class _HybridModel:
    def predict_inst(self, inference_state, box, multimask_output):
        masks = np.zeros((1, 100, 100), dtype=np.uint8)
        masks[0, 18:82, 18:82] = 1
        return masks, np.asarray([0.9]), np.zeros((1, 1, 1), dtype=np.float32)


class _HybridProcessor:
    model = _HybridModel()

    def reset_all_prompts(self, state):
        return None

    def set_text_prompt(self, prompt, state):
        import torch

        masks = torch.zeros((2, 1, 100, 100), dtype=torch.uint8)
        masks[0, 0, 25:28, 25:28] = 1
        masks[1, 0, 72:75, 72:75] = 1
        return {
            "masks": masks,
            "boxes": torch.tensor([[25, 25, 28, 28], [72, 72, 75, 75]]),
            "scores": torch.tensor([0.8, 0.9]),
        }


def test_sparse_semantic_detail_overrides_enclosing_pvs_without_rectangle():
    mask, metadata = segment_grounded_box(
        _HybridProcessor(), {}, "multiple facial piercings", [200, 200, 800, 800], (100, 100)
    )
    assert metadata["mask_source"] == "pcs"
    assert metadata["selection_reason"] == "SEMANTIC_DETAIL"
    assert not metadata["coverage_box_union"]
    assert mask.sum() == 18


def test_sam_prompt_keeps_body_part_semantics_from_verbose_action_label():
    assert _sam_text_prompt("man raises his right hand with fingers pinched together") == (
        "right hand with fingers pinched together"
    )


def test_target_mapping_and_aspect_ratio_audit():
    target = np.zeros((20, 40), dtype=np.uint8)
    target[8:12, 18:22] = 1
    mapped = map_target_mask_to_source(target, (100, 100), ar_mismatch=False)
    assert mapped.shape == (100, 100)
    assert mapped.sum() > 4 * 25
    assert aspect_ratio_delta((100, 100), (102, 100)) > 0
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
