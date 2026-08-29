import json

import cv2
import numpy as np

from crispedit_grounded_mask_pipeline import (
    _aggregate_semantic_connected_coverage,
    _color_surface_sam_prompt,
    _fuse_pcs_prompts,
    _sam_text_prompt,
    aspect_ratio_delta,
    box_iou,
    expand_box,
    map_target_mask_to_source,
    segment_grounded_box,
)
from crispedit_grounding import (
    TWO_PASS_PROMPT_VERSION,
    bbox_refinement_crop,
    box_needs_local_refinement,
    build_bbox_refinement_prompt,
    build_change_observation_prompt,
    build_grounding_prompt,
    build_grounding_requests,
    canonicalize_type,
    conservative_refined_bbox,
    grounding_is_complete,
    local_bbox_refinement_enabled,
    map_crop_bbox_to_full,
    parse_bbox_refinement_output,
    parse_change_observation,
    parse_grounding_output,
    prompt_version_for_mode,
)
from crispedit_mllm_grounding import GROUND_SCHEMA, prefilter_fields
from crispedit_grounded_mask_runner import MASK_SCHEMA, _copy_metadata
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


def test_latest_prefilter_manifest_metadata_survives_grounding_and_mask():
    manifest_row = {
        "prefilter_verdict": "PASS",
        "prefilter_confidence": 0.91,
        "prefilter_method": "fact_prefilter",
        "prefilter_evidence_schema": "fact_evidence",
        "prefilter_model_name": "Qwen3-VL",
        "prefilter_run_id": "run-id",
        "prefilter_reason": "supported",
        "prefilter_decision": "keep",
        "filter_reason_codes": "",
        "filter_mismatch_score": 0.02,
    }
    grounded = prefilter_fields(manifest_row)
    assert grounded["filter_decision"] == "keep"
    assert grounded["prefilter_evidence_schema"] == "fact_evidence"
    assert grounded["filter_mismatch_score"] == 0.02
    copied = _copy_metadata(grounded)
    assert {key: copied[key] for key in grounded} == grounded
    for field in (
        "prefilter_evidence_schema",
        "filter_reason_codes",
        "filter_mismatch_score",
    ):
        assert field in GROUND_SCHEMA.names
        assert field in MASK_SCHEMA.names


def test_add_remove_route_clear_collateral_opposite_side_changes():
    add_observation = {
        "changes": [
            {
                "source_ref": "blue fish",
                "target_ref": "",
                "change": "removed while flowers were added",
                "instruction_aligned": False,
            }
        ]
    }
    assert [
        request.grounding_image
        for request in build_grounding_requests("add", "add flowers", add_observation)
    ] == ["target", "source"]
    remove_observation = {
        "changes": [
            {
                "source_ref": "facial piercings",
                "target_ref": "",
                "change": "piercings removed",
                "instruction_aligned": True,
            },
            {
                "source_ref": "",
                "target_ref": "dangling earrings",
                "change": "earrings added while piercings were removed",
                "instruction_aligned": False,
            }
        ]
    }
    assert [
        request.grounding_image
        for request in build_grounding_requests("remove", "remove piercings", remove_observation)
    ] == ["source", "target"]

    add_removed_sentinel = {
        "changes": [
            {
                "source_ref": "small blue fish",
                "target_ref": "removed",
                "change": "fish disappeared while flowers were added",
            }
        ]
    }
    assert [
        request.grounding_image
        for request in build_grounding_requests(
            "add", "add flowers", add_removed_sentinel
        )
    ] == ["target", "source"]


def test_two_pass_observation_prompt_and_grounding_checklist():
    prompt = build_change_observation_prompt("color", "make the arms darker")
    assert "paired images are the source of truth" in prompt
    assert "arms and the face" in prompt
    assert "face/head, neck, each arm/hand" in prompt
    assert "rewrite the instruction into a precise specification" in prompt
    assert "nearby group" in prompt
    assert "checked_regions" in prompt
    observation = {
        "edit_summary": "arms and face became darker",
        "changes": [
            {
                "source_ref": "man's face",
                "target_ref": "man's face",
                "change": "skin tone became darker",
                "instruction_aligned": False,
            }
        ],
    }
    grounding_prompt = build_grounding_prompt(
        "color", "make the arms darker by changing the skin tone", "source", observation
    )
    assert "man's face" in grounding_prompt
    assert "Do not assume the original instruction" in grounding_prompt
    assert "never box the whole person" in grounding_prompt
    grouped_prompt = build_grounding_prompt(
        "add", "add scattered petals", "target", observation
    )
    assert "SPATIALLY SEPARATED EDIT REGION" in grouped_prompt
    assert "ALWAYS emit ONE aggregate_region box" in grouped_prompt
    assert "MUST be a superset" in grouped_prompt
    assert prompt_version_for_mode("two-pass") == TWO_PASS_PROMPT_VERSION


def test_small_box_local_refinement_geometry_is_recall_first():
    initial = {"ref": "black cap", "bbox_2d": [270, 183, 505, 333]}
    assert box_needs_local_refinement(initial)
    assert not box_needs_local_refinement(
        {"ref": "whole person", "bbox_2d": [100, 100, 700, 800]}
    )
    crop = bbox_refinement_crop(initial["bbox_2d"])
    assert crop == [35.0, 33.0, 740.0, 483.0]
    mapped = map_crop_bbox_to_full(crop, [100, 100, 700, 700])
    assert mapped == [105.5, 78.0, 528.5, 348.0]
    final = conservative_refined_bbox(initial["bbox_2d"], mapped)
    assert final == [81.5, 54.0, 552.5, 372.0]
    assert final[0] <= mapped[0] and final[1] <= mapped[1]
    assert final[2] >= initial["bbox_2d"][2] and final[3] >= initial["bbox_2d"][3]

    # A clearly shifted refinement replaces the initial proposal instead of
    # unioning a wrong nearby facial feature into the final search region.
    mouth = conservative_refined_bbox(
        [638, 425, 692, 465], [642.95, 483.64, 689.99, 498.76]
    )
    assert mouth == [630.95, 471.64, 701.99, 510.76]

    # A crop pass can lock onto a salient subpart such as a hammer head.  High
    # containment keeps the initial full-object extent even when IoU < 0.25.
    contained_subpart = conservative_refined_bbox(
        [392, 592, 442, 750], [388, 594, 434, 633]
    )
    assert contained_subpart == [376.0, 580.0, 454.0, 762.0]

    # A failed local pass still expands the initial proposal conservatively.
    fallback = conservative_refined_bbox(initial["bbox_2d"], None)
    assert fallback == [195.0, 108.0, 580.0, 408.0]


def test_bbox_refinement_prompt_and_parser_keep_candidate_ids():
    candidates = [
        {
            "candidate_id": 2,
            "ref": "speaker's mouth",
            "initial_bbox": [638, 425, 692, 465],
            "crop_bbox": [518, 305, 812, 585],
        }
    ]
    prompt = build_bbox_refinement_prompt(candidates)
    assert "complete mouth/lips rather than the nose" in prompt
    assert '"candidate_id":2' in prompt
    assert "638" not in prompt
    assert "812" not in prompt
    assert "derive every returned number solely from the visible candidate crop" in prompt
    parsed = parse_bbox_refinement_output(
        "```json\n"
        '[{"candidate_id":2,"ref":"speaker mouth","bbox_2d":[210,380,790,690]}]'
        "\n```"
    )
    assert parsed == [
        {
            "candidate_id": 2,
            "ref": "speaker mouth",
            "bbox_2d": [210.0, 380.0, 790.0, 690.0],
        }
    ]


def test_background_protection_boxes_are_not_local_refinement_candidates():
    # The pure geometry trigger may consider a thin foreground-protection box
    # small; the runner explicitly bypasses the entire background route before
    # constructing views.  Keep the trigger behavior documented here so future
    # refactors do not mistake size alone for route eligibility.
    assert box_needs_local_refinement(
        {"ref": "foreground person", "bbox_2d": [100, 100, 260, 700]}
    )
    assert not local_bbox_refinement_enabled("background change")
    assert not local_bbox_refinement_enabled("style")
    assert local_bbox_refinement_enabled("motion change")


def test_change_observation_parser_accepts_fence_and_normalizes():
    raw = """```json
{"edit_summary":"arms and face darkened","changes":[
  {"source_ref":"man's arms","target_ref":"dark arms","change":"skin became darker","instruction_aligned":true},
  {"source_ref":"man's face","target_ref":"dark face","change":"face also became darker","instruction_aligned":"no"}
]}
```"""
    assert parse_change_observation(raw) == {
        "edit_summary": "arms and face darkened",
        "checked_regions": [],
        "changes": [
            {
                "source_ref": "man's arms",
                "target_ref": "dark arms",
                "sam_ref": "man's arms",
                "region_description": "",
                "region_layout": "single",
                "change": "skin became darker",
                "instruction_aligned": True,
            },
            {
                "source_ref": "man's face",
                "target_ref": "dark face",
                "sam_ref": "man's face",
                "region_description": "",
                "region_layout": "single",
                "change": "face also became darker",
                "instruction_aligned": False,
            },
        ],
    }


def test_change_observation_parser_salvages_changes_after_broken_checks():
    raw = r'''{"edit_summary":"a red vase became blue","checked_regions":[
      {"ref":"vase","changed":true},
      "ref":"table","changed":false}],
      "changes":[{"source_ref":"red vase","target_ref":"blue vase",
      "sam_ref":"ceramic vase","region_description":"center of image",
      "region_layout":"single","change":"red surface became blue",
      "instruction_aligned":true}]}'''
    assert parse_change_observation(raw) == {
        "edit_summary": "a red vase became blue",
        "checked_regions": [],
        "changes": [
            {
                "source_ref": "red vase",
                "target_ref": "blue vase",
                "sam_ref": "ceramic vase",
                "region_description": "center of image",
                "region_layout": "single",
                "change": "red surface became blue",
                "instruction_aligned": True,
            }
        ],
    }


def test_subject_surface_audit_is_selected_only_by_instruction_surface():
    whole_object_prompt = build_grounding_prompt(
        "color",
        "change the color of Xenomorph to gold",
        "source",
        {
            "changes": [
                {
                    "source_ref": "alien head and hands",
                    "target_ref": "gold alien head and hands",
                    "change": "head and hands became gold",
                }
            ]
        },
    )
    assert "Independently ground a same-subject surface" not in whole_object_prompt
    assert "Find every complete object or sub-part" in whole_object_prompt

    skin_prompt = build_grounding_prompt(
        "color",
        "make the arms darker, changing the skin tone",
        "source",
        {
            "changes": [
                {
                    "source_ref": "bare arms",
                    "target_ref": "darker bare arms",
                    "change": "skin became darker",
                }
            ]
        },
    )
    assert "Independently ground a same-subject surface" in skin_prompt


def test_grounding_parser_accepts_fence_label_and_clips():
    text = """<think>ignored</think>\n```json
[{"label":"right arm","bbox_2d":[-2,10,1004,999]}]
```"""
    assert parse_grounding_output(text) == [
        {"ref": "right arm", "bbox_2d": [0.0, 10.0, 1000.0, 999.0]}
    ]


def test_grounding_parser_preserves_aggregate_region_mode():
    text = '[{"ref":"facial piercings","bbox_2d":[100,200,700,800],"region_mode":"nearby_group","mask_density":"sparse"}]'
    assert parse_grounding_output(text) == [
        {
            "ref": "facial piercings",
            "bbox_2d": [100.0, 200.0, 700.0, 800.0],
            "region_mode": "aggregate_region",
            "mask_density": "sparse",
        }
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


def test_adaptive_prompt_margin_only_uses_image_floor_for_tiny_dimensions():
    shape = (1000, 1500)
    # A normal face/limb region keeps the relative 2.5% expansion.
    normal = expand_box(
        [500, 400, 700, 800],
        shape,
        0.025,
        min_image_frac=0.015,
        min_margin_max_dimension_frac=0.05,
    )
    np.testing.assert_allclose(normal, [495, 390, 705, 810])

    # A tiny earring receives the 15px image-relative safety margin on both
    # dimensions instead of a sub-pixel percentage of its own extent.
    tiny = expand_box(
        [500, 400, 520, 430],
        shape,
        0.025,
        min_image_frac=0.015,
        min_margin_max_dimension_frac=0.05,
    )
    np.testing.assert_allclose(tiny, [485, 385, 535, 445])


def test_color_human_surface_prompt_targets_skin_parts_not_whole_person():
    assert (
        _color_surface_sam_prompt(
            "central male performer arms", "central male performer arms"
        )
        == "exposed human arms skin"
    )
    assert (
        _color_surface_sam_prompt(
            "man's face and neck", "man's face and neck"
        )
        == "exposed human face and neck skin"
    )
    assert _color_surface_sam_prompt("brown dog fur", "brown dog fur") == "brown dog fur"


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


def test_aggregate_dual_prompt_rejects_dense_enclosing_mask():
    text = np.zeros((100, 100), dtype=np.uint8)
    text[25:28, 25:28] = 1
    text[72:75, 72:75] = 1
    joint = np.zeros((100, 100), dtype=np.uint8)
    joint[20:80, 20:80] = 1
    metadata = {
        "candidate_count": 2,
        "selected_count": 2,
        "inside_ratio": 1.0,
        "box_iou": 0.1,
        "predicted_iou": 0.8,
    }
    mask, audit = _fuse_pcs_prompts(
        text,
        metadata,
        joint,
        {**metadata, "candidate_count": 1, "selected_count": 1},
        np.asarray([20, 20, 80, 80], dtype=np.float32),
        "aggregate_region",
    )
    assert mask.sum() == 18
    assert audit["pcs_fusion"] == "reject_dense_choose_text"


def test_aggregate_dual_prompt_unions_complementary_sparse_instances():
    text = np.zeros((100, 100), dtype=np.uint8)
    text[25:28, 25:28] = 1
    joint = np.zeros((100, 100), dtype=np.uint8)
    joint[72:75, 72:75] = 1
    metadata = {
        "candidate_count": 1,
        "selected_count": 1,
        "inside_ratio": 1.0,
        "box_iou": 0.05,
        "predicted_iou": 0.8,
    }
    mask, audit = _fuse_pcs_prompts(
        text,
        metadata,
        joint,
        metadata,
        np.asarray([20, 20, 80, 80], dtype=np.float32),
        "aggregate_region",
    )
    assert mask.sum() == 18
    assert audit["pcs_fusion"] == "aggregate_union"


def test_sparse_aggregate_prefers_text_when_joint_mask_is_much_denser():
    text = np.zeros((100, 100), dtype=np.uint8)
    text[25:28, 25:28] = 1
    text[72:75, 72:75] = 1
    joint = np.zeros((100, 100), dtype=np.uint8)
    joint[20:80, 20:80] = 1
    metadata = {
        "candidate_count": 2,
        "selected_count": 2,
        "inside_ratio": 1.0,
        "box_iou": 0.1,
        "predicted_iou": 0.8,
    }
    mask, audit = _fuse_pcs_prompts(
        text,
        metadata,
        joint,
        metadata,
        np.asarray([20, 20, 80, 80], dtype=np.float32),
        "aggregate_region",
        "sparse",
    )
    assert mask.sum() == 18
    assert audit["pcs_fusion"] == "sparse_reject_dense_choose_text"


def test_sparse_aggregate_recovers_joint_when_text_is_tiny_low_confidence():
    text = np.zeros((100, 100), dtype=np.uint8)
    text[25:27, 25:27] = 1
    joint = np.zeros((100, 100), dtype=np.uint8)
    joint[30:65, 25:55] = 1
    common = {
        "candidate_count": 2,
        "selected_count": 2,
        "inside_ratio": 1.0,
        "box_iou": 0.1,
    }
    mask, audit = _fuse_pcs_prompts(
        text,
        {**common, "predicted_iou": 0.33},
        joint,
        {**common, "selected_count": 1, "predicted_iou": 0.9},
        np.asarray([10, 10, 90, 90], dtype=np.float32),
        "aggregate_region",
        "sparse",
    )
    assert mask.sum() == joint.sum()
    assert audit["pcs_fusion"] == "recover_tiny_low_confidence_choose_text+box"


def test_nearby_tiny_semantic_components_get_one_filled_region():
    mask = np.zeros((100, 100), dtype=np.uint8)
    for y, x in ((12, 12), (16, 68), (32, 40), (48, 18), (58, 70), (66, 45)):
        mask[y : y + 2, x : x + 2] = 1
    metadata = {
        "ref": "colorful flowers",
        "bbox_2d": [80, 80, 780, 720],
        "region_mode": "aggregate_region",
        "mask_density": "sparse",
        "semantic_mask_source": "pcs",
        "selected_count": 12,
    }
    result = _aggregate_semantic_connected_coverage(
        mask, metadata, (100, 100), "target", 0
    )
    assert result is not None
    coverage, audit = result
    assert cv2.connectedComponents(coverage, connectivity=8)[0] - 1 == 1
    assert coverage[40, 40] == 1
    assert coverage.mean() < 0.50
    assert audit["selection_reason"] == "AGGREGATE_SEMANTIC_CONVEX_HULL"


def test_global_scattered_group_does_not_get_a_whole_image_hull():
    mask = np.zeros((100, 100), dtype=np.uint8)
    for y, x in ((5, 5), (5, 90), (45, 45), (75, 15), (85, 80), (92, 50)):
        mask[y : y + 2, x : x + 2] = 1
    metadata = {
        "ref": "scattered petals",
        "bbox_2d": [0, 0, 1000, 1000],
        "region_mode": "aggregate_region",
        "mask_density": "sparse",
        "semantic_mask_source": "pcs",
        "selected_count": 20,
    }
    assert (
        _aggregate_semantic_connected_coverage(
            mask, metadata, (100, 100), "target", 0
        )
        is None
    )


def test_sam_prompt_keeps_body_part_semantics_from_verbose_action_label():
    assert _sam_text_prompt("man raises his right hand with fingers pinched together") == (
        "right hand with fingers pinched together"
    )
    assert _sam_text_prompt("mixed wildflowers and foliage") == "mixed wildflowers"
    assert (
        _sam_text_prompt("white and pink flowers with dried foliage")
        == "white and pink flowers"
    )


def test_repeated_non_point_objects_do_not_get_connected_hulls():
    mask = np.zeros((100, 100), dtype=np.uint8)
    for y, x in ((12, 12), (16, 68), (32, 40), (48, 18), (58, 70), (66, 45)):
        mask[y : y + 2, x : x + 2] = 1
    metadata = {
        "ref": "rows of white boats",
        "bbox_2d": [80, 80, 780, 720],
        "region_mode": "aggregate_region",
        "mask_density": "sparse",
        "semantic_mask_source": "pcs",
        "selected_count": 12,
    }
    assert (
        _aggregate_semantic_connected_coverage(
            mask, metadata, (100, 100), "target", 0
        )
        is None
    )


def test_target_mapping_and_aspect_ratio_audit():
    target = np.zeros((20, 40), dtype=np.uint8)
    target[8:12, 18:22] = 1
    mapped = map_target_mask_to_source(target, (100, 100), ar_mismatch=False)
    assert mapped.shape == (100, 100)
    assert mapped.sum() > 4 * 25
    assert aspect_ratio_delta((100, 100), (102, 100)) > 0
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
