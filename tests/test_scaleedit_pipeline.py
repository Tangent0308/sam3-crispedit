import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw, ImageFont

from crispedit.mask.pipeline import MASK_POLICY_VERSION as CRISPEDIT_MASK_POLICY_VERSION
from scaleedit.grounding_runner import (
    CorruptSampleImageError,
    GroundingJob,
    Qwen35ScaleEditGrounder,
    _decode_sample_images,
    expand_dense_aggregate_localization_boxes,
    process_job,
    refine_text_localization_boxes,
    refine_surface_localization_boxes,
    upgrade_global_change_mask_mode,
)
from scaleedit.mask_pipeline import (
    _clean_semantic_object_mask,
    _negative_space_from_carrier_mask,
    _target_map_dilate_frac,
    annotate_sample,
)
from scaleedit.policy import (
    SUPPORTED_TASKS,
    apply_observation_plan_policy,
    apply_task_post_policy,
    build_grounding_prompt,
    build_observation_prompt,
    canonical_task,
    grounding_from_localization,
    object_viewpoint_ref,
    parse_bbox_localization,
    parse_observation,
)
from scaleedit.sam3_backend import MASK_POLICY_VERSION as SCALEEDIT_SAM_BACKEND_VERSION
from scripts.validate_scaleedit_masks import validate_mask_placeholder
from scripts.visualize_scaleedit_masks import COARSE_CATEGORY_GROUPS


def test_all_scaleedit_tasks_are_explicitly_supported():
    assert len(SUPPORTED_TASKS) == 23
    for task in SUPPORTED_TASKS:
        assert canonical_task(task.replace("_", " ")) == task


def test_dataset_specific_sam_policies_remain_isolated():
    assert CRISPEDIT_MASK_POLICY_VERSION == (
        "sam3-dual-prompt-region-fusion-v5-surface-aware"
    )
    assert SCALEEDIT_SAM_BACKEND_VERSION == (
        "sam3-dual-prompt-region-fusion-v10-fragment-aware-selection"
    )


def test_validator_accepts_only_well_formed_ground_fail_placeholders():
    placeholder = {
        "qc_flag": "GROUND_FAIL",
        "mask_png": b"",
        "mask_sum": 0,
        "mask_height": 0,
        "mask_width": 0,
        "instance_masks": [],
    }
    assert validate_mask_placeholder(placeholder) == (True, [])
    placeholder["qc_flag"] = "OK"
    is_placeholder, errors = validate_mask_placeholder(placeholder)
    assert is_placeholder
    assert errors == ["missing mask_png outside GROUND_FAIL"]


def test_coarse_review_groups_cover_every_scaleedit_task_once():
    grouped_tasks = [task for _, tasks in COARSE_CATEGORY_GROUPS for task in tasks]
    assert len(grouped_tasks) == len(set(grouped_tasks))
    assert set(grouped_tasks) == SUPPORTED_TASKS


def test_task_prompts_do_not_collapse_scaleedit_to_crispedit_routes():
    style = build_observation_prompt("style_transfer", "paint only the wall")
    extraction = build_observation_prompt("part_extraction", "extract the shirt to white")
    text = build_observation_prompt("gui_interface_text_editing", "replace OLD with NEW")
    assert "Do not assume style means full image" in style
    assert "Product extraction normally recenters" in extraction
    assert "exact old/new glyph block" in text
    grounding = build_grounding_prompt(
        "style_transfer",
        "paint only the wall",
        {
            "mask_mode": "regions",
            "localization_items": [
                {
                    "candidate_id": 0,
                    "image_side": "target",
                    "ref": "painted wall",
                    "spatial_hint": "wall behind the sofa",
                    "geometry": "dense_region",
                    "mask_method": "sam",
                    "region_mode": "object",
                    "mask_density": "dense",
                }
            ],
        },
    )
    assert "Locate these targets in this edited image" in grounding
    assert 'complete named material or subpart "painted wall"' in grounding
    assert "at wall behind the sofa" in grounding
    assert '"bbox_2d":[x1,y1,x2,y2]' in grounding
    assert "mask_mode" not in grounding
    assert "mask_method" not in grounding
    assert "mask_density" not in grounding

    count = build_grounding_prompt(
        "count_change",
        "Add two oranges, one on each side of the existing orange.",
        {
            "mask_mode": "regions",
            "localization_items": [
                {
                    "candidate_id": 0,
                    "image_side": "target",
                    "ref": "left added orange",
                    "spatial_hint": "left of the unchanged orange",
                }
            ],
        },
    )
    assert 'complete object "left added orange"' in count
    assert "at left of the unchanged orange" in count
    assert "Report one tight bbox per target" in count
    assert '"candidate_id":0' in count


def test_prompts_encode_dense_collection_extrema_and_text_pair_verification():
    plan_prompt = build_observation_prompt(
        "compositional_editing",
        "Recolor the fruit filling the basket and remove the person.",
    )
    assert "Many small touching items or material filling one named container" in plan_prompt
    assert "Never encode plural contained content as one ordinary" in plan_prompt

    text_locator = build_grounding_prompt(
        "movie_poster_text_editing",
        "Replace OLD with NEW.",
        {
            "mask_mode": "regions",
            "localization_items": [
                {
                    "candidate_id": 0,
                    "image_side": "source",
                    "ref": "text OLD",
                    "spatial_hint": "banner",
                    "geometry": "sparse_marks",
                    "mask_method": "box",
                    "region_mode": "object",
                },
                {
                    "candidate_id": 1,
                    "image_side": "target",
                    "ref": "text NEW",
                    "spatial_hint": "banner",
                    "geometry": "sparse_marks",
                    "mask_method": "box",
                    "region_mode": "object",
                },
            ],
        },
    )
    assert "Box only the requested glyphs" in text_locator
    assert "not nearby text or the carrier" in text_locator
    assert "position corresponding to the old text" in text_locator
    assert len(text_locator) < 1500


def test_aligned_pair_text_refinement_corrects_shifted_boxes_without_new_model_call():
    source = Image.new("RGB", (400, 220), "white")
    target = source.copy()
    source_draw = ImageDraw.Draw(source)
    target_draw = ImageDraw.Draw(target)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not font_path.exists():
        pytest.skip("DejaVu font is unavailable")
    font = ImageFont.truetype(str(font_path), 54)
    source_draw.text((100, 70), "SOLDIER", fill="black", font=font)
    target_draw.text((100, 70), "GUARD", fill="black", font=font)
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "text SOLDIER",
                "geometry": "sparse_marks",
                "mask_method": "box",
            },
            {
                "candidate_id": 1,
                "image_side": "target",
                "ref": "text GUARD",
                "geometry": "sparse_marks",
                "mask_method": "box",
            },
        ],
    }
    localized = [
        {"candidate_id": 0, "bbox_2d": [180, 180, 820, 430]},
        {"candidate_id": 1, "bbox_2d": [180, 350, 820, 600]},
    ]
    refined = refine_text_localization_boxes(
        source,
        target,
        "gui_interface_text_editing",
        plan,
        localized,
    )
    assert all(item.get("bbox_refinement") for item in refined)
    assert refined[0]["bbox_2d"] == refined[1]["bbox_2d"]
    assert refined[0]["bbox_2d"][1] < 400
    assert refined[0]["bbox_2d"][3] < 650
    assert refined[0]["bbox_refinement"]["rule"] == (
        "aligned_pair_text_component_refinement_v1"
    )


def test_text_pair_refinement_is_disabled_for_global_pair_change():
    source = Image.new("RGB", (200, 100), "black")
    target = Image.new("RGB", (200, 100), "white")
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "old text",
                "geometry": "sparse_marks",
                "mask_method": "box",
            }
        ],
    }
    localized = [{"candidate_id": 0, "bbox_2d": [100, 100, 400, 300]}]
    assert refine_text_localization_boxes(
        source,
        target,
        "building_surface_text_editing",
        plan,
        localized,
    ) == localized


def test_same_location_text_replacement_ignores_a_distant_duplicate_target_word():
    source = Image.new("RGB", (400, 240), "white")
    target = source.copy()
    source_draw = ImageDraw.Draw(source)
    target_draw = ImageDraw.Draw(target)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if not font_path.exists():
        pytest.skip("DejaVu font is unavailable")
    font = ImageFont.truetype(str(font_path), 42)
    source_draw.text((210, 55), "SPENCER", fill="black", font=font)
    target_draw.text((210, 55), "ART", fill="black", font=font)
    source_draw.text((210, 140), "ART GALLERY", fill="black", font=font)
    target_draw.text((210, 140), "ART GALLERY", fill="black", font=font)
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "text SPENCER",
                "spatial_hint": "upper center of the window above ART GALLERY",
                "geometry": "sparse_marks",
                "mask_method": "box",
            },
            {
                "candidate_id": 1,
                "image_side": "target",
                "ref": "text ART",
                "spatial_hint": "upper center of the window replacing SPENCER",
                "geometry": "sparse_marks",
                "mask_method": "box",
            },
        ],
    }
    localized = [
        {"candidate_id": 0, "bbox_2d": [510, 190, 960, 405]},
        {"candidate_id": 1, "bbox_2d": [510, 405, 950, 595]},
    ]
    refined = refine_text_localization_boxes(
        source,
        target,
        "building_surface_text_editing",
        plan,
        localized,
    )
    assert refined[0]["bbox_2d"] == refined[1]["bbox_2d"]
    assert refined[0]["bbox_2d"][3] < 550
    override = refined[1]["bbox_refinement"]["correspondence_anchor_override"]
    assert override["rule"] == "same_location_text_uses_source_anchor_v1"
    assert override["rejected_target_bbox_2d"] == [510.0, 405.0, 950.0, 595.0]
    assert override["spatial_hint_token_overlap"] >= 0.5


def test_dense_aggregate_box_gets_bounded_recall_margin_but_surface_completion_does_not():
    dense_plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "geometry": "dense_region",
                "mask_method": "sam",
                "region_mode": "aggregate_region",
            }
        ],
    }
    localized = [{"candidate_id": 0, "bbox_2d": [40, 550, 370, 860]}]
    expanded = expand_dense_aggregate_localization_boxes(dense_plan, localized)
    assert expanded[0]["bbox_2d"] == [13.6, 503.5, 396.4, 906.5]
    assert expanded[0]["bbox_refinement"]["rule"] == (
        "dense_aggregate_recall_margin_v1"
    )
    assert localized[0]["bbox_2d"] == [40, 550, 370, 860]

    surface_plan = {
        **dense_plan,
        "plan_policy_overrides": [
            {
                "rule": "appearance_surface_uses_all_changed_sections_v1",
                "candidate_id": 0,
            }
        ],
    }
    assert expand_dense_aggregate_localization_boxes(surface_plan, localized) == localized


def test_negative_space_plan_is_explicit_and_locator_targets_the_full_cutout():
    prompt = build_observation_prompt(
        "action_editing",
        "Rotate the cookie so the bite is at the bottom right.",
    )
    assert "negative_space" in prompt
    assert "carrier_ref" in prompt
    plan = parse_observation(
        json.dumps(
            {
                "realized_edit": "the cookie bite moved around the edge",
                "mask_mode": "regions",
                "localization_items": [
                    {
                        "image_side": "source",
                        "role": "edit_region",
                        "edit_op": "remove",
                        "ref": "empty bite cutout on the left",
                        "spatial_hint": "left edge of the cookie",
                        "geometry": "dense_region",
                        "mask_method": "sam",
                        "region_mode": "object",
                        "mask_density": "dense",
                        "negative_space": True,
                        "carrier_ref": "chocolate chip cookie",
                    }
                ],
                "confidence": "high",
            }
        )
    )
    item = plan["localization_items"][0]
    assert item["negative_space"] is True
    assert item["carrier_ref"] == "chocolate chip cookie"
    locator = build_grounding_prompt("action_editing", "ignored", plan)
    assert 'complete empty region "empty bite cutout on the left"' in locator
    assert 'inside "chocolate chip cookie"' in locator
    grounded = grounding_from_localization(
        plan, [{"candidate_id": 0, "bbox_2d": [100, 300, 350, 650]}]
    )
    assert grounded["source"][0]["negative_space"] is True
    assert grounded["source"][0]["carrier_ref"] == "chocolate chip cookie"


def test_negative_space_inverse_recovers_concavity_without_filling_carrier():
    shape = (100, 100)
    carrier = np.zeros(shape, dtype=np.uint8)
    cv2.circle(carrier, (50, 50), 35, 1, -1)
    # An open bite on the left edge of an otherwise convex carrier.
    cv2.circle(carrier, (20, 50), 15, 0, -1)
    negative, audit = _negative_space_from_carrier_mask(
        carrier, [50, 350, 300, 650], shape
    )
    assert negative[50, 20] == 1
    assert negative[50, 50] == 0
    assert negative.sum() < carrier.sum()
    assert audit["negative_space_strategy"] == "carrier_convex_inverse"


def test_bbox_locator_recovers_ordered_repeated_json_keys():
    malformed = (
        '[{"bbox_2d":[100,110,200,210],'
        '"bbox_2d":[300,310,400,410]}]'
    )
    parsed = parse_bbox_localization(malformed, [0, 1])
    assert parsed == [
        {"candidate_id": 0, "ref": "", "bbox_2d": [100.0, 110.0, 200.0, 210.0]},
        {"candidate_id": 1, "ref": "", "bbox_2d": [300.0, 310.0, 400.0, 410.0]},
    ]

    duplicate = (
        '[{"bbox_2d":[100,110,200,210],'
        '"bbox_2d":[300,310,400,410],'
        '"bbox_2d":[100,110,200,210]}]'
    )
    with pytest.raises(ValueError, match="candidate count mismatch"):
        parse_bbox_localization(duplicate, [0, 1])


def test_bbox_locator_accepts_candidate_id_at_start_of_qwen_label():
    response = json.dumps(
        [
            {"bbox_2d": [10, 20, 30, 40], "label": "0 | Image 1 | puddle"},
            {"bbox_2d": [50, 60, 70, 80], "label": "1 | Image 2 | dry soil"},
        ]
    )
    assert parse_bbox_localization(response, [0, 1]) == [
        {"candidate_id": 0, "ref": "", "bbox_2d": [10.0, 20.0, 30.0, 40.0]},
        {"candidate_id": 1, "ref": "", "bbox_2d": [50.0, 60.0, 70.0, 80.0]},
    ]


def test_bbox_locator_accepts_explicit_member_slot_labels():
    response = json.dumps(
        [
            {"bbox_2d": [10, 20, 30, 40], "label": "candidate_id=0 member=0"},
            {"bbox_2d": [50, 60, 70, 80], "label": "candidate_id=0|member=1"},
        ]
    )
    parsed = parse_bbox_localization(response, [0], multi_candidate_ids=[0])
    assert [item["candidate_id"] for item in parsed] == [0, 0]
    assert [item["member_index"] for item in parsed] == [0, 1]


def test_bbox_locator_accepts_native_semantic_labels_and_keeps_equal_cross_image_boxes():
    response = json.dumps(
        [
            {"bbox_2d": [0, 0, 1000, 510], "label": "Image 1 | sky"},
            {"bbox_2d": [0, 0, 1000, 510], "label": "Image 2 | sky"},
        ]
    )
    assert parse_bbox_localization(response, [0, 1]) == [
        {"candidate_id": 0, "ref": "", "bbox_2d": [0.0, 0.0, 1000.0, 510.0]},
        {"candidate_id": 1, "ref": "", "bbox_2d": [0.0, 0.0, 1000.0, 510.0]},
    ]


def test_bbox_locator_binds_semantic_labels_for_one_multi_instance_query():
    response = json.dumps(
        [
            {"bbox_2d": [10, 20, 30, 40], "label": "blue backpack"},
            {"bbox_2d": [50, 60, 70, 80], "label": "blue backpack"},
        ]
    )
    parsed = parse_bbox_localization(response, [0], multi_candidate_ids=[0])
    assert [item["candidate_id"] for item in parsed] == [0, 0]
    assert [item["member_index"] for item in parsed] == [0, 1]


def test_bbox_locator_binds_mixed_native_ids_for_one_multi_instance_query():
    response = json.dumps(
        [
            {"bbox_2d": [10, 20, 30, 40], "candidate_id": 0},
            {"bbox_2d": [50, 60, 70, 80], "label": "blue backpack"},
        ]
    )
    parsed = parse_bbox_localization(response, [0], multi_candidate_ids=[0])
    assert [item["candidate_id"] for item in parsed] == [0, 0]
    assert [item["member_index"] for item in parsed] == [0, 1]


def test_bbox_locator_recovers_last_coordinates_from_truncated_single_multi_query():
    response = """[
      {"bbox_2d":[10,20,30,40],"label":"candidate_id=0"},
      {"bbox_2d":[50,60,70,80],"label":"candidate_id=0 and truncated
    """
    parsed = parse_bbox_localization(response, [0], multi_candidate_ids=[0])
    assert [item["bbox_2d"] for item in parsed] == [
        [10.0, 20.0, 30.0, 40.0],
        [50.0, 60.0, 70.0, 80.0],
    ]
    assert all(
        item["bbox_parse_recovery"] == "partial_bbox_coordinates" for item in parsed
    )


def test_bbox_locator_reorders_complete_native_ids_but_rejects_duplicate_ids():
    reversed_response = json.dumps(
        [
            {"bbox_2d": [500, 500, 800, 800], "label": "candidate_id=1"},
            {"bbox_2d": [100, 100, 300, 300], "label": "candidate_id=0"},
        ]
    )
    parsed = parse_bbox_localization(reversed_response, [0, 1])
    assert [item["candidate_id"] for item in parsed] == [0, 1]
    assert parsed[0]["bbox_2d"] == [100.0, 100.0, 300.0, 300.0]

    duplicate_ids = json.dumps(
        [
            {"bbox_2d": [100, 100, 300, 300], "label": "candidate_id=0"},
            {"bbox_2d": [500, 500, 800, 800], "label": "candidate_id=0"},
        ]
    )
    with pytest.raises(ValueError, match="duplicate bbox candidate_id"):
        parse_bbox_localization(duplicate_ids, [0, 1])


def test_bbox_locator_unions_multiple_detections_only_for_aggregate_candidates():
    split_group = json.dumps(
        [
            {"bbox_2d": [100, 200, 250, 600], "label": "candidate_id=0"},
            {"bbox_2d": [400, 150, 550, 650], "label": "candidate_id=0"},
            {"bbox_2d": [120, 210, 270, 610], "label": "candidate_id=1"},
            {"bbox_2d": [420, 160, 570, 660], "label": "candidate_id=1"},
        ]
    )
    parsed = parse_bbox_localization(
        split_group, [0, 1], aggregate_candidate_ids=[0, 1]
    )
    assert parsed == [
        {"candidate_id": 0, "ref": "", "bbox_2d": [100.0, 150.0, 550.0, 650.0]},
        {"candidate_id": 1, "ref": "", "bbox_2d": [120.0, 160.0, 570.0, 660.0]},
    ]


def test_bbox_locator_unions_subboxes_for_complete_single_object_checklist():
    response = json.dumps(
        [
            {"bbox_2d": [100, 200, 180, 300], "label": "candidate_id=0"},
            {"bbox_2d": [185, 200, 270, 300], "label": "candidate_id=0"},
            {"bbox_2d": [600, 200, 800, 400], "label": "candidate_id=1"},
        ]
    )
    assert parse_bbox_localization(response, [0, 1]) == [
        {
            "candidate_id": 0,
            "ref": "",
            "bbox_2d": [100.0, 200.0, 270.0, 300.0],
            "bbox_parse_recovery": "single_candidate_subbox_union",
        },
        {"candidate_id": 1, "ref": "", "bbox_2d": [600.0, 200.0, 800.0, 400.0]},
    ]


def test_bbox_locator_recovers_complete_objects_from_truncated_multi_response():
    response = """```json
[
  {"bbox_2d":[10,20,30,40],"label":"candidate_id=0"},
  {"bbox_2d":[50,60,70,80],"label":"candidate_id=0"},
  {"bbox_2d":[90,100,110,120],"label":"candidate_id=1"},
  {"bbox_2d":[130,140
"""
    parsed = parse_bbox_localization(response, [0, 1], multi_candidate_ids=[0])
    assert [item["candidate_id"] for item in parsed] == [0, 0, 1]
    assert all(item["bbox_parse_recovery"] == "partial_json_objects" for item in parsed)


def test_bbox_locator_preserves_multi_instance_member_boxes_and_deduplicates_exact_repeats():
    response = json.dumps(
        [
            {"bbox_2d": [39, 547, 92, 692], "label": "candidate_id=0"},
            {"bbox_2d": [270, 532, 327, 650], "label": "candidate_id=0"},
            {"bbox_2d": [270, 532, 327, 650], "label": "candidate_id=0"},
            {"bbox_2d": [0, 570, 50, 692], "label": "candidate_id=1"},
            {"bbox_2d": [295, 547, 345, 665], "label": "candidate_id=1"},
        ]
    )
    parsed = parse_bbox_localization(
        response, [0, 1], multi_candidate_ids=[0, 1]
    )
    assert parsed == [
        {
            "candidate_id": 0,
            "ref": "",
            "bbox_2d": [39.0, 547.0, 92.0, 692.0],
            "member_index": 0,
        },
        {
            "candidate_id": 0,
            "ref": "",
            "bbox_2d": [270.0, 532.0, 327.0, 650.0],
            "member_index": 1,
        },
        {
            "candidate_id": 1,
            "ref": "",
            "bbox_2d": [0.0, 570.0, 50.0, 692.0],
            "member_index": 0,
        },
        {
            "candidate_id": 1,
            "ref": "",
            "bbox_2d": [295.0, 547.0, 345.0, 665.0],
            "member_index": 1,
        },
    ]


def test_multi_instance_plan_expands_members_for_independent_sam_anchors():
    plan = parse_observation(
        json.dumps(
            {
                "realized_edit": "all blue backpacks became green",
                "mask_mode": "regions",
                "localization_items": [
                    {
                        "image_side": "source",
                        "role": "edit_region",
                        "edit_op": "change",
                        "ref": "all blue backpacks worn by the group",
                        "spatial_hint": "across the line of people",
                        "geometry": "semantic_object",
                        "mask_method": "sam",
                        "region_mode": "multi_instance",
                        "selection_mode": "all_matching",
                        "mask_extent": "whole_object",
                        "expected_count": 2,
                        "mask_density": "object",
                    }
                ],
                "confidence": "high",
            }
        )
    )
    localized = parse_bbox_localization(
        json.dumps(
            [
                {"bbox_2d": [10, 20, 30, 50], "label": "candidate_id=0"},
                {"bbox_2d": [60, 20, 80, 50], "label": "candidate_id=0"},
            ]
        ),
        [0],
        multi_candidate_ids=[0],
    )
    grounded = grounding_from_localization(plan, localized)
    assert [item["bbox_2d"] for item in grounded["source"]] == [
        [10.0, 20.0, 30.0, 50.0],
        [60.0, 20.0, 80.0, 50.0],
    ]
    assert [item["member_index"] for item in grounded["source"]] == [0, 1]
    assert grounded["semantic_qc_flags"] == []


def test_multi_instance_count_mismatch_is_audited_without_dropping_boxes():
    plan = parse_observation(
        json.dumps(
            {
                "realized_edit": "three plates changed color",
                "mask_mode": "regions",
                "localization_items": [
                    {
                        "image_side": "source",
                        "role": "edit_region",
                        "edit_op": "change",
                        "ref": "all brown plates",
                        "spatial_hint": "left shelf",
                        "region_mode": "multi_instance",
                        "selection_mode": "all_matching",
                        "mask_extent": "whole_object",
                        "expected_count": 3,
                    }
                ],
                "confidence": "medium",
            }
        )
    )
    grounded = grounding_from_localization(
        plan,
        [{"candidate_id": 0, "member_index": 0, "bbox_2d": [10, 10, 20, 20]}],
    )
    assert len(grounded["source"]) == 1
    assert any(flag.startswith("COUNT_MISMATCH") for flag in grounded["semantic_qc_flags"])
    assert any(flag.startswith("MULTI_INSTANCE_SINGLETON") for flag in grounded["semantic_qc_flags"])


def test_bbox_spatial_order_mismatch_is_audited_without_an_mllm_retry():
    plan = parse_observation(
        json.dumps(
            {
                "realized_edit": "five liquids changed color",
                "mask_mode": "regions",
                "localization_items": [
                    {
                        "image_side": "source",
                        "role": "edit_region",
                        "edit_op": "change",
                        "ref": "left liquid",
                        "spatial_hint": "far left of the group",
                    },
                    {
                        "image_side": "source",
                        "role": "edit_region",
                        "edit_op": "change",
                        "ref": "right liquid",
                        "spatial_hint": "far right of the group",
                    },
                ],
                "confidence": "medium",
            }
        )
    )
    grounded = grounding_from_localization(
        plan,
        [
            {"candidate_id": 0, "bbox_2d": [700, 100, 800, 200]},
            {"candidate_id": 1, "bbox_2d": [100, 100, 200, 200]},
        ],
    )
    assert any(
        flag.startswith("SPATIAL_ORDER_MISMATCH")
        for flag in grounded["semantic_qc_flags"]
    )


def test_planner_prompts_encode_source_primary_color_and_whole_actor_scope():
    color = build_observation_prompt(
        "color_change", "Change all blue backpacks to green."
    )
    action = build_observation_prompt(
        "action_editing", "Make the boy raise his hand."
    )
    assert "localize the affected source content only" in color
    assert "region_mode=multi_instance" in color
    assert "different explicit roles or positions" in color
    assert "never only one member" in color
    assert "mask_extent=surface_region" in color
    assert "mask_extent=whole_actor" in action

    locator = build_grounding_prompt(
        "color_change",
        "ignored",
        {
            "mask_mode": "regions",
            "localization_items": [
                {
                    "candidate_id": 0,
                    "image_side": "source",
                    "ref": "blue backpacks",
                    "spatial_hint": "across the line of people",
                    "geometry": "semantic_object",
                    "region_mode": "multi_instance",
                    "selection_mode": "all_matching",
                    "mask_extent": "whole_object",
                    "expected_count": 8,
                }
            ],
        },
    )
    assert "exactly 8 different members" in locator
    assert '"candidate_id":0,"member_index":0' in locator
    assert '"candidate_id":0,"member_index":7' in locator


def test_planner_hard_limit_rejects_overlong_complete_plan():
    items = [
        {
            "image_side": "target",
            "role": "edit_region",
            "edit_op": "add",
            "ref": f"added item {index}",
            "spatial_hint": f"position {index}",
            "geometry": "semantic_object",
            "mask_method": "sam",
            "region_mode": "object",
            "mask_density": "object",
        }
        for index in range(9)
    ]
    with pytest.raises(ValueError, match="exceeds hard limit"):
        parse_observation(
            json.dumps(
                {
                    "realized_edit": "nine objects were added",
                    "mask_mode": "regions",
                    "localization_items": items,
                    "confidence": "high",
                }
            )
        )


def test_parse_retry_is_corrective_and_uses_stage_specific_token_limits():
    grounder = Qwen35ScaleEditGrounder.__new__(Qwen35ScaleEditGrounder)
    grounder.args = argparse.Namespace(
        request_batch_size=1,
        max_images_per_generate=8,
        parse_retries=1,
        planner_max_new_tokens=2048,
        locator_max_new_tokens=512,
        max_new_tokens=1024,
    )
    plan = json.dumps(
        {
            "realized_edit": "one plug was added",
            "mask_mode": "regions",
            "localization_items": [
                {
                    "image_side": "target",
                    "role": "edit_region",
                    "edit_op": "add",
                    "ref": "added power plug",
                    "spatial_hint": "beside the existing plug",
                    "geometry": "semantic_object",
                    "mask_method": "sam",
                    "region_mode": "object",
                    "mask_density": "object",
                }
            ],
            "confidence": "high",
        }
    )
    responses = [
        '{"realized_edit":"truncated"',
        plan,
        '[{"bbox_2d":[100,100,400,500],"label":"plug"},'
        '{"bbox_2d":[500,100,800,500],"label":"extra"}]',
        '[{"bbox_2d":[100,100,400,500],"label":"candidate_id=0"}]',
    ]
    calls = []

    def generate(conversations, *, max_tokens=None):
        calls.append((conversations, max_tokens))
        return [responses.pop(0)]

    grounder.generate = generate
    payload = grounder.infer(
        [
            {
                "source": Image.new("RGB", (64, 64), "white"),
                "target": Image.new("RGB", (64, 64), "white"),
                "instruction": "Add one more power plug next to the existing one.",
                "final_task": "object_addition",
            }
        ]
    )[0]

    assert responses == []
    assert [max_tokens for _, max_tokens in calls] == [2048, 2048, 512, 512]
    assert [message["role"] for message in calls[1][0][0]] == [
        "user",
        "assistant",
        "user",
    ]
    assert "could not be parsed" in calls[1][0][0][-1]["content"][0]["text"]
    assert "Every other ID returns exactly one box" in calls[3][0][0][-1]["content"][0]["text"]
    assert payload["ground_parse_ok"] is True
    assert len(payload["observation"]["attempts"]) == 2
    assert len(payload["grounding"]["attempts"]) == 2


def test_corrupt_image_cell_is_classified_for_row_level_skip():
    with pytest.raises(CorruptSampleImageError, match="source_image"):
        _decode_sample_images(
            {
                "source_image": b"<html>not an image</html>",
                "edited_image": b"also not reached",
            }
        )


def test_grounding_job_skips_only_corrupt_rows_and_preserves_source_row_idx(tmp_path):
    def png_bytes(color):
        buffer = io.BytesIO()
        Image.new("RGB", (12, 10), color).save(buffer, format="PNG")
        return buffer.getvalue()

    valid_image = png_bytes("white")
    raw_rows = [
        {
            "sample_id": "bad",
            "source_relative_path": "bad.png",
            "edit_task": "object_addition",
            "final_task": "object_addition",
            "original_instruction": "add an object",
            "final_instruction": "Add an object.",
            "source_image": b"<html>bad image</html>",
            "edited_image": valid_image,
        },
        {
            "sample_id": "good",
            "source_relative_path": "good.png",
            "edit_task": "tone_adjustment",
            "final_task": "tone_adjustment",
            "original_instruction": "grayscale",
            "final_instruction": "Convert the whole image to grayscale.",
            "source_image": valid_image,
            "edited_image": valid_image,
        },
    ]
    input_path = tmp_path / "part-00000.parquet"
    output_path = tmp_path / "grounding" / input_path.name
    pq.write_table(pa.Table.from_pylist(raw_rows), input_path)

    class FakeGrounder:
        def infer(self, samples):
            assert len(samples) == 1
            return [
                {
                    "schema_version": 1,
                    "prompt_version": "test",
                    "mask_mode": "full_image",
                    "source": [],
                    "target": [],
                    "protected_foreground": [],
                    "ground_parse_ok": True,
                }
            ]

    class FakeQueue:
        def __init__(self):
            self.messages = []

        def put(self, message):
            self.messages.append(message)

    queue = FakeQueue()
    summary = process_job(
        GroundingJob(str(input_path), str(output_path), 2),
        FakeGrounder(),
        argparse.Namespace(
            batch_size=2,
            fail_fast=True,
            model_path=Path("model"),
            compression="zstd",
        ),
        queue,
        worker_index=0,
    )

    assert summary["input_rows"] == 2
    assert summary["rows"] == 1
    assert summary["skipped_images"] == 1
    assert summary["skipped_samples"][0]["sample_id"] == "bad"
    output_rows = pq.read_table(output_path).to_pylist()
    assert [(row["row_idx"], row["sample_id"]) for row in output_rows] == [(1, "good")]
    assert any(message.get("kind") == "log" for message in queue.messages)


def test_count_change_uses_exactly_planner_and_locator_calls():
    grounder = Qwen35ScaleEditGrounder.__new__(Qwen35ScaleEditGrounder)
    grounder.args = argparse.Namespace(
        request_batch_size=1,
        max_images_per_generate=8,
        parse_retries=0,
    )
    observation = json.dumps(
        {
            "realized_edit": "two oranges were added beside the original",
            "mask_mode": "regions",
            "localization_items": [
                {
                    "image_side": "target",
                    "role": "edit_region",
                    "edit_op": "add",
                    "ref": "left added orange",
                    "spatial_hint": "left of the unchanged orange",
                    "geometry": "semantic_object",
                    "mask_method": "sam",
                    "region_mode": "object",
                    "mask_density": "object",
                },
                {
                    "image_side": "target",
                    "role": "edit_region",
                    "edit_op": "add",
                    "ref": "right added orange",
                    "spatial_hint": "right of the unchanged orange",
                    "geometry": "semantic_object",
                    "mask_method": "sam",
                    "region_mode": "object",
                    "mask_density": "object",
                },
            ],
            "confidence": "high",
        }
    )
    grounding = json.dumps(
        [
            {"candidate_id": 0, "bbox_2d": [70, 480, 330, 765]},
            {"candidate_id": 1, "bbox_2d": [670, 480, 930, 765]},
        ]
    )
    responses = [
        [observation],
        [grounding],
    ]

    def generate(_conversations, **_kwargs):
        return responses.pop(0)

    grounder.generate = generate
    sample = {
        "source": Image.new("RGB", (1024, 1024), "white"),
        "target": Image.new("RGB", (1024, 1024), "white"),
        "instruction": "Add two oranges, one on each side of the existing orange.",
        "final_task": "count_change",
    }
    payload = grounder.infer([sample])[0]
    assert responses == []
    assert payload["target"][0]["bbox_2d"] == [70.0, 480.0, 330.0, 765.0]
    assert payload["target"][1]["bbox_2d"] == [670.0, 480.0, 930.0, 765.0]


def test_second_pass_is_a_fresh_bbox_only_conversation():
    grounder = Qwen35ScaleEditGrounder.__new__(Qwen35ScaleEditGrounder)
    grounder.args = argparse.Namespace(
        request_batch_size=1,
        max_images_per_generate=8,
        parse_retries=0,
    )
    observation = json.dumps(
        {
            "realized_edit": "one orange was added",
            "mask_mode": "regions",
            "localization_items": [
                {
                    "image_side": "target",
                    "role": "edit_region",
                    "edit_op": "add",
                    "ref": "left added orange",
                    "spatial_hint": "left of the original orange",
                    "geometry": "semantic_object",
                    "mask_method": "sam",
                    "region_mode": "object",
                    "mask_density": "object",
                }
            ],
            "confidence": "high",
        }
    )
    calls = []

    def generate(conversations, **_kwargs):
        calls.extend(conversations)
        if len(calls) == 1:
            return [observation]
        return ['[{"candidate_id":0,"bbox_2d":[100,200,400,700]}]']

    grounder.generate = generate
    sample = {
        "source": Image.new("RGB", (100, 100), "white"),
        "target": Image.new("RGB", (100, 100), "white"),
        "instruction": "Add one orange to the left of the original.",
        "final_task": "object_addition",
    }
    payload = grounder.infer([sample])[0]
    assert len(calls) == 2
    assert len(calls[1]) == 1
    second_text = " ".join(
        part.get("text", "")
        for part in calls[1][0]["content"]
        if part.get("type") == "text"
    )
    assert "Locate these targets in this edited image" in second_text
    assert "realized_edit" not in second_text
    assert "Corrected edit instruction" not in second_text
    assert "mask_method" not in second_text
    assert "mask_density" not in second_text
    second_images = [
        part
        for part in calls[1][0]["content"]
        if part.get("type") == "image"
    ]
    assert len(second_images) == 1
    assert second_images[0]["image"] is sample["target"]
    assert payload["target"][0]["ref"] == "left added orange"


def test_mixed_side_locator_keeps_one_two_image_call():
    source = Image.new("RGB", (32, 24), "red")
    target = Image.new("RGB", (32, 24), "blue")
    plan = {
        "localization_items": [
            {"candidate_id": 0, "image_side": "source", "ref": "old object"},
            {"candidate_id": 1, "image_side": "target", "ref": "new object"},
        ]
    }
    conversation = Qwen35ScaleEditGrounder._localization_conversation(
        source, target, "locate", plan
    )
    images = [
        part
        for part in conversation[0]["content"]
        if part.get("type") == "image"
    ]
    assert [part["image"] for part in images] == [source, target]


def test_source_surface_locator_keeps_pair_evidence_in_same_call():
    source = Image.new("RGB", (32, 24), "red")
    target = Image.new("RGB", (32, 24), "blue")
    plan = {
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "changed facade",
                "mask_extent": "surface_region",
            }
        ]
    }
    conversation = Qwen35ScaleEditGrounder._localization_conversation(
        source, target, "locate", plan
    )
    images = [
        part["image"]
        for part in conversation[0]["content"]
        if part.get("type") == "image"
    ]
    assert images == [source, target]


def test_vllm_backend_preserves_chat_template_images_and_output_order():
    class FakeProcessor:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, conversation, **kwargs):
            self.calls.append((conversation, kwargs))
            return f"prompt-{len(self.calls)}"

    class FakeCandidate:
        def __init__(self, text):
            self.text = text

    class FakeResult:
        def __init__(self, text):
            self.outputs = [FakeCandidate(text)]

    class FakeLLM:
        def __init__(self):
            self.requests = None
            self.sampling_params = None
            self.use_tqdm = None

        def generate(self, requests, sampling_params, *, use_tqdm):
            self.requests = requests
            self.sampling_params = sampling_params
            self.use_tqdm = use_tqdm
            return [FakeResult(f"result-{index}") for index in range(len(requests))]

    grounder = Qwen35ScaleEditGrounder.__new__(Qwen35ScaleEditGrounder)
    grounder.args = argparse.Namespace(max_pixels=1_310_720)
    grounder.processor = FakeProcessor()
    grounder.model = FakeLLM()
    grounder.sampling_params = object()
    conversations = [
        grounder._conversation(
            Image.new("RGB", (32, 24), "red"),
            Image.new("RGB", (32, 24), "blue"),
            f"prompt {index}",
        )
        for index in range(2)
    ]

    outputs = grounder._generate_vllm(conversations)

    assert outputs == ["result-0", "result-1"]
    assert grounder.model.use_tqdm is False
    assert grounder.model.sampling_params is grounder.sampling_params
    for index, request in enumerate(grounder.model.requests):
        assert request["prompt"] == f"prompt-{index + 1}"
        assert len(request["multi_modal_data"]["image"]) == 2
        assert request["mm_processor_kwargs"] == {"max_pixels": 1_310_720}
    for _, kwargs in grounder.processor.calls:
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }


def test_observation_plan_and_bbox_join_are_deterministic():
    plan = parse_observation(
        json.dumps(
            {
                "realized_edit": "a toy was added to a stand",
                "mask_mode": "regions",
                "localization_items": [
                    {
                        "image_side": "source",
                        "role": "edit_region",
                        "edit_op": "add",
                        "ref": "empty white stand",
                        "spatial_hint": "left foreground",
                        "mask_method": "sam",
                        "region_mode": "object",
                        "mask_density": "object",
                    },
                    {
                        "image_side": "target",
                        "role": "edit_region",
                        "edit_op": "add",
                        "ref": "yellow plush toy",
                        "spatial_hint": "on the left stand",
                        "mask_method": "sam",
                        "region_mode": "object",
                        "mask_density": "object",
                    },
                ],
                "confidence": "high",
            }
        )
    )
    # The source support is rejected before Round 2 and candidate IDs are
    # reassigned after filtering, so the locator sees exactly one item.
    assert [item["ref"] for item in plan["localization_items"]] == [
        "yellow plush toy"
    ]
    localized = parse_bbox_localization(
        '[{"candidate_id":0,"bbox_2d":[200,300,500,800]}]', [0]
    )
    grounding = grounding_from_localization(plan, localized)
    assert grounding["source"] == []
    assert grounding["target"][0]["ref"] == "yellow plush toy"
    assert grounding["target"][0]["bbox_2d"] == [200.0, 300.0, 500.0, 800.0]


def test_observation_parser_accepts_wrapped_json():
    parsed = parse_observation(
        "answer:\n```json\n"
        + json.dumps(
            {
                "realized_edit": "the title changed",
                "mask_mode": "regions",
                "localization_items": [
                    {
                        "image_side": "target",
                        "role": "edit_region",
                        "edit_op": "change",
                        "ref": "new title",
                        "spatial_hint": "top of the interface",
                        "geometry": "sparse_marks",
                        "mask_method": "box",
                        "region_mode": "aggregate_region",
                        "mask_density": "sparse",
                    }
                ],
                "confidence": "high",
            }
        )
        + "\n```"
    )
    assert parsed["mask_mode"] == "regions"
    assert parsed["localization_items"][0]["ref"] == "new title"


def test_full_image_mask_does_not_call_sam():
    sample = {
        "source": Image.new("RGB", (11, 7), "black"),
        "target": Image.new("RGB", (11, 7), "white"),
    }
    ground_row = {
        "qc_flag": "OK",
        "final_task": "tone_adjustment",
        "ground_json": json.dumps(
            {
                "mask_mode": "full_image",
                "source": [],
                "target": [],
                "protected_foreground": [],
            }
        ),
    }
    result = annotate_sample(None, sample, ground_row, "sam-test")
    assert result["mask_source"] == "full_image"
    assert result["mask"].shape == (7, 11)
    assert np.all(result["mask"] == 1)


def test_direct_text_box_mask_does_not_call_sam_and_maps_target():
    sample = {
        "source": Image.new("RGB", (100, 80), "white"),
        "target": Image.new("RGB", (200, 160), "white"),
    }
    item = {
        "ref": "headline text glyphs",
        "bbox_2d": [100, 200, 500, 400],
        "mask_method": "box",
        "region_mode": "aggregate_region",
        "mask_density": "sparse",
    }
    ground_row = {
        "qc_flag": "OK",
        "final_task": "gui_interface_text_editing",
        "ground_json": json.dumps(
            {
                "mask_mode": "regions",
                "source": [item],
                "target": [item],
                "protected_foreground": [],
            }
        ),
    }
    result = annotate_sample(None, sample, ground_row, "sam-test")
    assert result["mask"].shape == (80, 100)
    assert result["mask"].sum() > 0
    assert result["mask_source"] == "direct_box"
    assert result["qc_flag"] == "OK"
    assert len(result["instances"]) == 2
    assert result["instances"][1]["mapped_from_target"] is True


def test_semantic_object_cleanup_keeps_anchor_components_and_drops_noise():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:45, 30:45] = 1
    mask[50:58, 50:58] = 1
    mask[35:37, 55:57] = 1
    mask[5:15, 5:15] = 1
    item = {
        "bbox_2d": [300, 300, 600, 600],
        "mask_method": "sam",
        "region_mode": "object",
        "mask_density": "object",
    }
    cleaned, audit = _clean_semantic_object_mask(mask, item, mask.shape, {})
    assert cleaned.sum() == 225 + 64
    assert audit["raw_component_count"] == 4
    assert audit["kept_component_count"] == 2
    assert audit["removed_component_count"] == 2
    assert audit["removed_area"] == mask.sum() - cleaned.sum()
    assert audit["output_guard_bbox_2d"][0] < 300
    assert audit["output_guard_bbox_2d"][0] > 200


def test_multi_instance_member_cleanup_uses_each_member_anchor():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:45, 30:45] = 1
    mask[70:80, 70:80] = 1
    item = {
        "bbox_2d": [280, 280, 470, 470],
        "mask_method": "sam",
        "region_mode": "multi_instance",
        "mask_density": "object",
    }
    cleaned, audit = _clean_semantic_object_mask(mask, item, mask.shape, {})
    assert cleaned[35, 35] == 1
    assert cleaned[75, 75] == 0
    assert audit["removed_component_count"] == 1


def test_compact_interaction_completion_closes_pcs_holes_and_fragments():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:60, 30:44] = 1
    mask[30:60, 47:61] = 1
    mask[38:42, 35:39] = 0
    item = {
        "ref": "driver's left hand holding phone",
        "bbox_2d": [250, 250, 650, 650],
        "mask_method": "sam",
        "region_mode": "object",
        "mask_density": "object",
    }
    cleaned, audit = _clean_semantic_object_mask(
        mask, item, mask.shape, {"mask_source": "pcs"}
    )
    assert audit["completion_applied"] is True
    assert audit["completion_rule"] == "compact_interaction_pcs_closing_v1"
    assert audit["completion_radius"] == 2
    assert audit["completion_added_area"] > 0
    assert audit["completion_component_count_before"] == 2
    assert audit["completion_component_count_after"] == 1
    assert cleaned[39, 36] == 1


@pytest.mark.parametrize(
    ("ref", "mask_source"),
    (("gold mirror frame", "pcs"), ("driver's left hand", "pvs")),
)
def test_compact_interaction_completion_does_not_touch_other_masks(
    ref, mask_source
):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:60, 30:60] = 1
    mask[38:42, 38:42] = 0
    item = {
        "ref": ref,
        "bbox_2d": [250, 250, 650, 650],
        "mask_method": "sam",
        "region_mode": "object",
        "mask_density": "object",
    }
    cleaned, audit = _clean_semantic_object_mask(
        mask, item, mask.shape, {"mask_source": mask_source}
    )
    assert audit["completion_applied"] is False
    assert np.array_equal(cleaned, mask)


def test_target_mapping_margin_is_small_and_geometry_aware():
    thin = np.zeros((100, 100), dtype=np.uint8)
    thin[20:80, 20] = 1
    thin[20:80, 79] = 1
    thin[20, 20:80] = 1
    thin[79, 20:80] = 1
    filled = np.zeros((100, 100), dtype=np.uint8)
    filled[20:80, 20:80] = 1
    object_item = {
        "mask_method": "sam",
        "region_mode": "object",
        "mask_density": "object",
    }
    assert _target_map_dilate_frac(object_item, thin) == 0.0025
    assert _target_map_dilate_frac(object_item, filled) == 0.005
    assert _target_map_dilate_frac(
        {**object_item, "mask_density": "dense"}, filled
    ) == 0.004
    assert _target_map_dilate_frac(
        {**object_item, "mask_method": "box"}, filled
    ) == 0.002


def test_product_extraction_post_policy_prevents_old_subject_holes():
    payload = {
        "ground_parse_ok": True,
        "mask_mode": "protect_foreground",
        "source": [],
        "target": [],
        "protected_foreground": [
            {
                "ref": "person rollerblading",
                "bbox_2d": [700, 200, 900, 800],
                "mask_method": "sam",
            }
        ],
    }
    updated = apply_task_post_policy(
        "part_extraction",
        "Extract the person over a white background, product photography style.",
        payload,
    )
    assert updated["mask_mode"] == "full_image"
    assert updated["protected_foreground"] == []
    assert updated["route_override"]["original_mask_mode"] == "protect_foreground"
    assert payload["mask_mode"] == "protect_foreground"

    local = apply_task_post_policy(
        "part_extraction",
        "Remove the outer shell of one chestnut to reveal its kernel.",
        payload,
    )
    assert local is payload


def test_pure_add_remove_post_policy_drops_opposite_support_side():
    removal = {
        "ground_parse_ok": True,
        "mask_mode": "regions",
        "source": [{"ref": "water puddle", "bbox_2d": [250, 550, 750, 950]}],
        "target": [{"ref": "revealed wet soil", "bbox_2d": [250, 550, 750, 950]}],
        "protected_foreground": [],
    }
    updated = apply_task_post_policy(
        "object_removal", "Remove the water puddle.", removal
    )
    assert updated["source"] == removal["source"]
    assert updated["target"] == []
    assert updated["side_override"]["dropped_side"] == "target"
    assert removal["target"]

    addition = {
        **removal,
        "source": [{"ref": "empty display stand", "bbox_2d": [100, 500, 400, 900]}],
        "target": [{"ref": "plush toy", "bbox_2d": [100, 400, 400, 900]}],
    }
    updated = apply_task_post_policy(
        "object_addition", "Add a plush toy to the stand.", addition
    )
    assert updated["source"] == []
    assert updated["target"] == addition["target"]
    assert updated["side_override"]["dropped_side"] == "source"


def test_appearance_only_post_policy_keeps_source_coordinates_only():
    payload = {
        "ground_parse_ok": True,
        "mask_mode": "regions",
        "source": [{"ref": "orange liquid", "bbox_2d": [10, 20, 30, 40]}],
        "target": [{"ref": "green liquid", "bbox_2d": [60, 20, 80, 40]}],
        "protected_foreground": [],
    }
    updated = apply_task_post_policy(
        "color_change", "Change the orange liquid to green.", payload
    )
    assert updated["source"] == payload["source"]
    assert updated["target"] == []
    assert updated["side_override"]["rule"] == (
        "appearance_only_edit_uses_source_coordinates_v1"
    )
    assert payload["target"]


def test_plan_policy_drops_unstated_model_counts_but_keeps_bound_counts():
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "blue backpacks",
                "selection_mode": "all_matching",
                "region_mode": "multi_instance",
                "expected_count": 12,
            }
        ],
    }
    updated = apply_observation_plan_policy(
        "color_change", "Change the blue backpacks to green.", plan
    )
    assert updated["localization_items"][0]["expected_count"] is None
    assert plan["localization_items"][0]["expected_count"] == 12

    explicit = apply_observation_plan_policy(
        "color_change", "Change the two blue backpacks to green.", plan
    )
    assert explicit["localization_items"][0]["expected_count"] == 2


def test_protected_foreground_keeps_explicit_multi_instance_topology():
    plan = parse_observation(
        json.dumps(
            {
                "realized_edit": "the background changed behind the students",
                "mask_mode": "protect_foreground",
                "localization_items": [
                    {
                        "image_side": "source",
                        "role": "protected_foreground",
                        "edit_op": "protect",
                        "ref": "all visible students",
                        "spatial_hint": "throughout the classroom",
                        "geometry": "semantic_object",
                        "mask_method": "sam",
                        "region_mode": "multi_instance",
                        "selection_mode": "all_matching",
                        "mask_extent": "whole_actor",
                        "expected_count": 7,
                        "mask_density": "object",
                        "negative_space": False,
                        "carrier_ref": "",
                    }
                ],
                "confidence": "high",
            }
        )
    )
    item = plan["localization_items"][0]
    assert item["region_mode"] == "multi_instance"
    assert item["selection_mode"] == "all_matching"
    assert item["mask_extent"] == "whole_actor"
    assert item["expected_count"] == 7


def test_plan_policy_uses_complete_changed_surface_in_locator():
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "gray middle facade section",
                "spatial_hint": "middle of building",
                "geometry": "semantic_object",
                "mask_method": "sam",
                "region_mode": "object",
                "selection_mode": "single",
                "mask_extent": "surface_region",
                "expected_count": 1,
                "mask_density": "object",
            }
        ],
    }
    updated = apply_observation_plan_policy(
        "color_change",
        "Change the right building's facade from two-tone gray and white to uniform white.",
        plan,
    )
    item = updated["localization_items"][0]
    assert item["ref"] == "gray middle facade section"
    assert item["region_mode"] == "aggregate_region"
    assert item["selection_mode"] == "compact_region"
    assert item["expected_count"] is None
    residual = updated["localization_items"][1]
    assert residual["optional"] is True
    assert "additional changed surface adjoining" in residual["ref"]
    locator = build_grounding_prompt("color_change", "unused", updated)
    assert "Image 1 is the source and Image 2 is the edited result" in locator
    assert "complete changed surface" in locator
    assert "one box enclosing complete changed surface" in locator
    assert "omit if absent" not in locator
    assert "additional changed surface adjoining" not in locator
    assert '"candidate_id":1' not in locator


def test_optional_residual_bbox_can_be_present_or_absent():
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "ref": "primary facade section",
                "optional": False,
            },
            {
                "candidate_id": 1,
                "image_side": "source",
                "ref": "additional changed facade section",
                "optional": True,
            },
        ],
    }
    primary_only = parse_bbox_localization(
        '[{"bbox_2d":[100,100,400,900],"label":"candidate_id=0"}]',
        [0, 1],
        optional_candidate_ids=[1],
    )
    assert [item["candidate_id"] for item in primary_only] == [0]
    payload = grounding_from_localization(plan, primary_only)
    assert len(payload["source"]) == 1

    both = parse_bbox_localization(
        '[{"bbox_2d":[100,100,400,900],"label":"candidate_id=0"},'
        '{"bbox_2d":[400,100,600,900],"label":"candidate_id=1"}]',
        [0, 1],
        optional_candidate_ids=[1],
    )
    assert [item["candidate_id"] for item in both] == [0, 1]
    assert len(grounding_from_localization(plan, both)["source"]) == 2


def test_aligned_surface_completion_expands_only_from_trusted_anchor():
    source = np.full((100, 100, 3), 120, dtype=np.uint8)
    target = source.copy()
    # Primary gray section and an adjoining yellow section both change. A
    # distant patch also changes but must not be joined to the trusted box.
    source[20:80, 40:60] = (80, 80, 80)
    target[20:80, 40:60] = (235, 235, 235)
    source[20:80, 20:40] = (210, 170, 45)
    target[20:80, 20:40] = (120, 120, 120)
    source[5:15, 80:95] = (20, 20, 20)
    target[5:15, 80:95] = (240, 240, 240)
    plan = {
        "mask_mode": "regions",
        "plan_policy_overrides": [
            {
                "rule": "appearance_surface_uses_all_changed_sections_v1",
                "candidate_id": 0,
            }
        ],
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "mask_extent": "surface_region",
                "optional": False,
            },
            {
                "candidate_id": 1,
                "image_side": "source",
                "mask_extent": "surface_region",
                "optional": True,
            },
        ],
    }
    localized = [{"candidate_id": 0, "bbox_2d": [400, 200, 600, 800]}]
    refined = refine_surface_localization_boxes(
        Image.fromarray(source), Image.fromarray(target), plan, localized
    )
    assert refined[0]["bbox_2d"] == [400, 200, 600, 800]
    assert len(refined) == 2
    assert refined[1]["candidate_id"] == 1
    assert refined[1]["bbox_2d"][0] < 300
    assert refined[1]["bbox_2d"][2] < 700
    assert refined[1]["bbox_refinement"]["rule"] == (
        "aligned_surface_low_frequency_completion_v1"
    )
    assert refined[1]["bbox_refinement"]["output"] == (
        "independent_residual_anchor"
    )
    assert localized[0]["bbox_2d"] == [400, 200, 600, 800]


def test_surface_completion_does_not_touch_ordinary_objects():
    source = Image.new("RGB", (100, 100), "black")
    target = Image.new("RGB", (100, 100), "white")
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": 0,
                "image_side": "source",
                "mask_extent": "whole_object",
            }
        ],
    }
    localized = [{"candidate_id": 0, "bbox_2d": [100, 100, 400, 400]}]
    assert refine_surface_localization_boxes(source, target, plan, localized) == localized


def test_global_pair_change_upgrades_grounded_background_and_foreground():
    source = Image.new("RGB", (100, 100), "black")
    target = Image.new("RGB", (100, 100), "white")
    grounded = {
        "prompt_version": "test",
        "mask_mode": "regions",
        "source": [
            {
                "candidate_id": 0,
                "ref": "background wall and floor",
                "bbox_2d": [0, 0, 1000, 650],
                "mask_extent": "surface_region",
            },
            {
                "candidate_id": 1,
                "ref": "metal wire",
                "bbox_2d": [410, 120, 590, 820],
                "mask_extent": "whole_object",
            },
        ],
        "target": [],
        "protected_foreground": [],
        "semantic_qc_flags": [],
    }
    updated = upgrade_global_change_mask_mode(source, target, grounded)
    assert updated["mask_mode"] == "full_image"
    assert updated["source"] == []
    assert updated["global_route_override"]["original_source"] == grounded["source"]
    assert updated["global_route_override"]["fraction_above_44"] == 1.0
    assert grounded["mask_mode"] == "regions"


def test_global_pair_change_requires_near_complete_grounded_envelope():
    source = Image.new("RGB", (100, 100), "black")
    target = Image.new("RGB", (100, 100), "white")
    grounded = {
        "mask_mode": "regions",
        "source": [
            {
                "ref": "background wall and floor",
                "bbox_2d": [0, 0, 1000, 650],
                "mask_extent": "surface_region",
            },
            {
                "ref": "small object",
                "bbox_2d": [400, 200, 600, 700],
                "mask_extent": "whole_object",
            },
        ],
        "target": [],
        "protected_foreground": [],
    }
    assert upgrade_global_change_mask_mode(source, target, grounded) is grounded


def test_plan_policy_collapses_multiple_body_parts_to_whole_actor():
    base = {
        "image_side": "source",
        "role": "edit_region",
        "edit_op": "remove",
        "spatial_hint": "at the desk",
        "geometry": "semantic_object",
        "mask_method": "sam",
        "region_mode": "object",
        "selection_mode": "single",
        "mask_extent": "subpart",
        "expected_count": 1,
        "mask_density": "object",
        "negative_space": False,
        "carrier_ref": "",
    }
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {**base, "candidate_id": 0, "ref": "boy's right arm and hand"},
            {**base, "candidate_id": 1, "ref": "boy's face"},
            {
                **base,
                "candidate_id": 2,
                "image_side": "target",
                "edit_op": "add",
                "ref": "boy's raised arm and smiling face",
            },
        ],
    }
    updated = apply_observation_plan_policy(
        "action_editing", "Make the boy raise his hand while sitting.", plan
    )
    items = updated["localization_items"]
    assert len(items) == 2
    assert [item["candidate_id"] for item in items] == [0, 1]
    assert {item["image_side"] for item in items} == {"source", "target"}
    assert all(item["ref"] == "complete visible boy" for item in items)
    assert all(item["mask_extent"] == "whole_actor" for item in items)


def test_plan_policy_keeps_a_truly_isolated_limb_as_subpart():
    plan = {
        "mask_mode": "regions",
        "localization_items": [
            {
                "candidate_id": side_id,
                "image_side": side,
                "ref": "man's left hand",
                "selection_mode": "single",
                "region_mode": "object",
                "mask_extent": "subpart",
                "expected_count": 1,
            }
            for side_id, side in enumerate(("source", "target"))
        ],
    }
    updated = apply_observation_plan_policy(
        "action_editing", "Turn the man's left hand palm-up.", plan
    )
    assert [item["mask_extent"] for item in updated["localization_items"]] == [
        "subpart",
        "subpart",
    ]


def test_object_viewpoint_detection_keeps_camera_edits_global():
    assert object_viewpoint_ref(
        "viewpoint_transformation",
        "Draw the rear view of the fire truck, including its rear lights.",
    ) == "fire truck"
    assert object_viewpoint_ref(
        "viewpoint_transformation",
        "Zoom in on the cluster of colorful balloons.",
    ) == "cluster of colorful balloons"
    assert not object_viewpoint_ref(
        "viewpoint_transformation",
        "Draw a view as if the camera moves back away from the car.",
    )


def test_maze_path_post_policy_covers_distant_endpoints():
    payload = {
        "ground_parse_ok": True,
        "mask_mode": "regions",
        "source": [],
        "target": [
            {
                "ref": "blue line path through maze",
                "bbox_2d": [320.0, 380.0, 680.0, 620.0],
                "mask_method": "box",
                "region_mode": "aggregate_region",
                "mask_density": "sparse",
            }
        ],
        "protected_foreground": [],
    }
    updated = apply_task_post_policy(
        "symbolic_reasoning",
        "Draw a blue line along the correct path in the maze.",
        payload,
    )
    assert updated["target"][0]["bbox_2d"] == [50.0, 100.0, 950.0, 900.0]
    assert updated["box_override"]["rule"] == "maze_path_must_cover_both_endpoints_v1"
    assert payload["target"][0]["bbox_2d"] == [320.0, 380.0, 680.0, 620.0]
