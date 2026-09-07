import argparse
import json

import cv2
import numpy as np
from PIL import Image

from scaleedit.mask_pipeline import (
    _clean_semantic_object_mask,
    _negative_space_from_carrier_mask,
    _target_map_dilate_frac,
    annotate_sample,
)
from scaleedit.policy import (
    SUPPORTED_TASKS,
    apply_task_post_policy,
    build_grounding_prompt,
    build_observation_prompt,
    canonical_task,
    grounding_from_localization,
    object_viewpoint_ref,
    parse_bbox_localization,
    parse_observation,
)
from scaleedit.grounding_runner import Qwen35ScaleEditGrounder
from scripts.visualize_scaleedit_masks import COARSE_CATEGORY_GROUPS


def test_all_scaleedit_tasks_are_explicitly_supported():
    assert len(SUPPORTED_TASKS) == 23
    for task in SUPPORTED_TASKS:
        assert canonical_task(task.replace("_", " ")) == task


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
    assert "Do not reinterpret the edit" in grounding
    assert 'complete visible material/region pixels of "painted wall"' in grounding
    assert "location: wall behind the sofa" in grounding
    assert '"bbox_2d": [x1, y1, x2, y2]' in grounding
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
    assert 'complete visible object "left added orange"' in count
    assert "location: left of the unchanged orange" in count
    assert "support surfaces" in count
    assert "Visual grounding only" in count


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
    assert "complete visible empty/white negative-space region" in locator
    assert 'foreground carrier "chocolate chip cookie"' in locator
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
    assert parse_bbox_localization(duplicate, [0, 1]) == parsed


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

    def generate(_conversations):
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

    def generate(conversations):
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
    assert "Do not reinterpret the edit" in second_text
    assert "realized_edit" not in second_text
    assert "Corrected edit instruction" not in second_text
    assert "mask_method" not in second_text
    assert "mask_density" not in second_text
    assert payload["target"][0]["ref"] == "left added orange"


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
