import json

import numpy as np
from PIL import Image

from scaleedit.mask_pipeline import annotate_sample
from scaleedit.policy import (
    SUPPORTED_TASKS,
    apply_task_post_policy,
    build_grounding_prompt,
    build_observation_prompt,
    canonical_task,
    grounding_status,
    object_viewpoint_ref,
    parse_grounding,
    parse_observation,
)


def test_all_scaleedit_tasks_are_explicitly_supported():
    assert len(SUPPORTED_TASKS) == 23
    for task in SUPPORTED_TASKS:
        assert canonical_task(task.replace("_", " ")) == task


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
        {"mask_mode": "regions", "changes": []},
    )
    assert "local style edit" in grounding
    assert "mask_method=box" in grounding


def test_observation_parser_accepts_wrapped_json():
    parsed = parse_observation(
        "answer:\n```json\n"
        + json.dumps(
            {
                "realized_edit": "the title changed",
                "mask_mode": "regions",
                "changes": [{"source_ref": "old title", "target_ref": "new title"}],
                "protected_foreground": [],
                "confidence": "high",
            }
        )
        + "\n```"
    )
    assert parsed["mask_mode"] == "regions"
    assert parsed["changes"][0]["target_ref"] == "new title"


def test_grounding_parser_normalizes_modes_boxes_and_methods():
    regions = parse_grounding(
        json.dumps(
            {
                "mask_mode": "regions",
                "source": [
                    {
                        "ref": "old title glyphs",
                        "bbox_2d": [-2, 100, 500.5, 260],
                        "mask_method": "box",
                        "region_mode": "aggregate_region",
                        "mask_density": "sparse",
                    }
                ],
                "target": [],
                "protected_foreground": [{"ref": "ignored", "bbox_2d": [1, 2, 3, 4]}],
            }
        )
    )
    assert regions["source"][0]["bbox_2d"] == [0.0, 100.0, 500.5, 260.0]
    assert regions["source"][0]["mask_method"] == "box"
    assert regions["protected_foreground"] == []

    protected = parse_grounding(
        json.dumps(
            {
                "mask_mode": "protect_foreground",
                "source": [{"ref": "ignored", "bbox_2d": [1, 2, 3, 4]}],
                "target": [],
                "protected_foreground": [
                    {
                        "ref": "woman on bench",
                        "bbox_2d": [100, 80, 600, 990],
                        "mask_method": "box",
                    }
                ],
            }
        )
    )
    assert protected["source"] == []
    assert protected["protected_foreground"][0]["mask_method"] == "sam"
    assert grounding_status({**protected, "ground_parse_ok": True}) == "PROTECT_FOREGROUND"


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
