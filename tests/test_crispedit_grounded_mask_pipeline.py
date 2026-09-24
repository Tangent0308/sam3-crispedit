import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from crispedit.mask.pipeline import (
    _aggregate_semantic_connected_coverage,
    _color_surface_sam_prompt,
    _fuse_pcs_prompts,
    _sam_text_prompt,
    annotate_grounded_sample,
    aspect_ratio_delta,
    box_iou,
    expand_box,
    map_target_mask_to_source,
    segment_grounded_box,
)
from crispedit.mask.grounding import (TWO_PASS_PROMPT_VERSION, build_change_observation_prompt, build_grounding_prompt, build_grounding_requests, canonicalize_type, grounding_is_complete, parse_change_observation, prompt_version_for_mode)
from crispedit.mask.grounding_runner import (
    GROUND_SCHEMA,
    conversation_image_count,
    conversations_for_vllm,
    prefilter_fields,
    split_conversations_by_image_budget,
)
from crispedit.mask.runner import MASK_SCHEMA, _copy_metadata, build_jobs








def test_visual_load_batching_preserves_order_and_single_large_request():
    def conversation(image_count):
        return [
            {
                "role": "user",
                "content": [{"type": "image", "image": object()}] * image_count,
            }
        ]

    requests = [conversation(2), conversation(2), conversation(10), conversation(2)]
    assert [conversation_image_count(item) for item in requests] == [2, 2, 10, 2]
    chunks = split_conversations_by_image_budget(requests, max_images=10)
    assert [[conversation_image_count(item) for item in chunk] for chunk in chunks] == [
        [2, 2],
        [10],
        [2],
    ]
    oversized = split_conversations_by_image_budget([conversation(12)], max_images=10)
    assert len(oversized) == 1
    assert [conversation_image_count(item) for item in oversized[0]] == [12]


def test_vllm_conversation_conversion_preserves_turns_and_image_order():
    source = Image.new("RGB", (3, 2), "red")
    target = Image.new("RGB", (3, 2), "blue")
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "source"},
                    {"type": "image", "image": source},
                    {"type": "text", "text": "target"},
                    {"type": "image", "image": target},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "observed"}]},
            {"role": "user", "content": [{"type": "text", "text": "ground it"}]},
        ]
    ]

    converted = conversations_for_vllm(conversations)
    assert [part["type"] for part in converted[0][0]["content"]] == [
        "text",
        "image_pil",
        "text",
        "image_pil",
    ]
    assert converted[0][0]["content"][1]["image_pil"] is source
    assert converted[0][0]["content"][3]["image_pil"] is target
    assert converted[0][1:] == conversations[0][1:]
    assert conversations[0][0]["content"][1]["type"] == "image"


def test_mask_jobs_are_scoped_by_input_directory(tmp_path):
    input_dir = tmp_path / "raw"
    grounding_dir = tmp_path / "grounding"
    output_dir = tmp_path / "mask"
    input_dir.mkdir()
    grounding_dir.mkdir()
    pq.write_table(pa.table({"value": [1]}), input_dir / "add_00000.parquet")
    pq.write_table(pa.table({"raw_type": ["add"], "row_idx": [0]}), grounding_dir / "add_00000.parquet")
    pq.write_table(pa.table({"raw_type": ["add"]}), grounding_dir / "add_00001.parquet")
    args = argparse.Namespace(
        input_dir=input_dir,
        grounding_dir=grounding_dir,
        output_dir=output_dir,
        include_types=None,
    )

    jobs = build_jobs(args)

    assert [Path(job.input_path).name for job in jobs] == ["add_00000.parquet"]
    assert [Path(job.grounding_path).name for job in jobs] == ["add_00000.parquet"]


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


def test_collateral_changes_do_not_add_opposite_canvas():
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
    ] == ["target"]
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
    ] == ["source"]

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
    ] == ["target"]


def test_two_pass_observation_prompt_and_grounding_checklist():
    prompt = build_change_observation_prompt("color", "make the arms darker")
    assert "co-edited instances" in prompt
    assert "only the changed parts" in prompt
    assert "make the arms darker" in prompt
    assert len(prompt.split()) < 400
    assert "nearby cluster" in prompt
    assert "checked_regions" not in prompt
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
    assert "EVERY change_id exactly once" in grounding_prompt
    assert "whole person for a local part" in grounding_prompt
    grouped_prompt = build_grounding_prompt(
        "add", "add scattered petals", "target", observation
    )
    assert "Do not merge different IDs" in grouped_prompt
    assert "tight boxes enclosing the complete visible contour" in grouped_prompt
    assert prompt_version_for_mode("two-pass") == TWO_PASS_PROMPT_VERSION








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




def test_checklist_identity_takes_precedence_over_instruction_surface():
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
    assert '"ref":"alien head and hands"' in whole_object_prompt

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
    assert "Independently ground a same-subject surface" not in skin_prompt
    assert '"ref":"bare arms"' in skin_prompt
















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


def test_aggregate_dual_prompt_selects_instead_of_blind_union():
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
    assert mask.sum() == 9
    assert audit["pcs_fusion"] == "aggregate_choose_joint"


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
