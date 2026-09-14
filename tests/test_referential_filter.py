import json

import numpy as np

from difficulty_filter.referential import (
    build_mllm_prompt,
    deduplicate_sam_instances,
    deterministic_screen,
    extract_grounding_refs,
    fuse_evidence,
    instruction_cues,
    parse_mllm_response,
)


def test_deterministic_screen_rejects_global_routes_and_types():
    assert not deterministic_screen("color_change", "full_image").eligible
    assert not deterministic_screen("style_transfer", "regions").eligible
    assert not deterministic_screen("object_addition", "regions").eligible
    assert deterministic_screen("color_change", "regions").eligible


def test_instruction_cues_are_audit_only_but_cover_core_reference_types():
    cues = instruction_cues("Change the second backpack from the left and two on the top shelf.")
    assert {"spatial", "ordinal", "cardinality", "relation"}.issubset(cues)


def test_extract_grounding_refs_handles_both_contract_shapes_source_first():
    scale = json.dumps(
        {
            "source": [{"ref": "blue backpack on the left"}],
            "target": [{"ref": "green backpack on the left"}],
        }
    )
    crisp = json.dumps(
        {
            "boxes": {
                "source": [{"ref": "right mirror frame"}],
                "target": [{"ref": "black mirror frame"}],
            }
        }
    )
    assert extract_grounding_refs(scale) == [
        "blue backpack on the left",
        "green backpack on the left",
    ]
    assert extract_grounding_refs(crisp) == ["right mirror frame", "black mirror frame"]


def test_prompt_requires_category_without_disambiguating_attributes():
    prompt = build_mllm_prompt(
        "Change the left blue backpack to green.",
        "color_change",
        ["left blue backpack"],
    )
    assert "short singular common-noun" in prompt
    assert "counting ALL" in prompt and "peer instances" in prompt
    assert "Derive `object_category` ONLY from `edited_subject_phrase`" in prompt
    assert "Change the left blue backpack to green." in prompt


def test_parse_mllm_response_accepts_fenced_json_and_normalizes():
    parsed = parse_mllm_response(
        """```json
        {"edited_subject_phrase":"left blue backpack",
        "object_category":"  Backpack ","visible_same_class_count":4,
        "selected_instance_count":1,"subset_relation":"yes",
        "reference_cues":["spatial","spatial"],
        "fine_grained_referential":"yes","confidence":1.2,"reason":"left one"}
        ```"""
    )
    assert parsed["edited_subject_phrase"] == "left blue backpack"
    assert parsed["object_category"] == "backpack"
    assert parsed["reference_cues"] == ["spatial"]
    assert parsed["confidence"] == 1.0


def test_deduplicate_sam_instances_suppresses_contained_duplicate():
    first = np.zeros((10, 10), dtype=bool)
    first[2:7, 2:7] = True
    duplicate = np.zeros((10, 10), dtype=bool)
    duplicate[3:6, 3:6] = True
    other = np.zeros((10, 10), dtype=bool)
    other[1:4, 7:10] = True
    kept, rejected = deduplicate_sam_instances(
        [
            {"score": 0.9, "bbox_xyxy": [2, 2, 7, 7], "_mask": first},
            {"score": 0.8, "bbox_xyxy": [3, 3, 6, 6], "_mask": duplicate},
            {"score": 0.7, "bbox_xyxy": [7, 1, 10, 4], "_mask": other},
        ]
    )
    assert len(kept) == 2
    assert len(rejected) == 1
    assert rejected[0]["duplicate_of"] == 0


def _positive_mllm(judgment="yes", subset="yes", visible=3):
    return {
        "object_category": "orange",
        "visible_same_class_count": visible,
        "selected_instance_count": 1,
        "subset_relation": subset,
        "reference_cues": ["spatial"],
        "fine_grained_referential": judgment,
        "confidence": 0.9,
        "reason": "the left orange is selected",
    }


def test_fusion_keeps_agreement_and_reviews_sam_undercount():
    keep = fuse_evidence(
        deterministic_eligible=True,
        deterministic_reason="eligible_local_edit",
        mllm=_positive_mllm(),
        sam_count=3,
        sam_selected_count=1,
    )
    review = fuse_evidence(
        deterministic_eligible=True,
        deterministic_reason="eligible_local_edit",
        mllm=_positive_mllm(),
        sam_count=1,
        sam_selected_count=1,
    )
    assert keep == {
        "decision": "keep",
        "loose_keep": True,
        "reason": "mllm_subset_and_sam_mask_subset",
    }
    assert review["decision"] == "review"
    assert review["loose_keep"]


def test_fusion_reviews_mllm_rejection_when_sam_mask_is_a_proper_subset():
    result = fuse_evidence(
        deterministic_eligible=True,
        deterministic_reason="eligible_local_edit",
        mllm=_positive_mllm(judgment="no", subset="no"),
        sam_count=5,
        sam_selected_count=1,
    )
    assert result["decision"] == "review"
    assert result["loose_keep"]


def test_fusion_still_drops_mllm_rejection_when_mask_covers_all_peers():
    result = fuse_evidence(
        deterministic_eligible=True,
        deterministic_reason="eligible_local_edit",
        mllm=_positive_mllm(judgment="no", subset="no"),
        sam_count=5,
        sam_selected_count=5,
    )
    assert result["decision"] == "drop"
    assert not result["loose_keep"]


def test_fusion_drops_when_training_mask_covers_every_sam_peer():
    result = fuse_evidence(
        deterministic_eligible=True,
        deterministic_reason="eligible_local_edit",
        mllm=_positive_mllm(),
        sam_count=4,
        sam_selected_count=4,
    )
    assert result["decision"] == "drop"
    assert result["reason"] == "edit_mask_covers_nearly_all_sam_instances"


def test_fusion_drops_nearly_all_instances_to_tolerate_one_sam_false_positive():
    result = fuse_evidence(
        deterministic_eligible=True,
        deterministic_reason="eligible_local_edit",
        mllm=_positive_mllm(visible=20),
        sam_count=20,
        sam_selected_count=19,
    )
    assert result["decision"] == "drop"
    assert result["reason"] == "edit_mask_covers_nearly_all_sam_instances"
