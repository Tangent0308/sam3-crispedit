import json
import argparse

import pytest
from PIL import Image

from refedit.quality_prefilter import (
    QUALITY_DIMENSIONS,
    build_quality_conversation,
    build_quality_prompt,
    extract_json_object,
    failed_dimensions,
    normalize_quality_assessment,
)
from refedit.quality_runner import build_jobs
from tests.test_refedit_pipeline import _tiny_refedit


def _assessment(statuses=None):
    statuses = statuses or {}
    value = {
        name: {
            "status": statuses.get(name, "PASS"),
            "evidence": f"evidence for {name}",
        }
        for name in QUALITY_DIMENSIONS
    }
    value.update({"reason_codes": [], "summary": "pair is usable", "confidence": 0.9})
    return value


def test_prompt_and_conversation_label_image_roles():
    prompt = build_quality_prompt("object_removal", "Remove the left cup")
    assert "Image 1 is the SOURCE" in prompt
    assert "Remove the left cup" in prompt
    conversation = build_quality_conversation(
        Image.new("RGB", (4, 4)),
        Image.new("RGB", (4, 4)),
        "object_removal",
        "Remove the left cup",
    )
    content = conversation[0]["content"]
    assert [item["type"] for item in content].count("image") == 2
    assert "SOURCE before editing" in content[0]["text"]
    assert "TARGET edited result" in content[2]["text"]


def test_json_extraction_accepts_code_fence_and_trailing_text():
    raw = "```json\n" + json.dumps(_assessment()) + "\n```\ndone"
    parsed = extract_json_object(raw)
    assert parsed["source_reference"]["status"] == "PASS"


def test_normalization_derives_pass_only_when_every_dimension_passes():
    result = normalize_quality_assessment(_assessment())
    assert result["verdict"] == "PASS"
    assert result["keep"] is True
    failed = normalize_quality_assessment(
        _assessment({"edit_completion": "FAIL"})
    )
    assert failed["verdict"] == "FAIL"
    assert failed["keep"] is False
    assert failed_dimensions(failed) == ("edit_completion",)
    unsure = normalize_quality_assessment(
        _assessment({"source_reference": "UNSURE"})
    )
    assert unsure["verdict"] == "UNSURE"
    assert unsure["keep"] is False


def test_normalization_rejects_missing_dimensions():
    value = _assessment()
    del value["target_integrity"]
    with pytest.raises(ValueError, match="missing quality dimension"):
        normalize_quality_assessment(value)


def test_unknown_reason_code_is_audited_as_other():
    value = _assessment()
    value["reason_codes"] = ["made_up_code"]
    assert normalize_quality_assessment(value)["reason_codes"] == ["OTHER"]


def test_reason_code_conflicting_with_dimension_status_is_removed():
    value = _assessment({"edit_completion": "FAIL"})
    value["reason_codes"] = ["NO_OP_ALREADY_SATISFIED", "EDIT_NOT_COMPLETED"]
    result = normalize_quality_assessment(value)
    assert result["reason_codes"] == ["EDIT_NOT_COMPLETED"]


def test_full_runner_builds_aligned_audit_and_manifest_jobs(tmp_path):
    dataset = tmp_path / "dataset"
    source = _tiny_refedit(dataset)
    output = tmp_path / "output"
    args = argparse.Namespace(
        input_dir=dataset,
        output_dir=output,
        img_id=[],
        limit_shards=None,
        limit_rows_per_shard=None,
    )
    jobs = build_jobs(args)
    assert len(jobs) == 1
    assert jobs[0].num_rows == 2
    assert jobs[0].input_path == str(source)
    assert jobs[0].audit_path == str(output / "audit" / source.name)
    assert jobs[0].manifest_path == str(output / "manifest" / source.name)
