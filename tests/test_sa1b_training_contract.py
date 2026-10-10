import pytest

from sa1b_pipeline.training import build_training_rows
from sa1b_pipeline.mllm import _safe_parse


def _pair(**updates):
    pair = {
        "case_id": "case",
        "unit_id": "U1",
        "operation": "attribute",
        "instruction": "Change the color of the left cat to navy.",
        "instruction_noref": "Change the color of this region to navy.",
        "ref_phrase": "the left cat",
        "source_image": "source.jpg",
        "edited_image": "edited.jpg",
    }
    pair.update(updates)
    return pair


def test_training_contract_rejects_invalid_codebook_ranges():
    with pytest.raises(ValueError, match="0000-0255"):
        build_training_rows(_pair(), "<|mt_start|><|mt_0256|><|mt_0257|><|mt_end|>")


def test_training_contract_rejects_ambiguous_ref_phrase():
    with pytest.raises(ValueError, match="cannot bind"):
        build_training_rows(
            _pair(
                instruction="Change the color of the left cat and the left cat to navy.",
                ref_phrase="the left cat",
            ),
            "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>",
        )


def test_training_contract_emits_exact_four_rows_and_region_noref():
    rows = build_training_rows(_pair(), "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>")
    assert [row["sample_type"] for row in rows] == ["edit_ntp", "edit_umt", "edit_umt", "edit"]
    assert rows[2]["prompt"] == "Change the color of this region <|mt_start|><|mt_0001|><|mt_0257|><|mt_end|> to navy."
    assert set(rows[0]) == {"sample_type", "edit_type", "edit_image", "prompt", "mt_cot"}


def test_mllm_batch_parse_failure_is_a_rejection():
    result = _safe_parse("not json", stage="selection")
    assert result["accept"] is False
    assert result["_parse_error"]


def test_mllm_audit_parse_failure_is_a_fail():
    result = _safe_parse("not json", stage="audit")
    assert result["decision"] == "fail"
