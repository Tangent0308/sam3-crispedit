import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from crispedit.prefilter.pair_quality import (
    QUALITY_DIMENSIONS,
    build_quality_conversation,
    build_quality_prompt,
    extract_json_object,
    failed_dimensions,
    normalize_quality_assessment,
    unresolved_dimensions,
)
from crispedit.prefilter.pair_runner import build_jobs, parse_cases


def _assessment(statuses=None, reason_codes=None):
    statuses = statuses or {}
    value = {
        name: {
            "status": statuses.get(name, "PASS"),
            "evidence": f"visible evidence for {name}",
        }
        for name in QUALITY_DIMENSIONS
    }
    value.update(
        {
            "edit_observation": {
                "source_state": "the requested object has its original state",
                "target_state": "the requested object has the requested state",
                "visible_change": "CLEAR",
                "instruction_match": "PASS",
            },
            "reason_codes": reason_codes or [],
            "summary": "The paired images provide enough evidence.",
            "confidence": 0.95,
        }
    )
    return value


def test_type_guidance_allows_expected_large_edits():
    replace = build_quality_prompt("replace", "replace the rabbit with a bear")
    background = build_quality_prompt(
        "background change", "change the background to a meadow"
    )
    style = build_quality_prompt("style", "render this as a cartoon")
    assert "Changing the main subject's category" in replace
    assert "Do not call those expected changes composition failure" in background
    assert "entire rendering medium" in style


def test_conversation_has_explicit_source_and_target_roles():
    conversation = build_quality_conversation(
        Image.new("RGB", (4, 4)),
        Image.new("RGB", (4, 4)),
        "add",
        "add a cup",
    )
    content = conversation[0]["content"]
    assert [item["type"] for item in content].count("image") == 2
    assert "SOURCE before editing" in content[0]["text"]
    assert "TARGET edited result" in content[2]["text"]


def test_normalization_derives_code_owned_verdicts():
    passed = normalize_quality_assessment(_assessment())
    assert passed["verdict"] == "PASS"
    assert passed["keep"] is True

    failed = normalize_quality_assessment(
        _assessment(
            {"edit_completion": "FAIL"}, reason_codes=["EDIT_NOT_COMPLETED"]
        )
    )
    assert failed["verdict"] == "FAIL"
    assert failed_dimensions(failed) == ("edit_completion",)

    unsure = normalize_quality_assessment(
        _assessment({"target_integrity": "UNSURE"})
    )
    assert unsure["verdict"] == "UNSURE"
    assert unresolved_dimensions(unsure) == ("target_integrity",)


def test_parser_accepts_fence_and_discards_contradictory_reason_code():
    payload = _assessment(
        {"edit_completion": "FAIL"},
        reason_codes=["NO_OP_ALREADY_SATISFIED", "EDIT_NOT_COMPLETED"],
    )
    parsed = extract_json_object("```json\n" + json.dumps(payload) + "\n```\ndone")
    result = normalize_quality_assessment(parsed)
    assert result["reason_codes"] == ["EDIT_NOT_COMPLETED"]


def test_normalization_rejects_missing_dimension():
    payload = _assessment()
    del payload["target_integrity"]
    with pytest.raises(ValueError, match="missing quality dimension"):
        normalize_quality_assessment(payload)


def test_pair_observation_prevents_contradictory_pass():
    no_change = _assessment()
    no_change["edit_observation"]["visible_change"] = "NONE"
    result = normalize_quality_assessment(no_change)
    assert result["verdict"] == "FAIL"
    assert failed_dimensions(result) == ("edit_completion",)

    mismatch = _assessment()
    mismatch["edit_observation"]["instruction_match"] = "UNSURE"
    result = normalize_quality_assessment(mismatch)
    assert result["verdict"] == "UNSURE"
    assert unresolved_dimensions(result) == ("edit_completion",)


def test_normalization_requires_pair_observation():
    payload = _assessment()
    del payload["edit_observation"]
    with pytest.raises(ValueError, match="missing edit_observation"):
        normalize_quality_assessment(payload)


def test_targeted_jobs_preserve_original_row_indices(tmp_path):
    source = tmp_path / "add_00000.parquet"
    pq.write_table(
        pa.table(
            {
                "input_img": [b"a", b"b", b"c"],
                "output_img": [b"d", b"e", b"f"],
                "instruction": ["one", "two", "three"],
                "type": ["add", "add", "add"],
            }
        ),
        source,
    )
    args = argparse.Namespace(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        case=["add_00000.parquet:2", "add_00000.parquet:0"],
        include_types="",
        limit_shards=None,
        limit_rows_per_shard=None,
    )
    jobs = build_jobs(args)
    assert len(jobs) == 1
    assert jobs[0].num_rows == 2
    assert jobs[0].row_indices == (0, 2)
    assert parse_cases(args.case) == {"add_00000.parquet": (0, 2)}
