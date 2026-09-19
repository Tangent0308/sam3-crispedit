import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from crispedit.difficulty.benchmark_scene import (
    build_scene_conversation,
    build_scene_prompt,
    deterministic_screen,
    normalize_scene_assessment,
    parse_scene_response,
)
from crispedit.difficulty.scene_runner import (
    AUDIT_SCHEMA,
    _error_row,
    _screened_row,
    build_jobs,
)


def _assessment(verdict="PASS", reference="the middle dog"):
    return {
        "verdict": verdict,
        "target": "dog's eyes",
        "reference": reference,
        "reason": "The middle dog is selected from several comparable dogs.",
    }


def test_prompt_is_source_only_concise_scene_judgment():
    prompt = build_scene_prompt("color", "make the middle dog's eyes blue")
    assert "only the SOURCE image" in prompt
    assert "PASS or DROP" in prompt
    assert "exact edit region" in prompt
    assert "peer_instances" not in prompt
    assert "scene_family" not in prompt
    assert len(prompt.split()) < 320
    conversation = build_scene_conversation(
        Image.new("RGB", (4, 4)), "color", "make the middle dog's eyes blue"
    )
    assert sum(item["type"] == "image" for item in conversation[0]["content"]) == 1


def test_type_screen_is_mask_independent_and_keeps_local_edit_types():
    assert deterministic_screen("add")["eligible"] is True
    assert deterministic_screen("motion change")["eligible"] is True
    assert deterministic_screen("background change")["eligible"] is False
    assert deterministic_screen("style")["eligible"] is False


def test_binary_assessment_accepts_only_pass_or_drop():
    assert normalize_scene_assessment(_assessment())["verdict"] == "PASS"
    assert normalize_scene_assessment(_assessment("DROP", "NONE"))["verdict"] == "DROP"
    with pytest.raises(ValueError, match="PASS or DROP"):
        normalize_scene_assessment(_assessment("REVIEW"))
    with pytest.raises(ValueError, match="explicit identifying reference"):
        normalize_scene_assessment(_assessment("PASS", "NONE"))


def test_assessment_requires_small_audit_fields():
    for field in ("target", "reference", "reason"):
        payload = _assessment()
        payload[field] = ""
        with pytest.raises(ValueError, match=field):
            normalize_scene_assessment(payload)


def test_parser_accepts_fenced_json_and_normalizes_case():
    payload = _assessment("pass")
    parsed = parse_scene_response("```json\n" + json.dumps(payload) + "\n```")
    assert parsed["verdict"] == "PASS"


def test_runner_rows_remain_binary_even_on_screen_or_parse_error():
    common = {
        "row_idx": 3,
        "record": {"type": "style", "instruction": "make it painterly"},
        "prefilter": {"prefilter_verdict": "PASS", "prefilter_run_id": "p"},
        "model_name": "qwen",
        "run_id": "r",
    }
    screened = _screened_row(**common, reason="ineligible_edit_type")
    failed = _error_row(**common, error=ValueError("bad json"))
    assert screened["scene_decision"] == failed["scene_decision"] == "DROP"
    assert screened["scene_pass"] is failed["scene_pass"] is False
    assert set(AUDIT_SCHEMA.names).isdisjoint({"scene_keep", "scene_loose_keep"})


def _runner_args(input_dir, manifest_dir, output_dir, cases):
    return argparse.Namespace(
        input_dir=input_dir,
        prefilter_manifest_dir=manifest_dir,
        output_dir=output_dir,
        case=cases,
        case_file=None,
        include_types="",
        limit_shards=None,
        limit_rows_per_shard=None,
    )


def test_jobs_select_only_qwen38_prefilter_pass_rows(tmp_path):
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
    manifest_dir = tmp_path / "prefilter"
    manifest_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "row_idx": [0, 1, 2],
                "prefilter_verdict": ["PASS", "FAIL", "PASS"],
            }
        ),
        manifest_dir / source.name,
    )

    jobs = build_jobs(_runner_args(tmp_path, manifest_dir, tmp_path / "out", []))
    assert jobs[0].row_indices == (0, 2)
    selected = build_jobs(
        _runner_args(tmp_path, manifest_dir, tmp_path / "out", ["add_00000.parquet:2"])
    )
    assert selected[0].row_indices == (2,)
    with pytest.raises(ValueError, match="not Qwen3.8 prefilter PASS"):
        build_jobs(
            _runner_args(tmp_path, manifest_dir, tmp_path / "out", ["add_00000.parquet:1"])
        )
