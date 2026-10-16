import pytest
import json
from pathlib import Path
from PIL import Image

from sa1b_pipeline.training import build_training_rows
from sa1b_pipeline.mllm import _safe_parse
from sa1b_pipeline.import_selected_sources import import_sources
from sa1b_pipeline.instructions import build_unit
from sa1b_pipeline.editor import edit_prompt


def _pair(**updates):
    pair = {
        "case_id": "case",
        "unit_id": "U1",
        "operation": "attribute",
        "instruction": "Change the color of the left cat to navy.",
        "instruction_noref": "Change the color of the selected cat to navy.",
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


def test_training_contract_emits_exact_four_rows_and_selected_noref():
    rows = build_training_rows(_pair(), "<|mt_start|><|mt_0001|><|mt_0257|><|mt_end|>")
    assert [row["sample_type"] for row in rows] == ["edit_ntp", "edit_umt", "edit_umt", "edit"]
    assert rows[2]["prompt"] == "Change the color of the selected cat <|mt_start|><|mt_0001|><|mt_0257|><|mt_end|> to navy."
    assert set(rows[0]) == {"sample_type", "edit_type", "edit_image", "prompt", "mt_cot"}


def test_mllm_batch_parse_failure_is_a_rejection():
    result = _safe_parse("not json", stage="selection")
    assert result["accept"] is False
    assert result["_parse_error"]


def test_mllm_audit_parse_failure_is_a_fail():
    result = _safe_parse("not json", stage="audit")
    assert result["decision"] == "fail"


def test_import_selected_sources_preserves_masks_and_provenance(tmp_path: Path):
    root = tmp_path / "assets"
    (root / "selected" / "sa_1").mkdir(parents=True)
    Image.new("RGB", (8, 6), "white").save(root / "selected" / "sa_1" / "image.jpg")
    Image.new("L", (8, 6), 255).save(root / "selected" / "sa_1" / "object_3.png")
    source_manifest = tmp_path / "selected_sources.jsonl"
    source_manifest.write_text(json.dumps({
        "id": "sa_1", "selection_index": 0,
        "image": "selected/sa_1/image.jpg", "image_sha256": "source-hash",
        "targets": [{"unit_id": "U1", "object_id": 3, "source_category": "sleeve",
                      "source_description": "visible sleeve", "mask": "selected/sa_1/object_3.png",
                      "bbox_xyxy": [0, 0, 8, 6], "area_fraction": 1.0,
                      "physical_parent_zh": "person"}],
    }) + "\n")
    output = tmp_path / "candidates.jsonl"
    rows = import_sources(source_manifest, root, output)
    assert rows[0]["source_id"] == "sa_1"
    assert rows[0]["objects"][0]["mask_path"].endswith("object_3.png")
    assert rows[0]["provenance"]["selection_status"].startswith("benchmark_shortlist")


def test_add_instruction_requires_a_new_item_payload(tmp_path: Path):
    obj = {"object_id": "1", "category": "chair", "description": "chair"}
    geometry = {"bbox": [0, 0, 4, 4], "point": [1, 1]}
    with pytest.raises(ValueError, match="new physical item"):
        build_unit({"operation": "add", "ref_phrase": "the chair", "scope": "chair",
                    "payload": "the top of the chair"}, obj, str(tmp_path / "m.png"), geometry, [])
    unit = build_unit({"operation": "add", "ref_phrase": "the chair", "scope": "chair",
                       "payload": "Add a small brass badge to the chair"}, obj,
                      str(tmp_path / "m.png"), geometry, [])
    assert unit.payload == "a small brass badge"
    assert unit.instruction_noref == "Add a small brass badge to the selected region."


def test_remove_editor_prompt_requires_background_reconstruction():
    prompt = edit_prompt({"operation": "remove", "instruction_ref": "Remove the selected chair."})
    assert "Completely erase" in prompt
    assert "For independent objects, reconstruct the exposed background" in prompt
    assert "Reveal the underlying host" in prompt
    assert "glove" not in prompt and "bare hand" not in prompt
    assert "cyan/gray patch" in prompt


def test_remove_glove_retains_host_guidance_without_priming_unrelated_objects():
    glove = edit_prompt({"operation": "remove", "scope": "gloves", "instruction_ref": "Remove the gloves."})
    vase = edit_prompt({"operation": "remove", "scope": "vase", "instruction_ref": "Remove the vase."})
    assert "bare hand" in glove and "Never erase the underlying limb" in glove
    assert "bare hand" not in vase and "glove" not in vase
    assert "underlying host" in vase
