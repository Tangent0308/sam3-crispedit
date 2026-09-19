from synthesis_pipeline.audit_edit_pairs_v2 import (
    apply_low_change_veto,
    build_prompt,
    parse_audit_json,
)


def test_attribute_threshold_is_optional_and_type_specific():
    result = {
        "quality": "pass", "instruction_match": "pass", "reason": "Looks blue.",
        "observed_instruction": "Change the shirt to blue.",
        "rewrite_candidate": None, "salvage_status": "none",
    }
    metrics = {"inside_changed_fraction": 0.16}
    assert apply_low_change_veto(result.copy(), "attribute", metrics, 0.30, 0.0)["quality"] == "pass"
    rejected = apply_low_change_veto(result.copy(), "attribute", metrics, 0.30, 0.20)
    assert rejected["quality"] == "fail"
    assert rejected["metric_veto"]["code"] == "low_mask_change_for_attribute"
    assert apply_low_change_veto(result.copy(), "add", metrics, 0.30, 0.20)["quality"] == "pass"


def test_conservative_prompt_requires_before_after_evidence():
    prompt = build_prompt(
        {
            "task_type": "attribute",
            "editing_instruction": "Change the selected railing ornament to gold.",
        },
        prompt_variant="conservative_gate",
    )
    assert "BEFORE and AFTER" in prompt
    assert "physical support" in prompt
    assert "Change the selected railing ornament to gold." in prompt
    assert "<<<" not in prompt


def test_recover_json_fields_only_response():
    raw = (
        '"source_target": "flag"\n'
        '"edited_target_area": "sky"\n'
        '"observed_edit": "flag removed"\n'
        '"target_match": true\n'
        '"unexpected_change": null\n'
        '"artifact": null\n'
        '"visual_quality": "pass"\n'
        '"instruction_match": "pass"\n'
        '"observed_instruction": null\n'
        '"reason": "The flag is gone."'
    )
    assert parse_audit_json(raw)["source_target"] == "flag"
    assert parse_audit_json("The flag is gone.") is None
