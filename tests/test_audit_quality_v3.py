import json

from synthesis_pipeline.audit_quality_v3 import (
    corrected_annotation,
    parse_compact,
    parse_rewrite,
    apply_target_change_gate,
    apply_add_scope_gate,
    rewrite_policy_error,
    apply_flat_fill_gate,
)


def test_natural_mismatch_is_eligible_for_relabeling_not_original_acceptance():
    result = parse_compact(
        json.dumps(
            {
                "visual_quality": "pass",
                "instruction_match": "fail",
                "reason": "The horse was replaced with a bear, not a dog.",
            }
        )
    )
    assert result["quality"] == "fail"
    assert result["visual_quality"] == "pass"


def test_invalid_or_uncertain_judgment_is_not_a_pass():
    assert (
        parse_compact(
            '{"quality":"review","observed_change":"x","reason":"unclear"}', True
        )
        is None
    )
    assert parse_compact("not json") is None


def test_reconstruction_abstention_and_annotation_leakage():
    assert parse_rewrite('{"task_type":null,"instruction":null}') is None
    assert (
        parse_rewrite(
            '{"task_type":"remove","instruction":"Remove the masked person."}'
        )
        is None
    )
    assert (
        parse_rewrite(
            '{"task_type":"replace","instruction":"Replace the left horse with a bear."}'
        )["task_type"]
        == "replace"
    )
    assert (
        parse_rewrite(
            '{"task_type":"remove","instruction":"Remove the highlighted horse."}'
        )
        is None
    )
    assert (
        parse_rewrite(
            '{"task_type":"remove","instruction":"Remove the surgical mask from the woman on the left."}'
        )
        is not None
    )


def test_thinking_parser_uses_final_answer_not_intermediate_json():
    raw = '<think>{"visual_quality":"pass","instruction_match":"pass","reason":"tentative"}</think>{"visual_quality":"fail","instruction_match":"fail","reason":"Visible old head remains."}'
    assert parse_compact(raw)["quality"] == "fail"
    assert (
        parse_rewrite(
            '<think>{"task_type":"add","instruction":"Add a dog."}</think>{"task_type":null,"instruction":null}'
        )
        is None
    )
    assert (
        parse_rewrite('<think>{"task_type":"add","instruction":"Add a dog."}') is None
    )
    assert (
        parse_compact(
            '<think>{"visual_quality":"pass","instruction_match":"pass","reason":"tentative"}'
        )
        is None
    )


def test_rewrite_cannot_rescue_failed_quality_or_mutate_original():
    row = {
        "image": "x.png",
        "task_type": "replace",
        "editing_instruction": "Replace the horse with a dog.",
        "mask": [1],
    }
    rewrite = {
        "status": "candidate",
        "task_type": "replace",
        "editing_instruction": "Replace the left horse with a bear.",
    }
    assert (
        corrected_annotation(row, rewrite, {"audit": {"visual_quality": "fail"}})
        is None
    )
    result = corrected_annotation(
        row,
        rewrite,
        {"audit": {"visual_quality": "pass", "reason": "Natural replacement."}},
    )
    assert result["editing_instruction"] == rewrite["editing_instruction"]
    assert result["mask"] == row["mask"]
    assert row["editing_instruction"].endswith("dog.")
    assert (
        result["instruction_revision"]["original_instruction"]
        == row["editing_instruction"]
    )


def test_target_gate_rejects_outside_only_changes_without_blocking_add_anchors():
    audit = {
        "visual_quality": "pass",
        "instruction_match": "pass",
        "quality": "pass",
        "reason": "Looks good.",
    }
    metrics = {"inside_changed_fraction": 0.006}
    assert (
        apply_target_change_gate(audit, "attribute", metrics, 0.02)["visual_quality"]
        == "fail"
    )
    assert apply_target_change_gate(audit, "add", metrics, 0.02)["quality"] == "pass"
    assert audit["quality"] == "pass"


def test_add_size_policy_is_explicit_and_does_not_mutate_model_decision():
    audit = {'visual_quality': 'pass', 'quality': 'pass', 'reason': 'A new sign appears.'}
    metrics = {'changed_to_target_area_ratio': 20}
    assert apply_add_scope_gate(audit, 'add', metrics, 3)['quality'] == 'fail'
    assert apply_add_scope_gate(audit, 'add', metrics, 0)['quality'] == 'pass'
    assert apply_add_scope_gate(audit, 'replace', metrics, 3)['quality'] == 'pass'
    assert audit['quality'] == 'pass'


def test_rewrite_cannot_automatically_change_operation_or_add_interaction():
    row = {'task_type': 'remove'}
    val = {'task_type': 'replace', 'editing_instruction': 'Replace the right woman with a man in pink.'}
    assert rewrite_policy_error(row, val) == 'task_type_change_requires_manual_verification'
    assert rewrite_policy_error(row, val, True) is None
    row = {'task_type': 'replace'}
    assert rewrite_policy_error(row, {'task_type': 'replace', 'editing_instruction':
        'Replace the backpack wearer with a woman walking a dog.'}) == 'replacement_introduces_multiple_interacting_entities'
    assert rewrite_policy_error(row, {'task_type': 'replace', 'editing_instruction':
        'Replace the left doll with a plush bear.'}) is None


def test_flat_fill_gate_is_only_for_new_texture_collapse():
    audit = {'visual_quality': 'pass', 'quality': 'pass', 'reason': 'Red target.'}
    metrics = {'source_dominant_color_fraction': .1, 'edited_dominant_color_fraction': .95}
    assert apply_flat_fill_gate(audit, 'attribute', metrics, True)['quality'] == 'fail'
    assert apply_flat_fill_gate(audit, 'replace', metrics, True)['quality'] == 'pass'
    metrics['source_dominant_color_fraction'] = .9
    assert apply_flat_fill_gate(audit, 'attribute', metrics, True)['quality'] == 'pass'


def test_outline_rewrite_and_count_drift_cannot_be_training_labels():
    assert parse_rewrite('{"task_type":"remove","instruction":"Remove the white outline around the hot dog."}') is None
    row = {'task_type': 'replace', 'edit_unit_status': 'complete_object'}
    val = {'task_type': 'replace', 'editing_instruction': 'Replace the two buses with a police car.'}
    assert rewrite_policy_error(row, val) == 'rewrite_changes_single_target_count'
def test_rewrite_cannot_add_revealed_background_person_as_second_replacement():
    from synthesis_pipeline.audit_quality_v3 import rewrite_policy_error
    row=dict(task_type='replace',edit_unit_status='complete_object')
    val=dict(task_type='replace',editing_instruction='Replace the woman with a parked bicycle and a different pedestrian.')
    assert rewrite_policy_error(row,val)=='replacement_introduces_multiple_entities'
