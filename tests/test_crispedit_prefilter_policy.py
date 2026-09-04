from crispedit.prefilter.runner import (
    _make_audit_and_manifest_rows,
    build_questionnaire,
    build_review_state_prompt,
    build_single_image_prompt,
    crop_normalized_bbox,
    needs_source_guided_target_crop,
    parse_simple_instruction_slots,
    select_focus_bbox,
)
from crispedit.legacy.runner import base_prefilter_fields
from crispedit.prefilter.policy import (
    adjudicate_evidence,
    adjudicate_terminal_no_change,
    derive_state_text_match,
    normalize_pair_evidence,
    normalize_single_image_evidence,
    normalize_slots,
    normalize_text_match,
)
from PIL import Image


def _slots(subgoals):
    return normalize_slots(
        {
            "instruction_status": "NORMAL",
            "subgoals": subgoals,
            "confidence": 0.95,
        },
        subgoals[0]["edit_type"],
    )


def _single(facts=None, scene="indoor", style="photo"):
    return normalize_single_image_evidence(
        {
            "scene_description": scene,
            "scene_label": scene,
            "style_label": style,
            "subject_identity": "same person",
            "facts": facts or [],
            "confidence": 0.95,
        }
    )


def _fact(index, role, present, count=None, attribute="", pose=""):
    return {
        "subgoal_index": index,
        "role": role,
        "query": "query",
        "present": present,
        "count": count,
        "attribute_value": attribute,
        "pose": pose,
        "confidence": 0.95,
    }


def _paired(scope="CONTROLLED_EDIT"):
    return normalize_pair_evidence(
        {
            "visible_differences": [
                {"description": "requested region visibly changed", "significance": "HIGH"}
            ],
            "same_subject": "YES",
            "composition_preserved": "YES",
            "unrelated_regions_preserved": "YES",
            "edit_scope": scope,
            "confidence": 0.95,
        }
    )


def _match(count, values=None):
    values = values or ["MATCH"] * count
    return normalize_text_match(
        {
            "subgoal_matches": [
                {
                    "subgoal_index": index,
                    "match": value,
                    "reason": "supported",
                    "confidence": 0.95,
                }
                for index, value in enumerate(values)
            ],
            "overall_match": "MATCH",
            "confidence": 0.95,
        },
        count,
    )


def _review(source, target, confidence=0.95):
    return {
        "method": "FOCUSED_INDEPENDENT_SINGLE_IMAGE_STATES",
        "source": source,
        "target": target,
        "confidence": confidence,
    }


def test_current_manifest_uses_method_name_without_version_fields():
    slots = _slots([{"edit_type": "add", "object_b": "tree"}])
    source = _single([_fact(0, "OBJECT_B", "NO", count=0)])
    target = _single([_fact(0, "OBJECT_B", "YES", count=1)])
    decision = adjudicate_evidence(slots, source, target, _paired(), _match(1))
    result = {
        "model_output": {
            "verdict": decision["verdict"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "change_presence": "CLEAR",
            "instruction_achievement": "YES",
            "failure_mode": "OTHER",
        },
        "decision": decision["decision"],
        "parse_ok": True,
        "evidence": {
            "slots": slots,
            "source": source,
            "target": target,
            "paired": _paired(),
            "text_match": _match(1),
            "predicates": decision["predicates"],
            "reason_codes": decision["reason_codes"],
        },
    }
    _, manifest = _make_audit_and_manifest_rows(
        {"type": "add", "instruction": "add a tree"},
        result,
        row_idx=0,
        raw_type="add",
        model_name="model",
        run_id="run",
    )
    assert manifest["prefilter_method"] == "fact"
    assert manifest["prefilter_evidence_schema"] == "fact_evidence"
    assert "prefilter_prompt_version" not in manifest
    assert "filter_version" not in manifest
    propagated = base_prefilter_fields(manifest)
    assert propagated["prefilter_method"] == "fact"
    assert propagated["prefilter_evidence_schema"] == "fact_evidence"


def test_simple_instruction_templates_skip_slot_mllm_conservatively():
    remove = parse_simple_instruction_slots("remove the white great dane", "remove")
    assert remove["subgoals"][0]["object_a"] == "white great dane"

    replace = parse_simple_instruction_slots(
        "replace the bottle with a glass of juice", "replace"
    )
    assert replace["subgoals"][0]["object_a"] == "bottle"
    assert replace["subgoals"][0]["object_b"] == "glass of juice"

    background = parse_simple_instruction_slots(
        "change the background to a misty forest", "background change"
    )
    assert background["subgoals"][0]["attribute"] == "a misty forest"

    packshot = parse_simple_instruction_slots(
        "replace the bottle in the packshot with a glass of juice", "replace"
    )
    assert packshot["subgoals"][0]["object_a"] == "bottle"
    assert packshot["subgoals"][0]["location"] == "in the packshot"

    color = parse_simple_instruction_slots(
        "Turn girl positioned in the central-left area into darker skin tone with blue attire",
        "color",
    )
    assert [item["attribute"] for item in color["subgoals"]] == [
        "darker skin tone",
        "blue attire",
    ]
    assert all(item["location"] == "central-left area" for item in color["subgoals"])

    simple_color = parse_simple_instruction_slots(
        "Turn cap positioned in the upper-central area into red", "color"
    )
    assert simple_color["subgoals"][0]["object_a"] == "cap"
    assert simple_color["subgoals"][0]["attribute"] == "red"
    assert simple_color["subgoals"][0]["location"] == "upper-central area"

    assert parse_simple_instruction_slots(
        "Turn wheel positioned in the center into black with red brake caliper",
        "color",
    ) is None

    motion = parse_simple_instruction_slots(
        "The woman lowers her hand and straightens her posture.", "motion change"
    )
    assert [item["object_a"] for item in motion["subgoals"]] == [
        "woman's hand",
        "woman's posture",
    ]
    assert [item["attribute"] for item in motion["subgoals"]] == [
        "lowered",
        "straightened",
    ]

    assert parse_simple_instruction_slots("add a garden with raised beds", "add") is None
    assert parse_simple_instruction_slots("Make the room feel warmer", "color") is None

    style = parse_simple_instruction_slots(
        "Would you create a pixel art version of this image?", "style"
    )
    assert style["subgoals"][0]["edit_type"] == "style"


def test_state_match_fast_path_only_accepts_decisive_positive_states():
    slots = _slots([{"edit_type": "replace", "object_a": "bottle", "object_b": "glass"}])
    source = _single(
        [_fact(0, "OBJECT_A", "YES", count=1), _fact(0, "OBJECT_B", "NO", count=0)]
    )
    target = _single(
        [_fact(0, "OBJECT_A", "NO", count=0), _fact(0, "OBJECT_B", "YES", count=1)]
    )
    match = derive_state_text_match(slots, source, target)
    assert match is not None
    assert match["method"] == "CODE_STATES"
    assert match["subgoal_matches"][0]["match"] == "MATCH"

    uncertain_target = _single(
        [_fact(0, "OBJECT_A", "NO", count=0), _fact(0, "OBJECT_B", "UNCLEAR")]
    )
    assert derive_state_text_match(slots, source, uncertain_target) is None

    color_slots = _slots(
        [{"edit_type": "color", "object_a": "shirt", "attribute": "blue"}]
    )
    color_source = _single([_fact(0, "OBJECT_A", "YES", attribute="red")])
    color_target = _single([_fact(0, "OBJECT_A", "YES", attribute="blue")])
    assert derive_state_text_match(color_slots, color_source, color_target) is None


def test_terminal_no_change_fast_path_is_high_precision():
    add_slots = _slots([{"edit_type": "add", "object_b": "family"}])
    same_source = _single([_fact(0, "OBJECT_B", "YES", count=3)])
    same_target = _single([_fact(0, "OBJECT_B", "YES", count=3)])
    decision = adjudicate_terminal_no_change(add_slots, same_source, same_target)
    assert decision is not None
    assert decision["verdict"] == "FAIL"
    assert decision["predicates"]["fast_terminal_no_change"] == "TRUE"

    remove_slots = _slots([{"edit_type": "remove", "object_a": "small dog"}])
    absent = _single([_fact(0, "OBJECT_A", "NO")])
    assert adjudicate_terminal_no_change(remove_slots, absent, absent) is None

    color_slots = _slots(
        [{"edit_type": "color", "object_a": "shirt", "attribute": "blue"}]
    )
    blue = _single([_fact(0, "OBJECT_A", "YES", attribute="blue")])
    assert adjudicate_terminal_no_change(color_slots, blue, blue) is None


def test_remove_requires_object_in_source_and_absence_in_target():
    slots = _slots([{"edit_type": "remove", "object_a": "white great dane"}])
    no_op = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_A", "NO")]),
        _single([_fact(0, "OBJECT_A", "NO")]),
        _paired(),
        _match(1),
    )
    assert no_op["verdict"] == "UNSURE"
    assert no_op["review_needed"]

    assert no_op["predicates"]["subgoal_0_change"] == "FALSE"

    valid = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_A", "YES")]),
        _single([_fact(0, "OBJECT_A", "NO")]),
        _paired(),
        _match(1),
    )
    assert valid["verdict"] == "PASS"
    assert valid["decision"] == "keep"


def test_add_requires_count_or_presence_increase():
    slots = _slots([{"edit_type": "add", "object_b": "family"}])
    no_op = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_B", "YES", count=1)]),
        _single([_fact(0, "OBJECT_B", "YES", count=1)]),
        _paired(),
        _match(1),
    )
    assert no_op["verdict"] == "UNSURE"
    assert no_op["review_needed"]

    reviewed_add = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_B", "YES", count=1)]),
        _single([_fact(0, "OBJECT_B", "YES", count=1)]),
        _paired(),
        _match(1),
        review=_review(
            _single([_fact(0, "OBJECT_B", "YES", count=1)]),
            _single([_fact(0, "OBJECT_B", "YES", count=2)]),
        ),
    )
    assert reviewed_add["verdict"] == "PASS"
    assert reviewed_add["predicates"]["subgoal_0_review_refined_change"] == "TRUE"

    valid = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_B", "NO", count=0)]),
        _single([_fact(0, "OBJECT_B", "YES", count=1)]),
        _paired(),
        _match(1),
    )
    assert valid["verdict"] == "PASS"


def test_replace_requires_full_a_to_b_transition():
    slots = _slots(
        [{"edit_type": "replace", "object_a": "spaceship", "object_b": "submarine"}]
    )
    valid = adjudicate_evidence(
        slots,
        _single(
            [_fact(0, "OBJECT_A", "YES"), _fact(0, "OBJECT_B", "NO")]
        ),
        _single(
            [_fact(0, "OBJECT_A", "NO"), _fact(0, "OBJECT_B", "YES")]
        ),
        _paired(),
        _match(1),
    )
    assert valid["verdict"] == "PASS"

    source_already_target = adjudicate_evidence(
        slots,
        _single(
            [_fact(0, "OBJECT_A", "NO"), _fact(0, "OBJECT_B", "YES")]
        ),
        _single(
            [_fact(0, "OBJECT_A", "NO"), _fact(0, "OBJECT_B", "YES")]
        ),
        _paired(),
        _match(1),
    )
    assert source_already_target["verdict"] == "UNSURE"
    assert source_already_target["review_needed"]

    one_of_two_replaced = adjudicate_evidence(
        slots,
        _single(
            [
                _fact(0, "OBJECT_A", "YES", count=2),
                _fact(0, "OBJECT_B", "NO", count=0),
            ]
        ),
        _single(
            [
                _fact(0, "OBJECT_A", "YES", count=1),
                _fact(0, "OBJECT_B", "YES", count=1),
            ]
        ),
        _paired(),
        _match(1),
    )
    assert one_of_two_replaced["verdict"] == "PASS"


def test_compound_color_subgoals_are_independent_and_all_required():
    slots = _slots(
        [
            {"edit_type": "color", "object_a": "girl", "attribute": "darker skin tone"},
            {"edit_type": "color", "object_a": "attire", "attribute": "blue"},
        ]
    )
    source = _single(
        [
            _fact(0, "OBJECT_A", "YES", attribute="light skin tone"),
            _fact(1, "OBJECT_A", "YES", attribute="white"),
        ]
    )
    target = _single(
        [
            _fact(0, "OBJECT_A", "YES", attribute="dark skin tone"),
            _fact(1, "OBJECT_A", "YES", attribute="blue"),
        ]
    )
    result = adjudicate_evidence(slots, source, target, _paired(), _match(2))
    assert result["verdict"] == "PASS"
    assert result["predicates"]["subgoal_0_change"] == "TRUE"
    assert result["predicates"]["subgoal_1_change"] == "TRUE"

    partial_match = adjudicate_evidence(
        slots, source, target, _paired(), _match(2, ["MATCH", "NOT_MENTIONED"])
    )
    assert partial_match["verdict"] == "UNSURE"
    assert partial_match["review_needed"]

    reviewed = adjudicate_evidence(
        slots,
        source,
        target,
        _paired(),
        _match(2, ["NOT_MENTIONED", "MATCH"]),
        review=_review(source, target),
    )
    assert reviewed["verdict"] == "PASS"


def test_fact_only_review_derives_compound_motion_result_from_states():
    slots = _slots(
        [
            {"edit_type": "motion", "object_a": "hand", "attribute": "lowered"},
            {
                "edit_type": "motion",
                "object_a": "posture",
                "attribute": "straightened",
            },
        ]
    )
    empty_source = _single(
        [
            _fact(0, "PART", "YES", pose=""),
            _fact(1, "PART", "YES", pose=""),
        ]
    )
    empty_target = _single(
        [
            _fact(0, "PART", "YES", pose=""),
            _fact(1, "PART", "YES", pose=""),
        ]
    )
    review_source = _single(
        [
            _fact(0, "PART", "YES", pose="raised and gesturing"),
            _fact(1, "PART", "YES", pose="forward-leaning seated posture"),
        ]
    )
    review_target = _single(
        [
            _fact(0, "PART", "YES", pose="lowered and resting on lap"),
            _fact(1, "PART", "YES", pose="seated posture"),
        ]
    )

    result = adjudicate_evidence(
        slots,
        empty_source,
        empty_target,
        _paired(),
        _match(2, ["MATCH", "NOT_MENTIONED"]),
        review=_review(review_source, review_target),
    )

    assert result["verdict"] == "PASS"
    assert result["decision"] == "keep"
    assert result["predicates"]["subgoal_0_review_change"] == "TRUE"
    assert result["predicates"]["subgoal_1_review_change"] == "TRUE"
    assert result["predicates"]["subgoal_0_review_target_match"] == "TRUE"
    assert result["predicates"]["subgoal_1_review_target_match"] == "TRUE"
    assert result["predicates"]["review_consistent"] == "TRUE"


def test_review_prompt_requests_states_and_has_no_support_label():
    slots = _slots([{"edit_type": "motion", "object_a": "hand", "attribute": "lowered"}])
    prompt = build_review_state_prompt(slots)
    assert '"change_supported"' not in prompt
    assert '"source_state"' not in prompt
    assert '"target_state"' not in prompt
    assert "lowered" not in prompt.lower()
    assert "requested_attribute_slot" not in prompt
    assert '"pose"' in prompt
    assert "ONE image" in prompt


def test_review_prompt_keeps_attribute_dimension_but_hides_target_values():
    slots = _slots(
        [
            {"edit_type": "color", "object_a": "girl", "attribute": "darker skin tone"},
            {"edit_type": "color", "object_a": "girl", "attribute": "blue attire"},
        ]
    )
    prompt = build_review_state_prompt(slots).lower()
    assert "apparent lightness" in prompt
    assert "seven-level scale" in prompt
    assert "attire/clothing" in prompt
    assert "darker" not in prompt
    assert "blue" not in prompt
    # The established base prompt is intentionally unchanged; only the focused
    # independent review must hide requested target values.
    base_prompt = build_single_image_prompt(slots).lower()
    assert "darker" in base_prompt
    assert "blue" in base_prompt


def test_review_can_use_target_state_when_source_state_is_unclear():
    slots = _slots(
        [{"edit_type": "color", "object_a": "people", "attribute": "darker skin tone"}]
    )
    unclear_source = _single([_fact(0, "OBJECT_A", "YES", attribute="")])
    dark_target = _single([_fact(0, "OBJECT_A", "YES", attribute="dark")])
    result = adjudicate_evidence(
        slots,
        unclear_source,
        dark_target,
        _paired(),
        _match(1, ["NOT_MENTIONED"]),
        review=_review(unclear_source, dark_target),
    )
    assert result["verdict"] == "PASS"
    assert result["predicates"]["subgoal_0_review_change"] == "UNKNOWN"
    assert result["predicates"]["subgoal_0_review_target_match"] == "TRUE"
    assert result["predicates"]["subgoal_0_review_blind_change_fallback"] == "TRUE"


def test_global_regeneration_cannot_keep():
    slots = _slots([{"edit_type": "add", "object_b": "tree"}])
    result = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_B", "NO", count=0)]),
        _single([_fact(0, "OBJECT_B", "YES", count=1)]),
        _paired(scope="GLOBAL_REGEN"),
        _match(1),
    )
    assert result["verdict"] == "FAIL"
    assert "not_global_regeneration" in result["reason_codes"]

    style_slots = _slots([{"edit_type": "style", "attribute": "watercolor"}])
    controlled_style = adjudicate_evidence(
        style_slots,
        _single(style="photo"),
        _single(style="watercolor"),
        _paired(scope="GLOBAL_REGEN"),
        _match(1),
    )
    assert controlled_style["verdict"] == "PASS"


def test_background_uses_specific_attribute_not_only_coarse_scene_label():
    slots = _slots([{"edit_type": "background", "attribute": "starry night sky"}])
    result = adjudicate_evidence(
        slots,
        _single([_fact(0, "OBJECT_A", "YES", attribute="warm orange clouds")], scene="space"),
        _single([_fact(0, "OBJECT_A", "YES", attribute="dark blue starry sky")], scene="space"),
        _paired(),
        _match(1),
    )
    assert result["verdict"] == "PASS"


def test_single_image_prompt_contains_slots_but_not_original_instruction():
    instruction = "remove the white great dane from the astronauts"
    slots = _slots([{"edit_type": "remove", "object_a": "white great dane"}])
    prompt = build_single_image_prompt(slots)
    assert "white great dane" in prompt
    assert instruction not in prompt
    assert build_questionnaire(slots)[0]["role"] == "OBJECT_A"


def test_color_outfit_is_not_misclassified_as_object_replacement():
    slots = normalize_slots(
        {
            "instruction_status": "NORMAL",
            "confidence": 0.95,
            "subgoals": [
                {
                    "edit_type": "replace",
                    "object_a": "person's outfit",
                    "object_b": "red outfit",
                }
            ],
        },
        "color",
    )
    subgoal = slots["subgoals"][0]
    assert subgoal["edit_type"] == "color"
    assert subgoal["attribute"] == "red outfit"
    assert subgoal["object_b"] == ""


def test_only_localized_edit_types_need_source_guided_target_crop():
    for edit_type in ("remove", "color", "motion"):
        assert needs_source_guided_target_crop(
            _slots([{"edit_type": edit_type, "object_a": "subject"}])
        )
    for edit_type in ("add", "replace", "background", "style"):
        assert not needs_source_guided_target_crop(
            _slots([{"edit_type": edit_type, "object_a": "subject"}])
        )


def test_normalized_crop_expands_and_stays_in_bounds():
    image = Image.new("RGB", (100, 80), "white")
    crop = crop_normalized_bbox(image, [0.2, 0.25, 0.4, 0.5], expansion=0.25)
    assert crop is not None
    assert crop.width > 20
    assert crop.height > 20


def test_qwen_0_to_1000_bboxes_are_normalized_and_location_can_refine_group_box():
    evidence = normalize_single_image_evidence(
        {
            "facts": [
                {
                    "subgoal_index": 0,
                    "role": "OBJECT_A",
                    "query": "person",
                    "present": "YES",
                    "bboxes": [[200, 250, 400, 500]],
                }
            ]
        }
    )
    assert evidence["facts"][0]["bboxes"][0] == [0.2, 0.25, 0.4, 0.5]

    slots = _slots(
        [
            {
                "edit_type": "color",
                "object_a": "people",
                "attribute": "blue clothing",
                "location": "right side",
            }
        ]
    )
    whole_image = _single([_fact(0, "OBJECT_A", "YES")])
    whole_image["facts"][0]["bboxes"] = [[0.0, 0.0, 1.0, 1.0]]
    assert select_focus_bbox(slots, whole_image) == [0.38, 0.1, 1.0, 0.9]
