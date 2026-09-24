"""Small regression tests for regional instruction planning inputs and wording."""

import re
import unittest

import numpy as np
from PIL import Image

from synthesis_pipeline.generate_samtok_plan import (
    PARTIAL_EDIT_UNIT_PATTERN,
    adds_same_category,
    adds_unmasked_wearable,
    build_prompt,
    contains_distinctive_reference,
    has_central_panel_seam,
    instruction_messages,
    largest_internal_hole,
    mask_geometry_hint,
    normalize_generated,
    protected_dependency_conflicts,
    ambiguous_anatomical_side,
    same_replacement_category,
    validation_feedback,
)
from synthesis_pipeline.audit_edit_pairs_v2 import (
    apply_low_change_veto,
    audit_messages as audit_v2_messages,
    build_prompt as build_audit_v2_prompt,
    normalize_result as normalize_audit_v2_result,
)
from synthesis_pipeline.visual_prompt_utils import (
    audit_two_image_inputs,
    audit_visual_inputs,
    instruction_target_crop,
)


def test_geometry_describes_membership_not_the_complement():
    mask = np.ones((90, 90), dtype=bool)
    mask[:, 40:45] = False
    hint = mask_geometry_hint(mask)
    assert '94.4%' in hint
    assert 'left, right, top, bottom' in hint
    assert 'middle-left 100%' in hint
    assert hint in build_prompt({'mask_geometry_hint': hint}, 'attribute')


def test_edge_connected_target_contour_closes_outside_photo():
    from synthesis_pipeline.visual_prompt_utils import instruction_target_crop
    source = Image.new('RGB', (100, 100), (30, 70, 120))
    crop = instruction_target_crop(source, np.ones((100, 100), dtype=bool), tile_size=128)
    pixels = np.asarray(crop)
    # Header occupies 42px; photograph has an 8px external margin on all sides.
    assert np.all(pixels[50:162, 8:120] == (30, 70, 120))
    assert np.all(pixels[48:50, 8:120] == 0)
    assert np.all(pixels[50:162, 6:8] == 0)


def test_retry_contains_the_actual_invalid_answer():
    source = Image.new('RGB', (32, 32))
    value = '{"editing_instruction":"Make the box white."}'
    messages = instruction_messages(source, source, {}, 'attribute', value, 'Retain left-side locator')
    assert value in messages[0]['content'][-1]['text']
    assert 'Retain left-side locator' in messages[0]['content'][-1]['text']


def test_same_category_with_multiple_adjectives_is_not_replacement():
    assert same_replacement_category('curved wall with metallic strips', 'smooth white curved wall')
    assert same_replacement_category('crane near archway', 'realistic lattice tower crane')
    assert not same_replacement_category('red bus behind motorcycle', 'black police cruiser')


def test_removed_owner_cannot_protect_its_explicitly_held_object():
    value = {'masked_content': 'Woman holding white game controllers in both hands.',
             'edit_unit_status': 'complete_object',
             'protected_objects': ['woman in black cardigan', 'white game controllers', 'floor lamp']}
    assert protected_dependency_conflicts(value, 'remove') == ['white game controllers']
    assert protected_dependency_conflicts(value, 'attribute') == []
    assert protected_dependency_conflicts(value, 'replace') == ['white game controllers']
    assert protected_dependency_conflicts({**value, 'edit_unit_status':'complete_part'}, 'remove') == []


def test_dependency_guard_does_not_infer_ownership_of_neighbor():
    value = {'masked_content': 'Woman holding a cup beside a man wearing a black cardigan.',
             'edit_unit_status': 'complete_object', 'protected_objects':['man', 'lamp']}
    assert protected_dependency_conflicts(value, 'remove') == []


def test_anatomical_side_policy_keeps_global_instance_locators():
    assert ambiguous_anatomical_side({'editing_instruction':
        'Add a watch to the left wrist of the man on the left.'})
    assert not ambiguous_anatomical_side({'editing_instruction':
        'Add a watch to the wrist holding the donut of the man on the left.'})
    assert not ambiguous_anatomical_side({'editing_instruction':
        'Change the left armrest of the central chair to gold.'})


def test_scope_prompt_is_type_specific_and_requires_visible_evidence():
    add = build_prompt({}, 'add')
    attribute = build_prompt({}, 'attribute')
    replace = build_prompt({}, 'replace')
    assert 'attachment surface actually exists and is visible' in add
    assert 'anatomical left/right' in add
    assert 'smallest complete visible part actually selected' in attribute
    assert 'use surface refinement' in attribute
    assert 'cannot repair that external interaction' in replace
    assert 'smallest complete visible part actually selected' not in add


class InstructionPlanningTest(unittest.TestCase):
    def test_two_image_audit_marks_only_source_and_preserves_target_pixels(self):
        array = np.full((100, 160, 3), (20, 80, 120), dtype=np.uint8)
        array[35:65, 65:95] = (150, 100, 40)
        source = Image.fromarray(array, mode="RGB")
        edited = Image.new("RGB", source.size, (40, 130, 75))
        mask = np.zeros((100, 160), dtype=bool)
        mask[35:65, 65:95] = True
        before, after = audit_two_image_inputs(source, edited, mask, longest_side=320)
        self.assertEqual(before.size, after.size)
        self.assertEqual(before.getpixel((160, 140)), (150, 100, 40))
        self.assertEqual(after.getpixel((160, 140)), (40, 130, 75))
        self.assertEqual(before.getpixel((127, 140)), (0, 0, 0))

    def test_two_image_audit_prompt_and_rewrite_are_independent_of_verdict(self):
        row = {
            "task_type": "replace",
            "editing_instruction": "Replace the center child with a small stuffed animal.",
            "refer_object": ["center child"],
        }
        source = Image.new("RGB", (32, 32), "white")
        edited = Image.new("RGB", (32, 32), "blue")
        content = audit_v2_messages(source, edited, row)[0]["content"]
        self.assertEqual(
            [item["image"] for item in content if item["type"] == "image"],
            [source, edited],
        )
        self.assertIn("Original instruction: Replace the center child", build_audit_v2_prompt(row))
        answer = {
            "source_target": "center child",
            "observed_edit": "The child becomes a cream teddy bear.",
            "target_match": True,
            "unexpected_change": None,
            "artifact": None,
            "visual_quality": "pass",
            "instruction_match": "fail",
            "observed_instruction": "Replace the center child with a cream teddy bear.",
            "reason": "The result is a larger bear.",
        }
        parsed = normalize_audit_v2_result(answer, "replace")
        self.assertEqual(parsed["quality"], "fail")
        self.assertEqual(parsed["salvage_status"], "manual_review_candidate")
        answer["visual_quality"] = "fail"
        self.assertIsNone(normalize_audit_v2_result(answer, "replace")["rewrite_candidate"])
        answer["visual_quality"] = "pass"
        answer["observed_instruction"] = "Add a cream bear beside the child."
        self.assertIsNone(normalize_audit_v2_result(answer, "replace")["rewrite_candidate"])

    def test_two_image_audit_vetoes_unrequested_change_and_no_edit_rewrite(self):
        answer = {
            "source_target": "brown teddy bear",
            "observed_edit": "No visible change was made.",
            "target_match": True,
            "unexpected_change": None,
            "artifact": None,
            "visual_quality": "pass",
            "instruction_match": "fail",
            "observed_instruction": "Remove the brown teddy bear.",
            "reason": "The edit is not visible.",
        }
        parsed = normalize_audit_v2_result(answer, "remove")
        self.assertEqual(parsed["visual_quality"], "fail")
        self.assertIsNone(parsed["rewrite_candidate"])
        answer["observed_edit"] = "The teddy bear becomes a second baby."
        answer["unexpected_change"] = "An unrequested baby appears."
        parsed = normalize_audit_v2_result(answer, "remove")
        self.assertEqual(parsed["visual_quality"], "fail")
        self.assertEqual(parsed["quality"], "fail")

    def test_two_image_audit_accepts_null_observed_edit_as_no_edit_failure(self):
        answer = {
            "source_target": "child in the dark coat",
            "observed_edit": None,
            "target_match": False,
            "unexpected_change": None,
            "artifact": None,
            "visual_quality": "pass",
            "instruction_match": "fail",
            "observed_instruction": None,
            "reason": "The child's skin tone is unchanged.",
        }
        parsed = normalize_audit_v2_result(answer, "attribute")
        self.assertEqual(parsed["quality"], "fail")
        self.assertEqual(parsed["visual_quality"], "fail")
        self.assertIsNone(parsed["rewrite_candidate"])

    def test_two_image_audit_low_change_veto_is_limited_to_remove_replace(self):
        answer = {
            "source_target": "the selected object",
            "observed_edit": "The selected object disappears.",
            "target_match": True,
            "unexpected_change": None,
            "artifact": None,
            "visual_quality": "pass",
            "instruction_match": "pass",
            "observed_instruction": "Remove the selected object.",
            "reason": "The object appears absent.",
        }
        for task_type in ("remove", "replace"):
            parsed = normalize_audit_v2_result(answer, task_type)
            gated = apply_low_change_veto(
                parsed, task_type, {"inside_changed_fraction": 0.2}, 0.3
            )
            self.assertEqual(gated["quality"], "fail")
            self.assertEqual(gated["instruction_match"], "fail")
            self.assertIsNone(gated["rewrite_candidate"])
            self.assertEqual(gated["metric_veto"]["threshold"], 0.3)
        parsed = normalize_audit_v2_result(answer, "attribute")
        self.assertEqual(
            apply_low_change_veto(
                parsed, "attribute", {"inside_changed_fraction": 0.2}, 0.3
            )["quality"],
            "pass",
        )

    def test_two_image_audit_checklist_old_target_veto(self):
        answer = {
            "source_target": "the selected person",
            "edited_target_area": "The same person remains in the edited image.",
            "old_target_visible": True,
            "requested_change_visible": True,
            "observed_edit": "The person seems to be removed.",
            "target_match": True,
            "unexpected_change": None,
            "artifact": None,
            "visual_quality": "pass",
            "instruction_match": "pass",
            "observed_instruction": None,
            "reason": "The person is no longer visible.",
        }
        parsed = normalize_audit_v2_result(answer, "remove", require_checklist=True)
        self.assertEqual(parsed["quality"], "fail")
        self.assertIn("old_target_still_visible", parsed["checklist_vetoes"])

    def test_two_image_audit_accepts_boolean_verdict_fields(self):
        answer = {
            "source_target": "the selected lamp",
            "observed_edit": "The lamp pole vanishes, leaving floating globes.",
            "target_match": True,
            "unexpected_change": "The pole disappears without the requested bench.",
            "artifact": "The globes float without support.",
            "visual_quality": True,
            "instruction_match": False,
            "observed_instruction": None,
            "reason": "A bench was not added and the lamps are unsupported.",
        }
        parsed = normalize_audit_v2_result(answer, "add")
        self.assertEqual(parsed["visual_quality"], "fail")
        self.assertEqual(parsed["instruction_match"], "fail")
        self.assertEqual(parsed["quality"], "fail")

    def test_two_image_audit_artifact_severity_parser(self):
        answer = {
            "source_target": "the selected camera",
            "observed_edit": "The camera is removed with a faint soft edge.",
            "target_match": True,
            "unexpected_change": None,
            "artifact": "A faint soft edge remains in the foliage.",
            "artifact_severity": "minor",
            "visual_quality": "pass",
            "instruction_match": "pass",
            "observed_instruction": None,
            "reason": "The removal is otherwise coherent.",
        }
        self.assertEqual(
            normalize_audit_v2_result(
                answer, "remove", require_artifact_severity=True
            )["quality"],
            "pass",
        )
        answer["artifact_severity"] = "major"
        self.assertEqual(
            normalize_audit_v2_result(
                answer, "remove", require_artifact_severity=True
            )["quality"],
            "fail",
        )

    def test_crop_preserves_photo_and_input_has_exactly_two_images(self):
        pixels = np.full((120, 160, 3), (70, 120, 160), dtype=np.uint8)
        pixels[45:75, 55:90] = (140, 90, 50)
        source = Image.fromarray(pixels, mode="RGB")
        mask = np.zeros((120, 160), dtype=bool)
        mask[45:75, 55:90] = True
        crop = instruction_target_crop(source, mask, tile_size=256)
        colors = {tuple(color) for color in np.asarray(crop).reshape(-1, 3)}
        self.assertIn((140, 90, 50), colors)
        self.assertIn((70, 120, 160), colors)
        messages = instruction_messages(source, crop, {}, "attribute")
        self.assertEqual(
            [item["image"] for item in messages[0]["content"] if item["type"] == "image"],
            [source, crop],
        )

    def test_audit_tight_crop_actually_magnifies_small_target(self):
        source = Image.new("RGB", (200, 200), (10, 50, 80))
        edited = source.copy()
        mask = np.zeros((200, 200), dtype=bool)
        mask[95:105, 95:105] = True
        local, context, full = audit_visual_inputs(source, edited, mask)
        self.assertEqual(local.getpixel((100, 100)), (10, 50, 80))
        self.assertEqual(context.getpixel((100, 100)), (10, 50, 80))
        self.assertEqual(full.getpixel((320, 350)), (10, 50, 80))

    def test_each_prompt_only_contains_its_required_edit_type(self):
        for task_type in ("add", "remove", "replace", "attribute"):
            prompt = build_prompt({}, task_type).lower()
            self.assertIn(f"required edit type: {task_type}", prompt)
            for other in ("add", "remove", "replace", "attribute"):
                if other != task_type:
                    self.assertIsNone(re.search(rf"\b{other}\b", prompt))

    def test_full_instruction_must_retain_located_reference(self):
        base = {
            "masked_content": "dark brown scaled serpentine railing ornament",
            "edit_unit_status": "complete_part",
            "outside_dependencies": "none",
            "mask_compatibility": "compatible",
            "compatibility_reason": "",
            "new_instruction": "Change the railing ornament to gold.",
        }
        ambiguous = {
            **base,
            "refer_object": "dark brown scaled serpentine railing ornament",
            "editing_instruction": "Change the dark brown scaled serpentine railing ornament to gold.",
        }
        self.assertIsNone(normalize_generated(ambiguous, "attribute"))
        located = {
            **base,
            "refer_object": "left serpentine railing ornament",
            "editing_instruction": "Change the left serpentine railing ornament to gold.",
        }
        self.assertIsNotNone(normalize_generated(located, "attribute"))
        self.assertTrue(
            contains_distinctive_reference(
                located["editing_instruction"], located["refer_object"]
            )
        )
        located["editing_instruction"] = "Change the railing ornament to gold."
        self.assertIsNone(normalize_generated(located, "attribute"))

    def test_small_grammatical_omission_keeps_landmark(self):
        reference = "person in blue shirt standing behind green umbrella"
        self.assertTrue(
            contains_distinctive_reference(
                "Add a sign to the person in blue shirt behind green umbrella.",
                reference,
            )
        )
        self.assertFalse(
            contains_distinctive_reference(
                "Add a sign to the person in blue shirt.", reference
            )
        )
        self.assertFalse(
            contains_distinctive_reference(
                "Make the small sheep near the blue fence wet.",
                "small sheep near blue fence, lower right",
            )
        )

    def test_add_placement_direction_cannot_replace_target_landmark(self):
        self.assertFalse(
            contains_distinctive_reference(
                "Add a blue napkin to the left of the white mug with handle.",
                "white mug with handle, left of glass jar",
            )
        )
        self.assertTrue(
            contains_distinctive_reference(
                "Add a blue napkin beside the white mug with handle left of glass jar.",
                "white mug with handle, left of glass jar",
            )
        )

    def test_background_alone_is_not_a_unique_locator(self):
        value = {
            "masked_content": "a tall evergreen tree",
            "edit_unit_status": "complete_object",
            "outside_dependencies": "none",
            "refer_object": "the evergreen tree in the background",
            "mask_compatibility": "compatible",
            "compatibility_reason": "",
            "editing_instruction": "Remove the evergreen tree in the background.",
            "new_instruction": "Remove the evergreen tree in the background.",
        }
        self.assertIsNone(normalize_generated(value, "remove"))
        value["refer_object"] = "the evergreen tree right of center"
        value["editing_instruction"] = "Remove the evergreen tree right of center."
        value["new_instruction"] = "Remove the evergreen tree right of center."
        self.assertIsNotNone(normalize_generated(value, "remove"))
        value["refer_object"] = "the evergreen tree in the front row"
        value["editing_instruction"] = "Remove the evergreen tree in the front row."
        value["new_instruction"] = "Remove the evergreen tree in the front row."
        self.assertIsNotNone(normalize_generated(value, "remove"))

    def test_retry_receives_specific_localization_feedback(self):
        value = {
            "masked_content": "dark scaled serpent railing",
            "edit_unit_status": "complete_part",
            "outside_dependencies": "none",
            "refer_object": "dark scaled serpent railing",
            "mask_compatibility": "compatible",
            "editing_instruction": "Change the dark scaled serpent railing to gold.",
            "new_instruction": "Change the serpent railing to gold.",
        }
        feedback = validation_feedback(value, "attribute")
        self.assertIn("full-image position", feedback)
        source = Image.new("RGB", (32, 32), "white")
        prompt = instruction_messages(
            source, source, {}, "attribute", str(value), feedback
        )[0]["content"][-1]["text"]
        self.assertIn(feedback, prompt)

    def test_duplicate_add_partial_replace_and_new_wearable_are_rejected(self):
        self.assertTrue(
            adds_same_category(
                "Add a glowing hexagonal lantern under the left lantern.",
                "left hexagonal lantern",
            )
        )
        self.assertFalse(
            adds_same_category(
                "Add a small bird perched on the left zebra.", "left zebra"
            )
        )
        self.assertIsNotNone(
            PARTIAL_EDIT_UNIT_PATTERN.search("A person's left arm and upper leg")
        )
        self.assertIsNotNone(
            PARTIAL_EDIT_UNIT_PATTERN.search("The back of a brown dog")
        )
        self.assertTrue(
            adds_unmasked_wearable(
                "Make the man wear a red tie.", "man in light blue shirt"
            )
        )
        self.assertFalse(
            adds_unmasked_wearable(
                "Make the man wear a blue shirt.", "man in orange shirt"
            )
        )

    def test_plural_mask_inventory_cannot_be_one_compatible_instance(self):
        value = {
            "masked_content": "Two giraffes near the tree trunk",
            "edit_unit_status": "complete_object",
            "outside_dependencies": "none",
            "refer_object": "the two giraffes on the left",
            "mask_compatibility": "compatible",
            "compatibility_reason": "",
            "editing_instruction": "Add a ribbon to the two giraffes on the left.",
            "new_instruction": "Add a ribbon to the giraffes.",
        }
        self.assertIsNone(normalize_generated(value, "add"))

    def test_remove_cannot_expand_adult_mask_to_unmasked_calf(self):
        value = {
            "masked_content": "Adult giraffe with calf nursing on dirt path",
            "edit_unit_status": "complete_object",
            "outside_dependencies": "none",
            "refer_object": "adult giraffe and calf near green tree",
            "mask_compatibility": "compatible",
            "compatibility_reason": "",
            "editing_instruction": "Remove the adult giraffe and calf near green tree.",
            "new_instruction": "Remove adult giraffe and calf near green tree.",
        }
        self.assertIsNone(normalize_generated(value, "remove"))

    def test_other_persons_board_is_not_target_dependency(self):
        value = {
            "masked_content": "person in gray snowsuit doing handstand",
            "edit_unit_status": "complete_object",
            "outside_dependencies": "red snowboard held by other person",
            "refer_object": "person in gray snowsuit doing handstand",
            "mask_compatibility": "compatible",
            "compatibility_reason": "",
            "editing_instruction": "Remove the person in gray snowsuit doing handstand.",
            "new_instruction": "Remove the person doing handstand.",
        }
        result = normalize_generated(value, "remove")
        self.assertIsNotNone(result)
        self.assertEqual(result["outside_dependencies"], "none")

    def test_replacement_changes_category_not_just_material(self):
        self.assertTrue(
            same_replacement_category(
                "dark wooden pulpit on the right",
                "modern glass and steel pulpit",
            )
        )
        self.assertTrue(
            same_replacement_category("foreground pizza with olives", "pizza topped with mushrooms")
        )
        self.assertFalse(
            same_replacement_category("blue boat near orange boat", "red sailboat")
        )

    def test_large_uneditable_island_is_measured(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        mask[40:60, 40:60] = False
        pixels, fraction = largest_internal_hole(mask)
        self.assertEqual(pixels, 400)
        self.assertAlmostEqual(fraction, 400 / 3200)

    def test_side_by_side_panel_is_filtered(self):
        pixels = np.full((64, 120, 3), 170, dtype=np.uint8)
        pixels[:, 59:62] = 0
        self.assertTrue(has_central_panel_seam(Image.fromarray(pixels)))
        pixels[:, 59:62] = 170
        self.assertFalse(has_central_panel_seam(Image.fromarray(pixels)))


if __name__ == "__main__":
    unittest.main()
