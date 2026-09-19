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
    normalize_generated,
    same_replacement_category,
    validation_feedback,
)
from synthesis_pipeline.visual_prompt_utils import instruction_target_crop


class InstructionPlanningTest(unittest.TestCase):
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
