from sa1b_pipeline.grounding import validate_spatial_grounding


def _unit(object_id, instruction):
    return {"object_id": object_id, "instruction_ref": instruction}


def _obj(object_id, x, category="door"):
    return {"object_id": object_id, "category": category, "bbox": [x, 0, x + 10, 20]}


def test_leftmost_and_rightmost_are_checked_against_same_category_instances():
    objects = [_obj("left", 10), _obj("middle", 50), _obj("right", 90)]
    units = [
        _unit("left", "Add a sign to the leftmost door."),
        _unit("middle", "Change the color of the rightmost door to blue."),
    ]
    errors = validate_spatial_grounding(units, objects)
    assert errors == ["middle:rightmost_target_is_not_rightmost_door"]


def test_right_of_center_requires_a_target_on_the_right_side():
    objects = [_obj("left", 10), _obj("center", 50), _obj("right", 90)]
    assert validate_spatial_grounding(
        [_unit("right", "Remove the door to the right of the center.")], objects
    ) == []
    assert validate_spatial_grounding(
        [_unit("left", "Remove the door to the right of the center.")], objects
    ) == ["left:right_of_center_target_is_not_right_of_center_door"]


def test_non_spatial_instruction_is_left_to_visual_review():
    objects = [_obj("a", 10), _obj("b", 90)]
    assert validate_spatial_grounding([_unit("a", "Change the color of the door to blue.")], objects) == []
