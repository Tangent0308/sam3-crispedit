import numpy as np

from crispedit.mask.attachments import attachment_phrase, complete_supported_holes


def test_container_completion_requires_actual_content_support_inside_envelope():
    from crispedit.mask.attachments import container_envelope, container_contents
    rim = np.zeros((100,100), dtype=np.uint8)
    rim[10:90,10:90] = 1
    rim[30:70,30:100] = 0  # Open cutout, not an enclosed hole.
    envelope = container_envelope(rim)
    assert envelope is not None
    food = np.zeros_like(rim)
    food[35:65,35:65] = 1
    output, audit = complete_supported_holes(rim, food, food, envelope=envelope)
    assert output[45,45] and not output[45,75] and not output[45,95]
    assert audit['added_pixels'] == 900
    unchanged, audit = complete_supported_holes(rim, food, None, envelope=envelope)
    np.testing.assert_array_equal(unchanged, rim)
    assert container_contents('bowl containing pie slice and fork') == ['pie slice', 'fork']


def test_complete_seat_queries_cushions_without_assuming_they_exist():
    assert attachment_phrase('brown fabric armchair') == 'cushions'
    assert attachment_phrase('sofa') == 'cushions'
    assert attachment_phrase('chair leg') == ''
    assert attachment_phrase('plate') == ''


def test_content_seam_cleanup_preserves_unrelated_openings_and_outer_boundary():
    from crispedit.mask.attachments import close_content_seams
    envelope = np.zeros((100,100), dtype=np.uint8)
    envelope[10:90,10:90] = 1
    mask = envelope.copy()
    mask[20:40,20:40] = 0  # Unrelated hole must survive.
    mask[60:75,60] = 0  # One-pixel seam beside supported food.
    food = np.zeros_like(mask)
    food[60:75,61:75] = 1
    result, area = close_content_seams(mask, food, envelope)
    assert area > 0 and result[65,60]
    assert not result[30,30] and np.all(result <= envelope)
    same, area = close_content_seams(mask, food*0, envelope)
    np.testing.assert_array_equal(same, mask)
    assert area == 0


def test_content_crossing_rim_only_adds_supported_pixels_inside_envelope():
    rim = np.zeros((100,100), dtype=np.uint8)
    rim[10:90,10:90] = 1
    rim[30:70,30:100] = 0
    from crispedit.mask.attachments import container_envelope
    envelope = container_envelope(rim)
    fork = np.zeros_like(rim)
    fork[45:50,60:100] = 1
    output, audit = complete_supported_holes(rim, fork, fork, envelope=envelope)
    assert audit['added_pixels'] > 0
    assert output[47,70] and not output[47,95]
    assert np.all((output & (rim == 0)) <= envelope)


def example():
    mask = np.zeros((100,100), dtype=np.uint8)
    mask[10:90,10:90] = 1
    mask[20:40,20:40] = 0  # Real opening, no attachment here.
    mask[50:80,50:80] = 0  # Missing cushion.
    attachment = np.zeros_like(mask)
    attachment[50:80,50:80] = 1
    return mask, attachment


def test_only_semantically_supported_hole_is_filled():
    mask, part = example()
    output, audit = complete_supported_holes(mask, part, part)
    assert output[60,60] and not output[30,30] and not output[0,0]
    assert audit['added_pixels'] == 900
    assert not mask[60,60]  # Do not mutate the cached base proposal.


def test_disagreeing_or_missing_proposal_does_not_fill_holes():
    mask, part = example()
    for alternative in [None, np.zeros_like(mask)]:
        output, audit = complete_supported_holes(mask, part, alternative)
        np.testing.assert_array_equal(output, mask)
        assert not audit['added_pixels']


def test_enclosing_object_degeneracy_does_not_fill_real_openings():
    mask, _ = example()
    dense = np.zeros_like(mask)
    dense[10:90,10:90] = 1
    output, audit = complete_supported_holes(mask, dense, dense)
    np.testing.assert_array_equal(output, mask)
    assert audit['reason'] == 'ATTACHMENT_GEOMETRY_REJECTED'


def test_attachment_outside_parent_is_rejected():
    mask, part = example()
    part[:, :10] = 1
    output, audit = complete_supported_holes(mask, part, part)
    np.testing.assert_array_equal(output, mask)
    assert not audit['added_pixels']


def test_attachment_must_be_named_not_invented():
    assert attachment_phrase('dark wicker chairs with blue cushions') == 'blue cushions'
    assert attachment_phrase('green robe') == ''


def test_larger_joint_proposal_cannot_contribute_unsupported_pixels():
    mask, part = example()
    joint = np.zeros_like(mask)
    joint[10:90,10:90] = 1
    output, audit = complete_supported_holes(mask, part, joint)
    assert output[60,60] and not output[30,30]
    assert audit['added_pixels'] == 900


def test_group_query_extras_do_not_discard_a_fully_matched_cushion():
    mask, part = example()
    text, joint = part.copy(), part.copy()
    text[0:30,0:30] = 1
    joint[0:30,70:100] = 1
    output,audit = complete_supported_holes(mask,text,joint)
    assert audit['component_matching'] and audit['added_pixels'] == 900
    assert output[60,60] and not output[30,30] and not output[0,0]
