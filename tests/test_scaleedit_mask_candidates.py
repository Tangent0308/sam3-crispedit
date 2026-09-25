# Ported with the current SAM3 core from crispedit-labeling b045184.
import numpy as np

from scaleedit.mask.candidates import select_object_candidate, topology, use_object_selection


def square():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[15:85, 15:85] = 1
    return mask


def semantic(mask, score=.95):
    return mask, {'concept_score': score}


def test_concept_confidence_is_not_mask_quality():
    whole = square()
    holes = whole.copy()
    holes[25:75:4, 25:75:4] = 0
    result = select_object_candidate([], semantic(whole, .6), semantic(holes, .99))
    np.testing.assert_array_equal(result[0], whole)
    assert result[1]['concept_score'] == .6
    assert 'predicted_iou' not in result[1]
    assert result[3]['selected'] == 'text'


def test_fragment_not_automatically_better_because_smaller():
    whole = square()
    tiny = np.zeros_like(whole)
    tiny[20:80:4, 20:80:4] = 1
    result = select_object_candidate([], semantic(whole), semantic(tiny))
    np.testing.assert_array_equal(result[0], whole)


def test_fragmented_semantic_can_use_non_top_scoring_pvs():
    whole = square()
    fragmented = whole.copy()
    fragmented[20:80:2, 20:80:2] = 0
    specks = np.zeros_like(whole)
    specks[5:95:4, 5:95:4] = 1
    result = select_object_candidate([(fragmented, {'predicted_iou': .96}),
                                      (whole, {'predicted_iou': .9})],
                                     (None, {}), semantic(specks, .99))
    np.testing.assert_array_equal(result[0], whole)
    assert result[2] == 'pvs'
    assert result[1]['predicted_iou'] == .9


def test_uncorroborated_enclosing_pvs_does_not_override_semantics():
    whole = square()
    enclosing = np.ones_like(whole)
    result = select_object_candidate([(enclosing, {'predicted_iou': .99})],
                                     semantic(whole), semantic(whole))
    np.testing.assert_array_equal(result[0], whole)
    assert result[2] == 'pcs'


def test_multi_object_is_not_rejected_for_disconnection():
    two = square()
    two[:, 45:55] = 0
    result = select_object_candidate([], semantic(two), semantic(two))
    assert not result[1]['selection_review']
    np.testing.assert_array_equal(result[0], two)


def test_fragmented_only_candidate_requires_review_without_inventing_pixels():
    specks = np.zeros((100, 100), dtype=np.uint8)
    specks[10:90:3, 10:90:3] = 1
    result = select_object_candidate([], (None, {}), semantic(specks))
    assert result[1]['selection_review']
    np.testing.assert_array_equal(result[0], specks)


def test_thin_and_sparse_regions_keep_their_policy():
    for ref in ['fairy lights', 'eye', 'tentacles', 'facial piercings', 'wire', 'wooden ladder']:
        assert not use_object_selection(ref, 'object', 'object')
    assert not use_object_selection('birds', 'aggregate_region', 'sparse')
    assert use_object_selection('porcelain jar', 'object', 'object')


def test_topology_measurement_does_not_modify_mask():
    original = square()
    original[35:65, 35:65] = 0
    copy = original.copy()
    features = topology(original)
    assert features['holes'] > 0
    np.testing.assert_array_equal(original, copy)


def test_complete_pvs_requires_both_semantic_prompts():
    whole = square()
    inner = np.zeros_like(whole)
    inner[20:80, 20:80] = 1
    perforated = whole.copy()
    perforated[20:80:4, 20:80:4] = 0
    result = select_object_candidate([(whole, {'predicted_iou': .95})],
                                     semantic(inner), semantic(perforated))
    assert result[2] == 'pvs'
    np.testing.assert_array_equal(result[0], whole)


def test_real_occlusion_hole_does_not_favor_incomplete_surface():
    whole = square()
    whole[35:55, 25:60] = 0
    truncated = whole.copy()
    truncated[55:] = 0
    result = select_object_candidate([], semantic(truncated), semantic(whole))
    assert topology(whole)['perforation'] == 0
    np.testing.assert_array_equal(result[0], whole)
def test_single_object_collapsed_extent_can_recover_supported_visual_candidate():
    import numpy as np
    from scaleedit.mask.candidates import select_object_candidate
    whole = np.zeros((200,200),dtype=np.uint8)
    whole[20:180,20:180] = 1
    scraps = np.zeros_like(whole)
    scraps[20:40,20:60] = 1
    scraps[174:180,174:180] = 1
    scraps[174:180,20:26] = 1
    selected = select_object_candidate([(whole,{'predicted_iou':.78})],
        (None,{}),(scraps,{}),allow_extent_recovery=True)
    assert selected[1]['selection_reason'] == 'OBJECT_RECOVER_COLLAPSED_EXTENT'
    np.testing.assert_array_equal(selected[0],whole)
    compound = select_object_candidate([(whole,{'predicted_iou':.78})],(None,{}),(scraps,{}))
    assert compound[1]['selection_reason'] != 'OBJECT_RECOVER_COLLAPSED_EXTENT'
