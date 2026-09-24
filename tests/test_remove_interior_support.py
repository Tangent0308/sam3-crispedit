"""Geometry-only regression tests; no object vocabulary or case identifiers."""
import numpy as np
import pytest
from PIL import Image

from utils.context_edit import compose_grounded_crop
from utils.remove_support import adaptive_remove_support


def pocket_scene():
    source = Image.new('RGB', (400, 400), (60, 60, 60))
    target = np.zeros((400, 400), bool)
    target[50:350, 50:350] = True
    target[140:260, 140:260] = False
    target[260:350, 190:210] = False  # Open notch between visible target parts.
    raw = np.asarray(source).copy()
    raw[target] = 180
    # Only part of an excluded interior changes, connected to selected pixels.
    raw[170:230, 130:205] = 180
    return source, Image.fromarray(raw), target


@pytest.mark.parametrize('rotation', range(4))
def test_changed_interior_pocket_completed_independent_of_orientation(rotation):
    source, raw, target = pocket_scene()
    source = Image.fromarray(np.rot90(np.asarray(source), rotation).copy())
    raw = Image.fromarray(np.rot90(np.asarray(raw), rotation).copy())
    target = np.rot90(target, rotation).copy()
    frozen = target.copy()
    v1, _ = adaptive_remove_support(source, raw, target)
    v2, evidence = adaptive_remove_support(source, raw, target, policy='adaptive-remove-v2')
    assert np.any(v2 & ~v1)
    assert evidence['interior_added_pixels'] > 0
    assert np.array_equal(target, frozen)
    # The center of the unmodified interior must not become writable.
    untouched = np.zeros(target.shape, bool)
    untouched[190:210, 220:235] = True
    untouched = np.rot90(untouched, rotation).copy()
    assert not np.any(v2 & untouched)


def test_protected_thin_occluder_survives_even_if_raw_erases_it():
    source, raw, target = pocket_scene()
    protected = np.zeros_like(target)
    protected[140:260, 192:196] = True
    result, _ = compose_grounded_crop(source, raw, target, 'remove', (0, 0, 400, 400),
        protected, remove_composition_policy='adaptive-remove-v2')
    assert np.array_equal(np.asarray(result)[protected], np.asarray(source)[protected])
    assert np.array_equal(np.asarray(result)[:20], np.asarray(source)[:20])


def test_large_excluded_interior_not_filled_even_when_raw_changes_it_all():
    source, _, target = pocket_scene()
    target[260:350, 190:210] = True  # Close a genuinely excluded large interior.
    raw = Image.new('RGB', source.size, (180, 180, 180))
    support, _ = adaptive_remove_support(source, raw, target, policy='adaptive-remove-v2')
    assert not support[200, 200]


def test_noop_and_disconnected_changes_do_not_expand():
    source, _, target = pocket_scene()
    support, _ = adaptive_remove_support(source, source, target, policy='adaptive-remove-v2')
    assert np.array_equal(support, target)
    pixels = np.asarray(source).copy()
    pixels[195:205, 195:205] = 180
    support, _ = adaptive_remove_support(source, Image.fromarray(pixels), target,
        policy='adaptive-remove-v2')
    assert not support[200, 200]


def test_invalid_policy_rejected():
    source, raw, target = pocket_scene()
    with pytest.raises(ValueError):
        adaptive_remove_support(source, raw, target, policy='unknown')


@pytest.mark.parametrize('policy',['erase-neutral-v1','erase-prefill-v1','erase-neutral-v2'])
def test_erased_condition_is_opt_in_and_preserves_source_label_and_neighbors(policy):
    from utils.remove_support import erased_removal_condition
    source=Image.new('RGB',(80,80),(31,63,95));target=np.zeros((80,80),bool)
    target[30:50,30:50]=True;guard=np.zeros_like(target);guard[:,50:55]=True
    before=target.copy();pixels=np.asarray(source).copy()
    result,evidence=erased_removal_condition(source,target,guard,policy)
    assert np.array_equal(target,before) and np.array_equal(np.asarray(source),pixels)
    assert np.array_equal(np.asarray(result)[guard],pixels[guard])
    assert evidence['protected_overlap_pixels']==0 and not evidence['source_mask_modified']


@pytest.mark.parametrize('task', ['add', 'replace', 'attribute'])
def test_non_remove_outputs_are_unchanged(task):
    source, raw, target = pocket_scene()
    outputs = []
    for policy in ['adaptive-remove-v1', 'adaptive-remove-v2']:
        final, alpha = compose_grounded_crop(source, raw, target, task, (0, 0, 400, 400),
            replacement_mask=target, remove_composition_policy=policy)
        outputs.append((np.asarray(final), np.asarray(alpha)))
    assert all(np.array_equal(a, b) for a, b in zip(*outputs))
