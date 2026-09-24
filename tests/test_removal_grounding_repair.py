import numpy as np
from PIL import Image
import pytest

from synthesis_pipeline.repair_removal_grounding import candidate_panel, safe_candidate_anchor


def test_repair_anchor_lies_inside_candidate():
    mask = np.zeros((100, 160), dtype=bool)
    mask[30:80, 90:95] = True
    x, y = safe_candidate_anchor(mask)
    assert mask[round(y * 100 / 1000), round(x * 160 / 1000)]


def test_candidate_panel_does_not_mutate_inputs():
    source = Image.new('RGB', (160, 100), (30, 80, 120))
    mask = np.zeros((100, 160), dtype=bool)
    mask[30:80, 90:95] = True
    original_mask = mask.copy()
    original_source = np.asarray(source).copy()
    panel = candidate_panel(source, [(mask, .9)])
    assert panel.size == (384, 432)
    np.testing.assert_array_equal(mask, original_mask)
    np.testing.assert_array_equal(np.asarray(source), original_source)


def test_empty_candidates_fail_before_model_call():
    with pytest.raises(ValueError, match='empty candidate'):
        candidate_panel(Image.new('RGB', (20, 20)), [])
    with pytest.raises(ValueError, match='nonempty'):
        safe_candidate_anchor(np.zeros((20, 20), dtype=bool))
