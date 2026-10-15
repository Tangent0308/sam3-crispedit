import copy

import pytest

from sa1b_pipeline.instructions import build_unit
from sa1b_pipeline.schema import EditUnit, validate_unit


def test_direct_site_and_mask_role_survive_training_serialization():
    unit = build_unit(
        {'operation': 'add', 'ref_phrase': 'the left sleeve', 'scope': 'sleeve',
         'payload': 'a red patch', 'placement': {'relation': 'sewn onto',
         'site': 'area below the cuff', 'anchor_mode': 'support'}},
        {'object_id': '1', 'category': 'sleeve'}, 'mask.png',
        {'bbox': [0, 0, 8, 8], 'point': [4, 4]}, [],
    )
    restored = EditUnit(**unit.to_dict())
    validate_unit(restored)
    assert restored.mask_role == 'anchor'
    assert restored.mask_semantics['anchor_mode'] == 'support'
    assert not restored.mask_semantics['output_difference_ground_truth']
    assert restored.instruction_noref == 'Add a red patch sewn onto the selected region.'
    assert 'below the cuff' in restored.instruction_ref
    assert restored.mask_semantics['source_region_resolution_required']
    broken = copy.deepcopy(restored)
    broken.mask_semantics['source_role'] = 'target'
    with pytest.raises(ValueError, match='mask semantics'):
        validate_unit(broken)
