from collections import Counter
from copy import deepcopy

import pytest

from test_multi_transactional import plan
from sa1b_pipeline import single_planning as sp
from sa1b_pipeline.group_planning import build_multi_group
from sa1b_pipeline.group_schema import _validate_member


def _inputs(plan):
    """The build_multi_group inputs the plan fixture was built from (4 cups: remove, replace, attribute, add)."""
    from sa1b_pipeline.geometry import mask_geometry
    import numpy as np
    from PIL import Image
    objects, paths, geoms, targets = [], [], [], []
    for i, u in enumerate(plan['units']):
        m = np.asarray(Image.open(u['mask']).convert('L')) > 0
        objects.append({'object_id': str(i), 'category': 'cup', 'mask_path': u['mask'], 'metadata': {}})
        paths.append(u['mask'])
        g = mask_geometry(m); g['point'] = u['point']; geoms.append(g)
        op = u['operation']
        targets.append(dict(unit_id=f'U{i+1}', object_id=str(i), operation=op, scope='cup', ref_phrase=u['ref'],
                            payload='a square cup' if op == 'replace' else 'a star decal' if op == 'add' else '',
                            attributes=[{'property': 'color', 'value': 'blue'}] if op == 'attribute' else [],
                            placement={'relation': 'on', 'site': '', 'anchor_mode': 'support'} if op == 'add' else None,
                            target_part='', remove_mode='whole', qualifier='', evidence='visible'))
    raw = {'feasible': True, 'targets': targets, 'generation_order': ['U1', 'U2', 'U3', 'U4']}
    assignments = {o['object_id']: u['operation'] for o, u in zip(objects, plan['units'])}
    return {'candidate': {'source_id': 'source', 'image_path': plan['source_image']}, 'objects': objects, 'raw': raw,
            'paths': paths, 'geoms': geoms, 'assignments': assignments}


def _build(fn, x, keep):
    k = [int(u[1:]) - 1 for u in keep]
    raw = deepcopy(x['raw'])
    raw['targets'] = [dict(raw['targets'][i], unit_id=f'U{j+1}') for j, i in enumerate(k)]
    raw['generation_order'] = [f'U{j+1}' for j in range(len(k))]
    objects = [x['objects'][i] for i in k]
    return fn(x['candidate'], objects, raw, [x['paths'][i] for i in k], split='train', group_id='g',
              geometry=[x['geoms'][i] for i in k], assignments={o['object_id']: x['assignments'][o['object_id']] for o in objects})


def test_single_target_builder_equals_shared_builder_and_allows_one_target(plan):
    x = _inputs(plan)
    builder = sp.single_target_builder()
    assert _build(builder, x, ['U1', 'U2', 'U3', 'U4']) == _build(build_multi_group, x, ['U1', 'U2', 'U3', 'U4'])
    with pytest.raises(ValueError, match='fewer than two'):
        _build(build_multi_group, x, ['U3'])
    one = _build(builder, x, ['U3'])
    assert one['n_targets'] == 1 and one['operations'] == ['attribute']
    retained = sp.retain_valid_targets(deepcopy(one))
    assert retained['units'] == one['units'] and retained['single_target_planning'] == sp.VERSION
    _validate_member(dict(retained['units'][0], support_mask=retained['units'][0]['mask']), require_resolved_regions=False)


def test_builder_refuses_when_the_shared_rule_changes(monkeypatch):
    sp._STATE.pop('builder', None)
    monkeypatch.setattr(sp, 'MULTI_MINIMUM', 'a line that is not in build_multi_group')
    with pytest.raises(RuntimeError, match='review single-target planning'):
        sp.single_target_builder()
    sp._STATE.pop('builder', None)


def test_retention_drops_invalid_members_and_protects_their_masks(plan):
    group = deepcopy(plan)
    group['units'][1] = dict(group['units'][1], instruction_spec={})
    out = sp.retain_valid_targets(group)
    assert [u['unit_id'] for u in out['units']] == ['U1', 'U3', 'U4'] and out['n_targets'] == 3
    assert plan['units'][1]['mask'] in out['protected_source_masks']
    group = deepcopy(plan)
    for u in group['units']:
        u['instruction_spec'] = {}
    with pytest.raises(ValueError, match='no target'):
        sp.retain_valid_targets(group)


def test_selection_rows_and_dropping_complete_operations():
    rows = [{'candidate': {'source_id': 'a'}, 'objects': [{'object_id': 1}, {'object_id': 2}], 'operations': ['remove', 'add'],
             'evidence_selection': {'members': {'1': {'score': 1}, '2': {'score': 2}}}},
            {'candidate': {'source_id': 'b'}, 'objects': [{'object_id': 3}, {'object_id': 4}], 'operations': ['add', 'attribute']}]
    kept, dropped = sp.drop_complete_operations(rows, {'remove': 5, 'add': 0, 'replace': 3, 'attribute': 0})
    assert len(kept) == 1 and kept[0]['operations'] == ['remove'] and kept[0]['objects'] == [{'object_id': 1}]
    assert kept[0]['evidence_selection']['members'] == {'1': {'score': 1}} and kept[0]['single_target_dropped'] == ['add']
    assert dropped == Counter({'add': 2, 'attribute': 1})
    sp.validate_selection_rows(kept)
    with pytest.raises(RuntimeError, match='distinct'):
        sp.validate_selection_rows([{'objects': [1, 2], 'operations': ['add', 'add']}])
    with pytest.raises(RuntimeError, match='distinct'):
        sp.validate_selection_rows([{'objects': [], 'operations': []}])


def test_enable_patches_this_process_only_for_the_given_module():
    import types
    from sa1b_pipeline import multi_execution_contract as mec
    original = mec.retain_feasible_plan
    module = types.SimpleNamespace(_validate_multi_selection_operations=None, build_multi_group=None)
    try:
        sp._STATE.pop('enabled', None)
        assert sp.enable(module) == {'version': sp.VERSION}
        assert module.build_multi_group is sp.single_target_builder()
        assert module._validate_multi_selection_operations is sp.validate_selection_rows
        assert mec.retain_feasible_plan is sp.retain_valid_targets
    finally:
        mec.retain_feasible_plan = original
        sp._STATE.pop('enabled', None)
