from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from sa1b_pipeline import per_mask_planning as pm, group_mask_planning as gm


def test_shared_grounding_and_distinct_content_with_key_binding():
    assignments = {'U1': 'replace', 'U2': 'replace'}
    raw = {'U2': {'ref': 'the left flowerpot', 'edit': {'replacement': 'a round terracotta pot'}},
           'U1': {'ref': 'the red bottle', 'edit': {'replacement': 'a glass water bottle'}}}
    result = gm.combine(raw, 'twins', assignments, [{'object_id': '1'}, {'object_id': '2'}])
    assert result['feasible'] and result['_planning_calls'] == 1
    assert [x['ref_phrase'] for x in result['members']] == ['the red bottle', 'the left flowerpot']
    assert result['members'][0]['payload'] != result['members'][1]['payload']
    assert set(gm.schema(assignments)['properties']) == {'U1', 'U2'}
    for text in [pm.prompt('replace'), gm.prompt(assignments, 'twins')]:
        assert 'THAT SAME region or object in the original Image 1' in text
        assert 'ONLY Image 1 and this phrase' in text
        assert 'verify any relation or ordinal in the FULL photo' in text
    assert 'SEPARATE copy' in gm.prompt(assignments, 'twins')
    assert 'coexist' in gm.prompt(assignments, 'multi')


def test_missing_or_invalid_target_is_not_assigned_to_its_neighbor():
    raw = {'U2': {'ref': 'the blue bag', 'edit': {'property': 'color', 'value': 'red'}}}
    assignments = {'U1': 'attribute', 'U2': 'attribute'}
    result = gm.combine(raw, 'multi', assignments, [{'object_id': '1'}, {'object_id': '2'}])
    assert not result['targets'][0]['feasible'] and result['targets'][1]['feasible']
    raw['U9'] = raw['U2']
    result = gm.combine(raw, 'multi', assignments, [{'object_id': '1'}, {'object_id': '2'}])
    assert not any(t['feasible'] for t in result['targets'])
    assert result['_group_mask_response'] == raw


def test_group_views_and_single_request_resume(tmp_path):
    source = tmp_path/'source.png'
    Image.new('RGB', (400, 300), 'gray').save(source)
    objects = []
    for i, x in enumerate([40, 280]):
        m = np.zeros((300, 400), dtype='uint8');m[100:170, x:x+50] = 255
        path = tmp_path/f'm{i}.png';Image.fromarray(m).save(path)
        objects.append({'object_id': str(i), 'mask_path': str(path)})
    pair, masks, digest = gm.views({'image_path': str(source)}, objects, tmp_path/'views')
    assert len(pair) == 2 and Image.open(pair[1]).size == Image.open(source).size
    assert np.array_equal(np.array(Image.open(pair[0])), np.array(Image.open(source)))
    labels = json.loads((Path(pair[0]).parent/'views.json').read_text())['targets']
    assert [(x['unit_id'], x['object_id']) for x in labels] == [('U1', '0'), ('U2', '1')]
    class Planner:
        model_id = 'test';batch_size = 8;calls = 0
        def generate_multimage(self, jobs, schemas=None):
            self.calls += len(jobs)
            assert len(jobs) == 1 and len(jobs[0][0]) == 2
            return [{'U1': {'ref': 'the left chair', 'edit': {}},
                     'U2': {'ref': 'the right lamp', 'edit': {}}}]
    planner = Planner()
    item = dict(gid='g', objects=objects, selection={'operation': 'remove'}, views=pair)
    first = gm.plan_batch(planner, [deepcopy(item)], kind='twins', root=tmp_path)
    assert first[0]['feasible'] and planner.calls == 1
    assert gm.plan_batch(planner, [deepcopy(item)], kind='twins', root=tmp_path) == first
    assert planner.calls == 1
    item['planning_source_hints'] = {'U1': 'changed'}
    with pytest.raises(ValueError, match='checkpoint'):
        gm.plan_batch(planner, [item], kind='twins', root=tmp_path)
