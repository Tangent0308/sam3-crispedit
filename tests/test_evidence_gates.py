import json

import numpy as np
from PIL import Image

from sa1b_pipeline import evidence_gates as eg
from sa1b_pipeline import evidence_selection as es


def test_reader_verdict_requires_every_box_on_target():
    target = [100, 100, 300, 300]
    assert eg.reader_verdict([[102, 98, 298, 305]], target)[0] == 'pass'
    assert eg.reader_verdict([[102, 98, 298, 305], [600, 600, 700, 700]], target)[0] == 'ambiguous'
    assert eg.reader_verdict([[600, 600, 700, 700]], target)[0] == 'wrong'
    assert eg.reader_verdict([], target)[0] == 'none'
    assert eg.reader_verdict(None, target)[0] == 'invalid'
    # Degenerate boxes are ignored rather than counted as hits.
    assert eg.reader_verdict([[5, 5, 5, 9]], target)[0] == 'none'


def _views(tmp_path, n):
    root = tmp_path / 'views'
    root.mkdir()
    Image.new('RGB', (100, 80)).save(root / 'original.png')
    targets = []
    for i in range(1, n + 1):
        mask = np.zeros((80, 100), dtype=np.uint8)
        mask[10:30, 10 * i:10 * i + 15] = 255
        path = root / f'mask_{i}.png'
        Image.fromarray(mask).save(path)
        Image.new('RGB', (40, 40)).save(root / f'U{i}_crop.png')
        targets.append({'unit_id': f'U{i}', 'object_id': str(i), 'mask_path': str(path)})
    (root / 'views.json').write_text(json.dumps({'targets': targets}))
    return [[str(root / 'original.png'), str(root / f'U{i}_crop.png')] for i in range(1, n + 1)]


class FakePlanner:
    """Reader always finds U1/U2 correctly, misses U3; identity says U2 differs."""
    def __init__(self):
        self.seen = []

    def generate_multimage(self, jobs, *, schemas, token_limits, thinking):
        assert thinking == [False] * len(jobs)
        out = []
        for (images, prompt), schema in zip(jobs, schemas):
            self.seen.append(prompt)
            if schema is eg.READER_SCHEMA:
                if 'third thing' in prompt:
                    out.append({'boxes': [[900, 900, 990, 990]]})
                elif 'first thing' in prompt:
                    out.append({'boxes': [[100, 125, 250, 375]]})
                else:
                    out.append({'boxes': [[200, 125, 350, 375]]})
            elif schema is eg.IDENTITY_SCHEMA:
                out.append({'relation': 'same'})
            else:
                out.append({'change': 'appearance_only'})
        return out


def _plan(ops, refs):
    return [{'unit_id': f'U{i}', 'object_id': str(i), 'feasible': True, 'ref_phrase': ref, 'operation': op,
             'payload': 'a red one' if op == 'replace' else '', 'grounding': {'edit_feasible': True}}
            for i, (op, ref) in enumerate(zip(ops, refs), 1)]


def test_twins_keep_two_passing_overgenerated_members(tmp_path):
    views = _views(tmp_path, 3)
    item = {'per_mask_views': views, 'objects': [{'object_id': str(i), 'category': 'chair'} for i in (1, 2, 3)],
            'selection': {'twin_keep': 2}}
    plan = {'members': _plan(['remove'] * 3, ['the first thing', 'the second thing', 'the third thing']),
            'feasible': True}
    gates = eg.EvidenceGates()
    gates.apply(FakePlanner(), [item], [plan], 'twins')
    assert plan['feasible'] is True
    assert plan['_twin_kept_object_ids'] == ['1', '2']
    assert [m['object_id'] for m in plan['members']] == ['1', '2']
    dropped = plan['twin_overgeneration']['dropped']
    assert [(d['object_id'], d['reason']) for d in dropped] == [('3', 'evidence_gate: reader=wrong')]
    assert dropped[0]['evidence_gates']['pass'] is False


def test_failed_payload_marks_multi_target_infeasible(tmp_path):
    views = _views(tmp_path, 2)
    item = {'per_mask_views': views, 'objects': [{'object_id': '1', 'category': 'flag'},
                                                 {'object_id': '2', 'category': 'chair'}], 'selection': {}}
    plan = {'targets': _plan(['replace', 'attribute'], ['the first thing', 'the second thing'])}
    eg.EvidenceGates().apply(FakePlanner(), [item], [plan], 'multi')
    first, second = plan['targets']
    assert first['feasible'] is False and first['reason'] == 'evidence_gate: payload=appearance_only'
    assert first['grounding']['edit_feasible'] is False
    assert second['feasible'] is True and second['evidence_gates']['pass'] is True


def test_twin_without_overgeneration_requires_both_members(tmp_path):
    views = _views(tmp_path, 3)
    item = {'per_mask_views': views[:2], 'objects': [{'object_id': '1', 'category': 'chair'},
                                                    {'object_id': '3', 'category': 'chair'}], 'selection': {}}
    members = _plan(['remove'] * 2, ['the first thing', 'the third thing'])
    members[1]['object_id'] = '3'
    plan = {'members': members, 'feasible': True}
    eg.EvidenceGates().apply(FakePlanner(), [item], [plan], 'twins')
    assert plan['feasible'] is False


def test_alias_heads_and_mixed_person_categories():
    assert es.alias_heads('bust,marble,sculpture,statue,stone') == {'bust', 'marble', 'sculpture', 'statue', 'stone'}
    assert es.alias_heads('cardboard boxes') == {'box'}
    assert es.mixed_person_category('advertisement,person,sign,signs')
    assert not es.mixed_person_category('man,person,run,runner')
    assert not es.mixed_person_category('chair,plastic chair')
