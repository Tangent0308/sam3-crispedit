import json
from pathlib import Path

import numpy as np
from PIL import Image

from sa1b_pipeline import single_salvage as ss

SPEC = {'version': '2.0', 'operation': 'remove', 'ref': 'the dark box', 'scope': 'dark box', 'payload': '',
        'target_part': '', 'remove_mode': 'whole', 'attributes': [], 'placement': {}, 'qualifier': ''}


def _files(tmp_path):
    rng = np.random.default_rng(0)
    source = np.clip(120 + rng.normal(0, 4, (400, 600, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((400, 600), bool)
    mask[180:220, 100:140] = True
    source[mask] = 20
    result = source.copy()
    result[mask] = 120
    paths = [tmp_path / 'source.png', tmp_path / 'result.png', tmp_path / 'mask.png']
    Image.fromarray(source).save(paths[0]); Image.fromarray(result).save(paths[1])
    Image.fromarray(mask.astype(np.uint8) * 255).save(paths[2])
    return [str(p) for p in paths]


def _member(uid, res, mpath):
    return {'unit_id': uid, 'operation': 'remove', 'mask': mpath, 'support_mask': mpath, 'edited_image': res,
            'instruction': 'Remove the dark box.', 'instruction_noref': 'Remove the selected region.',
            'instruction_spec': dict(SPEC), 'blind_eligible': True, 'object_id': '7',
            'deterministic': {'decision': 'pending_semantic_audit', 'failure_reasons': []}}


def _twin_group(tmp_path, deterministic_pass=False):
    src, res, mpath = _files(tmp_path)
    from sa1b_pipeline.twin_contract import ADAPTED
    pairs = [{'member': 'U1', 'other': 'U2', 'changed_fraction_in_other_source_mask': 0.0},
             {'member': 'U2', 'other': 'U1', 'changed_fraction_in_other_source_mask': 0.0}]
    checks = {'deterministic': {'pass': deterministic_pass}, 'support_gap': {'pass': True},
              'cross_target_preservation': {'pass': True, 'pairs': pairs}}
    return {'twin_id': 'twins-x', 'source_id': 'sa_1', 'source_image': src, 'operation': 'remove',
            'members': [_member('U1', res, mpath), _member('U2', res, mpath)], 'twin_contract_version': ADAPTED,
            'group_checks': checks}


class _Auditor:
    """Main audit: image_ok per unit from `ok`; verifier jobs answer `answers`."""
    batch_size = 8

    def __init__(self, ok, keep_singles=False, verifier=None, answers=None):
        self.ok, self.keep_singles, self.verifier, self.answers, self.calls = ok, keep_singles, verifier, answers, []

    def call_batch(self, jobs):
        self.calls.append([j['job_id'] for j in jobs])
        out = []
        for j in jobs:
            if '::verify::' in j['job_id']:
                out.append({'job_id': j['job_id'], 'semantic_audit': self.answers, 'finish_reason': 'stop'})
                continue
            ids = j['output_schema']['properties']['units']['items']['properties']['unit_id']['enum']
            units = [{'unit_id': u, 'source_state': 'a dark box', 'result_state': 'plain wall', 'image_ok': self.ok[u],
                      'scope_status': 'matched', 'observed_operation': 'remove', 'reference_unique': True,
                      'ref_issue': '', 'final_ref': ''} for u in ids]
            out.append({'job_id': j['job_id'], 'semantic_audit': {'units': units}, 'finish_reason': 'stop'})
        return out


def test_twin_member_eligibility(tmp_path):
    group = _twin_group(tmp_path)
    assert ss.twin_member_eligible(group, group['members'][0])
    group['group_checks']['cross_target_preservation']['pairs'][0]['changed_fraction_in_other_source_mask'] = .3
    assert not ss.twin_member_eligible(group, group['members'][0])       # its edit changed the other target
    assert ss.twin_member_eligible(group, group['members'][1])
    group['members'][1]['deterministic']['decision'] = 'fail'
    assert not ss.twin_member_eligible(group, group['members'][1])
    group['members'][1]['deterministic']['decision'] = 'pending_semantic_audit'
    group['members'][1]['blind_eligible'] = False
    assert not ss.twin_member_eligible(group, group['members'][1])


def test_twin_audit_salvages_members_only_with_keep_singles(tmp_path):
    from sa1b_pipeline import twin_compact_audit as tca
    # Group fails a group-level check (e.g. member size ratio): normally never audited.
    plain = _Auditor({'U1': True, 'U2': False})
    out = tca.audit([_twin_group(tmp_path)], plain, tmp_path / 'a')[0]
    assert plain.calls == [] and out['group_checks']['group_pass'] is False and 'singles' not in out
    keep = _Auditor({'U1': True, 'U2': False}, keep_singles=True)
    out = tca.audit([_twin_group(tmp_path)], keep, tmp_path / 'b')[0]
    assert keep.calls == [['twins-x']] and out['group_checks']['group_pass'] is False
    assert [s['unit_id'] for s in out['singles']] == ['U1']
    single = out['singles'][0]
    assert single['provenance']['kind'] == 'twins' and single['provenance']['other_units'] == ['U2']
    assert single['edited_image'] == out['members'][0]['edited_image'] and single['instruction'] == 'Remove the dark box.'
    assert single['image_size'] == [600, 400] and single['audit']['decision'] == 'pass'


def test_twin_partner_kept_when_verifier_rejects_one_member(tmp_path):
    from sa1b_pipeline import twin_compact_audit as tca
    from sa1b_pipeline import edit_verifier as ev
    answers = {k: False for k, _, _ in ev.QUESTIONS['remove']}

    class Split(_Auditor):
        def call_batch(self, jobs):
            out = super().call_batch(jobs)
            for o in out:
                if o['job_id'].endswith('::verify::U2'):
                    o['semantic_audit'] = {**answers, 'target_remains': True}
            return out

    auditor = Split({'U1': True, 'U2': True}, keep_singles=True, verifier={}, answers=answers)
    out = tca.audit([_twin_group(tmp_path, deterministic_pass=True)], auditor, tmp_path / 'c')[0]
    assert out['group_checks']['group_pass'] is False
    assert [s['unit_id'] for s in out['singles']] == ['U1']
    assert out['singles'][0]['audit']['verifier']['pass']
    # An accepted group ships as a twin; nothing is salvaged.
    ok = _Auditor({'U1': True, 'U2': True}, keep_singles=True, verifier={}, answers=answers)
    out = tca.audit([_twin_group(tmp_path, deterministic_pass=True)], ok, tmp_path / 'd')[0]
    assert out['group_checks']['group_pass'] is True and out['singles'] == []


def test_multi_single_only_for_exactly_one_passing_target(tmp_path):
    src, res, mpath = _files(tmp_path)
    passing = {'decision': 'pass', 'operation_correct': True, 'target_done': True, 'non_target_preserved': True,
               'natural': True, 'host_integrity': True, 'source_state': 'a box', 'result_state': 'wall'}
    unit = {**_member('U2', res, mpath), 'semantic_audit': passing, 'component_audit_image': res}
    group = {'multi_id': 'multi-x', 'source_id': 'sa_1', 'source_image': src, 'units': [unit], 'joint_target': res,
             'joint_audit': {'pass': False}, 'retention': {'decisions': [{'unit_id': 'U1'}, {'unit_id': 'U2'}]}}
    single = ss.multi_single(group)
    assert single['unit_id'] == 'U2' and single['provenance'] == {
        'version': ss.VERSION, 'kind': 'multi', 'group_id': 'multi-x', 'reason': 'multi_single_target_left',
        'other_units': ['U1']}
    assert ss.multi_single({**group, 'joint_audit': {'pass': True}}) is None
    assert ss.multi_single({**group, 'units': [unit, {**unit, 'unit_id': 'U3'}]}) is None
    assert ss.multi_single({**group, 'units': [{**unit, 'semantic_audit': {**passing, 'natural': False}}]}) is None


def test_lone_multi_target_is_audited_only_with_keep_singles(tmp_path, monkeypatch):
    from sa1b_pipeline import multi_component_audit as mca
    from sa1b_pipeline import multi_parallel as mp
    src, res, mpath = _files(tmp_path)
    audited = []
    monkeypatch.setattr(mca, 'generation_prefix', lambda provisional, group=None: provisional)
    monkeypatch.setattr(mp, 'fuse_components', lambda source, pack, out: (res, {}))
    monkeypatch.setattr(mca, 'audit_component', lambda group, ms, final, auditor, out, aligned=False:
                        audited.append([m['unit_id'] for m in ms]) or {'audits': {}, 'accepted': True})
    monkeypatch.setattr(mca, 'complete_component', lambda group, prov, cur, auditor, out, joint_report=None,
                        joint_image=None: {'decisions': [], 'kept': [], 'current': cur})

    def results():
        member = {**_member('U1', res, mpath), 'step_output_image': res}
        return [{'provisional': [{'member': member, 'accepted': True}],
                 'decisions': [{'unit_id': 'U1', 'accepted': True, 'reasons': []}], 'current': res}]

    group = {'multi_id': 'multi-x', 'source_image': src}
    out, packs = mca.complete_components_together(group, results(), _Auditor({}), tmp_path / 'a')
    assert audited == [] and packs == 0
    assert out[0]['decisions'][0]['reasons'] == ['fewer_than_two_provisionally_valid_targets']
    out, packs = mca.complete_components_together(group, results(), _Auditor({}, keep_singles=True), tmp_path / 'b')
    assert audited == [['U1']] and packs == 1
