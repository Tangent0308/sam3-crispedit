import json
from pathlib import Path

import numpy as np
from PIL import Image

from sa1b_pipeline import edit_verifier as ev


def _write(directory, source, result, mask):
    paths = [directory / 'source.png', directory / 'result.png', directory / 'mask.png']
    Image.fromarray(source).save(paths[0])
    Image.fromarray(result).save(paths[1])
    Image.fromarray(mask.astype(np.uint8) * 255).save(paths[2])
    return [str(p) for p in paths]


def _images(h=400, w=600):
    rng = np.random.default_rng(0)
    source = np.clip(120 + rng.normal(0, 4, (h, w, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), bool)
    mask[180:220, 100:140] = True
    return source, mask


def test_prompt_fills_slots_without_double_article():
    spec = {'operation': 'replace', 'ref': 'the black flag', 'scope': 'black flag', 'payload': 'a small red lantern'}
    text = ev.prompt('replace', spec, 'Replace the black flag with a small red lantern.')
    assert '"Replace the black flag with a small red lantern."' in text
    assert 'No small red lantern is visible in AFTER where the black flag was.' in text
    assert 'The new small red lantern' in text and 'new a small' not in text
    attr = {'operation': 'attribute', 'ref': 'the door', 'attributes': [{'property': 'color', 'value': 'dark bronze'}]}
    assert 'requested change (color dark bronze)' in ev.prompt('attribute', attr, 'x')


def test_schema_and_describe_fields():
    keys = [k for k, _, _ in ev.QUESTIONS['add']]
    assert ev.schema('add')['required'] == keys
    described = ev.schema('add', describe=True)
    assert described['required'][:2] == ['before', 'after'] and described['required'][2:] == keys
    assert ev.token_limit(True, 512) == ev.TOKEN_LIMIT + ev.DESCRIBE_LIMIT + 512


def test_judge_rejects_only_on_hard_defects_and_fails_closed():
    answers = {k: False for k, _, _ in ev.QUESTIONS['remove']}
    assert ev.judge('remove', answers)['pass']
    hard = ev.judge('remove', {**answers, 'left_unsupported': True, 'remnant': True})
    assert not hard['pass'] and hard['defects'] == ['left_unsupported', 'remnant']
    added = {k: False for k, _, _ in ev.QUESTIONS['add']}
    soft = ev.judge('add', {**added, 'pasted': True})
    assert soft['pass'] and soft['advisory'] == ['pasted'] and soft['defects'] == []
    assert ev.judge('remove', {**answers, 'remnant': 'yes'})['defects'] == ['invalid_verifier_response']
    assert ev.judge('remove', None)['pass'] is False
    observed = ev.judge('remove', {**answers, 'before': 'a cup', 'after': 'bare table'})
    assert observed['observed'] == {'before': 'a cup', 'after': 'bare table'}


def test_zoom_ignores_changes_of_other_edits():
    source, mask = _images()
    result = source.copy()
    result[185:215, 105:135] = 30          # this edit
    result[20:60, 500:560] = 250           # another target of the same multi image
    box = ev.zoom_box(source, result, mask)
    assert box[2] < 500 and box[1] > 60
    touching = result.copy()
    touching[200:260, 130:200] = 250       # e.g. a removed shadow next to the target
    wide = ev.zoom_box(source, touching, mask)
    assert wide[2] >= 200 and wide[3] >= 260


def test_views_layout_and_outline_only_on_before(tmp_path):
    source, mask = _images()
    result = source.copy()
    result[mask] = 30
    paths = ev.views(*_write(tmp_path, source, result, mask), tmp_path / 'v')
    pair = np.asarray(Image.open(paths[1]))
    meta = json.loads((tmp_path / 'v' / 'verifier_box.json').read_text())
    assert meta['layout'] in ('side', 'stacked') and pair.shape[1] <= 1536 and pair.shape[0] <= 1024
    yellow = (pair[..., 0] > 200) & (pair[..., 1] > 180) & (pair[..., 2] < 90)
    half = pair.shape[1] // 2 if meta['layout'] == 'side' else None
    if half:
        assert yellow[:, :half].sum() > 50 and yellow[:, half:].sum() == 0


class _Auditor:
    batch_size = 4

    def __init__(self, answers):
        self.answers, self.jobs = answers, []

    def call_batch(self, jobs):
        self.jobs.extend(jobs)
        return [{'job_id': j['job_id'], 'semantic_audit': self.answers, 'finish_reason': 'stop'} for j in jobs]


def test_apply_verifies_only_passed_members_and_records_failures(tmp_path):
    source, mask = _images()
    result = source.copy()
    result[mask] = 30
    src, res, mpath = _write(tmp_path, source, result, mask)
    members = [{'unit_id': u, 'operation': 'remove', 'mask': mpath, 'instruction': 'Remove the box.',
                'instruction_spec': {'operation': 'remove', 'ref': 'the box'}} for u in ('U1', 'U2')]
    audits = {'U1': {'decision': 'pass', 'failure_reasons': []},
              'U2': {'decision': 'fail', 'failure_reasons': ['natural_failed']}}
    answers = {k: k == 'target_remains' for k, _, _ in ev.QUESTIONS['remove']}
    auditor = _Auditor(answers)
    calls = ev.apply(members, audits, src, [res, res], auditor, tmp_path / 'audit', {}, prefix='g1')
    assert calls == 1 and [j['job_id'] for j in auditor.jobs] == ['g1::verify::U1']
    assert auditor.jobs[0]['enable_thinking'] is False and 'thinking_token_budget' not in auditor.jobs[0]
    assert audits['U1']['decision'] == 'fail' and audits['U1']['failure_reasons'] == ['verifier_target_remains']
    assert audits['U2']['failure_reasons'] == ['natural_failed'] and 'verifier' not in audits['U2']
    clean = {'U1': {'decision': 'pass', 'failure_reasons': []}}
    ev.apply(members[:1], clean, src, [res], _Auditor({k: False for k in answers}), tmp_path / 'a2',
             {'thinking': 512}, prefix='g2')
    assert clean['U1']['decision'] == 'pass' and clean['U1']['verifier']['pass']


def test_contract_records_settings():
    assert ev.contract({}) == {'version': ev.VERSION, 'describe': False, 'thinking': 0}
    assert ev.contract({'describe': True, 'thinking': 512})['thinking'] == 512


SPEC = {'version': '2.0', 'operation': 'remove', 'ref': 'the dark box', 'scope': 'dark box', 'payload': '',
        'target_part': '', 'remove_mode': 'whole', 'attributes': [], 'placement': {}, 'qualifier': ''}


class _PipelineAuditor:
    """Main audit passes every unit; verifier jobs get the given answers."""
    batch_size = 4

    def __init__(self, verifier_answers, verifier=None, image_ok=True):
        self.answers, self.verifier, self.jobs, self.calls, self.image_ok = verifier_answers, verifier, [], [], image_ok

    def call_batch(self, jobs):
        self.jobs.extend(jobs)
        self.calls.append([j['job_id'] for j in jobs])
        out = []
        for j in jobs:
            if '::verify::' in j['job_id']:
                out.append({'job_id': j['job_id'], 'semantic_audit': self.answers, 'finish_reason': 'stop'})
                continue
            ids = j['output_schema']['properties']['units']['items']['properties']['unit_id']['enum']
            units = [{'unit_id': u, 'source_state': 'a dark box on a gray wall', 'result_state': 'plain gray wall',
                      'image_ok': self.image_ok, 'scope_status': 'matched', 'observed_operation': 'remove',
                      'reference_unique': True, 'ref_issue': '', 'final_ref': ''} for u in ids]
            out.append({'job_id': j['job_id'], 'semantic_audit': {'units': units}, 'finish_reason': 'stop'})
        return out


def _member_files(tmp_path):
    source, mask = _images()
    source[mask] = 20
    result = source.copy()
    result[mask] = 120
    src, res, mpath = _write(tmp_path, source, result, mask)
    member = {'unit_id': 'U1', 'operation': 'remove', 'mask': mpath, 'support_mask': mpath,
              'instruction': 'Remove the dark box.', 'instruction_noref': 'Remove the selected region.',
              'instruction_spec': dict(SPEC)}
    return src, res, member


def test_multi_component_audit_applies_verifier(tmp_path):
    from sa1b_pipeline import multi_component_audit as mca
    src, res, member = _member_files(tmp_path)
    group = {'multi_id': 'multi-test', 'source_image': src, 'dependency_schedule': {}}
    clean = {k: False for k, _, _ in ev.QUESTIONS['remove']}
    report = mca.audit_component(group, [dict(member)], res, _PipelineAuditor(clean), tmp_path / 'plain', aligned=True)
    assert report['accepted'] and 'verifier' not in report['contract']
    auditor = _PipelineAuditor({**clean, 'target_remains': True}, verifier={'thinking': 512})
    report = mca.audit_component(group, [dict(member)], res, auditor, tmp_path / 'checked', aligned=True)
    assert report['contract']['verifier'] == ev.contract({'thinking': 512}) and report['verifier_calls'] == 1
    assert not report['accepted'] and 'verifier_target_remains' in report['audits']['U1']['failure_reasons']
    verify = [j for j in auditor.jobs if '::verify::' in j['job_id']][0]
    assert verify['enable_thinking'] and verify['thinking_token_budget'] == 512
    # Main audit and verifier share one round trip.
    assert auditor.calls == [['multi-test::component::U1', 'multi-test::component::U1::verify::U1']]
    # A member the main audit rejected keeps its own reasons; the verdict is recorded only.
    failed = mca.audit_component(group, [dict(member)], res, _PipelineAuditor(clean, verifier={}, image_ok=False),
                                 tmp_path / 'main_fail', aligned=True)
    row = failed['audits']['U1']
    assert row['decision'] == 'fail' and row['verifier']['pass'] and not any(
        r.startswith('verifier_') for r in row['failure_reasons'])
    # The ledger replays without another model call.
    again = mca.audit_component(group, [dict(member)], res, _PipelineAuditor(clean, verifier={'thinking': 512}),
                                tmp_path / 'checked', aligned=True)
    assert again['sha256'] == report['sha256']


def test_twin_compact_audit_applies_verifier(tmp_path, monkeypatch):
    from sa1b_pipeline import twin_compact_audit as tca
    from sa1b_pipeline.twin_contract import ADAPTED
    src, res, member = _member_files(tmp_path)

    def group():
        members = []
        for uid in ('U1', 'U2'):
            members.append({**member, 'unit_id': uid, 'edited_image': res, 'instruction_spec': dict(SPEC)})
        checks = {k: {'pass': True} for k in ('deterministic', 'cross_target_preservation', 'support_gap')}
        return {'twin_id': 'twins-test', 'source_image': src, 'operation': 'remove', 'members': members,
                'twin_contract_version': ADAPTED, 'group_checks': checks}

    clean = {k: False for k, _, _ in ev.QUESTIONS['remove']}
    settings = {'describe': False, 'thinking': 0}          # as run_dual_validation --audit-verifier nothink
    out = tca.audit([group()], _PipelineAuditor(clean), tmp_path / 'off')[0]
    assert out['group_checks']['group_pass']
    auditor = _PipelineAuditor(clean, verifier=settings)
    out = tca.audit([group()], auditor, tmp_path / 'a')[0]
    evidence = json.loads(Path(out['group_checks']['compact_audit_path']).read_text())
    assert out['group_checks']['group_pass'] and evidence['verifier_calls'] == 2
    assert auditor.calls == [['twins-test', 'twins-test::verify::U1', 'twins-test::verify::U2']]
    out = tca.audit([group()], _PipelineAuditor({**clean, 'left_unsupported': True}, verifier=settings), tmp_path / 'b')[0]
    assert out['group_checks']['group_pass'] is False
    assert any('verifier_left_unsupported' in m['semantic_audit'].get('failure_reasons', []) for m in out['members'])


def test_split_settings_skip_attribute_and_vary_thinking(tmp_path):
    src, res, member = _member_files(tmp_path)
    options = {'describe': False, 'thinking': 512, 'by_operation': dict(ev.SPLIT)}
    members = [{**member, 'unit_id': 'U1'},
               {**member, 'unit_id': 'U2', 'operation': 'attribute',
                'instruction_spec': {**SPEC, 'operation': 'attribute'}},
               {**member, 'unit_id': 'U3', 'operation': 'add', 'instruction_spec': {**SPEC, 'operation': 'add',
                                                                                    'payload': 'a red ball'}}]
    jobs = ev.member_jobs(members, src, [res] * 3, tmp_path / 'v', options, prefix='g')
    assert [j['job_id'] for j in jobs] == ['g::verify::U1', 'g::verify::U3']
    assert jobs[0]['enable_thinking'] and jobs[0]['thinking_token_budget'] == 512
    assert jobs[1]['enable_thinking'] is False and 'thinking_token_budget' not in jobs[1]
    assert ev.contract(options)['by_operation'] == {'add': 0, 'attribute': 'off', 'remove': 512, 'replace': 512}
    audits = {u: {'decision': 'pass', 'failure_reasons': []} for u in ('U1', 'U2', 'U3')}
    remove_answers = {k: k == 'target_remains' for k, _, _ in ev.QUESTIONS['remove']}
    add_answers = {k: False for k, _, _ in ev.QUESTIONS['add']}
    outputs = [{'job_id': 'g::verify::U3', 'semantic_audit': add_answers, 'finish_reason': 'stop'},
               {'job_id': 'g::verify::U1', 'semantic_audit': remove_answers, 'finish_reason': 'stop'}]
    ev.record(members, audits, outputs)                    # matched by job id, not by position
    assert audits['U1']['decision'] == 'fail' and audits['U1']['failure_reasons'] == ['verifier_target_remains']
    assert audits['U2']['decision'] == 'pass' and audits['U2']['verifier']['skipped']
    assert audits['U3']['decision'] == 'pass' and audits['U3']['verifier']['pass'] and 'skipped' not in audits['U3']['verifier']
    import pytest
    with pytest.raises(RuntimeError):
        ev.record(members, dict(audits), outputs + [{'job_id': 'g::verify::U9', 'semantic_audit': {}}])
