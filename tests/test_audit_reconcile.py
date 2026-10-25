import json

import numpy as np
from PIL import Image

from sa1b_pipeline import audit_reconcile as ar

SPEC = {'version': '2.0', 'operation': 'remove', 'ref': 'the small white cloud in the upper left of the sky',
        'scope': 'small white cloud', 'payload': '', 'target_part': '', 'remove_mode': 'whole', 'attributes': [],
        'placement': {}, 'qualifier': ''}


def _files(tmp_path):
    source = np.full((400, 600, 3), 150, np.uint8)
    mask = np.zeros((400, 600), bool)
    mask[40:80, 60:120] = True                     # 0-1000 box: [100, 100, 200, 200]
    result = source.copy()
    paths = [tmp_path / 'source.png', tmp_path / 'result.png', tmp_path / 'mask.png']
    Image.fromarray(source).save(paths[0]); Image.fromarray(result).save(paths[1])
    Image.fromarray(mask.astype(np.uint8) * 255).save(paths[2])
    return [str(p) for p in paths]


def _group(src, mpath, reader_status='pass', ref=SPEC['ref']):
    gates = {'verdicts': [{'gate': 'reader', 'status': reader_status, 'pass': reader_status == 'pass'}]}
    member = {'unit_id': 'U1', 'object_id': '7', 'operation': 'remove', 'mask': mpath, 'support_mask': mpath,
              'instruction': 'Remove the small white cloud in the upper left of the sky.',
              'instruction_noref': 'Remove the selected region.', 'instruction_spec': dict(SPEC)}
    group = {'twin_id': 'twins-x', 'source_image': src,
             'planning': {'raw': {'members': [{'unit_id': 'U3', 'object_id': '7', 'ref_phrase': ref,
                                               'evidence_gates': gates}]}}}
    return group, member


def _row(**kw):
    row = {'unit_id': 'U1', 'source_state': 'a small cloud', 'result_state': 'clear sky', 'image_ok': True,
           'scope_status': 'matched', 'observed_operation': 'remove', 'reference_unique': False,
           'ref_issue': 'Two small clouds sit in the upper left.', 'final_ref': ''}
    row.update(kw)
    return {'units': [row]}


class _Reader:
    batch_size = 4

    def __init__(self, boxes):
        self.boxes, self.jobs = boxes, []

    def call_batch(self, jobs):
        self.jobs.extend(jobs)
        return [{'job_id': j['job_id'], 'semantic_audit': {'boxes': self.boxes}, 'finish_reason': 'stop'} for j in jobs]


def test_planning_reader_lookup_matches_object_and_exact_ref(tmp_path):
    src, _, mpath = _files(tmp_path)
    group, member = _group(src, mpath)
    assert ar.planning_reader(group, member) == 'pass'          # unit IDs differ after twin over-selection
    assert ar.planning_reader(group, {**member, 'object_id': '8'}) is None
    changed = {**member, 'instruction_spec': {**SPEC, 'ref': 'the leftmost small white cloud'}}
    assert ar.planning_reader(group, changed) is None             # a different phrase was never read


def test_repair_adopted_only_when_reader_finds_one_box_on_target(tmp_path):
    src, _, mpath = _files(tmp_path)
    group, member = _group(src, mpath)
    raw = _row(final_ref='the leftmost small white cloud in the upper left of the sky')
    reader = _Reader([[100, 100, 200, 200]])
    out, records, calls = ar.reconcile(raw, [member], group, reader, prefix='g')
    assert calls == 1 and reader.jobs[0]['enable_thinking'] is False and reader.jobs[0]['images'] == [src]
    assert 'leftmost small white cloud' in reader.jobs[0]['prompt']
    assert out['units'][0]['reference_unique'] is True and out['units'][0]['final_ref'].startswith('the leftmost')
    assert records[0]['outcome'] == 'repair_verified' and raw['units'][0]['reference_unique'] is False  # input untouched
    # Reader finds two instances: the repair is not adopted; the reader-verified draft is kept.
    out, records, _ = ar.reconcile(raw, [member], group, _Reader([[100, 100, 200, 200], [300, 100, 400, 200]]), prefix='g')
    assert records[0]['outcome'] == 'kept_reader_verified_draft'
    assert out['units'][0]['final_ref'] == '' and out['units'][0]['ref_issue'] == '' and out['units'][0]['reference_unique']


def test_doubt_without_repair_needs_planning_reader_evidence(tmp_path):
    src, _, mpath = _files(tmp_path)
    group, member = _group(src, mpath)
    out, records, calls = ar.reconcile(_row(final_ref=SPEC['ref']), [member], group, _Reader([]), prefix='g')
    assert calls == 0 and records[0]['outcome'] == 'kept_reader_verified_draft' and out['units'][0]['reference_unique']
    ambiguous, _ = _group(src, mpath, reader_status='ambiguous')
    out, records, _ = ar.reconcile(_row(), [member], ambiguous, _Reader([]), prefix='g')
    assert records[0]['outcome'] == 'audit_verdict_kept' and out['units'][0]['reference_unique'] is False
    # Rows without a uniqueness doubt are not touched and cost nothing.
    out, records, calls = ar.reconcile(_row(reference_unique=True, ref_issue='', final_ref=''), [member], group,
                                       _Reader([]), prefix='g')
    assert records == [] and calls == 0


def test_reconciled_doubt_passes_the_normalized_audit(tmp_path):
    from sa1b_pipeline.multi_component_audit import normalize_result
    src, _, mpath = _files(tmp_path)
    group, member = _group(src, mpath)
    raw = _row(final_ref='the leftmost small white cloud in the upper left of the sky')
    before = normalize_result(json.loads(json.dumps(raw)), [dict(member)])['U1']
    assert before['decision'] == 'fail' and 'reference_not_unique' in before['failure_reasons']
    out, _, _ = ar.reconcile(raw, [member], group, _Reader([[100, 100, 200, 200]]), prefix='g')
    revised = dict(member)
    after = normalize_result(out, [revised])['U1']
    assert after['decision'] == 'pass'
    assert revised['instruction_spec']['ref'] == 'the leftmost small white cloud in the upper left of the sky'
    assert revised['instruction'] == 'Remove the leftmost small white cloud in the upper left of the sky.'


def test_scope_doubt_overruled_only_with_passing_image_and_reader(tmp_path):
    from sa1b_pipeline.multi_component_audit import normalize_result
    src, _, mpath = _files(tmp_path)
    group, member = _group(src, mpath)
    raw = _row(reference_unique=True, ref_issue='', final_ref='', scope_status='wrong_entity')
    before = normalize_result(json.loads(json.dumps(raw)), [dict(member)])['U1']
    assert before['decision'] == 'fail'
    out, records, calls = ar.reconcile(raw, [member], group, _Reader([]), prefix='g')
    assert calls == 0 and records[0]['kind'] == 'scope' and out['units'][0]['scope_status'] == 'matched'
    assert normalize_result(out, [dict(member)])['U1']['decision'] == 'pass'
    # The image failed, or another operation was observed, or no reader evidence: verdict stands.
    for kw in ({'image_ok': False}, {'observed_operation': 'replace'}):
        out, records, _ = ar.reconcile(_row(reference_unique=True, ref_issue='', final_ref='', scope_status='wrong_entity', **kw),
                                       [member], group, _Reader([]), prefix='g')
        assert records == [] and out['units'][0]['scope_status'] == 'wrong_entity'
    ambiguous, _ = _group(src, mpath, reader_status='ambiguous')
    out, records, _ = ar.reconcile(raw, [member], ambiguous, _Reader([]), prefix='g')
    assert records == [] and out['units'][0]['scope_status'] == 'wrong_entity'


def test_repair_the_revision_would_reject_keeps_the_verified_draft(tmp_path):
    from sa1b_pipeline.multi_component_audit import normalize_result
    src, _, mpath = _files(tmp_path)
    group, member = _group(src, mpath)
    raw = _row(final_ref='the small white cloud near the No. 9 tower')      # periods are not allowed in a Ref
    before = normalize_result(json.loads(json.dumps({**raw, 'units': [{**raw['units'][0], 'reference_unique': True}]})),
                              [dict(member)])['U1']
    assert before['decision'] == 'fail' and 'invalid_instruction_revision' in before['failure_reasons']
    reader = _Reader([[100, 100, 200, 200]])
    out, records, calls = ar.reconcile(raw, [member], group, reader, prefix='g')
    assert calls == 0 and reader.jobs == []
    assert records[0]['outcome'] == 'kept_reader_verified_draft' and records[0]['repair_invalid']
    revised = dict(member)
    assert normalize_result(out, [revised])['U1']['decision'] == 'pass'
    assert revised['instruction_spec']['ref'] == SPEC['ref']
    # Without planning reader evidence the audit verdict stands.
    ambiguous, _ = _group(src, mpath, reader_status='ambiguous')
    out, records, _ = ar.reconcile(raw, [member], ambiguous, _Reader([]), prefix='g')
    assert records[0]['outcome'] == 'audit_verdict_kept' and out['units'][0]['reference_unique'] is False
