import json
import sys
import threading
from collections import Counter
from types import SimpleNamespace

import jsonschema
import pytest

from sa1b_pipeline.operation_balance import OperationBalance, automatic_pass
from sa1b_pipeline.planning_candidates import assigned_operation_prompt
from sa1b_pipeline.planning_schema import typed_compact_schema
from sa1b_pipeline.shared_operation_quota import OPERATIONS
from sa1b_pipeline.work_queue import WorkQueue


def cases(n=8):
    return [{'id': f'{op}-{i}', 'source': {'source_id': f'{op}-source'},
             'units': [{'unit_id': 'U1', 'object_id': str(i), 'operation': op}]}
            for op in OPERATIONS for i in range(n)]


def outcome(job, passed=True):
    return {'job_id': job['job_id'], 'operation': job['case']['units'][0]['operation'],
            'decision': 'pass' if passed else 'fail',
            'semantic_audit': {'decision': 'pass' if passed else 'fail',
                               'operation_correct': True,
                               'target_done': passed, 'non_target_preserved': True, 'natural': True}}


def test_final_quota_replenishes_failed_type_only_without_overshoot(tmp_path):
    gen = WorkQueue(tmp_path/'gen.db')
    audit = WorkQueue(tmp_path/'audit.db')
    scheduler = OperationBalance(cases(), 3, 6)
    seen = Counter()
    while not scheduler.complete and not scheduler.exhausted:
        scheduler.dispatch(gen)
        for job, lease in reversed(gen.claim(6)):
            op = job['case']['units'][0]['operation']
            seen[op] += 1
            audit.enqueue(job)
            gen.finish(job['job_id'], lease, {'job_id': job['job_id']})
        for job, lease in reversed(audit.claim(6)):
            # Simulate systematically lower remove yield, including out-of-order completions.
            good = job['case']['units'][0]['operation'] != 'remove' or seen['remove'] > 3
            audit.finish(job['job_id'], lease, outcome(job, good))
        scheduler.observe(audit.completed_for(scheduler.pending))
    assert scheduler.complete
    assert Counter(r['operation'] for r in scheduler.balanced_rows()) == dict.fromkeys(OPERATIONS, 3)
    assert all(scheduler.attempts[op] == 3 for op in ('add', 'replace', 'attribute'))
    assert scheduler.attempts['remove'] > 3
    assert len(scheduler.outcomes) == sum(scheduler.attempts.values())
    assert gen.db.execute('SELECT MAX(attempts) FROM jobs').fetchone()[0] == 1
    gen.close()
    audit.close()


def test_insufficient_supply_is_explicit_and_preserves_extra_passes(tmp_path):
    q = WorkQueue(tmp_path/'q.db')
    scheduler = OperationBalance(cases(2), 2, 8)
    scheduler.dispatch(q)
    for job, lease in q.claim(8):
        row = outcome(job, job['case']['units'][0]['operation'] != 'remove')
        q.finish(job['job_id'], lease, row)
    scheduler.observe(q.results())
    scheduler.observe(q.results())  # Re-reading terminal rows is idempotent.
    assert scheduler.exhausted and not scheduler.complete
    assert scheduler.report()['shortfall'] == {'add': 0, 'replace': 0, 'remove': 2, 'attribute': 0}
    assert scheduler.report()['status'] == 'insufficient_supply'
    assert scheduler.balanced_rows() == []
    assert sum(map(len, scheduler.passed.values())) == 6
    with pytest.raises(ValueError, match='changed'):
        scheduler.observe([{**q.results()[0], 'decision': 'changed'}])
    q.close()


def test_balance_never_relabels_or_accepts_incomplete_audit(tmp_path):
    q = WorkQueue(tmp_path/'q.db')
    scheduler = OperationBalance(cases(1), 1, 1)
    scheduler.dispatch(q)
    job, _ = q.claim(1)[0]
    row = outcome(job)
    with pytest.raises(ValueError, match='ID/type'):
        scheduler.observe([{**row, 'operation': 'attribute'}])
    assert not automatic_pass({**row, 'semantic_audit': {'decision': 'pass'}})
    assert not automatic_pass({**row, 'benchmark_only': True})
    assert not automatic_pass({**row, 'decision': 'fail'})
    assert not automatic_pass({**row, 'semantic_audit': {**row['semantic_audit'], 'operation_correct': False}},
                              require_operation=True)
    with pytest.raises(ValueError, match='duplicate'):
        OperationBalance(cases(1)+cases(1), 1, 4)
    q.close()


def test_type_mismatch_does_not_fill_quota_even_when_other_audit_fields_pass(tmp_path):
    from sa1b_pipeline.validate import OPERATION_AUDIT_SCHEMA, audit_prompt
    q = WorkQueue(tmp_path/'q.db')
    scheduler = OperationBalance(cases(1), 1, 4)
    scheduler.dispatch(q)
    for job, lease in q.claim(4):
        row = outcome(job)
        if row['operation'] == 'replace':
            row['semantic_audit']['operation_correct'] = False
        q.finish(job['job_id'], lease, row)
    scheduler.observe(q.results())
    assert scheduler.report()['shortfall']['replace'] == 1
    assert not scheduler.complete
    assert 'Expected operation: replace' in audit_prompt('Replace water with calm water.', operation='replace')
    assert '"operation_correct":bool' in audit_prompt('Replace water with calm water.', operation='replace').splitlines()[1]
    assert 'operation_correct' not in audit_prompt('Replace water with calm water.')
    audit = {**outcome({'job_id': 'k', 'case': cases(1)[0]})['semantic_audit'], 'failure_reasons': [], 'evidence': ''}
    jsonschema.validate(audit, OPERATION_AUDIT_SCHEMA)
    del audit['operation_correct']
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(audit, OPERATION_AUDIT_SCHEMA)
    q.close()


@pytest.mark.parametrize('op', OPERATIONS)
def test_assigned_schema_and_prompt_allow_only_requested_type_or_no_edit(op):
    schema = typed_compact_schema([op], single_phrase=True)
    assert {b['properties']['op']['const'] for b in schema['anyOf']} == {op, 'no_edit'}
    no_edit = {'kind': 'anatomy', 'full_ref': 'the left forearm', 'rival': None,
               'scope': 'forearm', 'source_color': None, 'op': 'no_edit',
               'change': {'reason': 'The assigned operation is unnatural here.'},
               'skipped': {op: 'No natural in-scope edit of this type.'}}
    jsonschema.validate(no_edit, schema)
    obj = SimpleNamespace(category='forearm', description='forearm', metadata={})
    prompt = assigned_operation_prompt(obj, [op])
    assert 'no type fallback is permitted' in prompt
    assert 'fall back in this same response' not in prompt
    assert 'Removing an attached arm is infeasible' in prompt
