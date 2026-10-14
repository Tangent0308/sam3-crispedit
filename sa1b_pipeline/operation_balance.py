"""Fill final automatic-pass quotas without changing labels or audit decisions.

Only one supervisor owns this scheduler. Workers use the existing durable queues.
Pending generation AND audit reserve a slot; failures free that exact type's slot.
All unused plans and all audit outcomes are retained for inspection.
"""
from __future__ import annotations

import hashlib
from collections import Counter, deque

from .shared_operation_quota import OPERATIONS


def automatic_pass(row, *, require_operation=False):
    audit = row.get('semantic_audit') or {}
    return (not row.get('benchmark_only') and row.get('decision') == 'pass'
            and audit.get('decision') == 'pass'
            and (not (require_operation or row.get('operation_audit_required'))
                 or audit.get('operation_correct') is True)
            and all(audit.get(k) is True for k in ('target_done', 'non_target_preserved', 'natural')))


class OperationBalance:
    def __init__(self, cases, target_per_type, max_inflight):
        targets = (dict(target_per_type) if isinstance(target_per_type, dict)
                   else {op: target_per_type for op in OPERATIONS})
        if (set(targets) != set(OPERATIONS) or any(type(n) is not int or n < 0 for n in targets.values())
                or not any(targets.values()) or max_inflight < 1):
            raise ValueError('positive target and inflight limit required')
        self.target = target_per_type
        self.targets = targets
        self.max_inflight = max_inflight
        self.pool = {op: deque() for op in OPERATIONS}
        keys, masks = set(), set()
        for case in cases:
            if len(case['units']) != 1 or case.get('benchmark_only'):
                raise ValueError('balance requires unique real single-unit jobs')
            unit = case['units'][0]
            op = unit['operation']
            if op not in self.pool:
                raise ValueError('unknown operation')
            key = case['id']+'/'+unit['unit_id']
            mask = (str(case['source']['source_id']), str(unit['object_id']))
            if key in keys or mask in masks:
                raise ValueError('duplicate case/unit or source/mask')
            keys.add(key)
            masks.add(mask)
            self.pool[op].append({'job_id': key, 'case': case})
        # Frozen order, independent of worker completion or model confidence.
        for op in OPERATIONS:
            self.pool[op] = deque(sorted(self.pool[op], key=lambda j: hashlib.sha256(j['job_id'].encode()).digest()))
        self.initial = {op: len(self.pool[op]) for op in OPERATIONS}
        self.pending = {}
        self.outcomes = {}
        self.passed = {op: [] for op in OPERATIONS}
        self.attempts = Counter()

    def dispatch(self, queue):
        reservations = Counter(self.pending.values())
        added = []
        while len(self.pending) < self.max_inflight:
            available = [op for op in OPERATIONS if self.pool[op]
                         and len(self.passed[op])+reservations[op] < self.targets[op]]
            if not available:
                break
            op = min(available, key=lambda o: (len(self.passed[o])+reservations[o], OPERATIONS.index(o)))
            job = self.pool[op][0]
            queue.enqueue(job)  # Commit before updating in-memory reservations.
            self.pool[op].popleft()
            self.pending[job['job_id']] = op
            reservations[op] += 1
            self.attempts[op] += 1
            added.append(job['job_id'])
        return added

    def observe(self, rows):
        for row in rows:
            key = row['job_id']
            if key in self.outcomes:
                if row != self.outcomes[key]:
                    raise ValueError('completed audit outcome changed')
                continue
            if key not in self.pending or row['operation'] != self.pending[key]:
                raise ValueError('audit ID/type does not match dispatched plan')
            op = self.pending.pop(key)
            self.outcomes[key] = row
            if automatic_pass(row, require_operation=True):
                self.passed[op].append(row)

    @property
    def complete(self):
        return all(len(self.passed[op]) == self.targets[op] for op in OPERATIONS)

    @property
    def exhausted(self):
        return not self.pending and all(len(self.passed[op]) == self.targets[op] or not self.pool[op]
                                        for op in OPERATIONS)

    def report(self):
        accepted = {op: len(self.passed[op]) for op in OPERATIONS}
        return {'status': 'complete' if self.complete else 'insufficient_supply' if self.exhausted else 'running',
                'counting_unit': 'one independently edited source mask',
                'gate': 'deterministic, pipeline semantic and operation-type audit; manual release remains separate',
                'target_per_type': self.target, 'requested_total': sum(self.targets.values()),
                'planned_supply': self.initial, 'generated_or_dispatched': dict(self.attempts),
                'automatically_passed': accepted,
                'pending': dict(Counter(self.pending.values())),
                'shortfall': {op: self.targets[op]-accepted[op] for op in OPERATIONS},
                'unused_plans': {op: len(self.pool[op]) for op in OPERATIONS},
                'balanced_partial_per_type': min(accepted.values())}

    def balanced_rows(self):
        count = min(len(v) for v in self.passed.values())
        return [r for op in OPERATIONS for r in sorted(self.passed[op], key=lambda r: r['job_id'])[:count]]

    def unused_cases(self):
        return [job['case'] for op in OPERATIONS for job in self.pool[op]]

    def restore(self, dispatched, outcomes):
        """Rebuild reservations from immutable shared dispatch/result journals."""
        jobs = {job['job_id']: job for queue in self.pool.values() for job in queue}
        seen = set()
        for job in dispatched:
            key = job['job_id']
            if key in seen or jobs.get(key) != job:
                raise ValueError('duplicate or changed dispatched job on resume')
            seen.add(key)
            op = job['case']['units'][0]['operation']
            self.pending[key] = op
            self.attempts[op] += 1
        for op in OPERATIONS:
            self.pool[op] = deque(job for job in self.pool[op] if job['job_id'] not in seen)
        self.observe(outcomes)
        reserved = Counter(self.pending.values())
        if any(len(self.passed[op]) + reserved[op] > self.targets[op] for op in OPERATIONS):
            raise ValueError('resumed reservations exceed quota')
