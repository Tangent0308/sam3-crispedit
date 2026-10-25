"""Aggregate explicit visual review; never infer semantic grades from parsing.

Each target needs a verdict and reason for all four dimensions. Missing model
output is not_assessed, rather than four fabricated semantic failures. Report
both assessed denominators and the yield over all requested targets.
"""
from collections import Counter

DIMENSIONS = ('scope', 'reference', 'naturalness', 'expression')
VERDICTS = {'pass', 'fail', 'uncertain', 'not_assessed'}


def summarize(rows):
    seen = set()
    counts = {key: Counter() for key in DIMENSIONS}
    basic = 0
    for row in rows:
        identity = (row['group_id'], row['unit_id'])
        if identity in seen:
            raise ValueError('duplicate reviewed target')
        seen.add(identity)
        grades = row['dimensions']
        if set(grades) != set(DIMENSIONS):
            raise ValueError('all four review dimensions are required')
        for key, grade in grades.items():
            if grade.get('verdict') not in VERDICTS or not grade.get('reason', '').strip():
                raise ValueError('every review dimension needs a verdict and reason')
            counts[key][grade['verdict']] += 1
        basic += all(grade['verdict'] == 'pass' for grade in grades.values())
    return {'requested_targets': len(rows), 'basic_usable_targets': basic,
            'dimensions': {key: {'counts': dict(count),
                                 'assessed_targets': sum(n for v, n in count.items() if v != 'not_assessed'),
                                 'pass_targets': count['pass']}
                           for key, count in counts.items()}}
