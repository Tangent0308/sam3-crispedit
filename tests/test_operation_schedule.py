from collections import Counter
import pytest
from sa1b_pipeline.operation_schedule import fixed_balanced_schedule


def test_input_schedule_balanced_unique_order_invariant():
    keys=[(f'source{i%6}',str(i)) for i in range(207)]
    a=fixed_balanced_schedule(keys)
    assert a==fixed_balanced_schedule(reversed(keys))
    assert set(a)==set(keys)
    counts=Counter(v[0] for v in a.values())
    assert max(counts.values())-min(counts.values())<=1
    assert all(set(v)=={'add','replace','remove','attribute'} and len(v)==4 for v in a.values())
    for source in {k[0] for k in keys}:
        assert len({v[0] for k,v in a.items() if k[0]==source})==4
    assert a!=fixed_balanced_schedule(keys,seed=19)


def test_input_schedule_rejects_duplicate_identities_and_supports_empty_input():
    assert fixed_balanced_schedule([])=={}
    with pytest.raises(ValueError,match='duplicate'):
        fixed_balanced_schedule([('a','1'),('a',1)])
