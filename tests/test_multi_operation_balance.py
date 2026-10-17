from collections import Counter

from sa1b_pipeline.multi_operation_balance import balance_multi_operations, operation_schedules
from scripts.run_group_pipeline import _validate_multi_selection_operations


def _rows():
    rows = []
    for index, n in enumerate((2, 3, 4, 5, 2, 3, 4, 5)):
        rows.append({
            "candidate": {"source_id": f"source-{index}"},
            "objects": [{"object_id": f"{index}-{j}"} for j in range(n)],
        })
    return rows


def test_schedules_follow_multi_target_operation_contract():
    objects = [{"object_id": str(i)} for i in range(5)]
    schedules = operation_schedules(objects, (100, 100), lambda obj, op, size: True)
    assert schedules
    assert all(len(schedule) == 5 for schedule in schedules)
    assert all(set(schedule) == {"add", "replace", "remove", "attribute"} for schedule in schedules)


def test_balancer_reaches_exact_quota_when_all_operations_are_feasible():
    rows = _rows()
    report = balance_multi_operations(
        rows,
        image_size_for=lambda row: (100, 100),
        allowed=lambda obj, op, size: True,
        seed=17,
    )
    assert report["max_abs_deviation"] == 0
    assert Counter(op for row in rows for op in row["operations"]) == report["target_counts"]
    for row in rows:
        n = len(row["objects"])
        assert len(row["operations"]) == n
        if n <= 4:
            assert len(set(row["operations"])) == n
        else:
            assert set(row["operations"]) == {"add", "replace", "remove", "attribute"}


def test_balancer_never_assigns_a_rejected_operation():
    rows = _rows()

    def allowed(obj, operation, size):
        # The first object in every group cannot be removed, but all groups
        # still have feasible schedules through the remaining operation types.
        return not (obj["object_id"].endswith("-0") and operation == "remove")

    balance_multi_operations(rows, image_size_for=lambda row: (100, 100), allowed=allowed)
    for row in rows:
        for obj, operation in zip(row["objects"], row["operations"]):
            assert allowed(obj, operation, (100, 100))


def test_pipeline_preflight_rejects_fixed_order_skew():
    rows = []
    for index in range(40):
        rows.append({
            "objects": [{"object_id": f"{index}-0"}, {"object_id": f"{index}-1"}],
            "operations": ["add", "replace"],
        })
    try:
        _validate_multi_selection_operations(rows)
    except RuntimeError as error:
        assert "quota skew" in str(error)
    else:
        raise AssertionError("fixed-order multi selection must fail quota preflight")



def test_equal_quota_schedule_prefers_removing_standalone_item():
    rows=[{'candidate':{'source_id':'scene'},'objects':[
        {'object_id':'1','category':'arch'}, {'object_id':'2','category':'bottle'},
        {'object_id':'3','category':'chair'}, {'object_id':'4','category':'shadow'}]}]
    report=balance_multi_operations(rows,image_size_for=lambda row:(100,100),
                                    allowed=lambda obj,op,size:True)
    pairs=dict(zip((x['category'] for x in rows[0]['objects']),rows[0]['operations']))
    assert report['max_abs_deviation']==0
    assert pairs['arch']!='remove' and pairs['shadow']!='add'
    assert report['preference_cost']==0


def test_preferences_do_not_forbid_valid_masks_or_worsen_quotas():
    from sa1b_pipeline.multi_operation_balance import _milp_assignment
    options=[ [('remove','add'),('attribute','replace')] ]
    targets=dict(add=1,remove=1,attribute=0,replace=0)
    assert _milp_assignment(options,targets,[[1000,0]])==[('remove','add')]
    # When a restrictive eligibility gate leaves one costly schedule,
    # preference must not discard the source or silently relabel it.
    rows=[{'objects':[{'category':'arch'},{'category':'shadow'}]}]
    report=balance_multi_operations(rows,image_size_for=lambda row:(100,100),
       allowed=lambda obj,op,size:op==('remove' if obj['category']=='arch' else 'add'))
    assert rows[0]['operations']==['remove','add'] and report['preference_cost']==6


def test_preference_uses_head_nouns_without_broad_substring_exclusions():
    from sa1b_pipeline.multi_semantics import operation_preference_cost,operation_exclusion
    assert operation_preference_cost('stone arch','remove')==3
    assert operation_preference_cost('archery target','remove')==0
    assert operation_preference_cost('stone arches','remove')==3
    assert not operation_exclusion('arch','remove')


def test_greedy_fallback_preserves_original_group_operation_multisets(monkeypatch):
    import sa1b_pipeline.multi_operation_balance as balance
    rows=[{'candidate':{'source_id':str(i)},'objects':[
        {'object_id':str(j),'category':category} for j,category in enumerate(('arch','bottle','chair','shadow'))]}
        for i in range(4)]
    allowed=lambda obj,op,size:True
    options=[balance.operation_schedules(row['objects'],(100,100),allowed) for row in rows]
    original=balance._greedy_assignment(rows,options,balance._target_counts(16),0)
    monkeypatch.setattr(balance,'_milp_assignment',lambda *args,**kwargs:None)
    report=balance.balance_multi_operations(rows,image_size_for=lambda row:(100,100),allowed=allowed)
    for row,schedule in zip(rows,original):assert Counter(row['operations'])==Counter(schedule)
    assert report['preference_cost']==0
