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

