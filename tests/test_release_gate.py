import json

from sa1b_pipeline.release import release_rows


def _row(case, unit, *, decision="pending_semantic_audit", semantic=True):
    row = {"case_id": case, "unit_id": unit, "decision": decision}
    if semantic:
        row["semantic_audit"] = {
            "decision": "pass",
            "target_done": True,
            "non_target_preserved": True,
            "natural": True,
        }
    return row


def test_release_requires_all_gates_and_preserves_multi_target_cases():
    rows = [_row("a", f"U{i}") for i in range(1, 4)]
    rows += [_row("b", "U1"), _row("b", "U2", decision="fail")]
    review = {("a", f"U{i}"): {"status": "manual_pass"} for i in range(1, 4)}
    review[("b", "U1")] = {"status": "manual_pass"}
    review[("b", "U2")] = {"status": "manual_pass"}
    released, report = release_rows(rows, review, min_targets=3, require_semantic=True)
    assert {(r["case_id"], r["unit_id"]) for r in released} == {
        ("a", "U1"), ("a", "U2"), ("a", "U3")
    }
    assert report["released_cases"] == 1
    assert report["dropped_cases"] == {"b": "fewer_than_min_targets_after_row_gate"}


def test_visual_pilot_requires_explicit_override_for_missing_semantic_audit():
    rows = [_row("a", f"U{i}", semantic=False) for i in range(1, 4)]
    review = {("a", f"U{i}"): {"status": "manual_pass"} for i in range(1, 4)}
    released, _ = release_rows(rows, review, require_semantic=False)
    assert len(released) == 3
    released, report = release_rows(rows, review, require_semantic=True)
    assert not released
    assert report["rejected_rows"] == 3
