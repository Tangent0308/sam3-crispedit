import pytest

from synthesis_pipeline.merge_audit_splits import merge


def test_merge_restores_annotation_order():
    annotations = [{"image": "a.png"}, {"image": "b.png"}, {"image": "c.png"}]
    result = merge(
        annotations,
        [[{"image": "c.png", "quality": "pass"}, {"image": "a.png", "quality": "fail"}],
         [{"image": "b.png", "quality": "pass"}]],
    )
    assert [row["image"] for row in result] == ["a.png", "b.png", "c.png"]


def test_merge_rejects_duplicate_or_missing_audits():
    annotations = [{"image": "a.png"}, {"image": "b.png"}]
    with pytest.raises(ValueError, match="Duplicate audit"):
        merge(annotations, [[{"image": "a.png"}], [{"image": "a.png"}]])
    with pytest.raises(ValueError, match="mismatch"):
        merge(annotations, [[{"image": "a.png"}]])
