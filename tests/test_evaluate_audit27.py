from synthesis_pipeline.evaluate_audit27 import confusion


def test_confusion_separates_false_accept_from_false_reject():
    manual = {
        "a": {"quality": "pass"},
        "b": {"quality": "fail"},
        "c": {"quality": "fail"},
        "d": {"quality": "pass"},
    }
    audit = {
        "a": {"quality": "pass"},
        "b": {"quality": "pass"},
        "c": {"quality": "fail"},
        "d": {"quality": "fail"},
    }
    result = confusion(manual, audit, "quality")
    assert result["true_pass"] == 1
    assert result["true_fail"] == 1
    assert result["false_accept"] == 1
    assert result["false_reject"] == 1
    assert result["accuracy"] == 0.5
    assert result["false_accept_rate_among_bad_pairs"] == 0.5
