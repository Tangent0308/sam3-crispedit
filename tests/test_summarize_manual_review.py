import json
import sys

from synthesis_pipeline.summarize_manual_review import main


def test_new_two_criterion_review_schema(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations.jsonl"
    reviews = tmp_path / "manual.jsonl"
    summary = tmp_path / "summary.json"
    annotations.write_text(
        json.dumps({"image": "a.png", "task_type": "add", "source_subset": "gres"}) + "\n"
        + json.dumps({"image": "b.png", "task_type": "remove", "source_subset": "ver"}) + "\n"
    )
    reviews.write_text(
        json.dumps({
            "image": "a.png", "quality": "pass", "visual_quality": "pass",
            "instruction_match": "pass", "reason": "Good.",
        }) + "\n"
        + json.dumps({
            "image": "b.png", "quality": "fail", "visual_quality": "pass",
            "instruction_match": "fail", "reason": "Wrong target.",
        }) + "\n"
    )
    monkeypatch.setattr(sys, "argv", [
        "summarize_manual_review", "--annotations-jsonl", str(annotations),
        "--manual-review-jsonl", str(reviews), "--output-json", str(summary),
    ])
    main()
    result = json.loads(summary.read_text())
    assert result["overall"]["quality"] == {"pass": 1, "fail": 1}
    assert result["overall"]["visual_quality"] == {"pass": 2}
    assert result["by_task_type"]["remove"]["instruction_match"] == {"fail": 1}
