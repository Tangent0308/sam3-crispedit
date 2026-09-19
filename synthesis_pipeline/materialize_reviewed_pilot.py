"""Materialize human-accepted original edits and explicitly approved rewrites."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from synthesis_pipeline.audit_edit_pairs import load_jsonl, write_jsonl


def by_image(path: Path) -> dict[str, dict]:
    return {row["image"]: row for row in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--manual-jsonl", type=Path, required=True)
    parser.add_argument("--audit8-jsonl", type=Path, required=True)
    parser.add_argument("--audit27-jsonl", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    annotations = load_jsonl(args.annotations_jsonl)
    manual = by_image(args.manual_jsonl)
    audits = {"8b": by_image(args.audit8_jsonl), "27b": by_image(args.audit27_jsonl)}
    names = {row["image"] for row in annotations}
    if any(set(rows) != names for rows in (manual, *audits.values())):
        raise ValueError("All annotation, human, and audit case sets must match")

    accepted = []
    rewrites = []
    counts = Counter()
    for original in annotations:
        name = original["image"]
        review = manual[name]
        if review.get("quality") not in {"pass", "fail"}:
            raise ValueError(f"Unreviewed case: {name}")
        approved_models = [
            model for model in audits
            if review.get(f"rewrite_{model}_valid") is True
        ]
        if len(approved_models) > 1:
            raise ValueError(f"Multiple approved model rewrites require arbitration: {name}")
        if approved_models:
            model = approved_models[0]
            audit = audits[model][name]
            candidate = audit.get("rewrite_candidate")
            evidence = audit.get("audit") or {}
            if (
                not candidate
                or review["quality"] != "fail"
                or review.get("visual_quality") != "pass"
                or review.get("instruction_match") != "fail"
                or audit.get("quality") != "fail"
                or evidence.get("visual_quality") != "pass"
                or evidence.get("instruction_match") != "fail"
                or not evidence.get("target_match")
            ):
                raise ValueError(f"Rewrite approval and visual evidence conflict: {name}")
            row = dict(original)
            row["original_editing_instruction"] = row["editing_instruction"]
            row["editing_instruction"] = candidate
            row["reviewed_instruction_source"] = f"{model}_human_approved_rewrite"
            row["human_review_reason"] = review.get("reason")
            rewrites.append(row)
            accepted.append(row)
            counts["approved_model_rewrite"] += 1
        elif review["quality"] == "pass":
            if (
                review.get("visual_quality") != "pass"
                or review.get("instruction_match") != "pass"
            ):
                raise ValueError(f"Human pass dimensions conflict: {name}")
            row = dict(original)
            clarified = review.get("clarified_instruction")
            if clarified:
                row["original_editing_instruction"] = row["editing_instruction"]
                row["editing_instruction"] = clarified
                row["reviewed_instruction_source"] = "human_clarity_rewrite"
                counts["human_clarity_rewrite"] += 1
            else:
                row["reviewed_instruction_source"] = "original_human_pass"
                counts["original_human_pass"] += 1
            row["human_review_reason"] = review.get("reason")
            accepted.append(row)
        else:
            counts["rejected"] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "accepted_annotations.jsonl", accepted)
    write_jsonl(args.out_dir / "approved_model_rewrites.jsonl", rewrites)
    summary = {
        "reviewed_cases": len(annotations),
        "accepted_cases": len(accepted),
        "approved_model_rewrites": len(rewrites),
        "counts": dict(counts),
        "scope": "pilot_human_review_only_not_production_admission",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
