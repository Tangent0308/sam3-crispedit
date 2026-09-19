"""Validate per-case manual reviews and write aggregate quality statistics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


QUALITIES = {"pass", "review", "fail"}
BOOLEAN_FIELDS = (
    "edit_visible",
    "correct_region",
    "non_target_preserved",
    "artifact_free",
    "difficult_localization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--manual-review-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rates(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    return {
        "cases": total,
        "counts": {name: counts.get(name, 0) for name in sorted(QUALITIES)},
        "pass_rate": round(counts.get("pass", 0) / total, 4) if total else 0.0,
        "usable_rate": (
            round((counts.get("pass", 0) + counts.get("review", 0)) / total, 4)
            if total
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations_jsonl)
    reviews = load_jsonl(args.manual_review_jsonl)
    annotation_by_image = {str(row["image"]): row for row in annotations}
    new_schema = bool(reviews) and all(
        "visual_quality" in row and "instruction_match" in row for row in reviews
    )
    review_by_image: dict[str, dict[str, Any]] = {}
    for row in reviews:
        image = str(row.get("image", ""))
        if not image or image in review_by_image:
            raise ValueError(f"Missing or duplicate review image: {image!r}")
        if image not in annotation_by_image:
            raise ValueError(f"Review is not present in annotations: {image}")
        if row.get("quality") not in ({"pass", "fail"} if new_schema else QUALITIES):
            raise ValueError(f"Invalid quality for {image}: {row.get('quality')!r}")
        if new_schema:
            for field in ("visual_quality", "instruction_match"):
                if row.get(field) not in {"pass", "fail"}:
                    raise ValueError(f"{image}: {field} must be pass or fail")
            if row["quality"] == "pass" and (
                row["visual_quality"] != "pass" or row["instruction_match"] != "pass"
            ):
                raise ValueError(f"{image}: overall pass requires both criteria to pass")
        else:
            for field in BOOLEAN_FIELDS:
                if not isinstance(row.get(field), bool):
                    raise ValueError(f"{image}: {field} must be boolean")
        if not str(row.get("reason", "")).strip():
            raise ValueError(f"{image}: reason is required")
        review_by_image[image] = row

    missing = [name for name in annotation_by_image if name not in review_by_image]
    if missing:
        raise ValueError(f"Missing {len(missing)} manual reviews: {missing[:10]}")
    if len(reviews) != len(annotations):
        raise ValueError(
            f"Expected {len(annotations)} reviews, found {len(reviews)}"
        )

    if new_schema:
        fields = ("quality", "visual_quality", "instruction_match")
        summary = {
            "cases": len(reviews),
            "review_coverage": f"{len(reviews)}/{len(annotations)}",
            "overall": {
                field: dict(Counter(row[field] for row in reviews)) for field in fields
            },
            "by_task_type": {
                name: {
                    field: dict(Counter(review_by_image[image][field] for image, annotation in annotation_by_image.items() if annotation.get("task_type") == name))
                    for field in fields
                }
                for name in sorted({str(row.get("task_type")) for row in annotations})
            },
            "by_source_subset": {
                name: {
                    field: dict(Counter(review_by_image[image][field] for image, annotation in annotation_by_image.items() if annotation.get("source_subset") == name))
                    for field in fields
                }
                for name in sorted({str(row.get("source_subset")) for row in annotations})
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    overall: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_subset: dict[str, Counter[str]] = defaultdict(Counter)
    criterion_true: Counter[str] = Counter()
    reason_tags: Counter[str] = Counter()
    for image, annotation in annotation_by_image.items():
        row = review_by_image[image]
        quality = str(row["quality"])
        overall[quality] += 1
        by_type[str(annotation.get("task_type"))][quality] += 1
        by_subset[str(annotation.get("source_subset"))][quality] += 1
        for field in BOOLEAN_FIELDS:
            criterion_true[field] += int(row[field])
        for tag in row.get("issue_tags", []):
            reason_tags[str(tag)] += 1

    total = len(reviews)
    summary = {
        "cases": total,
        "review_coverage": f"{total}/{len(annotations)}",
        "overall": rates(overall),
        "by_task_type": {
            name: rates(counts) for name, counts in sorted(by_type.items())
        },
        "by_source_subset": {
            name: rates(counts) for name, counts in sorted(by_subset.items())
        },
        "criterion_true_counts": {
            field: criterion_true[field] for field in BOOLEAN_FIELDS
        },
        "difficult_localization_rate": round(
            criterion_true["difficult_localization"] / total, 4
        ),
        "issue_tag_counts": dict(reason_tags.most_common()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
