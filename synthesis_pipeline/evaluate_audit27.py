"""Score 27B audit variants against independent case-level review labels.

False acceptance is reported separately because admitting a bad training pair
is more costly than rejecting a usable one.  Rewrite proposals are scored only
when a human explicitly marked their validity.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {str(row["image"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate images in {path}")
    return result


def confusion(manual: dict[str, dict], audit: dict[str, dict], field: str) -> dict:
    counts: Counter[str] = Counter()
    for image, expected_row in manual.items():
        expected = expected_row.get(field)
        predicted_row = audit.get(image)
        observed = (
            predicted_row.get("quality") if field == "quality"
            else (predicted_row.get("audit") or {}).get(field)
        ) if predicted_row else None
        if expected not in {"pass", "fail"}:
            counts["missing_manual_label"] += 1
        elif observed not in {"pass", "fail"}:
            counts["missing_or_parse_error"] += 1
        elif expected == observed == "pass":
            counts["true_pass"] += 1
        elif expected == observed == "fail":
            counts["true_fail"] += 1
        elif observed == "pass":
            counts["false_accept"] += 1
        else:
            counts["false_reject"] += 1
    scored = sum(counts[key] for key in ("true_pass", "true_fail", "false_accept", "false_reject"))
    actual_fail = counts["true_fail"] + counts["false_accept"]
    return {
        "cases": len(manual),
        **{key: counts[key] for key in ("true_pass", "true_fail", "false_accept", "false_reject", "missing_or_parse_error", "missing_manual_label")},
        "accuracy": round((counts["true_pass"] + counts["true_fail"]) / scored, 4) if scored else None,
        "false_accept_rate_among_bad_pairs": round(counts["false_accept"] / actual_fail, 4) if actual_fail else None,
    }


def evaluate(manual: dict[str, dict], audit: dict[str, dict]) -> dict:
    by_type: dict[str, dict[str, dict]] = defaultdict(dict)
    for task_type in sorted({str(row.get("task_type")) for row in manual.values()}):
        subset = {image: row for image, row in manual.items() if row.get("task_type") == task_type}
        by_type[task_type] = {field: confusion(subset, audit, field) for field in ("quality", "visual_quality", "instruction_match")}
    proposals = [(image, row["rewrite_candidate"]) for image, row in audit.items() if image in manual and row.get("rewrite_candidate")]
    validity = [manual[image].get("rewrite_valid") for image, _ in proposals]
    return {
        "overall": {field: confusion(manual, audit, field) for field in ("quality", "visual_quality", "instruction_match")},
        "by_type": by_type,
        "rewrite": {
            "proposed": len(proposals),
            "human_valid": sum(value is True for value in validity),
            "human_invalid": sum(value is False for value in validity),
            "unreviewed": sum(value is None for value in validity),
            "cases": [{"image": image, "candidate": candidate, "valid": manual[image].get("rewrite_valid")} for image, candidate in proposals],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-jsonl", type=Path, required=True)
    parser.add_argument(
        "--split-annotations-jsonl",
        type=Path,
        help="Restrict scoring to image IDs in this source-disjoint split.",
    )
    parser.add_argument("--audit", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    manual = read_jsonl(args.manual_jsonl)
    if args.split_annotations_jsonl:
        split = read_jsonl(args.split_annotations_jsonl)
        missing = sorted(set(split) - set(manual))
        if missing:
            raise ValueError(f"Missing manual labels for split images: {missing[:10]}")
        manual = {image: manual[image] for image in split}
    result = {}
    for item in args.audit:
        name, separator, path = item.partition("=")
        if not separator or not name:
            raise ValueError(f"Expected NAME=PATH, received {item!r}")
        result[name] = evaluate(manual, read_jsonl(Path(path)))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"overall": value["overall"]["quality"], "rewrite": {k: v for k, v in value["rewrite"].items() if k != "cases"}} for name, value in result.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
