"""Compare audit outputs with manual pair-level quality decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rows_by_image(path: Path) -> dict[str, dict]:
    return {
        row["image"]: row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def score(manual: dict[str, dict], automatic: dict[str, dict]) -> dict:
    counts = Counter()
    for image, reference in manual.items():
        if image not in automatic:
            counts["missing"] += 1
            continue
        predicted = automatic[image]["quality"]
        expected = reference["quality"]
        if predicted not in {"pass", "fail"}:
            counts["parse_error"] += 1
        elif predicted == expected == "pass":
            counts["true_pass"] += 1
        elif predicted == expected == "fail":
            counts["true_fail"] += 1
        elif predicted == "pass":
            counts["false_accept"] += 1
        else:
            counts["false_reject"] += 1
    valid = sum(counts[key] for key in (
        "true_pass", "true_fail", "false_accept", "false_reject"
    ))
    return {
        **dict(counts),
        "evaluated": valid,
        "accuracy": round(
            (counts["true_pass"] + counts["true_fail"]) / valid, 4
        ) if valid else None,
        "false_accept_rate_among_manual_fail": round(
            counts["false_accept"] /
            (counts["false_accept"] + counts["true_fail"]), 4
        ) if counts["false_accept"] + counts["true_fail"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-jsonl", type=Path, required=True)
    parser.add_argument("--legacy-jsonl", type=Path, required=True)
    parser.add_argument("--qwen8b-jsonl", type=Path, required=True)
    parser.add_argument("--qwen38-jsonl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    manual = rows_by_image(args.manual_jsonl)
    audits = {
        "legacy_8b_three_images": rows_by_image(args.legacy_jsonl),
        "new_8b_two_images": rows_by_image(args.qwen8b_jsonl),
        "new_27b_two_images": rows_by_image(args.qwen38_jsonl),
    }
    if any(set(rows) != set(manual) for rows in audits.values()):
        raise ValueError("The manual and audit case sets must match exactly")
    result = {
        "cases": len(manual),
        "scores": {name: score(manual, rows) for name, rows in audits.items()},
        "cases_detail": [
            {
                "image": image,
                "manual": manual[image]["quality"],
                **{name: rows[image]["quality"] for name, rows in audits.items()},
                "qwen8b_rewrite": audits["new_8b_two_images"][image].get(
                    "rewrite_candidate"
                ),
                "qwen38_rewrite": audits["new_27b_two_images"][image].get(
                    "rewrite_candidate"
                ),
            }
            for image in sorted(manual)
        ],
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["scores"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
