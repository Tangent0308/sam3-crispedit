#!/usr/bin/env python3
"""Evaluate the fact prefilter on a generated raw-row regression slice."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def expected_decision(issue_class: str) -> str:
    return "drop" if issue_class == "A_FALSE_KEEP_NOOP" else "keep"


def main() -> None:
    args = parse_args()
    rows = []
    counts: Counter = Counter()
    for input_path in sorted(args.input_dir.glob("*.parquet")):
        audit_path = args.audit_dir / input_path.name
        if not audit_path.exists():
            raise FileNotFoundError(audit_path)
        inputs = pq.read_table(
            input_path,
            columns=[
                "instruction",
                "type",
                "source_shard",
                "source_row_idx",
                "expected_issue_class",
            ],
        ).to_pylist()
        audits = {
            row["row_idx"]: row
            for row in pq.read_table(
                audit_path,
                columns=[
                    "row_idx",
                    "filter_decision",
                    "prefilter_verdict",
                    "prefilter_reason",
                    "filter_reason_codes",
                    "prefilter_review_triggered",
                    "prefilter_review_resolution",
                    "prefilter_predicates_json",
                ],
            ).to_pylist()
        }
        for row_idx, input_row in enumerate(inputs):
            audit = audits[row_idx]
            issue_class = input_row["expected_issue_class"]
            expected = expected_decision(issue_class)
            actual = audit["filter_decision"]
            correct = actual == expected
            result = {
                "mini_shard": input_path.name,
                "row_idx": row_idx,
                "source_shard": input_row["source_shard"],
                "source_row_idx": input_row["source_row_idx"],
                "raw_type": input_row["type"],
                "instruction": input_row["instruction"],
                "issue_class": issue_class,
                "expected_decision": expected,
                "actual_decision": actual,
                "verdict": audit["prefilter_verdict"],
                "correct": correct,
                "reason": audit["prefilter_reason"],
                "reason_codes": audit["filter_reason_codes"],
                "review_triggered": audit["prefilter_review_triggered"],
                "review_resolution": audit["prefilter_review_resolution"],
                "predicates": json.loads(audit["prefilter_predicates_json"]),
            }
            rows.append(result)
            counts["rows"] += 1
            counts["correct"] += int(correct)
            counts["incorrect"] += int(not correct)
            counts[f"class_{issue_class}_rows"] += 1
            counts[f"class_{issue_class}_correct"] += int(correct)
            counts[f"actual_{actual}"] += 1
            counts["review_triggered"] += int(audit["prefilter_review_triggered"])
    summary = dict(counts)
    summary["accuracy"] = round(counts["correct"] / max(counts["rows"], 1), 6)
    report = {"summary": summary, "rows": rows}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        marker = "OK" if row["correct"] else "MISS"
        print(
            f"{marker} {row['source_shard']}:{row['source_row_idx']} "
            f"{row['issue_class']} expected={row['expected_decision']} "
            f"actual={row['actual_decision']}"
        )


if __name__ == "__main__":
    main()
