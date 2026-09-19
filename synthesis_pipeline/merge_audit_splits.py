"""Merge source-disjoint audit splits in original annotation order."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def merge(annotations: list[dict], audit_splits: list[list[dict]]) -> list[dict]:
    names = [str(row["image"]) for row in annotations]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate images in annotations")
    by_image: dict[str, dict] = {}
    for split in audit_splits:
        for row in split:
            name = str(row["image"])
            if name in by_image:
                raise ValueError(f"Duplicate audit image across splits: {name}")
            by_image[name] = row
    if set(by_image) != set(names):
        raise ValueError(
            f"Audit/annotation mismatch: missing={sorted(set(names) - set(by_image))[:5]}, "
            f"extra={sorted(set(by_image) - set(names))[:5]}"
        )
    return [by_image[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    args = parser.parse_args()
    rows = merge(
        load_jsonl(args.annotations_jsonl),
        [load_jsonl(path) for path in args.audit_jsonl],
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "cases": len(rows),
        "source_audits": [str(path) for path in args.audit_jsonl],
        "quality_counts": dict(Counter(row.get("quality") for row in rows)),
        "rewrite_candidate_count": sum(bool(row.get("rewrite_candidate")) for row in rows),
    }
    args.output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
