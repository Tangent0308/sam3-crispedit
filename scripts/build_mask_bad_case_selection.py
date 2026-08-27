#!/usr/bin/env python3
"""Extract the mask-side bad cases from BAD_CASE_REVIEW into a selection JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CASE_PATTERN = re.compile(r"^#### `(.+?\.parquet)` row `(\d+)`", re.MULTILINE)


def extract_mask_cases(markdown: str) -> list[dict]:
    try:
        section = markdown.split("## 2. Mask 打标这一侧 bad case", 1)[1]
    except IndexError as exc:
        raise ValueError("mask bad-case section was not found") from exc
    section = section.split("## 3.", 1)[0]
    cases = [
        {"shard": match.group(1), "row_idx": int(match.group(2))}
        for match in CASE_PATTERN.finditer(section)
    ]
    if not cases:
        raise ValueError("no mask bad cases were found")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-doc",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "BAD_CASE_REVIEW_20260826.md",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = extract_mask_cases(args.review_doc.read_text(encoding="utf-8"))
    payload = {
        "name": "mask_bad_cases_20260826",
        "source_doc": str(args.review_doc.resolve()),
        "num_cases": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
