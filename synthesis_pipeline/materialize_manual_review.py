"""Expand reviewed defaults and per-image decisions into a full review JSONL."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--decisions-json", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    annotations = load_jsonl(args.annotations_jsonl)
    decisions = json.loads(args.decisions_json.read_text(encoding="utf-8"))
    default = decisions.get("default")
    overrides = decisions.get("overrides", {})
    if not isinstance(default, dict) or not isinstance(overrides, dict):
        raise ValueError("Decisions JSON requires object fields: default and overrides")

    annotation_names = {str(row["image"]) for row in annotations}
    unknown = sorted(set(overrides) - annotation_names)
    if unknown:
        raise ValueError(f"Decision overrides not found in annotations: {unknown}")

    rows = []
    for annotation in annotations:
        image = str(annotation["image"])
        decision = {**default, **overrides.get(image, {})}
        reason = str(decision.get("reason", "")).replace(
            "{task_type}", str(annotation.get("task_type", "edit"))
        )
        rows.append(
            {
                "image": image,
                "task_type": annotation.get("task_type"),
                "source_subset": annotation.get("source_subset"),
                "parquet_row_index": annotation.get("parquet_row_index"),
                "mask_index": annotation.get("mask_index"),
                **decision,
                "reason": reason,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output_jsonl)
    print(json.dumps({"cases": len(rows), "output": str(args.output_jsonl)}, indent=2))


if __name__ == "__main__":
    main()
