#!/usr/bin/env python3
"""Build the documented Class A/Class B prefilter regression slice."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = Path(
    "/mnt/bn/strategy-mllm-train/user/tanyue/datasets/CrispEdit-2M"
)

REGRESSION_ROWS: List[Tuple[str, int, str]] = [
    ("remove_00009.parquet", 148, "A_FALSE_KEEP_NOOP"),
    ("remove_00000.parquet", 55, "A_FALSE_KEEP_NOOP"),
    ("add_00002.parquet", 70, "A_FALSE_KEEP_NOOP"),
    ("background change_00002.parquet", 101, "A_FALSE_KEEP_NOOP"),
    ("motion change_00000.parquet", 46, "A_FALSE_KEEP_NOOP"),
    ("replace_00008.parquet", 34, "A_FALSE_KEEP_NOOP"),
    ("color_00000.parquet", 17, "B_KEEP_REASON_MISMATCH"),
    ("color_00001.parquet", 118, "B_KEEP_REASON_MISMATCH"),
]

POSITIVE_CONTROL_ROWS: List[Tuple[str, int, str]] = [
    ("add_00002.parquet", 115, "HIGH_CONFIDENCE_KEEP_CONTROL"),
    ("remove_00000.parquet", 0, "HIGH_CONFIDENCE_KEEP_CONTROL"),
    ("replace_00003.parquet", 109, "HIGH_CONFIDENCE_KEEP_CONTROL"),
    ("color_00001.parquet", 130, "HIGH_CONFIDENCE_KEEP_CONTROL"),
    ("motion change_00000.parquet", 2, "HIGH_CONFIDENCE_KEEP_CONTROL"),
    ("background change_00000.parquet", 1, "HIGH_CONFIDENCE_KEEP_CONTROL"),
    ("style_00000.parquet", 0, "HIGH_CONFIDENCE_KEEP_CONTROL"),
]

PREFILTER_BAD_CASE_ROWS: List[Tuple[str, int, str, str]] = [
    ("motion change_00060.parquet", 183, "AMBIGUOUS_SUBTLE_CHANGE", "drop"),
    ("remove_00011.parquet", 225, "A_FALSE_KEEP_NOOP", "drop"),
    ("remove_00023.parquet", 208, "WRONG_THING_EDITED", "drop"),
    ("remove_00071.parquet", 249, "A_FALSE_KEEP_NOOP", "drop"),
    ("motion change_00007.parquet", 219, "AMBIGUOUS_SUBTLE_CHANGE", "drop"),
    ("motion change_00038.parquet", 194, "AMBIGUOUS_SUBTLE_CHANGE", "drop"),
    ("motion change_00048.parquet", 176, "AMBIGUOUS_SUBTLE_CHANGE", "drop"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression", type=str, default="zstd")
    parser.add_argument(
        "--shards-per-type",
        type=int,
        default=1,
        help="Split each edit type into up to this many mini-shards (use 2 to exercise 8 GPUs)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-positive-controls",
        action="store_true",
        help="Also include one expected-keep control row per edit type",
    )
    parser.add_argument(
        "--include-prefilter-bad-cases",
        action="store_true",
        help="Also include the reviewed historical prefilter failures",
    )
    return parser.parse_args()


def raw_type_from_name(name: str) -> str:
    return name.rsplit("_", 1)[0]


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.glob("*.parquet")) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} already contains parquet files; use --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.shards_per_type < 1:
        raise ValueError("--shards-per-type must be at least 1")
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    manifest = []
    table_cache: Dict[str, pa.Table] = {}
    selected_rows = [
        (*row, "drop" if row[2] == "A_FALSE_KEEP_NOOP" else "keep")
        for row in REGRESSION_ROWS
    ]
    if args.include_positive_controls:
        selected_rows.extend((*row, "keep") for row in POSITIVE_CONTROL_ROWS)
    if args.include_prefilter_bad_cases:
        selected_rows.extend(PREFILTER_BAD_CASE_ROWS)
    for source_shard, source_row_idx, issue_class, expected_decision in selected_rows:
        source_path = args.input_dir / source_shard
        if source_shard not in table_cache:
            table_cache[source_shard] = pq.read_table(source_path)
        row = table_cache[source_shard].slice(source_row_idx, 1).to_pylist()[0]
        row["source_shard"] = source_shard
        row["source_row_idx"] = source_row_idx
        row["expected_issue_class"] = issue_class
        row["expected_decision"] = expected_decision
        raw_type = str(row.get("type") or raw_type_from_name(source_shard))
        grouped[raw_type].append(row)
        manifest.append(
            {
                "source_shard": source_shard,
                "source_row_idx": source_row_idx,
                "raw_type": raw_type,
                "instruction": row.get("instruction", ""),
                "expected_issue_class": issue_class,
                "expected_decision": expected_decision,
            }
        )
    for raw_type, rows in sorted(grouped.items()):
        shard_count = min(args.shards_per_type, len(rows))
        for shard_idx in range(shard_count):
            shard_rows = rows[shard_idx::shard_count]
            output_path = args.output_dir / f"{raw_type}_{shard_idx:05d}.parquet"
            pq.write_table(
                pa.Table.from_pylist(shard_rows), output_path, compression=args.compression
            )
            print(f"wrote {output_path.name}: {len(shard_rows)} rows")
    (args.output_dir / "regression_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
