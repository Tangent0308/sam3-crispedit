#!/usr/bin/env python3
"""Overlay targeted grounding rows onto a full grounding run by sample_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--updates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()
    args.base_dir = args.base_dir.resolve()
    args.updates_dir = args.updates_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir in {args.base_dir, args.updates_dir}:
        raise ValueError("output directory must differ from both inputs")

    updates = {}
    for path in sorted(args.updates_dir.glob("part-*.parquet")):
        for row in pq.read_table(path).to_pylist():
            sample_id = str(row["sample_id"])
            if sample_id in updates:
                raise ValueError(f"duplicate update sample_id: {sample_id}")
            updates[sample_id] = row
    if not updates:
        raise FileNotFoundError(f"no update rows under {args.updates_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    applied = set()
    total_rows = 0
    for base_path in sorted(args.base_dir.glob("part-*.parquet")):
        table = pq.read_table(base_path)
        rows = table.to_pylist()
        for index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            replacement = updates.get(sample_id)
            if replacement is None:
                continue
            if int(replacement["row_idx"]) != int(row["row_idx"]):
                raise ValueError(f"row_idx mismatch for {sample_id}")
            rows[index] = replacement
            applied.add(sample_id)
        total_rows += len(rows)
        output_path = args.output_dir / base_path.name
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema),
            tmp_path,
            compression=args.compression,
        )
        tmp_path.replace(output_path)

    missing = set(updates) - applied
    if missing:
        raise KeyError(f"update sample_id(s) absent from base: {sorted(missing)}")
    summary = {"rows": total_rows, "updates": len(updates), "updated_sample_ids": sorted(applied)}
    (args.output_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
