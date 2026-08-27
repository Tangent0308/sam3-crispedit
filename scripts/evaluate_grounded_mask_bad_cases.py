#!/usr/bin/env python3
"""Summarize the sparse 2026-08-26 grounded-mask regression run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow.parquet as pq


def read_sparse_outputs(output_dir: Path) -> List[Dict]:
    rows = []
    for path in sorted(output_dir.glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            row["shard"] = path.name
            rows.append(row)
    return rows


def read_parquet_row(path: Path, row_idx: int, columns: Iterable[str]) -> Dict:
    pf = pq.ParquetFile(path)
    start = 0
    for group_index in range(pf.num_row_groups):
        count = pf.metadata.row_group(group_index).num_rows
        if start <= row_idx < start + count:
            return (
                pf.read_row_group(group_index, columns=list(columns))
                .slice(row_idx - start, 1)
                .to_pylist()[0]
            )
        start += count
    raise IndexError(f"row {row_idx} out of range for {path}")


def _area_stats(values: List[float]) -> Dict:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min": round(min(clean), 6),
        "median": round(statistics.median(clean), 6),
        "mean": round(statistics.fmean(clean), 6),
        "max": round(max(clean), 6),
    }


def evaluate(mask_dir: Path, selection_path: Path, old_mask_dir: Path | None) -> Dict:
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = {
        (str(item["shard"]), int(item["row_idx"]))
        for item in selection_payload["cases"]
    }
    rows = read_sparse_outputs(mask_dir)
    actual = {(row["shard"], int(row["row_idx"])) for row in rows}
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_type[str(row["raw_type"])].append(row)

    old_areas: Dict[tuple, float] = {}
    if old_mask_dir is not None:
        for shard, row_idx in expected:
            old_path = old_mask_dir / shard
            old = read_parquet_row(old_path, row_idx, ["area_frac"])
            old_areas[(shard, row_idx)] = float(old["area_frac"])

    cases = []
    for row in sorted(rows, key=lambda item: (item["raw_type"], item["shard"], item["row_idx"])):
        key = (row["shard"], int(row["row_idx"]))
        instance_masks = row.get("instance_masks") or []
        cases.append(
            {
                "shard": row["shard"],
                "row_idx": int(row["row_idx"]),
                "raw_type": row["raw_type"],
                "qc_flag": row["qc_flag"],
                "mask_source": row["mask_source"],
                "area_frac": row["area_frac"],
                "old_area_frac": old_areas.get(key),
                "instance_count": len(instance_masks),
                "coverage_box_unions": sum(
                    bool(instance.get("coverage_box_union")) for instance in instance_masks
                ),
            }
        )

    type_stats = {}
    for raw_type, type_rows in sorted(by_type.items()):
        type_stats[raw_type] = {
            "rows": len(type_rows),
            "qc_flags": dict(Counter(str(row["qc_flag"]) for row in type_rows)),
            "mask_sources": dict(Counter(str(row["mask_source"]) for row in type_rows)),
            "area": _area_stats([row["area_frac"] for row in type_rows if row["area_frac"] is not None]),
            "instances": sum(len(row.get("instance_masks") or []) for row in type_rows),
        }

    return {
        "expected_rows": len(expected),
        "actual_rows": len(actual),
        "missing": [f"{shard}:{row}" for shard, row in sorted(expected - actual)],
        "unexpected": [f"{shard}:{row}" for shard, row in sorted(actual - expected)],
        "qc_flags": dict(Counter(str(row["qc_flag"]) for row in rows)),
        "mask_sources": dict(Counter(str(row["mask_source"]) for row in rows)),
        "area": _area_stats([row["area_frac"] for row in rows if row["area_frac"] is not None]),
        "by_type": type_stats,
        "cases": cases,
    }


def markdown_report(report: Dict) -> str:
    lines = [
        "# Grounded mask bad-case evaluation",
        "",
        f"Rows: {report['actual_rows']} / {report['expected_rows']}",
        "",
        f"QC: `{json.dumps(report['qc_flags'], ensure_ascii=False)}`",
        "",
        f"Mask source: `{json.dumps(report['mask_sources'], ensure_ascii=False)}`",
        "",
        "| type | rows | QC | source | median area | instances |",
        "|---|---:|---|---|---:|---:|",
    ]
    for raw_type, stats in report["by_type"].items():
        lines.append(
            f"| {raw_type} | {stats['rows']} | `{json.dumps(stats['qc_flags'], ensure_ascii=False)}` "
            f"| `{json.dumps(stats['mask_sources'], ensure_ascii=False)}` | "
            f"{stats['area'].get('median', float('nan')):.4f} | {stats['instances']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--old-mask-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.mask_dir, args.selection_file, args.old_mask_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("expected_rows", "actual_rows", "missing", "unexpected", "qc_flags", "mask_sources", "area")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
