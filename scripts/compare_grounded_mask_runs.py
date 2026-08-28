#!/usr/bin/env python3
"""Compare two sparse grounded-mask runs without requiring ground-truth masks."""

from __future__ import annotations

import argparse
import io
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


Key = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sparse grounded-mask A/B runs")
    parser.add_argument("--baseline-mask-dir", type=Path, required=True)
    parser.add_argument("--candidate-mask-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def _read_rows(directory: Path) -> Dict[Key, Dict[str, Any]]:
    rows: Dict[Key, Dict[str, Any]] = {}
    columns = [
        "row_idx", "raw_type", "canonical_type", "area_frac", "mask_png",
        "ground_json", "mask_source", "qc_flag",
    ]
    for path in sorted(directory.glob("*.parquet")):
        available = set(pq.ParquetFile(path).schema.names)
        for row in pq.read_table(path, columns=[item for item in columns if item in available]).to_pylist():
            key = (path.name, int(row.get("row_idx", -1)))
            if key in rows:
                raise ValueError(f"duplicate row {key} in {directory}")
            rows[key] = row
    return rows


def _payload(row: Dict[str, Any]) -> Dict[str, Any]:
    try:
        value = json.loads(str(row.get("ground_json", "{}")))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _box_refs(row: Dict[str, Any]) -> Dict[str, List[str]]:
    boxes = _payload(row).get("boxes", {})
    if not isinstance(boxes, dict):
        boxes = {}
    result: Dict[str, List[str]] = {}
    for side in ("source", "target"):
        items = boxes.get(side, [])
        result[side] = [
            str(item.get("ref", ""))
            for item in items if isinstance(item, dict) and str(item.get("ref", "")).strip()
        ] if isinstance(items, list) else []
    return result


def _observation_counts(row: Dict[str, Any]) -> Tuple[int, int, int]:
    observation = _payload(row).get("observation", {})
    parsed = observation.get("parsed", {}) if isinstance(observation, dict) else {}
    changes = parsed.get("changes", []) if isinstance(parsed, dict) else []
    checks = parsed.get("checked_regions", []) if isinstance(parsed, dict) else []
    aligned_false = sum(
        isinstance(item, dict) and not bool(item.get("instruction_aligned", True))
        for item in changes if isinstance(changes, list)
    )
    changed_checks = sum(
        isinstance(item, dict) and bool(item.get("changed"))
        for item in checks if isinstance(checks, list)
    )
    return len(changes) if isinstance(changes, list) else 0, changed_checks, aligned_false


def _decode_mask(row: Dict[str, Any]) -> np.ndarray:
    data = row.get("mask_png")
    if not data:
        return np.zeros((0, 0), dtype=bool)
    return np.asarray(Image.open(io.BytesIO(data)).convert("L")) > 0


def _mask_metrics(baseline: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    if baseline.shape != candidate.shape:
        raise ValueError(f"mask shape mismatch: {baseline.shape} vs {candidate.shape}")
    intersection = int(np.logical_and(baseline, candidate).sum())
    union = int(np.logical_or(baseline, candidate).sum())
    baseline_sum = int(baseline.sum())
    candidate_sum = int(candidate.sum())
    added = int(np.logical_and(candidate, ~baseline).sum())
    lost = int(np.logical_and(baseline, ~candidate).sum())
    return {
        "iou": intersection / union if union else 1.0,
        "candidate_added_frac": added / candidate_sum if candidate_sum else 0.0,
        "baseline_lost_frac": lost / baseline_sum if baseline_sum else 0.0,
    }


def _stats(values: Iterable[float]) -> Dict[str, float | int]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "mean": round(statistics.fmean(clean), 6),
        "median": round(statistics.median(clean), 6),
        "min": round(min(clean), 6),
        "max": round(max(clean), 6),
    }


def compare(baseline_dir: Path, candidate_dir: Path) -> Dict[str, Any]:
    baseline = _read_rows(baseline_dir)
    candidate = _read_rows(candidate_dir)
    common = sorted(set(baseline) & set(candidate))
    cases: List[Dict[str, Any]] = []
    for shard, row_idx in common:
        before = baseline[(shard, row_idx)]
        after = candidate[(shard, row_idx)]
        before_refs = _box_refs(before)
        after_refs = _box_refs(after)
        observation_changes, observation_changed_checks, observation_extra = _observation_counts(after)
        mask_metrics = _mask_metrics(_decode_mask(before), _decode_mask(after))
        before_count = sum(len(items) for items in before_refs.values())
        after_count = sum(len(items) for items in after_refs.values())
        cases.append(
            {
                "shard": shard,
                "row_idx": row_idx,
                "raw_type": str(after.get("raw_type", before.get("raw_type", ""))),
                "baseline_refs": before_refs,
                "candidate_refs": after_refs,
                "baseline_box_count": before_count,
                "candidate_box_count": after_count,
                "box_count_delta": after_count - before_count,
                "observation_changes": observation_changes,
                "observation_changed_checks": observation_changed_checks,
                "observation_instruction_unaligned": observation_extra,
                "baseline_area_frac": float(before.get("area_frac", 0.0) or 0.0),
                "candidate_area_frac": float(after.get("area_frac", 0.0) or 0.0),
                "area_delta": float(after.get("area_frac", 0.0) or 0.0)
                - float(before.get("area_frac", 0.0) or 0.0),
                "baseline_mask_source": str(before.get("mask_source", "")),
                "candidate_mask_source": str(after.get("mask_source", "")),
                "baseline_qc": str(before.get("qc_flag", "")),
                "candidate_qc": str(after.get("qc_flag", "")),
                **mask_metrics,
            }
        )

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["raw_type"]].append(case)

    def summarize(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "rows": len(items),
            "baseline_boxes": sum(item["baseline_box_count"] for item in items),
            "candidate_boxes": sum(item["candidate_box_count"] for item in items),
            "rows_with_changed_refs_or_count": sum(
                item["baseline_refs"] != item["candidate_refs"] for item in items
            ),
            "observation_changes": sum(item["observation_changes"] for item in items),
            "observation_instruction_unaligned": sum(
                item["observation_instruction_unaligned"] for item in items
            ),
            "mask_iou": _stats(item["iou"] for item in items),
            "area_delta": _stats(item["area_delta"] for item in items),
            "candidate_added_frac": _stats(item["candidate_added_frac"] for item in items),
            "baseline_lost_frac": _stats(item["baseline_lost_frac"] for item in items),
            "candidate_qc": dict(sorted(Counter(item["candidate_qc"] for item in items).items())),
        }

    summary = summarize(cases)
    summary["matched_rows"] = len(common)
    summary["baseline_only"] = [f"{shard}:{row}" for shard, row in sorted(set(baseline) - set(candidate))]
    summary["candidate_only"] = [f"{shard}:{row}" for shard, row in sorted(set(candidate) - set(baseline))]
    return {
        "baseline_mask_dir": str(baseline_dir),
        "candidate_mask_dir": str(candidate_dir),
        "summary": summary,
        "by_type": {key: summarize(value) for key, value in sorted(grouped.items())},
        "largest_candidate_additions": sorted(
            cases, key=lambda item: item["candidate_added_frac"], reverse=True
        )[:10],
        "largest_baseline_losses": sorted(
            cases, key=lambda item: item["baseline_lost_frac"], reverse=True
        )[:10],
        "cases": cases,
    }


def markdown_report(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Grounded-mask A/B comparison",
        "",
        f"Matched rows: {summary['matched_rows']}",
        "",
        f"Boxes: baseline {summary['baseline_boxes']} → candidate {summary['candidate_boxes']}; "
        f"changed refs/count in {summary['rows_with_changed_refs_or_count']} rows.",
        "",
        f"Mask IoU median: {summary['mask_iou']['median']:.4f}; "
        f"area delta median: {summary['area_delta']['median']:+.4f}.",
        "",
        "| type | rows | boxes A→B | changed rows | mask IoU median | area delta median | unaligned observations |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for raw_type, stats in report["by_type"].items():
        lines.append(
            f"| {raw_type} | {stats['rows']} | {stats['baseline_boxes']}→{stats['candidate_boxes']} "
            f"| {stats['rows_with_changed_refs_or_count']} | {stats['mask_iou']['median']:.4f} "
            f"| {stats['area_delta']['median']:+.4f} | {stats['observation_instruction_unaligned']} |"
        )
    lines.extend(["", "## Largest candidate-only mask fractions", ""])
    for item in report["largest_candidate_additions"]:
        lines.append(
            f"- `{item['shard']}` row {item['row_idx']}: "
            f"added={item['candidate_added_frac']:.3f}, IoU={item['iou']:.3f}, "
            f"boxes {item['baseline_box_count']}→{item['candidate_box_count']}"
        )
    lines.extend(["", "## Largest baseline-only mask fractions", ""])
    for item in report["largest_baseline_losses"]:
        lines.append(
            f"- `{item['shard']}` row {item['row_idx']}: "
            f"lost={item['baseline_lost_frac']:.3f}, IoU={item['iou']:.3f}, "
            f"boxes {item['baseline_box_count']}→{item['candidate_box_count']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = compare(args.baseline_mask_dir.resolve(), args.candidate_mask_dir.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
