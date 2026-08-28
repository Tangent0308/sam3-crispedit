#!/usr/bin/env python3
"""Export Qwen grounding parquet rows to human-readable JSONL/CSV/Markdown.

The mask pipeline intentionally keeps parquet as its scalable interchange format.
This utility is a read-only view/export layer: it does not alter parquet files and
keeps each Qwen raw response alongside the parsed boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export grounding parquet to JSONL/CSV/Markdown")
    parser.add_argument("--grounding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-types", default=None, help="Comma-separated canonical/raw types")
    parser.add_argument(
        "--write-json-array",
        action="store_true",
        help="Also write grounding_outputs.json (convenient for small review slices)",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    """Convert Arrow/Python values to strict JSON-compatible values."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _parse_payload(raw: Any) -> Dict[str, Any]:
    try:
        payload = json.loads(str(raw or "{}"))
    except (TypeError, ValueError):
        return {"parse_error": "ground_json is not valid JSON", "raw": str(raw or "")}
    return payload if isinstance(payload, dict) else {"parse_error": "ground_json is not an object", "raw": payload}


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return value or "unknown"


def _fence(text: str) -> str:
    runs = re.findall(r"`+", text or "")
    width = max([len(item) for item in runs] + [3]) + 1
    return "`" * width


def _read_records(grounding_dir: Path, include: Optional[set[str]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(grounding_dir.glob("*.parquet")):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            raw_type = str(row.get("raw_type", ""))
            canonical_type = str(row.get("canonical_type", raw_type))
            if include and raw_type not in include and canonical_type not in include:
                continue
            payload = _parse_payload(row.get("ground_json", "{}"))
            record = {
                "shard": path.name,
                "row_idx": int(row.get("row_idx", -1)),
                "raw_type": raw_type,
                "canonical_type": canonical_type,
                "instruction": str(row.get("instruction", "")),
                "grounding_status": str(row.get("grounding_status", "")),
                "qc_flag": str(row.get("qc_flag", "")),
                "ground_parse_ok": bool(row.get("ground_parse_ok", False)),
                "source_width": int(row.get("source_width", 0) or 0),
                "source_height": int(row.get("source_height", 0) or 0),
                "target_width": int(row.get("target_width", 0) or 0),
                "target_height": int(row.get("target_height", 0) or 0),
                "mllm_model": str(row.get("mllm_model", "")),
                "prompt_version": str(row.get("prompt_version", "")),
                "grounding_seconds": _json_safe(row.get("grounding_seconds")),
                "prefilter_verdict": str(row.get("prefilter_verdict", "")),
                "prefilter_confidence": _json_safe(row.get("prefilter_confidence")),
                "grounding": _json_safe(payload),
            }
            records.append(record)
    records.sort(key=lambda item: (item["canonical_type"], item["shard"], item["row_idx"]))
    return records


def _requests(records: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for record in records:
        payload = record.get("grounding", {})
        request_items = payload.get("requests", []) if isinstance(payload, dict) else []
        if not request_items:
            yield {
                "canonical_type": record["canonical_type"],
                "shard": record["shard"],
                "row_idx": record["row_idx"],
                "instruction": record["instruction"],
                "grounding_status": record["grounding_status"],
                "qc_flag": record["qc_flag"],
                "grounding_image": "",
                "parse_ok": False,
                "error": "no request in ground_json",
                "raw_text": "",
                "boxes_json": "[]",
            }
            continue
        for request in request_items:
            request = request if isinstance(request, dict) else {}
            yield {
                "canonical_type": record["canonical_type"],
                "shard": record["shard"],
                "row_idx": record["row_idx"],
                "instruction": record["instruction"],
                "grounding_status": record["grounding_status"],
                "qc_flag": record["qc_flag"],
                "grounding_image": str(request.get("grounding_image", "")),
                "parse_ok": bool(request.get("parse_ok", False)),
                "error": str(request.get("error", "")),
                "raw_text": str(request.get("raw_text", "")),
                "boxes_json": json.dumps(request.get("boxes", []), ensure_ascii=False, separators=(",", ":")),
            }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_markdown(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["canonical_type"]].append(record)
    lines = [
        "# Qwen3.5 grounding outputs",
        "",
        "每个样本保留解析后的 `grounding`，其中 `requests[].raw_text` 是模型原始输出。",
        "坐标均为对应 grounding image 上的 `[0, 1000]` 归一化坐标。",
        "",
    ]
    for category in sorted(grouped):
        rows = grouped[category]
        lines.extend([f"## {category} ({len(rows)})", ""])
        for record in rows:
            lines.extend(
                [
                    f"### `{record['shard']}` row `{record['row_idx']}`",
                    "",
                    f"- instruction: {record['instruction']}",
                    f"- status: `{record['grounding_status']}`; qc: `{record['qc_flag']}`; parse: `{record['ground_parse_ok']}`",
                    f"- model: `{record['mllm_model']}`; prompt: `{record['prompt_version']}`",
                    "",
                ]
            )
            payload = record.get("grounding", {})
            for request in payload.get("requests", []) if isinstance(payload, dict) else []:
                image_name = str(request.get("grounding_image", ""))
                raw_text = str(request.get("raw_text", ""))
                lines.extend(
                    [
                        f"#### {image_name}",
                        "",
                        f"- parse: `{request.get('parse_ok', False)}`; error: `{request.get('error', '')}`",
                        "- parsed boxes:",
                        "",
                        "```json",
                        json.dumps(request.get("boxes", []), ensure_ascii=False, indent=2),
                        "```",
                        "",
                        "- raw model response:",
                        "",
                    ]
                )
                fence = _fence(raw_text)
                lines.extend([f"{fence}text", raw_text, fence, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    include = {item.strip() for item in args.include_types.split(",") if item.strip()} if args.include_types else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(args.grounding_dir.resolve(), include)
    if not records:
        raise SystemExit("no grounding parquet rows matched")

    all_jsonl = args.output_dir / "grounding_outputs.jsonl"
    _write_jsonl(all_jsonl, records)
    if args.write_json_array:
        (args.output_dir / "grounding_outputs.json").write_text(
            json.dumps(_json_safe(records), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    by_category = args.output_dir / "by_category"
    by_category.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["canonical_type"]].append(record)
    for category, category_records in sorted(grouped.items()):
        _write_jsonl(by_category / f"{_slug(category)}.jsonl", category_records)

    request_rows = list(_requests(records))
    with (args.output_dir / "grounding_requests.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "canonical_type", "shard", "row_idx", "instruction", "grounding_status", "qc_flag",
            "grounding_image", "parse_ok", "error", "raw_text", "boxes_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(request_rows)
    _write_markdown(args.output_dir / "grounding_outputs.md", records)

    summary = {
        "rows": len(records),
        "requests": len(request_rows),
        "categories": dict(sorted(Counter(record["canonical_type"] for record in records).items())),
        "statuses": dict(sorted(Counter(record["grounding_status"] for record in records).items())),
        "qc_flags": dict(sorted(Counter(record["qc_flag"] for record in records).items())),
        "files": [
            "grounding_outputs.jsonl",
            "grounding_requests.csv",
            "grounding_outputs.md",
            "by_category/",
        ] + (["grounding_outputs.json"] if args.write_json_array else []),
    }
    (args.output_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
