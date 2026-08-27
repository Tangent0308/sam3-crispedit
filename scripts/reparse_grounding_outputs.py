#!/usr/bin/env python3
"""Reparse saved raw Qwen responses after parser improvements, without inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from crispedit_grounding import grounding_is_complete, parse_grounding_output
from crispedit_mllm_grounding import GROUND_SCHEMA


def repair_row(row: dict) -> dict:
    if row.get("qc_flag") == "PREFILTER_SKIP":
        return row
    payload = json.loads(row["ground_json"])
    boxes = payload.setdefault("boxes", {"source": [], "target": []})
    parse_ok = True
    for request in payload.get("requests", []):
        try:
            request["boxes"] = parse_grounding_output(request.get("raw_text", ""))
            request["parse_ok"] = True
            request["error"] = ""
            request["reparsed"] = True
        except Exception as exc:
            request["parse_ok"] = False
            request["error"] = repr(exc)
        parse_ok &= bool(request.get("parse_ok"))
        boxes[request["grounding_image"]] = request.get("boxes", [])
    etype = str(row["canonical_type"])
    complete = parse_ok and grounding_is_complete(etype, boxes)
    if not parse_ok:
        status = "PARSE_ERROR"
    elif complete and etype == "replace" and not all(boxes.get(side) for side in ("source", "target")):
        status = "PARTIAL_OK"
    elif complete:
        status = "OK"
    else:
        status = "GROUND_FAIL"
    row["ground_json"] = json.dumps(payload, ensure_ascii=False)
    row["ground_parse_ok"] = parse_ok
    row["grounding_status"] = status
    row["qc_flag"] = "OK" if complete else "GROUND_FAIL"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding-dir", type=Path, required=True)
    args = parser.parse_args()
    counts = Counter()
    for path in sorted(args.grounding_dir.glob("*.parquet")):
        rows = [repair_row(row) for row in pq.read_table(path).to_pylist()]
        table = pa.Table.from_pylist(rows, schema=GROUND_SCHEMA)
        tmp = path.with_suffix(path.suffix + ".reparse.tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)
        counts.update(row["grounding_status"] for row in rows)
    summary_path = args.grounding_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary["statuses"] = dict(counts)
    summary["ground_fail"] = sum(
        count for status, count in counts.items() if status in {"GROUND_FAIL", "PARSE_ERROR"}
    )
    summary["reparsed_from_saved_raw"] = True
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
