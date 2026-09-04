#!/usr/bin/env python3
"""Re-apply deterministic ScaleEdit post-policy without rerunning the MLLM."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scaleedit.policy import apply_task_post_policy, grounding_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.input_dir == args.output_dir:
        raise ValueError("use a separate output directory to preserve raw MLLM grounding")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"rows": 0, "overrides": 0, "modes": Counter(), "statuses": Counter()}
    for input_path in sorted(args.input_dir.glob("part-*.parquet")):
        table = pq.read_table(input_path)
        rows = table.to_pylist()
        for row in rows:
            payload = json.loads(row["ground_json"])
            updated = apply_task_post_policy(
                row["final_task"], row["final_instruction"], payload
            )
            summary["overrides"] += int(updated is not payload)
            status = grounding_status(updated)
            row["ground_json"] = json.dumps(updated, ensure_ascii=False)
            row["grounding_status"] = status
            good = status not in {"PARSE_ERROR", "RUNTIME_ERROR", "GROUND_FAIL"}
            row["ground_parse_ok"] = bool(row["ground_parse_ok"] and good)
            row["qc_flag"] = "OK" if row["ground_parse_ok"] else "GROUND_FAIL"
            summary["modes"][str(updated.get("mask_mode", "unresolved"))] += 1
            summary["statuses"][status] += 1
            summary["rows"] += 1
        output_path = args.output_dir / input_path.name
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema),
            tmp_path,
            compression=args.compression,
        )
        tmp_path.replace(output_path)

    serializable = {
        **summary,
        "modes": dict(sorted(summary["modes"].items())),
        "statuses": dict(sorted(summary["statuses"].items())),
    }
    (args.output_dir / "post_policy_summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
