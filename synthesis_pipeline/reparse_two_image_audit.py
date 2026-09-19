"""Re-evaluate saved VLM JSON with the current two-image audit parser, without inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from synthesis_pipeline.audit_edit_pairs import load_jsonl, normalized_task_type, write_jsonl
from synthesis_pipeline.audit_edit_pairs_v2 import (
    AUDIT_VERSION,
    apply_low_change_veto,
    normalize_result,
    parse_audit_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--input-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--low-change-veto-threshold", type=float, default=0.0)
    parser.add_argument("--attribute-low-change-veto-threshold", type=float, default=0.0)
    args = parser.parse_args()
    if not 0 <= args.low_change_veto_threshold <= 1:
        raise ValueError("--low-change-veto-threshold must be within [0, 1]")
    if not 0 <= args.attribute_low_change_veto_threshold <= 1:
        raise ValueError("--attribute-low-change-veto-threshold must be within [0, 1]")
    summary = json.loads(args.input_summary.read_text(encoding="utf-8"))
    rows = load_jsonl(args.input_jsonl)
    for row in rows:
        parsed = normalize_result(
            parse_audit_json(str(row.get("raw_response", ""))),
            normalized_task_type(row),
            require_checklist=summary.get("prompt_variant") == "checklist",
            require_artifact_severity=summary.get("prompt_variant") == "legacy_severity",
            require_edited_area=summary.get("prompt_variant") in {"evidence_gate", "conservative_gate"},
        )
        parsed = apply_low_change_veto(
            parsed, normalized_task_type(row), row["locality_metrics"],
            args.low_change_veto_threshold,
            args.attribute_low_change_veto_threshold,
        )
        row["audit"] = parsed
        row["quality"] = parsed["quality"] if parsed else "parse_error"
        row["rewrite_candidate"] = parsed.get("rewrite_candidate") if parsed else None
        row["salvage_status"] = parsed.get("salvage_status") if parsed else "none"
    summary["audit_version"] = AUDIT_VERSION
    summary["low_change_veto_threshold"] = args.low_change_veto_threshold
    summary["attribute_low_change_veto_threshold"] = args.attribute_low_change_veto_threshold
    summary["low_change_veto_count"] = sum(
        bool((row.get("audit") or {}).get("metric_veto")) for row in rows
    )
    summary["quality_counts"] = dict(Counter(row["quality"] for row in rows))
    summary["visual_quality_counts"] = dict(Counter(
        (row.get("audit") or {}).get("visual_quality", "parse_error")
        for row in rows
    ))
    summary["salvage_candidate_count"] = sum(
        row["salvage_status"] == "manual_review_candidate" for row in rows
    )
    summary["reparsed_saved_model_outputs"] = True
    summary["timing_source"] = str(args.input_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "edit_audit.jsonl", rows)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "cases": len(rows),
        "quality_counts": summary["quality_counts"],
        "reparsed_saved_model_outputs": True,
    }, indent=2))


if __name__ == "__main__":
    main()
