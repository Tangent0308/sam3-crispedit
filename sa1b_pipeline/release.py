"""Conservative post-generation release gate for multi-target edit cases.

Generation and semantic review are intentionally separate from release.  A
row reaches a release manifest only when its deterministic checks pass, its
manual review says ``manual_pass``, and (unless explicitly overridden for a
visual pilot) its semantic audit is a complete pass.  Cases are then kept or
dropped as a whole so every released source image still has several distinct
targets.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .io import read_jsonl, write_jsonl


def _semantic_pass(row: dict[str, Any]) -> bool:
    audit = row.get("semantic_audit")
    return (
        isinstance(audit, dict)
        and audit.get("decision") == "pass"
        and audit.get("target_done") is True
        and audit.get("non_target_preserved") is True
        and audit.get("natural") is True
    )


def release_rows(
    rows: Iterable[dict[str, Any]],
    manual_review: dict[tuple[str, str], dict[str, Any]],
    *,
    min_targets: int = 3,
    max_targets: int = 5,
    require_semantic: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return rows that meet the row and per-source release contract.

    ``require_semantic=False`` is for a clearly labelled visual pilot when a
    semantic model cannot run.  It still requires deterministic checks and an
    explicit manual pass; it must not be used as a silent fallback.
    """
    if min_targets < 1 or max_targets < min_targets:
        raise ValueError("invalid target-count bounds")
    rows = list(rows)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("case_id")), str(row.get("unit_id")))
        review = manual_review.get(key, {})
        reasons: list[str] = []
        if row.get("decision") == "fail":
            reasons.append("deterministic_check_failed")
        if review.get("status") != "manual_pass":
            reasons.append("manual_review_not_pass")
        if require_semantic and not _semantic_pass(row):
            reasons.append("semantic_audit_not_pass")
        if reasons:
            rejected.append({"case_id": key[0], "unit_id": key[1], "reasons": reasons})
            continue
        value = dict(row)
        value["release_status"] = "released"
        accepted.append(value)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        by_case[str(row["case_id"])].append(row)
    released: list[dict[str, Any]] = []
    dropped_cases: dict[str, str] = {}
    for case_id, case_rows in sorted(by_case.items()):
        if len(case_rows) < min_targets:
            dropped_cases[case_id] = "fewer_than_min_targets_after_row_gate"
            continue
        if len(case_rows) > max_targets:
            dropped_cases[case_id] = "more_than_max_targets_after_row_gate"
            continue
        released.extend(sorted(case_rows, key=lambda row: str(row["unit_id"])))

    report = {
        "input_rows": len(rows),
        "row_gate_pass": len(accepted),
        "released_rows": len(released),
        "rejected_rows": len(rows) - len(accepted),
        "input_cases": len({str(row.get("case_id")) for row in rows}),
        "released_cases": len({str(row.get("case_id")) for row in released}),
        "dropped_cases": dropped_cases,
        "min_targets": min_targets,
        "max_targets": max_targets,
        "require_semantic": require_semantic,
        "rejected": rejected,
    }
    return released, report


def _read_manual(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["case_id"]), str(row["unit_id"])): row
        for row in read_jsonl(path)
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-targets", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=5)
    parser.add_argument(
        "--allow-manual-without-semantic",
        action="store_true",
        help="visual-pilot mode; still requires deterministic and manual pass",
    )
    args = parser.parse_args(argv)
    released, report = release_rows(
        read_jsonl(args.generated),
        _read_manual(args.manual_review),
        min_targets=args.min_targets,
        max_targets=args.max_targets,
        require_semantic=not args.allow_manual_without_semantic,
    )
    write_jsonl(args.output, released)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": len(released), "cases": report["released_cases"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
