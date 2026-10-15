#!/usr/bin/env python3
"""Rebalance operation labels in an already selected multi-target shard.

Only the per-target ``operations`` arrays are changed.  Source candidates,
object IDs, exact adjacency evidence, n-target quotas, and split isolation are
left byte-for-byte intact.  The script is intended for the materialized
selection files produced before planning, and is safe to rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from sa1b_pipeline.group_selection import _cheap_operation_allowed, _image_size
from sa1b_pipeline.multi_operation_balance import balance_multi_operations


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path,
                        help="selected multi-target JSONL files to update")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for path in args.paths:
        rows = _read(path)
        before = Counter(op for row in rows for op in row.get("operations", []))
        report = balance_multi_operations(
            rows,
            image_size_for=lambda row: tuple(row.get("image_size") or _image_size(row["candidate"])),
            allowed=lambda obj, op, size: _cheap_operation_allowed(obj, op, size),
            seed=args.seed,
        )
        if not args.dry_run:
            _write_atomic(path, rows)
            summary_path = path.with_suffix(path.suffix + ".operation_balance.json")
            summary_path.write_text(
                json.dumps({"path": str(path), "before": dict(before), **report},
                           ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({"path": str(path), "before": dict(before), **report,
                          "written": not args.dry_run}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

