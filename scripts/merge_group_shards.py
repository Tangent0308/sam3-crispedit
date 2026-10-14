#!/usr/bin/env python3
"""Merge and validate final JSONL files produced by grouped rank shards."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


OPS = ("add", "replace", "remove", "attribute")


def _group_id(row):
    return row.get("twin_id") or row.get("multi_id") or row.get("group_id")


def _source_id(row):
    return str(row.get("source_id") or row.get("source", {}).get("source_id") or "")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="root containing rank*/final/*.jsonl")
    parser.add_argument("--kind", choices=("twins", "multi"), required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    name = ("twins_" if args.kind == "twins" else "multi_") + args.split + ".jsonl"
    files = sorted(args.input_root.glob(f"rank*/final/{name}"))
    if not files:
        raise RuntimeError(f"no shard outputs found for {name} under {args.input_root}")
    rows = []
    seen_ids = set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            gid = _group_id(row)
            if not gid or gid in seen_ids:
                raise RuntimeError(f"missing or duplicate group id: {gid!r}")
            seen_ids.add(gid)
            rows.append(row)
    if len(rows) != args.target:
        raise RuntimeError(f"merged {len(rows)} groups, expected {args.target}")
    sources = Counter(_source_id(row) for row in rows)
    if any(count > 2 for count in sources.values()):
        raise RuntimeError("source has more than two groups: " + repr({k: v for k, v in sources.items() if v > 2}))
    operation_counts = Counter()
    member_counts = Counter()
    if args.kind == "twins":
        for row in rows:
            operation_counts[str(row.get("operation"))] += 1
            member_counts[len(row.get("members", []))] += 1
        expected = args.target // 4
        if any(operation_counts[op] not in {expected, expected + (1 if i < args.target % 4 else 0)}
               for i, op in enumerate(OPS)):
            raise RuntimeError(f"twin operation quotas are not balanced: {dict(operation_counts)}")
    else:
        for row in rows:
            units = row.get("units", [])
            n = len(units)
            if n not in (2, 3, 4, 5):
                raise RuntimeError(f"multi-target group has invalid unit count: {n}")
            operations = [str(unit.get("operation") or unit.get("instruction_spec", {}).get("operation", ""))
                          for unit in units]
            if n <= 4 and (len(set(operations)) != n or not set(operations).issubset(set(OPS))):
                raise RuntimeError(f"multi-target operations are not distinct valid types: {operations}")
            if n == 5 and (set(operations) != set(OPS) or len(set(operations)) != 4):
                raise RuntimeError(f"five-target group must contain four types and one repeat: {operations}")
            if not row.get("adjacent_pairs"):
                raise RuntimeError(f"multi-target group has no adjacent-pair evidence: {_group_id(row)}")
            member_counts[n] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temp.replace(args.output)
    summary = {"kind": args.kind, "split": args.split, "groups": len(rows),
               "source_count": len(sources), "max_groups_per_source": max(sources.values(), default=0),
               "operation_counts": dict(operation_counts), "member_counts": dict(member_counts),
               "inputs": [str(path) for path in files], "output": str(args.output)}
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
