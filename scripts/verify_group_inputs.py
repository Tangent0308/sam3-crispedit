#!/usr/bin/env python3
"""Verify grouped train/eval manifests before planning or publication.

The verifier checks group counts, twin operation balance, benchmark exclusion,
50k source exclusion for eval, reservation overlap, and duplicate eval sources.
It is intentionally independent of model runtimes and writes a JSON report.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
import sys

# Keep direct invocation from any working directory self-contained.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sa1b_pipeline.source_isolation import normalize_source_id


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        return [value] if isinstance(value, dict) else []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keys(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    # Selection manifests wrap the source under candidate; final manifests put
    # source_id/source_image at group level. Support both without guessing IDs.
    source = row.get("candidate") or row.get("source") or row
    ids: set[str] = set()
    hashes: set[str] = set()
    if isinstance(source, dict):
        for key in ("source_id", "image_id", "family_id", "id"):
            if source.get(key):
                ids.add(normalize_source_id(source[key]))
        for key in ("image_sha256", "sha256", "image_hash"):
            if source.get(key):
                hashes.add(str(source[key]))
    return ids, hashes


def _manifest_keys(paths: list[Path]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    for path in paths:
        for row in _rows(path):
            row_ids, row_hashes = _keys(row)
            ids.update(row_ids)
            hashes.update(row_hashes)
    return ids, hashes


def _frozen_keys(root: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    for path in sorted((root / "inputs").glob("node*.jsonl")):
        for row in _rows(path):
            if row.get("source_id"):
                ids.add(normalize_source_id(row["source_id"]))
    hashes: set[str] = set()
    index = root / "source_cache" / "index.json"
    if index.is_file():
        value = json.loads(index.read_text(encoding="utf-8"))
        for record in value.values():
            if normalize_source_id(record.get("source_id")) in ids and record.get("sha256"):
                hashes.add(str(record["sha256"]))
    return ids, hashes


def _benchmark_keys(path: Path) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    for row in _rows(path):
        source = row.get("source", row)
        row_ids, row_hashes = _keys({"source": source})
        ids.update(row_ids)
        hashes.update(row_hashes)
    return ids, hashes


def _reservation(path: Path) -> tuple[set[str], set[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return ({normalize_source_id(item) for item in value.get("reserved_ids", [])},
            {str(item) for item in value.get("reserved_image_sha256", [])})


def _check_counts(rows: list[dict[str, Any]], *, kind: str, expected: int) -> dict[str, Any]:
    if len(rows) != expected:
        raise ValueError(f"{kind}: expected {expected} rows, found {len(rows)}")
    groups = []
    operations: Counter[str] = Counter()
    member_counts: Counter[str] = Counter()
    for row in rows:
        gid = row.get("twin_id") or row.get("multi_id") or row.get("group_id")
        if not gid and row.get("candidate"):
            candidate = row["candidate"]
            source_id = candidate.get("source_id", "")
            object_ids = [str(obj.get("object_id", "")) for obj in row.get("objects", [])]
            op_key = row.get("operation") or "-".join(row.get("operations", []))
            gid = f"selection:{source_id}:{','.join(object_ids)}:{op_key}"
        if not gid:
            raise ValueError(f"{kind}: missing group id")
        groups.append(str(gid))
        if kind == "twins":
            op = str(row.get("operation", ""))
            if op not in {"add", "replace", "remove", "attribute"}:
                raise ValueError(f"{kind}: invalid operation {op!r}")
            operations[op] += 1
            member_counts[str(len(row.get("members") or row.get("objects", [])))] += 1
        else:
            units = row.get("units") or row.get("objects", [])
            n = len(units)
            if n not in {2, 3, 4, 5}:
                raise ValueError(f"{kind}: invalid target count for {gid}")
            if row.get("operations") and not row.get("units"):
                ops = [str(op) for op in row["operations"]]
            else:
                ops = [str(unit.get("operation") or unit.get("instruction_spec", {}).get("operation", ""))
                       for unit in units]
            if n <= 4 and len(set(ops)) != n:
                raise ValueError(f"{kind}: repeated operation in {gid}: {ops}")
            if n == 5 and set(ops) != {"add", "replace", "remove", "attribute"}:
                raise ValueError(f"{kind}: five-target operation contract failed for {gid}: {ops}")
            operations.update(ops)
            member_counts[str(n)] += 1
    if len(set(groups)) != len(groups):
        raise ValueError(f"{kind}: duplicate group id")
    if kind == "twins" and set(operations) != {"add", "replace", "remove", "attribute"}:
        raise ValueError(f"{kind}: missing operation quota: {dict(operations)}")
    return {"groups": len(rows), "operation_counts": dict(operations), "member_counts": dict(member_counts)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--benchmark-exclusion", type=Path, required=True)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--twin-train", type=Path, required=True)
    parser.add_argument("--twin-eval", type=Path, required=True)
    parser.add_argument("--multi-train", type=Path, required=True)
    parser.add_argument("--multi-eval", type=Path, required=True)
    parser.add_argument("--twin-train-count", type=int, default=5000)
    parser.add_argument("--twin-eval-count", type=int, default=200)
    parser.add_argument("--multi-train-count", type=int, default=1500)
    parser.add_argument("--multi-eval-count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    paths = {"twin_train": args.twin_train, "twin_eval": args.twin_eval,
             "multi_train": args.multi_train, "multi_eval": args.multi_eval}
    rows = {name: _rows(path) for name, path in paths.items()}
    summaries = {
        "twin_train": _check_counts(rows["twin_train"], kind="twins", expected=args.twin_train_count),
        "twin_eval": _check_counts(rows["twin_eval"], kind="twins", expected=args.twin_eval_count),
        "multi_train": _check_counts(rows["multi_train"], kind="multi", expected=args.multi_train_count),
        "multi_eval": _check_counts(rows["multi_eval"], kind="multi", expected=args.multi_eval_count),
    }
    keys = {name: _manifest_keys([path]) for name, path in paths.items()}
    frozen_ids, frozen_hashes = _frozen_keys(args.frozen_root)
    benchmark_ids, benchmark_hashes = _benchmark_keys(args.benchmark_exclusion)
    reservation_ids, reservation_hashes = _reservation(args.reservation)
    train_ids = set().union(*(keys[name][0] for name in ("twin_train", "multi_train")), frozen_ids)
    train_hashes = set().union(*(keys[name][1] for name in ("twin_train", "multi_train")), frozen_hashes)
    eval_ids = set().union(*(keys[name][0] for name in ("twin_eval", "multi_eval")))
    eval_hashes = set().union(*(keys[name][1] for name in ("twin_eval", "multi_eval")))
    if eval_ids & train_ids or eval_hashes & train_hashes:
        raise ValueError("evaluation source leaked into training by ID or image hash")
    if eval_ids & benchmark_ids or eval_hashes & benchmark_hashes:
        raise ValueError("evaluation source overlaps benchmark exclusion")
    if any(keys[name][0] & benchmark_ids or keys[name][1] & benchmark_hashes for name in paths):
        raise ValueError("group source overlaps benchmark exclusion")
    dataset_summary = {}
    for name, (ids, hashes) in keys.items():
        dataset_summary[name] = {"source_ids": len(ids), "image_hashes": len(hashes),
                                 "benchmark_id_overlap": len(ids & benchmark_ids),
                                 "benchmark_hash_overlap": len(hashes & benchmark_hashes),
                                 "reservation_id_overlap": len(ids & reservation_ids),
                                 "reservation_hash_overlap": len(hashes & reservation_hashes)}
    report = {"status": "verified", "datasets": summaries, "source_sets": dataset_summary,
              "frozen_50k": {"source_ids": len(frozen_ids), "image_hashes": len(frozen_hashes)},
              "benchmark_exclusion": {"source_ids": len(benchmark_ids), "image_hashes": len(benchmark_hashes)},
              "reservation": {"source_ids": len(reservation_ids), "image_hashes": len(reservation_hashes)},
              "eval_vs_all_train": {"source_id_overlap": len(eval_ids & train_ids),
                                    "image_hash_overlap": len(eval_hashes & train_hashes)},
              "eval_internal_overlap": {"source_id_overlap": len(keys["twin_eval"][0] & keys["multi_eval"][0]),
                                        "image_hash_overlap": len(keys["twin_eval"][1] & keys["multi_eval"][1])},
              "inputs": {name: str(path.resolve()) for name, path in paths.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
