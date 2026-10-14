#!/usr/bin/env python3
"""Deterministically shard a selected grouped-input JSONL for multi-node runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.world_size < 1:
        parser.error("--world-size must be positive")
    rows = [json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("selection is empty")
    shards = [[] for _ in range(args.world_size)]
    # The selector already enforces quotas before shuffling. Round-robin keeps
    # each rank close to the same mix without changing group order within a
    # shard, which makes retries and audit logs deterministic.
    for index, row in enumerate(rows):
        shards[index % args.world_size].append(row)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"selection": str(args.selection.resolve()), "world_size": args.world_size,
                "rows": len(rows), "shards": []}
    for rank, shard in enumerate(shards):
        path = args.output_root / f"selected_rank{rank}.jsonl"
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in shard), encoding="utf-8")
        temp.replace(path)
        manifest["shards"].append({"rank": rank, "path": str(path), "rows": len(shard)})
    (args.output_root / "shards.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
