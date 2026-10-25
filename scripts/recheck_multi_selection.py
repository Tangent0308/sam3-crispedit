#!/usr/bin/env python3
"""Recheck fast multi-target selections with exact masks, without a distance gate.

Preserves target-size quotas and the two-groups-per-source cap. Source selectors
must be nonempty and disjoint; writable support overlaps are scheduled later.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sa1b_pipeline.geometry import load_mask
from sa1b_pipeline.group_selection import _target_sizes


def _exact_geometry(row: dict) -> dict | None:
    objects = row.get("objects", [])
    if not 2 <= len(objects) <= 5 or len({str(o.get("object_id")) for o in objects}) != len(objects):
        return None
    try:
        width, height = map(int, row["image_size"])
        occupied = None
        for obj in objects:
            mask = load_mask(obj, (width, height))
            if not mask.any() or (occupied is not None and (occupied & mask).any()):
                return None
            occupied = mask.copy() if occupied is None else occupied | mask
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return None
    return {"eligible": True, "distance_required": False, "geometry_exact": True,
            "geometry_method": "rle_source_mask_nonoverlap"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    args = parser.parse_args(argv)
    if args.target < 1:
        parser.error("--target must be positive")
    need = _target_sizes(args.target)
    got: Counter[int] = Counter()
    sources: Counter[str] = Counter()
    accepted = []
    checked = 0
    exact_pass = 0
    with args.input.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or all(got[n] >= need[n] for n in need):
                break
            row = json.loads(line)
            n = len(row.get("objects", []))
            if n not in need or got[n] >= need[n]:
                continue
            source = str(row.get("candidate", {}).get("source_id", ""))
            if sources[source] >= 2:
                continue
            checked += 1
            evidence = _exact_geometry(row)
            if evidence is None:
                continue
            exact_pass += 1
            row = dict(row)
            row["adjacent_pairs"] = []
            row["selection_geometry"] = evidence
            accepted.append(row)
            got[n] += 1
            sources[source] += 1
            if checked % 100 == 0:
                print(json.dumps({"checked": checked, "exact_pass": exact_pass,
                                  "accepted": len(accepted), "quota": dict(need)},
                                 ensure_ascii=False), flush=True)
    if any(got[n] != need[n] for n in need):
        raise RuntimeError(f"exact mask recheck shortage: got={dict(got)} need={need}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted),
                    encoding="utf-8")
    temp.replace(args.output)
    summary = {"requested": args.target, "selected": len(accepted),
               "checked": checked, "exact_pass": exact_pass,
               "member_counts": dict(sorted((str(k), v) for k, v in got.items())),
               "source_count": len(sources), "max_groups_per_source": max(sources.values()),
               "geometry_exact": True, "input": str(args.input.resolve()),
               "output": str(args.output.resolve())}
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
