#!/usr/bin/env python3
"""Recheck fast multi-target capacity selections with exact RLE adjacency.

The fast selector is deliberately bbox-only so large SA-1B manifests can be
scanned quickly.  This pass streams its oversampled rows, recomputes the
preselected adjacent pair from the RLE masks, and emits the requested target
count with the 20/30/25/25 target-size quotas and a two-groups-per-source cap.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sa1b_pipeline.geometry import load_mask
from sa1b_pipeline.group_geometry import gap_pixels
from sa1b_pipeline.group_selection import _local_pair_view, _short_side, _target_sizes


def _exact_pair(row: dict) -> dict | None:
    pairs = row.get("adjacent_pairs") or []
    if not pairs:
        return None
    ids = [str(x) for x in pairs[0].get("object_ids", [])]
    if len(ids) != 2:
        return None
    objects = {str(o.get("object_id")): o for o in row.get("objects", [])}
    a, b = (objects.get(ids[0]), objects.get(ids[1]))
    if a is None or b is None:
        return None
    size = row.get("image_size") or []
    if len(size) != 2:
        return None
    try:
        width, height = int(size[0]), int(size[1])
        ma, mb = load_mask(a, (width, height)), load_mask(b, (width, height))
        short = min(_short_side(a), _short_side(b))
        local_a, local_b = _local_pair_view(a, b, ma, mb, short)
        gap = gap_pixels(local_a, local_b)
        overlap = float((local_a & local_b).sum()) / max(1, int(local_b.sum()))
        eligible = gap <= 0.1 * max(1.0, short) and overlap <= 0.10
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return None
    evidence = dict(pairs[0])
    evidence.update({"eligible": bool(eligible), "gap_pixels": float(gap),
                     "threshold_pixels": 0.1 * max(1.0, short),
                     "support_a_over_source_b": overlap,
                     "geometry_exact": True,
                     "geometry_method": "rle_local_adaptive_gap"})
    return evidence if eligible else None


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
            evidence = _exact_pair(row)
            if evidence is None:
                continue
            exact_pass += 1
            row = dict(row)
            row["adjacent_pairs"] = [evidence]
            accepted.append(row)
            got[n] += 1
            sources[source] += 1
            if checked % 100 == 0:
                print(json.dumps({"checked": checked, "exact_pass": exact_pass,
                                  "accepted": len(accepted), "quota": dict(need)},
                                 ensure_ascii=False), flush=True)
    if any(got[n] != need[n] for n in need):
        raise RuntimeError(f"exact adjacency recheck shortage: got={dict(got)} need={need}")
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
