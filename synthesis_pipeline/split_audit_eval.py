"""Split edit pairs by source image for leakage-free audit prompt evaluation.

Both regions of a source remain together.  The split is stratified by source
subset and edit-type pair so dev and holdout have matching task distributions.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def split_rows(rows: list[dict], seed: int, dev_fraction: float) -> tuple[list[dict], list[dict]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must lie strictly between zero and one")
    sources: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (str(row["source_subset"]), int(row["parquet_row_index"]))
        sources[key].append(row)
    strata: dict[tuple[str, tuple[str, ...]], list[tuple[str, int]]] = defaultdict(list)
    for key, group in sources.items():
        kinds = tuple(sorted(str(row["task_type"]) for row in group))
        strata[(key[0], kinds)].append(key)
    rng = random.Random(seed)
    dev_keys: set[tuple[str, int]] = set()
    for keys in strata.values():
        ordered = sorted(keys)
        rng.shuffle(ordered)
        count = round(len(ordered) * dev_fraction)
        if len(ordered) > 1:
            count = min(len(ordered) - 1, max(1, count))
        dev_keys.update(ordered[:count])
    dev = [row for row in rows if (str(row["source_subset"]), int(row["parquet_row_index"])) in dev_keys]
    holdout = [row for row in rows if (str(row["source_subset"]), int(row["parquet_row_index"])) not in dev_keys]
    return dev, holdout


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--dev-fraction", type=float, default=0.6)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.annotations_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    dev, holdout = split_rows(rows, args.seed, args.dev_fraction)
    write_jsonl(args.output_dir / "dev_annotations.jsonl", dev)
    write_jsonl(args.output_dir / "holdout_annotations.jsonl", holdout)
    summary = {
        "seed": args.seed,
        "source_disjoint": not ({(r["source_subset"], r["parquet_row_index"]) for r in dev} & {(r["source_subset"], r["parquet_row_index"]) for r in holdout}),
        "dev": {"cases": len(dev), "types": dict(Counter(r["task_type"] for r in dev)), "subsets": dict(Counter(r["source_subset"] for r in dev))},
        "holdout": {"cases": len(holdout), "types": dict(Counter(r["task_type"] for r in holdout)), "subsets": dict(Counter(r["source_subset"] for r in holdout))},
    }
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
