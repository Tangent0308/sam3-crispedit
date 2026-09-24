#!/usr/bin/env python3
"""Choose double-PASS examples and both skip controls without copying raw images."""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--difficulty-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expand-selection", type=Path, help="Keep existing regression cases and sample new shards")
    parser.add_argument("--extra-per-type", type=int, default=16, help="Additional PASS rows per type; multiple of 8")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--fresh-per-type", type=int, default=0,
                        help="Sample one PASS per previously unused shard for independent validation")
    parser.add_argument("--exclude-selection", type=Path, action="append", default=[],
                        help="Exclude all shards named in a previous selection; repeatable")
    args = parser.parse_args()
    cases = []
    rng = random.Random(args.seed)
    if args.fresh_per_type:
        if args.fresh_per_type < 0 or args.expand_selection:
            parser.error('--fresh-per-type must be positive and cannot be combined with expansion')
        excluded = set()
        for selection in args.exclude_selection:
            excluded.update(c['shard'] for c in json.loads(selection.read_text())['cases'])
        for kind in ['add', 'color', 'motion change', 'remove', 'replace']:
            paths = sorted(p for p in args.difficulty_dir.glob(f'{kind}_*.parquet') if p.name not in excluded)
            rng.shuffle(paths)
            selected = 0
            for path in paths:
                passes = [r for r in pq.read_table(path).to_pylist() if r['scene_decision'] == 'PASS']
                if not passes:
                    continue
                row = rng.choice(passes)
                quality = {r['row_idx']: r for r in pq.read_table(args.quality_dir/path.name).to_pylist()}
                if quality[row['row_idx']]['prefilter_verdict'] != 'PASS':
                    raise ValueError(f'inconsistent double PASS: {path}:{row["row_idx"]}')
                cases.append({'shard': path.name, 'row_idx': row['row_idx'],
                              'expected_selection': 'SELECTED', 'group': 'fresh'})
                selected += 1
                if selected == args.fresh_per_type:
                    break
            if selected != args.fresh_per_type:
                raise ValueError(f'not enough unused PASS shards: {kind}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'seed': args.seed, 'excluded_shards': sorted(excluded),
                                         'cases': cases}, indent=2)+'\n')
        print(f'Selected {len(cases)} PASS cases from previously unused shards')
        return
    if args.expand_selection:
        if args.extra_per_type < 8 or args.extra_per_type % 8:
            parser.error('--extra-per-type must be a positive multiple of 8')
        cases = json.loads(args.expand_selection.read_text())["cases"]
        previous_count = len(cases)
        excluded = {item["shard"] for item in cases}
        for kind in ["add", "color", "motion change", "remove", "replace"]:
            paths = sorted(p for p in args.difficulty_dir.glob(f"{kind}_*.parquet") if p.name not in excluded)
            rng.shuffle(paths)
            groups = {"original": 0, "additional": 0}
            total = 0
            for path in paths:
                passes = [r for r in pq.read_table(path).to_pylist() if r["scene_decision"] == "PASS"]
                if len(passes) < 4:
                    continue
                group = "original" if passes[0]["source_prefilter_run_id"] == "pair_prefilter_20260916_041550" else "additional"
                if kind != "motion change" and groups[group] >= args.extra_per_type // 2:
                    continue
                quality = {r["row_idx"]: r for r in pq.read_table(args.quality_dir/path.name).to_pylist()}
                for row in rng.sample(passes, 4):
                    if quality[row["row_idx"]]["prefilter_verdict"] != "PASS":
                        raise ValueError(f"inconsistent double PASS: {path}:{row['row_idx']}")
                    cases.append({"shard":path.name,"row_idx":row["row_idx"],
                                  "expected_selection":"SELECTED","group":group})
                groups[group] += 4
                total += 4
                if total == args.extra_per_type:
                    break
            if total != args.extra_per_type:
                raise ValueError(f"not enough expansion cases: {kind}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"seed":args.seed,"cases":cases}, indent=2)+"\n")
        selected = [c for c in cases if c["expected_selection"] == "SELECTED"]
        args.output.with_name("selected.json").write_text(json.dumps({"cases":selected}, indent=2)+"\n")
        new_cases = cases[previous_count:]
        args.output.with_name('new_selected.json').write_text(json.dumps({'cases':new_cases},indent=2)+'\n')
        # Choose before inference: one randomly sampled row from each new shard,
        # not cases cherry-picked by their eventual mask quality or QC status.
        review, seen_shards = [], set()
        for case in new_cases:
            if case['shard'] not in seen_shards:
                review.append(case)
                seen_shards.add(case['shard'])
        args.output.with_name('review_selected.json').write_text(json.dumps({'seed':args.seed,'cases':review},indent=2)+'\n')
        print(f"Selected {len(selected)} double PASS + {len(cases)-len(selected)} skip controls")
        return
    for kind in ["add", "color", "motion change", "remove", "replace"]:
        paths = sorted(args.difficulty_dir.glob(f"{kind}_*.parquet"))
        rng.shuffle(paths)
        groups = set()
        for path in paths:
            rows = pq.read_table(path).to_pylist()
            passes = [r for r in rows if r["scene_decision"] == "PASS"]
            drops = [r for r in rows if r["scene_decision"] == "DROP"]
            if len(passes) < 2 or not drops:
                continue
            group = "original" if passes[0]["source_prefilter_run_id"] == "pair_prefilter_20260916_041550" else "additional"
            # Motion has no newly downloaded shards, so use two original shards.
            if kind == "motion change": group = path.name
            if group in groups:
                continue
            quality = pq.read_table(args.quality_dir / path.name).to_pylist()
            qdrops = [r for r in quality if r["prefilter_verdict"] != "PASS"]
            if not qdrops:
                continue
            for row, expected in [(passes[0], "SELECTED"), (passes[1], "SELECTED"),
                                  (drops[0], "SCENE_DROP"), (qdrops[0], "QUALITY_DROP")]:
                cases.append({"shard": path.name, "row_idx": row["row_idx"],
                              "expected_selection": expected, "group": group})
            groups.add(group)
            if len(groups) == 2:
                break
        if len(groups) != 2:
            raise ValueError(f"insufficient smoke cases: {kind}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed":args.seed,"cases":cases}, indent=2) + "\n")
    print(f"Selected {len(cases)} rows: 20 double PASS + 20 skip controls -> {args.output}")


if __name__ == "__main__":
    main()
