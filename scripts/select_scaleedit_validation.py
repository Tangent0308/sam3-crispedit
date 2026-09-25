#!/usr/bin/env python3
"""Reproducible old/new, random/reference-enriched development sample (not an unbiased estimate)."""
import argparse
import json
from pathlib import Path
import random
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(42)
    cases = []
    cue = re.compile(r'\b(left|right|middle|center|central|second|third|between|behind|beside|closest|furthest|nearest|leftmost|rightmost)\b', re.I)
    for pattern in ('part-*.parquet', 'expand-*.parquet'):
        paths = sorted(args.input_dir.glob(pattern))
        rng.shuffle(paths)
        for path in paths[:16]:
            rows = pq.read_table(path, columns=['sample_id', 'final_task', 'final_instruction']).to_pylist()
            candidates = list(range(len(rows)))
            rng.shuffle(candidates)
            references = [i for i in candidates if cue.search(rows[i]['final_instruction'])][:8]
            selected = references + [i for i in candidates if i not in references][:16-len(references)]
            for i in selected:
                cases.append(dict(shard=path.name, row_idx=i, sample_id=rows[i]['sample_id'],
                                  final_task=rows[i]['final_task'], final_instruction=rows[i]['final_instruction'],
                                  stratum='reference_enriched' if i in references else 'random'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(seed=42, purpose='development; not population-rate estimation', cases=cases), indent=2))
    print('Selected', len(cases), 'pairs:', args.output)


if __name__ == '__main__':
    main()
