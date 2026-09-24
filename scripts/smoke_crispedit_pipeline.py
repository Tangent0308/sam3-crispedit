#!/usr/bin/env python3
"""Validate all four pipeline ranks on one 8-GPU host with isolated sample copies."""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--selection-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--per-type', type=int, default=4)
    parser.add_argument('--python', default=sys.executable)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = args.output_dir / 'source'
    source.mkdir()
    selection = json.loads(args.selection_file.read_text())['cases']
    groups = defaultdict(list)
    for case in selection:
        kind = case['shard'].rsplit('_', 1)[0]
        if len(groups[kind]) < args.per_type:
            groups[kind].append(case)
    provenance = []
    for kind, cases in sorted(groups.items()):
        rows = []
        for case in cases:
            rows.append(pq.read_table(args.source_dir / case['shard']).slice(case['row_idx'], 1))
        name = kind + '_90000.parquet'
        pq.write_table(pa.concat_tables(rows), source / name)
        provenance += [{'shard': name, 'row_idx': i, 'original': case} for i, case in enumerate(cases)]
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    processes, handles = [], []
    try:
        for rank in range(4):
            log = (args.output_dir / f'coordinator.node{rank}.log').open('w')
            handles.append(log)
            command = [args.python, '-u', str(REPO / 'scripts/run_crispedit_pipeline.py'),
                       '--run-dir', str(args.output_dir / 'run'), '--source-dir', str(source),
                       '--quality-dir', str(args.output_dir / 'quality'),
                       '--scene-dir', str(args.output_dir / 'scene'),
                       '--python', args.python, '--sam-python', args.python,
                       '--rank', str(rank), '--nodes', '4', '--local-test',
                       '--devices', f'{rank*2},{rank*2+1}', '--grounding-batch-size', '4',
                       '--wait-seconds', '3600']
            processes.append(subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in processes]
        print(json.dumps({'exit_codes': codes, 'root': str(args.output_dir)}, indent=2), flush=True)
        if any(codes):
            raise SystemExit(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
