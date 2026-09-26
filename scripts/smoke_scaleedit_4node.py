#!/usr/bin/env python3
"""Run real Qwen/SAM stages on eight GPUs as four local smoke-test ranks."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scaleedit.distributed import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=Path('/mnt/bn/strategy-mllm-train/user/tanyue/datasets/ScaleEdit-filtered-source'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attempt', default='initial')
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    selection = args.run_dir / 'smoke_selection.json'
    if not args.resume:
        if selection.exists():
            raise ValueError('Existing smoke run; use --resume or a new run directory')
        gallery = json.loads((REPO / 'scripts/scaleedit_review_cases.json').read_text())
        # Fixed examples cover both filter decisions, multi-instance masks, text and additions.
        cases = {(item['shard'], item['row_idx']) for group, items in gallery.items()
                 for item in (items[:8] if group == 'mask' else items)}
        atomic_json(selection, dict(cases=[dict(shard=s, row_idx=i) for s, i in sorted(cases)]))
    processes, handles = [], []
    try:
        for rank in range(4):
            handle = (args.run_dir / f'smoke.node{rank}.log').open('a')
            handles.append(handle)
            command = [sys.executable, '-u', str(REPO / 'scripts/run_scaleedit_4node.py'),
                       '--run-dir', str(args.run_dir), '--source-dir', str(args.source_dir),
                       '--selection-file', str(selection), '--rank', str(rank),
                       '--devices', f'{2*rank},{2*rank+1}', '--local-test', '--attempt', args.attempt]
            if args.resume:
                command.append('--resume')
            processes.append(subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Smoke rank failed: {[p.poll() for p in processes]}; see {args.run_dir}')
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError('A smoke rank failed')
        print((args.run_dir / 'reports/run_manifest.json').read_text())
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for h in handles:
            h.close()


if __name__ == '__main__':
    main()
