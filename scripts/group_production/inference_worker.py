#!/usr/bin/env python3
"""Persistent single-GPU inference worker for committed multi-target chains."""
from pathlib import Path
import argparse
import json
import sys
import time
import traceback
import os
import signal
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sa1b_pipeline.production_state import atomic_json


def watch_parent(owner_pid):
    # Each service owns a fresh process group. A killed coordinator must not
    # leave GPU model descendants running and block the next resume.
    while True:
        if os.getppid() != owner_pid:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            return
        time.sleep(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--role', choices=['editor', 'audit'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--dtype', default='bf16')
    p.add_argument('--owner-pid', type=int, required=True)
    args = p.parse_args(); root = args.directory
    threading.Thread(target=watch_parent, args=(args.owner_pid,), daemon=True).start()
    if args.role == 'editor':
        from sa1b_pipeline.editor import DiffusersQwenImageEditor
        model = DiffusersQwenImageEditor(args.model, steps=40, dtype=args.dtype, crop_instruction=False)
    else:
        from sa1b_pipeline.audit_backend import VLLMAuditor
        model = VLLMAuditor(args.model, batch_size=args.batch_size, max_tokens=1000, check_operation=False)
    atomic_json(root/'ready.json', {'time': time.time(), 'role': args.role})
    handled = set()
    try:
        while not (root/'stop').exists():
            pending = sorted(p for p in (root/'requests').glob('*.json') if p.name not in handled)
            if not pending:
                time.sleep(.05); continue
            if args.role == 'audit' and len(pending) < args.batch_size:
                time.sleep(.1)
                pending = sorted(p for p in (root/'requests').glob('*.json') if p.name not in handled)
            batch = pending[:args.batch_size if args.role == 'audit' else 1]
            start = time.monotonic()
            try:
                requests = [json.loads(p.read_text()) for p in batch]
                if args.role == 'editor':
                    item = requests[0]; output = model.generate(item['source'], item['unit'], seed=item['seed'])
                    path = root/'responses'/f'{batch[0].stem}.png'; output.save(path)
                    results = [{'image': str(path), 'context': model.last_generation_context}]
                else:
                    results = model.call_batch(requests)
                if len(results) != len(batch):
                    raise RuntimeError('incomplete service batch')
                for path, result in zip(batch, results):
                    atomic_json(root/'responses'/path.name, {'result': result})
            except Exception as exc:
                traceback.print_exc()
                for path in batch:
                    atomic_json(root/'responses'/path.name, {'error': str(exc)})
                raise  # An infrastructure failure is not an edit rejection.
            handled.update(p.name for p in batch)
            print(json.dumps({'role': args.role, 'batch': len(batch), 'seconds': time.monotonic()-start,
                              'completed': len(handled)}), flush=True)
    finally:
        if args.role == 'editor':
            model.pipe.close()
        else:
            model.close()


if __name__ == '__main__':
    main()
