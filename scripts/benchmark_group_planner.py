#!/usr/bin/env python3
"""Replay frozen grouped planning inputs without editing production checkpoints."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.production_state import atomic_json, read_rows, write_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--batches', default='8,16')
    p.add_argument('--warmup', type=int, default=2)
    args = p.parse_args()
    batches = [int(x) for x in args.batches.split(',')]
    if not batches or min(batches) < 1 or args.warmup < 0:
        p.error('invalid batch sizes or warmup')
    root = args.output_root; root.mkdir(parents=True, exist_ok=True)
    if (root/'summary.json').exists():
        raise ValueError('use a fresh benchmark output directory')
    os.environ['SA1B_FORCE_DIRECT_VLLM'] = '1'
    # Runtime executables (including ninja) must come from this interpreter.
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
    from sa1b_pipeline.mllm import VLLMSelector
    from vllm.sampling_params import StructuredOutputsParams
    requests = read_rows(args.requests)
    start = time.perf_counter()
    model = VLLMSelector(args.model, tensor_parallel_size=args.tp,
                         batch_size=max(batches), max_new_tokens=1600)
    model.params.structured_outputs = StructuredOutputsParams(json_object=True)
    summary = {'model': args.model, 'tp': args.tp, 'load_seconds': time.perf_counter()-start,
               'requests': str(args.requests), 'variants': [], 'warmup_requests': min(args.warmup,len(requests))}
    try:
        start = time.perf_counter()
        if args.warmup:
            model.generate_multimage([(r['views'], r['prompt']) for r in requests[:args.warmup]])
        summary['warmup_seconds'] = time.perf_counter()-start
        for batch_size in batches:
            model.batch_size = batch_size
            rows = []; metrics = []; start = time.perf_counter()
            for offset in range(0, len(requests), batch_size):
                batch = requests[offset:offset+batch_size]
                outputs = model.generate_multimage([(r['views'], r['prompt']) for r in batch])
                if len(outputs) != len(batch):
                    raise RuntimeError('incomplete planner batch')
                metrics.append(model.last_batch_metrics)
                for request, raw in zip(batch, outputs):
                    rows.append({'group_id': request['group_id'], 'operation': request['operation'], 'raw': raw})
                write_rows(root/f'batch{batch_size}.jsonl', rows)
                print(json.dumps({'batch_size':batch_size, 'completed':len(rows), 'total':len(requests),
                                  'last_batch':metrics[-1]}), flush=True)
            summary['variants'].append({'batch_size':batch_size,'seconds':time.perf_counter()-start,
                                        'groups':len(rows),'metrics':metrics})
            atomic_json(root/'summary.json', summary)
    finally:
        model.close()


if __name__ == '__main__':
    main()
