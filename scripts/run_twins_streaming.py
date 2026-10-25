#!/usr/bin/env python3
"""Local twin validation: resident planning, blind/audit, and editor GPU pools."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sa1b_pipeline.inference_services import ServicePool, RemoteSelector
from sa1b_pipeline.production_state import atomic_json, immutable_json, read_rows, write_rows, file_hash
from sa1b_pipeline.twin_streaming import process_twin


def device_ids(value):
    parts = [x.strip() for x in value.split(',') if x.strip()]
    if not parts or len(parts) != len(set(parts)):
        raise ValueError('GPU lists must be nonempty and distinct')
    return parts


def main(argv=None, *, services=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--selected-manifest', type=Path)
    source.add_argument('--planned-manifest', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--planner-model')
    p.add_argument('--twin-planning-style', choices=['concise-27b-v5','concise-27b-v8','content-v1'], default='concise-27b-v5')
    p.add_argument('--per-mask-planning', action='store_true')
    p.add_argument('--planner-thinking-budget', type=int, default=512)
    p.add_argument('--audit-model', required=True)
    p.add_argument('--editor-model', required=True)
    p.add_argument('--mllm-python', required=True)
    p.add_argument('--editor-python', required=True)
    p.add_argument('--planner-devices', default='0')
    p.add_argument('--editor-devices', default='1,2,3,4,5,6')
    p.add_argument('--audit-devices', default='7')
    p.add_argument('--planner-batch-size', type=int, default=16)
    p.add_argument('--audit-batch-size', type=int, default=8)
    p.add_argument('--groups-in-flight', type=int, default=32)
    p.add_argument('--local-generation-reference', action='store_true')
    p.add_argument('--site-model', required=True)
    p.add_argument('--reservation', type=Path)
    args = p.parse_args(argv)
    if min(args.planner_batch_size, args.audit_batch_size, args.groups_in_flight) < 1:
        p.error('batch sizes and concurrency must be positive')
    if args.selected_manifest and not args.planner_model:
        p.error('selected inputs require --planner-model')
    if args.per_mask_planning and args.twin_planning_style != 'content-v1':
        p.error('per-mask planning requires content-v1')
    planner_options = ({'max_new_tokens':2300,'thinking_token_budget':args.planner_thinking_budget}
                       if args.per_mask_planning else {})
    planner_gpus = device_ids(args.planner_devices) if args.selected_manifest else []
    editor_gpus, audit_gpus = device_ids(args.editor_devices), device_ids(args.audit_devices)
    all_gpus = planner_gpus+editor_gpus+audit_gpus
    if len(all_gpus) != len(set(all_gpus)):
        p.error('planner, editor, and audit GPU pools must be disjoint')
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('unset CUDA_VISIBLE_DEVICES; assign GPUs using the explicit device options')
    root = args.output_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    os.environ['SA1B_SITE_MODEL'] = args.site_model
    os.environ.setdefault('SA1B_MM_ENCODER_ATTN_BACKEND', 'TORCH_SDPA')
    os.environ.setdefault('DIFFUSION_ATTENTION_BACKEND', 'TORCH_SDPA')
    from scripts.run_group_production_4node import owner_lock
    from scripts import run_group_pipeline
    def stop(signum, frame):
        raise KeyboardInterrupt(f'received signal {signum}')
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, stop)
    with owner_lock(root/'streaming.lock'):
        manifest = args.selected_manifest or args.planned_manifest
        rows = read_rows(manifest)
        ids = [run_group_pipeline._group_id('twins', g) if args.selected_manifest else g['twin_id'] for g in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('duplicate input group IDs')
        immutable_json(root/'run.json', {
            'version': 'twins-stream-v1', 'manifest': str(manifest.resolve()), 'input_sha256': file_hash(manifest),
            'planner_model': args.planner_model if args.selected_manifest else None,
            'twin_planning_style':args.twin_planning_style,
            **({'per_mask_planning':True,'planner_options':planner_options} if args.per_mask_planning else {}),
            'editor_model': args.editor_model, 'audit_model': args.audit_model, 'site_model': args.site_model,
            'steps': 40, 'instruction_protocol': '2.0', 'mask_semantics': '2.0',
            'planner_batch_size': args.planner_batch_size,
            'local_generation_reference':args.local_generation_reference,
            'reservation_sha256': file_hash(args.reservation) if args.reservation else None})
        # Service queues and model workers are local to this coordinator. All
        # durable results live under output-root, independently of service UUIDs.
        editors = ServicePool(python=args.editor_python, role='twins', devices=editor_gpus,
                              model=args.editor_model, directory=root/'services/editor')
        audits = ServicePool(python=args.mllm_python, role='audit', devices=audit_gpus,
                             model=args.audit_model, directory=root/'services/audit', batch_size=args.audit_batch_size)
        planners = ServicePool(python=args.mllm_python, role='planner', devices=[','.join(planner_gpus)],
                               model=args.planner_model, directory=root/'services/planner',
                               batch_size=args.planner_batch_size, planner_options=planner_options) if planner_gpus else None
        if services is not None:
            editors, audits, planners = services
            if (editors.model != args.editor_model or audits.model != args.audit_model
                    or (args.selected_manifest and (planners is None or planners.model != args.planner_model
                                                   or planners.batch_size != args.planner_batch_size
                                                   or getattr(planners,'planner_options',{}) != planner_options))):
                raise ValueError('resident services do not match the frozen run contract')
        pools = [editors, audits]+([planners] if planners else [])
        start = time.monotonic(); results = []; lock = threading.Lock(); planned_count = 0
        # Atomic checkpoints keep per-group progress visible while planning the
        # next microbatch; wall throughput never sums overlapping stage times.
        failures = []
        data_errors = (ValueError, KeyError, IndexError, AssertionError, ZeroDivisionError)
        def process(group):
            try:
                result = process_twin(group, root, editors, audits, local_reference=args.local_generation_reference)
            except data_errors as exc:
                # A data-dependent error of one group is recorded and the group left out (see the multi job).
                import traceback
                with lock:
                    failures.append(group['twin_id'])
                    with (root/'failed_groups.jsonl').open('a') as f:
                        f.write(json.dumps({'twin_id': group['twin_id'], 'error': repr(exc),
                                            'traceback': traceback.format_exc(), 'time': time.time()}) + '\n')
                    print(json.dumps({'group_failed': group['twin_id'], 'error': repr(exc)}), flush=True)
                    if len(failures) > max(3, math.ceil(.05 * max(1, len(rows)))):
                        raise RuntimeError(f'{len(failures)} twin groups failed with data errors; '
                                           f'see {root}/failed_groups.jsonl')
                return None
            with lock:
                results.append(result)
                atomic_json(root/'progress.json', {
                    'input_candidates': len(rows), 'planned': planned_count, 'completed': len(results),
                    'statuses': dict(Counter(x['status'] for x in results)),
                    'accepted_by_operation': dict(Counter(x['group']['operation'] for x in results if x['status']=='accepted')),
                    'session_seconds': time.monotonic()-start})
                print(json.dumps({'completed':len(results), 'status':result['status'],
                                  'group_id':group['twin_id'], 'seconds':time.monotonic()-start}), flush=True)
            return result
        def plans():
            nonlocal planned_count
            if not args.selected_manifest:
                planned_count = len(rows)
                yield from rows
                return
            for index in range(0, len(rows), args.planner_batch_size):
                chunk = rows[index:index+args.planner_batch_size]
                directory = root/'planning'/f'{index//args.planner_batch_size:05d}'
                selected = directory/'inputs/selected.jsonl'
                selected.parent.mkdir(parents=True, exist_ok=True)
                if selected.exists():
                    if read_rows(selected) != chunk:
                        raise ValueError('planning chunk changed on resume')
                else:
                    write_rows(selected, chunk)
                    write_rows(directory/'inputs/manifest.jsonl', [r['candidate'] for r in chunk])
                cmd = ['--kind','twins','--manifest',str(directory/'inputs/manifest.jsonl'),
                       '--output-root',str(directory),'--target',str(len(chunk)), '--quota-from-selection',
                       '--outline-planning-crops','--locate-planning-scene','--structured-planning-json',
                       '--skip-unused-planning-views',
                       '--twin-planning-style',args.twin_planning_style,
                       '--plan-only','--defer-blind','--site-model',args.site_model]
                if args.reservation:
                    cmd += ['--reservation', str(args.reservation)]
                if args.per_mask_planning:
                    cmd += ['--per-mask-planning','--planning-operation-retry','0','--planning-source-hints','none',
                            '--planner-model',args.planner_model,'--planner-batch-size',str(args.planner_batch_size),
                            '--planner-thinking-budget',str(args.planner_thinking_budget),'--max-new-tokens','2300']
                run_group_pipeline.main(cmd, planner_override=RemoteSelector(planners))
                ready = read_rows(directory/'planned.jsonl')
                planned_count += len(ready)
                yield from ready
        executor = ThreadPoolExecutor(max_workers=args.groups_in_flight)
        pending = set()
        try:
            # Cold starts overlap. A fully completed resume remains lazy and
            # validates checkpoints without loading any GPU model.
            if not (root/'summary.json').exists():
                for pool in pools: pool.start()
            for group in plans():
                pending.add(executor.submit(process, group))
                if len(pending) >= args.groups_in_flight:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done: f.result()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for f in done: f.result()
        except BaseException:
            for f in pending: f.cancel()
            import traceback
            traceback.print_exc()  # visible even when a concurrent job reports first
            if services is None:  # shared services belong to the caller (other jobs use them)
                for pool in pools: pool.close()
            raise
        finally:
            if services is None:
                for pool in pools: pool.close()
            executor.shutdown(wait=True, cancel_futures=True)
        results.sort(key=lambda x: x['group']['twin_id'])
        write_rows(root/'audited.jsonl', [x['group'] for x in results if x['status'] in ('accepted','audit_rejected')])
        write_rows(root/'accepted.jsonl', [x['group'] for x in results if x['status']=='accepted'])
        # Opt-in single-target salvage (audit service keep_singles); empty otherwise.
        write_rows(root/'singles.jsonl', [s for x in results for s in (x['group'].get('singles') or [])])
        atomic_json(root/'summary.json', {'candidates':len(rows), 'planned':planned_count,
                    'statuses':dict(Counter(x['status'] for x in results)),
                    'session_seconds':time.monotonic()-start,
                    'published':False, 'note':'Local validation only; final production quotas remain external.'})


if __name__ == '__main__':
    main()
