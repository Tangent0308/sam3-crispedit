"""Shared audit queue: enqueue frozen jobs, serve with multiple GPUs, export."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def prepare_job(job, panel_dir, *, check_operation=False):
    if 'images' in job:
        if check_operation:
            raise ValueError('type-aware audit requires a generated row, not a frozen old prompt')
        for path, expected in zip(job['images'], job['images_sha256']):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('frozen audit image changed: ' + path)
        return job
    from sa1b_pipeline.validate import _target_panel, audit_prompt, deterministic_check
    row = job['row']
    return {**job, 'images': [row['source_image'], row['edited_image'], _target_panel(row, panel_dir)],
            'prompt': audit_prompt(row['instruction'], operation=row['operation'] if check_operation else None),
            'check': deterministic_check(row)}


def worker(a, queue):
    import importlib.metadata

    from sa1b_pipeline.audit_backend import TransformersAuditor, VLLMAuditor
    from sa1b_pipeline.validate import (
        AUDIT_PROMPT,
        OPERATION_AUDIT_RULES,
        OPERATION_AUDIT_SCHEMA,
    )
    config = {'model': a.model, 'backend': a.backend, 'max_tokens': 512,
              'thumbnail': 1536, 'prompt_sha256': hashlib.sha256(AUDIT_PROMPT.encode()).hexdigest(),
              'vllm_version': importlib.metadata.version('vllm'),
              'transformers_version': importlib.metadata.version('transformers')}
    check_operation = getattr(a, 'check_operation', False)
    result_dir = getattr(a, 'result_dir', None)
    if check_operation:
        config['operation_audit_sha256'] = hashlib.sha256(OPERATION_AUDIT_RULES.encode()).hexdigest()
        config['operation_schema_sha256'] = hashlib.sha256(json.dumps(OPERATION_AUDIT_SCHEMA, sort_keys=True).encode()).hexdigest()
    if a.disable_input_cache:
        config['cache_inputs'] = False
    queue.bind_config(config)
    if queue.sealed() and not any(queue.counts().get(s, 0) for s in ['pending', 'running']):
        print('queue already complete; model not loaded')
        return
    cls = VLLMAuditor if a.backend == 'vllm' else TransformersAuditor
    start = time.perf_counter()
    model = cls(a.model, batch_size=a.batch_size, cache_inputs=not a.disable_input_cache,
                **({'check_operation': True} if check_operation else {}))
    loaded = time.perf_counter()
    batches = []
    idle_since = time.monotonic()
    try:
        while True:
            claimed = queue.claim(a.batch_size, min_count=a.min_batch_size,
                                  max_wait_seconds=a.max_batch_wait)
            if not claimed:
                if queue.sealed() and not queue.counts().get('running', 0):
                    break
                if time.monotonic()-idle_since > a.idle_timeout:
                    raise TimeoutError('audit queue idle timeout; results retained')
                time.sleep(.2)
                continue
            # The queue atomically waits for a minimum batch or oldest-job
            # deadline; sealing flushes the tail immediately.
            before = time.perf_counter()
            started_unix = time.time()
            try:
                jobs = [prepare_job(job, (result_dir.parent if result_dir else a.db.parent)/'panels', check_operation=check_operation) for job, _ in claimed]
                outputs = model.call_batch(jobs)
                if [o['job_id'] for o in outputs] != [j['job_id'] for j in jobs]:
                    raise RuntimeError('backend result IDs do not match input batch')
                elapsed = time.perf_counter()-before
                for job, (_, lease), result in zip(jobs, claimed, outputs):
                    audit = result['semantic_audit']
                    ok = audit.get('decision') == 'pass' and all(audit.get(k) is True for k in
                        ['target_done', 'non_target_preserved', 'natural'])
                    if check_operation:
                        ok = ok and audit.get('operation_correct') is True
                    final = {**job['row'], **job['check'], **result,
                             'decision': 'pass' if ok and job['check']['decision'] != 'fail' else 'fail',
                             'audit_seconds': elapsed/len(jobs), 'audit_seconds_kind': 'amortized_batch_wall',
                             'audit_batch_seconds': elapsed, 'audit_batch_size': len(jobs),
                             'audit_completed_unix': time.time(),
                             'semantic_audit_backend': a.backend, 'audit_runtime': config}
                    if check_operation:
                        final['operation_audit_required'] = True
                    if result_dir:
                        from sa1b_pipeline.production_state import atomic_json, digest
                        # Persist before queue.finish: loss of a local DB cannot
                        # discard an audit or select a different retry verdict.
                        atomic_json(result_dir/(hashlib.sha256(job['job_id'].encode()).hexdigest()+'.json'),
                                    {'payload_sha256': digest({'job_id': job['job_id'], 'row': job['row']}),
                                     'result': final})
                    queue.finish(job['job_id'], lease, final)
                batches.append({'count': len(jobs), 'seconds': elapsed,
                                'started_unix': started_unix, 'completed_unix': time.time()})
                idle_since = time.monotonic()
                print(json.dumps({'completed': sum(b['count'] for b in batches),
                                  'batch': batches[-1], 'queue': queue.counts()}), flush=True)
            except BaseException:
                for job, lease in claimed:
                    queue.release(job['job_id'], lease)
                raise
    finally:
        stats = {'load_seconds': loaded-start, 'work_seconds': time.perf_counter()-loaded,
                 'batches': batches, 'count': sum(b['count'] for b in batches), 'runtime': config,
                 'batch_policy': {'max_size': a.batch_size, 'min_size': a.min_batch_size,
                                  'max_wait_seconds': a.max_batch_wait}}
        a.metrics.write_text(json.dumps(stats, indent=2))
        model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    enqueue = sub.add_parser('enqueue')
    enqueue.add_argument('--jobs', type=Path, required=True)
    enqueue.add_argument('--seal', action='store_true')
    sub.add_parser('seal')
    serve = sub.add_parser('worker')
    serve.add_argument('--backend', choices=['vllm', 'transformers'], default='vllm')
    serve.add_argument('--batch-size', type=int, default=8)
    serve.add_argument('--min-batch-size', type=int, default=1)
    serve.add_argument('--max-batch-wait', type=float, default=0)
    serve.add_argument('--disable-input-cache', action='store_true',
                       help='Disable prefix and multimodal processor caches for repeated-cohort stress tests')
    serve.add_argument('--check-operation', action='store_true', help='Check true operation semantics in the same model call')
    serve.add_argument('--model', default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen3.8-27B')
    serve.add_argument('--metrics', type=Path, required=True)
    serve.add_argument('--result-dir', type=Path, help='Shared durable per-job audit results for production recovery')
    serve.add_argument('--idle-timeout', type=float, default=1800)
    export = sub.add_parser('export')
    export.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'worker' and (not 1 <= a.min_batch_size <= a.batch_size or a.max_batch_wait < 0):
        p.error('invalid batch wait policy')
    from sa1b_pipeline.work_queue import WorkQueue
    queue = WorkQueue(a.db)
    try:
        if a.command == 'enqueue':
            for line in a.jobs.read_text().splitlines():
                queue.enqueue(json.loads(line))
            if a.seal:
                queue.seal()
        elif a.command == 'seal':
            queue.seal()
        elif a.command == 'worker':
            worker(a, queue)
        else:
            queue.export_results(a.output)
        print(json.dumps(queue.counts()))
    finally:
        queue.close()


if __name__ == '__main__':
    main()
