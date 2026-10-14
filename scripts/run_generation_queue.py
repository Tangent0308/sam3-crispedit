"""Resident single-GPU editor consuming independently leased source/mask jobs."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--queue-root', type=Path, help='Node-local SQLite directory, separate from shared images')
    p.add_argument('--model', default='/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/Qwen-Image-2.1')
    p.add_argument('--idle-timeout', type=float, default=1800)
    p.add_argument('--reuse-generation-from', type=Path,
                   help='Copy prior per-job outputs; generate_cases validates the full generation fingerprint')
    a = p.parse_args()
    from sa1b_pipeline.work_queue import WorkQueue
    queue_root = a.queue_root or a.root
    generation = WorkQueue(queue_root/'generation.db')
    audit = WorkQueue(queue_root/'audit.db')
    model = None
    try:
        if generation.sealed() and not generation.counts().get('pending', 0) and not generation.counts().get('running', 0):
            print('generation queue already complete; model not loaded')
            return
        from sa1b_pipeline.editor import DiffusersQwenImageEditor, generate_cases
        started = time.perf_counter()
        model = DiffusersQwenImageEditor(
            a.model,
            steps=40, crop_instruction=False)
        loaded = time.perf_counter()
        events = []
        last_work = time.monotonic()
        while True:
            claimed = generation.claim(1)
            if not claimed:
                if generation.sealed() and not generation.counts().get('running', 0):
                    break
                if time.monotonic()-last_work > a.idle_timeout:
                    raise TimeoutError('no generation jobs; state retained')
                time.sleep(.2)
                continue
            job, lease = claimed[0]
            before = time.perf_counter()
            try:
                directory = a.root/'units'/hashlib.sha256(job['job_id'].encode()).hexdigest()
                directory.mkdir(parents=True, exist_ok=True)
                reused = False
                if a.reuse_generation_from:
                    cached = a.reuse_generation_from/'units'/directory.name
                    if cached.is_dir():
                        shutil.copytree(cached, directory, dirs_exist_ok=True)
                        reused = any(directory.glob('*.png'))
                manifest = directory/'input.jsonl'
                manifest.write_text(json.dumps(job['case'])+'\n')
                def publish(row, job_id=job['job_id']):
                    audit.enqueue({'job_id': job_id, 'row': row})
                rows = generate_cases(manifest, directory, model, seed=0, on_result=publish)
                if len(rows) != 1:
                    raise ValueError('generation job must contain exactly one unit')
                result = {'job_id': job['job_id'], 'row': rows[0], 'worker': a.worker,
                          'reused_generation': reused,
                          'service_seconds': time.perf_counter()-before, 'completed_unix': time.time()}
                generation.finish(job['job_id'], lease, result)
                events.append({k: v for k, v in result.items() if k != 'row'})
                last_work = time.monotonic()
                print(json.dumps(events[-1]), flush=True)
            except BaseException:
                generation.release(job['job_id'], lease)
                raise
        metrics = {'load_seconds': loaded-started, 'work_seconds': time.perf_counter()-loaded,
                   'count': len(events), 'steps': 40, 'seed': 0, 'events': events}
        (a.root/f'generate_worker_{a.worker}.json').write_text(json.dumps(metrics, indent=2))
    finally:
        generation.close()
        audit.close()
        if model is not None:
            model.pipe.close()


if __name__ == '__main__':
    main()
