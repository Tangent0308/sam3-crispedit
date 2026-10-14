"""Resumable 6 editor + 2 audit pool; shared immutable journals, local queues."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from .operation_balance import OperationBalance
from .production_state import atomic_json, immutable_json, read_rows, write_rows
from .work_queue import WorkQueue

REPO = Path(__file__).resolve().parents[1]


def runtime_env(python, gpu):
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=gpu,
               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1', CUDA_MODULE_LOADING='LAZY',
               WANDB_MODE='disabled', WANDB_DISABLED='true', DIFFUSION_ATTENTION_BACKEND='TORCH_SDPA')
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    libs = sorted(p for p in Path(python).parent.parent.glob('lib/python*/site-packages/nvidia/*/lib')
                  if p.parent.name != 'cu13')
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, libs)) + os.pathsep + env.get('LD_LIBRARY_PATH', '')
    return env


def stop_processes(processes):
    # Terminate child process groups as well as the launcher (vLLM has children).
    for proc, _ in processes:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    for proc, log in processes:
        try:
            proc.wait(timeout=max(.1, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        log.close()


class JournalQueue:
    def __init__(self, queue, directory):
        self.queue, self.directory = queue, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def enqueue(self, job):
        path = self.directory/(hashlib.sha256(job['job_id'].encode()).hexdigest()+'.json')
        immutable_json(path, job)  # Intent precedes local queue insertion.
        return self.queue.enqueue(job)


def run_generation(manifest, root, local_root, targets, config, progress):
    root, local_root = Path(root), Path(local_root)
    root.mkdir(parents=True, exist_ok=True)
    immutable_json(root/'contract.json', {'manifest': read_rows(manifest), 'targets': targets,
                                        'editor_model': config['editor_model'], 'audit_model': config['audit_model'],
                                        'steps': 40, 'audit_operation': True})
    completed = root/'complete.json'
    if completed.exists():
        return read_rows(root/'validation.jsonl'), read_rows(root/'unused_plans.jsonl')
    # A fresh local queue per supervised attempt. Shared results are authoritative.
    qroot = local_root/('queues-'+uuid.uuid4().hex)
    generation, audit = WorkQueue(qroot/'generation.db'), WorkQueue(qroot/'audit.db')
    balance = OperationBalance(read_rows(manifest), targets, 24)
    dispatch = JournalQueue(generation, root/'dispatch')
    dispatched = [json.loads(p.read_text()) for p in sorted((root/'dispatch').glob('*.json'))]
    results = [json.loads(p.read_text())['result'] for p in sorted((root/'results').glob('*.json'))]
    balance.restore(dispatched, results)
    for job in dispatched:
        if job['job_id'] in balance.pending:
            generation.enqueue(job)
    balance.dispatch(dispatch)
    started = time.monotonic()
    processes = []
    try:
        if not balance.exhausted:
            for role, count in [('generate', 6), ('audit', 2)]:
                for i in range(count):
                    gpu = config['devices'][i if role == 'generate' else i+6]
                    if role == 'generate':
                        python = config['editor_python']
                        command = [python, '-m', 'scripts.run_generation_queue', '--root', str(root),
                                   '--queue-root', str(qroot), '--worker', str(i), '--model', config['editor_model']]
                    else:
                        python = config['audit_python']
                        command = [python, '-m', 'scripts.run_audit_queue', '--db', str(qroot/'audit.db'),
                                   'worker', '--batch-size', '8', '--min-batch-size', '8', '--max-batch-wait', '5',
                                   '--check-operation', '--model', config['audit_model'],
                                   '--result-dir', str(root/'results'), '--metrics', str(root/f'audit_{i}.json')]
                    log = (root/f'{role}_{i}.log').open('a')
                    proc = subprocess.Popen(command, cwd=REPO, env=runtime_env(python, gpu),
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append((proc, log))
        last_log = 0
        while processes and any(proc.poll() is None for proc, _ in processes):
            if any(proc.poll() not in (None, 0) for proc, _ in processes):
                raise RuntimeError(f'generation/audit worker failed: {root}')
            balance.observe(audit.completed_for(balance.pending))
            balance.dispatch(dispatch)
            if balance.exhausted:
                generation.seal()
                audit.seal()
            if time.monotonic()-last_log >= 10:
                report = {**balance.report(), 'elapsed_seconds_this_attempt': time.monotonic()-started,
                          'generation_queue': generation.counts(), 'audit_queue': audit.counts()}
                atomic_json(root/'progress.json', report)
                progress(report)
                last_log = time.monotonic()
            time.sleep(1)
        if any(proc.returncode for proc, _ in processes):
            raise RuntimeError('worker failed')
        balance.observe(audit.completed_for(balance.pending))
        if not balance.exhausted:
            raise RuntimeError('workers exited with unfinished reservations')
        rows = list(balance.outcomes.values())
        write_rows(root/'validation.jsonl', rows)
        write_rows(root/'unused_plans.jsonl', balance.unused_cases())
        atomic_json(root/'complete.json', {**balance.report(), 'seconds_this_attempt': time.monotonic()-started})
        return rows, balance.unused_cases()
    finally:
        stop_processes(processes)
        generation.close()
        audit.close()
