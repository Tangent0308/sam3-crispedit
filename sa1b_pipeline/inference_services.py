"""Run editor and auditor in separate pinned Python environments and GPU pools."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import signal
import subprocess
import threading
import time
import uuid

from PIL import Image

from .production_state import atomic_json


class ServicePool:
    """One atomic-file inbox per GPU; audit workers gather requests into batches."""
    def __init__(self, *, python, role, devices, model, directory, batch_size=4, dtype="bf16", timeout=1800):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("service devices must be distinct and nonempty")
        self.python, self.role, self.devices, self.model = python, role, devices, model
        self.directory = Path(directory) / uuid.uuid4().hex
        self.batch_size, self.dtype, self.timeout = batch_size, dtype, timeout
        self.lock = threading.Lock(); self.cursor = 0; self.workers = []
        self.closed = False

    def _start(self):
        # Called under lock, does not block for model load: GPUs initialize concurrently.
        if self.closed:
            raise RuntimeError("inference service is closed")
        if self.workers:
            return
        script = Path(__file__).resolve().parents[1] / 'scripts/group_production/inference_worker.py'
        for i, device in enumerate(self.devices):
            folder = self.directory / str(i); folder.mkdir(parents=True)
            (folder / 'requests').mkdir(); (folder / 'responses').mkdir()
            log = (folder / 'worker.log').open('a')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), PYTHONUNBUFFERED='1')
            cmd = [self.python, str(script), '--role', self.role, '--model', self.model,
                   '--directory', str(folder), '--batch-size', str(self.batch_size), '--dtype', self.dtype,
                   '--owner-pid', str(os.getpid())]
            try:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            self.workers.append((folder, process, log))

    def request(self, payload):
        with self.lock:
            self._start()
            folder, process, _ = self.workers[self.cursor % len(self.workers)]; self.cursor += 1
        request_id = uuid.uuid4().hex
        response = folder / 'responses' / f'{request_id}.json'
        atomic_json(folder / 'requests' / f'{request_id}.json', payload)
        start = time.monotonic()
        while not response.exists():
            if self.closed or process.poll() is not None:
                raise RuntimeError(f'{self.role} worker stopped; see {folder}/worker.log')
            if time.monotonic() - start > self.timeout:
                raise TimeoutError(f'{self.role} request timed out; see {folder}/worker.log')
            time.sleep(.1)
        value = json.loads(response.read_text())
        if 'error' in value:
            raise RuntimeError(f'{self.role} inference failed: {value["error"]}; see {folder}/worker.log')
        return value['result']

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            for folder, _, _ in self.workers:
                (folder / 'stop').touch()
        deadline = time.monotonic() + 15
        for _, process, log in self.workers:
            try:
                process.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            finally:
                # Workers can leave descendants even after the parent exits.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                log.close()


class RemoteEditor:
    def __init__(self, pool):
        self.pool = pool
        self.config = {'backend': 'qwen-omni-transactional-1', 'model_id': pool.model,
                       'steps': 40, 'dtype': pool.dtype, 'instruction_frame': 'full-ref',
                       'prompt_version': 'region-edit-5-contextual-removal'}
        # Each group gets its own adapter; pool is shared, context is not.
        self.last_generation_context = {}

    def generate(self, source, unit, seed=0):
        result = self.pool.request({'source': source, 'unit': unit, 'seed': seed})
        self.last_generation_context = result['context']
        with Image.open(result['image']) as image:
            output = image.convert('RGB')
        Path(result['image']).unlink()
        return output


class RemoteAuditor:
    def __init__(self, pool):
        from .group_audit import group_audit_contract
        self.pool = pool; self.batch_size = pool.batch_size
        self.config = {**group_audit_contract('multi', pool.model), 'max_tokens': 1000}

    def call_batch(self, jobs):
        # Futures preserve input ordering. Actual batching happens inside the GPU worker.
        with ThreadPoolExecutor(max_workers=len(jobs)) as threads:
            return list(threads.map(self.pool.request, jobs))
