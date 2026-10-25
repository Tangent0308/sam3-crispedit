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
    def __init__(self, *, python, role, devices, model, directory, batch_size=4, dtype="bf16", timeout=1800,
                 planner_options=None, editor_tools=None, dispatch='round_robin'):
        if not devices or len(set(devices)) != len(devices):
            raise ValueError("service devices must be distinct and nonempty")
        if dispatch not in ('round_robin', 'least_loaded'):
            raise ValueError("dispatch must be round_robin or least_loaded")
        # least_loaded (opt-in): send each request to the worker with the fewest requests still
        # waiting or running in this process, so that long requests do not pile up on one inbox.
        self.dispatch = dispatch
        # Opt-in: adapters tag requests on a critical path with a priority (see inference_worker).
        self.priority_scheduling = False
        self.python, self.role, self.devices, self.model = python, role, devices, model
        self.directory = Path(directory) / uuid.uuid4().hex
        self.batch_size, self.dtype, self.timeout = batch_size, dtype, timeout
        self.planner_options = dict(planner_options or {})
        self.editor_tools = dict(editor_tools or {})
        if set(self.editor_tools) - {'lama_model', 'restore_texture', 'text_verify', 'finish'}:
            raise ValueError('unsupported editor tool options')
        if self.editor_tools and role not in ('editor', 'twins'):
            raise ValueError('editor tools require an editor service')
        if set(self.planner_options) - {'max_new_tokens','thinking_token_budget'}:
            raise ValueError('unsupported planner service options')
        self.lock = threading.Lock(); self.cursor = 0; self.workers = []; self.outstanding = []
        self.closed = False

    def start(self):
        """Load independent GPU services concurrently, before their first job."""
        with self.lock:
            self._start()

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
            env['PATH'] = str(Path(self.python).parent) + os.pathsep + env.get('PATH', '')
            cmd = [self.python, str(script), '--role', self.role, '--model', self.model,
                   '--directory', str(folder), '--batch-size', str(self.batch_size), '--dtype', self.dtype,
                   '--owner-pid', str(os.getpid())]
            for key, value in self.planner_options.items():
                cmd += ['--'+key.replace('_','-'), str(value)]
            if self.editor_tools.get('lama_model'):
                cmd += ['--lama-repair-model', self.editor_tools['lama_model']]
            if self.editor_tools.get('restore_texture'):
                cmd += ['--restore-color-texture']
            if self.editor_tools.get('text_verify'):
                cmd += ['--text-verify']
            if self.editor_tools.get('finish'):
                cmd += ['--editor-finish', json.dumps(self.editor_tools['finish'], sort_keys=True)]
            try:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                log.close()
                raise
            self.workers.append((folder, process, log))
        self.outstanding = [0] * len(self.workers)

    def request(self, payload):
        with self.lock:
            self._start()
            index = self._pick()
            self.outstanding[index] += 1
            folder, process, _ = self.workers[index]
        try:
            return self._request(folder, process, payload)
        finally:
            with self.lock:
                self.outstanding[index] -= 1

    def _pick(self):
        """Worker index for the next request (called under the lock)."""
        n = len(self.workers)
        if self.dispatch == 'least_loaded':
            # Round-robin among the least loaded workers.
            index = min(range(n), key=lambda i: (self.outstanding[i], (i - self.cursor) % n))
        else:
            index = self.cursor % n
        self.cursor += 1
        return index

    def _request(self, folder, process, payload):
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
        # Durable group checkpoints own the evidence. Keep resident inbox scans
        # bounded across thousands of waves; image responses are consumed later.
        (folder / 'requests' / f'{request_id}.json').unlink(missing_ok=True)
        response.unlink(missing_ok=True)
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


def _with_priority(payload, priority):
    return {**payload, 'priority': priority} if priority else payload


class RemoteEditor:
    """Multi-target steps; with pool.priority_scheduling their requests go first (each step
    waits for the previous one, so a multi group is a chain of dependent requests)."""
    def __init__(self, pool):
        self.pool = pool
        self.priority = 1 if getattr(pool, 'priority_scheduling', False) else 0
        self.config = {'backend': 'qwen-omni-transactional-1', 'model_id': pool.model,
                       'steps': 40, 'dtype': pool.dtype, 'instruction_frame': 'full-ref',
                       'prompt_version': 'region-edit-8-explicit-content'}
        if getattr(pool, 'editor_tools', None):
            self.config['experimental_tools'] = dict(pool.editor_tools)
            self.config['tools_version'] = 'experimental-2'
        # Each group gets its own adapter; pool is shared, context is not.
        self.last_generation_context = {}
        if (getattr(pool, 'editor_tools', None) or {}).get('finish', {}).get('removal_effects'):
            # Multi steps resolve the removal's shadow/reflection before generation
            # so the execution mask, checks and paste all share one support.
            self.removal_effect = self._removal_effect

    def _removal_effect(self, source, unit, output):
        result = self.pool.request(_with_priority({'request_kind': 'removal_effect', 'source': source, 'unit': unit,
                                                   'output': str(output)}, self.priority))
        return result.get('mask'), result.get('info')

    def generate(self, source, unit, seed=0):
        result = self.pool.request(_with_priority({'source': source, 'unit': unit, 'seed': seed}, self.priority))
        self.last_generation_context = result['context']
        with Image.open(result['image']) as image:
            output = image.convert('RGB')
        Path(result['image']).unlink()
        return output


class RemoteAuditor:
    def __init__(self, pool, kind='multi'):
        from .group_audit import group_audit_contract
        self.pool = pool; self.batch_size = pool.batch_size
        self.grounding = getattr(pool, 'grounding', None)
        # Optional post-audit defect check (edit_verifier); None keeps the audit unchanged.
        self.verifier = getattr(pool, 'verifier', None)
        # Optional single-target salvage (single_salvage); audits groups that cannot ship as groups.
        self.keep_singles = getattr(pool, 'keep_singles', False)
        # Optional reader check of the audit's Ref-uniqueness doubts (audit_reconcile).
        self.ref_reconcile = getattr(pool, 'ref_reconcile', False)
        self.config = {**group_audit_contract(kind, pool.model), 'max_tokens': 1000}
        # Multi audits sit on their group's chain (see RemoteEditor); twin audits do not.
        self.priority = 1 if getattr(pool, 'priority_scheduling', False) and kind == 'multi' else 0

    def call_batch(self, jobs):
        # Futures preserve input ordering. Actual batching happens inside the GPU worker.
        with ThreadPoolExecutor(max_workers=len(jobs)) as threads:
            return list(threads.map(lambda job: self.pool.request(_with_priority(job, self.priority)), jobs))


class RemoteGrounding:
    """Optional small-model evidence service, independent of the MLLM response."""
    def __init__(self, pool):
        self.pool = pool

    def inspect(self, source_path, member, result_path=None):
        return self.pool.request(dict(source_path=source_path, member=member, result_path=result_path))


class RemoteSelector:
    """Caller-owned persistent planner; close never terminates a shared pool."""
    def __init__(self, pool):
        self.pool = pool
        self.model_id = pool.model
        # Fill each replica's own batch. Keeping the per-device batch here
        # serializes callers into small global waves and leaves replicas idle.
        self.batch_size = pool.batch_size * len(pool.devices)

    def generate_multimage(self, jobs, *, schemas=None, token_limits=None, thinking=None):
        if not jobs:
            return []
        if schemas is not None and len(schemas) != len(jobs):
            raise ValueError('schema count must match planning requests')
        if thinking is not None and (len(thinking) != len(jobs) or any(t not in (None, False) for t in thinking)):
            raise ValueError('per-request thinking may only disable reasoning')
        if token_limits is not None and (len(token_limits) != len(jobs) or
                any(type(n) is not int or not 1 <= n <= 4096 for n in token_limits)):
            raise ValueError('invalid per-request token limits')
        requests = [{'images': paths, 'prompt': prompt,
                     **({'max_new_tokens': token_limits[i]} if token_limits is not None else {}),
                     **({'schema': schemas[i]} if schemas is not None else {}),
                     **({'thinking': False} if thinking is not None and thinking[i] is False else {})}
                    for i,(paths,prompt) in enumerate(jobs)]
        with ThreadPoolExecutor(max_workers=min(len(jobs), self.batch_size)) as threads:
            return list(threads.map(self.pool.request, requests))

    def close(self):
        pass


class RemoteBlind(RemoteSelector):
    """Use the resident audit model with the unchanged image-only blind input."""
    def generate_multimage(self, jobs):
        if not jobs:
            return []
        requests = [{'job_id': uuid.uuid4().hex, 'images': paths, 'prompt': prompt,
                     'image_max_edge': 0, 'request_kind':'blind'} for paths, prompt in jobs]
        with ThreadPoolExecutor(max_workers=min(len(jobs), self.batch_size)) as threads:
            results = list(threads.map(self.pool.request, requests))
        return [{**r['semantic_audit'], '_generation_metadata': {
                    'raw_output': r['semantic_audit_raw'], 'finish_reason': r['finish_reason'],
                    'input_tokens': r['input_tokens'], 'output_tokens': r['output_tokens']}}
                for r in results]
