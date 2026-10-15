"""Four independent Arnold ranks, ~50k balanced automatic-pass edit pairs."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sa1b_pipeline.operation_balance import automatic_pass
from sa1b_pipeline.production_generation import (
    REPO,
    run_generation,
    runtime_env,
    stop_processes,
)
from sa1b_pipeline.production_sources import materialize, prepare_reservoir
from sa1b_pipeline.production_state import (
    OPERATIONS,
    atomic_json,
    file_hash,
    mask_key,
    portable,
    read_rows,
    write_rows,
)

DEFAULT_ROOT = '/mnt/bn/strategy-mllm-train/user/tanyue/datasets/sa1b_samtok_edit_50k_20261014'


def emit(event):
    print(json.dumps({'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}, ensure_ascii=False), flush=True)


def load_config(root):
    config = json.loads((root/'config.json').read_text())
    if not (root/'inputs/ready.json').exists():
        raise RuntimeError('input reservoir is not prepared')
    for relative, hashed in config['code_hashes'].items():
        if file_hash(REPO/relative) != hashed:
            raise ValueError(f'production code differs from frozen run: {relative}')
    for relative, hashed in config.get('input_hashes', {}).items():
        if file_hash(root/relative) != hashed:
            raise ValueError(f'input partition/exclusion changed: {relative}')
    if config.get('source_backend') == 'shared_record_cache_v1':
        from sa1b_pipeline.production_source_cache import validate_cache
        validate_cache(root/'source_cache', root/'inputs/reservoir.jsonl', config.get('source_cache_ready_sha256'))
    return config


def prepare(a):
    a.root.mkdir(parents=True, exist_ok=True)
    config_path = a.root/'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config['target_total'] != a.target_total or config['source_capacity'] != a.source_capacity:
            raise ValueError('existing run has different target/capacity')
    else:
        if a.target_total <= 0 or a.target_total % 16:
            raise ValueError('target must be a positive multiple of 16 (4 nodes x 4 types)')
        code = [p for directory in ['sa1b_pipeline', 'scripts', 'utils', 'requirements/production']
                for p in (REPO/directory).rglob('*')
                if p.is_file() and (p.suffix in {'.py', '.sh', '.txt', '.json'})]
        config = {'version': 1, 'target_total': a.target_total, 'target_per_node_per_type': a.target_total//16,
                  'nodes': 4, 'gpus_per_node': 8, 'generation_steps': 40, 'seed': 0,
                  'batch_sources': a.batch_sources, 'source_capacity': a.source_capacity,
                  'source_root': str(a.source_root), 'benchmark_sha256': file_hash(a.benchmark),
                  'benchmark_original': str(a.benchmark), 'sampling_seed': a.seed,
                  'planner_model': a.planner_model, 'editor_model': a.editor_model, 'audit_model': a.audit_model,
                  'counting_unit': 'one independent source-mask edit, pipeline automatic pass, no manual approval claim',
                  'code_hashes': {str(p.relative_to(REPO)): file_hash(p) for p in sorted(code)}}
        atomic_json(config_path, config)
        write_rows(a.root/'inputs/benchmark_exclusion.jsonl', read_rows(a.benchmark))
    emit({'stage': 'prepare_reservoir', 'capacity': config['source_capacity']})
    prepare_reservoir(a.root, config['source_root'], a.root/'inputs/benchmark_exclusion.jsonl',
                      config['source_capacity'], config['sampling_seed'])
    hashes = {str(path.relative_to(a.root)): file_hash(path) for path in sorted((a.root/'inputs').glob('*'))
              if path.is_file()}
    if 'input_hashes' in config and config['input_hashes'] != hashes:
        raise ValueError('prepared input reservoir changed')
    config['input_hashes'] = hashes
    atomic_json(config_path, config)
    emit({'stage': 'prepared', 'root': str(a.root), 'target': config['target_total']})


def collect(node):
    rows, unused = [], []
    waves = sorted((node/'waves').glob('*')) if (node/'waves').exists() else []
    for wave in waves:
        if not (wave/'complete.json').exists():
            break
        if (wave/'generation/validation.jsonl').exists():
            rows.extend(read_rows(wave/'generation/validation.jsonl'))
            unused = read_rows(wave/'generation/unused_plans.jsonl')
    ids = [r['job_id'] for r in rows]
    masks = [mask_key(r) for r in rows]
    if len(ids) != len(set(ids)) or len(masks) != len(set(masks)):
        raise ValueError('duplicate generated source/mask across waves')
    return rows, unused


def planner(wave, candidates, weights, config, progress):
    processes = []
    try:
        for i in range(2):
            manifest = wave/f'planning_input_{i}.jsonl'
            if not manifest.exists():
                write_rows(manifest, candidates[i::2])
            if not candidates[i::2]:
                continue
            out = wave/f'planning_{i}'
            if (out/'think_256/run/summary.json').exists():
                summary = json.loads((out/'think_256/run/summary.json').read_text())
                completed_plans = read_rows(out/'think_256/planned.jsonl')
                if (summary['input_masks'] != sum(len(c['objects']) for c in candidates[i::2])
                        or len(completed_plans) != summary['accepted_masks']
                        or summary['model_calls']+summary.get('interrupted_unknown', 0) != summary['input_masks']):
                    raise ValueError('planner completion checkpoint has inconsistent coverage')
                continue
            python = config['planner_python']
            cmd = [python, '-m', 'scripts.run_balanced_planning', '--root', str(out),
                   '--manifest', str(manifest), '--model', config['planner_model'],
                   '--production-resume', '--operation-weights', json.dumps(weights)]
            log = (wave/f'planning_{i}.log').open('a')
            proc = subprocess.Popen(cmd, cwd=REPO, env=runtime_env(python, ','.join(config['devices'][i*4:i*4+4])),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((proc, log))
        while any(proc.poll() is None for proc, _ in processes):
            if any(proc.poll() not in (None, 0) for proc, _ in processes):
                raise RuntimeError(f'planning worker failed; see {wave}')
            progress({'stage': 'planning', 'log_files': [str(wave/f'planning_{i}.log') for i in range(2)]})
            time.sleep(10)
        if any(proc.returncode for proc, _ in processes):
            raise RuntimeError(f'planning failed: {wave}')
    finally:
        stop_processes(processes)
    plans = []
    for i in range(2):
        path = wave/f'planning_{i}/think_256/planned.jsonl'
        if path.exists():
            plans.extend(read_rows(path))
    return plans


def export_node(root, node, rows, target):
    passed = [r for r in rows if automatic_pass(r, require_operation=True)]
    counts = Counter(r['operation'] for r in passed)
    if any(counts[op] > target for op in OPERATIONS):
        raise ValueError('quota overshoot')
    cache = {}
    write_rows(node/'auto_pass.jsonl', [portable(r, root, cache=cache) for r in passed])
    # All failures and their images stay available; accepted list is separate.
    write_rows(node/'validation.jsonl', [portable(r, root, cache=cache) for r in rows])
    return counts


def run_node(a):
    config = load_config(a.root)
    if not 0 <= a.rank < 4:
        raise ValueError('rank must be 0..3')
    devices = a.devices.split(',')
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError('each production rank requires 8 distinct GPUs')
    config.update(devices=devices, planner_python=a.planner_python, editor_python=a.editor_python,
                  audit_python=a.audit_python)
    for key in ['planner_python', 'editor_python', 'audit_python']:
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    for key, filename in [('planner_model', 'config.json'), ('editor_model', 'model_index.json'), ('audit_model', 'config.json')]:
        if not (Path(config[key])/filename).is_file():
            raise FileNotFoundError(config[key])
    # Never place SQLite WAL files on the shared dataset mount.
    if a.local_root.resolve().is_relative_to(Path('/mnt')):
        raise ValueError('local-root must use node-local storage, not /mnt')
    node = a.root/f'nodes/node{a.rank}'
    node.mkdir(parents=True, exist_ok=True)
    lock = (node/'owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    atomic_json(node/'owner.json', {'pid': os.getpid(), 'host': socket.gethostname(), 'started_unix': time.time()})
    def stopped(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    entries = read_rows(a.root/f'inputs/node{a.rank}.jsonl')
    exclusion = json.loads((a.root/'inputs/exclusion.json').read_text())
    target = config['target_per_node_per_type']
    started = time.time()
    batch_size = config['batch_sources']
    try:
        for wave_index, offset in enumerate(range(0, len(entries), batch_size)):
            wave = node/'waves'/f'{wave_index:05d}'
            if (wave/'complete.json').exists():
                continue
            rows, unused = collect(node)
            counts = export_node(a.root, node, rows, target)
            targets = {op: target-counts[op] for op in OPERATIONS}
            if not any(targets.values()):
                break
            wave.mkdir(parents=True, exist_ok=True)
            def progress(details, counts=counts, wave_index=wave_index):
                details = dict(details)
                if 'status' in details:
                    details['wave_status'] = details.pop('status')
                if 'target_per_type' in details:
                    details['wave_target_per_type'] = details.pop('target_per_type')
                wave_pass = details.get('automatically_passed', {})
                totals = {op: counts[op]+wave_pass.get(op, 0) for op in OPERATIONS}
                status = {'rank': a.rank, 'status': 'running', 'wave': wave_index,
                          'started_unix_this_attempt': started, 'updated_unix': time.time(),
                          'target_per_type': target, 'automatically_passed': totals,
                          'remaining': {op: target-totals[op] for op in OPERATIONS}, **details}
                # Details' wave counts must not replace cumulative node totals.
                status['automatically_passed'] = totals
                atomic_json(node/'progress.json', status)
                emit(status)
            wave_config = wave/'config.json'
            if wave_config.exists():
                frozen = json.loads(wave_config.read_text())
                if frozen['targets'] != targets:
                    raise ValueError('completed outcomes changed during wave resume')
                weights = frozen['weights']
            else:
                attempts = Counter(r['operation'] for r in rows)
                pool = Counter(c['units'][0]['operation'] for c in unused)
                # Smooth observed generation pass rates; devote more planning
                # to final shortages, without changing masks' rejected types.
                rates = {op: (counts[op]+4)/(attempts[op]+8) for op in OPERATIONS}
                weights = {op: max(0., targets[op]/rates[op]-pool[op]) for op in OPERATIONS}
                if not any(weights.values()):
                    weights = {op: float(targets[op]) for op in OPERATIONS}
                atomic_json(wave_config, {'targets': targets, 'weights': weights})
            progress({'stage': 'source_materialization'})
            input_file = wave/'candidates.jsonl'
            if input_file.exists():
                candidates = read_rows(input_file)
            else:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    materialized = list(pool.map(lambda e: materialize(e, a.root, config['source_root'], exclusion),
                                                  entries[offset:offset+batch_size]))
                write_rows(wave/'source_status.jsonl', [{k: v for k, v in r.items() if k != 'candidate'} for r in materialized])
                candidates = [r['candidate'] for r in materialized if r['status'] == 'ready']
                write_rows(input_file, candidates)
            plans = planner(wave, candidates, weights, config, progress) if candidates else []
            generation_input = wave/'generation_input.jsonl'
            expected = unused+plans
            if generation_input.exists():
                if read_rows(generation_input) != expected:
                    raise ValueError('generation manifest changed during resume')
            else:
                write_rows(generation_input, expected)
            run_generation(generation_input, wave/'generation', a.local_root/a.root.name/f'node{a.rank}',
                           targets, config, lambda r: progress({'stage': 'generation_and_audit', **r}))
            atomic_json(wave/'complete.json', {'completed_unix': time.time(), 'sources': len(candidates)})
        rows, _ = collect(node)
        counts = export_node(a.root, node, rows, target)
        complete = all(counts[op] == target for op in OPERATIONS)
        status = {'rank': a.rank, 'status': 'complete' if complete else 'insufficient_source_supply',
                  'automatically_passed': {op: counts[op] for op in OPERATIONS}, 'updated_unix': time.time(),
                  'remaining': {op: target-counts[op] for op in OPERATIONS}, 'generated': len(rows)}
        atomic_json(node/'progress.json', status)
        if complete:
            atomic_json(node/'done.json', status)
            finalize(a.root, require_complete=False)
        else:
            raise RuntimeError('source reservoir exhausted; partial data retained, no complete 50k claim')
        emit(status)
    except BaseException as exc:
        atomic_json(node/'failure.json', {'error': str(exc), 'time': time.time(), 'host': socket.gethostname()})
        previous = json.loads((node/'progress.json').read_text()) if (node/'progress.json').exists() else {}
        atomic_json(node/'progress.json', {**previous, 'status': 'failed', 'error': str(exc),
                                          'updated_unix': time.time()})
        raise
    finally:
        lock.close()


def finalize(root, require_complete=True):
    config = load_config(root)
    if not all((root/f'nodes/node{i}/done.json').exists() for i in range(4)):
        if require_complete:
            raise ValueError('not all four nodes have completed their quotas')
        return
    from scripts.finalize_production_exports import finalize_exports
    finalize_exports(root)


def status(root):
    config = json.loads((root/'config.json').read_text())
    nodes = []
    for rank in range(4):
        path = root/f'nodes/node{rank}/progress.json'
        node = json.loads(path.read_text()) if path.exists() else {'rank': rank, 'status': 'not_started'}
        if 'updated_unix' in node:
            node['heartbeat_age_seconds'] = round(time.time()-node['updated_unix'], 1)
        nodes.append(node)
    counts = {op: sum(n.get('automatically_passed', {}).get(op, 0) for n in nodes) for op in OPERATIONS}
    emit({'target': config['target_total'], 'automatic_passes': sum(counts.values()), 'operations': counts, 'nodes': nodes})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--target-total', type=int, default=50000)
    prep.add_argument('--source-capacity', type=int, default=16000)
    prep.add_argument('--batch-sources', type=int, default=64)
    prep.add_argument('--seed', type=int, default=20261014)
    prep.add_argument('--source-root', type=Path, default=Path('/mnt/hdfs/sg_byte_ttlive_strategy_llm/user/haobo.yuan/datasets'))
    prep.add_argument('--benchmark', type=Path, default=Path('/opt/tiger/tanyue/samtok_edit_benchmark_v2branch/data/v2/cases.jsonl'))
    models = '/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models/'
    prep.add_argument('--planner-model', default=models+'Qwen3.8-Flash-Next-FP8')
    prep.add_argument('--editor-model', default=models+'Qwen-Image-2.1')
    prep.add_argument('--audit-model', default=models+'Qwen3.8-27B')
    run = sub.add_parser('run')
    run.add_argument('--rank', type=int, default=int(os.environ.get('ARNOLD_ID', '0')))
    run.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    run.add_argument('--local-root', type=Path, default=Path('/opt/tiger/sa1b-production-local'))
    run.add_argument('--planner-python', default='/tmp/sa1b-next-runtime/bin/python')
    run.add_argument('--editor-python', default='/tmp/sa1b-editor-runtime/bin/python')
    run.add_argument('--audit-python', default='/tmp/sa1b-mllm-runtime/bin/python')
    sub.add_parser('finalize')
    sub.add_parser('status')
    a = p.parse_args()
    a.root = a.root.resolve()
    if a.command == 'prepare':
        if a.batch_sources < 1 or a.source_capacity < 4:
            p.error('positive batch size and at least four source candidates required')
        prepare(a)
    elif a.command == 'run':
        run_node(a)
    elif a.command == 'finalize':
        finalize(a.root)
    else:
        status(a.root)


if __name__ == '__main__':
    main()
