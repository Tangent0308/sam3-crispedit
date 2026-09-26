"""Shared-filesystem orchestration for four independent ScaleEdit GPU nodes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

import pyarrow.parquet as pq
from tqdm import tqdm

from scaleedit import runner
from scaleedit.validation import validate_shard

REPO = Path(__file__).resolve().parents[1]
STAGES = ('quality', 'scene', 'grounding', 'mask')
BASE = Path('/mnt/bn/strategy-mllm-train/user/tanyue')


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def code_digest():
    digest = hashlib.sha256()
    paths = list((REPO / 'scaleedit').rglob('*.py')) + list((REPO / 'scripts').glob('*.py'))
    paths += list((REPO / 'scripts').glob('*.sh')) + [REPO / 'scripts/scaleedit_packages.txt', REPO / 'pyproject.toml']
    for path in sorted(paths):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_peers(args):
    failures = sorted(args.control.glob('*.failed'))
    failures += sorted((args.run_dir / 'bootstrap_control' / args.attempt).glob('*.failed'))
    if failures:
        raise RuntimeError(f'Peer failed: {failures[0]}: {failures[0].read_text()[:1000]}')


def wait_for(args, path):
    started = time.monotonic()
    while True:
        check_peers(args)
        if path.exists():
            return
        if time.monotonic() - started > args.wait_seconds:
            raise TimeoutError(f'Timed out waiting for {path}')
        time.sleep(2)


def run_config(args):
    config = {key: str(getattr(args, key)) for key in (
        'source_dir', 'nodes', 'model_path', 'checkpoint_path', 'batch_size',
        'grounding_tp', 'grounding_batch_size', 'local_test')}
    config['selection'] = runner.load_selection(args.selection_file)
    return config


def source_plan(args):
    selected = runner.load_selection(args.selection_file)
    paths = runner.discover(args.source_dir)
    if selected is not None and set(selected) - {p.name for p in paths}:
        raise ValueError('Selection contains unknown source shards')
    shards = {}
    for path in tqdm(paths, desc='ScaleEdit source plan', unit='shard', mininterval=5):
        if selected is not None and path.name not in selected:
            continue
        pf = pq.ParquetFile(path)
        if not runner.REQUIRED_COLUMNS <= set(pf.schema_arrow.names):
            raise ValueError(f'Invalid source schema: {path}')
        indices = selected[path.name] if selected is not None else list(range(pf.metadata.num_rows))
        if any(i >= pf.metadata.num_rows for i in indices):
            raise ValueError(f'Selection outside source: {path}')
        stat = path.stat()
        shards[path.name] = dict(indices=indices, rows=pf.metadata.num_rows,
                                 bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if not shards:
        raise ValueError('Empty source selection')
    return dict(config=run_config(args), code_sha256=code_digest(), shards=shards)


def balanced(names, weights, nodes):
    groups, loads = [[] for _ in range(nodes)], [0] * nodes
    for name in sorted(names, key=lambda n: (-weights[n], n)):
        rank = min(range(nodes), key=lambda r: (loads[r], len(groups[r]), r))
        groups[rank].append(name)
        loads[rank] += weights[name]
    return [sorted(g) for g in groups], loads


def eligible(args, plan, stage, name):
    indices = plan['shards'][name]['indices']
    for previous in ('quality', 'scene'):
        if stage == previous:
            break
        upstream = runner.load_index(args.run_dir / previous / 'manifest' / name)
        if set(upstream) != set(indices):
            raise ValueError(f'{previous} coverage mismatch: {name}')
        indices = [i for i in indices if upstream[i]['keep']]
    return indices


def stage_plan(args, plan, stage):
    weights = {name: len(eligible(args, plan, stage, name)) for name in plan['shards']}
    groups, loads = balanced(weights, weights, args.nodes)
    return dict(assignments=groups, node_rows=loads, selected_rows=sum(loads))


def execute(args, command, stage):
    log = args.run_dir / 'logs' / f'{stage}.node{args.rank}.log'
    print(f'node{args.rank} {stage}: log={log}', flush=True)
    with log.open('a') as handle:
        handle.write('COMMAND ' + json.dumps(command) + '\n')
        handle.flush()
        environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=handle,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                check_peers(args)
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f'{stage} exited {process.returncode}; see {log}')
        finally:
            # Also stop vLLM grandchildren after an abnormal worker exit.
            if process.poll() is None or process.returncode:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def stage_command(args, stage, output, selection):
    command = [sys.executable, '-u', str(REPO / 'scripts/run_scaleedit_pipeline.py'),
               '--stage', stage, '--input-dir', str(args.source_dir), '--output-dir', str(output),
               '--selection-file', str(selection), '--devices', args.devices,
               '--model-path', args.model_path, '--checkpoint-path', args.checkpoint_path,
               '--tensor-parallel-size', str(args.grounding_tp if stage == 'grounding' else 1),
               '--batch-size', str(args.grounding_batch_size if stage == 'grounding' else args.batch_size),
               '--vllm-max-num-seqs', str(args.grounding_batch_size if stage == 'grounding' else args.batch_size),
               '--vllm-enforce-eager']
    if stage != 'quality':
        command += ['--quality-dir', str(args.run_dir / 'quality')]
    if stage in ('grounding', 'mask'):
        command += ['--scene-dir', str(args.run_dir / 'scene')]
    if stage == 'mask':
        command += ['--grounding-dir', str(args.run_dir / 'grounding')]
    return command


def publish(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        if not os.path.samefile(source, target):
            raise FileExistsError(f'Refusing to replace unrelated output: {target}')


def merge_stage(args, plan, stage, scheduled):
    totals = Counter()
    for rank, names in enumerate(scheduled['assignments']):
        root = args.run_dir / 'work' / stage / f'node{rank}'
        for name in tqdm(names, desc=f'Validate {stage} node{rank}', unit='shard', mininterval=5):
            rel = Path('audit') / name if stage in ('quality', 'scene') else Path(name)
            source = root / rel
            totals.update(validate_shard(args.source_dir / name, source,
                                         eligible(args, plan, stage, name), stage))
            publish(source, args.run_dir / stage / rel)
            if stage in ('quality', 'scene'):
                manifest = root / 'manifest' / name
                if not pq.read_table(source).equals(pq.read_table(manifest), check_metadata=True):
                    raise ValueError(f'audit/manifest disagreement: {manifest}')
                publish(manifest, args.run_dir / stage / 'manifest' / name)
    result = dict(stage=stage, counts=dict(totals), selected_rows=scheduled['selected_rows'])
    atomic_json(args.run_dir / stage / 'run_summary.json', result)
    return result


def pipeline(args):
    report = dict(hostname=socket.gethostname(), rank=args.rank, config=run_config(args),
                  code_sha256=code_digest(), devices=args.devices)
    atomic_json(args.control / f'joined.node{args.rank}.json', report)
    for rank in range(args.nodes):
        wait_for(args, args.control / f'joined.node{rank}.json')
    reports = [json.loads((args.control / f'joined.node{r}.json').read_text()) for r in range(args.nodes)]
    if any(r['config'] != report['config'] or r['code_sha256'] != report['code_sha256'] for r in reports):
        raise ValueError('Node code/settings mismatch')
    if not args.local_test and len({r['hostname'] for r in reports}) != args.nodes:
        raise ValueError('Distinct physical hosts required; --local-test is only for smoke validation')
    if args.local_test:
        used = set()
        for r in reports:
            devices = set(r['devices'].split(','))
            if used & devices:
                raise ValueError('Local smoke ranks must use disjoint GPUs')
            used |= devices
    path = args.run_dir / 'plan.json'
    if args.rank == 0:
        current = source_plan(args)
        if path.exists() and json.loads(path.read_text()) != current:
            raise ValueError('Code, configuration, selection or source changed; use a new run')
        if not args.resume and path.exists():
            raise ValueError('Run already exists; use --resume and a new shared --attempt')
        atomic_json(path, current)
        atomic_json(args.control / 'plan.ready', {})
    wait_for(args, args.control / 'plan.ready')
    plan = json.loads(path.read_text())
    for name, snapshot in plan['shards'].items():
        stat = (args.source_dir / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (snapshot['bytes'], snapshot['mtime_ns']):
            raise ValueError(f'Source snapshot mismatch on node{args.rank}: {name}')
    for stage in STAGES:
        schedule_path = args.run_dir / f'{stage}_plan.json'
        if args.rank == 0:
            scheduled = stage_plan(args, plan, stage)
            atomic_json(schedule_path, scheduled)
            atomic_json(args.control / f'{stage}.ready', {})
        wait_for(args, args.control / f'{stage}.ready')
        scheduled = json.loads(schedule_path.read_text())
        names = scheduled['assignments'][args.rank]
        selection = args.run_dir / 'selections' / f'{stage}.node{args.rank}.json'
        # Preserve the original scope for cache signatures; gates select eligible rows.
        atomic_json(selection, dict(cases=[dict(shard=n, row_idx=i)
                    for n in names for i in plan['shards'][n]['indices']]))
        if names:
            execute(args, stage_command(args, stage, args.run_dir / 'work' / stage / f'node{args.rank}', selection), stage)
        atomic_json(args.control / f'{stage}.node{args.rank}.done', dict(eligible_rows=scheduled['node_rows'][args.rank]))
        if args.rank == 0:
            for rank in range(args.nodes):
                wait_for(args, args.control / f'{stage}.node{rank}.done')
            result = merge_stage(args, plan, stage, scheduled)
            atomic_json(args.control / f'{stage}.merged', result)
        wait_for(args, args.control / f'{stage}.merged')
    if args.rank == 0:
        summaries = {stage: json.loads((args.run_dir / stage / 'run_summary.json').read_text()) for stage in STAGES}
        result = dict(completed_utc=datetime.now(timezone.utc).isoformat(), code_sha256=report['code_sha256'],
                      local_test=args.local_test, hosts=[r['hostname'] for r in reports], stages=summaries,
                      note='Structural completion; per-record errors and semantic QC remain explicit in stage counts.')
        atomic_json(args.run_dir / 'reports' / 'run_manifest.json', result)
        atomic_json(args.control / 'complete.ok', result)
    wait_for(args, args.control / 'complete.ok')
    print(f'node{args.rank}: complete, results={args.run_dir}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=BASE / 'datasets/ScaleEdit-filtered-source')
    parser.add_argument('--selection-file', type=Path)
    parser.add_argument('--model-path', default=str(BASE / 'models/pretrained_models/Qwen3.8-27B'))
    parser.add_argument('--checkpoint-path', default='/mnt/bn/strategy-mllm-train/common/models/sam3/sam3.pt')
    parser.add_argument('--nodes', type=int, default=4)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grounding-tp', type=int, default=2)
    parser.add_argument('--grounding-batch-size', type=int, default=4)
    parser.add_argument('--wait-seconds', type=int, default=172800)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attempt', default='initial')
    parser.add_argument('--local-test', action='store_true')
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    if args.nodes != 4 or not 0 <= args.rank < args.nodes:
        parser.error('Requires four ranks, rank=0..3')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.attempt) or args.attempt in ('.', '..'):
        parser.error('Invalid attempt name')
    if args.resume == (args.attempt == 'initial'):
        parser.error('New runs use initial; resumes require a new shared attempt name')
    if args.batch_size < 1 or args.grounding_batch_size < 1:
        parser.error('Batch sizes must be positive')
    devices = runner.parse_device_groups(args.devices, args.grounding_tp)
    flat = [d for g in devices for d in g]
    if len(set(flat)) != len(flat) or (not args.local_test and len(flat) != 8):
        parser.error('Production needs eight distinct GPUs per node')
    args.run_dir, args.source_dir = args.run_dir.resolve(), args.source_dir.resolve()
    if args.run_dir == args.source_dir or args.source_dir in args.run_dir.parents or args.run_dir in args.source_dir.parents:
        parser.error('Source and run directories must be disjoint')
    args.control = args.run_dir / 'control' / args.attempt
    args.control.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'logs').mkdir(exist_ok=True)
    with (args.run_dir / f'node{args.rank}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = args.control / f'node{args.rank}.started'
        if started.exists():
            raise ValueError('Attempt already used; resume with a new shared attempt name')
        atomic_json(started, dict(pid=os.getpid()))
        try:
            pipeline(args)
        except BaseException as exc:
            atomic_json(args.control / f'pipeline.node{args.rank}.failed', dict(error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
