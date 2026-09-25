"""Shared-filesystem data parallelism around the unchanged removal pipeline.

No DDP, distributed model, new prompt, altered seed, or success-based resampling.
Every source (all its regions) belongs to exactly one node. Outputs stay sharded.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
import traceback

PROFILE = {
    'planner': 'relations-v16', 'thinking': True, 'ground_keeps': True,
    'auxiliary': 'ownership-v1', 'keep_fallback': 'box-protect-v1',
    'editor': 'relation-spatial-v1', 'backend': 'vllm-omni',
    'latent': 'guard-any-v1', 'geometry': 'visible-v1',
    'composition': 'adaptive-remove-v5', 'steps': 40, 'seed': 0,
    'audit': 'completion-v5', 'layout': 'adaptive', 'pixel_veto': True,
    'rewrite': False,
}
REPO = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_digest():
    h = hashlib.sha256()
    for folder in ('utils', 'synthesis_pipeline'):
        for file in sorted((REPO / folder).glob('*.py')):
            h.update(str(file.relative_to(REPO)).encode())
            h.update(file.read_bytes())
    return h.hexdigest()


def model_identity():
    """Require the same shared model locations/configs, without rehashing 100GB per worker."""
    locations = {
        'qwen38': (os.environ.get('SAMTOK_QWEN38_MODEL', ''), 'config.json'),
        'qwen21': (os.environ.get('SAMTOK_QWEN21_MODEL', ''), 'model_index.json'),
        'sam3': (os.environ.get('SAMTOK_SAM3_CHECKPOINT', ''), None),
    }
    result = {}
    for name,(location,config) in locations.items():
        if not location: raise ValueError(f'Missing deployed model location: {name}')
        path=Path(location).resolve(); check=path/config if config else path
        if not check.is_file(): raise FileNotFoundError(check)
        result[name]={'path':str(path),'bytes':check.stat().st_size}
        if config: result[name]['config_sha256']=digest(check)
        if (path/'staging_manifest.json').is_file():
            result[name]['staged_weights_sha256']=digest(path/'staging_manifest.json')
    return result


def split_sources(rows, nodes):
    """Greedy source-group balancing preserves IDs and source-neighbor context."""
    groups = {}
    names, ids = set(), set()
    for row in rows:
        name = row['image']
        if Path(name).name != name or not re.fullmatch(r'\d+_[\w.-]+\.png', name):
            raise ValueError(f'Unsafe/noncanonical image ID: {name!r}')
        number = int(name.split('_')[0])
        if name in names or number in ids:
            raise ValueError(f'Duplicate image or numeric ID: {name}')
        if row.get('task_type') != 'remove':
            raise ValueError('Frozen current profile is removal-only; do not silently convert other types')
        if not row.get('mask') or str(row.get('answer', '')).strip().lower().rstrip('.') == 'no target':
            raise ValueError(f'Unprepared/No target input: {name}')
        source = row['source_image']
        if Path(source).name != source:
            raise ValueError(f'Unsafe source path: {source}')
        names.add(name); ids.add(number)
        groups.setdefault(source, []).append(row)
    shards = [[] for _ in range(nodes)]
    for group in groups.values():
        owner = min(range(nodes), key=lambda i: (len(shards[i]), i))
        shards[owner].extend(group)
    order = {r['image']: i for i, r in enumerate(rows)}
    for shard in shards:
        shard.sort(key=lambda r: order[r['image']])
    return shards


def failed(root):
    files = sorted((root / 'control').glob('*.failed.json'))
    return files[0] if files else None


def check_peer_failure(root):
    if root is not None:
        problem = failed(Path(root))
        if problem:
            raise RuntimeError(f'Peer failed: {problem}')


def wait_for(paths, root, timeout):
    start = time.monotonic()
    while True:
        problem = failed(root)
        if problem:
            raise RuntimeError(f'Peer failed: {problem}: {problem.read_text()[:2000]}')
        if all(path.is_file() for path in paths):
            return
        if time.monotonic() - start > timeout:
            raise TimeoutError(f'Waiting for {[str(p) for p in paths if not p.exists()]}')
        time.sleep(2)


def run_stage(command, logfile, root, timeout):
    """Kill the complete local stage process group when any node fails."""
    print('RUN', json.dumps(command), 'LOG', logfile, flush=True)
    logfile.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with logfile.open('x') as log:
        proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'})
        try:
            while proc.poll() is None:
                problem = failed(root)
                if problem:
                    raise RuntimeError(f'Peer failed: {problem}')
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f'Stage timeout: {logfile}')
                time.sleep(2)
            if proc.returncode:
                raise RuntimeError(f'Stage exited {proc.returncode}: {logfile}')
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                proc.wait()
            raise
    return time.monotonic() - start


def pipeline_command(data, out, gpus):
    return [sys.executable, '-m', 'synthesis_pipeline.run_relation_cohort',
            '--data-root', str(data), '--out-root', str(out), '--gpus', gpus,
            '--policy', PROFILE['planner'], '--thinking', '--ground-keeps',
            '--auxiliary-policy', PROFILE['auxiliary'], '--keep-fallback', PROFILE['keep_fallback'],
            '--editor-policy', PROFILE['editor'], '--qwen21-backend', PROFILE['backend'],
            '--latent-protection-policy', PROFILE['latent'], '--relation-geometry-policy', PROFILE['geometry'],
            '--remove-composition-policy', PROFILE['composition']]


def prepare(data, root, nodes):
    rows = read_rows(data / 'annotations.jsonl')
    if not rows:
        raise ValueError('Empty dataset')
    shards = split_sources(rows, nodes)
    # Validate every source exists before launching GPU work, not just node 0's.
    for name in {r['source_image'] for r in rows}:
        if not (data / 'sources' / name).is_file():
            raise FileNotFoundError(data / 'sources' / name)
    for rank, shard in enumerate(shards):
        dest = root / 'inputs' / f'node{rank}'
        dest.mkdir(parents=True, exist_ok=False)
        (dest / 'sources').symlink_to((data / 'sources').resolve())
        write_rows(dest / 'annotations.jsonl', shard)
        write_rows(dest / 'input_annotations.jsonl', shard)
    manifest = dict(cases=len(rows), source_images=len({r['source_image'] for r in rows}),
                    counts=[len(s) for s in shards], input_sha256=digest(data/'annotations.jsonl'),
                    shard_sha256=[digest(root/'inputs'/f'node{i}'/'annotations.jsonl') for i in range(nodes)])
    atomic(root / 'reports' / 'partition.json', manifest)
    return manifest


def merge(root, nodes):
    inventory, passed, audited = [], [], []
    seen = set()
    for rank in range(nodes):
        rows = read_rows(root/'inputs'/f'node{rank}'/'annotations.jsonl')
        out = root/'nodes'/f'node{rank}'
        plans = {r['image']: r for r in read_rows(out/'pipeline/relations/annotations.jsonl')} if rows else {}
        resolutions = {r['image']: r for r in read_rows(out/'pipeline/regions/resolution.jsonl')} if rows else {}
        finalroot = out/'pipeline/editing/context_grounded_v4_qwen21'
        generated = {r['image']: r for r in read_rows(finalroot/'annotations.jsonl')} if (finalroot/'annotations.jsonl').exists() else {}
        audits = {r['image']: r for r in read_rows(out/'audit/audit.jsonl')}
        expected = {r['image'] for r in rows}
        if set(plans) != expected or set(resolutions) != expected or set(audits) != set(generated):
            raise ValueError(f'Incomplete stage coverage on node {rank}')
        if not set(generated) <= expected:
            raise ValueError('Unexpected generated IDs')
        for row in rows:
            name = row['image']
            if name in seen: raise ValueError(f'Duplicate merged ID {name}')
            seen.add(name)
            resolved = resolutions[name]['status']
            if (resolved == 'accepted') != (name in generated):
                raise ValueError(f'Missing/unexpected generated image: {name}')
            image = finalroot/'edited'/name
            if name in generated and not image.is_file(): raise FileNotFoundError(image)
            status = audits[name]['decision'] if name in audits else 'no_output'
            item = dict(image=name, node=rank, status=status, resolution_status=resolved,
                        source_path=str((root/'inputs'/f'node{rank}'/'sources'/row['source_image']).resolve()),
                        edited_path=str(image) if image.is_file() else None,
                        evidence_root=str(out), original_mask=row['mask'])
            inventory.append(item)
            if name in audits: audited.append({**audits[name], 'node': rank})
            if status == 'pass':
                passed.append({**generated[name], **item, 'quality_label': 'model_pass_not_human_verified'})
    order = {r['image']: i for i, r in enumerate(inventory)}
    audited.sort(key=lambda r: order[r['image']])
    write_rows(root/'results/all_cases.jsonl', inventory)
    write_rows(root/'results/audit.jsonl', audited)
    write_rows(root/'results/model_pass.jsonl', passed)
    result = dict(input_cases=len(inventory), decisions=dict(Counter(r['status'] for r in inventory)),
                  model_calls_audited=len(audited), all_nodes_completed=True, profile=PROFILE,
                  quality_note='Model pass is not independently reviewed ground truth',
                  inventory_sha256=digest(root/'results/all_cases.jsonl'))
    atomic(root/'reports/final.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--rank', type=int, default=int(os.environ.get('ARNOLD_ID', '0')))
    p.add_argument('--nodes', type=int, default=4)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--timeout', type=int, default=604800, help='Whole stage timeout; default 7 days')
    p.add_argument('--join-timeout', type=int, default=10800)
    p.add_argument('--local-test', action='store_true', help='Allow same-host ranks and fewer GPUs; real models still run')
    p.add_argument('--coordination-only', action='store_true', help='No models or generated outputs; test control plane only')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', a.run_id): p.error('Invalid run ID')
    if not 0 <= a.rank < a.nodes: p.error('Invalid rank')
    if a.nodes != 4: p.error('This entry expects four nodes')
    gpus = a.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus): p.error('Invalid GPU list')
    if not a.local_test and len(gpus) != 8: p.error('Production requires 8 GPUs per node')
    if a.coordination_only and not a.local_test: p.error('coordination-only requires local-test')
    a.run_root = a.run_root.resolve(); a.data_root = a.data_root.resolve()
    root = a.run_root; control = root/'control'; control.mkdir(parents=True, exist_ok=True)
    # Atomic exclusive claim prevents stale done markers or duplicate node launches.
    claim = control/f'node{a.rank}.claim'
    with claim.open('x') as f: f.write(f'{socket.gethostname()} {os.getpid()}\n')
    def interrupted(signum, frame):
        raise RuntimeError(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        env_report = os.environ.get('SAMTOK_ENV_REPORT')
        versions = json.loads(Path(env_report).read_text()) if env_report else None
        if not a.local_test and versions is None:
            raise ValueError('Production requires SAMTOK_ENV_REPORT from environment preflight')
        if not a.local_test and any(v.get('gpu_count',0)<8 for v in versions.values()):
            raise ValueError('Fewer than 8 visible physical GPUs in runtime preflight')
        report = dict(run_id=a.run_id, nodes=a.nodes, rank=a.rank, hostname=socket.gethostname(),
                      gpu_count=len(gpus), code_sha256=code_digest(), profile=PROFILE,
                      input_sha256=digest(a.data_root/'annotations.jsonl'),
                      local_test=a.local_test, coordination_only=a.coordination_only,
                      environment=versions, models=None if a.coordination_only else model_identity())
        atomic(root/'reports'/f'topology.node{a.rank}.json', report)
        reports = [root/'reports'/f'topology.node{i}.json' for i in range(a.nodes)]
        wait_for(reports, root, a.join_timeout)
        peers = [json.loads(f.read_text()) for f in reports]
        comparable = lambda r: {k:v for k,v in r.items() if k not in {'rank','hostname'}}
        if any(comparable(r) != comparable(report) for r in peers):
            raise ValueError('Nodes disagree on code, inputs, environment, GPU count or policy')
        if not a.local_test and len({r['hostname'] for r in peers}) != 4:
            raise ValueError('Production requires four distinct hostnames')
        if a.rank == 0:
            atomic(root/'reports/topology.json', peers)
            prepare(a.data_root, root, a.nodes)
            atomic(control/'partition.ok.json', {'ready': True})
        wait_for([control/'partition.ok.json'], root, a.join_timeout)
        data = root/'inputs'/f'node{a.rank}'; out = root/'nodes'/f'node{a.rank}'
        out.mkdir(parents=True, exist_ok=False)
        rows = read_rows(data/'annotations.jsonl')
        times = {}; started = time.monotonic()
        def progress(stage):
            atomic(root/'reports'/f'progress.node{a.rank}.json',
                   dict(stage=stage, input_cases=len(rows), seconds=time.monotonic()-started, timings=times))
            print(f'node{a.rank}: {stage}, input={len(rows)}', flush=True)
        if not a.coordination_only:
            if rows:
                progress('planning_grounding_editing')
                times['pipeline'] = run_stage(pipeline_command(data,out/'pipeline',a.gpus),
                    root/'logs'/f'pipeline.node{a.rank}.log',root,a.timeout)
            final = out/'pipeline/editing/context_grounded_v4_qwen21'
            ready = read_rows(final/'annotations.jsonl') if (final/'annotations.jsonl').exists() else []
            progress('audit')
            if ready:
                times['audit'] = run_stage([sys.executable,'-m','synthesis_pipeline.audit_removal_concise',
                    '--data-root',str(out/'pipeline/regions'),'--edited-dir',str(final/'edited'),
                    '--out-root',str(out/'audit'),'--gpus',a.gpus,'--policy',PROFILE['audit'],
                    '--input-layout',PROFILE['layout'],'--pixel-veto'],
                    root/'logs'/f'audit.node{a.rank}.log',root,a.timeout)
            else:
                write_rows(out/'audit/audit.jsonl', [])
                atomic(out/'audit/summary.json',dict(cases=0,calls=0,reason='no_executable_cases'))
        progress('node_done' if not a.coordination_only else 'coordination_only_done')
        atomic(control/f'node{a.rank}.done.json',dict(timings=times,coordination_only=a.coordination_only))
        wait_for([control/f'node{i}.done.json' for i in range(a.nodes)], root, a.timeout)
        if a.rank == 0:
            if a.coordination_only:
                atomic(root/'reports/final.json',dict(coordination_only=True,generated=0,profile=PROFILE))
            else:
                print(json.dumps(merge(root,a.nodes)), flush=True)
            atomic(control/'finalize.ok.json',dict(coordination_only=a.coordination_only))
        wait_for([control/'finalize.ok.json'], root, a.timeout)
    except BaseException:
        atomic(control/f'node{a.rank}.failed.json',dict(error=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
