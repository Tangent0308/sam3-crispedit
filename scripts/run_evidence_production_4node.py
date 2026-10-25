#!/usr/bin/env python3
"""Evidence-gated twins + multi production on Arnold workers (4 x 8 GPUs), one experiment directory.

Each rank runs the validated wave pipeline (scripts/run_evidence_waves.py, full-run-5 settings) on
its own SA-1B shards: before every wave, ``prepare-pool`` adds shard pairs until the wave can be
served (shard_feeder -> affordance labels on the idle GPUs -> per-image menus); every wave balances
the final ACCEPTED edit types; the last wave is skipped when at most ``stop_gap`` units are missing.

  prepare   once, on any machine: config.json (quotas with explicit units), inputs/
  bundle    once after prepare: code bundle (working tree) + checksums, model inventory, entry scripts
  node      on every worker (deployment/arnold_entry.sh): stage models, run the rank's waves
  status    progress of all ranks (also written by rank 0 to progress.json / logs/progress.log)
  merge     global final/ from the ranks' balanced exports (--partial before all ranks are done)
  prefeed   optional, before submitting: feed the first shards of every rank here (HDFS mount needed)
  recover-lock  release a dead worker's lock (only after that worker has stopped)

Layout: nodes/node<r>/{pool, affordance_cache.json, waves/, done.json}; pool_claims/ (image-hash
claims shared by all ranks); final/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
from sa1b_pipeline.production_state import atomic_json, file_hash, immutable_json  # noqa: E402

OPERATIONS = ('add', 'remove', 'replace', 'attribute')
DATA = Path('/mnt/bn/strategy-mllm-train/user/tanyue/experiments2/SAMTokEdit/data_pipeline')
MODELS = Path('/mnt/bn/strategy-mllm-train/user/tanyue/models/pretrained_models')
PROTECTED = [Path('/mnt/bn/strategy-mllm-train/user/tanyue/datasets'),
             DATA / '02_production_runs/sa1b_samtok_edit_50k_20261014']
DEFAULT_INPUTS = {
    'reservation': DATA / '02_production_runs/sa1b_samtok_group_train_20k_20261016/inputs/eval_reservation.json',
    'benchmark_exclusion': DATA / '02_production_runs/sa1b_samtok_edit_50k_20261014/inputs/exclusion.json',
    'affordance_cache': DATA / '04_current_validation/sa1b_audit_verifier_20261024/affordance_cache.json'}
# Full run 5 (04_current_validation/sa1b_audit_verifier_20261024/reproduce/run_waves5.sh), plus planning views
# prepared in 16 processes (same files; production-size waves otherwise idle the planners for minutes) and
# no per-wave aggregate multi balance check (the wave coordinator balances accepted counts across waves).
DUAL_FLAGS = ['--planner-devices', '0,1,2,3,4,5,6,7', '--audit-devices', '5,6,7', '--concurrent-kinds',
              '--priority-scheduling', '--evidence-gates', '--text-verify', '--twin-single-add',
              '--evidence-gate-lighting', '--evidence-gate-medium', '--evidence-gate-viewpoint', '--replace-retry',
              '--garment-extent-check', '--edit-finish', '--audit-verifier', 'split', '--audit-ref-reconcile',
              '--keep-singles', '--planning-view-workers', '16', '--coordinated-quotas']
# Busy GPU-seconds per accepted unit in full runs 3-5 (planning + editing + audit), for the ETA only.
COST = {'twins': 160.0, 'multi': 88.0}
CHILDREN = []


def config(root: Path) -> dict:
    return json.loads((root / 'config.json').read_text())


def quotas(twin_groups: int, multi_groups: int | None, multi_targets: int | None, world: int,
           twin_samples: int | None = None) -> dict:
    per_op = math.ceil(twin_groups / world / len(OPERATIONS))
    q = {'twins': {'unit': 'groups', 'total': twin_groups, 'per_rank_per_operation': per_op,
                   'exported_total_if_complete': per_op * world * len(OPERATIONS),
                   'samples_per_group': 2,
                   **({'requested_samples': twin_samples} if twin_samples is not None else {})}}
    if multi_groups is not None:
        q['multi'] = {'unit': 'groups', 'total': multi_groups, 'per_rank': math.ceil(multi_groups / world),
                      'balance': 'targets per operation = groups x mean targets per accepted group / 4, '
                                 're-estimated after every wave'}
    else:
        per = math.ceil(multi_targets / world / len(OPERATIONS))
        q['multi'] = {'unit': 'targets', 'total': multi_targets, 'per_rank_per_operation': per}
    return q


def set_aside_unstarted(root: Path):
    """Move config.json and deployment/ to planning/replaced_<time>/ (only before any rank has started)."""
    if (root / 'nodes').exists():
        raise ValueError('ranks have started in this root; quotas and code stay frozen (use a new root)')
    kept = root / 'planning' / f'replaced_{time.strftime("%Y%m%dT%H%M%S")}'
    kept.mkdir(parents=True)
    for name in ('config.json', 'deployment'):
        if (root / name).exists():
            (root / name).rename(kept / name)
    print(json.dumps({'set_aside': str(kept)}))


def prepare(args):
    from sa1b_pipeline.production_pool import shard_order, write_tables
    root = args.root
    if args.replace_unstarted:
        set_aside_unstarted(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'inputs').mkdir(exist_ok=True)
    reservation = json.loads(Path(args.reservation).read_text())
    bench = json.loads(Path(args.benchmark_exclusion).read_text())
    exclusion = {'source_ids': sorted(set(bench['source_ids']) | set(reservation['reserved_ids'])),
                 'image_hashes': sorted(set(bench['image_hashes']) | set(reservation['reserved_image_sha256'])),
                 'benchmark_sha256': 'benchmark-200 + group-eval-reservation-1200',
                 'sources': {'benchmark': str(args.benchmark_exclusion), 'reservation': str(args.reservation)}}
    for name, value in (('eval_reservation.json', reservation), ('exclusion.json', exclusion)):
        path = root / 'inputs' / name
        if not path.exists():
            atomic_json(path, value)
    if not (root / 'inputs/affordance_cache_base.json').exists():
        shutil.copyfile(args.affordance_cache, root / 'inputs/affordance_cache_base.json')
    if not (root / 'inputs/shard_order.json').exists():
        atomic_json(root / 'inputs/shard_order.json', shard_order(args.seed, args.world_size))
    order = json.loads((root / 'inputs/shard_order.json').read_text())
    if sorted(order) != [str(r) for r in range(args.world_size)]:
        raise ValueError('inputs/shard_order.json was made for another world size; use a new root')
    write_tables(root, sorted({s for r in range(args.world_size) for s in order[str(r)][:args.tables_per_rank]}))
    c = {
        'schema': 'evidence-production-1', 'experiment': root.name, 'world_size': args.world_size,
        'gpus_per_node': 8, 'seed': args.seed, 'protocol': '2.0', 'editor_steps': 40,
        'quotas': quotas(args.twin_groups, args.multi_groups, args.multi_targets, args.world_size, args.twin_samples),
        'models': {'planner': str(MODELS / 'Qwen3.8-27B'), 'editor': str(MODELS / 'Qwen-Image-2.1'),
                   'site': str(MODELS / 'sam3')},
        'selection': {'twin_members': 3, 'twin_members_add': 4, 'worn_check': True, 'covering_check': True,
                      'no_surface_recolor': True, 'skip_grayscale': True},
        'waves': {'wave_budget_gpu_seconds': args.wave_budget, 'wave_cap_twins': args.wave_cap_twins,
                  'wave_cap_multi': args.wave_cap_multi,
                  'selection_margin': 1.25, 'fill_margin': 2.0, 'stop_gap': args.stop_gap,
                  'stop_gap_fraction': 0.0, 'max_waves': args.max_waves},
        'dual_flags': DUAL_FLAGS,
        'pool': {'feeder_streams': 16, 'feeder_workers': 64, 'feeder_parallel_shards': 2, 'menu_workers': 64,
                 'selection_workers': 32, 'label_batch': 128, 'label_chunk': 8192, 'availability_factor': 1.5,
                 'max_shards_per_wave': 4, 'service_dir': '/tmp/sa1b-evidence-label-services'},
        'inputs': {name: file_hash(root / 'inputs' / name) for name in
                   ('eval_reservation.json', 'exclusion.json', 'affordance_cache_base.json', 'shard_order.json')},
        'model_cache_root': '/opt/tiger/sa1b-evidence-model-cache'}
    immutable_json(root / 'config.json', c)
    print(json.dumps({'root': str(root), 'quotas': c['quotas'],
                      'tables': len(list((root / 'inputs/shards').glob('*.json')))}, indent=1))


def bundle(args):
    """Freeze the code (working tree: tracked and untracked, not ignored) with checksums, the model inventory
    and the entry scripts. Nothing is committed."""
    root, dest = args.root, args.root / 'deployment'
    dest.mkdir(exist_ok=True)
    if (dest / 'code.tar.gz').exists():
        raise ValueError('deployment already frozen; use a new experiment root for code changes')
    names = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '--', 'sa1b_pipeline', 'scripts',
                                     'utils', 'requirements', 'pyproject.toml', 'requirements.txt'],
                                    cwd=REPO, text=True).split()
    names = sorted(n for n in names if (REPO / n).is_file() and '__pycache__' not in n)
    for needed in ('sa1b_pipeline/instruction_protocol.py', 'sa1b_pipeline/assets/fonts/sha256.json',
                   'requirements/group_production/mllm.lock.txt', 'scripts/evidence_production/launch_4node.sh',
                   'scripts/evidence_production/setup_extras.sh'):
        if needed not in names:
            raise ValueError(f'code bundle would miss {needed}')
    temp = dest / 'code.tar.gz.partial'
    import gzip
    with gzip.GzipFile(temp, 'wb', mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as tar:
        for name in names:
            info = tar.gettarinfo(str(REPO / name), arcname=name)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, '', ''
            with (REPO / name).open('rb') as f:
                tar.addfile(info, f)
    temp.replace(dest / 'code.tar.gz')
    (dest / 'code.sha256').write_text(file_hash(dest / 'code.tar.gz') + '\n')
    (dest / 'config.sha256').write_text(file_hash(root / 'config.json') + '\n')
    c = config(root)
    models = {}
    for path in map(Path, c['models'].values()):
        models[str(path)] = {str(p.relative_to(path)): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                             for p in sorted(path.rglob('*'))
                             if p.is_file() and not any(x.startswith('.') for x in p.relative_to(path).parts)}
    atomic_json(dest / 'models.json', models)
    shutil.copyfile(REPO / 'scripts/evidence_production/launch_4node.sh', dest / 'launch_4node.sh')
    import shlex
    (dest / 'arnold_entry.sh').write_text(
        '#!/usr/bin/env bash\n# Same command on every Arnold worker.\nset -euo pipefail\n'
        f'export EVIDENCE_RUN_ROOT={shlex.quote(str(root))}\n'
        'export EVIDENCE_RUNTIME_ROOT="${EVIDENCE_RUNTIME_ROOT:-/opt/tiger/sa1b-evidence-production-runtime}"\n'
        'exec bash "$EVIDENCE_RUN_ROOT/deployment/launch_4node.sh"\n')
    git = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).count('\n')
    atomic_json(dest / 'manifest.json', {
        'code_sha256': file_hash(dest / 'code.tar.gz'), 'files': len(names), 'git_head': git,
        'uncommitted_paths_at_bundle': dirty, 'config_sha256': file_hash(root / 'config.json'),
        'model_inventory_sha256': file_hash(dest / 'models.json'),
        'instruction_protocol_sha256': file_hash(REPO / 'sa1b_pipeline/instruction_protocol.py')})
    print(json.dumps({'root': str(root), 'files': len(names), 'code_sha256': file_hash(dest / 'code.tar.gz')}))


def verify_frozen(root: Path):
    dest = root / 'deployment'
    if file_hash(root / 'config.json') != (dest / 'config.sha256').read_text().strip():
        raise ValueError('frozen config changed')
    manifest = json.loads((dest / 'manifest.json').read_text())
    if file_hash(dest / 'models.json') != manifest['model_inventory_sha256']:
        raise ValueError('model inventory changed')
    for name, digest in config(root)['inputs'].items():
        if file_hash(root / 'inputs' / name) != digest:
            raise ValueError(f'frozen input changed: {name}')
    for model, inventory in json.loads((dest / 'models.json').read_text()).items():
        for name, meta in inventory.items():
            stat = (Path(model) / name).stat()
            if stat.st_size != meta['size'] or stat.st_mtime_ns != meta['mtime_ns']:
                raise ValueError(f'frozen model changed: {model}/{name}')


def coordinator(root: Path, c: dict, rank: int, mllm: str, editor: str) -> list:
    node = root / 'nodes' / f'node{rank}'
    q, w, sel = c['quotas'], c['waves'], c['selection']
    multi = (['--multi-groups', str(q['multi']['per_rank'])] if q['multi']['unit'] == 'groups'
             else ['--multi-targets-per-op', str(q['multi']['per_rank_per_operation'])])
    flags = [flag for key, flag in (('worn_check', '--worn-check'), ('covering_check', '--covering-check'),
                                    ('no_surface_recolor', '--no-surface-recolor'), ('skip_grayscale', '--skip-grayscale'))
             if sel[key]]
    hook = [mllm, str(Path(__file__).resolve()), '--root', str(root), 'prepare-pool', '--rank', str(rank)]
    return [mllm, str(REPO / 'scripts/run_evidence_waves.py'), '--root', str(node / 'waves'),
            '--sources-root', str(node / 'pool/sources'), '--reservation', str(root / 'inputs/eval_reservation.json'),
            '--exclude', str(root / 'inputs/exclusion.json'), '--affordance-cache', str(node / 'affordance_cache.json'),
            '--twin-groups-per-op', str(q['twins']['per_rank_per_operation']), *multi,
            '--max-waves', str(w['max_waves']), '--wave-cap-twins', str(w['wave_cap_twins']),
            '--wave-cap-multi', str(w['wave_cap_multi']), '--seed', str(c['seed'] + rank),
            *(['--wave-budget', str(w['wave_budget_gpu_seconds'])] if w.get('wave_budget_gpu_seconds') else []),
            '--selection-margin', str(w['selection_margin']), '--fill-margin', str(w['fill_margin']),
            '--stop-gap', str(w['stop_gap']), '--stop-gap-fraction', str(w['stop_gap_fraction']),
            '--twin-members-add', str(sel['twin_members_add']), *flags,
            '--menu-dir', str(node / 'pool/menus'), '--selection-workers', str(c['pool']['selection_workers']),
            '--before-wave-json', json.dumps(hook), '--',
            '--planner-model', c['models']['planner'], '--editor-model', c['models']['editor'],
            '--site-model', c['models']['site'], '--mllm-python', mllm, '--editor-python', editor, *c['dual_flags']]


def node(args):
    root, rank = args.root, args.rank
    c = config(root)
    world = int(os.environ.get('ARNOLD_WORKER_NUM', '0'))
    if world != c['world_size'] or int(os.environ.get('ARNOLD_WORKER_GPU', '0')) != 8 \
            or os.environ.get('ARNOLD_ID') != str(rank) or rank not in range(c['world_size']):
        raise ValueError(f"needs {c['world_size']} Arnold workers x 8 GPUs and ARNOLD_ID == --rank")
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('unset CUDA_VISIBLE_DEVICES; ranks use physical GPUs 0..7')
    verify_frozen(root)
    from scripts.run_group_production_4node import owner_lock
    node_dir = root / 'nodes' / f'node{rank}'
    node_dir.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        for child in CHILDREN:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise RuntimeError(f'worker interrupted by signal {signum}; waves resume from their checkpoints')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    lock = root / 'control' / f'node{rank}.lock.d'
    trial = os.environ.get('ARNOLD_TRIAL_ID')
    owner = json.loads((lock / 'owner.json').read_text()) if (lock / 'owner.json').exists() else {}
    if trial and owner.get('trial') and owner['trial'] != trial:
        # Left by an earlier Arnold trial: its containers are gone, so its workers have stopped.
        lock.rename(lock.with_name(f'{lock.name}.recovered.{time.time_ns()}'))
    with owner_lock(lock):
        stop, monitor_thread = threading.Event(), None
        if rank == 0:
            monitor_thread = threading.Thread(target=monitor, args=(root, stop), daemon=True)
            monitor_thread.start()
        try:
            started = json.loads((node_dir / 'started.json').read_text()) if (node_dir / 'started.json').exists() \
                else {'first_start': time.time(), 'starts': []}
            started['starts'].append({'time': time.time(), 'host': os.uname().nodename,
                                      'trial': os.environ.get('ARNOLD_TRIAL_ID')})
            if (node_dir / 'failure.json').exists():  # a previous attempt; kept beside, no longer current
                (node_dir / 'failure.json').rename(node_dir / f'failure.{int(time.time())}.json')
            atomic_json(node_dir / 'started.json', started)
            if args.model_map:
                mapping = json.loads(Path(args.model_map).read_text())
            else:
                from sa1b_pipeline.model_cache import stage_model
                mapping = {m: stage_model(m, c['model_cache_root'], progress=lambda x: print(json.dumps(x), flush=True))
                           for m in c['models'].values()}
            atomic_json(node_dir / 'model_map.json', mapping)
            env = dict(os.environ, SA1B_LOCAL_MODEL_MAP=json.dumps(mapping), SA1B_FORCE_DIRECT_VLLM='1',
                       PYTHONUNBUFFERED='1')
            if not (node_dir / 'affordance_cache.json').exists():
                shutil.copyfile(root / 'inputs/affordance_cache_base.json', node_dir / 'affordance_cache.json')
            cmd = coordinator(root, c, rank, args.mllm_python, args.editor_python)
            atomic_json(node_dir / 'command.json', cmd)
            for attempt in range(args.retries + 1):
                # Waves resume from their checkpoints, so a transient failure (a lost service worker,
                # a mount hiccup) costs one restart of the current stage instead of the worker.
                child = subprocess.Popen(cmd, env=env, start_new_session=True)
                CHILDREN.append(child)
                if child.wait() == 0:
                    break
                atomic_json(node_dir / f'failure.{int(time.time())}.json',
                            {'time': time.time(), 'attempt': attempt, 'error': f'coordinator exited with {child.returncode}'})
                if attempt == args.retries:
                    raise RuntimeError(f'wave coordinator exited with {child.returncode} ({attempt + 1} attempts)')
                time.sleep(120)
            atomic_json(node_dir / 'done.json', {'time': time.time(),
                                                 'summary': json.loads((node_dir / 'waves/final/summary.json').read_text())})
            if rank == 0:
                # Keep the monitor running until every rank is done or has failed (then merge by hand).
                while not all(rank_status(root, c, r)['state'] in ('done', 'failed') for r in range(c['world_size'])):
                    time.sleep(60)
                if all(rank_status(root, c, r)['state'] == 'done' for r in range(c['world_size'])):
                    merge(SimpleNamespace(root=root, partial=False))
        except BaseException as exc:
            atomic_json(node_dir / 'failure.json', {'time': time.time(), 'error': repr(exc)})
            raise
        finally:
            stop.set()
            if monitor_thread:
                monitor_thread.join(2)


def prepare_pool(args):
    from sa1b_pipeline.production_pool import Pool, used_sources
    c = config(args.root)
    plan = json.loads(Path(args.plan).read_text())
    pool = Pool(args.root, args.rank, c)
    node_dir = args.root / 'nodes' / f'node{args.rank}'
    started = time.time()
    info = pool.prepare(plan, used_sources(node_dir / 'waves'), mllm_python=sys.executable,
                        planner_model=c['models']['planner'], devices=[str(i) for i in range(c['gpus_per_node'])])
    info['seconds'] = round(time.time() - started, 1)
    with (node_dir / 'pool_log.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'wave': plan['wave'], **info}) + '\n')
    print(json.dumps({'stage': 'prepare_pool', 'wave': plan['wave'], 'shards': info['shards'],
                      'seconds': info['seconds'], 'available': info['log'][-1].get('available')}), flush=True)


def prefeed(args):
    """Feed shards for all ranks on this machine (CPU and the HDFS mount only): shard k of every rank before
    shard k+1 of any, so every rank can start early. Workers without the mount wait for these shards;
    workers still label and compute menus themselves. Safe to run while the ranks are running."""
    from concurrent.futures import ThreadPoolExecutor
    from sa1b_pipeline.production_pool import Pool
    c = config(args.root)
    pools = [Pool(args.root, rank, c) for rank in range(c['world_size'])]
    todo = []
    for k in range(args.shards_per_rank):
        for pool in pools:
            if k < len(pool.order) and pool.order[k] not in set(pool.done('shards')):
                todo.append((pool, pool.order[k]))

    from sa1b_pipeline.production_pool import write_tables
    write_tables(args.root, sorted({shard for _, shard in todo}))   # one pass over the SG2 indexes

    def one(item):
        pool, shard = item
        started = time.time()
        pool.feed([shard])
        print(json.dumps({'stage': 'prefeed', 'rank': pool.rank, 'shard': shard,
                          'seconds': round(time.time() - started)}), flush=True)
    with ThreadPoolExecutor(max(1, args.parallel)) as threads:
        list(threads.map(one, todo))


def rank_status(root: Path, c: dict, rank: int) -> dict:
    node_dir = root / 'nodes' / f'node{rank}'
    p = node_dir / 'waves/progress.json'
    s = {'rank': rank, 'state': 'done' if (node_dir / 'done.json').exists() else
         'failed' if (node_dir / 'failure.json').exists() and not (root / 'control' / f'node{rank}.lock.d').exists()
         else 'running' if (root / 'control' / f'node{rank}.lock.d').exists() else 'not_started'}
    if not p.exists():
        return s
    prog = json.loads(p.read_text())
    t, m = prog['targets'], prog['exportable']
    done = {'twins': sum(m['twins_groups'].values()), 'multi': sum(m['multi_targets'].values())}
    total = {'twins': 4 * t['twins'], 'multi': 4 * t['multi']}
    work = sum(COST[k] * total[k] for k in COST)
    finished = sum(COST[k] * done[k] for k in COST)
    first = json.loads((node_dir / 'started.json').read_text())['first_start'] if (node_dir / 'started.json').exists() else None
    elapsed = time.time() - first if first else None
    eta = elapsed * (work - finished) / finished if elapsed and finished else None
    s.update(waves_finished=prog['waves_finished'], targets=t, exportable=m, multi_groups_accepted=prog['multi_groups_accepted'],
             progress=round(finished / work, 4) if work else None, elapsed_hours=round(elapsed / 3600, 2) if elapsed else None,
             eta_hours=round(eta / 3600, 1) if eta is not None else None, updated=prog['updated'])
    return s


def status(root: Path) -> dict:
    c = config(root)
    ranks = [rank_status(root, c, r) for r in range(c['world_size'])]
    out = {'time': time.time(), 'quotas': c['quotas'], 'ranks': ranks}
    known = [r for r in ranks if r.get('progress') is not None]
    if known:
        out['progress'] = round(sum(r['progress'] for r in known) / c['world_size'], 4)
        etas = [r['eta_hours'] for r in known if r.get('eta_hours') is not None]
        out['eta_hours_slowest_rank'] = max(etas) if etas else None
    return out


def monitor(root: Path, stop: threading.Event):
    while not stop.is_set():
        try:
            s = status(root)
            atomic_json(root / 'progress.json', s)
            line = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' | ' + ' | '.join(
                f"r{r['rank']} {r['state']} w{r.get('waves_finished', 0)} "
                f"{100 * (r.get('progress') or 0):.1f}% eta {r.get('eta_hours')}h" for r in s['ranks'])
            (root / 'logs').mkdir(exist_ok=True)
            with (root / 'logs/progress.log').open('a') as f:
                f.write(line + '\n')
        except (OSError, ValueError, KeyError) as exc:  # a rank file being replaced; try again next round
            print(json.dumps({'monitor_error': repr(exc)}), flush=True)
        stop.wait(600)


def merge(args):
    """final/: the ranks' balanced exports side by side (each rank is balanced on its own quota)."""
    import run_evidence_waves as rw
    root = args.root
    c = config(root)
    q = c['quotas']
    out = root / ('final_partial' if args.partial else 'final')
    out.mkdir(exist_ok=True)
    totals = {'twins_groups': {op: 0 for op in OPERATIONS}, 'multi_targets': {op: 0 for op in OPERATIONS},
              'twins_group_count': 0, 'multi_group_count': 0, 'singles': 0}
    ranks = []
    files = {name: (out / name).open('w') for name in ('twins_balanced.jsonl', 'multi_balanced.jsonl', 'singles.jsonl')}
    try:
        for r in range(c['world_size']):
            waves = root / 'nodes' / f'node{r}' / 'waves'
            if not args.partial and not (root / 'nodes' / f'node{r}' / 'done.json').exists():
                raise FileNotFoundError(f'rank {r} has not finished; use --partial')
            if not waves.exists():
                continue
            a = SimpleNamespace(twin_groups_per_op=q['twins']['per_rank_per_operation'],
                                multi_groups=q['multi'].get('per_rank') if q['multi']['unit'] == 'groups' else None,
                                multi_targets_per_op=q['multi'].get('per_rank_per_operation'),
                                stop_gap=c['waves']['stop_gap'], stop_gap_fraction=c['waves']['stop_gap_fraction'])
            rank_out = waves / 'final' if not args.partial else out / f'rank{r}'
            summary = (json.loads((waves / 'final/summary.json').read_text()) if not args.partial
                       else rw.export(waves, a, len(rw.waves(waves)), out=rank_out))
            for name, f in files.items():
                path = rank_out / name
                if path.exists():
                    for line in path.read_text().splitlines():
                        if line.strip():
                            f.write(line + '\n')
            e = summary['exported']
            for op in OPERATIONS:
                totals['twins_groups'][op] += e['twins_groups'].get(op, 0)
                totals['multi_targets'][op] += e['multi_targets'].get(op, 0)
            totals['twins_group_count'] += e['twins_group_count']
            totals['multi_group_count'] += e['multi_group_count']
            totals['singles'] += summary['singles']['count']
            ranks.append({'rank': r, **summary})
    finally:
        for f in files.values():
            f.close()
    report = {'quotas': q, 'complete': not args.partial, 'exported': totals,
              'twin_samples': 2 * totals['twins_group_count'],
              'multi_samples_targets': sum(totals['multi_targets'].values()), 'ranks': ranks,
              'note': 'twins: group = two images edited with one shared instruction; multi: group = one image with '
                      '2-4 targets edited in sequence. Singles are a separate product, outside the balance.'}
    atomic_json(out / 'summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'ranks'}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    twins = prep.add_mutually_exclusive_group(required=True)
    twins.add_argument('--twin-groups', type=int, help='total accepted twin GROUPS (2 samples each)')
    twins.add_argument('--twin-samples', type=int, help='total accepted twin SAMPLES; groups = ceil(samples / 2)')
    group = prep.add_mutually_exclusive_group(required=True)
    group.add_argument('--multi-groups', type=int, help='total accepted multi GROUPS (2-4 targets each)')
    group.add_argument('--multi-targets', '--multi-samples', dest='multi_targets', type=int,
                       help='total accepted multi TARGETS (one edit sample each)')
    prep.add_argument('--world-size', type=int, default=4)
    prep.add_argument('--seed', type=int, default=20261025)
    prep.add_argument('--wave-budget', type=float, default=30000,
                      help='busy GPU-seconds per wave and rank (~1.3 h on 8 GPUs), split between twins and multi by '
                           'remaining need; 0: fixed caps only')
    prep.add_argument('--wave-cap-twins', type=int, default=600, help='at most this many twin groups per wave and rank')
    prep.add_argument('--wave-cap-multi', type=int, default=600, help='at most this many multi groups per wave and rank')
    prep.add_argument('--stop-gap', type=int, default=2,
                      help='skip a further wave of a kind when at most this many units (per rank) are missing')
    prep.add_argument('--max-waves', type=int, default=100000, help='per rank (tests: a few)')
    prep.add_argument('--replace-unstarted', action='store_true',
                      help='before any rank has started: set the current config and deployment aside and prepare again '
                           '(e.g. other quota units); run bundle again afterwards')
    prep.add_argument('--tables-per-rank', type=int, default=24, help='shard tables written now (more on demand)')
    prep.add_argument('--reservation', type=Path, default=DEFAULT_INPUTS['reservation'])
    prep.add_argument('--benchmark-exclusion', type=Path, default=DEFAULT_INPUTS['benchmark_exclusion'])
    prep.add_argument('--affordance-cache', type=Path, default=DEFAULT_INPUTS['affordance_cache'])
    sub.add_parser('bundle')
    n = sub.add_parser('node')
    n.add_argument('--rank', type=int, required=True)
    n.add_argument('--mllm-python', required=True)
    n.add_argument('--editor-python', required=True)
    n.add_argument('--model-map', type=Path, help='existing {model: local copy} map instead of staging (tests)')
    n.add_argument('--retries', type=int, default=3, help='restarts of the wave coordinator after a failure')
    pp = sub.add_parser('prepare-pool', help='called by the wave coordinator before each selection')
    pp.add_argument('--rank', type=int, required=True)
    pp.add_argument('--plan', type=Path, required=True)
    pf = sub.add_parser('prefeed', help='feed the first shards of every rank here (needs the HDFS mount)')
    pf.add_argument('--shards-per-rank', type=int, required=True)
    pf.add_argument('--parallel', type=int, default=2, help='shards fed at the same time')
    sub.add_parser('status')
    rec = sub.add_parser('recover-lock', help='after a worker died without cleanup (e.g. Arnold restart on a new host)')
    rec.add_argument('--rank', type=int, required=True)
    rec.add_argument('--previous-workers-stopped', action='store_true', required=True)
    m = sub.add_parser('merge')
    m.add_argument('--partial', action='store_true', help='export what is finished so far into final_partial/')
    args = p.parse_args()
    args.root = args.root.resolve()
    if args.command == 'prepare' and args.twin_samples is not None:
        args.twin_groups = math.ceil(args.twin_samples / 2)
    if any(args.root == x or x in args.root.parents for x in PROTECTED):
        raise ValueError('protected location: new production runs go under 02_production_runs/')
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'bundle':
        bundle(args)
    elif args.command == 'node':
        node(args)
    elif args.command == 'prepare-pool':
        prepare_pool(args)
    elif args.command == 'status':
        print(json.dumps(status(args.root), indent=1))
    elif args.command == 'prefeed':
        prefeed(args)
    elif args.command == 'recover-lock':
        lock = args.root / 'control' / f'node{args.rank}.lock.d'
        if lock.exists():
            kept = lock.with_name(f'{lock.name}.recovered.{time.time_ns()}')
            lock.rename(kept)
            print(f'kept the previous lock at {kept}')
    else:
        merge(args)


if __name__ == '__main__':
    main()
